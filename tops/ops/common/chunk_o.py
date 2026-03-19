"""Pallas fused backward kernel for simple GLA (g_gamma only).

Simple GLA uses per-head scalar gates g_gamma: [H] instead of per-element
gates gk: [B,T,H,K]. The gate for position t within a chunk is:
    b_g[t] = gamma * (t + 1)

This kernel computes dq, dk, dv in a single fused pass (no dg needed since
g_gamma is a hyperparameter with no gradient).
"""

import functools

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu


def chunk_simple_gla_bwd_kernel(
    q_ref, k_ref, v_ref, g_gamma_ref, h_ref, a_ref, do_ref, dh_ref,
    dq_ref, dk_ref, dv_ref,
    *,
    BT: int,
    scale: float,
):
    """Fused backward kernel for simple GLA with g_gamma.

    Grid: (H, total_NT)
    Refs (after block spec indexing):
      q_ref/k_ref: (1, 1, BT, K)
      v_ref/do_ref: (1, 1, BT, V)
      g_gamma_ref: [H] in SMEM
      h_ref/dh_ref: (1, 1, K, V)
      a_ref: (1, 1, BT, BT)
      dq_ref/dk_ref: (1, 1, BT, K)
      dv_ref: (1, 1, BT, V)
    """
    b_q = q_ref[0, 0]    # (BT, K)
    b_k = k_ref[0, 0]    # (BT, K)
    b_v = v_ref[0, 0]    # (BT, V)
    b_h = h_ref[0, 0].astype(jnp.float32)    # (K, V)
    b_a = a_ref[0, 0].astype(jnp.float32)    # (BT, BT)
    b_do = do_ref[0, 0]  # (BT, V)
    b_dh = dh_ref[0, 0].astype(jnp.float32)  # (K, V)

    # Build per-position decay from g_gamma
    head_idx = pl.program_id(0)
    b_gamma = g_gamma_ref[head_idx]
    b_g = b_gamma * (jnp.arange(BT) + 1).astype(jnp.float32)  # [BT]
    b_gn = b_g[BT - 1]  # scalar — last position decay

    # 1. dA = do @ v^T * scale, masked lower-triangular
    b_A_do = b_do.astype(b_v.dtype)
    b_dA = jnp.dot(b_A_do, b_v.T, precision=jax.lax.Precision.HIGHEST,
                    preferred_element_type=jnp.float32) * scale
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_dA = jnp.where(mask, b_dA, 0.0)

    # 2. dv = A^T @ do + k_decay @ dh
    b_a_masked = jnp.where(mask, b_a, 0.0)
    b_dv_intra = jnp.dot(b_a_masked.T.astype(b_do.dtype), b_do,
                          precision=jax.lax.Precision.HIGHEST,
                          preferred_element_type=jnp.float32)
    k_decay = (b_k * jnp.exp(b_gn - b_g)[:, None]).astype(b_k.dtype)
    b_dv_inter = jnp.dot(k_decay, b_dh.astype(k_decay.dtype),
                          precision=jax.lax.Precision.HIGHEST,
                          preferred_element_type=jnp.float32)
    b_dv = b_dv_intra + b_dv_inter
    dv_ref[0, 0] = b_dv.astype(dv_ref.dtype)

    # 3. dq = dA @ k_neg * exp(b_g) + do @ h^T * scale * exp(b_g)
    k_neg = (b_k * jnp.exp(-b_g)[:, None]).astype(b_k.dtype)
    b_dq_intra = jnp.dot(b_dA.astype(k_neg.dtype), k_neg,
                          precision=jax.lax.Precision.HIGHEST,
                          preferred_element_type=jnp.float32) * jnp.exp(b_g)[:, None]
    b_dq_inter = jnp.dot(b_do, b_h.astype(b_do.dtype).T,
                          precision=jax.lax.Precision.HIGHEST,
                          preferred_element_type=jnp.float32) * (scale * jnp.exp(b_g)[:, None])
    b_dq = b_dq_intra + b_dq_inter
    dq_ref[0, 0] = b_dq.astype(dq_ref.dtype)

    # 4. dk = dA^T @ q_pos * exp(-b_g) + v @ dh^T * exp(b_gn - b_g)
    q_pos = (b_q * jnp.exp(b_g)[:, None]).astype(b_q.dtype)
    b_dk_intra = jnp.dot(b_dA.T.astype(q_pos.dtype), q_pos,
                          precision=jax.lax.Precision.HIGHEST,
                          preferred_element_type=jnp.float32) * jnp.exp(-b_g)[:, None]
    b_dk_inter = jnp.dot(b_v, b_dh.astype(b_v.dtype).T,
                          precision=jax.lax.Precision.HIGHEST,
                          preferred_element_type=jnp.float32) * jnp.exp(b_gn - b_g)[:, None]
    b_dk = b_dk_intra + b_dk_inter
    dk_ref[0, 0] = b_dk.astype(dk_ref.dtype)


def chunk_simple_gla_bwd_o_pl(
    q: jax.Array,   # [B, T, H, K]
    k: jax.Array,   # [B, T, H, K]
    v: jax.Array,   # [B, T, H, V]
    g_gamma: jax.Array,  # [H]
    h: jax.Array,   # [B, NT, H, K, V]
    A: jax.Array,   # [B, T, H, BT]
    do: jax.Array,  # [B, T, H, V]
    dh: jax.Array,  # [B, NT, H, K, V]
    scale: float,
    chunk_size: int,
):
    """Launcher for the fused simple GLA backward kernel.

    Returns: (dq, dk, dv)
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    # Reshape: [B, T, H, D] -> [H, total_NT, BT, D]
    def _reshape_bt(x, D):
        return x.reshape(B, NT, BT, H, D).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, D)

    _q = _reshape_bt(q, K)
    _k = _reshape_bt(k, K)
    _v = _reshape_bt(v, V)
    _do = _reshape_bt(do, V)
    _A = A.reshape(B, NT, BT, H, BT).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, BT)
    # h/dh: [B, NT, H, K, V] -> [H, total_NT, K, V]
    _h = h.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)
    _dh = dh.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)

    # BlockSpecs
    grid = (H, total_NT)
    spec_K = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_V = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_h = pl.BlockSpec([1, 1, K, V], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_A = pl.BlockSpec([1, 1, BT, BT], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_gamma = pl.BlockSpec(memory_space=pltpu.SMEM)

    dq_shape = jax.ShapeDtypeStruct([H, total_NT, BT, K], q.dtype)
    dk_shape = jax.ShapeDtypeStruct([H, total_NT, BT, K], k.dtype)
    dv_shape = jax.ShapeDtypeStruct([H, total_NT, BT, V], v.dtype)

    dq, dk, dv = pl.pallas_call(
        functools.partial(chunk_simple_gla_bwd_kernel, BT=BT, scale=scale),
        grid=grid,
        out_shape=[dq_shape, dk_shape, dv_shape],
        in_specs=[spec_K, spec_K, spec_V, spec_gamma, spec_h, spec_A, spec_V, spec_h],
        out_specs=[spec_K, spec_K, spec_V],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=32 * 1024 * 1024,
        ),
    )(_q, _k, _v, g_gamma, _h, _A, _do, _dh)

    # Post-process: (H, total_NT, BT, D) -> (B, T, H, D)
    def _unreshape(x, D):
        x = x.reshape(H, B, NT, BT, D)
        x = x.transpose(1, 0, 2, 3, 4)  # (B, H, NT, BT, D)
        x = x.reshape(B, H, T, D)
        return x.transpose(0, 2, 1, 3)   # (B, T, H, D)

    dq = _unreshape(dq, K)
    dk = _unreshape(dk, K)
    dv = _unreshape(dv, V)

    return dq, dk, dv
