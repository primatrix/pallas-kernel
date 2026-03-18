import functools

import jax
import jax.experimental.pallas as pl
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from tops.utils import pad_to_multiple
from tops.ops.utils import is_tpu_runtime
from tops.ops.common.chunk_h import chunk_fwd_h_kernel
from tops.ops.common.chunk_h import chunk_fwd_h_ref
from tops.ops.common.chunk_h import chunk_bwd_dh_kernel
from tops.ops.common.chunk_h import chunk_bwd_dh_ref


# =============================================================================
# Helper: chunk_local_cumsum for scalar gates [B, T, H]
# =============================================================================


def chunk_local_cumsum_scalar(
    g: jax.Array,
    chunk_size: int,
) -> jax.Array:
    """Chunk-local cumulative sum for scalar gates.

    Args:
        g: [B, T, H] — log-space scalar gates
        chunk_size: block size

    Returns:
        g_cumsum: [B, T, H] — chunk-local cumsum
    """
    B, T, H = g.shape
    C = chunk_size
    assert T % C == 0
    NT = T // C
    g = g.reshape(B * NT, C, H)
    g_cumsum = jnp.cumsum(g, axis=1)
    return g_cumsum.reshape(B, T, H)


def chunk_local_cumsum_scalar_reverse(
    dg: jax.Array,
    chunk_size: int,
) -> jax.Array:
    """Reverse chunk-local cumulative sum for scalar gate gradients.

    Args:
        dg: [B, T, H] — gradient w.r.t. cumsummed gates
        chunk_size: block size

    Returns:
        dg_raw: [B, T, H] — gradient w.r.t. raw gates
    """
    B, T, H = dg.shape
    C = chunk_size
    assert T % C == 0
    NT = T // C
    dg = dg.reshape(B * NT, C, H)
    dg_raw = jnp.cumsum(dg[:, ::-1, :], axis=1)[:, ::-1, :]
    return dg_raw.reshape(B, T, H)


def _prepare_g_cumsum(
    g: jax.Array | None,
    g_gamma: jax.Array | None,
    B: int,
    T: int,
    H: int,
    chunk_size: int,
) -> jax.Array:
    """Prepare chunk-local cumsum of scalar gates.

    Handles both data-dependent g and data-independent g_gamma.

    Args:
        g: [B, T, H] or None — raw log-space scalar gates
        g_gamma: [H] or None — constant per-head decay
        B, T, H: dimensions
        chunk_size: block size

    Returns:
        g_cumsum: [B, T, H] — chunk-local cumsum
    """
    C = chunk_size
    NT = T // C

    if g is not None:
        return chunk_local_cumsum_scalar(g, C)
    elif g_gamma is not None:
        # g_gamma is constant: cumsum within chunk = g_gamma * (1, 2, ..., C)
        positions = jnp.arange(1, C + 1, dtype=jnp.float32)  # [C]
        g_chunk = positions[:, None] * g_gamma[None, :]  # [C, H]
        g_cumsum = jnp.tile(g_chunk[None, :, :], (B * NT, 1, 1))
        return g_cumsum.reshape(B, T, H)
    else:
        return jnp.zeros((B, T, H), dtype=jnp.float32)


# =============================================================================
# Reference: simple_gla_fwd_o (for testing)
# =============================================================================


def simple_gla_fwd_o_ref(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    h: jax.Array,
    scale: float,
    chunk_size: int,
) -> jax.Array:
    """Reference: compute output with fused intra-chunk attention.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H] — chunk-local cumsum of scalar gates
        h: [B, NT, H, K, V]
        scale: scaling factor
        chunk_size: block size

    Returns:
        o: [B, T, H, V]
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C

    q_c = q.reshape(B, NT, C, H, K)
    k_c = k.reshape(B, NT, C, H, K)
    v_c = v.reshape(B, NT, C, H, V)
    g_c = g.reshape(B, NT, C, H)

    # Inter-chunk: q * exp(g) @ h
    qg = q_c * jnp.exp(g_c)[..., None]  # [B, NT, C, H, K]
    o_inter = scale * jnp.einsum(
        "bnchk,bnhkv->bnchv", qg, h,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    # Intra-chunk attention with scalar gating
    A = jnp.einsum(
        "bnihk,bnjhk->bnihj", q_c, k_c,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    # g_c is [B, NT, C, H]. A is [B, NT, C_i, H, C_j].
    # g_i: [B, NT, C, H, 1], g_j: [B, NT, 1, H, C]
    g_i = g_c[:, :, :, :, None]
    g_j = g_c.transpose(0, 1, 3, 2)[:, :, None, :, :]
    A = A * jnp.exp(g_i - g_j) * scale

    causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
    A = jnp.where(causal_mask[None, None, :, None, :], A, 0.0)

    o_intra = jnp.einsum(
        "bnihj,bnjhv->bnihv", A, v_c,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    o = (o_inter + o_intra).reshape(B, T, H, V)
    return o


# =============================================================================
# Pallas kernel: simple_gla_fwd_o (fused A computation)
# =============================================================================


def _simple_gla_fwd_o_kernel(
    q_ref,
    k_ref,
    v_ref,
    g_ref,
    h_ref,
    o_ref,
    *,
    BT: int,
    scale: float,
):
    """Fused forward output kernel for SimpleGLA.

    Computes both intra-chunk attention A and output o in one kernel.
    g is scalar per head (broadcast across K dimension).

    Grid: (H, total_NT).
    Refs (after block spec indexing):
      q_ref/k_ref: (1, 1, BT, K)
      v_ref:       (1, 1, BT, V)
      g_ref:       (1, 1, BT, 1)
      h_ref:       (1, 1, K, V)
      o_ref:       (1, 1, BT, V)
    """
    b_q = q_ref[0, 0]  # (BT, K)
    b_k = k_ref[0, 0]  # (BT, K)
    b_v = v_ref[0, 0]  # (BT, V)
    b_g = g_ref[0, 0, :, 0].astype(jnp.float32)  # (BT,)
    b_h = h_ref[0, 0]  # (K, V)

    # Inter-chunk: scale * (q * exp(g)) @ h
    b_qg = (b_q * jnp.exp(b_g)[:, None]).astype(b_q.dtype)
    b_o = (
        jnp.dot(
            b_qg,
            b_h.astype(b_qg.dtype),
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        * scale
    )

    # Intra-chunk: A = (q @ k^T) * exp(g_i - g_j) * scale; o += tril(A) @ v
    b_A = jnp.dot(
        b_q,
        b_k.T,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    b_A = b_A * jnp.exp(b_g[:, None] - b_g[None, :]) * scale
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_A = jnp.where(mask, b_A, 0.0).astype(b_v.dtype)

    b_o += jnp.dot(
        b_A,
        b_v,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    o_ref[0, 0] = b_o.astype(o_ref.dtype)


def simple_gla_fwd_o(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    h: jax.Array,
    scale: float,
    chunk_size: int,
) -> jax.Array:
    """Launcher for fused forward output Pallas kernel.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H] — chunk-local cumsum of scalar gates
        h: [B, NT, H, K, V]
        scale: scaling factor
        chunk_size: block size

    Returns:
        o: [B, T, H, V]
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    # Reshape to [H, total_NT, ...]
    _q = q.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _k = k.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _v = v.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _g = g.reshape(B, NT, BT, H).transpose(3, 0, 1, 2).reshape(H, total_NT, BT, 1)
    _h = h.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)

    spec_K = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_V = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_g = pl.BlockSpec([1, 1, BT, 1], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_h = pl.BlockSpec([1, 1, K, V], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_o = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))

    o_shape = jax.ShapeDtypeStruct([H, total_NT, BT, V], v.dtype)
    interpret = not is_tpu_runtime()

    o = pl.pallas_call(
        functools.partial(_simple_gla_fwd_o_kernel, BT=BT, scale=scale),
        grid=(H, total_NT),
        out_shape=o_shape,
        in_specs=[spec_K, spec_K, spec_V, spec_g, spec_h],
        out_specs=spec_o,
        interpret=interpret,
    )(_q, _k, _v, _g, _h)

    # Post-reshape: [H, total_NT, BT, V] -> [B, T, H, V]
    o = o.reshape(H, B, NT, BT, V)
    o = o.transpose(1, 0, 2, 3, 4).reshape(B, H, T, V)
    o = o.transpose(0, 2, 1, 3)
    return o


# =============================================================================
# Pallas kernel: simple_gla_bwd_dv (recomputes A)
# =============================================================================


def _simple_gla_bwd_dv_kernel(
    q_ref,
    k_ref,
    g_ref,
    do_ref,
    dh_ref,
    dv_ref,
    *,
    BT: int,
    scale: float,
):
    """Backward kernel: compute dv by recomputing A.

    dv = tril(A)^T @ do + k_decay @ dh
    """
    b_q = q_ref[0, 0]  # (BT, K)
    b_k = k_ref[0, 0]  # (BT, K)
    b_g = g_ref[0, 0, :, 0].astype(jnp.float32)  # (BT,)
    b_do = do_ref[0, 0]  # (BT, V)
    b_dh = dh_ref[0, 0].astype(jnp.float32)  # (K, V)

    b_gn = b_g[BT - 1]  # scalar

    # Recompute A = (q @ k^T) * exp(g_i - g_j) * scale
    b_A = jnp.dot(
        b_q,
        b_k.T,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    b_A = b_A * jnp.exp(b_g[:, None] - b_g[None, :]) * scale
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_A = jnp.where(mask, b_A, 0.0)

    # dv_intra = A^T @ do
    b_dv_intra = jnp.dot(
        b_A.T.astype(b_do.dtype),
        b_do,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    # dv_inter = k_decay @ dh, k_decay = k * exp(g_n - g)
    k_decay = (b_k * jnp.exp(b_gn - b_g)[:, None]).astype(b_k.dtype)
    b_dv_inter = jnp.dot(
        k_decay,
        b_dh.astype(k_decay.dtype),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    b_dv = b_dv_intra + b_dv_inter
    dv_ref[0, 0] = b_dv.astype(dv_ref.dtype)


def simple_gla_bwd_dv(
    q: jax.Array,
    k: jax.Array,
    g: jax.Array,
    do: jax.Array,
    dh: jax.Array,
    scale: float,
    chunk_size: int,
) -> jax.Array:
    """Launcher for backward dv Pallas kernel.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        g: [B, T, H] — chunk-local cumsum
        do: [B, T, H, V]
        dh: [B, NT, H, K, V]
        scale: scaling factor
        chunk_size: block size

    Returns:
        dv: [B, T, H, V]
    """
    B, T, H, K = q.shape
    V = do.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    _q = q.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _k = k.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _g = g.reshape(B, NT, BT, H).transpose(3, 0, 1, 2).reshape(H, total_NT, BT, 1)
    _do = do.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _dh = dh.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)

    spec_K = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_V = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_g = pl.BlockSpec([1, 1, BT, 1], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_h = pl.BlockSpec([1, 1, K, V], index_map=lambda h, nt: (h, nt, 0, 0))

    dv_shape = jax.ShapeDtypeStruct([H, total_NT, BT, V], do.dtype)
    interpret = not is_tpu_runtime()

    dv = pl.pallas_call(
        functools.partial(_simple_gla_bwd_dv_kernel, BT=BT, scale=scale),
        grid=(H, total_NT),
        out_shape=dv_shape,
        in_specs=[spec_K, spec_K, spec_g, spec_V, spec_h],
        out_specs=spec_V,
        interpret=interpret,
    )(_q, _k, _g, _do, _dh)

    # Post-reshape: [H, total_NT, BT, V] -> [B, T, H, V]
    dv = dv.reshape(H, B, NT, BT, V)
    dv = dv.transpose(1, 0, 2, 3, 4).reshape(B, H, T, V)
    dv = dv.transpose(0, 2, 1, 3)
    return dv


# =============================================================================
# Pallas kernel: simple_gla_bwd_dqkwg (dq, dk, dg)
# =============================================================================


def _simple_gla_bwd_dqkwg_kernel(
    q_ref,
    k_ref,
    v_ref,
    g_ref,
    h_ref,
    do_ref,
    dh_ref,
    dq_ref,
    dk_ref,
    dg_ref,
    *,
    BT: int,
    scale: float,
):
    """Backward kernel: compute dq, dk, dg.

    Internally computes dA = (do @ v^T) * scale, then derives dq, dk.
    Also computes scalar gate gradient dg (w.r.t. g_cumsum).
    """
    b_q = q_ref[0, 0]  # (BT, K)
    b_k = k_ref[0, 0]  # (BT, K)
    b_v = v_ref[0, 0]  # (BT, V)
    b_g = g_ref[0, 0, :, 0].astype(jnp.float32)  # (BT,)
    b_h = h_ref[0, 0].astype(jnp.float32)  # (K, V)
    b_do = do_ref[0, 0]  # (BT, V)
    b_dh = dh_ref[0, 0].astype(jnp.float32)  # (K, V)

    b_gn = b_g[BT - 1]  # scalar

    # 1. dA = (do @ v^T) * scale, lower-triangular masked
    b_dA = (
        jnp.dot(
            b_do.astype(b_v.dtype),
            b_v.T,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        * scale
    )
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_dA = jnp.where(mask, b_dA, 0.0)

    # 2. dq
    # dq_intra = exp(g) * (dA @ k_neg), k_neg = k * exp(-g)
    k_neg = (b_k * jnp.exp(-b_g)[:, None]).astype(b_k.dtype)
    b_dq_intra = (
        jnp.dot(
            b_dA.astype(k_neg.dtype),
            k_neg,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        * jnp.exp(b_g)[:, None]
    )
    # dq_inter = scale * exp(g) * (do @ h^T)
    b_dq_inter = (
        jnp.dot(
            b_do,
            b_h.astype(b_do.dtype).T,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        * (scale * jnp.exp(b_g)[:, None])
    )
    b_dq = b_dq_intra + b_dq_inter
    dq_ref[0, 0] = b_dq.astype(dq_ref.dtype)

    # 3. dk
    # dk_intra = exp(-g) * (dA^T @ q_pos), q_pos = q * exp(g)
    q_pos = (b_q * jnp.exp(b_g)[:, None]).astype(b_q.dtype)
    b_dk_intra = (
        jnp.dot(
            b_dA.T.astype(q_pos.dtype),
            q_pos,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        * jnp.exp(-b_g)[:, None]
    )
    # dk_inter = exp(g_n - g) * (v @ dh^T)
    b_dk_inter = (
        jnp.dot(
            b_v,
            b_dh.astype(b_v.dtype).T,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        * jnp.exp(b_gn - b_g)[:, None]
    )
    b_dk = b_dk_intra + b_dk_inter
    dk_ref[0, 0] = b_dk.astype(dk_ref.dtype)

    # 4. dg (scalar gate gradient w.r.t. g_cumsum)
    # dg_raw = sum_k(q * dq - k * dk)
    dg_raw = jnp.sum(
        b_q.astype(jnp.float32) * b_dq - b_k.astype(jnp.float32) * b_dk,
        axis=-1,
    )  # (BT,)

    # dgk_inter: inter-chunk correction
    dgk_inter = jnp.exp(b_gn) * jnp.sum(b_h * b_dh) + jnp.sum(
        b_dk_inter * b_k.astype(jnp.float32)
    )

    # Reverse cumsum via upper-triangular matmul
    mask_upper = jnp.arange(BT)[None, :] >= jnp.arange(BT)[:, None]
    M_upper = jnp.where(mask_upper, 1.0, 0.0).astype(jnp.float32)
    dg_rev_cumsum = (M_upper @ dg_raw[:, None])[:, 0]  # (BT,)

    b_dg = dg_rev_cumsum + dgk_inter  # (BT,)
    dg_ref[0, 0] = b_dg[:, None].astype(dg_ref.dtype)


def simple_gla_bwd_dqkwg(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    h: jax.Array,
    do: jax.Array,
    dh: jax.Array,
    scale: float,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Launcher for backward dqkwg Pallas kernel.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H] — chunk-local cumsum
        h: [B, NT, H, K, V]
        do: [B, T, H, V]
        dh: [B, NT, H, K, V]
        scale: scaling factor
        chunk_size: block size

    Returns:
        dq: [B, T, H, K]
        dk: [B, T, H, K]
        dg: [B, T, H] — gradient w.r.t. g_cumsum
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    _q = q.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _k = k.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _v = v.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _g = g.reshape(B, NT, BT, H).transpose(3, 0, 1, 2).reshape(H, total_NT, BT, 1)
    _h = h.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)
    _do = do.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _dh = dh.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)

    spec_K = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_V = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_g = pl.BlockSpec([1, 1, BT, 1], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_h = pl.BlockSpec([1, 1, K, V], index_map=lambda h, nt: (h, nt, 0, 0))

    dq_shape = jax.ShapeDtypeStruct([H, total_NT, BT, K], q.dtype)
    dk_shape = jax.ShapeDtypeStruct([H, total_NT, BT, K], k.dtype)
    dg_shape = jax.ShapeDtypeStruct([H, total_NT, BT, 1], jnp.float32)

    interpret = not is_tpu_runtime()

    dq, dk, dg = pl.pallas_call(
        functools.partial(_simple_gla_bwd_dqkwg_kernel, BT=BT, scale=scale),
        grid=(H, total_NT),
        out_shape=[dq_shape, dk_shape, dg_shape],
        in_specs=[spec_K, spec_K, spec_V, spec_g, spec_h, spec_V, spec_h],
        out_specs=[spec_K, spec_K, spec_g],
        interpret=interpret,
    )(_q, _k, _v, _g, _h, _do, _dh)

    def _unreshape_K(x):
        x = x.reshape(H, B, NT, BT, K)
        x = x.transpose(1, 0, 2, 3, 4).reshape(B, H, T, K)
        return x.transpose(0, 2, 1, 3)

    dq = _unreshape_K(dq)
    dk = _unreshape_K(dk)

    # dg: [H, total_NT, BT, 1] -> [B, T, H]
    dg = dg.reshape(H, B, NT, BT, 1)
    dg = dg.transpose(1, 0, 2, 3, 4).reshape(B, H, T, 1)
    dg = dg.transpose(0, 2, 1, 3)[:, :, :, 0]

    return dq, dk, dg


# =============================================================================
# Reference backward (for testing)
# =============================================================================


def simple_gla_bwd_ref(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g_cumsum: jax.Array,
    h: jax.Array,
    do: jax.Array,
    dh: jax.Array,
    scale: float,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Reference backward: compute dq, dk, dv, dg from g_cumsum.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g_cumsum: [B, T, H] — chunk-local cumsum of scalar gates
        h: [B, NT, H, K, V]
        do: [B, T, H, V]
        dh: [B, NT, H, K, V]
        scale: scaling factor
        chunk_size: block size

    Returns:
        dq: [B, T, H, K]
        dk: [B, T, H, K]
        dv: [B, T, H, V]
        dg: [B, T, H] — gradient w.r.t. g_cumsum
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C

    q_c = q.reshape(B, NT, C, H, K)
    k_c = k.reshape(B, NT, C, H, K)
    v_c = v.reshape(B, NT, C, H, V)
    gc_c = g_cumsum.reshape(B, NT, C, H)
    do_c = do.reshape(B, NT, C, H, V)

    gn = gc_c[:, :, -1, :]  # [B, NT, H]

    # --- dA ---
    dA = (
        jnp.einsum("bnihv,bnjhv->bnihj", do_c, v_c, precision=lax.Precision.HIGHEST)
        * scale
    )
    causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
    dA = jnp.where(causal_mask[None, None, :, None, :], dA, 0.0)

    # --- Recompute A for dv ---
    A = jnp.einsum(
        "bnihk,bnjhk->bnihj", q_c, k_c,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    # gc_c is [B, NT, C, H]. A is [B, NT, C_i, H, C_j].
    gc_i = gc_c[:, :, :, :, None]
    gc_j = gc_c.transpose(0, 1, 3, 2)[:, :, None, :, :]
    A = A * jnp.exp(gc_i - gc_j) * scale
    A_masked = jnp.where(causal_mask[None, None, :, None, :], A, 0.0)

    # --- dv ---
    dv_intra = jnp.einsum(
        "bnihj,bnihv->bnjhv", A_masked, do_c, precision=lax.Precision.HIGHEST
    )
    # gn is [B,NT,H], gc_c is [B,NT,C,H], k_c is [B,NT,C,H,K]
    k_decay = k_c * jnp.exp(gn[:, :, None, :, None] - gc_c[..., None])
    dv_inter = jnp.einsum(
        "bnchk,bnhkv->bnchv", k_decay, dh, precision=lax.Precision.HIGHEST
    )
    dv = (dv_intra + dv_inter).reshape(B, T, H, V)

    # --- dq ---
    k_neg = k_c * jnp.exp(-gc_c)[..., None]
    dq_intra = jnp.exp(gc_c)[..., None] * jnp.einsum(
        "bnihj,bnjhk->bnihk", dA, k_neg, precision=lax.Precision.HIGHEST
    )
    dq_inter = (
        scale
        * jnp.exp(gc_c)[..., None]
        * jnp.einsum("bnchv,bnhkv->bnchk", do_c, h, precision=lax.Precision.HIGHEST)
    )
    dq = (dq_intra + dq_inter).reshape(B, T, H, K)

    # --- dk ---
    q_pos = q_c * jnp.exp(gc_c)[..., None]
    dk_intra = jnp.exp(-gc_c)[..., None] * jnp.einsum(
        "bnihj,bnihk->bnjhk", dA, q_pos, precision=lax.Precision.HIGHEST
    )
    dk_inter = jnp.exp(gn[:, :, None, :, None] - gc_c[..., None]) * jnp.einsum(
        "bnchv,bnhkv->bnchk", v_c, dh, precision=lax.Precision.HIGHEST
    )
    dk = (dk_intra + dk_inter).reshape(B, T, H, K)

    # --- dg (scalar, sum over K) ---
    dq_total = (dq_intra + dq_inter)
    dk_total = (dk_intra + dk_inter)
    dg_raw = jnp.sum(q_c * dq_total - k_c * dk_total, axis=-1)  # [B, NT, C, H]

    dgk_inter = jnp.exp(gn)[..., None] * jnp.einsum(
        "bnhkv,bnhkv->bnhk", h, dh, precision=lax.Precision.HIGHEST
    )
    dgk_inter = jnp.sum(dgk_inter, axis=-1)  # [B, NT, H] — sum over K
    dgk_inter = dgk_inter + jnp.sum(dk_inter * k_c, axis=(2, -1))  # sum over C and K

    dg = (
        jnp.cumsum(dg_raw[:, :, ::-1, :], axis=2)[:, :, ::-1, :]
        + dgk_inter[:, :, None, :]
    )
    dg = dg.reshape(B, T, H)

    return dq, dk, dv, dg


# =============================================================================
# Orchestrator: chunk_simple_gla_fwd
# =============================================================================


def chunk_simple_gla_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array | None,
    g_gamma: jax.Array | None,
    scale: float,
    initial_state: jax.Array | None,
    output_final_state: bool,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """SimpleGLA forward orchestrator.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H] or None — raw scalar gates (log-space)
        g_gamma: [H] or None — constant per-head decay
        scale: scaling factor
        initial_state: [B, H, K, V] or None
        output_final_state: whether to return final state
        chunk_size: block size

    Returns:
        (g_cumsum, o, ht)
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    # Pad T to multiple of chunk_size
    T_orig = T
    if T % C != 0:
        q, k, v = (pad_to_multiple(x, C, axis=1, val=0) for x in (q, k, v))
        if g is not None:
            g = pad_to_multiple(g, C, axis=1, val=0)

    T_padded = q.shape[1]

    # Prepare g_cumsum [B, T_padded, H]
    g_cumsum = _prepare_g_cumsum(g, g_gamma, B, T_padded, H, C)

    # Broadcast g_cumsum to gk shape for h computation
    gk = jnp.broadcast_to(g_cumsum[..., None], q.shape)

    # Forward hidden state propagation (using ref for portability)
    h, ht = chunk_fwd_h_ref(
        k, v, gk=gk, h0=initial_state,
        output_final_state=output_final_state,
        chunk_size=C,
    )

    # Fused forward output
    o = simple_gla_fwd_o(q, k, v, g_cumsum, h, scale, chunk_size=C)

    # Unpad T
    o = o[:, :T_orig]

    return g_cumsum, o, ht


# =============================================================================
# Orchestrator: chunk_simple_gla_bwd
# =============================================================================


def chunk_simple_gla_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g_cumsum: jax.Array,
    scale: float,
    initial_state: jax.Array | None,
    do: jax.Array,
    dht: jax.Array | None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None]:
    """SimpleGLA backward orchestrator.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g_cumsum: [B, T, H] — chunk-local cumsum of scalar gates (from forward)
        scale: scaling factor
        initial_state: [B, H, K, V] or None
        do: [B, T, H, V] — output gradient
        dht: [N, H, K, V] or None — terminal state gradient
        chunk_size: block size

    Returns:
        (dq, dk, dv, dg_cumsum, dh0)
        dg_cumsum is the gradient w.r.t. g_cumsum (caller applies reverse cumsum
        to get gradient w.r.t. raw gates).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    # Pad T
    T_orig = T
    if T % C != 0:
        pad_T = lambda x: pad_to_multiple(x, C, axis=1, val=0)
        q, k, v, do = (pad_T(x) for x in (q, k, v, do))
        g_cumsum = pad_to_multiple(g_cumsum, C, axis=1, val=0)

    T_padded = q.shape[1]

    # Broadcast g_cumsum to gk shape
    gk = jnp.broadcast_to(g_cumsum[..., None], q.shape)

    # 1. Replay forward to get h (using ref for portability)
    h, _ = chunk_fwd_h_ref(
        k, v, gk=gk, h0=initial_state,
        output_final_state=False, chunk_size=C,
    )

    # 2. Backward hidden state gradients (using ref for portability)
    dh, dh0 = chunk_bwd_dh_ref(
        q, k, v, gk, do,
        h0=initial_state, dht=dht,
        scale=scale, chunk_size=C,
    )

    # 3. Backward dv
    dv = simple_gla_bwd_dv(q, k, g_cumsum, do, dh, scale, chunk_size=C)

    # 4. Backward dq, dk, dg
    dq, dk, dg = simple_gla_bwd_dqkwg(
        q, k, v, g_cumsum, h, do, dh, scale, chunk_size=C
    )

    # Unpad T
    dq = dq[:, :T_orig]
    dk = dk[:, :T_orig]
    dv = dv[:, :T_orig]
    dg = dg[:, :T_orig]

    return dq, dk, dv, dg, dh0


# =============================================================================
# Public API: chunk_simple_gla
# =============================================================================


def chunk_simple_gla(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    chunk_size: int = 16,
) -> tuple[jax.Array, jax.Array | None]:
    """Chunked Simple GLA — JAX/Pallas implementation.

    Simple GLA uses head-wise scalar gates (g: [B, T, H]) instead of
    element-wise gates (gk: [B, T, H, K]) used by full GLA.

    Either ``g`` or ``g_gamma`` (or both) may be provided:
    - ``g``: [B, T, H] — per-step scalar log-space gates.
    - ``g_gamma``: [H] — constant per-head decay.
    If neither is given, gates default to zero (no decay).

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H] or None — scalar gates (log-space, after logsigmoid)
        g_gamma: [H] or None — constant per-head decay
        scale: scaling factor, default K^{-0.5}
        initial_state: [N, H, K, V]
        output_final_state: whether to return final state
        chunk_size: block size, default 16

    Returns:
        o: [B, T, H, V]
        final_state: [N, H, K, V] or None
    """
    dtype = q.dtype
    q, k, v = (x.astype(jnp.float32) for x in (q, k, v))
    if g is not None:
        g = g.astype(jnp.float32)
    if g_gamma is not None:
        g_gamma = g_gamma.astype(jnp.float32)
    B, T, H, K = q.shape

    if scale is None:
        scale = K**-0.5

    _, o, ht = chunk_simple_gla_fwd(
        q,
        k,
        v,
        g,
        g_gamma,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
    )
    final_state = ht if output_final_state else None
    return o.astype(dtype), final_state
