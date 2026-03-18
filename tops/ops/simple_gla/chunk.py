# pallas-kernel/tops/ops/simple_gla/chunk.py
import functools

import jax
import jax.experimental.pallas as pl
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu
from tops.utils import pad_to_multiple
from tops.ops.common.chunk_h import chunk_fwd_h_kernel, chunk_fwd_h_ref
from tops.ops.gla.chunk import (
    chunk_gla_bwd,
    chunk_gla_fwd_intra_gk_ref,
    chunk_gla_fwd_o_gk_ref,
    chunk_local_cumsum_ref,
)


# =============================================================================
# Reference implementations (pure JAX, no Pallas)
# =============================================================================


def chunk_simple_gla_fwd_intra_ref(
    q: jax.Array,
    k: jax.Array,
    g_gamma: jax.Array,
    scale: float,
    chunk_size: int = 64,
) -> jax.Array:
    """Intra-chunk attention for Simple GLA (reference, pure JAX).

    Uses standard matmul + Toeplitz decay mask instead of per-K-dim gating.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        g_gamma: (1, 1, H, 1) or (H,) — constant scalar gate per head
        scale: scaling factor
        chunk_size: block size

    Returns:
        A: [B, T, H, C] — intra-chunk attention matrix
    """
    B, T, H, K = q.shape
    C = chunk_size
    NT = T // C

    q_c = q.reshape(B, NT, C, H, K)
    k_c = k.reshape(B, NT, C, H, K)

    # Standard attention (no per-element gating)
    A = jnp.einsum(
        "bnihk,bnjhk->bnihj", q_c, k_c,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ) * scale

    # Toeplitz decay mask: exp(g_gamma[h] * (i - j))
    g_h = g_gamma.reshape(H)
    pos = jnp.arange(1, C + 1, dtype=jnp.float32)
    # decay[i, h, j] = exp(g_h[h] * (pos[i] - pos[j]))
    decay = jnp.exp(g_h[None, :, None] * (pos[:, None, None] - pos[None, None, :]))
    A = A * decay[None, None]  # broadcast over B, NT

    A = A.reshape(B, T, H, C)
    return A


def chunk_simple_gla_fwd_o_ref(
    q: jax.Array,
    v: jax.Array,
    g_gamma: jax.Array,
    A: jax.Array,
    h: jax.Array,
    scale: float,
    chunk_size: int = 64,
) -> jax.Array:
    """Output combination for Simple GLA (reference, pure JAX).

    Inter-chunk: q @ h * exp(g_gamma * pos) * scale
    Intra-chunk: tril(A) @ v

    Args:
        q: [B, T, H, K]
        v: [B, T, H, V]
        g_gamma: (1, 1, H, 1) or (H,) — constant scalar gate per head
        A: [B, T, H, C] — intra-chunk attention matrix
        h: [B, NT, H, K, V] — hidden state at start of each chunk
        scale: scaling factor
        chunk_size: block size

    Returns:
        o: [B, T, H, V]
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = B * T // C

    q = q.reshape(-1, C, H, K)
    v = v.reshape(-1, C, H, V)
    h = h.reshape(-1, H, K, V)
    A = A.reshape(-1, C, H, C)

    # Inter-chunk: scale * q * exp(g_cumsum) @ h
    g_h = g_gamma.reshape(H)
    pos = jnp.arange(1, C + 1, dtype=jnp.float32)
    g_exp = jnp.exp(g_h[None, :] * pos[:, None])  # (C, H)
    qg = q * g_exp[None, :, :, None]  # (NT, C, H, K) * (1, C, H, 1)

    o_inter = scale * jnp.einsum("nchk,nhkv->nchv", qg, h)

    # Intra-chunk: tril(A) @ v
    causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))[:, None, :]
    n_A = jnp.where(causal_mask, A, 0.0)
    o_intra = jnp.einsum("nihj,njhv->nihv", n_A, v)

    o = (o_inter + o_intra).reshape(B, T, H, V)
    return o


def chunk_simple_gla_fwd_ref(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g_gamma: jax.Array,
    scale: float,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> tuple[jax.Array | None, jax.Array]:
    """Full Simple GLA forward (reference, pure JAX).

    Returns:
        (ht, o) — final state and output
    """
    B, T, H, K = q.shape
    C = chunk_size

    # Pad T
    if T % C != 0:
        q, k, v = (pad_to_multiple(x, C, axis=1, val=0) for x in (q, k, v))

    g_gamma_1d = g_gamma.reshape(-1)
    assert g_gamma_1d.shape[0] == H

    # Stage 1: state propagation using g_gamma
    _, T_pad = q.shape[:2]
    pos = jnp.arange(1, C + 1, dtype=jnp.float32)
    pos = jnp.tile(pos, T_pad // C).reshape(1, T_pad, 1, 1)
    g_cumsum = jnp.broadcast_to(g_gamma * pos, q.shape)

    h, ht = chunk_fwd_h_ref(
        k, v, gk=g_cumsum, h0=initial_state,
        output_final_state=output_final_state, chunk_size=C,
    )

    # Stage 2: intra-chunk attention (Simple GLA)
    A = chunk_simple_gla_fwd_intra_ref(q, k, g_gamma, scale, chunk_size=C)

    # Stage 3: output (Simple GLA)
    o = chunk_simple_gla_fwd_o_ref(q, v, g_gamma, A, h, scale, chunk_size=C)

    o = o[:, :T]
    return ht, o
