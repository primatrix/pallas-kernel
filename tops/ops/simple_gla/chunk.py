"""Simple GLA (g_gamma only) backward orchestrator using Pallas kernels.

Reuses existing infrastructure:
  - chunk_fwd_h_kernel (supports g_gamma natively)
  - chunk_bwd_dh_kernel (via synthetic gk from g_gamma)
  - chunk_gla_fwd_intra_gk (via synthetic g_cumsum from g_gamma)
  - chunk_simple_gla_bwd_o_pl (fused dq/dk/dv kernel)
"""

import jax
import jax.numpy as jnp

from tops.utils import pad_to_multiple
from tops.ops.common.chunk_h import chunk_fwd_h_kernel, chunk_bwd_dh_kernel
from tops.ops.gla.chunk import chunk_gla_fwd_intra_gk_ref
from tops.ops.common.chunk_o import chunk_simple_gla_bwd_o_pl


def _build_gk_from_gamma(g_gamma: jax.Array, B: int, T: int, H: int, K: int, chunk_size: int) -> jax.Array:
    """Build equivalent gk [B, T, H, K] from g_gamma [H] for reuse with existing kernels.

    For each chunk of size C, position t_in_chunk gets gate value gamma[h] * (t_in_chunk + 1).
    This is the chunk-local cumsum of a constant per-step gate of gamma[h].
    """
    C = chunk_size
    NT = T // C
    # pos = [1, 2, ..., C] tiled NT times -> [T]
    pos = jnp.tile(jnp.arange(1, C + 1), NT)  # [T]
    # gc[t, h] = gamma[h] * pos[t_in_chunk]
    gc = pos[:, None] * g_gamma[None, :]  # [T, H]
    # broadcast to [B, T, H, K]
    return jnp.broadcast_to(gc[None, :, :, None], (B, T, H, K))


def chunk_simple_gla_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g_gamma: jax.Array,
    scale: float,
    initial_state: jax.Array | None,
    do: jax.Array,
    dht: jax.Array | None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
    """Simple GLA backward orchestrator (g_gamma only).

    Args:
        q:  [B, T, H, K]
        k:  [B, T, H, K]
        v:  [B, T, H, V]
        g_gamma: [H] — fixed per-head log-decay
        scale: scaling factor
        initial_state: [B, H, K, V] or None
        do: [B, T, H, V] — output gradient
        dht: [B, H, K, V] or None — terminal state gradient
        chunk_size: block size

    Returns:
        (dq, dk, dv, dh0)
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    # --- T padding ---
    orig_T = T
    NT = (T + C - 1) // C
    T_padded = NT * C
    if T_padded > T:
        pad = T_padded - T
        pad_width = ((0, 0), (0, pad), (0, 0), (0, 0))
        q = jnp.pad(q, pad_width)
        k = jnp.pad(k, pad_width)
        v = jnp.pad(v, pad_width)
        do = jnp.pad(do, pad_width)
        T = T_padded

    # --- K/V padding to multiple of 128 ---
    orig_K, orig_V = K, V
    q, k = (pad_to_multiple(x, 128, axis=3, val=0) for x in (q, k))
    v = pad_to_multiple(v, 128, axis=3, val=0)
    do = pad_to_multiple(do, 128, axis=3, val=0)
    K, V = q.shape[-1], v.shape[-1]
    if initial_state is not None:
        initial_state = pad_to_multiple(initial_state, [128, 128], axis=[2, 3], val=0)

    NT = T // C

    # Build synthetic gk from g_gamma for kernels that need it
    gk = _build_gk_from_gamma(g_gamma, B, T, H, K, C)

    # 1. Recompute h via chunk_fwd_h_kernel
    h, _ = chunk_fwd_h_kernel(
        k, v,
        g=None,
        g_gamma=g_gamma,
        gk=None,
        h0=initial_state,
        output_final_state=False,
        chunk_size=C,
    )
    h = h.reshape(B, NT, H, K, V)

    # 2. Compute dh via chunk_bwd_dh_kernel with synthetic gk
    dh, dh0 = chunk_bwd_dh_kernel(
        q, k, v,
        gk=gk,
        do=do,
        dht=dht,
        scale=scale,
        chunk_size=C,
    )
    dh = dh.reshape(B, NT, H, K, V)
    if dh0 is not None:
        dh0 = dh0.reshape(B, H, K, V)

    # 3. Compute A via existing intra-chunk attention with synthetic gk
    A = chunk_gla_fwd_intra_gk_ref(q, k, gk, scale, chunk_size=C)

    # 4. Fused dq/dk/dv via simple GLA pallas kernel
    dq, dk, dv = chunk_simple_gla_bwd_o_pl(
        q, k, v, g_gamma, h, A, do, dh,
        scale=scale, chunk_size=C,
    )

    # --- Unpad ---
    dq = dq[:, :orig_T, :, :orig_K]
    dk = dk[:, :orig_T, :, :orig_K]
    dv = dv[:, :orig_T, :, :orig_V]
    if dh0 is not None:
        dh0 = dh0[:, :, :orig_K, :orig_V]

    return dq, dk, dv, dh0
