import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from src.utils import cdiv, align_up, pad_to_multiple


# ============================================================================
# Pure JAX reference implementation
# ============================================================================


def fused_recurrent_gla_fwd_ref(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray | None = None,
    g_gamma: jnp.ndarray | None = None,
    gk: jnp.ndarray | None = None,
    gv: jnp.ndarray | None = None,
    scale: float | None = None,
    initial_state: jnp.ndarray | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: np.ndarray | None = None,
):
    """Pure JAX implementation of fused_recurrent_gla_fwd.

    Recurrence per time step t (or reversed when reverse=True):
        h_t = decay(h_{t-1}) + k_t @ v_t
        o_t = q_t . h_t

    Decay gates (all in log domain, kernel applies exp internally):
        g        : [B, T, H]    scalar log-gate per (batch, time, head)
        g_gamma  : [H]          per-head constant log-gate (same every step)
        gk       : [B, T, H, K] key-wise log-gate
        gv       : [B, T, H, V] value-wise log-gate

    Args:
        q:             [B, T, H, K]
        k:             [B, T, H, K]
        v:             [B, T, H, V]
        g:             [B, T, H]    scalar log-gate (optional)
        g_gamma:       [H]          per-head constant log-gate (optional)
        gk:            [B, T, H, K] key-wise log-gate (optional)
        gv:            [B, T, H, V] value-wise log-gate (optional)
        scale:         scalar, default K^-0.5
        initial_state: [N, H, K, V]
        output_final_state: whether to return final hidden state
        reverse:       if True, iterate time steps from T-1 to 0
        cu_seqlens:    [N+1] cumulative sequence lengths (varlen mode, requires B=1)

    Returns:
        o:  [B, T, H, V]
        ht: [N, H, K, V] if output_final_state else None
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    if scale is None:
        scale = K**-0.5

    USE_G = g is not None
    USE_G_GAMMA = g_gamma is not None
    USE_GK = gk is not None
    USE_GV = gv is not None

    q_f = q.astype(jnp.float32)
    k_f = k.astype(jnp.float32)
    v_f = v.astype(jnp.float32)
    g_f = g.astype(jnp.float32) if USE_G else None
    g_gamma_f = g_gamma.astype(jnp.float32) if USE_G_GAMMA else None
    gk_f = gk.astype(jnp.float32) if USE_GK else None
    gv_f = gv.astype(jnp.float32) if USE_GV else None

    o = jnp.zeros((B, T, H, V), dtype=jnp.float32)

    ht_list = []

    def _run_seq(batch_idx, bos, seq_len, o_arr):
        """Run recurrence for one sequence, return (final_h, updated_o)."""
        if initial_state is not None:
            h = initial_state[batch_idx].astype(jnp.float32)  # [H, K, V]
        else:
            h = jnp.zeros((H, K, V), dtype=jnp.float32)

        time_range = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

        b = 0 if cu_seqlens is not None else batch_idx

        for i_t in time_range:
            t_idx = bos + i_t

            q_t = q_f[b, t_idx] * scale  # [H, K]
            k_t = k_f[b, t_idx]  # [H, K]
            v_t = v_f[b, t_idx]  # [H, V]

            if USE_G:
                h = h * jnp.exp(g_f[b, t_idx])[:, None, None]
            if USE_G_GAMMA:
                h = h * jnp.exp(g_gamma_f)[:, None, None]
            if USE_GK:
                h = h * jnp.exp(gk_f[b, t_idx])[:, :, None]
            if USE_GV:
                h = h * jnp.exp(gv_f[b, t_idx])[:, None, :]

            h = h + k_t[:, :, None] * v_t[:, None, :]

            o_arr = o_arr.at[b, t_idx].set((h * q_t[:, :, None]).sum(1))

        return h, o_arr

    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
        for i_n in range(N):
            bos = int(cu_seqlens[i_n])
            eos = int(cu_seqlens[i_n + 1])
            h_final, o = _run_seq(i_n, bos, eos - bos, o)
            if output_final_state:
                ht_list.append(h_final)
    else:
        for i_n in range(B):
            h_final, o = _run_seq(i_n, 0, T, o)
            if output_final_state:
                ht_list.append(h_final)

    ht = jnp.stack(ht_list, axis=0) if output_final_state else None
    return o, ht


# ============================================================================
# Pallas kernel
# ============================================================================
# TODO(0xaskr) support large tokens by tiling in time dimension
def _fused_recurrent_gla_fwd_kernel(
    # in
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    gk: jax.Array,
    gv: jax.Array,
    h0: jax.Array,
    seqlens: jax.Array,
    # out
    o: jax.Array,
    ht: jax.Array,
    # static args
    SCALE: float,
    T: int,
    B: int,
    H: int,
    K: int,
    V: int,
    BK: int,
    BV: int,
    USE_G: bool,
    USE_GK: bool,
    USE_GV: bool,
    USE_INIT_STATE: bool,
    OUTPUT_FINAL_STATE: bool,
    REVERSE: bool,
    USE_SEQLENS: bool,
):
    assert K % BK == 0, f"K must be a multiple of BK={BK}, got {K}"
    assert V % BV == 0, f"V must be a multiple of BV={BV}, got {V}"
    q = q.reshape(T, BK)
    k = k.reshape(T, BK)
    v = v.reshape(T, BV)
    if USE_GK:
        gk = gk.reshape(T, BK)
    if USE_GV:
        gv = gv.reshape(T, BV)
    if USE_INIT_STATE:
        h0 = h0.reshape(BK, BV)

    if USE_SEQLENS:
        idx_nh = pl.program_id(2)
        seq_idx = idx_nh // H
        bos = seqlens[seq_idx]
        eos = seqlens[seq_idx + 1]

        # Zero-init the output block: on TPU, output blocks are not
        # guaranteed to be zero-initialized.  Each varlen grid cell only
        # writes to [bos, eos), so the remaining positions must be
        # explicitly cleared before the sum-over-N reduction in the launcher.
        def _zero_output(t, _):
            o[0, 0, 0, t, 0:BV] = jnp.zeros(BV, dtype=o.dtype)
            return _

        jax.lax.fori_loop(0, T, _zero_output, None)
    else:
        bos = 0
        eos = T

    b_h = jnp.zeros((BK, BV), dtype=jnp.float32)
    if USE_INIT_STATE:
        b_h += h0[...].astype(jnp.float32)

    def body(idx_t, b_h):
        if REVERSE:
            t = (eos - 1) - (idx_t - bos)
        else:
            t = idx_t
        if USE_GK:
            b_gk = gk[t, 0:BK].astype(jnp.float32)
            b_h = b_h * jnp.exp(b_gk[:, None])
        if USE_GV:
            b_gv = gv[t, 0:BV].astype(jnp.float32)
            b_h = b_h * jnp.exp(b_gv[None, :])
        b_q = q[t, 0:BK].astype(jnp.float32) * SCALE
        b_k = k[t, 0:BK].astype(jnp.float32)
        b_v = v[t, 0:BV].astype(jnp.float32)

        b_h += b_k[:, None] * b_v[None, :]
        b_o = b_h * b_q[:, None]
        b_o = jnp.sum(b_o, axis=0)
        o[0, 0, 0, t, 0:BV] = b_o.astype(o.dtype)
        return b_h

    b_h = jax.lax.fori_loop(bos, eos, body, b_h)

    if OUTPUT_FINAL_STATE:
        ht[0, 0, :, :] = b_h.astype(ht.dtype)


# ============================================================================
# Pallas launcher
# ============================================================================


def _fused_recurrent_gla_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float,
    gk: jax.Array | None = None,
    gv: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: jax.Array | None = None,
    use_gk: bool = False,
    use_gv: bool = False,
    use_init_state: bool = False,
    use_final_state: bool = False,
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    N = B if cu_seqlens is None else cu_seqlens.shape[0] - 1
    BK, BV = 128, 128
    NK, NV = cdiv(K, BK), cdiv(V, BV)
    origin_K = K
    origin_V = V
    K = align_up(K, BK)
    V = align_up(V, BV)

    h0 = initial_state
    ht = jnp.zeros((N, H, K, V), dtype=jnp.float32)
    o = jnp.zeros([NK, H, N, T, V], dtype=jnp.float32)
    o_spec = jax.ShapeDtypeStruct(o.shape, o.dtype)
    ht_spec = jax.ShapeDtypeStruct(ht.shape, ht.dtype)

    # [B, T, H, K] -> [H, B, T, K]
    q_trans = pad_to_multiple(q, BK, 3, 0).transpose(2, 0, 1, 3)
    k_trans = pad_to_multiple(k, BK, 3, 0).transpose(2, 0, 1, 3)
    v_trans = pad_to_multiple(v, BV, 3, 0).transpose(2, 0, 1, 3)
    gk_trans = pad_to_multiple(gk, BK, 3, 0).transpose(2, 0, 1, 3) if use_gk else None
    gv_trans = pad_to_multiple(gv, BV, 3, 0).transpose(2, 0, 1, 3) if use_gv else None
    # h0: [N, H, K, V] -> transpose(1,0,2,3) -> [H, N, K, V]
    h0_trans = (
        pad_to_multiple(h0, [BK, BV], [2, 3], 0).transpose(1, 0, 2, 3)
        if use_init_state
        else None
    )

    USE_SEQLENS = cu_seqlens is not None

    if USE_SEQLENS:
        # varlen: B=1, batch dim is always 0
        def qk_index_map(idx_v, idx_k, idx_nh):
            return (idx_nh % H, 0, 0, idx_k)

        def v_index_map(idx_v, idx_k, idx_nh):
            return (idx_nh % H, 0, 0, idx_v)
    else:

        def qk_index_map(idx_v, idx_k, idx_nh):
            return (idx_nh % H, idx_nh // H, 0, idx_k)

        def v_index_map(idx_v, idx_k, idx_nh):
            return (idx_nh % H, idx_nh // H, 0, idx_v)

    # o: [NK, H, N, T, V] — N=B for non-varlen, N=num_seqs for varlen
    def o_index_map(idx_v, idx_k, idx_nh):
        return (idx_k, idx_nh % H, idx_nh // H, 0, idx_v)

    # h0_trans: [H, N, K, V]
    def h0_index_map(idx_v, idx_k, idx_nh):
        return (idx_nh % H, idx_nh // H, idx_k, idx_v)

    # ht: [N, H, K, V] (N and H swapped vs h0_trans)
    def ht_index_map(idx_v, idx_k, idx_nh):
        return (idx_nh // H, idx_nh % H, idx_k, idx_v)

    q_blockspec = pl.BlockSpec([1, 1, T, BK], qk_index_map)
    k_blockspec = pl.BlockSpec([1, 1, T, BK], qk_index_map)
    v_blockspec = pl.BlockSpec([1, 1, T, BV], v_index_map)
    gk_blockspec = pl.BlockSpec([1, 1, T, BK], qk_index_map) if use_gk else None
    gv_blockspec = pl.BlockSpec([1, 1, T, BV], v_index_map) if use_gv else None
    h0_blockspec = (
        pl.BlockSpec([1, 1, BK, BV], h0_index_map) if use_init_state else None
    )
    seqlens_blockspec = (
        pl.BlockSpec([N + 1], lambda *idx: (0,), memory_space=pltpu.TPUMemorySpace.SMEM)
        if USE_SEQLENS
        else None
    )

    o_blockspec = pl.BlockSpec([1, 1, 1, T, BV], o_index_map)
    ht_blockspec = pl.BlockSpec([1, 1, BK, BV], ht_index_map)
    call_func = functools.partial(
        _fused_recurrent_gla_fwd_kernel,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_G=False,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        USE_INIT_STATE=initial_state is not None,
        OUTPUT_FINAL_STATE=output_final_state,
        REVERSE=reverse,
        SCALE=scale,
        USE_SEQLENS=USE_SEQLENS,
    )

    grid = (NV, NK, N * H)
    results = pl.pallas_call(
        call_func,
        out_shape=[o_spec, ht_spec],
        grid=grid,
        in_specs=[
            q_blockspec,
            k_blockspec,
            v_blockspec,
            gk_blockspec,
            gv_blockspec,
            h0_blockspec,
            seqlens_blockspec,
        ],
        out_specs=[o_blockspec, ht_blockspec],
    )(q_trans, k_trans, v_trans, gk_trans, gv_trans, h0_trans, cu_seqlens)
    o, ht = results
    o = o.transpose(0, 2, 3, 1, 4)  # [NK, H, N, T, V] -> [NK, N, T, H, V]
    o = o.sum(0)  # [N, T, H, V]
    if USE_SEQLENS:
        # In varlen mode, each seq wrote to non-overlapping T ranges; sum over N to merge
        o = o.sum(0, keepdims=True)  # [1, T, H, V]
    o = o[:, :, :, :origin_V]  # [B, T, H, V]
    ht = ht[:, :, :origin_K, :origin_V] if use_final_state else None
    return o, ht


# ============================================================================
# Entry point: validation + dispatch
# ============================================================================


def fused_recurrent_gla_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    gk: jax.Array | None = None,
    gv: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: jax.Array | None = None,
):
    B, T, H, K = q.shape
    V = v.shape[-1]
    N = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else B
    scale = scale if scale is not None else K**-0.5

    assert k.shape == (B, T, H, K), f"Expected k shape {(B, T, H, K)}, got {k.shape}"
    assert v.shape == (B, T, H, V), f"Expected v shape {(B, T, H, V)}, got {v.shape}"
    assert (gk is None) or (gk.shape == (B, T, H, K)), (
        f"Expected gk shape {(B, T, H, K)}, got {gk.shape}"
    )
    assert (gv is None) or (gv.shape == (B, T, H, V)), (
        f"Expected gv shape {(B, T, H, V)}, got {gv.shape}"
    )
    assert (initial_state is None) or (initial_state.shape == (N, H, K, V)), (
        f"Expected initial_state shape {(N, H, K, V)}, got {initial_state.shape}"
    )
    assert (cu_seqlens is None) or (B == 1), (
        f"Batch size must be 1 when using cu_seqlens, got {B}"
    )
    assert scale is not None, (
        "ignore pylance warning about unused variable `scale`, which is actually used in the kernel call"
    )

    o, ht = _fused_recurrent_gla_fwd(
        q=q,
        k=k,
        v=v,
        gk=gk,
        gv=gv,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        use_gk=gk is not None,
        use_gv=gv is not None,
        use_init_state=initial_state is not None,
        use_final_state=output_final_state,
    )
    return o, ht


# ============================================================================
# Public API matching FLA signature
# ============================================================================


def fused_recurrent_gla(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    gk: jax.Array | None = None,
    gv: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Fused recurrent GLA — public API matching FLA signature.

    Wraps fused_recurrent_gla_fwd, mapping gk/gv to the internal API.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        gk: [B, T, H, K] key-wise log-gate (optional)
        gv: [B, T, H, V] value-wise log-gate (optional)
        scale: scalar, default K^-0.5
        initial_state: [N, H, K, V]
        output_final_state: whether to return final hidden state
        reverse: if True, iterate time steps from T-1 to 0
        cu_seqlens: [N+1] cumulative sequence lengths

    Returns:
        o: [B, T, H, V]
        ht: [N, H, K, V] if output_final_state else None
    """
    return fused_recurrent_gla_fwd(
        q=q,
        k=k,
        v=v,
        gk=gk,
        gv=gv,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )
