import jax.numpy as jnp
import numpy as np


def fused_recurrent_gla_fwd(
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
        scale = K ** -0.5

    USE_G       = g is not None
    USE_G_GAMMA = g_gamma is not None
    USE_GK      = gk is not None
    USE_GV      = gv is not None

    q_f  = q.astype(jnp.float32)
    k_f  = k.astype(jnp.float32)
    v_f  = v.astype(jnp.float32)
    g_f  = g.astype(jnp.float32)       if USE_G       else None
    g_gamma_f = g_gamma.astype(jnp.float32) if USE_G_GAMMA else None
    gk_f = gk.astype(jnp.float32)      if USE_GK      else None
    gv_f = gv.astype(jnp.float32)      if USE_GV      else None

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
            k_t = k_f[b, t_idx]           # [H, K]
            v_t = v_f[b, t_idx]           # [H, V]

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


def fused_recurrent_gla(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    gk: jnp.ndarray | None = None,
    gv: jnp.ndarray | None = None,
    scale: float | None = None,
    initial_state: jnp.ndarray | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: np.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
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
        q=q, k=k, v=v,
        gk=gk, gv=gv,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )
