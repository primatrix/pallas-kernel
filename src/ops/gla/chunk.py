import jax.numpy as jnp
import numpy as np


# =============================================================================
# Sub-function 1: chunk_local_cumsum
# =============================================================================

def chunk_local_cumsum(
    g: jnp.ndarray,
    chunk_size: int,
    cu_seqlens: np.ndarray | None = None,
) -> jnp.ndarray:
    """Chunk-local cumulative sum of gates.

    Args:
        g: [B, T, H, K] — log-space gates (T must be a multiple of chunk_size)
        chunk_size: block size
        cu_seqlens: unused, kept for interface compatibility

    Returns:
        g_cumsum: [B, T, H, K] — chunk-local cumsum
    """
    B, T, H, K = g.shape
    C = chunk_size
    NT = T // C
    g_cumsum = jnp.cumsum(g.reshape(B, NT, C, H, K), axis=2).reshape(B, T, H, K)
    return g_cumsum


# =============================================================================
# Sub-function 2: chunk_fwd_h
# =============================================================================

def chunk_fwd_h(
    k: jnp.ndarray,
    v: jnp.ndarray,
    gk: jnp.ndarray,
    h0: jnp.ndarray | None = None,
    output_final_state: bool = False,
    cu_seqlens: np.ndarray | None = None,
    chunk_size: int = 64,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Inter-chunk hidden state propagation.

    Computes the hidden state at the start of each chunk by
    sequentially propagating through chunks.

    Args:
        k:  [B, T, H, K] — keys (T must be a multiple of chunk_size)
        v:  [B, T, H, V] — values
        gk: [B, T, H, K] — chunk-local cumsum of gates
        h0: [B, H, K, V] — initial hidden state (optional)
        output_final_state: whether to return final state
        cu_seqlens: unused, kept for interface compatibility
        chunk_size: block size

    Returns:
        h:  [B, NT, H, K, V] — hidden state at the start of each chunk
        ht: [B, H, K, V] or None — final hidden state
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C

    k_c = k.reshape(B, NT, C, H, K)
    v_c = v.reshape(B, NT, C, H, V)
    gk_c = gk.reshape(B, NT, C, H, K)

    h = jnp.zeros((B, H, K, V), dtype=jnp.float32)
    if h0 is not None:
        h = h + h0.astype(jnp.float32)

    h_list = []
    for i in range(NT):
        h_list.append(h)

        gc = gk_c[:, i]        # [B, C, H, K]
        ki = k_c[:, i]         # [B, C, H, K]
        vi = v_c[:, i]         # [B, C, H, V]

        g_total = gc[:, -1]    # [B, H, K]

        # h = h * exp(g_total) + sum_j k_j * exp(g_total - gc_j) @ v_j
        h = h * jnp.exp(g_total[:, :, :, None])
        k_state = ki * jnp.exp(g_total[:, None, :, :] - gc)    # [B, C, H, K]
        h = h + jnp.einsum('bchk,bchv->bhkv', k_state, vi)

    h_all = jnp.stack(h_list, axis=1)    # [B, NT, H, K, V]
    ht = h if output_final_state else None
    return h_all, ht


# =============================================================================
# Sub-function 3: chunk_gla_fwd_intra_gk
# =============================================================================

def chunk_gla_fwd_intra_gk(
    q: jnp.ndarray,
    k: jnp.ndarray,
    g: jnp.ndarray,
    scale: float,
    cu_seqlens: np.ndarray | None = None,
    chunk_size: int = 64,
) -> jnp.ndarray:
    """Intra-chunk attention matrix with causal mask.

    Args:
        q: [B, T, H, K] — queries (T must be a multiple of chunk_size)
        k: [B, T, H, K] — keys
        g: [B, T, H, K] — chunk-local cumsum of gates
        scale: scaling factor
        cu_seqlens: unused, kept for interface compatibility
        chunk_size: block size

    Returns:
        A: [B, NT, H, C, C] — intra-chunk causal attention matrix
    """
    B, T, H, K = q.shape
    C = chunk_size
    NT = T // C

    q_c = q.reshape(B, NT, C, H, K)
    k_c = k.reshape(B, NT, C, H, K)
    g_c = g.reshape(B, NT, C, H, K)

    q_gated = q_c * jnp.exp(g_c)
    k_gated = k_c * jnp.exp(-g_c)

    A = jnp.einsum('bnihk,bnjhk->bnhij', q_gated, k_gated)

    causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
    A = jnp.where(causal_mask, A, 0.0)

    return A


# =============================================================================
# Sub-function 4: chunk_gla_fwd_o_gk
# =============================================================================

def chunk_gla_fwd_o_gk(
    q: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray,
    A: jnp.ndarray,
    h: jnp.ndarray,
    scale: float,
    cu_seqlens: np.ndarray | None = None,
    chunk_size: int = 64,
) -> jnp.ndarray:
    """Combine inter-chunk and intra-chunk contributions to produce output.

    Args:
        q: [B, T, H, K] — queries (T must be a multiple of chunk_size)
        v: [B, T, H, V] — values
        g: [B, T, H, K] — chunk-local cumsum of gates
        A: [B, NT, H, C, C] — intra-chunk attention matrix
        h: [B, NT, H, K, V] — hidden state at start of each chunk
        scale: scaling factor
        cu_seqlens: unused, kept for interface compatibility
        chunk_size: block size

    Returns:
        o: [B, T, H, V]
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C

    q_c = q.reshape(B, NT, C, H, K)
    v_c = v.reshape(B, NT, C, H, V)
    g_c = g.reshape(B, NT, C, H, K)

    q_gated = q_c * jnp.exp(g_c)

    # Inter-chunk: o_inter = scale * q_gated @ h
    o_inter = scale * jnp.einsum('bnchk,bnhkv->bnchv', q_gated, h)

    # Intra-chunk: o_intra = scale * A @ v
    o_intra = scale * jnp.einsum('bnhij,bnjhv->bnihv', A, v_c)

    o = (o_inter + o_intra).reshape(B, T, H, V)
    return o


# =============================================================================
# Orchestrator: chunk_gla_fwd
# =============================================================================

def chunk_gla_fwd(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray,
    g_cumsum: jnp.ndarray | None,
    scale: float,
    initial_state: jnp.ndarray | None,
    output_final_state: bool,
    cu_seqlens: np.ndarray | None = None,
    chunk_size: int = 64,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray]:
    """Chunk GLA forward orchestrator.

    Pads inputs to a multiple of chunk_size, then calls the 4 sub-functions.

    Returns:
        (g_cumsum, A, h, ht, o)
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = (T + C - 1) // C
    T_padded = NT * C

    if T_padded > T:
        pad = T_padded - T
        pad_width = ((0, 0), (0, pad), (0, 0), (0, 0))
        q = jnp.pad(q, pad_width)
        k = jnp.pad(k, pad_width)
        v = jnp.pad(v, ((0, 0), (0, pad), (0, 0), (0, 0)))
        g = jnp.pad(g, pad_width)

    if g_cumsum is None:
        g_cumsum = chunk_local_cumsum(g, C)

    h, ht = chunk_fwd_h(
        k, v, g_cumsum,
        h0=initial_state,
        output_final_state=output_final_state,
        chunk_size=C,
    )
    A = chunk_gla_fwd_intra_gk(q, k, g_cumsum, scale, chunk_size=C)
    o = chunk_gla_fwd_o_gk(q, v, g_cumsum, A, h, scale, chunk_size=C)

    o = o[:, :T]
    return g_cumsum, A, h, ht, o


# =============================================================================
# Public API: chunk_gla
# =============================================================================

def chunk_gla(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray,
    scale: float | None = None,
    initial_state: jnp.ndarray | None = None,
    output_final_state: bool = False,
    cu_seqlens: np.ndarray | None = None,
    chunk_size: int = 16,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Chunked GLA — pure JAX implementation.

    Splits the sequence into blocks of chunk_size and computes in parallel
    within each block, propagating hidden states across blocks.
    Mathematically equivalent to naive_recurrent_gla.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H, K] — gates (log-space, after logsigmoid)
        scale: scaling factor, default K^{-0.5}
        initial_state: [N, H, K, V]
        output_final_state: whether to return final state
        cu_seqlens: [N+1] variable-length cumulative lengths
        chunk_size: block size, default 16

    Returns:
        o: [B, T, H, V]
        final_state: [N, H, K, V] or None
    """
    dtype = q.dtype
    q, k, v, g = (x.astype(jnp.float32) for x in (q, k, v, g))
    B, T, H, K = q.shape

    if scale is None:
        scale = K ** -0.5

    if cu_seqlens is not None:
        assert B == 1, "cu_seqlens requires B=1"
        N = len(cu_seqlens) - 1
        o = jnp.zeros_like(v)   # [1, T, H, V]
        final_states = [] if output_final_state else None

        for i in range(N):
            bos = int(cu_seqlens[i])
            eos = int(cu_seqlens[i + 1])
            h0 = initial_state[i:i + 1] if initial_state is not None else None

            _, _, _, ht_seg, o_seg = chunk_gla_fwd(
                q[:, bos:eos], k[:, bos:eos],
                v[:, bos:eos], g[:, bos:eos],
                g_cumsum=None, scale=scale,
                initial_state=h0,
                output_final_state=output_final_state,
                chunk_size=chunk_size,
            )
            o = o.at[:, bos:eos].set(o_seg)
            if output_final_state:
                final_states.append(ht_seg[0])   # [H, K, V]

        final_state = jnp.stack(final_states, axis=0) if output_final_state else None
        return o.astype(dtype), final_state
    else:
        _, _, _, ht, o = chunk_gla_fwd(
            q, k, v, g,
            g_cumsum=None, scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
        )
        final_state = ht if output_final_state else None
        return o.astype(dtype), final_state
