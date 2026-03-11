import torch
import torch.nn.functional as F


# =============================================================================
# Sub-function 1: chunk_local_cumsum
# =============================================================================


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """Chunk-local cumulative sum of gates.

    Corresponds to FLA ``fla.ops.utils.cumsum``.

    Args:
        g: [B, T, H, K] — log-space gates (T must be a multiple of chunk_size)
        chunk_size: block size
        cu_seqlens: unused, kept for FLA interface compatibility

    Returns:
        g_cumsum: [B, T, H, K] — chunk-local cumsum
    """
    B, T, H, K = g.shape
    C = chunk_size
    NT = T // C
    g_cumsum = g.view(B, NT, C, H, K).cumsum(dim=2).view(B, T, H, K)
    return g_cumsum


# =============================================================================
# Sub-function 2: chunk_fwd_h
# =============================================================================


def chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    h0: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Inter-chunk hidden state propagation.

    Corresponds to FLA ``fla.ops.common.chunk_h.chunk_fwd_h``.

    Computes the hidden state at the **start** of each chunk by
    sequentially propagating through chunks.

    Args:
        k:  [B, T, H, K] — keys (T must be a multiple of chunk_size)
        v:  [B, T, H, V] — values
        gk: [B, T, H, K] — chunk-local cumsum of gates
        h0: [B, H, K, V] — initial hidden state (optional)
        output_final_state: whether to return final state
        cu_seqlens: unused, kept for FLA interface compatibility
        chunk_size: block size

    Returns:
        h:  [B, NT, H, K, V] — hidden state at the start of each chunk
        ht: [B, H, K, V] or None — final hidden state
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C

    k_c = k.view(B, NT, C, H, K)
    v_c = v.view(B, NT, C, H, V)
    gk_c = gk.view(B, NT, C, H, K)

    h = k.new_zeros(B, H, K, V, dtype=torch.float32)
    if h0 is not None:
        h = h + h0.float()

    h_list = []
    for i in range(NT):
        h_list.append(h.clone())

        gc = gk_c[:, i]  # [B, C, H, K]
        ki = k_c[:, i]  # [B, C, H, K]
        vi = v_c[:, i]  # [B, C, H, V]

        g_total = gc[:, -1]  # [B, H, K]

        # h = h * exp(g_total) + Σ_j k_j * exp(g_total - gc_j) ⊗ v_j
        h = h * g_total.unsqueeze(-1).exp()
        k_state = ki * (g_total.unsqueeze(1) - gc).exp()  # [B, C, H, K]
        h = h + torch.einsum("bchk,bchv->bhkv", k_state, vi)

    h_all = torch.stack(h_list, dim=1)  # [B, NT, H, K, V]
    ht = h if output_final_state else None
    return h_all, ht


# =============================================================================
# Sub-function 3: chunk_gla_fwd_intra_gk
# =============================================================================


def chunk_gla_fwd_intra_gk(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Intra-chunk attention matrix with causal mask.

    Corresponds to FLA ``fla.ops.gla.chunk.chunk_gla_fwd_intra_gk``.

    Args:
        q: [B, T, H, K] — queries (T must be a multiple of chunk_size)
        k: [B, T, H, K] — keys
        g: [B, T, H, K] — chunk-local cumsum of gates
        scale: scaling factor
        cu_seqlens: unused, kept for FLA interface compatibility
        chunk_size: block size

    Returns:
        A: [B, NT, H, C, C] — intra-chunk causal attention matrix
    """
    B, T, H, K = q.shape
    C = chunk_size
    NT = T // C

    q_c = q.view(B, NT, C, H, K)
    k_c = k.view(B, NT, C, H, K)
    g_c = g.view(B, NT, C, H, K)

    q_gated = q_c * g_c.exp()  # [B, NT, C, H, K]
    k_gated = k_c * (-g_c).exp()  # [B, NT, C, H, K]

    # A[b,n,h,i,j] = scale * Σ_k q_gated[b,n,i,h,k] * k_gated[b,n,j,h,k]
    A = torch.einsum("bnihk,bnjhk->bnihj", q_gated, k_gated) * scale  # [B, NT, H, C, C]

    return A


# =============================================================================
# Sub-function 4: chunk_gla_fwd_o_gk
# =============================================================================


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Combine inter-chunk and intra-chunk contributions to produce output.

    Corresponds to FLA ``fla.ops.gla.chunk.chunk_gla_fwd_o_gk``.

    When ``cu_seqlens`` is ``None`` (default), T must be a multiple of
    chunk_size.  When ``cu_seqlens`` is provided, the inputs are packed
    variable-length sequences (B must be 1).  ``q``, ``v``, ``g`` are
    packed along dim-1 (total length = sum of sequence lengths), while
    ``A`` and ``h`` are packed along the chunk dimension (dim-1) with
    each sequence contributing ``ceil(Li / C)`` chunks.

    Args:
        q: [B, T, H, K] — queries
        v: [B, T, H, V] — values
        g: [B, T, H, K] — chunk-local cumsum of gates
        A: [B, NT, H, C, C] — intra-chunk attention matrix
        h: [B, NT, H, K, V] — hidden state at start of each chunk
        scale: scaling factor
        cu_seqlens: [N+1] cumulative sequence lengths for variable-length mode
        chunk_size: block size

    Returns:
        o: [B, T, H, V]
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = B * T // C
    assert T % C == 0, "T must be a multiple of chunk_size for chunk_gla_fwd_o_gk_ref"
    assert (cu_seqlens is None) or (cu_seqlens % C == 0).all(), (
        "cu_seqlens must be multiples of chunk_size for chunk_fwd_h"
    )

    q = q.reshape(-1, C, H, K)
    v = v.reshape(-1, C, H, V)
    gk = gk.reshape(-1, C, H, K)
    h = h.reshape(-1, H, K, V)
    A = A.reshape(-1, C, H, C)

    qg = q * gk.exp()

    # Inter-chunk: o_inter = scale * (q_gated @ h)
    o_inter = scale * torch.einsum(
        "nchk,nhkv->nchv", qg, h
    )  # [C, K] @ [K, V] -> [C, V]

    causal_mask = torch.tril(torch.ones((C, C), dtype=torch.bool))[
        :, None, :
    ]  # (C, 1, C) → broadcasts to (NT, C, H, C)
    n_A = torch.where(causal_mask, A, 0.0)

    # [C, C] @ [C, V] -> [C, V]
    # Intra-chunk: o_intra = A @ v, contract over j (key position within chunk)
    o_intra = torch.einsum("nihj,njhv->nihv", n_A, v)

    o = (o_inter + o_intra).reshape(B, T, H, V)
    return o


# =============================================================================
# Orchestrator: chunk_gla_fwd
# =============================================================================


def chunk_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_cumsum: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Chunk GLA forward orchestrator.

    Pads inputs to a multiple of chunk_size, then calls the 4 sub-functions
    in order: chunk_local_cumsum → chunk_fwd_h → chunk_gla_fwd_intra_gk →
    chunk_gla_fwd_o_gk.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H, K] — raw log-space gates
        g_cumsum: [B, T_padded, H, K] or None — pre-computed chunk-local cumsum
        scale: scaling factor
        initial_state: [B, H, K, V] or None
        output_final_state: whether to return final hidden state
        cu_seqlens: unused, kept for FLA interface compatibility
        chunk_size: block size

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
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        g = F.pad(g, (0, 0, 0, 0, 0, pad))

    if g_cumsum is None:
        g_cumsum = chunk_local_cumsum(g, C)

    h, ht = chunk_fwd_h(
        k,
        v,
        g_cumsum,
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
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked GLA — pure PyTorch CPU implementation.

    Splits the sequence into blocks of chunk_size and computes in parallel
    within each block, propagating hidden states across blocks.
    Mathematically equivalent to naive_recurrent_gla.

    The gate parameter is named ``g`` (matching the FLA chunk_gla API),
    not ``gk`` as in fused_recurrent_gla.

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
    q, k, v, g = (x.float() for x in (q, k, v, g))
    B, T, H, K = q.shape

    if scale is None:
        scale = K**-0.5

    if cu_seqlens is not None:
        assert B == 1, "cu_seqlens requires B=1"
        N = len(cu_seqlens) - 1
        o = torch.zeros_like(v)  # [1, T, H, V]
        final_states = [] if output_final_state else None

        for i in range(N):
            bos = cu_seqlens[i].item()
            eos = cu_seqlens[i + 1].item()
            h0 = initial_state[i : i + 1] if initial_state is not None else None

            _, _, _, ht_seg, o_seg = chunk_gla_fwd(
                q[:, bos:eos],
                k[:, bos:eos],
                v[:, bos:eos],
                g[:, bos:eos],
                g_cumsum=None,
                scale=scale,
                initial_state=h0,
                output_final_state=output_final_state,
                chunk_size=chunk_size,
            )
            o[:, bos:eos] = o_seg
            if output_final_state:
                final_states.append(ht_seg.squeeze(0))  # [H, K, V]

        final_state = torch.stack(final_states, dim=0) if output_final_state else None
        return o.to(dtype), final_state
    else:
        _, _, _, ht, o = chunk_gla_fwd(
            q,
            k,
            v,
            g,
            g_cumsum=None,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
        )
        final_state = ht if output_final_state else None
        return o.to(dtype), final_state
