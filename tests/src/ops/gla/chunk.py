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
# Backward sub-function 1: chunk_bwd_dh
# =============================================================================


def chunk_bwd_dh(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    do: torch.Tensor,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float = 1.0,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Backward hidden state gradient propagation.

    Args:
        q:   [B, T, H, K]
        k:   [B, T, H, K]
        v:   [B, T, H, V]
        gk:  [B, T, H, K] — chunk-local cumsum of gates
        do:  [B, T, H, V] — output gradient
        h0:  [B, H, K, V] — initial hidden state (optional)
        dht: [B, H, K, V] — terminal state gradient (optional)
        scale: scaling factor
        cu_seqlens: unused
        chunk_size: block size

    Returns:
        dh:  [B, NT, H, K, V]
        dh0: [B, H, K, V] or None
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C

    q_c = q.view(B, NT, C, H, K)
    do_c = do.view(B, NT, C, H, V)
    gk_c = gk.view(B, NT, C, H, K)

    dh = q.new_zeros(B, H, K, V, dtype=torch.float32)
    if dht is not None:
        dh = dh + dht.float()

    dh_list = [None] * NT
    for i in range(NT - 1, -1, -1):
        dh_list[i] = dh.clone()

        b_q = q_c[:, i]     # [B, C, H, K]
        b_do = do_c[:, i]   # [B, C, H, V]
        gc = gk_c[:, i]     # [B, C, H, K]
        g_total = gc[:, -1]  # [B, H, K]

        b_q_hat = b_q * gc.exp() * scale  # [B, C, H, K]
        dh = dh * g_total.unsqueeze(-1).exp()
        dh = dh + torch.einsum("bchk,bchv->bhkv", b_q_hat, b_do)

    dh_all = torch.stack(dh_list, dim=1)  # [B, NT, H, K, V]
    dh0 = dh if (h0 is not None or dht is not None) else None
    return dh_all, dh0


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


# =============================================================================
# Backward sub-function 2: chunk_gla_bwd_dA
# =============================================================================


def chunk_gla_bwd_dA(
    v: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Gradient of the intra-chunk attention matrix.

    Args:
        v:  [B, T, H, V]
        do: [B, T, H, V]
        scale: scaling factor
        chunk_size: block size

    Returns:
        dA: [B, T, H, C] — lower-triangular masked
    """
    B, T, H, V = v.shape
    C = chunk_size
    NT = T // C

    v_c = v.view(B, NT, C, H, V)
    do_c = do.view(B, NT, C, H, V)

    dA = torch.einsum("bnihv,bnjhv->bnihj", do_c, v_c) * scale
    causal_mask = torch.tril(torch.ones(C, C, dtype=torch.bool))
    dA = torch.where(causal_mask[None, None, :, None, :], dA, 0.0)
    dA = dA.reshape(B, T, H, C)
    return dA


# =============================================================================
# Backward sub-function 3: chunk_gla_bwd_dv
# =============================================================================


def chunk_gla_bwd_dv(
    k: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Gradient of v.

    Args:
        k:        [B, T, H, K]
        g_cumsum: [B, T, H, K]
        A:        [B, T, H, C] — intra-chunk attention matrix
        do:       [B, T, H, V]
        dh:       [B, NT, H, K, V]
        chunk_size: block size

    Returns:
        dv: [B, T, H, V]
    """
    B, T, H, K = k.shape
    V = do.shape[-1]
    C = chunk_size
    NT = T // C

    k_c = k.view(B, NT, C, H, K)
    gc_c = g_cumsum.view(B, NT, C, H, K)
    do_c = do.view(B, NT, C, H, V)
    A_c = A.view(B, NT, C, H, C)

    # Intra: dv[j] = sum_{i>=j} A[i,j] * do[i]
    # A is lower-triangular (nonzero when i >= j), keep those entries
    causal_mask = torch.tril(torch.ones(C, C, dtype=torch.bool))
    A_masked = torch.where(causal_mask[None, None, :, None, :], A_c, 0.0)
    dv_intra = torch.einsum("bnihj,bnihv->bnjhv", A_masked, do_c)

    # Inter: k_decay @ dh
    gn = gc_c[:, :, -1, :, :]  # [B, NT, H, K]
    k_decay = k_c * (gn[:, :, None, :, :] - gc_c).exp()
    dv_inter = torch.einsum("bnchk,bnhkv->bnchv", k_decay, dh)

    dv = (dv_intra + dv_inter).reshape(B, T, H, V)
    return dv


# =============================================================================
# Backward sub-function 4: chunk_gla_bwd_dqk_intra
# =============================================================================


def chunk_gla_bwd_dqk_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g_cumsum: torch.Tensor,
    dA: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Intra-chunk dq, dk from dA.

    Args:
        q:        [B, T, H, K]
        k:        [B, T, H, K]
        g_cumsum: [B, T, H, K]
        dA:       [B, T, H, C]
        chunk_size: block size

    Returns:
        dq, dk: [B, T, H, K]
    """
    B, T, H, K = q.shape
    C = chunk_size
    NT = T // C

    q_c = q.view(B, NT, C, H, K)
    k_c = k.view(B, NT, C, H, K)
    gc_c = g_cumsum.view(B, NT, C, H, K)
    dA_c = dA.view(B, NT, C, H, C)

    k_neg = k_c * (-gc_c).exp()
    dq = gc_c.exp() * torch.einsum("bnihj,bnjhk->bnihk", dA_c, k_neg)

    q_pos = q_c * gc_c.exp()
    dk = (-gc_c).exp() * torch.einsum("bnihj,bnihk->bnjhk", dA_c, q_pos)

    return dq.reshape(B, T, H, K), dk.reshape(B, T, H, K)


# =============================================================================
# Backward sub-function 5: chunk_gla_bwd_dqkg
# =============================================================================


def chunk_gla_bwd_dqkg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g_cumsum: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inter-chunk dq, dk + gate gradient dg.

    Args:
        q, k:     [B, T, H, K]
        v:        [B, T, H, V]
        h:        [B, NT, H, K, V]
        g_cumsum: [B, T, H, K]
        do:       [B, T, H, V]
        dh:       [B, NT, H, K, V]
        dq, dk:   [B, T, H, K] — intra-chunk gradients
        scale: scaling factor
        chunk_size: block size

    Returns:
        dq, dk, dg: [B, T, H, K]
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C

    q_c = q.view(B, NT, C, H, K)
    k_c = k.view(B, NT, C, H, K)
    v_c = v.view(B, NT, C, H, V)
    gc_c = g_cumsum.view(B, NT, C, H, K)
    do_c = do.view(B, NT, C, H, V)
    dq_c = dq.view(B, NT, C, H, K)
    dk_c = dk.view(B, NT, C, H, K)

    gn = gc_c[:, :, -1, :, :]  # [B, NT, H, K]

    # Inter-chunk dq
    dq_inter = scale * gc_c.exp() * torch.einsum("bnchv,bnhkv->bnchk", do_c, h)

    # Inter-chunk dk
    dk_inter = (gn[:, :, None, :, :] - gc_c).exp() * torch.einsum(
        "bnchv,bnhkv->bnchk", v_c, dh
    )

    dq_total = dq_c + dq_inter
    dk_total = dk_c + dk_inter

    # Gate gradient
    dgk_inter = (
        gn.exp() * torch.einsum("bnhkv,bnhkv->bnhk", h, dh)
        + (dk_inter * k_c).sum(dim=2)
    )

    dg_raw = q_c * dq_total - k_c * dk_total
    dg = dg_raw.flip(2).cumsum(2).flip(2) + dgk_inter[:, :, None, :, :]

    return (
        dq_total.reshape(B, T, H, K),
        dk_total.reshape(B, T, H, K),
        dg.reshape(B, T, H, K),
    )


# =============================================================================
# Backward orchestrator: chunk_gla_bwd
# =============================================================================


def chunk_gla_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Chunk GLA backward orchestrator.

    Returns:
        (dq, dk, dv, dg, dh0)
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
        do = F.pad(do, (0, 0, 0, 0, 0, pad))

    g_cumsum = chunk_local_cumsum(g, C)

    h, _ = chunk_fwd_h(k, v, g_cumsum, h0=initial_state, output_final_state=False, chunk_size=C)

    dh, dh0 = chunk_bwd_dh(
        q, k, v, g_cumsum, do, h0=initial_state, dht=dht, scale=scale, chunk_size=C,
    )

    dA = chunk_gla_bwd_dA(v, do, scale, chunk_size=C)

    A = chunk_gla_fwd_intra_gk(q, k, g_cumsum, scale, chunk_size=C)
    # A is [B, NT, C, H, C], need [B, T, H, C] for dv
    A_flat = A.reshape(B, T_padded, H, C)
    dv = chunk_gla_bwd_dv(k, g_cumsum, A_flat, do, dh, chunk_size=C)

    dq, dk = chunk_gla_bwd_dqk_intra(q, k, g_cumsum, dA, chunk_size=C)

    dq, dk, dg = chunk_gla_bwd_dqkg(
        q, k, v, h, g_cumsum, do, dh, dq, dk, scale, chunk_size=C,
    )

    dq = dq[:, :T]
    dk = dk[:, :T]
    dv = dv[:, :T]
    dg = dg[:, :T]

    return dq, dk, dv, dg, dh0
