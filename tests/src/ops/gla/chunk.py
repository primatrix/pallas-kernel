import torch
import torch.nn.functional as F

def cdiv(x: torch.Tensor, y: int):
    return (x + y - 1) // y

def align_up(x: torch.Tensor, align: int):
    return cdiv(x, align) * align

def pad_to_multiple(x: torch.Tensor, multiple: int | list, axis: int | list, val):
    if isinstance(multiple, int):
        multiple = [multiple]
    if isinstance(axis, int):
        axis = [axis]

    assert len(multiple) == len(axis), (
        f"Length of multiple {len(multiple)} must match length of axis {len(axis)}"
    )

    shape = list(x.shape)
    pad_width = [(0, 0)] * len(shape)
    for idx in range(0, len(axis)):
        ax = axis[idx]
        mu = multiple[idx]
        length = shape[ax]
        remainder = length % mu
        if remainder == 0:
            continue
        pad_len = mu - remainder
        pad_width[ax] = (0, pad_len)
    return F.pad(x, pad_width, value=val)


def pad_varlen_seqs(
    tensors: list[torch.Tensor],
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> tuple[list[torch.Tensor], torch.Tensor, list[int] | None, list[int] | None]:
    """Pad each variable-length segment along dim=1 to a multiple of chunk_size.

    Args:
        tensors: list of [1, T_total, ...] tensors sharing the same sequence layout
        cu_seqlens: [N+1] cumulative sequence lengths
        chunk_size: block size

    Returns:
        (padded_tensors, new_cu_seqlens, orig_seqlens, padded_seqlens)
        If no padding is needed, returns (tensors, cu_seqlens, None, None).
    """
    N = len(cu_seqlens) - 1
    orig_seqlens = torch.diff(cu_seqlens).tolist()
    padded_seqlens = [((L + chunk_size - 1) // chunk_size) * chunk_size for L in orig_seqlens]

    if orig_seqlens == padded_seqlens:
        return tensors, cu_seqlens, None, None

    padded = [[] for _ in tensors]
    for i in range(N):
        bos = cu_seqlens[i].item()
        L = orig_seqlens[i]
        pad = padded_seqlens[i] - L
        for j, t in enumerate(tensors):
            seg = t[:, bos:bos + L]
            padded[j].append(
                F.pad(seg, (0, 0, 0, 0, 0, pad)) if pad > 0 else seg
            )

    padded_tensors = [torch.cat(p, dim=1) for p in padded]
    offsets = [0]
    for pl in padded_seqlens:
        offsets.append(offsets[-1] + pl)
    new_cu_seqlens = torch.tensor(offsets, dtype=torch.long)

    return padded_tensors, new_cu_seqlens, orig_seqlens, padded_seqlens


def unpad_varlen_seqs(
    tensor: torch.Tensor,
    orig_seqlens: list[int],
    padded_seqlens: list[int],
) -> torch.Tensor:
    """Remove per-segment padding from a variable-length tensor along dim=1.

    Args:
        tensor: [1, T_padded_total, ...] padded tensor
        orig_seqlens: original segment lengths
        padded_seqlens: padded segment lengths

    Returns:
        [1, T_total, ...] tensor with padding removed
    """
    parts = []
    offset = 0
    for L, PL in zip(orig_seqlens, padded_seqlens):
        parts.append(tensor[:, offset:offset + L])
        offset += PL
    return torch.cat(parts, dim=1)

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
        cu_seqlens: [N+1] cumulative sequence lengths for variable-length mode.
            When provided, B must be 1 and each segment length must be a
            multiple of chunk_size.

    Returns:
        g_cumsum: [B, T, H, K] — chunk-local cumsum
    """
    B, T, H, K = g.shape
    C = chunk_size

    if cu_seqlens is not None:
        assert B == 1
        N = len(cu_seqlens) - 1
        g_cumsum = torch.zeros_like(g)
        for i in range(N):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            g_cumsum[:, bos:eos] = chunk_local_cumsum(g[:, bos:eos], chunk_size)
        return g_cumsum

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
        h0: [B, H, K, V] or [N, H, K, V] (varlen) — initial hidden state (optional)
        output_final_state: whether to return final state
        cu_seqlens: [N+1] cumulative sequence lengths for variable-length mode.
            When provided, B must be 1.  h0 shape is [N, H, K, V].
            Returns h [1, NT_total, H, K, V] and ht [N, H, K, V].
        chunk_size: block size

    Returns:
        h:  [B, NT, H, K, V] — hidden state at the start of each chunk
        ht: [B, H, K, V] or [N, H, K, V] (varlen) or None — final hidden state
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    C = chunk_size

    if cu_seqlens is not None:
        assert B == 1
        N = len(cu_seqlens) - 1
        h_list, ht_list = [], []
        for i in range(N):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            h0_i = h0[i:i+1] if h0 is not None else None
            h_seg, ht_seg = chunk_fwd_h(
                k[:, bos:eos], v[:, bos:eos], gk[:, bos:eos],
                h0=h0_i, output_final_state=output_final_state, chunk_size=chunk_size)
            h_list.append(h_seg)
            if ht_seg is not None:
                ht_list.append(ht_seg.squeeze(0))
        h_all = torch.cat(h_list, dim=1)
        ht = torch.stack(ht_list) if ht_list else None
        return h_all, ht

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
        h0:  [B, H, K, V] or [N, H, K, V] (varlen) — initial hidden state (optional)
        dht: [B, H, K, V] or [N, H, K, V] (varlen) — terminal state gradient (optional)
        scale: scaling factor
        cu_seqlens: [N+1] cumulative sequence lengths for variable-length mode.
            When provided, B must be 1.  h0/dht shape is [N, H, K, V].
            Returns dh [1, NT_total, H, K, V] and dh0 [N, H, K, V].
        chunk_size: block size

    Returns:
        dh:  [B, NT, H, K, V]
        dh0: [B, H, K, V] or [N, H, K, V] (varlen) or None
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    if cu_seqlens is not None:
        assert B == 1
        N = len(cu_seqlens) - 1
        dh_list, dh0_list = [], []
        for i in range(N):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            h0_i = h0[i:i+1] if h0 is not None else None
            dht_i = dht[i:i+1] if dht is not None else None
            dh_seg, dh0_seg = chunk_bwd_dh(
                q[:, bos:eos], k[:, bos:eos], v[:, bos:eos],
                gk[:, bos:eos], do[:, bos:eos],
                h0=h0_i, dht=dht_i, scale=scale, chunk_size=chunk_size)
            dh_list.append(dh_seg)
            if dh0_seg is not None:
                dh0_list.append(dh0_seg.squeeze(0))
        dh = torch.cat(dh_list, dim=1)
        dh0 = torch.stack(dh0_list) if dh0_list else None
        return dh, dh0

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
        cu_seqlens: [N+1] cumulative sequence lengths for variable-length mode.
            When provided, B must be 1.
        chunk_size: block size

    Returns:
        A: [B, T, H, BT] — intra-chunk attention matrix (float32)
    """
    B, T, H, K = q.shape
    BT = chunk_size

    if cu_seqlens is not None:
        assert B == 1
        N = len(cu_seqlens) - 1
        A_list = []
        for i in range(N):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            A_seg = chunk_gla_fwd_intra_gk(
                q[:, bos:eos], k[:, bos:eos], g[:, bos:eos],
                scale, chunk_size=chunk_size)
            A_list.append(A_seg)
        return torch.cat(A_list, dim=1)

    NT = T // BT

    q_c = q.view(B, NT, BT, H, K)
    k_c = k.view(B, NT, BT, H, K)
    g_c = g.view(B, NT, BT, H, K)

    # 数值稳定: 引入参考点 g_n (每个 chunk 第一行)
    g_n = g_c[:, :, 0:1, :, :]  # [B, NT, 1, H, K]
    q_gated = q_c * (g_c - g_n).exp()  # [B, NT, BT, H, K]
    k_gated = k_c * (g_n - g_c).exp()  # [B, NT, BT, H, K]

    # A[b,n,i,h,j] = scale * Σ_k q_gated[b,n,i,h,k] * k_gated[b,n,j,h,k]
    A = torch.einsum("bnihk,bnjhk->bnihj", q_gated, k_gated) * scale

    # Causal mask: 上三角清零
    causal_mask = torch.tril(torch.ones(BT, BT, dtype=torch.bool, device=q.device))
    A = A.masked_fill(~causal_mask[None, None, :, None, :], 0.0)

    return A.reshape(B, T, H, BT)


# =============================================================================
# Sub-function 4: chunk_gla_fwd_o_gk
# =============================================================================


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
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
        A: [B, T, H, BT] — intra-chunk attention matrix
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
    g = g.reshape(-1, C, H, K)
    h = h.reshape(-1, H, K, V)
    A = A.reshape(-1, C, H, C)

    qg = q * g.exp()

    # Inter-chunk: o_inter = scale * (q_gated @ h)
    o_inter = scale * torch.einsum(
        "nchk,nhkv->nchv", qg, h
    )  # [C, K] @ [K, V] -> [C, V]

    # [C, C] @ [C, V] -> [C, V]
    # Intra-chunk: o_intra = A @ v, contract over j (key position within chunk)
    o_intra = torch.einsum("nihj,njhv->nihv", A, v)

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
        cu_seqlens: [N+1] cumulative sequence lengths for variable-length mode
        chunk_size: block size

    Returns:
        (g_cumsum, A, h, ht, o)  — o has the original (unpadded) T length
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    # --- padding ---
    orig_seqlens = None
    padded_seqlens = None

    if cu_seqlens is not None:
        assert B == 1
        [q, k, v, g], cu_seqlens, orig_seqlens, padded_seqlens = pad_varlen_seqs(
            [q, k, v, g], cu_seqlens, C
        )
    else:
        T_padded = ((T + C - 1) // C) * C
        if T_padded > T:
            pad = T_padded - T
            pads = (0, 0, 0, 0, 0, pad)
            q = F.pad(q, pads)
            k = F.pad(k, pads)
            v = F.pad(v, pads)
            g = F.pad(g, pads)

    T_padded = q.shape[1]
    NT = T_padded // C

    # --- forward ---

    if g_cumsum is None:
        g_cumsum = chunk_local_cumsum(g, C, cu_seqlens)
        assert g_cumsum.shape == g.shape
        assert g_cumsum.dtype == g.dtype

    h, ht = chunk_fwd_h(
        k,
        v,
        gk=g_cumsum,
        h0=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=C,
    )
    assert (B, NT, H, K, V) == h.shape
    assert (N, H, K, V) == ht.shape if ht is not None else True
    assert h.dtype == k.dtype
    assert ht.dtype == k.dtype if ht is not None else True

    A = chunk_gla_fwd_intra_gk(q, k, g=g_cumsum, scale=scale, cu_seqlens=cu_seqlens, chunk_size=C)
    assert (B, T_padded, H, C) == A.shape
    assert A.dtype == torch.float32, "Attention matrix must be float32 for numerical stability"

    o = chunk_gla_fwd_o_gk(q, v, g=g_cumsum, A=A, h=h, scale=scale, cu_seqlens=cu_seqlens, chunk_size=C)
    assert (B, T_padded, H, V) == o.shape
    assert o.dtype == v.dtype

    # --- unpadding ---
    if orig_seqlens is not None:
        g_cumsum = unpad_varlen_seqs(g_cumsum, orig_seqlens, padded_seqlens)
        o = unpad_varlen_seqs(o, orig_seqlens, padded_seqlens)
        A = unpad_varlen_seqs(A, orig_seqlens, padded_seqlens)
    else:
        o = o[:, :T]
        g_cumsum = g_cumsum[:, :T]
        A = A[:, :T]

    return g_cumsum, A, h, ht, o


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
        cu_seqlens: [N+1] cumulative sequence lengths for variable-length mode.
            When provided, B must be 1.
        chunk_size: block size

    Returns:
        dA: [B, T, H, C] — lower-triangular masked
    """
    B, T, H, V = v.shape
    C = chunk_size

    if cu_seqlens is not None:
        assert B == 1
        N = len(cu_seqlens) - 1
        dA = torch.zeros(1, T, H, C, dtype=v.dtype, device=v.device)
        for i in range(N):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            dA[:, bos:eos] = chunk_gla_bwd_dA(v[:, bos:eos], do[:, bos:eos], scale, chunk_size=chunk_size)
        return dA

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
        cu_seqlens: [N+1] cumulative sequence lengths for variable-length mode.
            When provided, B must be 1.  dh is [1, NT_total, H, K, V] with
            chunks from all segments concatenated along dim-1.
        chunk_size: block size

    Returns:
        dv: [B, T, H, V]
    """
    B, T, H, K = k.shape
    V = do.shape[-1]
    C = chunk_size

    if cu_seqlens is not None:
        assert B == 1
        N = len(cu_seqlens) - 1
        dv = torch.zeros(1, T, H, V, dtype=do.dtype, device=do.device)
        chunk_offset = 0
        for i in range(N):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            NT_seg = (eos - bos) // C
            dv[:, bos:eos] = chunk_gla_bwd_dv(
                k[:, bos:eos], g_cumsum[:, bos:eos], A[:, bos:eos],
                do[:, bos:eos], dh[:, chunk_offset:chunk_offset+NT_seg],
                chunk_size=chunk_size)
            chunk_offset += NT_seg
        return dv

    NT = T // C

    k_c = k.view(B, NT, C, H, K)
    gc_c = g_cumsum.view(B, NT, C, H, K)
    do_c = do.view(B, NT, C, H, V)
    A_c = A.view(B, NT, C, H, C)

    # Intra: dv[j] = sum_{i>=j} A[i,j] * do[i]
    # A is already lower-triangular masked by chunk_gla_fwd_intra_gk
    dv_intra = torch.einsum("bnihj,bnihv->bnjhv", A_c, do_c)

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
        cu_seqlens: [N+1] cumulative sequence lengths for variable-length mode.
            When provided, B must be 1.
        chunk_size: block size

    Returns:
        dq, dk: [B, T, H, K]
    """
    B, T, H, K = q.shape
    C = chunk_size

    if cu_seqlens is not None:
        assert B == 1
        N = len(cu_seqlens) - 1
        dq_out = torch.zeros_like(q)
        dk_out = torch.zeros_like(k)
        for i in range(N):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            dq_seg, dk_seg = chunk_gla_bwd_dqk_intra(
                q[:, bos:eos], k[:, bos:eos], g_cumsum[:, bos:eos],
                dA[:, bos:eos], chunk_size=chunk_size)
            dq_out[:, bos:eos] = dq_seg
            dk_out[:, bos:eos] = dk_seg
        return dq_out, dk_out

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
        cu_seqlens: [N+1] cumulative sequence lengths for variable-length mode.
            When provided, B must be 1.  h and dh are [1, NT_total, H, K, V]
            with chunks from all segments concatenated along dim-1.
        chunk_size: block size

    Returns:
        dq, dk, dg: [B, T, H, K]
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    if cu_seqlens is not None:
        assert B == 1
        N = len(cu_seqlens) - 1
        dq_out, dk_out, dg_out = (torch.zeros_like(q) for _ in range(3))
        chunk_offset = 0
        for i in range(N):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            NT_seg = (eos - bos) // C
            dq_seg, dk_seg, dg_seg = chunk_gla_bwd_dqkg(
                q[:, bos:eos], k[:, bos:eos], v[:, bos:eos],
                h[:, chunk_offset:chunk_offset+NT_seg],
                g_cumsum[:, bos:eos], do[:, bos:eos],
                dh[:, chunk_offset:chunk_offset+NT_seg],
                dq[:, bos:eos], dk[:, bos:eos],
                scale, chunk_size=chunk_size)
            dq_out[:, bos:eos] = dq_seg
            dk_out[:, bos:eos] = dk_seg
            dg_out[:, bos:eos] = dg_seg
            chunk_offset += NT_seg
        return dq_out, dk_out, dg_out

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
    g_cumsum: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor | None,
    h: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Chunk GLA backward orchestrator.

    Args:
        q, k:     [B, T, H, K]
        v:        [B, T, H, V]
        g:        [B, T, H, K] — raw log-space gates
        scale: scaling factor
        initial_state: [B, H, K, V] or [N, H, K, V] (varlen) — initial hidden state
        do:       [B, T, H, V] — output gradient
        dht:      [B, H, K, V] or [N, H, K, V] (varlen) — terminal state gradient
        cu_seqlens: [N+1] cumulative sequence lengths for variable-length mode.
            When provided, B must be 1.
        chunk_size: block size

    Returns:
        (dq, dk, dv, dg, dh0)
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    # --- padding ---
    orig_seqlens = None
    padded_seqlens = None

    if cu_seqlens is not None:
        assert B == 1
        [q, k, v, g, do], cu_seqlens, orig_seqlens, padded_seqlens = pad_varlen_seqs(
            [q, k, v, g, do], cu_seqlens, C
        )
    else:
        T_padded = ((T + C - 1) // C) * C
        if T_padded > T:
            pad = T_padded - T
            pads = (0, 0, 0, 0, 0, pad)
            q = F.pad(q, pads)
            k = F.pad(k, pads)
            v = F.pad(v, pads)
            g = F.pad(g, pads)
            do = F.pad(do, pads)

    T_padded = q.shape[1]

    # --- recompute forward intermediates ---
    g_cumsum = chunk_local_cumsum(g, C, cu_seqlens)

    h, _ = chunk_fwd_h(k, v, g_cumsum, h0=initial_state, output_final_state=False,
                        cu_seqlens=cu_seqlens, chunk_size=C)

    # --- backward ---
    dh, dh0 = chunk_bwd_dh(
        q, k, v, g_cumsum, do, h0=initial_state, dht=dht, scale=scale,
        cu_seqlens=cu_seqlens, chunk_size=C,
    )

    dA = chunk_gla_bwd_dA(v, do, scale, cu_seqlens=cu_seqlens, chunk_size=C)

    A = chunk_gla_fwd_intra_gk(q, k, g_cumsum, scale, cu_seqlens=cu_seqlens, chunk_size=C)
    A_flat = A.reshape(B, T_padded, H, C)
    dv = chunk_gla_bwd_dv(k, g_cumsum, A_flat, do, dh, cu_seqlens=cu_seqlens, chunk_size=C)

    dq, dk = chunk_gla_bwd_dqk_intra(q, k, g_cumsum, dA, cu_seqlens=cu_seqlens, chunk_size=C)

    dq, dk, dg = chunk_gla_bwd_dqkg(
        q, k, v, h, g_cumsum, do, dh, dq, dk, scale,
        cu_seqlens=cu_seqlens, chunk_size=C,
    )

    # --- unpadding ---
    if orig_seqlens is not None:
        dq = unpad_varlen_seqs(dq, orig_seqlens, padded_seqlens)
        dk = unpad_varlen_seqs(dk, orig_seqlens, padded_seqlens)
        dv = unpad_varlen_seqs(dv, orig_seqlens, padded_seqlens)
        dg = unpad_varlen_seqs(dg, orig_seqlens, padded_seqlens)
    else:
        dq = dq[:, :T]
        dk = dk[:, :T]
        dv = dv[:, :T]
        dg = dg[:, :T]

    return dq, dk, dv, dg, dh0
