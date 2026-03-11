import torch

# =============================================================================
# GLA core operation: naive recurrent (Replacement for Triton version chunk_gla / fused_recurrent_gla)
# Core GLA recurrent operation, pure PyTorch implementation
# =============================================================================


def naive_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Naive recurrent GLA — Pure PyTorch CPU implementation.

    Mathematically equivalent to chunk_gla / fused_recurrent_gla / fused_chunk_gla,
    but using naive step-by-step recurrence instead of Triton block or fused kernels.

    Core recurrence:
        h_t = h_{t-1} * exp(gk_t) + k_t^T v_t
        o_t = q_t * h_t  (then sum along K dimension)

    Args:
        q: [B, T, H, K] — Queries
        k: [B, T, H, K] — Keys
        v: [B, T, H, V] — Values
        gk: [B, T, H, K] — Gating (log-space, i.e., values after logsigmoid)
        scale: Scaling factor, default K^{-0.5}
        initial_state: [N, H, K, V] — Initial state
        output_final_state: Whether to output the final state
        cu_seqlens: [N+1] — Cumulative length of variable-length sequences (B must be 1 in this case)

    Returns:
        o: [B, T, H, V] — Output
        final_state: [N, H, K, V] or None
    """
    dtype = q.dtype
    # transpose: [B, T, H, K/V] → [B, H, T, K/V], float32 calculation
    q, k, v, gk = (x.transpose(1, 2).float() for x in (q, k, v, gk))
    B, H, T_total, K = q.shape  # q: [B, H, T, K]
    V = v.shape[-1]  # v: [B, H, T, V]

    if scale is None:
        scale = K**-0.5

    if cu_seqlens is not None:
        # Variable-length sequence mode: B=1, recurrent independently in segments by cu_seqlens
        assert B == 1, "cu_seqlens requires B=1"
        N = len(cu_seqlens) - 1
        o = torch.zeros_like(v)  # [1, H, T_total, V]
        final_states = [] if output_final_state else None

        for i in range(N):
            bos = cu_seqlens[i].item()
            eos = cu_seqlens[i + 1].item()
            seg_len = eos - bos

            # Extract data for this segment [1, H, seg_len, K/V]
            q_seg = q[:, :, bos:eos, :]  # [1, H, seg_len, K]
            k_seg = k[:, :, bos:eos, :]  # [1, H, seg_len, K]
            v_seg = v[:, :, bos:eos, :]  # [1, H, seg_len, V]
            gk_seg = gk[:, :, bos:eos, :]  # [1, H, seg_len, K]

            # Initial state
            h = q.new_zeros(1, H, K, V, dtype=torch.float32)  # [1, H, K, V]
            if initial_state is not None:
                h = h + initial_state[i : i + 1].float()  # [1, H, K, V]

            for t in range(seg_len):
                q_t = q_seg[:, :, t] * scale  # [1, H, K]
                k_t = k_seg[:, :, t]  # [1, H, K]
                v_t = v_seg[:, :, t]  # [1, H, V]
                gk_t = gk_seg[:, :, t].exp()  # [1, H, K]
                kv_t = k_t[..., None] * v_t[..., None, :]  # [1, H, K, V]
                h = h * gk_t[..., None] + kv_t  # [1, H, K, V]
                o[:, :, bos + t] = (q_t[..., None] * h).sum(-2)  # [1, H, V]

            if output_final_state:
                final_states.append(h.squeeze(0))  # [H, K, V]

        final_state = (
            torch.stack(final_states, dim=0) if output_final_state else None
        )  # [N, H, K, V]
        return o.transpose(1, 2).to(dtype), final_state  # o: [B, T, H, V]
    else:
        # Standard batch mode
        o = torch.zeros_like(v)  # [B, H, T, V]
        h = q.new_zeros(B, H, K, V, dtype=torch.float32)  # [B, H, K, V]
        if initial_state is not None:
            h = h + initial_state.float()  # [B, H, K, V]

        for t in range(T_total):
            q_t = q[:, :, t] * scale  # [B, H, K]
            k_t = k[:, :, t]  # [B, H, K]
            v_t = v[:, :, t]  # [B, H, V]
            gk_t = gk[:, :, t].exp()  # [B, H, K]
            kv_t = k_t[..., None] * v_t[..., None, :]  # [B, H, K, V]
            h = h * gk_t[..., None] + kv_t  # [B, H, K, V]
            o[:, :, t] = (q_t[..., None] * h).sum(-2)  # [B, H, V]

        final_state = h if output_final_state else None  # [B, H, K, V] or None
        return o.transpose(1, 2).to(dtype), final_state  # o: [B, T, H, V]
