import torch


def fused_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, H, K = q.shape
    V = v.shape[-1]

    if scale is None:
        scale = K**-0.5

    USE_G = g is not None
    USE_G_GAMMA = g_gamma is not None
    USE_GK = gk is not None
    USE_GV = gv is not None

    # All accumulation in float32, matching the Triton kernel's tl.float32 accumulators
    q_f = q.float().cpu()
    k_f = k.float().cpu()
    v_f = v.float().cpu()
    g_f = g.float().cpu() if USE_G else None
    g_gamma_f = g_gamma.float().cpu() if USE_G_GAMMA else None
    gk_f = gk.float().cpu() if USE_GK else None
    gv_f = gv.float().cpu() if USE_GV else None

    o = torch.zeros(B, T, H, V, dtype=torch.float32)

    ht_list = []

    def _run_seq(batch_idx, bos, seq_len):
        """Run recurrence for one sequence, return final hidden state [H, K, V]."""
        if initial_state is not None:
            h = initial_state[batch_idx].clone().float().cpu()  # [H, K, V]
        else:
            h = torch.zeros(H, K, V, dtype=torch.float32)

        time_range = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

        for i_t in time_range:
            t_idx = bos + i_t

            # b_idx is always 0 in varlen mode (B=1), otherwise the batch index
            b = 0 if cu_seqlens is not None else batch_idx

            q_t = q_f[b, t_idx] * scale  # [H, K]
            k_t = k_f[b, t_idx]  # [H, K]
            v_t = v_f[b, t_idx]  # [H, V]

            # Apply log-gates to h: [H, K, V]
            if USE_G:
                # g[b, t_idx] -> [H]; broadcast to [H, 1, 1]
                h = h * torch.exp(g_f[b, t_idx])[:, None, None]
            if USE_G_GAMMA:
                # g_gamma -> [H]; broadcast to [H, 1, 1]
                h = h * torch.exp(g_gamma_f)[:, None, None]
            if USE_GK:
                # gk[b, t_idx] -> [H, K]; reshape to [H, K, 1]
                h = h * torch.exp(gk_f[b, t_idx])[:, :, None]
            if USE_GV:
                # gv[b, t_idx] -> [H, V]; reshape to [H, 1, V]
                h = h * torch.exp(gv_f[b, t_idx])[:, None, :]

            # h += k ⊗ v  (outer product per head: [H, K] x [H, V] -> [H, K, V])
            h = h + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)

            # o = sum_K(h * q[:, :, None])  -> [H, V]
            b_out = 0 if cu_seqlens is not None else batch_idx
            o[b_out, t_idx] = (h * q_t.unsqueeze(-1)).sum(1)

        return h

    if cu_seqlens is not None:
        # Varlen mode: B must be 1, sequences are packed into dim-1
        N = len(cu_seqlens) - 1
        for i_n in range(N):
            bos = cu_seqlens[i_n].item()
            eos = cu_seqlens[i_n + 1].item()
            h_final = _run_seq(i_n, bos, eos - bos)
            if output_final_state:
                ht_list.append(h_final)
    else:
        for i_n in range(B):
            h_final = _run_seq(i_n, 0, T)
            if output_final_state:
                ht_list.append(h_final)

    ht = torch.stack(ht_list, dim=0) if output_final_state else None
    return o, ht
