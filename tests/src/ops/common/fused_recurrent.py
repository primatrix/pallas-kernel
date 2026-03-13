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
    # =========================================================================
    # Forward recurrence (per head, per sequence):
    #
    # Gate matrix applied to hidden state at each step:
    #   G_t[k,v] = exp(g_t) * exp(g_gamma) * exp(gk_t[k]) * exp(gv_t[v])
    # (each gate factor is optional; omitted ones are treated as 1)
    #
    # Recurrence:
    #   h_t[k,v] = G_t[k,v] * h_{t-1}[k,v] + k_t[k] * v_t[v]
    #
    # Output:
    #   o_t[v] = sum_k ( h_t[k,v] * q_t[k] * scale )
    #
    # h_0 = initial_state  (default zeros)
    # =========================================================================
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


def fused_recurrent_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    o: torch.Tensor | None = None,
    do: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    reverse: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor | None, torch.Tensor | None, torch.Tensor | None,
           torch.Tensor | None]:
    """CPU reference backward for fused_recurrent, equivalent to the Triton kernel.

    Returns:
        dq, dk, dv, dg, dgk, dgv, dh0
    """
    # =========================================================================
    # Backward derivation (per head, per sequence):
    #
    # Forward recap:
    #   h_gated_t = G_t ⊙ h_{t-1}          (element-wise gating)
    #   h_t       = h_gated_t + k_t ⊗ v_t  (outer product update)
    #   o_t[v]    = scale * sum_k h_t[k,v] * q_t[k]
    #
    # Define dh_t = dL/dh_t (gradient flowing back from future steps).
    # Initialize: dh_T = dht  (final state gradient, or zeros).
    #
    # At each step t (iterating backward through time):
    #
    #   (1) Output gradient contributes to dh_t:
    #       dh_t += scale * q_t[:, None] * do_t[None, :]    (outer product)
    #
    #   (2) Gradients of inputs through  h_t = h_gated_t + k_t ⊗ v_t:
    #       dk_t[k] = sum_v  dh_t[k,v] * v_t[v]
    #       dv_t[v] = sum_k  dh_t[k,v] * k_t[k]
    #
    #   (3) Gradients of the forward output o_t w.r.t. q:
    #       dq_t[k] = scale * sum_v  h_t[k,v] * do_t[v]
    #
    #   (4) Gate parameter gradients (using h_gated_t = G_t ⊙ h_{t-1}):
    #       Since  d(G_t ⊙ h_{t-1}) / d(log_gate_t) = h_gated_t,
    #       dg_t       = sum_{k,v} dh_t[k,v] * h_gated_t[k,v]    (scalar gate)
    #       dgk_t[k]   = sum_v     dh_t[k,v] * h_gated_t[k,v]    (key gate)
    #       dgv_t[v]   = sum_k     dh_t[k,v] * h_gated_t[k,v]    (value gate)
    #
    #   (5) Propagate dh through the gate to h_{t-1}:
    #       dh_{t-1} = G_t ⊙ dh_t
    #       (gates applied in reverse order: gv -> gk -> g_gamma -> g)
    #
    # Final: dh_0 is the gradient w.r.t. initial_state.
    # =========================================================================
    B, T, H, K = q.shape
    V = v.shape[-1]

    if scale is None:
        scale = K ** -0.5

    USE_G = g is not None
    USE_G_GAMMA = g_gamma is not None
    USE_GK = gk is not None
    USE_GV = gv is not None

    q_f = q.float().cpu()
    k_f = k.float().cpu()
    v_f = v.float().cpu()
    do_f = do.float().cpu()
    g_f = g.float().cpu() if USE_G else None
    g_gamma_f = g_gamma.float().cpu() if USE_G_GAMMA else None
    gk_f = gk.float().cpu() if USE_GK else None
    gv_f = gv.float().cpu() if USE_GV else None

    dq = torch.zeros(B, T, H, K, dtype=torch.float32)
    dk = torch.zeros(B, T, H, K, dtype=torch.float32)
    dv = torch.zeros(B, T, H, V, dtype=torch.float32)
    dg = torch.zeros_like(g_f) if USE_G else None
    dgk = torch.zeros(B, T, H, K, dtype=torch.float32) if USE_GK else None
    dgv = torch.zeros(B, T, H, V, dtype=torch.float32) if USE_GV else None

    dh0_list = []

    def _bwd_seq(seq_idx, bos, seq_len):
        """Backward for one sequence. Returns dh0 [H, K, V]."""
        b = 0 if cu_seqlens is not None else seq_idx

        if initial_state is not None:
            h_init = initial_state[seq_idx].clone().float().cpu()
        else:
            h_init = torch.zeros(H, K, V, dtype=torch.float32)

        time_range = list(range(seq_len - 1, -1, -1) if reverse else range(seq_len))

        # ---- Forward replay: store h_t and h_gated_t at each step ----
        # h_gated_t = G_t ⊙ h_{t-1}   (needed for gate gradients in step 4)
        # h_t = h_gated_t + k_t ⊗ v_t  (needed for dq in step 3)
        h_states = {}   # h_t after update at each step
        h_gateds = {}   # h_gated_t: fully-gated h_{t-1} (before outer-product update)

        h = h_init.clone()
        for i_t in time_range:
            t_idx = bos + i_t

            h_gated = h.clone()
            if USE_G:
                h_gated = h_gated * torch.exp(g_f[b, t_idx])[:, None, None]
            if USE_G_GAMMA:
                h_gated = h_gated * torch.exp(g_gamma_f)[:, None, None]
            if USE_GK:
                h_gated = h_gated * torch.exp(gk_f[b, t_idx])[:, :, None]
            if USE_GV:
                h_gated = h_gated * torch.exp(gv_f[b, t_idx])[:, None, :]

            h_gateds[i_t] = h_gated
            h = h_gated + k_f[b, t_idx].unsqueeze(-1) * v_f[b, t_idx].unsqueeze(-2)
            h_states[i_t] = h.clone()

        # ---- Step (3): dq_t[k] = scale * sum_v h_t[k,v] * do_t[v] ----
        for i_t in time_range:
            t_idx = bos + i_t
            h_t = h_states[i_t]
            do_t = do_f[b, t_idx]
            dq[b, t_idx] = scale * (h_t * do_t.unsqueeze(-2)).sum(-1)

        # ---- Backward pass (reverse of time_range) ----
        dh = torch.zeros(H, K, V, dtype=torch.float32)
        if dht is not None:
            dh = dht[seq_idx].clone().float().cpu()

        for i_t in reversed(time_range):
            t_idx = bos + i_t

            q_t = q_f[b, t_idx]       # [H, K]
            k_t = k_f[b, t_idx]       # [H, K]
            v_t = v_f[b, t_idx]       # [H, V]
            do_t = do_f[b, t_idx]     # [H, V]
            h_gated = h_gateds[i_t]   # [H, K, V]

            # Step (1): dh_t += scale * q_t[:, None] * do_t[None, :]
            dh = dh + scale * q_t.unsqueeze(-1) * do_t.unsqueeze(-2)

            # Step (2): dk_t[k] = sum_v dh_t[k,v] * v_t[v]
            #           dv_t[v] = sum_k dh_t[k,v] * k_t[k]
            dk[b, t_idx] = (dh * v_t.unsqueeze(-2)).sum(-1)    # [H, K]
            dv[b, t_idx] = (dh * k_t.unsqueeze(-1)).sum(-2)    # [H, V]

            # Step (4): gate gradients — d(G⊙h)/d(log_gate) = h_gated_t
            #   dg_t     = sum_{k,v} dh_t[k,v] * h_gated_t[k,v]
            #   dgk_t[k] = sum_v     dh_t[k,v] * h_gated_t[k,v]
            #   dgv_t[v] = sum_k     dh_t[k,v] * h_gated_t[k,v]
            if USE_G:
                dg[b, t_idx] = (dh * h_gated).sum((-1, -2))    # [H]
            if USE_GK:
                dgk[b, t_idx] = (dh * h_gated).sum(-1)         # [H, K]
            if USE_GV:
                dgv[b, t_idx] = (dh * h_gated).sum(-2)         # [H, V]

            # Step (5): dh_{t-1} = G_t ⊙ dh_t  (reverse gate order)
            if USE_GV:
                dh = dh * torch.exp(gv_f[b, t_idx])[:, None, :]
            if USE_GK:
                dh = dh * torch.exp(gk_f[b, t_idx])[:, :, None]
            if USE_G_GAMMA:
                dh = dh * torch.exp(g_gamma_f)[:, None, None]
            if USE_G:
                dh = dh * torch.exp(g_f[b, t_idx])[:, None, None]

        return dh  # dh0

    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
        for i_n in range(N):
            bos = cu_seqlens[i_n].item()
            eos = cu_seqlens[i_n + 1].item()
            dh0 = _bwd_seq(i_n, bos, eos - bos)
            dh0_list.append(dh0)
    else:
        for i_n in range(B):
            dh0 = _bwd_seq(i_n, 0, T)
            dh0_list.append(dh0)

    dh0_out = torch.stack(dh0_list, dim=0) if initial_state is not None else None
    return dq, dk, dv, dg, dgk, dgv, dh0_out
