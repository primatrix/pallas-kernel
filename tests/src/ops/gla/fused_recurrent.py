from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch

from tests.src.ops.common.fused_recurrent import fused_recurrent

def fused_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fused recurrent GLA — Torch CPU reference with native cu_seqlens support.

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
        cu_seqlens: [N+1] cumulative sequence lengths (requires B=1)

    Returns:
        o: [B, T, H, V]
        ht: [N, H, K, V] if output_final_state else None
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return fused_recurrent(
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
