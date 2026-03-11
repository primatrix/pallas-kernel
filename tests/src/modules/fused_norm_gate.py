import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# FusedRMSNormGated (Replacement for fla.modules.FusedRMSNormGated, pure PyTorch)
# =============================================================================


class FusedRMSNormGated(nn.Module):
    """RMSNorm + SiLU gate fusion (Pure CPU version).

    output = RMSNorm(x) * silu(g)
    """

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H, V]  — kernel output (needs normalization)
        # g: [B, T, H, V]  — gate values (from g_proj projection + rearrange)
        # -> [B, T, H, V]  = RMSNorm(x) * SiLU(g)
        input_dtype = x.dtype
        x = x.float()
        g = g.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # [B, T, H, 1]
        x = x * rms  # [B, T, H, V]
        if self.weight is not None:
            x = x * self.weight.float()  # [B, T, H, V]  element-wise
        x = x * F.silu(g)  # [B, T, H, V]
        return x.to(input_dtype)
