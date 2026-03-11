import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# =============================================================================
# ShortConvolution (Replacement for fla.modules.ShortConvolution, pure PyTorch)
# Causal depthwise separable convolution based on nn.Conv1d
# =============================================================================


class ShortConvolution(nn.Conv1d):
    """Causal depthwise 1D convolution (Pure CPU version).

    - groups = hidden_size (depthwise separable)
    - causal padding = kernel_size - 1 (left padding)
    - optional SiLU activation
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = "silu",
        **kwargs,
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
        )
        self.hidden_size = hidden_size
        self.activation = activation

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation in ("silu", "swish"):
            return F.silu(x)
        return x

    def _causal_conv1d(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """Apply causal conv1d.
        x: [B, T, D] -> [B, T, D]  (depthwise causal convolution + optional SiLU)
        """
        W = self.kernel_size[0]

        if cu_seqlens is not None:
            # Variable length sequences: causal convolution independently in segments by cu_seqlens
            B, T_total, D = x.shape
            assert B == 1, "cu_seqlens requires B=1"
            N = len(cu_seqlens) - 1
            segments = []
            for i in range(N):
                bos = cu_seqlens[i].item()
                eos = cu_seqlens[i + 1].item()
                seg = x[:, bos:eos, :]  # [1, seg_len, D]
                seg = rearrange(seg, "b t d -> b d t")  # [1, D, seg_len]
                # Manual left padding
                seg_padded = F.pad(seg, (W - 1, 0))
                seg_out = F.conv1d(
                    seg_padded, self.weight, self.bias, groups=self.hidden_size
                )
                seg_out = rearrange(seg_out, "b d t -> b t d")
                segments.append(seg_out)
            y = torch.cat(segments, dim=1)
        else:
            # Standard case: convolution across the entire sequence
            x_t = rearrange(x, "b t d -> b d t")
            # nn.Conv1d with padding=kernel_size-1 will pad both sides
            # We need causal convolution, so we handle it manually
            x_padded = F.pad(x_t, (W - 1, 0))
            y = F.conv1d(x_padded, self.weight, self.bias, groups=self.hidden_size)
            y = rearrange(y, "b d t -> b t d")

        return self._apply_activation(y)

    def step(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Single-step decoding with cache update.

        x: [B, 1, D] or [1, N, D] with cu_seqlens
        cache: [N, D, W] where W is kernel_size
        """
        W = self.kernel_size[0]
        B = x.shape[0]
        D = self.hidden_size
        N = B if cu_seqlens is None else len(cu_seqlens) - 1

        if output_final_state and cache is None:
            cache = x.new_zeros(N, D, W)

        # Get current token
        x_step = (
            x.squeeze(0) if cu_seqlens is not None else x.squeeze(1)
        )  # [N, D] or [B, D]

        if cache is not None:
            # Roll cache and put new token at the last position
            cache = cache.roll(shifts=-1, dims=-1)
            cache[:, :, -1] = x_step
            # Dot product with convolution weights
            w = rearrange(self.weight, "d 1 w -> d w")
            y = (cache * w).sum(dim=-1)  # [N, D]
            if self.bias is not None:
                y = y + self.bias
        else:
            # Case without cache, calculate directly
            w = rearrange(self.weight, "d 1 w -> d w")
            # Use only the last weight
            y = x_step * w[:, -1]
            if self.bias is not None:
                y = y + self.bias

        y = self._apply_activation(y)
        y = y.view(x.shape)
        return y, cache

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # x: [B, T, D]  — projected q/k/v (D = key_dim or value_dim_per_group etc.)
        # cache: [N, D, W] or None (W = kernel_size)
        # cu_seqlens: [N+1] or None
        # -> y: [B, T, D], final_state: [N, D, W] or None
        B, T, D = x.shape
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        W = self.kernel_size[0]

        # Decoding mode: single token per sequence
        if B * T == N:
            return self.step(x, cache, output_final_state, cu_seqlens)

        # Prefill / training mode
        y = self._causal_conv1d(x, cu_seqlens)

        # Calculate final state (window of the last W tokens)
        final_state = None
        if output_final_state:
            if cu_seqlens is not None:
                final_states = []
                for i in range(N):
                    bos = cu_seqlens[i].item()
                    eos = cu_seqlens[i + 1].item()
                    seg = x[0, bos:eos, :]  # [seg_len, D]
                    seg_t = rearrange(seg, "t d -> d t")
                    # Left pad to ensure at least W tokens
                    if seg_t.shape[-1] < W:
                        seg_t = F.pad(seg_t, (W - seg_t.shape[-1], 0))
                    final_states.append(seg_t[:, -W:])  # [D, W]
                final_state = torch.stack(final_states, dim=0)  # [N, D, W]
            else:
                x_t = rearrange(x, "b t d -> b d t")
                if T < W:
                    x_t = F.pad(x_t, (W - T, 0))
                final_state = x_t[:, :, -W:]  # [B, D, W]

        # If cache is provided, write final_state to it
        if cache is not None and final_state is not None:
            cache.copy_(final_state)
            final_state = cache

        return y, final_state
