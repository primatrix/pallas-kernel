"""Module-level tests: RMSNorm, FusedRMSNormGated, ShortConvolution.

Triton (FLA CUDA) vs Torch CPU, plus standalone correctness checks.
"""

from __future__ import annotations

import unittest

import pytest
import torch
import torch.nn.functional as F

from tests.src.modules.layernorm import RMSNorm as CpuRMSNorm
from tests.src.modules.fused_norm_gate import FusedRMSNormGated as CpuFNG
from tests.src.modules.convolution import ShortConvolution as CpuShortConv
from tests.utils import compare_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_CUDA = torch.cuda.is_available()

triton_imports_available = False
try:
    from fla.modules import RMSNorm as TritonRMSNorm
    from fla.modules import FusedRMSNormGated as TritonFNG
    from fla.modules import ShortConvolution as TritonShortConv

    triton_imports_available = True
except ImportError:
    pass

requires_triton = pytest.mark.skipif(
    not (HAS_CUDA and triton_imports_available),
    reason="Triton / CUDA not available",
)


def to_cpu(sd: dict) -> dict:
    return {k: v.cpu() for k, v in sd.items()}


# ============================================================================
# TestRMSNorm
# ============================================================================


class TestRMSNorm(unittest.TestCase):
    @requires_triton
    def test_triton_vs_cpu(self):
        """Triton RMSNorm (CUDA) vs Torch CPU."""
        torch.manual_seed(0)
        dim = 64
        triton_norm = TritonRMSNorm(dim, elementwise_affine=True, eps=1e-5).to(DEVICE)
        cpu_norm = CpuRMSNorm(dim, elementwise_affine=True, eps=1e-5)
        cpu_norm.load_state_dict(to_cpu(triton_norm.state_dict()))
        triton_norm.eval()
        cpu_norm.eval()

        x = torch.randn(2, 10, dim)
        with torch.no_grad():
            y_triton = triton_norm(x.to(DEVICE)).cpu()
            y_cpu = cpu_norm(x)
        assert compare_tensor("RMSNorm", y_triton, y_cpu, atol=1e-5, rtol=1e-5)

    def test_standalone_correctness(self):
        """Manual RMS computation vs module output."""
        torch.manual_seed(0)
        dim = 64
        norm = CpuRMSNorm(dim, elementwise_affine=True, eps=1e-5)
        x = torch.randn(2, 10, dim)
        y = norm(x)
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + 1e-5)
        y_exp = (x_f * rms * norm.weight.float()).to(x.dtype)
        assert compare_tensor("RMSNorm vs manual", y, y_exp, atol=1e-6, rtol=1e-6)

    def test_no_affine(self):
        """No affine parameters: shape is preserved."""
        dim = 64
        norm = CpuRMSNorm(dim, elementwise_affine=False)
        x = torch.randn(2, 10, dim)
        y = norm(x)
        assert y.shape == x.shape


# ============================================================================
# TestFusedRMSNormGated
# ============================================================================


class TestFusedRMSNormGated(unittest.TestCase):
    @requires_triton
    def test_triton_vs_cpu(self):
        """Triton FusedRMSNormGated (CUDA) vs Torch CPU."""
        torch.manual_seed(0)
        dim = 64
        triton_fn = TritonFNG(dim, elementwise_affine=True, eps=1e-5).to(DEVICE)
        cpu_fn = CpuFNG(dim, elementwise_affine=True, eps=1e-5)
        cpu_fn.load_state_dict(to_cpu(triton_fn.state_dict()))
        triton_fn.eval()
        cpu_fn.eval()

        x = torch.randn(2, 10, dim)
        g = torch.randn(2, 10, dim)
        with torch.no_grad():
            y_triton = triton_fn(x.to(DEVICE), g.to(DEVICE)).cpu()
            y_cpu = cpu_fn(x, g)
        assert compare_tensor(
            "FusedRMSNormGated", y_triton, y_cpu, atol=1e-5, rtol=1e-5
        )

    def test_standalone_correctness(self):
        """Manual RMSNorm + SiLU vs module output."""
        torch.manual_seed(0)
        dim = 64
        fnorm = CpuFNG(dim, elementwise_affine=True, eps=1e-5)
        x = torch.randn(2, 10, dim)
        g = torch.randn(2, 10, dim)
        y = fnorm(x, g)
        x_f, g_f = x.float(), g.float()
        rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + 1e-5)
        y_exp = (x_f * rms * fnorm.weight.float() * F.silu(g_f)).to(x.dtype)
        assert compare_tensor("FNG vs manual", y, y_exp, atol=1e-6, rtol=1e-6)


# ============================================================================
# TestShortConvolution
# ============================================================================


class TestShortConvolution(unittest.TestCase):
    @requires_triton
    def test_triton_vs_cpu(self):
        """Triton ShortConvolution (CUDA) vs Torch CPU."""
        triton_conv = TritonShortConv(
            hidden_size=32, kernel_size=4, bias=True, activation="silu"
        ).to(DEVICE)
        cpu_conv = CpuShortConv(
            hidden_size=32, kernel_size=4, bias=True, activation="silu"
        )
        cpu_conv.load_state_dict(to_cpu(triton_conv.state_dict()))
        triton_conv.eval()
        cpu_conv.eval()

        x = torch.randn(2, 16, 32)
        with torch.no_grad():
            y_triton, _ = triton_conv(x.to(DEVICE))
            y_cpu, _ = cpu_conv(x)
        assert compare_tensor(
            "ShortConv output", y_triton.cpu(), y_cpu, atol=1e-5, rtol=1e-5
        )

    @requires_triton
    def test_triton_vs_cpu_cache(self):
        """Triton ShortConvolution cache output vs Torch CPU."""
        triton_conv = TritonShortConv(
            hidden_size=16, kernel_size=4, bias=True, activation="silu"
        ).to(DEVICE)
        cpu_conv = CpuShortConv(
            hidden_size=16, kernel_size=4, bias=True, activation="silu"
        )
        cpu_conv.load_state_dict(to_cpu(triton_conv.state_dict()))
        triton_conv.eval()
        cpu_conv.eval()

        x = torch.randn(1, 8, 16)
        with torch.no_grad():
            y_triton, cache_triton = triton_conv(x.to(DEVICE), output_final_state=True)
            y_cpu, cache_cpu = cpu_conv(x, output_final_state=True)
        assert compare_tensor("output", y_triton.cpu(), y_cpu, atol=1e-5, rtol=1e-5)
        assert compare_tensor(
            "cache", cache_triton.cpu(), cache_cpu, atol=1e-5, rtol=1e-5
        )

    def test_causality(self):
        """Causal property: future changes don't affect past outputs."""
        conv = CpuShortConv(hidden_size=8, kernel_size=3, bias=True, activation="silu")
        x1 = torch.randn(1, 16, 8)
        x2 = x1.clone()
        x2[:, 10:, :] = torch.randn(1, 6, 8)
        y1, _ = conv(x1)
        y2, _ = conv(x2)
        assert torch.allclose(y1[:, :10], y2[:, :10], atol=1e-6), (
            "past outputs affected by future"
        )
        assert not torch.allclose(y1[:, 10:], y2[:, 10:], atol=1e-6), (
            "future should differ"
        )

    def test_cu_seqlens(self):
        """cu_seqlens packed == separate sequences."""
        conv = CpuShortConv(hidden_size=8, kernel_size=3, bias=False, activation="silu")
        x_s1 = torch.randn(1, 6, 8)
        x_s2 = torch.randn(1, 10, 8)
        y_s1, _ = conv(x_s1)
        y_s2, _ = conv(x_s2)
        x_packed = torch.cat([x_s1, x_s2], dim=1)
        cu = torch.tensor([0, 6, 16], dtype=torch.long)
        y_packed, _ = conv(x_packed, cu_seqlens=cu)
        assert compare_tensor("seg1", y_s1, y_packed[:, :6])
        assert compare_tensor("seg2", y_s2, y_packed[:, 6:])

    def test_step_decode(self):
        """Single-step decode with cache."""
        conv = CpuShortConv(hidden_size=16, kernel_size=4, bias=True, activation="silu")
        x_pre = torch.randn(1, 8, 16)
        y_pre, cache = conv(x_pre, output_final_state=True)
        assert cache is not None and cache.shape == (1, 16, 4)
        x_dec = torch.randn(1, 1, 16)
        y_dec, cache_dec = conv.step(x_dec, cache, output_final_state=True)
        assert y_dec.shape == (1, 1, 16)
        assert cache_dec.shape == (1, 16, 4)
