"""chunk_gla_bwd_dA: FLA Triton GPU (gold) vs Torch CPU kernel tests.

Both the FLA Triton kernel and our CPU reference compute:

    dA[b,n,i,j] = scale * Σ_v do[b,n,i,h,v] * v[b,n,j,h,v]   (lower-tri masked)

The test builds random inputs and verifies both implementations produce
the same output.
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

# Disable TF32 for deterministic float32 results on Ampere+ GPUs.
os.environ["TRITON_F32_DEFAULT"] = "ieee"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import torch

from tests.src.ops.gla.chunk import chunk_gla_bwd_dA as cpu_chunk_gla_bwd_dA
from tests.utils import compare_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_CUDA = torch.cuda.is_available()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

triton_imports_available = False
try:
    from fla.ops.gla.chunk import chunk_gla_bwd_dA as triton_chunk_gla_bwd_dA

    triton_imports_available = True
except ImportError:
    pass

requires_triton = pytest.mark.skipif(
    not (HAS_CUDA and triton_imports_available),
    reason="Triton / CUDA not available",
)

# ============================================================================
# Unified test configs
#
# Each dict can contain:
#   B, T, H, K, V   — shape (required, T must be a multiple of chunk_size)
#   seed             — random seed (required)
#   chunk_size       — block size (default 64)
#   atol             — absolute tolerance (default 1e-2)
#   rtol             — relative tolerance (default 1e-2)
#   scale            — float or None (default None = K^{-0.5})
# ============================================================================

CASES = [
    # ── standard shapes ──
    dict(B=2, T=64, H=4, K=32, V=64, seed=42),
    dict(B=1, T=128, H=2, K=64, V=128, seed=7),
    # ── single head ──
    dict(B=2, T=64, H=1, K=32, V=64, seed=10),
    # ── K != V ──
    dict(B=2, T=64, H=4, K=16, V=128, seed=20),
    dict(B=2, T=64, H=4, K=128, V=16, seed=21),
    # ── T = 2 * chunk_size ──
    dict(B=2, T=128, H=4, K=16, V=32, seed=40),
    # ── large batch ──
    dict(B=8, T=64, H=4, K=32, V=64, seed=50),
    # ── many heads ──
    dict(B=1, T=64, H=16, K=32, V=64, seed=60),
    # ── small dims ──
    dict(B=2, T=64, H=2, K=8, V=16, seed=70),
    # ── various ──
    dict(B=4, T=64, H=8, K=32, V=64, seed=99),
    # ── custom scale ──
    dict(B=2, T=64, H=4, K=32, V=64, seed=200, scale=0.1),
    # ── long sequence ──
    dict(B=1, T=256, H=2, K=32, V=64, seed=300),
    # ── non-default chunk_size ──
    dict(B=2, T=64, H=4, K=32, V=64, seed=500, chunk_size=16),
    dict(B=1, T=128, H=2, K=32, V=64, seed=501, chunk_size=32),
]


def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    cs = c.get("chunk_size", 64)
    if cs != 64:
        parts.append(f"C{cs}")
    if c.get("scale") is not None:
        parts.append(f"scale={c['scale']}")
    return "-".join(parts)


# ============================================================================
# Helpers
# ============================================================================


def _to_device(*tensors):
    return tuple(t.contiguous().to(DEVICE) if t is not None else None for t in tensors)


# ============================================================================
# Main parametrized test — gold (Triton GPU) vs cpu (Torch)
# ============================================================================


@requires_triton
@pytest.mark.parametrize("cfg", CASES, ids=[_case_id(c) for c in CASES])
def test_gold_vs_cpu(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    atol = cfg.get("atol", 1e-5)
    rtol = cfg.get("rtol", 1e-5)
    scale = cfg.get("scale", K**-0.5)
    chunk_size = cfg.get("chunk_size", 64)

    assert T % chunk_size == 0, f"T={T} must be a multiple of chunk_size={chunk_size}"

    torch.manual_seed(cfg["seed"])
    v = torch.randn(B, T, H, V)
    do = torch.randn(B, T, H, V)

    # CPU reference
    dA_cpu = cpu_chunk_gla_bwd_dA(v, do, scale, chunk_size=chunk_size)

    # Triton GPU
    (v_g, do_g) = _to_device(v, do)
    dA_gold = triton_chunk_gla_bwd_dA(v_g, do_g, scale, chunk_size=chunk_size).cpu()

    assert compare_tensor("dA", dA_gold, dA_cpu, atol=atol, rtol=rtol)


# ============================================================================
# Structural test — varlen packed vs separate
# ============================================================================


def test_varlen_packed_vs_separate():
    """Backward dA: packed varlen (CPU cu_seqlens) == separate per-segment."""
    torch.manual_seed(700)
    H, K, V = 2, 32, 64
    C = 64
    s1_len, s2_len = 64, 128
    scale = K**-0.5

    v1 = torch.randn(1, s1_len, H, V)
    do1 = torch.randn(1, s1_len, H, V)
    v2 = torch.randn(1, s2_len, H, V)
    do2 = torch.randn(1, s2_len, H, V)

    dA1 = cpu_chunk_gla_bwd_dA(v1, do1, scale, chunk_size=C)
    dA2 = cpu_chunk_gla_bwd_dA(v2, do2, scale, chunk_size=C)

    v_cat = torch.cat([v1, v2], dim=1)
    do_cat = torch.cat([do1, do2], dim=1)
    cu = torch.tensor([0, s1_len, s1_len + s2_len], dtype=torch.long)

    dA_p = cpu_chunk_gla_bwd_dA(v_cat, do_cat, scale, cu_seqlens=cu, chunk_size=C)

    atol, rtol = 1e-5, 1e-5
    assert compare_tensor("seg1 dA", dA1, dA_p[:, :s1_len], atol=atol, rtol=rtol)
    assert compare_tensor("seg2 dA", dA2, dA_p[:, s1_len:], atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
