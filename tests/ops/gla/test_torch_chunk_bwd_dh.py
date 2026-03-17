"""chunk_bwd_dh: FLA Triton GPU (gold) vs Torch CPU kernel tests.

Both the FLA Triton kernel and our CPU reference compute the backward pass
for hidden-state gradients dh and initial-state gradient dh0.

The test computes g_cumsum on CPU as shared truth, then passes it to both
implementations.

Triton call signature:
    chunk_bwd_dh(q, k, v, do, h0, dht, scale, gk=g_cumsum, ...)
CPU call signature:
    chunk_bwd_dh(q, k, v, gk, do, h0=h0, dht=dht, scale=scale, ...)
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
import torch.nn.functional as F

from tests.src.ops.gla.chunk import (
    chunk_bwd_dh as cpu_chunk_bwd_dh,
    chunk_local_cumsum as cpu_chunk_local_cumsum,
)
from tests.utils import compare_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_CUDA = torch.cuda.is_available()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

triton_imports_available = False
try:
    from fla.ops.common.chunk_h import chunk_bwd_dh as triton_chunk_bwd_dh

    triton_imports_available = True
except ImportError:
    pass

requires_triton = pytest.mark.skipif(
    not (HAS_CUDA and triton_imports_available),
    reason="Triton / CUDA not available",
)

# ============================================================================
# Unified test configs
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
    # ── with h0 ──
    dict(B=2, T=64, H=4, K=32, V=64, seed=13, h0=True),
    # ── with dht ──
    dict(B=2, T=64, H=4, K=32, V=64, seed=14, dht=True),
    # ── with h0 + dht ──
    dict(B=2, T=64, H=4, K=32, V=64, seed=15, h0=True, dht=True),
    # ── long + h0 + dht ──
    dict(B=1, T=256, H=2, K=32, V=64, seed=303, h0=True, dht=True),
    # ── non-default chunk_size ──
    dict(B=2, T=64, H=4, K=32, V=64, seed=500, chunk_size=16),
    dict(B=1, T=128, H=2, K=32, V=64, seed=501, chunk_size=32, h0=True, dht=True),
]


def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    cs = c.get("chunk_size", 64)
    if cs != 64:
        parts.append(f"C{cs}")
    if c.get("h0"):
        parts.append("h0")
    if c.get("dht"):
        parts.append("dht")
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
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    g = F.logsigmoid(torch.randn(B, T, H, K))
    do = torch.randn(B, T, H, V)
    h0 = torch.randn(B, H, K, V) if cfg.get("h0") else None
    dht = torch.randn(B, H, K, V) if cfg.get("dht") else None

    # Compute g_cumsum on CPU as shared truth
    g_cumsum = cpu_chunk_local_cumsum(g, chunk_size)

    # CPU reference
    dh_cpu, dh0_cpu = cpu_chunk_bwd_dh(
        q, k, v, g_cumsum, do,
        h0=h0, dht=dht, scale=scale, chunk_size=chunk_size,
    )

    # Triton GPU
    (q_g, k_g, v_g, do_g, h0_g, dht_g, gc_g) = _to_device(
        q, k, v, do, h0, dht, g_cumsum,
    )
    dh_gold, dh0_gold = triton_chunk_bwd_dh(
        q_g, k_g, v_g, do_g, h0_g, dht_g, scale,
        gk=gc_g, states_in_fp32=True, chunk_size=chunk_size,
    )
    dh_gold = dh_gold.cpu()
    dh0_gold = dh0_gold.cpu() if dh0_gold is not None else None

    assert compare_tensor("dh", dh_gold, dh_cpu, atol=atol, rtol=rtol)
    if cfg.get("h0"):
        assert compare_tensor("dh0", dh0_gold, dh0_cpu, atol=atol, rtol=rtol)


# ============================================================================
# Structural test — varlen packed vs separate
# ============================================================================


def test_varlen_packed_vs_separate():
    """Backward dh: packed varlen (CPU cu_seqlens) == separate per-segment."""
    torch.manual_seed(720)
    H, K, V = 2, 32, 64
    C = 64
    s1_len, s2_len = 64, 128
    scale = K**-0.5

    q1 = torch.randn(1, s1_len, H, K)
    k1 = torch.randn(1, s1_len, H, K)
    v1 = torch.randn(1, s1_len, H, V)
    g1 = F.logsigmoid(torch.randn(1, s1_len, H, K))
    do1 = torch.randn(1, s1_len, H, V)
    h01 = torch.randn(1, H, K, V)
    dht1 = torch.randn(1, H, K, V)

    q2 = torch.randn(1, s2_len, H, K)
    k2 = torch.randn(1, s2_len, H, K)
    v2 = torch.randn(1, s2_len, H, V)
    g2 = F.logsigmoid(torch.randn(1, s2_len, H, K))
    do2 = torch.randn(1, s2_len, H, V)
    h02 = torch.randn(1, H, K, V)
    dht2 = torch.randn(1, H, K, V)

    # Per-segment
    gc1 = cpu_chunk_local_cumsum(g1, C)
    dh1, dh01 = cpu_chunk_bwd_dh(q1, k1, v1, gc1, do1, h0=h01, dht=dht1, scale=scale, chunk_size=C)
    gc2 = cpu_chunk_local_cumsum(g2, C)
    dh2, dh02 = cpu_chunk_bwd_dh(q2, k2, v2, gc2, do2, h0=h02, dht=dht2, scale=scale, chunk_size=C)

    # Packed
    cu = torch.tensor([0, s1_len, s1_len + s2_len], dtype=torch.long)
    q_cat = torch.cat([q1, q2], dim=1)
    k_cat = torch.cat([k1, k2], dim=1)
    v_cat = torch.cat([v1, v2], dim=1)
    gc_cat = torch.cat([gc1, gc2], dim=1)
    do_cat = torch.cat([do1, do2], dim=1)
    h0_cat = torch.cat([h01, h02], dim=0)   # [N, H, K, V]
    dht_cat = torch.cat([dht1, dht2], dim=0)

    dh_p, dh0_p = cpu_chunk_bwd_dh(
        q_cat, k_cat, v_cat, gc_cat, do_cat,
        h0=h0_cat, dht=dht_cat, scale=scale,
        cu_seqlens=cu, chunk_size=C,
    )

    NT1 = s1_len // C
    atol, rtol = 1e-5, 1e-5
    assert compare_tensor("seg1 dh", dh1, dh_p[:, :NT1], atol=atol, rtol=rtol)
    assert compare_tensor("seg2 dh", dh2, dh_p[:, NT1:], atol=atol, rtol=rtol)
    assert compare_tensor("seg1 dh0", dh01, dh0_p[:1], atol=atol, rtol=rtol)
    assert compare_tensor("seg2 dh0", dh02, dh0_p[1:], atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
