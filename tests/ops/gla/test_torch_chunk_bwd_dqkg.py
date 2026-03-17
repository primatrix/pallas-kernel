"""chunk_gla_bwd_dqkg: FLA Triton GPU (gold) vs Torch CPU kernel tests.

Both the FLA Triton kernel and our CPU reference compute the combined
inter-chunk + intra-chunk gradients for q, k, and g.

Shared intermediates (g_cumsum, h, dh, dq_intra, dk_intra) are computed
on CPU as shared truth, then passed to both implementations.
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
    chunk_gla_bwd_dqkg as cpu_chunk_gla_bwd_dqkg,
    chunk_gla_bwd_dqk_intra as cpu_chunk_gla_bwd_dqk_intra,
    chunk_gla_bwd_dA as cpu_chunk_gla_bwd_dA,
    chunk_local_cumsum as cpu_chunk_local_cumsum,
    chunk_fwd_h as cpu_chunk_fwd_h,
    chunk_bwd_dh as cpu_chunk_bwd_dh,
)
from tests.utils import compare_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_CUDA = torch.cuda.is_available()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

triton_imports_available = False
try:
    from fla.ops.gla.chunk import chunk_gla_bwd_dqkg as triton_chunk_gla_bwd_dqkg

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
    # ── K >> V ──
    dict(B=2, T=64, H=4, K=128, V=16, seed=21),
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
    C = chunk_size

    assert T % C == 0, f"T={T} must be a multiple of chunk_size={C}"

    torch.manual_seed(cfg["seed"])
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    g = F.logsigmoid(torch.randn(B, T, H, K))
    do = torch.randn(B, T, H, V)
    h0 = torch.randn(B, H, K, V) if cfg.get("h0") else None
    dht = torch.randn(B, H, K, V) if cfg.get("dht") else None

    # Compute shared intermediates on CPU
    g_cumsum = cpu_chunk_local_cumsum(g, C)
    h_cpu, _ = cpu_chunk_fwd_h(k, v, g_cumsum, h0=h0, chunk_size=C)
    dh_cpu, _ = cpu_chunk_bwd_dh(
        q, k, v, g_cumsum, do,
        h0=h0, dht=dht, scale=scale, chunk_size=C,
    )
    dA_cpu = cpu_chunk_gla_bwd_dA(v, do, scale, chunk_size=C)
    dq_intra_cpu, dk_intra_cpu = cpu_chunk_gla_bwd_dqk_intra(
        q, k, g_cumsum, dA_cpu, chunk_size=C,
    )

    # CPU reference
    dq_cpu, dk_cpu, dg_cpu = cpu_chunk_gla_bwd_dqkg(
        q, k, v, h_cpu, g_cumsum, do, dh_cpu,
        dq_intra_cpu, dk_intra_cpu, scale, chunk_size=C,
    )

    # Triton GPU
    (q_g, k_g, v_g, h_g, gc_g, do_g, dh_g, dq_i_g, dk_i_g) = _to_device(
        q, k, v, h_cpu, g_cumsum, do, dh_cpu, dq_intra_cpu, dk_intra_cpu,
    )
    dq_gold, dk_gold, dg_gold = triton_chunk_gla_bwd_dqkg(
        q_g, k_g, v_g, h_g, gc_g, do_g, dh_g, dq_i_g, dk_i_g,
        scale=scale, chunk_size=C,
    )
    dq_gold = dq_gold.cpu()
    dk_gold = dk_gold.cpu()
    dg_gold = dg_gold.cpu()

    assert compare_tensor("dq", dq_gold, dq_cpu, atol=atol, rtol=rtol)
    assert compare_tensor("dk", dk_gold, dk_cpu, atol=atol, rtol=rtol)
    assert compare_tensor("dg", dg_gold, dg_cpu, atol=atol, rtol=rtol)


# ============================================================================
# Structural test — varlen packed vs separate
# ============================================================================


def test_varlen_packed_vs_separate():
    """Backward dqkg: packed varlen (CPU cu_seqlens) == separate per-segment."""
    torch.manual_seed(740)
    H, K, V = 2, 32, 64
    C = 64
    s1_len, s2_len = 64, 128
    scale = K**-0.5

    q1 = torch.randn(1, s1_len, H, K)
    k1 = torch.randn(1, s1_len, H, K)
    v1 = torch.randn(1, s1_len, H, V)
    g1 = F.logsigmoid(torch.randn(1, s1_len, H, K))
    do1 = torch.randn(1, s1_len, H, V)

    q2 = torch.randn(1, s2_len, H, K)
    k2 = torch.randn(1, s2_len, H, K)
    v2 = torch.randn(1, s2_len, H, V)
    g2 = F.logsigmoid(torch.randn(1, s2_len, H, K))
    do2 = torch.randn(1, s2_len, H, V)

    # Per-segment: compute all intermediates and dqkg separately
    gc1 = cpu_chunk_local_cumsum(g1, C)
    h1, _ = cpu_chunk_fwd_h(k1, v1, gc1, chunk_size=C)
    dh1, _ = cpu_chunk_bwd_dh(q1, k1, v1, gc1, do1, scale=scale, chunk_size=C)
    dA1 = cpu_chunk_gla_bwd_dA(v1, do1, scale, chunk_size=C)
    dqi1, dki1 = cpu_chunk_gla_bwd_dqk_intra(q1, k1, gc1, dA1, chunk_size=C)
    dq1, dk1, dg1 = cpu_chunk_gla_bwd_dqkg(
        q1, k1, v1, h1, gc1, do1, dh1, dqi1, dki1, scale, chunk_size=C,
    )

    gc2 = cpu_chunk_local_cumsum(g2, C)
    h2, _ = cpu_chunk_fwd_h(k2, v2, gc2, chunk_size=C)
    dh2, _ = cpu_chunk_bwd_dh(q2, k2, v2, gc2, do2, scale=scale, chunk_size=C)
    dA2 = cpu_chunk_gla_bwd_dA(v2, do2, scale, chunk_size=C)
    dqi2, dki2 = cpu_chunk_gla_bwd_dqk_intra(q2, k2, gc2, dA2, chunk_size=C)
    dq2, dk2, dg2 = cpu_chunk_gla_bwd_dqkg(
        q2, k2, v2, h2, gc2, do2, dh2, dqi2, dki2, scale, chunk_size=C,
    )

    # Packed
    cu = torch.tensor([0, s1_len, s1_len + s2_len], dtype=torch.long)
    q_cat = torch.cat([q1, q2], dim=1)
    k_cat = torch.cat([k1, k2], dim=1)
    v_cat = torch.cat([v1, v2], dim=1)
    gc_cat = torch.cat([gc1, gc2], dim=1)
    do_cat = torch.cat([do1, do2], dim=1)
    h_cat = torch.cat([h1, h2], dim=1)       # [1, NT_total, H, K, V]
    dh_cat = torch.cat([dh1, dh2], dim=1)
    dqi_cat = torch.cat([dqi1, dqi2], dim=1)
    dki_cat = torch.cat([dki1, dki2], dim=1)

    dq_p, dk_p, dg_p = cpu_chunk_gla_bwd_dqkg(
        q_cat, k_cat, v_cat, h_cat, gc_cat, do_cat, dh_cat,
        dqi_cat, dki_cat, scale,
        cu_seqlens=cu, chunk_size=C,
    )

    atol, rtol = 1e-5, 1e-5
    assert compare_tensor("seg1 dq", dq1, dq_p[:, :s1_len], atol=atol, rtol=rtol)
    assert compare_tensor("seg2 dq", dq2, dq_p[:, s1_len:], atol=atol, rtol=rtol)
    assert compare_tensor("seg1 dk", dk1, dk_p[:, :s1_len], atol=atol, rtol=rtol)
    assert compare_tensor("seg2 dk", dk2, dk_p[:, s1_len:], atol=atol, rtol=rtol)
    assert compare_tensor("seg1 dg", dg1, dg_p[:, :s1_len], atol=atol, rtol=rtol)
    assert compare_tensor("seg2 dg", dg2, dg_p[:, s1_len:], atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
