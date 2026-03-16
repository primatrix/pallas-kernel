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


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
