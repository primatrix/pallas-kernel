"""chunk_gla_bwd: end-to-end FLA Triton GPU (gold) vs Torch CPU orchestrator tests.

Both the FLA Triton orchestrator and our CPU reference compute the full
backward pass for GLA chunk attention.  Each side computes all intermediates
independently — no shared truth between them.

The Triton orchestrator requires g_cumsum, h, and A from the forward pass.
We run Triton's forward first to obtain these, then pass them to backward.

CPU signature:
    chunk_gla_bwd(q, k, v, g, g_cumsum, scale, initial_state, h, A, do, dht, chunk_size)

Triton signature:
    chunk_gla_bwd(q, k, v, g, g_cumsum, scale, initial_state, h, A, do, dht, chunk_size)
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

from tests.src.ops.gla.chunk import chunk_gla_bwd as cpu_chunk_gla_bwd
from tests.utils import compare_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_CUDA = torch.cuda.is_available()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

triton_imports_available = False
try:
    from fla.ops.gla.chunk import (
        chunk_gla_bwd as triton_chunk_gla_bwd,
        chunk_gla_fwd as triton_chunk_gla_fwd,
    )

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
    # ── non-aligned T (padding exercised) ──
    dict(B=2, T=100, H=4, K=32, V=64, seed=400),
    dict(B=1, T=100, H=2, K=32, V=64, seed=401, h0=True, dht=True),
    # ── non-default chunk_size ──
    dict(B=2, T=64, H=4, K=32, V=64, seed=500, chunk_size=16),
    dict(B=1, T=128, H=2, K=32, V=64, seed=501, chunk_size=32, h0=True, dht=True),
    # ── varlen: equal segments ──
    dict(B=1, T=64, H=4, K=32, V=64, seed=600, cu_seqlens=[0, 32, 64]),
    # ── varlen: unequal segments ──
    dict(B=1, T=48, H=4, K=32, V=64, seed=601, cu_seqlens=[0, 10, 24, 48]),
    # ── varlen + h0 ──
    dict(B=1, T=64, H=4, K=32, V=64, seed=602, cu_seqlens=[0, 32, 64], h0=True),
    # ── varlen + dht ──
    dict(B=1, T=64, H=4, K=32, V=64, seed=603, cu_seqlens=[0, 32, 64], dht=True),
    # ── varlen + h0 + dht ──
    dict(B=1, T=64, H=4, K=32, V=64, seed=604, cu_seqlens=[0, 32, 64], h0=True, dht=True),
    # ── varlen: single token segments ──
    dict(B=1, T=4, H=2, K=16, V=32, seed=605, cu_seqlens=[0, 1, 2, 3, 4]),
    # ── varlen: long + short ──
    dict(B=1, T=67, H=2, K=32, V=64, seed=606, cu_seqlens=[0, 3, 67]),
    # ── varlen: many segments ──
    dict(B=1, T=48, H=2, K=16, V=32, seed=607, cu_seqlens=[0, 8, 16, 24, 32, 40, 48]),
    # ── varlen: single segment ──
    dict(B=1, T=64, H=4, K=32, V=64, seed=608, cu_seqlens=[0, 64]),
    # ── varlen + h0 + dht + unequal ──
    dict(B=1, T=48, H=4, K=32, V=64, seed=609, cu_seqlens=[0, 10, 24, 48], h0=True, dht=True),
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
    if c.get("cu_seqlens"):
        parts.append(f"segs{len(c['cu_seqlens']) - 1}")
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
    scale = cfg.get("scale", K**-0.5)
    chunk_size = cfg.get("chunk_size", 64)
    C = chunk_size

    cu_list = cfg.get("cu_seqlens")
    cu = torch.tensor(cu_list, dtype=torch.long) if cu_list else None
    N = len(cu_list) - 1 if cu_list else B

    torch.manual_seed(cfg["seed"])
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    g = F.logsigmoid(torch.randn(B, T, H, K))
    do = torch.randn(B, T, H, V)
    h0 = torch.randn(N, H, K, V) if cfg.get("h0") else None
    dht = torch.randn(N, H, K, V) if cfg.get("dht") else None

    # CPU reference (handles padding internally)
    dq_cpu, dk_cpu, dv_cpu, dg_cpu, dh0_cpu = cpu_chunk_gla_bwd(
        q.float(), k.float(), v.float(), g.float(),
        None, scale, h0, None, None, do.float(), dht,
        cu_seqlens=cu, chunk_size=C,
    )

    if cu is not None:
        # Triton handles cu_seqlens internally — no manual padding needed
        (q_g, k_g, v_g, g_g, do_g, h0_g, dht_g) = _to_device(
            q, k, v, g, do, h0, dht,
        )
        cu_g = cu.to(DEVICE)

        g_cumsum_g, A_g, h_g, _, _ = triton_chunk_gla_fwd(
            q_g, k_g, v_g, g_g, None, scale, h0_g,
            output_final_state=False, cu_seqlens=cu_g, chunk_size=C,
        )

        dq_gold, dk_gold, dv_gold, dg_gold, dh0_gold = triton_chunk_gla_bwd(
            q_g, k_g, v_g, g_g, g_cumsum_g, scale, h0_g, h_g, A_g, do_g, dht_g,
            cu_seqlens=cu_g, chunk_size=C,
        )

        dq_gold = dq_gold.cpu()
        dk_gold = dk_gold.cpu()
        dv_gold = dv_gold.cpu()
        dg_gold = dg_gold.cpu()
        dh0_gold = dh0_gold.cpu() if dh0_gold is not None else None
    else:
        # Pad T to multiple of chunk_size for Triton
        NT = (T + C - 1) // C
        T_padded = NT * C
        q_t, k_t, v_t, g_t, do_t = q, k, v, g, do
        if T_padded > T:
            pad = T_padded - T
            q_t = F.pad(q, (0, 0, 0, 0, 0, pad))
            k_t = F.pad(k, (0, 0, 0, 0, 0, pad))
            v_t = F.pad(v, (0, 0, 0, 0, 0, pad))
            g_t = F.pad(g, (0, 0, 0, 0, 0, pad))
            do_t = F.pad(do, (0, 0, 0, 0, 0, pad))

        (q_g, k_g, v_g, g_g, do_g, h0_g, dht_g) = _to_device(
            q_t, k_t, v_t, g_t, do_t, h0, dht,
        )

        # Run Triton forward to get intermediates (g_cumsum, A, h)
        g_cumsum_g, A_g, h_g, _, _ = triton_chunk_gla_fwd(
            q_g, k_g, v_g, g_g, None, scale, h0_g,
            output_final_state=False, chunk_size=C,
        )

        # Triton backward with precomputed intermediates
        dq_gold, dk_gold, dv_gold, dg_gold, dh0_gold = triton_chunk_gla_bwd(
            q_g, k_g, v_g, g_g, g_cumsum_g, scale, h0_g, h_g, A_g, do_g, dht_g,
            chunk_size=C,
        )

        # Slice back to original T for comparison
        dq_gold = dq_gold[:, :T].cpu()
        dk_gold = dk_gold[:, :T].cpu()
        dv_gold = dv_gold[:, :T].cpu()
        dg_gold = dg_gold[:, :T].cpu()
        dh0_gold = dh0_gold.cpu() if dh0_gold is not None else None

    # Tight tolerance for dq, dk
    atol_qk = cfg.get("atol", 5e-5)
    rtol_qk = cfg.get("rtol", 5e-5)
    assert compare_tensor("dq", dq_gold, dq_cpu, atol=atol_qk, rtol=rtol_qk)
    assert compare_tensor("dk", dk_gold, dk_cpu, atol=atol_qk, rtol=rtol_qk)

    # dg (reverse cumsum) and dv (accumulated intermediates)
    assert compare_tensor("dg", dg_gold, dg_cpu, atol=atol_qk, rtol=rtol_qk)
    assert compare_tensor("dv", dv_gold, dv_cpu, atol=atol_qk, rtol=rtol_qk)

    if cfg.get("h0"):
        assert compare_tensor("dh0", dh0_gold, dh0_cpu, atol=atol_qk, rtol=rtol_qk)


# ============================================================================
# Structural test — varlen packed vs separate
# ============================================================================


def test_bwd_varlen_packed_vs_separate():
    """Backward: packed varlen (CPU cu_seqlens) == separate batch (CPU per-segment)."""
    torch.manual_seed(700)
    H, K, V = 2, 32, 64
    C = 64
    s1_len, s2_len = 10, 14

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

    scale = K**-0.5

    # Separate backward via CPU (one segment at a time)
    dq1, dk1, dv1, dg1, _ = cpu_chunk_gla_bwd(
        q1, k1, v1, g1, None, scale, None, None, None, do1, None, chunk_size=C,
    )
    dq2, dk2, dv2, dg2, _ = cpu_chunk_gla_bwd(
        q2, k2, v2, g2, None, scale, None, None, None, do2, None, chunk_size=C,
    )

    # Packed backward via CPU with cu_seqlens
    q_cat = torch.cat([q1, q2], dim=1)
    k_cat = torch.cat([k1, k2], dim=1)
    v_cat = torch.cat([v1, v2], dim=1)
    g_cat = torch.cat([g1, g2], dim=1)
    do_cat = torch.cat([do1, do2], dim=1)
    cu = torch.tensor([0, s1_len, s1_len + s2_len], dtype=torch.long)

    dq_p, dk_p, dv_p, dg_p, _ = cpu_chunk_gla_bwd(
        q_cat, k_cat, v_cat, g_cat, None, scale, None, None, None, do_cat, None,
        cu_seqlens=cu, chunk_size=C,
    )

    atol, rtol = 1e-5, 1e-5
    assert compare_tensor("seg1 dq", dq1, dq_p[:, :s1_len], atol=atol, rtol=rtol)
    assert compare_tensor("seg2 dq", dq2, dq_p[:, s1_len:], atol=atol, rtol=rtol)
    assert compare_tensor("seg1 dk", dk1, dk_p[:, :s1_len], atol=atol, rtol=rtol)
    assert compare_tensor("seg2 dk", dk2, dk_p[:, s1_len:], atol=atol, rtol=rtol)
    assert compare_tensor("seg1 dv", dv1, dv_p[:, :s1_len], atol=atol, rtol=rtol)
    assert compare_tensor("seg2 dv", dv2, dv_p[:, s1_len:], atol=atol, rtol=rtol)
    assert compare_tensor("seg1 dg", dg1, dg_p[:, :s1_len], atol=atol, rtol=rtol)
    assert compare_tensor("seg2 dg", dg2, dg_p[:, s1_len:], atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
