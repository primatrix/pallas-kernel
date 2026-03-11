"""fused_recurrent_gla: FLA Triton GPU (gold) vs Torch CPU kernel tests."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import torch
import torch.nn.functional as F

from tests.src.ops.gla import fused_recurrent_gla as cpu_fused_recurrent
from tests.utils import compare_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_CUDA = torch.cuda.is_available()

triton_imports_available = False
try:
    from fla.ops.gla import fused_recurrent_gla as triton_fused_recurrent

    triton_imports_available = True
except ImportError:
    pass

requires_triton = pytest.mark.skipif(
    not (HAS_CUDA and triton_imports_available),
    reason="Triton / CUDA not available",
)

# ============================================================================
# Unified test configs — add new cases here
#
# Each dict can contain:
#   B, T, H, K, V  — shape (required)
#   seed            — random seed (required)
#   atol            — absolute tolerance (default 1e-6)
#   rtol            — relative tolerance (default 1e-5)
#   gate            — "gk" | "gv" | "gk+gv" | "none" (default "gk")
#   h0              — True/False, whether to use initial_state (default False)
#   cu_seqlens      — list[int] or None (default None)
#   reverse         — True/False (default False)
#   scale           — float or None (default None = K^{-0.5})
#
# np.allclose check: |a - b| <= atol + rtol * |b|
# ============================================================================

CASES = [
    # ── standard shapes ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=42),
    dict(B=2, T=32, H=4, K=32, V=64, seed=13, h0=True),
    dict(B=1, T=128, H=2, K=64, V=128, seed=7),
    # ── single head ──
    dict(B=2, T=32, H=1, K=32, V=64, seed=10),
    # ── K != V ──
    dict(B=2, T=32, H=4, K=16, V=128, seed=20),
    dict(B=2, T=32, H=4, K=128, V=16, seed=21),
    # ── very short T ──
    dict(B=1, T=1, H=2, K=32, V=64, seed=30),
    dict(B=1, T=3, H=2, K=32, V=64, seed=31),
    # ── odd T ──
    dict(B=2, T=37, H=4, K=16, V=32, seed=40),
    dict(B=1, T=50, H=2, K=32, V=64, seed=41),
    dict(B=1, T=100, H=2, K=32, V=64, seed=42, h0=True),
    # ── large batch ──
    dict(B=8, T=32, H=4, K=32, V=64, seed=50),
    # ── many heads ──
    dict(B=1, T=64, H=16, K=32, V=64, seed=60),
    # ── small dims ──
    dict(B=2, T=32, H=2, K=8, V=16, seed=70),
    dict(B=2, T=32, H=2, K=8, V=16, seed=71, h0=True),
    # ── various ──
    dict(B=1, T=16, H=1, K=16, V=16, seed=99),
    dict(B=4, T=64, H=8, K=32, V=64, seed=99),
    dict(B=2, T=48, H=4, K=32, V=32, seed=99),
    # ── gate modes ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=100, gate="gv"),
    dict(B=2, T=32, H=4, K=32, V=64, seed=101, gate="gk+gv"),
    dict(B=2, T=32, H=4, K=32, V=64, seed=102, gate="none", atol=5e-6),
    # ── reverse ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=110, reverse=True),
    dict(B=2, T=32, H=4, K=32, V=64, seed=111, reverse=True, h0=True),
    dict(B=1, T=1, H=2, K=16, V=32, seed=112, reverse=True),
    # ── custom scale ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=200, scale=0.1),
    # ── varlen: equal segments ──
    dict(B=1, T=32, H=4, K=32, V=64, seed=7, cu_seqlens=[0, 16, 32]),
    dict(B=1, T=96, H=4, K=32, V=64, seed=10, cu_seqlens=[0, 32, 64, 96]),
    # ── varlen: unequal segments ──
    dict(B=1, T=48, H=4, K=32, V=64, seed=11, cu_seqlens=[0, 10, 24, 48]),
    dict(B=1, T=23, H=2, K=16, V=32, seed=12, cu_seqlens=[0, 7, 20, 23]),
    # ── varlen + h0 ──
    dict(B=1, T=32, H=4, K=32, V=64, seed=17, cu_seqlens=[0, 16, 32], h0=True),
    dict(B=1, T=48, H=4, K=32, V=64, seed=18, cu_seqlens=[0, 10, 24, 48], h0=True),
    # ── varlen: single token segments ──
    dict(B=1, T=4, H=2, K=16, V=32, seed=20, cu_seqlens=[0, 1, 2, 3, 4]),
    # ── varlen: long + short ──
    dict(B=1, T=67, H=2, K=32, V=64, seed=25, cu_seqlens=[0, 3, 67]),
    dict(B=1, T=64, H=2, K=32, V=64, seed=26, cu_seqlens=[0, 61, 64]),
    # ── varlen: many segments ──
    dict(B=1, T=48, H=2, K=16, V=32, seed=30, cu_seqlens=[0, 8, 16, 24, 32, 40, 48]),
    # ── varlen: single segment (degenerates to no varlen) ──
    dict(B=1, T=64, H=4, K=32, V=64, seed=35, cu_seqlens=[0, 64]),
    # ── varlen + reverse ──
    dict(B=1, T=32, H=4, K=32, V=64, seed=120, cu_seqlens=[0, 16, 32], reverse=True),
    dict(B=1, T=23, H=2, K=16, V=32, seed=121, cu_seqlens=[0, 7, 20, 23], reverse=True),
    dict(
        B=1,
        T=32,
        H=4,
        K=32,
        V=64,
        seed=122,
        cu_seqlens=[0, 16, 32],
        reverse=True,
        h0=True,
    ),
    # ── gate × varlen ──
    dict(B=1, T=32, H=4, K=32, V=64, seed=130, gate="gv", cu_seqlens=[0, 16, 32]),
    dict(B=1, T=32, H=4, K=32, V=64, seed=131, gate="gk+gv", cu_seqlens=[0, 16, 32]),
    dict(
        B=1,
        T=32,
        H=4,
        K=32,
        V=64,
        seed=132,
        gate="none",
        cu_seqlens=[0, 16, 32],
        atol=5e-6,
    ),
    # ── gate × h0 ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=133, gate="gk+gv", h0=True),
    dict(B=2, T=32, H=4, K=32, V=64, seed=134, gate="gv", h0=True),
    # ── gate × reverse ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=135, gate="gv", reverse=True),
    dict(B=2, T=32, H=4, K=32, V=64, seed=136, gate="gk+gv", reverse=True),
    # ── scale × h0 / varlen ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=140, scale=0.1, h0=True),
    dict(B=1, T=32, H=4, K=32, V=64, seed=141, scale=0.1, cu_seqlens=[0, 16, 32]),
    # ── long sequence numerical stability ──
    dict(B=1, T=256, H=2, K=32, V=64, seed=300),
    dict(B=1, T=512, H=2, K=32, V=64, seed=301),
    dict(B=1, T=1024, H=2, K=32, V=64, seed=302),
    # ── long + h0 ──
    dict(B=1, T=256, H=2, K=32, V=64, seed=303, h0=True),
    dict(B=1, T=512, H=2, K=32, V=64, seed=304, h0=True),
    # ── long + gate modes ──
    dict(B=1, T=256, H=2, K=32, V=64, seed=310, gate="gv"),
    dict(B=1, T=256, H=2, K=32, V=64, seed=311, gate="gk+gv"),
    dict(B=1, T=512, H=2, K=32, V=64, seed=312, gate="gk+gv"),
    dict(B=1, T=256, H=2, K=32, V=64, seed=313, gate="none", atol=2e-5),
    # ── long + reverse ──
    dict(B=1, T=256, H=2, K=32, V=64, seed=320, reverse=True),
    dict(B=1, T=512, H=2, K=32, V=64, seed=321, reverse=True),
    dict(B=1, T=256, H=2, K=32, V=64, seed=322, reverse=True, h0=True),
    # ── long + scale ──
    dict(B=1, T=256, H=2, K=32, V=64, seed=330, scale=0.1),
    dict(B=1, T=512, H=2, K=32, V=64, seed=331, scale=0.01),
    # ── long + varlen ──
    dict(B=1, T=256, H=2, K=32, V=64, seed=340, cu_seqlens=[0, 128, 256]),
    dict(B=1, T=512, H=2, K=32, V=64, seed=341, cu_seqlens=[0, 100, 300, 512]),
    dict(B=1, T=256, H=2, K=32, V=64, seed=342, cu_seqlens=[0, 128, 256], h0=True),
    # ── long + large dims (累积误差 × 高维) ──
    dict(B=1, T=256, H=2, K=64, V=128, seed=350),
    dict(B=1, T=256, H=2, K=128, V=64, seed=351),
    dict(B=1, T=512, H=2, K=64, V=128, seed=352, h0=True),
    # ── long + multi-batch ──
    dict(B=4, T=256, H=2, K=32, V=64, seed=360),
    dict(B=2, T=512, H=4, K=32, V=64, seed=361),
    # ── long + many heads ──
    dict(B=1, T=256, H=8, K=32, V=64, seed=370),
    dict(B=1, T=512, H=8, K=32, V=64, seed=371, h0=True),
    # ── long + combo (kitchen sink) ──
    dict(
        B=1,
        T=256,
        H=4,
        K=32,
        V=64,
        seed=380,
        gate="gk+gv",
        h0=True,
        reverse=True,
        scale=0.1,
    ),
    dict(
        B=1,
        T=256,
        H=4,
        K=32,
        V=64,
        seed=381,
        gate="gk+gv",
        h0=True,
        cu_seqlens=[0, 64, 180, 256],
    ),
]


def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    gate = c.get("gate", "gk")
    if gate != "gk":
        parts.append(f"gate={gate}")
    if c.get("h0"):
        parts.append("h0")
    if c.get("cu_seqlens"):
        parts.append(f"segs{len(c['cu_seqlens']) - 1}")
    if c.get("reverse"):
        parts.append("rev")
    if c.get("scale") is not None:
        parts.append(f"scale={c['scale']}")
    return "-".join(parts)


# ============================================================================
# Helpers
# ============================================================================


def _to_device(*tensors):
    return tuple(t.to(DEVICE) for t in tensors)


def _run_gold(
    q, k, v, *, gk=None, gv=None, h0=None, cu=None, scale=None, reverse=False
):
    args = dict(output_final_state=True, reverse=reverse)
    if gk is not None:
        args["gk"] = gk.to(DEVICE)
    if gv is not None:
        args["gv"] = gv.to(DEVICE)
    if h0 is not None:
        args["initial_state"] = h0.to(DEVICE)
    if cu is not None:
        args["cu_seqlens"] = cu.to(DEVICE)
    if scale is not None:
        args["scale"] = scale
    o, s = triton_fused_recurrent(*_to_device(q, k, v), **args)
    return o.cpu(), s.cpu() if s is not None else None


def _run_cpu(q, k, v, *, gk=None, gv=None, h0=None, cu=None, scale=None, reverse=False):
    return cpu_fused_recurrent(
        q,
        k,
        v,
        gk=gk,
        gv=gv,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu,
        scale=scale,
        reverse=reverse,
    )


# ============================================================================
# Main parametrized test — gold vs cpu
# ============================================================================


@requires_triton
@pytest.mark.parametrize("cfg", CASES, ids=[_case_id(c) for c in CASES])
def test_gold_vs_cpu(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    atol = cfg.get("atol", 1e-6)
    rtol = cfg.get("rtol", 1e-5)
    gate = cfg.get("gate", "gk")
    reverse = cfg.get("reverse", False)
    scale = cfg.get("scale", None)

    torch.manual_seed(cfg["seed"])
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K)) if "gk" in gate else None
    gv = F.logsigmoid(torch.randn(B, T, H, V)) if "gv" in gate else None

    cu_list = cfg.get("cu_seqlens")
    cu = torch.tensor(cu_list, dtype=torch.long) if cu_list else None
    N = len(cu_list) - 1 if cu_list else B
    h0 = torch.randn(N, H, K, V) if cfg.get("h0") else None

    o_gold, s_gold = _run_gold(
        q, k, v, gk=gk, gv=gv, h0=h0, cu=cu, scale=scale, reverse=reverse
    )
    o_cpu, s_cpu = _run_cpu(
        q, k, v, gk=gk, gv=gv, h0=h0, cu=cu, scale=scale, reverse=reverse
    )
    assert compare_tensor("output", o_gold, o_cpu, atol=atol, rtol=rtol)
    assert compare_tensor("final_state", s_gold, s_cpu, atol=atol, rtol=rtol)


# ============================================================================
# Structural tests (different assertion logic)
# ============================================================================


@requires_triton
def test_state_split():
    """Split sequence in 2 halves: state continuity."""
    torch.manual_seed(77)
    B, T, H, K, V = 1, 40, 2, 16, 32
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))
    T1 = T // 2

    _, s_full_gold = _run_gold(q, k, v, gk=gk)
    _, s_full_cpu = _run_cpu(q, k, v, gk=gk)

    _, s1_gold = _run_gold(q[:, :T1], k[:, :T1], v[:, :T1], gk=gk[:, :T1])
    _, s2_gold = _run_gold(q[:, T1:], k[:, T1:], v[:, T1:], gk=gk[:, T1:], h0=s1_gold)

    _, s1_cpu = _run_cpu(q[:, :T1], k[:, :T1], v[:, :T1], gk=gk[:, :T1])
    _, s2_cpu = _run_cpu(q[:, T1:], k[:, T1:], v[:, T1:], gk=gk[:, T1:], h0=s1_cpu)

    assert compare_tensor(
        "full state (gold vs cpu)", s_full_gold, s_full_cpu, atol=1e-4, rtol=1e-4
    )
    assert compare_tensor(
        "gold: full vs split", s_full_gold, s2_gold, atol=1e-4, rtol=1e-4
    )
    assert compare_tensor(
        "cpu: full vs split", s_full_cpu, s2_cpu, atol=1e-5, rtol=1e-5
    )
    assert compare_tensor(
        "split state (gold vs cpu)", s2_gold, s2_cpu, atol=1e-4, rtol=1e-4
    )


@requires_triton
def test_varlen_packed_vs_separate():
    """Packed varlen == separate batch (gold vs cpu)."""
    torch.manual_seed(123)
    H, K, V = 2, 32, 64
    s1_len, s2_len = 10, 14

    q1, k1, v1 = (
        torch.randn(1, s1_len, H, K),
        torch.randn(1, s1_len, H, K),
        torch.randn(1, s1_len, H, V),
    )
    g1 = F.logsigmoid(torch.randn(1, s1_len, H, K))
    q2, k2, v2 = (
        torch.randn(1, s2_len, H, K),
        torch.randn(1, s2_len, H, K),
        torch.randn(1, s2_len, H, V),
    )
    g2 = F.logsigmoid(torch.randn(1, s2_len, H, K))

    o1_gold, s1_gold = _run_gold(q1, k1, v1, gk=g1)
    o2_gold, s2_gold = _run_gold(q2, k2, v2, gk=g2)

    q_cat = torch.cat([q1, q2], dim=1)
    k_cat = torch.cat([k1, k2], dim=1)
    v_cat = torch.cat([v1, v2], dim=1)
    g_cat = torch.cat([g1, g2], dim=1)
    cu = torch.tensor([0, s1_len, s1_len + s2_len], dtype=torch.long)

    o_cpu, s_cpu = _run_cpu(q_cat, k_cat, v_cat, gk=g_cat, cu=cu)

    assert compare_tensor(
        "seg1 output", o1_gold, o_cpu[:, :s1_len], atol=1e-4, rtol=1e-4
    )
    assert compare_tensor(
        "seg2 output", o2_gold, o_cpu[:, s1_len:], atol=1e-4, rtol=1e-4
    )
    assert compare_tensor(
        "seg1 state", s1_gold.squeeze(0), s_cpu[0], atol=1e-4, rtol=1e-4
    )
    assert compare_tensor(
        "seg2 state", s2_gold.squeeze(0), s_cpu[1], atol=1e-4, rtol=1e-4
    )


@requires_triton
def test_no_final_state():
    """output_final_state=False returns None for final_state."""
    torch.manual_seed(210)
    B, T, H, K, V = 2, 32, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))

    o_gold, s_gold = triton_fused_recurrent(
        *_to_device(q, k, v),
        gk=gk.to(DEVICE),
        output_final_state=False,
    )
    o_cpu, s_cpu = cpu_fused_recurrent(q, k, v, gk=gk, output_final_state=False)

    assert s_gold is None, f"gold final_state should be None, got {type(s_gold)}"
    assert s_cpu is None, f"cpu final_state should be None, got {type(s_cpu)}"
    assert compare_tensor("output", o_gold.cpu(), o_cpu, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
