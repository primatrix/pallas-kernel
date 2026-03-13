"""fused_recurrent_gla backward: FLA Triton GPU (gold) vs Torch CPU kernel tests."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import torch
import torch.nn.functional as F

from tests.src.ops.gla import fused_recurrent_gla_bwd as cpu_bwd
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
# Test configs — mirrors forward test structure
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
    # ── varlen: single segment ──
    dict(B=1, T=64, H=4, K=32, V=64, seed=35, cu_seqlens=[0, 64]),
    # ── varlen + reverse ──
    dict(B=1, T=32, H=4, K=32, V=64, seed=120, cu_seqlens=[0, 16, 32], reverse=True),
    dict(B=1, T=23, H=2, K=16, V=32, seed=121, cu_seqlens=[0, 7, 20, 23], reverse=True),
    dict(
        B=1, T=32, H=4, K=32, V=64, seed=122,
        cu_seqlens=[0, 16, 32], reverse=True, h0=True,
    ),
    # ── gate × varlen ──
    dict(B=1, T=32, H=4, K=32, V=64, seed=130, gate="gv", cu_seqlens=[0, 16, 32]),
    dict(B=1, T=32, H=4, K=32, V=64, seed=131, gate="gk+gv", cu_seqlens=[0, 16, 32]),
    dict(
        B=1, T=32, H=4, K=32, V=64, seed=132,
        gate="none", cu_seqlens=[0, 16, 32], atol=5e-6,
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
    # ── longer sequences ──
    dict(B=1, T=256, H=2, K=32, V=64, seed=300),
    dict(B=1, T=256, H=2, K=32, V=64, seed=303, h0=True),
    dict(B=1, T=256, H=2, K=32, V=64, seed=310, gate="gv"),
    dict(B=1, T=256, H=2, K=32, V=64, seed=311, gate="gk+gv"),
    dict(B=1, T=256, H=2, K=32, V=64, seed=313, gate="none", atol=2e-5),
    # ── long + reverse ──
    dict(B=1, T=256, H=2, K=32, V=64, seed=320, reverse=True),
    dict(B=1, T=256, H=2, K=32, V=64, seed=322, reverse=True, h0=True),
    # ── long + scale ──
    dict(B=1, T=256, H=2, K=32, V=64, seed=330, scale=0.1),
    # ── long + varlen ──
    dict(B=1, T=256, H=2, K=32, V=64, seed=340, cu_seqlens=[0, 128, 256]),
    dict(B=1, T=256, H=2, K=32, V=64, seed=342, cu_seqlens=[0, 128, 256], h0=True),
    # ── combo ──
    dict(
        B=1, T=256, H=4, K=32, V=64, seed=380,
        gate="gk+gv", h0=True, reverse=True, scale=0.1,
    ),
    dict(
        B=1, T=256, H=4, K=32, V=64, seed=381,
        gate="gk+gv", h0=True, cu_seqlens=[0, 64, 180, 256],
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


def _run_triton_bwd(
    q, k, v, *, gk=None, gv=None, h0=None, cu=None, scale=None,
    reverse=False, do_target=None, dht_target=None,
):
    """Run forward + backward via Triton autograd and return grads on CPU."""
    device = DEVICE

    q_g = q.to(device).requires_grad_(True)
    k_g = k.to(device).requires_grad_(True)
    v_g = v.to(device).requires_grad_(True)
    gk_g = gk.to(device).requires_grad_(True) if gk is not None else None
    gv_g = gv.to(device).requires_grad_(True) if gv is not None else None
    h0_g = h0.to(device).requires_grad_(True) if h0 is not None else None

    kwargs = dict(output_final_state=True, reverse=reverse)
    if gk_g is not None:
        kwargs["gk"] = gk_g
    if gv_g is not None:
        kwargs["gv"] = gv_g
    if h0_g is not None:
        kwargs["initial_state"] = h0_g
    if cu is not None:
        kwargs["cu_seqlens"] = cu.to(device)
    if scale is not None:
        kwargs["scale"] = scale

    o, ht = triton_fused_recurrent(q_g, k_g, v_g, **kwargs)

    loss = (o * do_target.to(device)).sum()
    if ht is not None and dht_target is not None:
        loss = loss + (ht * dht_target.to(device)).sum()
    loss.backward()

    return dict(
        dq=q_g.grad.cpu(),
        dk=k_g.grad.cpu(),
        dv=v_g.grad.cpu(),
        dgk=gk_g.grad.cpu() if gk_g is not None else None,
        dgv=gv_g.grad.cpu() if gv_g is not None else None,
        dh0=h0_g.grad.cpu() if h0_g is not None else None,
        o=o.detach().cpu(),
    )


def _run_cpu_bwd(
    q, k, v, *, gk=None, gv=None, h0=None, cu=None, scale=None,
    reverse=False, do_target=None, dht_target=None,
):
    """Run CPU backward and return grads."""
    dq, dk, dv, dgk, dgv, dh0 = cpu_bwd(
        q=q, k=k, v=v, gk=gk, gv=gv,
        do=do_target, dht=dht_target,
        scale=scale, initial_state=h0,
        reverse=reverse, cu_seqlens=cu,
    )
    return dict(dq=dq, dk=dk, dv=dv, dgk=dgk, dgv=dgv, dh0=dh0)


# ============================================================================
# Main parametrized test
# ============================================================================


@requires_triton
@pytest.mark.parametrize("cfg", CASES, ids=[_case_id(c) for c in CASES])
def test_bwd_gold_vs_cpu(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    atol = cfg.get("atol", 1e-4)
    rtol = cfg.get("rtol", 1e-4)
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

    # Random upstream gradients
    do_target = torch.randn(B, T, H, V)
    dht_target = torch.randn(N, H, K, V) if cfg.get("h0") else None

    gold = _run_triton_bwd(
        q, k, v, gk=gk, gv=gv, h0=h0, cu=cu, scale=scale,
        reverse=reverse, do_target=do_target, dht_target=dht_target,
    )
    cpu = _run_cpu_bwd(
        q, k, v, gk=gk, gv=gv, h0=h0, cu=cu, scale=scale,
        reverse=reverse, do_target=do_target, dht_target=dht_target,
    )

    assert compare_tensor("dq", gold["dq"], cpu["dq"], atol=atol, rtol=rtol)
    assert compare_tensor("dk", gold["dk"], cpu["dk"], atol=atol, rtol=rtol)
    assert compare_tensor("dv", gold["dv"], cpu["dv"], atol=atol, rtol=rtol)
    if gk is not None:
        assert compare_tensor("dgk", gold["dgk"], cpu["dgk"], atol=atol, rtol=rtol)
    if gv is not None:
        assert compare_tensor("dgv", gold["dgv"], cpu["dgv"], atol=atol, rtol=rtol)
    if h0 is not None:
        assert compare_tensor("dh0", gold["dh0"], cpu["dh0"], atol=atol, rtol=rtol)


# ============================================================================
# Structural tests
# ============================================================================


@requires_triton
def test_bwd_no_gate():
    """Backward with no gates at all."""
    torch.manual_seed(500)
    B, T, H, K, V = 2, 32, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    do_target = torch.randn(B, T, H, V)

    gold = _run_triton_bwd(q, k, v, do_target=do_target)
    cpu = _run_cpu_bwd(q, k, v, do_target=do_target)

    assert compare_tensor("dq", gold["dq"], cpu["dq"], atol=1e-4, rtol=1e-4)
    assert compare_tensor("dk", gold["dk"], cpu["dk"], atol=1e-4, rtol=1e-4)
    assert compare_tensor("dv", gold["dv"], cpu["dv"], atol=1e-4, rtol=1e-4)


@requires_triton
def test_bwd_with_dht():
    """Backward with final state gradient."""
    torch.manual_seed(501)
    B, T, H, K, V = 2, 32, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))
    h0 = torch.randn(B, H, K, V)
    do_target = torch.randn(B, T, H, V)
    dht_target = torch.randn(B, H, K, V)

    gold = _run_triton_bwd(
        q, k, v, gk=gk, h0=h0,
        do_target=do_target, dht_target=dht_target,
    )
    cpu = _run_cpu_bwd(
        q, k, v, gk=gk, h0=h0,
        do_target=do_target, dht_target=dht_target,
    )

    assert compare_tensor("dq", gold["dq"], cpu["dq"], atol=1e-4, rtol=1e-4)
    assert compare_tensor("dk", gold["dk"], cpu["dk"], atol=1e-4, rtol=1e-4)
    assert compare_tensor("dv", gold["dv"], cpu["dv"], atol=1e-4, rtol=1e-4)
    assert compare_tensor("dgk", gold["dgk"], cpu["dgk"], atol=1e-4, rtol=1e-4)
    assert compare_tensor("dh0", gold["dh0"], cpu["dh0"], atol=1e-4, rtol=1e-4)


@requires_triton
def test_bwd_varlen_packed_vs_separate():
    """Backward: packed varlen == separate batch."""
    torch.manual_seed(502)
    H, K, V = 2, 32, 64
    s1_len, s2_len = 10, 14

    q1, k1, v1 = (
        torch.randn(1, s1_len, H, K),
        torch.randn(1, s1_len, H, K),
        torch.randn(1, s1_len, H, V),
    )
    g1 = F.logsigmoid(torch.randn(1, s1_len, H, K))
    do1 = torch.randn(1, s1_len, H, V)

    q2, k2, v2 = (
        torch.randn(1, s2_len, H, K),
        torch.randn(1, s2_len, H, K),
        torch.randn(1, s2_len, H, V),
    )
    g2 = F.logsigmoid(torch.randn(1, s2_len, H, K))
    do2 = torch.randn(1, s2_len, H, V)

    # Separate backward via Triton
    gold1 = _run_triton_bwd(q1, k1, v1, gk=g1, do_target=do1)
    gold2 = _run_triton_bwd(q2, k2, v2, gk=g2, do_target=do2)

    # Packed backward via CPU
    q_cat = torch.cat([q1, q2], dim=1)
    k_cat = torch.cat([k1, k2], dim=1)
    v_cat = torch.cat([v1, v2], dim=1)
    g_cat = torch.cat([g1, g2], dim=1)
    do_cat = torch.cat([do1, do2], dim=1)
    cu = torch.tensor([0, s1_len, s1_len + s2_len], dtype=torch.long)

    cpu = _run_cpu_bwd(q_cat, k_cat, v_cat, gk=g_cat, cu=cu, do_target=do_cat)

    assert compare_tensor(
        "seg1 dq", gold1["dq"], cpu["dq"][:, :s1_len], atol=1e-4, rtol=1e-4
    )
    assert compare_tensor(
        "seg2 dq", gold2["dq"], cpu["dq"][:, s1_len:], atol=1e-4, rtol=1e-4
    )
    assert compare_tensor(
        "seg1 dk", gold1["dk"], cpu["dk"][:, :s1_len], atol=1e-4, rtol=1e-4
    )
    assert compare_tensor(
        "seg2 dk", gold2["dk"], cpu["dk"][:, s1_len:], atol=1e-4, rtol=1e-4
    )
    assert compare_tensor(
        "seg1 dv", gold1["dv"], cpu["dv"][:, :s1_len], atol=1e-4, rtol=1e-4
    )
    assert compare_tensor(
        "seg2 dv", gold2["dv"], cpu["dv"][:, s1_len:], atol=1e-4, rtol=1e-4
    )


if __name__ == "__main__":
    pytest.main([__file__])
