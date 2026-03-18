"""fused_recurrent_gla backward: Pallas kernel vs Torch CPU reference tests."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp

from tests.src.ops.gla import fused_recurrent_gla_bwd as cpu_bwd
from tops.ops.gla import fused_recurrent_gla_bwd as pallas_bwd
from tops.ops.gla import fused_recurrent_gla as pallas_fwd
from tests.utils import compare_tensor

# ============================================================================
# Test cases
# ============================================================================

PALLAS_BWD_CASES = [
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
    # ── dht without h0 ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=400, dht=True),
    dict(B=2, T=32, H=4, K=32, V=64, seed=401, dht=True, gate="gv"),
    dict(B=2, T=32, H=4, K=32, V=64, seed=402, dht=True, gate="gk+gv"),
    dict(B=2, T=32, H=4, K=32, V=64, seed=403, dht=True, gate="none", atol=5e-6),
    dict(B=2, T=32, H=4, K=32, V=64, seed=404, dht=True, reverse=True),
    dict(
        B=1, T=32, H=4, K=32, V=64, seed=405,
        dht=True, cu_seqlens=[0, 16, 32],
    ),
    # ── h0 without dht ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=410, h0=True, dht=False),
    dict(B=2, T=32, H=4, K=32, V=64, seed=411, h0=True, dht=False, gate="gk+gv"),
    # ── gv + dht + varlen (dgv varlen correction) ──
    dict(
        B=1, T=32, H=4, K=32, V=64, seed=420,
        gate="gv", h0=True, cu_seqlens=[0, 16, 32],
    ),
    dict(
        B=1, T=48, H=2, K=16, V=32, seed=421,
        gate="gv", dht=True, cu_seqlens=[0, 10, 24, 48],
    ),
    # ── gk+gv + dht + varlen (both dgk & dgv varlen correction) ──
    dict(
        B=1, T=32, H=4, K=32, V=64, seed=430,
        gate="gk+gv", dht=True, cu_seqlens=[0, 16, 32],
    ),
    dict(
        B=1, T=48, H=2, K=16, V=32, seed=431,
        gate="gk+gv", h0=True, cu_seqlens=[0, 10, 24, 48],
    ),
]


def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    gate = c.get("gate", "gk")
    if gate != "gk":
        parts.append(f"gate={gate}")
    if c.get("h0"):
        parts.append("h0")
    if "dht" in c:
        # Explicit dht control (independent of h0)
        if c["dht"] and not c.get("h0"):
            parts.append("dht")
        elif not c["dht"] and c.get("h0"):
            parts.append("no-dht")
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


def _torch_to_jax(t: torch.Tensor) -> jax.Array:
    """Convert a torch tensor to a JAX array."""
    return jnp.array(t.detach().to(torch.float32).numpy())


def _run_cpu_bwd(
    q, k, v, *, gk=None, gv=None, h0=None, cu=None,
    scale=None, reverse=False, do_target=None, dht_target=None,
):
    dq, dk, dv, dgk, dgv, dh0 = cpu_bwd(
        q=q, k=k, v=v, gk=gk, gv=gv,
        do=do_target, dht=dht_target,
        scale=scale, initial_state=h0,
        reverse=reverse, cu_seqlens=cu,
    )
    return dict(dq=dq, dk=dk, dv=dv, dgk=dgk, dgv=dgv, dh0=dh0)


def _run_pallas_bwd(
    q, k, v, *, gk=None, gv=None, h0=None, cu=None,
    scale=None, reverse=False, do_target=None, dht_target=None,
):
    q_j = _torch_to_jax(q)
    k_j = _torch_to_jax(k)
    v_j = _torch_to_jax(v)
    gk_j = _torch_to_jax(gk) if gk is not None else None
    gv_j = _torch_to_jax(gv) if gv is not None else None
    h0_j = _torch_to_jax(h0) if h0 is not None else None
    do_j = _torch_to_jax(do_target)
    dht_j = _torch_to_jax(dht_target) if dht_target is not None else None
    cu_j = (
        jnp.array(cu.numpy(), dtype=jnp.int32)
        if cu is not None
        else None
    )

    # Compute forward output for dgv (only needed when gv is used)
    o_j = None
    if gv is not None:
        o_j, _ = pallas_fwd(
            q_j, k_j, v_j, gk=gk_j, gv=gv_j,
            scale=scale, initial_state=h0_j,
            output_final_state=False, reverse=reverse,
            cu_seqlens=cu_j,
        )

    dq, dk, dv, dgk, dgv, dh0 = pallas_bwd(
        q_j, k_j, v_j, gk=gk_j, gv=gv_j,
        o=o_j, do=do_j, dht=dht_j,
        scale=scale, initial_state=h0_j,
        reverse=reverse, cu_seqlens=cu_j,
    )
    return dict(dq=dq, dk=dk, dv=dv, dgk=dgk, dgv=dgv, dh0=dh0)


# ============================================================================
# Parametrized test — Torch CPU vs Pallas
# ============================================================================


@pytest.mark.parametrize(
    "cfg", PALLAS_BWD_CASES, ids=[_case_id(c) for c in PALLAS_BWD_CASES]
)
def test_bwd_cpu_vs_pallas(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    atol = cfg.get("atol", 1e-4)
    rtol = cfg.get("rtol", 1e-4)
    gate = cfg.get("gate", "gk")
    scale = cfg.get("scale", None)
    reverse = cfg.get("reverse", False)

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

    # dht defaults to same as h0 for backward compat; can be overridden explicitly
    use_dht = cfg.get("dht", cfg.get("h0", False))
    do_target = torch.randn(B, T, H, V)
    dht_target = torch.randn(N, H, K, V) if use_dht else None

    cpu = _run_cpu_bwd(
        q, k, v, gk=gk, gv=gv, h0=h0, cu=cu, scale=scale,
        reverse=reverse, do_target=do_target, dht_target=dht_target,
    )
    pallas = _run_pallas_bwd(
        q, k, v, gk=gk, gv=gv, h0=h0, cu=cu, scale=scale,
        reverse=reverse, do_target=do_target, dht_target=dht_target,
    )

    assert compare_tensor("dq", cpu["dq"], pallas["dq"], atol=atol, rtol=rtol)
    assert compare_tensor("dk", cpu["dk"], pallas["dk"], atol=atol, rtol=rtol)
    assert compare_tensor("dv", cpu["dv"], pallas["dv"], atol=atol, rtol=rtol)
    if gk is not None:
        assert compare_tensor("dgk", cpu["dgk"], pallas["dgk"], atol=atol, rtol=rtol)
    if gv is not None:
        assert compare_tensor("dgv", cpu["dgv"], pallas["dgv"], atol=atol, rtol=rtol)
    if h0 is not None:
        assert compare_tensor("dh0", cpu["dh0"], pallas["dh0"], atol=atol, rtol=rtol)


# ============================================================================
# Structural tests
# ============================================================================


def test_bwd_no_gate_pallas():
    """Backward with no gates at all."""
    torch.manual_seed(500)
    B, T, H, K, V = 2, 32, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    do_target = torch.randn(B, T, H, V)

    cpu = _run_cpu_bwd(q, k, v, do_target=do_target)
    pallas = _run_pallas_bwd(q, k, v, do_target=do_target)

    assert compare_tensor("dq", cpu["dq"], pallas["dq"], atol=1e-4, rtol=1e-4)
    assert compare_tensor("dk", cpu["dk"], pallas["dk"], atol=1e-4, rtol=1e-4)
    assert compare_tensor("dv", cpu["dv"], pallas["dv"], atol=1e-4, rtol=1e-4)


def test_bwd_with_dht_pallas():
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

    cpu = _run_cpu_bwd(
        q, k, v, gk=gk, h0=h0,
        do_target=do_target, dht_target=dht_target,
    )
    pallas = _run_pallas_bwd(
        q, k, v, gk=gk, h0=h0,
        do_target=do_target, dht_target=dht_target,
    )

    assert compare_tensor("dq", cpu["dq"], pallas["dq"], atol=1e-4, rtol=1e-4)
    assert compare_tensor("dk", cpu["dk"], pallas["dk"], atol=1e-4, rtol=1e-4)
    assert compare_tensor("dv", cpu["dv"], pallas["dv"], atol=1e-4, rtol=1e-4)
    assert compare_tensor("dgk", cpu["dgk"], pallas["dgk"], atol=1e-4, rtol=1e-4)
    assert compare_tensor("dh0", cpu["dh0"], pallas["dh0"], atol=1e-4, rtol=1e-4)


def test_bwd_varlen_packed_vs_separate():
    """Backward: packed varlen == separate batch (via Pallas)."""
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

    # Separate backward via Pallas
    sep1 = _run_pallas_bwd(q1, k1, v1, gk=g1, do_target=do1)
    sep2 = _run_pallas_bwd(q2, k2, v2, gk=g2, do_target=do2)

    # Packed backward via Pallas
    q_cat = torch.cat([q1, q2], dim=1)
    k_cat = torch.cat([k1, k2], dim=1)
    v_cat = torch.cat([v1, v2], dim=1)
    g_cat = torch.cat([g1, g2], dim=1)
    do_cat = torch.cat([do1, do2], dim=1)
    cu = torch.tensor([0, s1_len, s1_len + s2_len], dtype=torch.long)

    packed = _run_pallas_bwd(q_cat, k_cat, v_cat, gk=g_cat, cu=cu, do_target=do_cat)

    assert compare_tensor(
        "seg1 dq", sep1["dq"], packed["dq"][:, :s1_len], atol=1e-4, rtol=1e-4
    )
    assert compare_tensor(
        "seg2 dq", sep2["dq"], packed["dq"][:, s1_len:], atol=1e-4, rtol=1e-4
    )
    assert compare_tensor(
        "seg1 dk", sep1["dk"], packed["dk"][:, :s1_len], atol=1e-4, rtol=1e-4
    )
    assert compare_tensor(
        "seg2 dk", sep2["dk"], packed["dk"][:, s1_len:], atol=1e-4, rtol=1e-4
    )
    assert compare_tensor(
        "seg1 dv", sep1["dv"], packed["dv"][:, :s1_len], atol=1e-4, rtol=1e-4
    )
    assert compare_tensor(
        "seg2 dv", sep2["dv"], packed["dv"][:, s1_len:], atol=1e-4, rtol=1e-4
    )


if __name__ == "__main__":
    pytest.main([__file__])
