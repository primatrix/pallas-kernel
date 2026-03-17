"""fused_recurrent_gla: Pallas kernel vs Torch CPU reference tests."""

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

from tests.src.ops.gla import fused_recurrent_gla as cpu_fused_recurrent
from tops.ops.gla import fused_recurrent_gla as pallas_fused_recurrent
from tests.utils import compare_tensor

# ============================================================================
# Filter CASES: exclude reverse=True (not yet supported)
# ============================================================================

PALLAS_CASES = [
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


def _torch_to_jax(t: torch.Tensor) -> jax.Array:
    """Convert a torch tensor to a JAX array."""
    return jnp.array(t.detach().to(torch.float32).numpy())


def _run_cpu(
    q, k, v, *, gk=None, gv=None, h0=None, scale=None, cu_seqlens=None, reverse=False
):
    return cpu_fused_recurrent(
        q,
        k,
        v,
        gk=gk,
        gv=gv,
        initial_state=h0,
        output_final_state=True,
        scale=scale,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )


def _run_pallas(
    q,
    k,
    v,
    *,
    gk=None,
    gv=None,
    h0=None,
    scale=None,
    output_final_state=True,
    cu_seqlens=None,
    reverse=False,
):
    q_j, k_j, v_j = _torch_to_jax(q), _torch_to_jax(k), _torch_to_jax(v)
    gk_j = _torch_to_jax(gk) if gk is not None else None
    gv_j = _torch_to_jax(gv) if gv is not None else None
    h0_j = _torch_to_jax(h0) if h0 is not None else None
    cu_j = (
        jnp.array(cu_seqlens.numpy(), dtype=jnp.int32)
        if cu_seqlens is not None
        else None
    )
    return pallas_fused_recurrent(
        q_j,
        k_j,
        v_j,
        gk=gk_j,
        gv=gv_j,
        initial_state=h0_j,
        output_final_state=output_final_state,
        scale=scale,
        reverse=reverse,
        cu_seqlens=cu_j,
    )


# ============================================================================
# Parametrized test — Torch CPU vs Pallas
# ============================================================================


@pytest.mark.parametrize("cfg", PALLAS_CASES, ids=[_case_id(c) for c in PALLAS_CASES])
def test_cpu_vs_pallas(cfg):
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

    o_cpu, s_cpu = _run_cpu(
        q, k, v, gk=gk, gv=gv, h0=h0, scale=scale, cu_seqlens=cu, reverse=reverse
    )
    o_pallas, s_pallas = _run_pallas(
        q, k, v, gk=gk, gv=gv, h0=h0, scale=scale, cu_seqlens=cu, reverse=reverse
    )

    assert compare_tensor("output", o_cpu, o_pallas, atol=atol, rtol=rtol)
    assert compare_tensor("final_state", s_cpu, s_pallas, atol=atol, rtol=rtol)


# ============================================================================
# Structural tests
# ============================================================================


def test_state_split_pallas():
    """Split sequence in 2 halves: state continuity via Pallas."""
    torch.manual_seed(77)
    B, T, H, K, V = 1, 40, 2, 16, 32
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))
    T1 = T // 2

    _, s_full_cpu = _run_cpu(q, k, v, gk=gk)
    _, s_full_pallas = _run_pallas(q, k, v, gk=gk)

    _, s1_pallas = _run_pallas(q[:, :T1], k[:, :T1], v[:, :T1], gk=gk[:, :T1])
    # Convert JAX state back to torch for second half
    s1_torch = torch.from_numpy(np.array(s1_pallas))
    _, s2_pallas = _run_pallas(
        q[:, T1:], k[:, T1:], v[:, T1:], gk=gk[:, T1:], h0=s1_torch
    )

    assert compare_tensor(
        "full state (cpu vs pallas)", s_full_cpu, s_full_pallas, atol=1e-4, rtol=1e-4
    )
    assert compare_tensor(
        "pallas: full vs split", s_full_pallas, s2_pallas, atol=1e-4, rtol=1e-4
    )


def test_no_final_state_pallas():
    """output_final_state=False returns None for final_state."""
    torch.manual_seed(210)
    B, T, H, K, V = 2, 32, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))

    o_cpu, s_cpu = cpu_fused_recurrent(q, k, v, gk=gk, output_final_state=False)
    o_pallas, s_pallas = _run_pallas(q, k, v, gk=gk, output_final_state=False)

    assert s_cpu is None, f"cpu final_state should be None, got {type(s_cpu)}"
    assert s_pallas is None, f"pallas final_state should be None, got {type(s_pallas)}"
    assert compare_tensor("output", o_cpu, o_pallas, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
