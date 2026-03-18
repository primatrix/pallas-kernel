"""chunk_gla: Pallas kernel vs Torch CPU reference tests."""

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

from tests.src.ops.gla import chunk_gla_fwd as cpu_chunk_gla_fwd
from tops.ops.gla import chunk_gla as pallas_chunk_gla
from tests.utils import compare_tensor

# ============================================================================
# Test configs — reuse from test_torch_chunk_gla.py
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
    # ── odd T (chunk padding boundary) ──
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
    # ── custom scale ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=200, scale=0.1),
    # ── varlen: equal segments ──
    dict(B=1, T=32, H=4, K=32, V=64, seed=7, cu_seqlens=[0, 16, 32]),
    dict(B=1, T=96, H=4, K=32, V=64, seed=10, cu_seqlens=[0, 32, 64, 96]),
    # ── varlen: unequal segments (chunk-aligned) ──
    dict(B=1, T=48, H=4, K=32, V=64, seed=11, cu_seqlens=[0, 16, 48]),
    dict(B=1, T=64, H=2, K=16, V=32, seed=12, cu_seqlens=[0, 16, 48, 64]),
    # ── varlen + h0 ──
    dict(B=1, T=32, H=4, K=32, V=64, seed=17, cu_seqlens=[0, 16, 32], h0=True),
    dict(B=1, T=48, H=4, K=32, V=64, seed=18, cu_seqlens=[0, 16, 48], h0=True),
    # ── varlen: minimum-size segments ──
    dict(B=1, T=64, H=2, K=16, V=32, seed=20, cu_seqlens=[0, 16, 32, 48, 64]),
    # ── varlen: long + short ──
    dict(B=1, T=80, H=2, K=32, V=64, seed=25, cu_seqlens=[0, 16, 80]),
    dict(B=1, T=64, H=2, K=32, V=64, seed=26, cu_seqlens=[0, 48, 64]),
    # ── varlen: many segments ──
    dict(B=1, T=96, H=2, K=16, V=32, seed=30, cu_seqlens=[0, 16, 32, 48, 64, 80, 96]),
    # ── varlen: single segment (degenerates to no varlen) ──
    dict(B=1, T=64, H=4, K=32, V=64, seed=35, cu_seqlens=[0, 64]),
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
    # ── long + scale ──
    dict(B=1, T=256, H=2, K=32, V=64, seed=330, scale=0.1),
    dict(B=1, T=512, H=2, K=32, V=64, seed=331, scale=0.01),
    # ── long + varlen ──
    dict(B=1, T=256, H=2, K=32, V=64, seed=340, cu_seqlens=[0, 128, 256]),
    dict(B=1, T=512, H=2, K=32, V=64, seed=341, cu_seqlens=[0, 128, 320, 512]),
    dict(B=1, T=256, H=2, K=32, V=64, seed=342, cu_seqlens=[0, 128, 256], h0=True),
    # ── long + large dims ──
    dict(B=1, T=256, H=2, K=64, V=128, seed=350),
    dict(B=1, T=256, H=2, K=128, V=64, seed=351),
    dict(B=1, T=512, H=2, K=64, V=128, seed=352, h0=True),
    # ── long + multi-batch ──
    dict(B=4, T=256, H=2, K=32, V=64, seed=360),
    dict(B=2, T=512, H=4, K=32, V=64, seed=361),
    # ── long + many heads ──
    dict(B=1, T=256, H=8, K=32, V=64, seed=370),
    dict(B=1, T=512, H=8, K=32, V=64, seed=371, h0=True),
    # ── long + combo ──
    dict(
        B=1,
        T=256,
        H=4,
        K=32,
        V=64,
        seed=380,
        h0=True,
        scale=0.1,
    ),
    dict(
        B=1,
        T=256,
        H=4,
        K=32,
        V=64,
        seed=381,
        h0=True,
        cu_seqlens=[0, 64, 192, 256],
    ),
    # ── varlen: non-chunk-aligned segments (per-segment padding) ──
    dict(B=1, T=30, H=2, K=32, V=64, seed=400, cu_seqlens=[0, 10, 30]),
    dict(B=1, T=45, H=2, K=32, V=64, seed=401, cu_seqlens=[0, 20, 45]),
    dict(B=1, T=50, H=2, K=16, V=32, seed=402, cu_seqlens=[0, 13, 37, 50]),
    dict(B=1, T=30, H=4, K=16, V=32, seed=403, cu_seqlens=[0, 10, 30], h0=True),
    dict(B=1, T=45, H=2, K=32, V=64, seed=404, cu_seqlens=[0, 20, 45], h0=True),
]


def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    if c.get("h0"):
        parts.append("h0")
    if c.get("cu_seqlens"):
        parts.append(f"segs{len(c['cu_seqlens']) - 1}")
    if c.get("scale") is not None:
        parts.append(f"scale={c['scale']}")
    return "-".join(parts)


# ============================================================================
# Helpers
# ============================================================================


def _torch_to_jax(t: torch.Tensor) -> jax.Array:
    """Convert a torch tensor to a JAX array."""
    return jnp.array(t.detach().to(torch.float32).numpy())


def _run_cpu(q, k, v, *, g, h0=None, cu=None, scale=None):
    dtype = q.dtype
    q, k, v, g = (x.float() for x in (q, k, v, g))
    K = q.shape[-1]
    if scale is None:
        scale = K**-0.5
    _, _, _, ht, o = cpu_chunk_gla_fwd(
        q, k, v, g,
        g_cumsum=None,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu,
    )
    return o.to(dtype), ht


def _run_pallas(q, k, v, *, g, h0=None, cu=None, scale=None, output_final_state=True):
    q_j, k_j, v_j = _torch_to_jax(q), _torch_to_jax(k), _torch_to_jax(v)
    g_j = _torch_to_jax(g)
    h0_j = _torch_to_jax(h0) if h0 is not None else None
    cu_j = (
        jnp.array(cu.numpy(), dtype=jnp.int32)
        if cu is not None
        else None
    )
    return pallas_chunk_gla(
        q_j,
        k_j,
        v_j,
        g=g_j,
        initial_state=h0_j,
        output_final_state=output_final_state,
        scale=scale,
        cu_seqlens=cu_j,
    )


# ============================================================================
# Parametrized test — Torch CPU vs Pallas
# ============================================================================


@pytest.mark.parametrize("cfg", CASES, ids=[_case_id(c) for c in CASES])
def test_cpu_vs_pallas(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    atol = cfg.get("atol", 1e-5)
    rtol = cfg.get("rtol", 1e-5)
    scale = cfg.get("scale", None)

    torch.manual_seed(cfg["seed"])
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    g = F.logsigmoid(torch.randn(B, T, H, K))

    cu_list = cfg.get("cu_seqlens")
    cu = torch.tensor(cu_list, dtype=torch.long) if cu_list else None
    N = len(cu_list) - 1 if cu_list else B
    h0 = torch.randn(N, H, K, V) if cfg.get("h0") else None

    o_cpu, s_cpu = _run_cpu(q, k, v, g=g, h0=h0, cu=cu, scale=scale)
    o_pallas, s_pallas = _run_pallas(q, k, v, g=g, h0=h0, cu=cu, scale=scale)

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
    g = F.logsigmoid(torch.randn(B, T, H, K))
    T1 = T // 2

    _, s_full_cpu = _run_cpu(q, k, v, g=g)
    _, s_full_pallas = _run_pallas(q, k, v, g=g)

    _, s1_pallas = _run_pallas(q[:, :T1], k[:, :T1], v[:, :T1], g=g[:, :T1])
    # Convert JAX state back to torch for second half
    s1_torch = torch.from_numpy(np.array(s1_pallas))
    _, s2_pallas = _run_pallas(
        q[:, T1:], k[:, T1:], v[:, T1:], g=g[:, T1:], h0=s1_torch
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
    g = F.logsigmoid(torch.randn(B, T, H, K))

    q_f, k_f, v_f, g_f = (x.float() for x in (q, k, v, g))
    K = q.shape[-1]
    _, _, _, ht_cpu, o_cpu = cpu_chunk_gla_fwd(
        q_f, k_f, v_f, g_f,
        g_cumsum=None,
        scale=K**-0.5,
        initial_state=None,
        output_final_state=False,
    )
    o_pallas, s_pallas = _run_pallas(q, k, v, g=g, output_final_state=False)

    assert ht_cpu is None, f"cpu final_state should be None, got {type(ht_cpu)}"
    assert s_pallas is None, f"pallas final_state should be None, got {type(s_pallas)}"
    assert compare_tensor("output", o_cpu, o_pallas, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
