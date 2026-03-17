"""chunk_gla_bwd: end-to-end JAX orchestrator vs Torch CPU orchestrator tests."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp

from tests.src.ops.gla.chunk import chunk_gla_bwd as cpu_chunk_gla_bwd
from src.ops.gla.chunk import chunk_gla_bwd as jax_chunk_gla_bwd
from tests.utils import compare_tensor


CASES = [
    dict(B=2, T=32, H=4, K=32, V=64, seed=42),
    dict(B=2, T=32, H=4, K=32, V=64, seed=13, h0=True),
    dict(B=2, T=32, H=4, K=32, V=64, seed=14, dht=True),
    dict(B=2, T=32, H=4, K=32, V=64, seed=15, h0=True, dht=True),
    dict(B=1, T=128, H=2, K=64, V=128, seed=7),
    dict(B=2, T=32, H=1, K=32, V=64, seed=10),
    dict(B=2, T=32, H=4, K=16, V=128, seed=20),
    dict(B=2, T=32, H=4, K=128, V=16, seed=21),
    # odd T (needs padding)
    dict(B=2, T=37, H=4, K=16, V=32, seed=40),
    dict(B=1, T=50, H=2, K=32, V=64, seed=41),
    dict(B=2, T=64, H=4, K=16, V=32, seed=40),
    dict(B=8, T=32, H=4, K=32, V=64, seed=50),
    dict(B=1, T=64, H=16, K=32, V=64, seed=60),
    dict(B=2, T=32, H=2, K=8, V=16, seed=70),
    dict(B=1, T=16, H=1, K=16, V=16, seed=99),
    dict(B=4, T=64, H=8, K=32, V=64, seed=99),
    dict(B=2, T=32, H=4, K=32, V=64, seed=200, scale=0.1),
    dict(B=1, T=256, H=2, K=32, V=64, seed=300),
    dict(B=1, T=256, H=2, K=32, V=64, seed=303, h0=True, dht=True),
    # ── non-default chunk_size ──
    dict(B=2, T=64, H=4, K=32, V=64, seed=500, chunk_size=32),
    dict(B=1, T=128, H=2, K=32, V=64, seed=501, chunk_size=8, h0=True, dht=True),
    dict(B=2, T=128, H=4, K=32, V=64, seed=502, chunk_size=64),
    dict(B=1, T=64, H=2, K=32, V=64, seed=503, chunk_size=4),
    # ── varlen: equal segments ──
    dict(B=1, T=32, H=4, K=32, V=64, seed=7, cu_seqlens=[0, 16, 32]),
    dict(B=1, T=96, H=4, K=32, V=64, seed=10, cu_seqlens=[0, 32, 64, 96]),
    # ── varlen: unequal (chunk-aligned) ──
    dict(B=1, T=48, H=4, K=32, V=64, seed=11, cu_seqlens=[0, 16, 48]),
    dict(B=1, T=64, H=2, K=16, V=32, seed=12, cu_seqlens=[0, 16, 48, 64]),
    # ── varlen + h0 ──
    dict(B=1, T=32, H=4, K=32, V=64, seed=17, cu_seqlens=[0, 16, 32], h0=True),
    dict(B=1, T=48, H=4, K=32, V=64, seed=18, cu_seqlens=[0, 16, 48], h0=True),
    # ── varlen + dht ──
    dict(B=1, T=32, H=4, K=32, V=64, seed=19, cu_seqlens=[0, 16, 32], dht=True),
    # ── varlen + h0 + dht ──
    dict(B=1, T=48, H=4, K=32, V=64, seed=20, cu_seqlens=[0, 16, 48], h0=True, dht=True),
    # ── varlen: minimum-size segments ──
    dict(B=1, T=64, H=2, K=16, V=32, seed=22, cu_seqlens=[0, 16, 32, 48, 64]),
    # ── varlen: long + short ──
    dict(B=1, T=80, H=2, K=32, V=64, seed=25, cu_seqlens=[0, 16, 80]),
    dict(B=1, T=64, H=2, K=32, V=64, seed=26, cu_seqlens=[0, 48, 64]),
    # ── varlen: many segments ──
    dict(B=1, T=96, H=2, K=16, V=32, seed=30, cu_seqlens=[0, 16, 32, 48, 64, 80, 96]),
    # ── varlen: single segment ──
    dict(B=1, T=64, H=4, K=32, V=64, seed=35, cu_seqlens=[0, 64]),
    # ── long + varlen ──
    dict(B=1, T=256, H=2, K=32, V=64, seed=340, cu_seqlens=[0, 128, 256]),
    dict(B=1, T=256, H=2, K=32, V=64, seed=342, cu_seqlens=[0, 128, 256], h0=True, dht=True),
    # ── varlen + non-default chunk_size ──
    dict(B=1, T=64, H=4, K=32, V=64, seed=510, chunk_size=32, cu_seqlens=[0, 32, 64]),
    dict(B=1, T=64, H=2, K=16, V=32, seed=511, chunk_size=8, cu_seqlens=[0, 32, 64]),
    # ── varlen: non-chunk-aligned segments (per-segment padding) ──
    dict(B=1, T=30, H=4, K=16, V=32, seed=600, cu_seqlens=[0, 10, 30]),
    dict(B=1, T=45, H=2, K=32, V=64, seed=601, cu_seqlens=[0, 20, 45]),
    dict(B=1, T=50, H=2, K=16, V=32, seed=602, cu_seqlens=[0, 13, 37, 50]),
    dict(B=1, T=30, H=4, K=16, V=32, seed=603, cu_seqlens=[0, 10, 30], h0=True),
    dict(B=1, T=45, H=2, K=32, V=64, seed=604, cu_seqlens=[0, 20, 45], h0=True, dht=True),
]


def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    cs = c.get("chunk_size", 16)
    if cs != 16:
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


def _torch_to_jax(t: torch.Tensor) -> jax.Array:
    return jnp.array(t.detach().to(torch.float32).numpy())


@pytest.mark.parametrize("cfg", CASES, ids=[_case_id(c) for c in CASES])
def test_chunk_gla_bwd(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    scale = cfg.get("scale", K**-0.5)
    C = cfg.get("chunk_size", 16)

    torch.manual_seed(cfg["seed"])
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    g = F.logsigmoid(torch.randn(B, T, H, K))
    do = torch.randn(B, T, H, V)

    cu_list = cfg.get("cu_seqlens")
    cu = torch.tensor(cu_list, dtype=torch.long) if cu_list else None
    N = len(cu_list) - 1 if cu_list else B

    h0 = torch.randn(N, H, K, V) if cfg.get("h0") else None
    dht = torch.randn(N, H, K, V) if cfg.get("dht") else None

    # Torch CPU (handles padding internally)
    dq_cpu, dk_cpu, dv_cpu, dg_cpu, dh0_cpu = cpu_chunk_gla_bwd(
        q.float(), k.float(), v.float(), g.float(),
        None, scale, h0, None, None, do.float(), dht,
        cu_seqlens=cu, chunk_size=C,
    )

    # JAX (handles padding internally)
    q_j, k_j, v_j = _torch_to_jax(q), _torch_to_jax(k), _torch_to_jax(v)
    g_j = _torch_to_jax(g)
    do_j = _torch_to_jax(do)
    h0_j = _torch_to_jax(h0) if h0 is not None else None
    dht_j = _torch_to_jax(dht) if dht is not None else None
    cu_j = jnp.array(cu_list, dtype=jnp.int32) if cu_list else None

    dq_jax, dk_jax, dv_jax, dg_jax, dh0_jax = jax_chunk_gla_bwd(
        q_j, k_j, v_j, g_j,
        g_cumsum=None, scale=scale,
        initial_state=h0_j, h=None, A=None,
        do=do_j, dht=dht_j,
        cu_seqlens=cu_j, chunk_size=C,
    )

    # fp32 end-to-end tolerance: 2e-5 accounts for accumulated rounding
    # across independently-computed intermediates (h, dh, A)
    assert compare_tensor("dq", dq_cpu, dq_jax, atol=2e-5, rtol=1e-5)
    assert compare_tensor("dk", dk_cpu, dk_jax, atol=2e-5, rtol=1e-5)
    assert compare_tensor("dg", dg_cpu, dg_jax, atol=5e-5, rtol=1e-5)
    assert compare_tensor("dv", dv_cpu, dv_jax, atol=2e-5, rtol=1e-5)
    if dh0_cpu is not None:
        assert compare_tensor("dh0", dh0_cpu, dh0_jax, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
