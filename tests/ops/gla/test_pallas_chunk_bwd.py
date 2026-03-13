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
]


def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    if c.get("h0"):
        parts.append("h0")
    if c.get("dht"):
        parts.append("dht")
    if c.get("scale") is not None:
        parts.append(f"scale={c['scale']}")
    return "-".join(parts)


def _torch_to_jax(t: torch.Tensor) -> jax.Array:
    return jnp.array(t.detach().to(torch.float32).numpy())


@pytest.mark.parametrize("cfg", CASES, ids=[_case_id(c) for c in CASES])
def test_chunk_gla_bwd(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    scale = cfg.get("scale", K**-0.5)
    C = 16

    torch.manual_seed(cfg["seed"])
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    g = F.logsigmoid(torch.randn(B, T, H, K))
    do = torch.randn(B, T, H, V)
    h0 = torch.randn(B, H, K, V) if cfg.get("h0") else None
    dht = torch.randn(B, H, K, V) if cfg.get("dht") else None

    # Torch CPU (handles padding internally)
    dq_cpu, dk_cpu, dv_cpu, dg_cpu, dh0_cpu = cpu_chunk_gla_bwd(
        q.float(), k.float(), v.float(), g.float(), scale,
        initial_state=h0, do=do.float(), dht=dht, chunk_size=C,
    )

    # JAX (caller must pad T to multiple of chunk_size)
    NT = (T + C - 1) // C
    T_padded = NT * C
    if T_padded > T:
        pad = T_padded - T
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        g = F.pad(g, (0, 0, 0, 0, 0, pad))
        do = F.pad(do, (0, 0, 0, 0, 0, pad))

    q_j, k_j, v_j = _torch_to_jax(q), _torch_to_jax(k), _torch_to_jax(v)
    g_j = _torch_to_jax(g)
    do_j = _torch_to_jax(do)
    h0_j = _torch_to_jax(h0) if h0 is not None else None
    dht_j = _torch_to_jax(dht) if dht is not None else None

    dq_jax, dk_jax, dv_jax, dg_jax, dh0_jax = jax_chunk_gla_bwd(
        q_j, k_j, v_j, g_j,
        g_cumsum=None, scale=scale,
        initial_state=h0_j, h=None, A=None,
        do=do_j, dht=dht_j, chunk_size=C,
    )

    # Slice back to original T for comparison
    dq_jax = dq_jax[:, :T]
    dk_jax = dk_jax[:, :T]
    dv_jax = dv_jax[:, :T]
    dg_jax = dg_jax[:, :T]

    # Tight tolerance for dq, dk (same computation graph, minimal drift)
    assert compare_tensor("dq", dq_cpu, dq_jax, atol=1e-5, rtol=1e-5)
    assert compare_tensor("dk", dk_cpu, dk_jax, atol=1e-5, rtol=1e-5)
    # Slightly relaxed for dg (reverse cumsum accumulates small float32 differences)
    assert compare_tensor("dg", dg_cpu, dg_jax, atol=5e-5, rtol=5e-5)
    # Relaxed tolerance for dv (accumulates precision differences from h, dh, A intermediates)
    assert compare_tensor("dv", dv_cpu, dv_jax, atol=5e-2, rtol=5e-2)
    if dh0_cpu is not None:
        assert compare_tensor("dh0", dh0_cpu, dh0_jax, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
