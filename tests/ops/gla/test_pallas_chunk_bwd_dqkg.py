"""chunk_gla_bwd_dqkg: JAX ref vs Torch CPU reference tests."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp

from tests.src.ops.gla.chunk import (
    chunk_gla_bwd_dqkg as cpu_chunk_gla_bwd_dqkg,
    chunk_gla_bwd_dqk_intra as cpu_chunk_gla_bwd_dqk_intra,
    chunk_gla_bwd_dA as cpu_chunk_gla_bwd_dA,
    chunk_local_cumsum as cpu_chunk_local_cumsum,
    chunk_fwd_h as cpu_chunk_fwd_h,
    chunk_bwd_dh as cpu_chunk_bwd_dh,
)
from src.ops.gla.chunk import chunk_gla_bwd_dqkg_ref as jax_chunk_gla_bwd_dqkg
from tests.utils import compare_tensor


CASES = [
    dict(B=2, T=32, H=4, K=32, V=64, seed=42),
    dict(B=2, T=32, H=4, K=32, V=64, seed=13, h0=True),
    dict(B=2, T=32, H=4, K=32, V=64, seed=14, dht=True),
    dict(B=2, T=32, H=4, K=32, V=64, seed=15, h0=True, dht=True),
    dict(B=1, T=128, H=2, K=64, V=128, seed=7),
    dict(B=2, T=32, H=1, K=32, V=64, seed=10),
    dict(B=2, T=32, H=4, K=16, V=128, seed=20),
    dict(B=2, T=64, H=4, K=16, V=32, seed=40),
    dict(B=8, T=32, H=4, K=32, V=64, seed=50),
    dict(B=1, T=16, H=1, K=16, V=16, seed=99),
    dict(B=4, T=64, H=8, K=32, V=64, seed=99),
    dict(B=2, T=32, H=4, K=32, V=64, seed=200, scale=0.1),
    dict(B=1, T=256, H=2, K=32, V=64, seed=300),
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
def test_chunk_gla_bwd_dqkg(cfg):
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

    # Compute intermediates on torch side (shared truth)
    gk_cpu = cpu_chunk_local_cumsum(g, C)
    h_cpu, _ = cpu_chunk_fwd_h(k, v, gk_cpu, h0=h0, chunk_size=C)
    dh_cpu, _ = cpu_chunk_bwd_dh(q, k, v, gk_cpu, do, h0=h0, dht=dht, scale=scale, chunk_size=C)
    dA_cpu = cpu_chunk_gla_bwd_dA(v, do, scale, chunk_size=C)
    dq_intra_cpu, dk_intra_cpu = cpu_chunk_gla_bwd_dqk_intra(q, k, gk_cpu, dA_cpu, chunk_size=C)

    # Torch CPU dqkg
    dq_cpu, dk_cpu, dg_cpu = cpu_chunk_gla_bwd_dqkg(
        q, k, v, h_cpu, gk_cpu, do, dh_cpu, dq_intra_cpu, dk_intra_cpu, scale, chunk_size=C,
    )

    # JAX dqkg using same intermediates
    q_j, k_j, v_j = _torch_to_jax(q), _torch_to_jax(k), _torch_to_jax(v)
    gk_j = _torch_to_jax(gk_cpu)
    do_j = _torch_to_jax(do)
    h_j = _torch_to_jax(h_cpu)
    dh_j = _torch_to_jax(dh_cpu)
    dq_intra_j = _torch_to_jax(dq_intra_cpu)
    dk_intra_j = _torch_to_jax(dk_intra_cpu)

    dq_jax, dk_jax, dg_jax = jax_chunk_gla_bwd_dqkg(
        q_j, k_j, v_j, h_j, gk_j, do_j, dh_j, dq_intra_j, dk_intra_j, scale, chunk_size=C,
    )

    assert compare_tensor("dq", dq_cpu, dq_jax, atol=1e-5, rtol=1e-5)
    assert compare_tensor("dk", dk_cpu, dk_jax, atol=1e-5, rtol=1e-5)
    assert compare_tensor("dg", dg_cpu, dg_jax, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
