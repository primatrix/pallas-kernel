"""chunk_bwd_fused: Pallas kernel vs Jax TPU reference tests."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import jax
import jax.numpy as jnp

from src.ops.gla.chunk import chunk_gla_bwd_fused_pl
from src.ops.gla.chunk import (
    chunk_gla_bwd_dA_ref,
    chunk_gla_bwd_dv_ref,
    chunk_gla_bwd_dqk_intra_ref,
    chunk_gla_bwd_dqkg_ref,
)
from tests.utils import compare_tensor

PALLAS_CASES = [
    dict(
        B=1,
        T=128,
        H=2,
        K=128,
        chunk_size=64,
        seed=11,
    ),
    dict(
        B=2,
        T=256,
        H=4,
        K=64,
        chunk_size=64,
        seed=11,
    ),
]

def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}"]
    if c.get("chunk_size") is not None:
        parts.append(f"chunk{c['chunk_size']}")
    return "-".join(parts)


def _run_tpu_ref(q, k, v, g_cumsum, h, do, dh, scale, chunk_size):
    # 5. dA
    dA = chunk_gla_bwd_dA_ref(v, do, scale, chunk_size=chunk_size)
    # 4. dv
    dv = chunk_gla_bwd_dv_ref(k, g_cumsum, dA, do, dh, chunk_size=chunk_size)
    # 6. Intra-chunk dq, dk
    dq_intra, dk_intra = chunk_gla_bwd_dqk_intra_ref(q, k, g_cumsum, dA, chunk_size=chunk_size)
    # 7. Inter-chunk dq, dk + gate gradient
    dq, dk, dg = chunk_gla_bwd_dqkg_ref(
        q, k, v, h, g_cumsum, do, dh, dq_intra, dk_intra, scale, chunk_size=chunk_size
    )
    return dq, dk, dv, dg


def _run_pallas(q, k, v, g_cumsum, h, do, dh, scale, chunk_size):
    dq, dk, dv, dg = chunk_gla_bwd_fused_pl(
        q, k, v, g_cumsum, h, do, dh, scale=scale, chunk_size=chunk_size
    )
    return dq, dk, dv, dg


@pytest.mark.parametrize("cfg", PALLAS_CASES, ids=[_case_id(c) for c in PALLAS_CASES])
def test_native_tpu_vs_pallas(cfg):
    B, T, H, K = cfg["B"], cfg["T"], cfg["H"], cfg["K"]
    V = K # In these tests V == K
    atol = cfg.get("atol", 1e-8)
    rtol = cfg.get("rtol", 1e-8)
    chunk_size = cfg.get("chunk_size", 64)
    NT = T // chunk_size
    
    key = jax.random.PRNGKey(cfg.get("seed", 11))
    
    q = jax.random.normal(key, (B, T, H, K))
    k = jax.random.normal(key, (B, T, H, K))
    v = jax.random.normal(key, (B, T, H, V))
    do = jax.random.normal(key, (B, T, H, V))
    g_cumsum = jax.random.normal(key, (B, T, H, K))
    h = jax.random.normal(key, (B, NT, H, K, V))
    dh = jax.random.normal(key, (B, NT, H, K, V))
    scale = K ** -0.5

    ref_dq, ref_dk, ref_dv, ref_dg = _run_tpu_ref(
        q, k, v, g_cumsum, h, do, dh, scale, chunk_size
    )

    pallas_dq, pallas_dk, pallas_dv, pallas_dg = _run_pallas(
        q, k, v, g_cumsum, h, do, dh, scale, chunk_size
    )

    assert compare_tensor("dq", ref_dq, pallas_dq, atol=atol, rtol=rtol)
    assert compare_tensor("dk", ref_dk, pallas_dk, atol=atol, rtol=rtol)
    assert compare_tensor("dv", ref_dv, pallas_dv, atol=atol, rtol=rtol)
    assert compare_tensor("dg", ref_dg, pallas_dg, atol=atol, rtol=rtol)

if __name__ == "__main__":
    pytest.main([__file__])