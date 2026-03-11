"""chunk_gla_fwd_intra_gk: Pallas TPU kernel vs Torch CPU reference tests.

Tests that the Pallas kernel produces the same output as the
Torch CPU reference for both non-varlen and varlen cases.

Both compute the full (non-causal) intra-chunk attention matrix:

    A[i, j] = scale * sum_k (q[i] * exp(g[i])) * (k[j] * exp(-g[j]))

CPU returns [B, NT, C_i, H, C_j] -> reshaped to [B, T, H, C].
Pallas returns [B, T, H, BT].
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp

from tests.src.ops.gla.chunk import chunk_gla_fwd_intra_gk as cpu_chunk_gla_fwd_intra_gk
from src.ops.gla.chunk import chunk_gla_fwd_intra_gk_pl as pallas_chunk_gla_fwd_intra_gk
from tests.utils import compare_tensor

# ============================================================================
# Test configs
# ============================================================================

CASES = [
    # -- standard shapes --
    dict(B=2, T=64, H=4, K=32, seed=42),
    dict(B=1, T=128, H=2, K=64, seed=7),
    # -- single head --
    dict(B=2, T=64, H=1, K=32, seed=10),
    # -- K variations --
    dict(B=2, T=64, H=4, K=16, seed=20),
    dict(B=2, T=64, H=4, K=128, seed=21),
    # -- small T (= chunk_size) --
    dict(B=1, T=64, H=2, K=32, seed=30),
    # -- T = 2 * chunk_size --
    dict(B=2, T=128, H=4, K=16, seed=40),
    # -- large batch --
    dict(B=8, T=64, H=4, K=32, seed=50),
    # -- many heads --
    dict(B=1, T=64, H=16, K=32, seed=60),
    # -- small dims --
    dict(B=2, T=64, H=2, K=8, seed=70),
    # -- various --
    dict(B=1, T=64, H=1, K=16, seed=99),
    dict(B=4, T=64, H=8, K=32, seed=99),
    dict(B=2, T=128, H=4, K=32, seed=99),
    # -- custom scale --
    dict(B=2, T=64, H=4, K=32, seed=200, scale=0.1),
    # -- smaller chunk_size --
    dict(B=2, T=64, H=4, K=32, seed=210, chunk_size=16),
    dict(B=2, T=128, H=4, K=32, seed=211, chunk_size=32),
    dict(B=1, T=128, H=2, K=64, seed=212, chunk_size=16),
    # -- long sequence --
    dict(B=1, T=256, H=2, K=32, seed=300),
    dict(B=1, T=512, H=2, K=32, seed=301),
    dict(B=1, T=1024, H=2, K=32, seed=302),
    # -- long + large K --
    dict(B=1, T=256, H=2, K=64, seed=350),
    dict(B=1, T=256, H=2, K=128, seed=351),
    # -- long + multi-batch --
    dict(B=4, T=256, H=2, K=32, seed=360),
    dict(B=2, T=512, H=4, K=32, seed=361),
    # -- long + many heads --
    dict(B=1, T=256, H=8, K=32, seed=370),
    # -- long + combo --
    dict(B=1, T=256, H=4, K=32, seed=380, scale=0.1),
    dict(B=2, T=256, H=4, K=32, seed=381, chunk_size=32),
]


def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}"]
    cs = c.get("chunk_size", 64)
    if cs != 64:
        parts.append(f"C{cs}")
    if c.get("scale") is not None:
        parts.append(f"scale={c['scale']}")
    return "-".join(parts)


# ============================================================================
# Helpers
# ============================================================================


def _torch_to_jax(t: torch.Tensor) -> jax.Array:
    return jnp.array(t.detach().to(torch.float32).numpy())


def _make_inputs(B, T, H, K):
    """Generate random inputs for chunk_gla_fwd_intra_gk."""
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    g = F.logsigmoid(torch.randn(B, T, H, K))
    return q, k, g


# ============================================================================
# Non-varlen test: Pallas TPU vs Torch CPU
# ============================================================================


@pytest.mark.parametrize("cfg", CASES, ids=[_case_id(c) for c in CASES])
def test_pallas_vs_cpu(cfg):
    B, T, H, K = cfg["B"], cfg["T"], cfg["H"], cfg["K"]
    atol = cfg.get("atol", 1e-5)
    rtol = cfg.get("rtol", 1e-5)
    scale = cfg.get("scale", K**-0.5)
    chunk_size = cfg.get("chunk_size", 64)
    C = chunk_size

    assert T % C == 0, f"T={T} must be a multiple of chunk_size={C}"

    torch.manual_seed(cfg["seed"])
    q, k, g = _make_inputs(B, T, H, K)

    # CPU: returns [B, NT, C_i, H, C_j]
    A_cpu = cpu_chunk_gla_fwd_intra_gk(q, k, g, scale=scale, chunk_size=C)
    # Reshape to flat format [B, T, H, C]
    A_cpu_flat = A_cpu.reshape(B, T, H, C)

    # Pallas TPU: returns [B, T, H, BT]
    A_pallas = pallas_chunk_gla_fwd_intra_gk(
        _torch_to_jax(q),
        _torch_to_jax(k),
        _torch_to_jax(g),
        scale=scale,
        chunk_size=C,
    )

    assert compare_tensor("A_intra", A_cpu_flat, A_pallas, atol=atol, rtol=rtol)


# ============================================================================
# Varlen test configs
# ============================================================================

VARLEN_CASES = [
    # -- basic --
    dict(seqlens=[64, 64], H=4, K=32, seed=1000),
    dict(seqlens=[128, 64], H=4, K=32, seed=1001),
    dict(seqlens=[64, 128, 64], H=2, K=32, seed=1002),
    dict(seqlens=[64, 64, 64, 64], H=4, K=16, seed=1003),
    # -- single sequence --
    dict(seqlens=[128], H=4, K=32, seed=1007),
    # -- K variations --
    dict(seqlens=[64, 128], H=4, K=16, seed=1009),
    dict(seqlens=[64, 128], H=4, K=128, seed=1010),
    # -- custom scale --
    dict(seqlens=[64, 64], H=4, K=32, seed=1011, scale=0.1),
    # -- small chunk_size --
    dict(seqlens=[64, 64], H=4, K=32, seed=1012, chunk_size=16),
    dict(seqlens=[64, 128, 64], H=2, K=32, seed=1013, chunk_size=16),
    # -- long sequences --
    dict(seqlens=[256, 128], H=2, K=32, seed=1014),
    dict(seqlens=[128, 256, 64], H=2, K=32, seed=1015),
    # -- many heads --
    dict(seqlens=[64, 64], H=16, K=32, seed=1016),
    # -- small dims --
    dict(seqlens=[64, 64], H=2, K=8, seed=1017),
    # -- long + scale --
    dict(seqlens=[128, 256], H=4, K=32, seed=1018, scale=0.1),
    # -- many short sequences --
    dict(seqlens=[16, 32, 16, 48, 16], H=4, K=32, seed=1019, chunk_size=16),
]


def _varlen_case_id(c):
    seqlens_str = ",".join(str(s) for s in c["seqlens"])
    parts = [f"seqlens=[{seqlens_str}]_H{c['H']}_K{c['K']}"]
    cs = c.get("chunk_size", 64)
    if cs != 64:
        parts.append(f"C{cs}")
    if c.get("scale") is not None:
        parts.append(f"scale={c['scale']}")
    return "-".join(parts)


def _make_segment_inputs_intra(B, L, H, K, chunk_size, scale):
    """Generate random inputs for one segment, with padding for CPU reference.

    Returns:
        q_raw:     [B, L, H, K]   -- unpadded
        k_raw:     [B, L, H, K]   -- unpadded
        g_raw:     [B, L, H, K]   -- unpadded
        A_ref:     [B, L, H, C]   -- reference output (valid positions only, flat format)
    """
    C = chunk_size
    NT = (L + C - 1) // C
    T_pad = NT * C
    pad = T_pad - L

    q_raw = torch.randn(B, L, H, K)
    k_raw = torch.randn(B, L, H, K)
    g_raw = F.logsigmoid(torch.randn(B, L, H, K))

    # Pad for CPU computation
    q_p = F.pad(q_raw, (0, 0, 0, 0, 0, pad)) if pad > 0 else q_raw
    k_p = F.pad(k_raw, (0, 0, 0, 0, 0, pad)) if pad > 0 else k_raw
    g_p = F.pad(g_raw, (0, 0, 0, 0, 0, pad)) if pad > 0 else g_raw

    A_ref = cpu_chunk_gla_fwd_intra_gk(q_p, k_p, g_p, scale=scale, chunk_size=C)
    # [B, NT, C_i, H, C_j] -> [B, T_pad, H, C] -> take first L rows
    A_ref_flat = A_ref.reshape(B, T_pad, H, C)[:, :L]

    return q_raw, k_raw, g_raw, A_ref_flat


# ============================================================================
# Varlen test: Pallas varlen path vs per-segment CPU computation
# ============================================================================


@pytest.mark.parametrize(
    "cfg", VARLEN_CASES, ids=[_varlen_case_id(c) for c in VARLEN_CASES]
)
def test_pallas_varlen_vs_segments(cfg):
    """Test that Pallas varlen path matches per-segment CPU computation."""
    seqlens = cfg["seqlens"]
    H, K = cfg["H"], cfg["K"]
    C = cfg.get("chunk_size", 64)
    scale = cfg.get("scale", K**-0.5)
    atol = cfg.get("atol", 1e-5)
    rtol = cfg.get("rtol", 1e-5)
    B = 1

    # Only test chunk-aligned seqlens (unified kernel requires alignment)
    for L in seqlens:
        assert L % C == 0, f"seqlen {L} not aligned to chunk_size {C}"

    torch.manual_seed(cfg["seed"])

    qs, ks, gs, A_refs = [], [], [], []

    for L in seqlens:
        q_raw, k_raw, g_raw, A_ref = _make_segment_inputs_intra(B, L, H, K, C, scale)
        qs.append(q_raw)
        ks.append(k_raw)
        gs.append(g_raw)
        A_refs.append(A_ref)

    # Pack into varlen format
    q_packed = torch.cat(qs, dim=1)
    k_packed = torch.cat(ks, dim=1)
    g_packed = torch.cat(gs, dim=1)
    A_ref_packed = torch.cat(A_refs, dim=1)

    # Run Pallas on packed inputs
    A_pallas = pallas_chunk_gla_fwd_intra_gk(
        _torch_to_jax(q_packed),
        _torch_to_jax(k_packed),
        _torch_to_jax(g_packed),
        scale=scale,
        chunk_size=C,
    )

    assert compare_tensor(
        "A_intra_varlen", A_ref_packed, A_pallas, atol=atol, rtol=rtol
    )


if __name__ == "__main__":
    pytest.main([__file__])
