"""chunk_gla_fwd_o_gk: Pallas TPU kernel vs Torch CPU reference tests.

Tests that the unified Pallas kernel produces the same output as the
Torch CPU reference for both non-varlen and varlen cases.
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

from tests.src.ops.gla.chunk import chunk_gla_fwd_o_gk as cpu_chunk_gla_fwd_o_gk
from src.ops.gla.chunk import chunk_gla_fwd_o_gk as pallas_chunk_gla_fwd_o_gk
from tests.utils import compare_tensor

# ============================================================================
# Test configs
# ============================================================================

CASES = [
    # ── standard shapes ──
    dict(B=2, T=64, H=4, K=32, V=64, seed=42),
    dict(B=1, T=128, H=2, K=64, V=128, seed=7),
    # ── single head ──
    dict(B=2, T=64, H=1, K=32, V=64, seed=10),
    # ── K != V ──
    dict(B=2, T=64, H=4, K=16, V=128, seed=20),
    dict(B=2, T=64, H=4, K=128, V=16, seed=21),
    # ── small T (= chunk_size) ──
    dict(B=1, T=64, H=2, K=32, V=64, seed=30),
    # ── T = 2 * chunk_size ──
    dict(B=2, T=128, H=4, K=16, V=32, seed=40),
    # ── large batch ──
    dict(B=8, T=64, H=4, K=32, V=64, seed=50),
    # ── many heads ──
    dict(B=1, T=64, H=16, K=32, V=64, seed=60),
    # ── small dims ──
    dict(B=2, T=64, H=2, K=8, V=16, seed=70),
    # ── various ──
    dict(B=1, T=64, H=1, K=16, V=16, seed=99),
    dict(B=4, T=64, H=8, K=32, V=64, seed=99),
    dict(B=2, T=128, H=4, K=32, V=32, seed=99),
    # ── custom scale ──
    dict(B=2, T=64, H=4, K=32, V=64, seed=200, scale=0.1),
    # ── smaller chunk_size ──
    dict(B=2, T=64, H=4, K=32, V=64, seed=210, chunk_size=16),
    dict(B=2, T=128, H=4, K=32, V=64, seed=211, chunk_size=32),
    dict(B=1, T=128, H=2, K=64, V=128, seed=212, chunk_size=16),
    # ── long sequence ──
    dict(B=1, T=256, H=2, K=32, V=64, seed=300),
    dict(B=1, T=512, H=2, K=32, V=64, seed=301),
    # ── long + large dims ──
    dict(B=1, T=256, H=2, K=64, V=128, seed=350),
    dict(B=1, T=256, H=2, K=128, V=64, seed=351),
    # ── long + multi-batch ──
    dict(B=4, T=256, H=2, K=32, V=64, seed=360),
    dict(B=2, T=512, H=4, K=32, V=64, seed=361),
    # ── long + many heads ──
    dict(B=1, T=256, H=8, K=32, V=64, seed=370),
    # ── long + combo ──
    dict(B=1, T=256, H=4, K=32, V=64, seed=380, scale=0.1),
    dict(B=2, T=256, H=4, K=32, V=64, seed=381, chunk_size=32),
]


def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
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


def _make_inputs(B, T, H, K, V, chunk_size, scale):
    """Generate random inputs for chunk_gla_fwd_o_gk.

    A is generated in CPU format [B, NT, H, C, C] (causal-masked),
    then converted to Pallas format [B, T, H, C] via permute+reshape.
    """
    C = chunk_size
    NT = T // C

    q = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    g = F.logsigmoid(torch.randn(B, T, H, K))
    h = torch.randn(B, NT, H, K, V)

    A_cpu = torch.randn(B, NT, H, C, C)
    causal_mask = torch.tril(torch.ones(C, C, dtype=torch.bool))
    A_cpu = A_cpu.masked_fill(~causal_mask, 0.0)

    # Pallas format: [B, NT, H, C_i, C_j] -> [B, NT, C_i, H, C_j] -> [B, T, H, C]
    A_pallas = A_cpu.permute(0, 1, 3, 2, 4).reshape(B, T, H, C).contiguous()

    return q, v, g, A_cpu, A_pallas, h


# ============================================================================
# Non-varlen test: Pallas TPU vs Torch CPU
# ============================================================================


@pytest.mark.parametrize("cfg", CASES, ids=[_case_id(c) for c in CASES])
def test_pallas_vs_cpu(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    atol = cfg.get("atol", 1e-2)
    rtol = cfg.get("rtol", 1e-2)
    scale = cfg.get("scale", K**-0.5)
    chunk_size = cfg.get("chunk_size", 64)

    assert T % chunk_size == 0, f"T={T} must be a multiple of chunk_size={chunk_size}"

    torch.manual_seed(cfg["seed"])
    q, v, g, A_cpu, A_pallas, h = _make_inputs(B, T, H, K, V, chunk_size, scale)

    # CPU reference
    o_cpu = cpu_chunk_gla_fwd_o_gk(
        q, v, g, A_cpu, h, scale=scale, chunk_size=chunk_size
    )

    # Pallas TPU
    o_pallas = pallas_chunk_gla_fwd_o_gk(
        _torch_to_jax(q),
        _torch_to_jax(v),
        _torch_to_jax(g),
        _torch_to_jax(A_pallas),
        _torch_to_jax(h),
        scale=scale,
        chunk_size=chunk_size,
    )

    assert compare_tensor("output", o_cpu, o_pallas, atol=atol, rtol=rtol)


# ============================================================================
# Varlen test configs
# ============================================================================

VARLEN_CASES = [
    # ── basic ──
    dict(seqlens=[64, 64], H=4, K=32, V=64, seed=1000),
    dict(seqlens=[128, 64], H=4, K=32, V=64, seed=1001),
    dict(seqlens=[64, 128, 64], H=2, K=32, V=64, seed=1002),
    dict(seqlens=[64, 64, 64, 64], H=4, K=16, V=32, seed=1003),
    # ── single sequence ──
    dict(seqlens=[128], H=4, K=32, V=64, seed=1007),
    # ── K != V ──
    dict(seqlens=[64, 128], H=4, K=16, V=128, seed=1009),
    dict(seqlens=[64, 128], H=4, K=128, V=16, seed=1010),
    # ── custom scale ──
    dict(seqlens=[64, 64], H=4, K=32, V=64, seed=1011, scale=0.1),
    # ── small chunk_size ──
    dict(seqlens=[64, 64], H=4, K=32, V=64, seed=1012, chunk_size=16),
    dict(seqlens=[64, 128, 64], H=2, K=32, V=64, seed=1013, chunk_size=16),
    # ── long sequences ──
    dict(seqlens=[256, 128], H=2, K=32, V=64, seed=1014),
    dict(seqlens=[128, 256, 64], H=2, K=32, V=64, seed=1015),
    # ── many heads ──
    dict(seqlens=[64, 64], H=16, K=32, V=64, seed=1016),
    # ── small dims ──
    dict(seqlens=[64, 64], H=2, K=8, V=16, seed=1017),
    # ── long + scale ──
    dict(seqlens=[128, 256], H=4, K=32, V=64, seed=1018, scale=0.1),
    # ── many short sequences ──
    dict(seqlens=[16, 32, 16, 48, 16], H=4, K=32, V=64, seed=1019, chunk_size=16),
]


def _varlen_case_id(c):
    seqlens_str = ",".join(str(s) for s in c["seqlens"])
    parts = [f"seqlens=[{seqlens_str}]_H{c['H']}_K{c['K']}_V{c['V']}"]
    cs = c.get("chunk_size", 64)
    if cs != 64:
        parts.append(f"C{cs}")
    if c.get("scale") is not None:
        parts.append(f"scale={c['scale']}")
    return "-".join(parts)


def _make_segment_inputs(B, L, H, K, V, chunk_size, scale):
    """Generate random inputs for one segment, with padding to chunk_size."""
    C = chunk_size
    NT = (L + C - 1) // C
    T_pad = NT * C
    pad = T_pad - L

    q_raw = torch.randn(B, L, H, K)
    v_raw = torch.randn(B, L, H, V)
    g_raw = F.logsigmoid(torch.randn(B, L, H, K))

    A_seg = torch.randn(B, NT, H, C, C)
    causal_mask = torch.tril(torch.ones(C, C, dtype=torch.bool))
    A_seg = A_seg.masked_fill(~causal_mask, 0.0)
    h_seg = torch.randn(B, NT, H, K, V)

    # Pad and compute reference
    q_p = F.pad(q_raw, (0, 0, 0, 0, 0, pad)) if pad > 0 else q_raw
    v_p = F.pad(v_raw, (0, 0, 0, 0, 0, pad)) if pad > 0 else v_raw
    g_p = F.pad(g_raw, (0, 0, 0, 0, 0, pad)) if pad > 0 else g_raw

    o_ref_full = cpu_chunk_gla_fwd_o_gk(
        q_p, v_p, g_p, A_seg, h_seg, scale=scale, chunk_size=C
    )
    return q_raw, v_raw, g_raw, A_seg, h_seg, o_ref_full[:, :L]


@pytest.mark.parametrize(
    "cfg", VARLEN_CASES, ids=[_varlen_case_id(c) for c in VARLEN_CASES]
)
def test_pallas_varlen_vs_segments(cfg):
    """Test Pallas varlen path matches per-segment non-varlen computation."""
    seqlens = cfg["seqlens"]
    H, K, V = cfg["H"], cfg["K"], cfg["V"]
    C = cfg.get("chunk_size", 64)
    scale = cfg.get("scale", K**-0.5)
    atol = cfg.get("atol", 1e-2)
    rtol = cfg.get("rtol", 1e-2)
    B = 1

    # Only test chunk-aligned seqlens (unified kernel requires alignment)
    for L in seqlens:
        assert L % C == 0, f"seqlen {L} not aligned to chunk_size {C}"

    torch.manual_seed(cfg["seed"])

    qs, vs, gs = [], [], []
    A_all, h_all, o_refs = [], [], []

    for L in seqlens:
        q_raw, v_raw, g_raw, A_seg, h_seg, o_ref = _make_segment_inputs(
            B, L, H, K, V, C, scale
        )
        qs.append(q_raw)
        vs.append(v_raw)
        gs.append(g_raw)
        A_all.append(A_seg)
        h_all.append(h_seg)
        o_refs.append(o_ref)

    # Pack into varlen format
    q_packed = torch.cat(qs, dim=1)
    v_packed = torch.cat(vs, dim=1)
    g_packed = torch.cat(gs, dim=1)
    # A in Pallas format: [B, NT, H, C, C] -> [B, NT, C, H, C] -> [B, T_seg, H, C]
    # then concat along T dimension
    A_pallas_segs = []
    for A_seg in A_all:
        BN, NT_seg = A_seg.shape[:2]
        A_p = A_seg.permute(0, 1, 3, 2, 4).reshape(BN, NT_seg * C, H, C)
        A_pallas_segs.append(A_p)
    A_packed = torch.cat(A_pallas_segs, dim=1)
    h_packed = torch.cat(h_all, dim=1)
    o_ref_packed = torch.cat(o_refs, dim=1)

    # Run Pallas
    o_pallas = pallas_chunk_gla_fwd_o_gk(
        _torch_to_jax(q_packed),
        _torch_to_jax(v_packed),
        _torch_to_jax(g_packed),
        _torch_to_jax(A_packed),
        _torch_to_jax(h_packed),
        scale=scale,
        chunk_size=C,
    )

    assert compare_tensor("output", o_ref_packed, o_pallas, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__])
