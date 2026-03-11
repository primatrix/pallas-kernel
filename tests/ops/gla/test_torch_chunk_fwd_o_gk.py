"""chunk_gla_fwd_o_gk: FLA Triton GPU (gold) vs Torch CPU kernel tests.

Both the FLA Triton kernel and our CPU reference compute:

    o = scale * (q * exp(g)) @ h  +  A @ v        # A has scale baked in

The test builds random inputs and verifies both implementations produce
the same output.  A is generated in CPU format [B, NT, H, C, C] and
converted to Triton format [B, T, H, C] via permute+reshape.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import torch
import torch.nn.functional as F

from tests.src.ops.gla.chunk import chunk_gla_fwd_o_gk as cpu_chunk_gla_fwd_o_gk
from tests.utils import compare_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_CUDA = torch.cuda.is_available()

triton_imports_available = False
try:
    from fla.ops.gla.chunk import chunk_gla_fwd_o_gk as triton_chunk_gla_fwd_o_gk

    triton_imports_available = True
except ImportError:
    pass

requires_triton = pytest.mark.skipif(
    not (HAS_CUDA and triton_imports_available),
    reason="Triton / CUDA not available",
)

# ============================================================================
# Unified test configs
#
# Each dict can contain:
#   B, T, H, K, V   — shape (required, T must be a multiple of chunk_size)
#   seed             — random seed (required)
#   chunk_size       — block size (default 64)
#   atol             — absolute tolerance (default 1e-2)
#   rtol             — relative tolerance (default 1e-2)
#   scale            — float or None (default None = K^{-0.5})
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
    dict(B=1, T=1024, H=2, K=32, V=64, seed=302),
    # ── long + large dims ──
    dict(B=1, T=256, H=2, K=64, V=128, seed=350),
    dict(B=1, T=256, H=2, K=128, V=64, seed=351),
    dict(B=1, T=512, H=2, K=64, V=128, seed=352),
    # ── long + multi-batch ──
    dict(B=4, T=256,  H=2, K=32,  V=64,  seed=360, atol=5e-2, rtol=5e-2),
    dict(B=2, T=512,  H=4, K=32,  V=64,  seed=361),
    # ── long + many heads ──
    dict(B=1, T=256, H=8, K=32, V=64, seed=370),
    dict(B=1, T=512, H=8, K=32, V=64, seed=371),
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


def _to_device(*tensors):
    return tuple(t.to(DEVICE) for t in tensors)


def _make_inputs(B, T, H, K, V, chunk_size, scale):
    """Generate random inputs for chunk_gla_fwd_o_gk.

    A is generated with scale baked in (matching Triton/Pallas convention).

    Format conversion for Triton:
        CPU:    [B, NT, H, C, C]  — A[b, n, h, i, j]
        Triton: [B, T, H, C]     — A[b, n*C+i, h, j]

    Returns:
        q, v, g, A, A_triton, h
    """
    C = chunk_size
    NT = T // C

    q = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    g = F.logsigmoid(torch.randn(B, T, H, K))
    h = torch.randn(B, NT, H, K, V)

    A = torch.randn(B, NT, C, H, C)
    causal_mask = torch.tril(torch.ones(C, C, dtype=torch.bool))[:, None, :]  # [C, 1, C]
    A = A.masked_fill(~causal_mask, 0.0)

    # Triton format: [B, NT, C, H, C] -> [B, T, H, C]
    A_triton = A.reshape(B, T, H, C).contiguous()

    return q, v, g, A, A_triton, h


# ============================================================================
# Main parametrized test — gold (Triton GPU) vs cpu (Torch)
# ============================================================================


@requires_triton
@pytest.mark.parametrize("cfg", CASES, ids=[_case_id(c) for c in CASES])
def test_gold_vs_cpu(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    atol = cfg.get("atol", 1e-2)
    rtol = cfg.get("rtol", 1e-2)
    scale = cfg.get("scale", K**-0.5)
    chunk_size = cfg.get("chunk_size", 64)

    assert T % chunk_size == 0, f"T={T} must be a multiple of chunk_size={chunk_size}"

    torch.manual_seed(cfg["seed"])
    q, v, g, A, A_triton, h = _make_inputs(B, T, H, K, V, chunk_size, scale)

    # CPU reference
    o_cpu = cpu_chunk_gla_fwd_o_gk(q, v, g, A, h, scale=scale, chunk_size=chunk_size)
    # Triton GPU
    o_gold = triton_chunk_gla_fwd_o_gk(
        *_to_device(q, v, g, A_triton, h), scale=scale, chunk_size=chunk_size
    ).cpu()

    assert compare_tensor("output", o_gold, o_cpu, atol=atol, rtol=rtol)


# ============================================================================
# Varlen test configs
#
# Each dict can contain:
#   seqlens      — list of per-sequence lengths (required)
#   H, K, V      — shape (required)
#   seed         — random seed (required)
#   chunk_size   — block size (default 64)
#   atol         — absolute tolerance (default 1e-4)
#   rtol         — relative tolerance (default 1e-4)
#   scale        — float or None (default None = K^{-0.5})
# ============================================================================

VARLEN_CASES = [
    # ── basic (all seqlens are multiples of chunk_size) ──
    dict(seqlens=[64, 64], H=4, K=32, V=64, seed=1000),
    dict(seqlens=[128, 64], H=4, K=32, V=64, seed=1001),
    dict(seqlens=[64, 128, 64], H=2, K=32, V=64, seed=1002),
    dict(seqlens=[64, 64, 64, 64], H=4, K=16, V=32, seed=1003),
    # ── varied multiples of chunk_size ──
    dict(seqlens=[64, 128], H=4, K=32, V=64, seed=1004),
    dict(seqlens=[32, 64, 96], H=2, K=32, V=64, seed=1005, chunk_size=32),
    dict(seqlens=[128, 64], H=2, K=32, V=64, seed=1006),
    # ── single sequence ──
    dict(seqlens=[128], H=4, K=32, V=64, seed=1007),
    dict(seqlens=[192], H=4, K=32, V=64, seed=1008),
    # ── K != V ──
    dict(seqlens=[64, 128], H=4, K=16, V=128, seed=1009),
    dict(seqlens=[64, 128], H=4, K=128, V=16, seed=1010),
    # ── custom scale ──
    dict(seqlens=[64, 64], H=4, K=32, V=64, seed=1011, scale=0.1),
    # ── small chunk_size ──
    dict(seqlens=[64, 48], H=4, K=32, V=64, seed=1012, chunk_size=16),
    dict(seqlens=[32, 64, 48], H=2, K=32, V=64, seed=1013, chunk_size=16),
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
    """Generate random inputs for one segment, with padding.

    Returns:
        q_raw:  [B, L, H, K]   — unpadded
        v_raw:  [B, L, H, V]   — unpadded
        g_raw:  [B, L, H, K]   — unpadded
        A_seg:  [B, NT, H, C, C] — causal-masked
        h_seg:  [B, NT, H, K, V]
        o_ref:  [B, L, H, V]   — reference output (valid positions only)
    """
    C = chunk_size
    NT = (L + C - 1) // C
    T_pad = NT * C
    pad = T_pad - L

    q_raw = torch.randn(B, L, H, K)
    v_raw = torch.randn(B, L, H, V)
    g_raw = F.logsigmoid(torch.randn(B, L, H, K))

    A_seg = torch.randn(B, NT, C, H, C)
    causal_mask = torch.tril(torch.ones(C, C, dtype=torch.bool))[:, None, :]  # [C, 1, C]
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


# ============================================================================
# Varlen test — varlen path vs per-segment non-varlen computation
# ============================================================================


@pytest.mark.parametrize(
    "cfg", VARLEN_CASES, ids=[_varlen_case_id(c) for c in VARLEN_CASES]
)
def test_varlen_vs_segments(cfg):
    """Test that varlen path matches per-segment computation."""
    seqlens = cfg["seqlens"]
    H, K, V = cfg["H"], cfg["K"], cfg["V"]
    C = cfg.get("chunk_size", 64)
    scale = cfg.get("scale", K**-0.5)
    atol = cfg.get("atol", 1e-4)
    rtol = cfg.get("rtol", 1e-4)
    B = 1

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
    A_packed = torch.cat(A_all, dim=1)
    h_packed = torch.cat(h_all, dim=1)
    o_ref_packed = torch.cat(o_refs, dim=1)

    offsets = [0]
    for s in seqlens:
        offsets.append(offsets[-1] + s)
    cu_seqlens = torch.tensor(offsets, dtype=torch.long)

    # Run varlen path
    o_varlen = cpu_chunk_gla_fwd_o_gk(
        q_packed,
        v_packed,
        g_packed,
        A_packed,
        h_packed,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=C,
    )

    assert compare_tensor("output", o_varlen, o_ref_packed, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__])
