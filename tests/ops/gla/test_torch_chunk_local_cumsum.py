from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pytest
import torch

from tests.utils import compare_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_CUDA = torch.cuda.is_available()

triton_imports_available = False
if HAS_CUDA:
    try:
        from fla.ops.gla.chunk import (
            chunk_local_cumsum as triton_chunk_local_cumsum_vector,
        )

        triton_imports_available = True
    except ImportError:
        pass

requires_triton = pytest.mark.skipif(
    not (HAS_CUDA and triton_imports_available),
    reason="Triton / CUDA not available",
)

CASES = [
    dict(
        B=2,
        T=32,
        H=4,
        K=128,
        chunk_size=16,
        reverse=False,
        scale=None,
        cu_seqlens=None,
        head_first=False,
        output_dtype=torch.float32,
        chunk_indices=None,
        seed=11,
    ),
    dict(
        B=2,
        T=32,
        H=3,
        K=128,
        chunk_size=8,
        reverse=True,
        scale=0.5,
        cu_seqlens=None,
        head_first=False,
        output_dtype=torch.float32,
        chunk_indices=None,
        seed=22,
    ),
    dict(
        B=1,
        T=33,
        H=2,
        K=128,
        chunk_size=16,
        reverse=False,
        scale=0.75,
        cu_seqlens=[0, 17, 33],
        head_first=False,
        output_dtype=torch.float32,
        chunk_indices=None,
        seed=33,
    ),
    dict(
        B=1,
        T=33,
        H=2,
        K=128,
        chunk_size=16,
        reverse=True,
        scale=None,
        cu_seqlens=[0, 17, 33],
        head_first=False,
        output_dtype=torch.float32,
        chunk_indices=[[0, 0], [0, 1], [1, 0]],
        seed=55,
    ),
    dict(
        B=3,
        T=128,
        H=6,
        K=128,
        chunk_size=32,
        reverse=False,
        scale=0.125,
        cu_seqlens=None,
        head_first=False,
        output_dtype=torch.float32,
        chunk_indices=None,
        seed=66,
    ),
    dict(
        B=10,
        T=1280,
        H=6,
        K=128,
        chunk_size=32,
        reverse=False,
        scale=0.125,
        cu_seqlens=None,
        head_first=True,
        output_dtype=torch.float32,
        chunk_indices=None,
        seed=67,
    ),
    dict(
        B=10,
        T=128,
        H=64,
        K=128,
        chunk_size=32,
        reverse=False,
        scale=0.125,
        cu_seqlens=None,
        head_first=True,
        output_dtype=torch.float16,
        chunk_indices=None,
        seed=66,
        atol=1e-3,
        rtol=1e-3,
    ),
]


def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}"]
    if c.get("cu_seqlens"):
        parts.append(f"segs{len(c['cu_seqlens']) - 1}")
    if c.get("reverse"):
        parts.append("rev")
    if c.get("scale") is not None:
        parts.append(f"scale={c['scale']}")
    if c.get("chunk_size") is not None:
        parts.append(f"chunk{c['chunk_size']}")
    if c.get("head_first"):
        parts.append("head_first")
    if c.get("output_dtype") is not None:
        dtype = c["output_dtype"]
        dtype_name = getattr(dtype, "__name__", None)
        if dtype_name is None:
            dtype_name = str(dtype).replace("torch.", "")
        parts.append(f"dtype={dtype_name}")
    if c.get("chunk_indices") is not None:
        parts.append("custom_idx")
    return "-".join(parts)


def _to_torch_dtype(dtype) -> torch.dtype | None:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    return torch.from_numpy(np.empty((), dtype=np.dtype(dtype))).dtype


def _to_torch_cu_seqlens(cu_seqlens) -> torch.Tensor | None:
    if cu_seqlens is None:
        return None
    if isinstance(cu_seqlens, torch.Tensor):
        return cu_seqlens.to(dtype=torch.long)
    return torch.tensor(cu_seqlens, dtype=torch.long)


def _chunk_local_cumsum_cpu(
    g: torch.Tensor,
    chunk_size: int,
    *,
    reverse: bool = False,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    if g.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tuple(g.shape)}")

    B, T, _, _ = g.shape
    out = torch.zeros_like(g, dtype=torch.float32)

    if cu_seqlens is None:
        segments = [(0, T)]
    else:
        segments = [
            (max(0, int(start)), min(T, int(end)))
            for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False)
        ]

    for b in range(B):
        for start, end in segments:
            if start >= end:
                continue
            for chunk_start in range(start, end, chunk_size):
                chunk_end = min(chunk_start + chunk_size, end)
                chunk = g[b, chunk_start:chunk_end].to(torch.float32)
                if reverse:
                    chunk = torch.flip(
                        torch.cumsum(torch.flip(chunk, dims=[0]), dim=0), dims=[0]
                    )
                else:
                    chunk = torch.cumsum(chunk, dim=0)
                if scale is not None:
                    chunk = chunk * scale
                out[b, chunk_start:chunk_end] = chunk

    return out


def _run_cpu(
    g,
    chunk_size,
    *,
    reverse=False,
    scale=None,
    cu_seqlens=None,
    head_first=False,
    output_dtype=None,
    chunk_indices=None,
):
    del chunk_indices

    cu = _to_torch_cu_seqlens(cu_seqlens)
    torch_dtype = _to_torch_dtype(output_dtype)

    if head_first:
        g = g.transpose(1, 2)

    out = _chunk_local_cumsum_cpu(
        g,
        chunk_size=chunk_size,
        reverse=reverse,
        scale=scale,
        cu_seqlens=cu,
    )

    if head_first:
        out = out.transpose(1, 2)

    if torch_dtype is not None:
        out = out.to(dtype=torch_dtype)

    return out


@requires_triton
def _run_triton(
    g,
    chunk_size,
    *,
    reverse=False,
    scale=None,
    cu_seqlens=None,
    head_first=False,
    output_dtype=None,
    chunk_indices=None,
):
    device = torch.device("cuda")
    g_cuda = g.to(device=device)
    cu_cuda = _to_torch_cu_seqlens(cu_seqlens)
    chunk_indices_cuda = _to_torch_cu_seqlens(chunk_indices)
    if cu_cuda is not None:
        cu_cuda = cu_cuda.to(device=device)
    if chunk_indices_cuda is not None:
        chunk_indices_cuda = chunk_indices_cuda.to(device=device)
    return triton_chunk_local_cumsum_vector(
        g_cuda,
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens=cu_cuda,
        reverse=reverse,
        head_first=head_first,
        output_dtype=output_dtype,
        chunk_indices=chunk_indices_cuda,
    ).cpu()


# ============================================================================
# Parametrized test — Torch CPU vs Triton
# ============================================================================


@requires_triton
@pytest.mark.parametrize("cfg", CASES, ids=[_case_id(c) for c in CASES])
def test_cpu_vs_triton(cfg):
    B, T, H, K = cfg["B"], cfg["T"], cfg["H"], cfg["K"]
    atol = cfg.get("atol", 1e-4)
    rtol = cfg.get("rtol", 1e-4)
    scale = cfg.get("scale", None)
    reverse = cfg.get("reverse", False)
    chunk_size = cfg.get("chunk_size", 64)

    torch.manual_seed(cfg["seed"])
    if cfg.get("head_first", False):
        g = torch.randn(B, H, T, K)
    else:
        g = torch.randn(B, T, H, K)

    cu = cfg.get("cu_seqlens", None)
    o_cpu = _run_cpu(
        g=g,
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens=cu,
        reverse=reverse,
        head_first=cfg.get("head_first", False),
        output_dtype=cfg.get("output_dtype"),
        chunk_indices=cfg.get("chunk_indices"),
    )
    o_triton = _run_triton(
        g=g,
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens=cu,
        reverse=reverse,
        head_first=cfg.get("head_first", False),
        output_dtype=cfg.get("output_dtype"),
        chunk_indices=cfg.get("chunk_indices"),
    )

    assert compare_tensor("output", o_cpu, o_triton, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__])
