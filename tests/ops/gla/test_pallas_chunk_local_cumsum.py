"""chunk_local_cumsum: Pallas kernel vs Torch CPU reference tests."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from src.ops.gla import chunk_local_cumsum as pallas_chunk_local_cumsum
from src.utils import prepare_chunk_indices
from tests.utils import compare_tensor


PALLAS_CASES = [
    dict(
        B=2,
        T=32,
        H=4,
        K=32,
        chunk_size=16,
        reverse=False,
        scale=None,
        cu_seqlens=None,
        head_first=False,
        output_dtype=jnp.float32,
        chunk_indices=None,
        seed=11,
    ),
    dict(
        B=2,
        T=32,
        H=3,
        K=17,
        chunk_size=8,
        reverse=True,
        scale=0.5,
        cu_seqlens=None,
        head_first=False,
        output_dtype=jnp.float32,
        chunk_indices=None,
        seed=22,
    ),
    dict(
        B=1,
        T=33,
        H=2,
        K=65,
        chunk_size=16,
        reverse=False,
        scale=0.75,
        cu_seqlens=[0, 17, 33],
        head_first=False,
        output_dtype=jnp.float32,
        chunk_indices=None,
        seed=33,
    ),
    dict(
        B=2,
        T=32,
        H=3,
        K=8,
        chunk_size=16,
        reverse=True,
        scale=1.25,
        cu_seqlens=[0, 24, 32],
        head_first=True,
        output_dtype=jnp.float16,
        chunk_indices=None,
        seed=44,
    ),
    dict(
        B=1,
        T=33,
        H=2,
        K=7,
        chunk_size=16,
        reverse=True,
        scale=None,
        cu_seqlens=[0, 17, 33],
        head_first=False,
        output_dtype=jnp.float32,
        chunk_indices=prepare_chunk_indices(
            jnp.array([0, 17, 33], dtype=jnp.int32), 16
        ).tolist(),
        seed=55,
    ),
    dict(
        B=3,
        T=128,
        H=6,
        K=96,
        chunk_size=32,
        reverse=False,
        scale=0.125,
        cu_seqlens=None,
        head_first=False,
        output_dtype=jnp.float32,
        chunk_indices=None,
        seed=66,
    ),
    dict(
        B=1,
        T=160,
        H=4,
        K=129,
        chunk_size=32,
        reverse=True,
        scale=0.375,
        cu_seqlens=[0, 49, 111, 160],
        head_first=False,
        output_dtype=jnp.float32,
        chunk_indices=prepare_chunk_indices(
            jnp.array([0, 49, 111, 160], dtype=jnp.int32), 32
        ).tolist(),
        seed=77,
    ),
    dict(
        B=2,
        T=96,
        H=5,
        K=64,
        chunk_size=32,
        reverse=True,
        scale=1.5,
        cu_seqlens=[0, 45, 96],
        head_first=True,
        output_dtype=jnp.float16,
        chunk_indices=None,
        seed=88,
        atol=5e-3,
        rtol=5e-3,
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
        parts.append(f"dtype={c['output_dtype'].__name__}")
    if c.get("chunk_indices") is not None:
        parts.append("custom_idx")
    return "-".join(parts)


# ============================================================================
# Helpers
# ============================================================================


def _torch_to_jax(t: torch.Tensor) -> jax.Array:
    """Convert a torch tensor to a JAX array."""
    return jnp.array(t.detach().to(torch.float32).numpy())


def _to_torch_dtype(dtype) -> torch.dtype | None:
    if dtype is None:
        return None
    return torch.from_numpy(np.empty((), dtype=np.dtype(dtype))).dtype


def _to_torch_cu_seqlens(cu_seqlens) -> torch.Tensor | None:
    if cu_seqlens is None:
        return None
    if isinstance(cu_seqlens, torch.Tensor):
        return cu_seqlens.to(dtype=torch.long)
    return torch.tensor(cu_seqlens, dtype=torch.long)


def _to_jax_cu_seqlens(cu_seqlens) -> jax.Array | None:
    if cu_seqlens is None:
        return None
    return jnp.asarray(cu_seqlens, dtype=jnp.int32)


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


def _run_pallas(
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
    return pallas_chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens=_to_jax_cu_seqlens(cu_seqlens),
        reverse=reverse,
        head_first=head_first,
        output_dtype=output_dtype,
        chunk_indices=_to_jax_cu_seqlens(chunk_indices),
    )


# ============================================================================
# Parametrized test — Torch CPU vs Pallas
# ============================================================================


@pytest.mark.parametrize("cfg", PALLAS_CASES, ids=[_case_id(c) for c in PALLAS_CASES])
def test_cpu_vs_pallas(cfg):
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
    o_pallas = _run_pallas(
        g=_torch_to_jax(g),
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens=cu,
        reverse=reverse,
        head_first=cfg.get("head_first", False),
        output_dtype=cfg.get("output_dtype"),
        chunk_indices=cfg.get("chunk_indices"),
    )

    assert compare_tensor("output", o_cpu, o_pallas, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__])
