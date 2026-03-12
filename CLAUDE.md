# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX/Pallas TPU kernels for Gated Linear Attention (GLA) operations. The project implements optimized GLA layers using Google's Pallas framework with support for TPU, GPU, and CPU.

## Common Commands

```bash
# Install dependencies
uv sync                      # Base install
uv sync --extra gpu          # With GPU support (CUDA 12.6)
uv sync --extra tpu          # With TPU support

# Run tests
uv run pytest tests/ -v                                    # All tests
uv run pytest tests/ops/gla/test_pallas_fused_recurrent.py # Single file
uv run pytest tests/ops/gla/test_pallas_fused_recurrent.py::test_fused_recurrent_gla_fwd -v  # Single test

# Lint and format
uv run ruff check src/ tests/           # Lint
uv run ruff check --fix src/ tests/     # Lint with auto-fix
uv run ruff format src/ tests/          # Format

# Pre-commit hooks
pre-commit install           # Install hooks
pre-commit run --all-files   # Run manually

# Launch cloud clusters (SkyPilot)
./scripts/launch_gpu.sh L4 my-cluster        # GPU cluster
./scripts/launch_tpu.sh tpu-v6e-1 my-cluster # TPU cluster
```

## Architecture

```
src/
├── ops/gla/           # Core kernel implementations
│   ├── naive.py       # Pure JAX reference (step-by-step recurrence)
│   ├── chunk.py       # Chunked GLA with Pallas TPU kernels
│   ├── fused_recurrent.py  # Fused recurrent Pallas kernel
│   └── fused_chunk.py      # Wrapper for fused chunk ops
├── layers/
│   └── gla.py         # GatedLinearAttention layer (Flax NNX)
└── modules/           # Building blocks: RMSNorm, ShortConvolution, FusedRMSNormGated

tests/
├── ops/gla/           # Kernel tests (test_pallas_* vs test_torch_*)
├── layers/            # Layer integration tests
├── modules/           # Module unit tests
└── src/               # CPU reference implementations mirroring src/
```

### Key Concepts

**GLA Recurrence Formula:**
```
h_t = h_{t-1} * exp(gk_t) + k_t^T @ v_t
o_t = q_t^T @ h_t
```

**Testing Pattern:** Each Pallas kernel has a corresponding CPU reference test. Tests compare optimized kernels against naive implementations with tolerance-based assertions.

**Gate Types:**
- `gk`: K-dimension gate `[B, T, H, K]` - standard GLA
- `gv`: V-dimension gate `[B, T, H, V]` - optional
- `g`: Scalar gate `[B, T, H]` - whole matrix

## Third-Party Dependencies

- `third_party/flash-linear-attention/` - Reference linear attention implementation (git submodule)
- `third_party/tokamax/` - Related library (git submodule)

## Test Utilities

```python
# tests/utils.py
compare_tensor(name, gold, tensor, atol=1e-5, rtol=1e-5)

# tests/conftest.py provides fixtures:
# - seed(): sets torch.manual_seed(42)
# - device(): returns "cuda" or "cpu"
```

## Pallas Kernel: Syntax, Features, and Optimization

### 1. Overview
Pallas is a JAX primitive that allows writing custom kernels for accelerators (TPU/GPU) using a subset of JAX. It provides fine-grained control over memory hierarchy (HBM vs. VMEM/SMEM) and execution grids, enabling high-performance implementations of operations like FlashAttention.

### 2. Syntax & Structure

*   **Imports:**
    ```python
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    ```

*   **Kernel Definition:**
    Decorated with `jax.jit` or called via `pl.pallas_call`.
    ```python
    def kernel_func(
        # Refs to inputs/outputs/scratch in VMEM/SMEM
        q_ref, k_ref, v_ref, o_ref,
        # Scalar args
        sm_scale, block_size
    ):
        ...
    ```

*   **Invocation:**
    ```python
    pl.pallas_call(
        kernel_func,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=...,
            grid=grid, # Tuple defining the launch grid (e.g., (batch, heads, blocks))
            in_specs=[pl.BlockSpec(...), ...], # Mapping HBM -> VMEM
            out_specs=[pl.BlockSpec(...), ...],
            scratch_shapes=[pltpu.VMEM(...), ...],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", ...), # "parallel" or "arbitrary"
            vmem_limit_bytes=...,
        ),
        out_shape=...
    )(q, k, v)
    ```

*   **BlockSpec:**
    Defines how a block of data is mapped from HBM to VMEM for a specific grid index.
    ```python
    def index_map(batch_idx, head_idx, block_idx):
        return (batch_idx, head_idx, block_idx * block_size, 0)

    pl.BlockSpec(
        block_shape=(batch, heads, block_size, head_dim),
        index_map=index_map
    )
    ```

### 3. Core Features

*   **Program Identity:**
    *   `pl.program_id(axis)`: Get current index in the launch grid.
    *   `pl.num_programs(axis)`: Get total size of the grid dimension.

*   **Memory Spaces:**
    *   `pltpu.HBM`: High Bandwidth Memory (Global).
    *   `pltpu.VMEM`: Vector Memory (Local to TPU core).
    *   `pltpu.SMEM`: Scalar Memory (limited size).

*   **Control Flow:**
    *   `@pl.when(condition)`: Conditional execution (if-block).
    *   `@pl.loop(start, stop, step)`: For-loop.
    *   `jax.lax.cond`, `jax.lax.select`, `jax.lax.while_loop`: Standard JAX control flow.

*   **Data Movement:**
    *   Automatic: Via `in_specs` / `out_specs` in `pl.pallas_call`.
    *   Manual (Async):
        ```python
        # Create a semaphore for synchronization
        sem = pltpu.SemaphoreType.DMA((4, 2))
        # Async copy
        cp = pltpu.make_async_copy(src_ref, dst_ref, sem)
        cp.start()
        cp.wait()
        ```

*   **Compute:**
    *   Standard `jax.numpy` operations (`jnp.matmul`, `jnp.exp`, `jnp.sum`, etc.).
    *   `jax.lax.dot_general`: For matrix multiplication with precision control.

### 4. Optimization Strategies

*   **Pipelining:**
    *   Overlap Compute and Data Transfer (DMA).
    *   Use **Semaphores** and **Async Copies**.
    *   Divide the loop into stages (e.g., Load Next, Compute Current, Store Previous).
    *   Use `sem.wait()` strategically to ensure data is ready only when needed.

*   **Double Buffering:**
    *   Allocate scratch space for *current* and *next* iteration (e.g., `shape=(2, block_size, ...)`).
    *   Compute on buffer `i % 2` while loading into `(i + 1) % 2`.

*   **Tiling (Blocking):**
    *   Process large tensors in small blocks that fit in VMEM.
    *   Typical block sizes: 128x128, 128xHEAD_DIM.
    *   Align block sizes with hardware lanes (multiples of 128 or 8).

*   **Vectorization:**
    *   Pallas kernels are implicitly vectorized over the `BlockSpec` shape.
    *   Avoid scalar loops inside the kernel where possible; use `jnp` array operations.

*   **Memory Layout:**
    *   Ensure data is contiguous in memory for efficient DMA.
    *   Use `padded` shapes if necessary to maintain alignment.
    *   Bitcasting (`pltpu.bitcast`) can be used for zero-cost type reinterpretation.