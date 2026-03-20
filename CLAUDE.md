# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

High-performance JAX/Pallas kernels for sequence modeling operations (including GLA, attention, etc.). The project implements optimized layers using Google's Pallas framework with support for TPU and GPU.

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
uv run ruff check tops/ tests/           # Lint
uv run ruff check --fix tops/ tests/     # Lint with auto-fix
uv run ruff format tops/ tests/          # Format

# Pre-commit hooks
pre-commit install           # Install hooks
pre-commit run --all-files   # Run manually

# Launch cloud clusters (SkyPilot)
./scripts/launch_gpu.sh L4 my-cluster        # GPU cluster
./scripts/launch_tpu.sh tpu-v6e-1 my-cluster # TPU cluster
