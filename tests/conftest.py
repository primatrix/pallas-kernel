from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure `tests/` parent (project root) is importable so that
# ``from tests.src.…`` style imports work regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def device():
    return DEVICE
