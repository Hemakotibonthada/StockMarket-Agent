"""Shared test configuration and fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _set_seed():
    """Set deterministic seed before each test."""
    from src.core.utils import set_seed
    set_seed(42)
