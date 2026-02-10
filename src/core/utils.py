"""Common utility functions."""

from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division with zero-safe fallback."""
    if denominator == 0:
        return default
    return numerator / denominator


def round_to_tick(price: float, tick_size: float = 0.05) -> float:
    """Round price to the nearest tick size."""
    return round(round(price / tick_size) * tick_size, 2)


def round_to_lot(quantity: int, lot_size: int = 1) -> int:
    """Round quantity to the nearest lot size."""
    return max(lot_size, (quantity // lot_size) * lot_size)


def file_hash(path: str) -> str:
    """Compute SHA256 hash of a file for versioning."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dictionary."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
