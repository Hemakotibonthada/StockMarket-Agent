"""Symbol universe management and liquidity filters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.core.logging_utils import get_logger

logger = get_logger("data.universe")


class Universe:
    """Manages the tradeable symbol universe with optional filters."""

    def __init__(
        self,
        symbols: list[str] | None = None,
        config_path: str | Path | None = None,
    ):
        self.symbols: list[str] = []
        self.filters: dict[str, Any] = {}

        if config_path:
            self._load_from_config(config_path)
        if symbols:
            self.symbols = symbols

    def _load_from_config(self, path: str | Path) -> None:
        """Load universe from YAML config file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        self.symbols = data.get("symbols", [])
        self.filters = data.get("filters", {})
        logger.info(f"Loaded universe with {len(self.symbols)} symbols from {path}")

    def apply_liquidity_filter(
        self,
        market_data: pd.DataFrame,
        min_avg_volume: int | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        lookback_days: int = 20,
    ) -> list[str]:
        """Filter symbols based on liquidity and price criteria.

        Args:
            market_data: DataFrame with columns: symbol, date, close, volume.
            min_avg_volume: Minimum average daily volume.
            max_price: Maximum price filter.
            min_price: Minimum price filter.
            lookback_days: Number of days for average calculations.

        Returns:
            Filtered list of symbols.
        """
        min_avg_volume = min_avg_volume or self.filters.get("min_avg_volume", 0)
        min_price = min_price or self.filters.get("min_price", 0)
        max_price = max_price or self.filters.get("max_price", float("inf"))

        if market_data.empty:
            return self.symbols

        # Get recent data
        latest_date = market_data["date"].max()
        cutoff = latest_date - pd.Timedelta(days=lookback_days * 2)
        recent = market_data[market_data["date"] >= cutoff]

        passed = []
        for symbol in self.symbols:
            sym_data = recent[recent["symbol"] == symbol]
            if sym_data.empty:
                continue

            avg_vol = sym_data["volume"].mean()
            last_close = sym_data["close"].iloc[-1]

            if avg_vol >= min_avg_volume and min_price <= last_close <= max_price:
                passed.append(symbol)

        logger.info(f"Liquidity filter: {len(self.symbols)} -> {len(passed)} symbols")
        return passed

    def get_symbols(self) -> list[str]:
        """Return the current symbol list."""
        return self.symbols.copy()
