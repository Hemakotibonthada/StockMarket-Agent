"""Portfolio-level risk constraints: sector caps, asset caps, correlation limits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.core.logging_utils import get_logger

logger = get_logger("risk.portfolio")


class PortfolioRisk:
    """Enforce portfolio-level constraints.

    - Maximum allocation per symbol.
    - Maximum allocation per sector.
    - Correlation caps to avoid over-concentration.
    """

    def __init__(
        self,
        max_single_pct: float = 20.0,
        max_sector_pct: float = 40.0,
        max_correlation: float = 0.8,
        sector_map: dict[str, str] | None = None,
    ):
        self.max_single_pct = max_single_pct
        self.max_sector_pct = max_sector_pct
        self.max_correlation = max_correlation
        self.sector_map = sector_map or {}

    def check_position_limit(
        self,
        symbol: str,
        position_value: float,
        portfolio_value: float,
    ) -> tuple[bool, str]:
        """Check if adding/holding this position exceeds single-stock limit."""
        if portfolio_value <= 0:
            return False, "Portfolio value is zero"

        pct = position_value / portfolio_value * 100
        if pct > self.max_single_pct:
            return False, f"{symbol} allocation {pct:.1f}% exceeds limit {self.max_single_pct}%"

        return True, "OK"

    def check_sector_limit(
        self,
        symbol: str,
        positions: dict[str, float],
        portfolio_value: float,
    ) -> tuple[bool, str]:
        """Check if the sector allocation would exceed the limit."""
        if not self.sector_map or portfolio_value <= 0:
            return True, "OK"

        sector = self.sector_map.get(symbol, "Unknown")
        sector_value = sum(
            v for s, v in positions.items()
            if self.sector_map.get(s, "Unknown") == sector
        )

        sector_pct = sector_value / portfolio_value * 100
        if sector_pct > self.max_sector_pct:
            return False, f"Sector '{sector}' at {sector_pct:.1f}% exceeds limit {self.max_sector_pct}%"

        return True, "OK"

    def check_correlation(
        self,
        symbol: str,
        positions: list[str],
        returns_data: pd.DataFrame,
    ) -> tuple[bool, str]:
        """Check if the new symbol is too correlated with existing positions."""
        if not positions or symbol not in returns_data.columns:
            return True, "OK"

        for existing in positions:
            if existing not in returns_data.columns:
                continue
            corr = returns_data[symbol].corr(returns_data[existing])
            if abs(corr) > self.max_correlation:
                return False, (
                    f"{symbol} corr with {existing} is {corr:.2f}, "
                    f"exceeds limit {self.max_correlation}"
                )

        return True, "OK"

    def portfolio_summary(
        self,
        positions: dict[str, float],
        portfolio_value: float,
    ) -> dict[str, Any]:
        """Generate portfolio allocation summary."""
        summary: dict[str, Any] = {"positions": {}, "sectors": {}}

        for sym, val in positions.items():
            pct = val / portfolio_value * 100 if portfolio_value > 0 else 0
            summary["positions"][sym] = {"value": val, "pct": pct}

            sector = self.sector_map.get(sym, "Unknown")
            if sector not in summary["sectors"]:
                summary["sectors"][sector] = 0
            summary["sectors"][sector] += pct

        summary["n_positions"] = len(positions)
        summary["gross_exposure_pct"] = sum(
            abs(v) / portfolio_value * 100 for v in positions.values()
        ) if portfolio_value > 0 else 0

        return summary
