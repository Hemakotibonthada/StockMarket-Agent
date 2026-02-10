"""Transaction cost model for Indian equity markets.

Covers: brokerage, STT, GST, stamp duty, SEBI turnover charges,
slippage, and market impact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.config import CostsConfig, SlippageConfig


@dataclass
class TransactionCost:
    """Breakdown of costs for a single trade."""

    brokerage: float = 0.0
    stt: float = 0.0
    gst: float = 0.0
    stamp_duty: float = 0.0
    sebi_charges: float = 0.0
    slippage: float = 0.0
    total: float = 0.0

    def __post_init__(self) -> None:
        self.total = self.brokerage + self.stt + self.gst + self.stamp_duty + self.sebi_charges + self.slippage


class CostModel:
    """Compute transaction costs for Indian equity trades."""

    def __init__(
        self,
        costs_config: CostsConfig | None = None,
        slippage_config: SlippageConfig | None = None,
    ):
        self.costs = costs_config or CostsConfig()
        self.slippage = slippage_config or SlippageConfig()

    def compute(
        self,
        price: float,
        quantity: int,
        side: str = "BUY",
        seed: int | None = None,
    ) -> TransactionCost:
        """Compute total transaction costs for a trade.

        Args:
            price: Execution price per share.
            quantity: Number of shares.
            side: 'BUY' or 'SELL'.
            seed: Random seed for slippage.

        Returns:
            TransactionCost breakdown.
        """
        turnover = price * abs(quantity)

        brokerage = turnover * self.costs.brokerage_bps / 10_000
        stt = turnover * self.costs.stt_bps / 10_000
        gst = brokerage * self.costs.gst_bps / 10  # GST on brokerage (18%)
        stamp_duty = turnover * self.costs.stamp_bps / 10_000
        sebi_charges = turnover * self.costs.sebi_bps / 10_000

        # Slippage
        if self.slippage.mode == "random":
            rng = np.random.RandomState(seed) if seed else np.random
            slip_bps = max(0, rng.normal(self.slippage.bps_mean, self.slippage.bps_std))
        elif self.slippage.mode == "fixed":
            slip_bps = self.slippage.bps_mean
        else:
            slip_bps = 0.0

        slippage_cost = turnover * slip_bps / 10_000

        return TransactionCost(
            brokerage=brokerage,
            stt=stt,
            gst=gst,
            stamp_duty=stamp_duty,
            sebi_charges=sebi_charges,
            slippage=slippage_cost,
        )

    def apply_slippage_to_price(
        self,
        price: float,
        side: str = "BUY",
        seed: int | None = None,
    ) -> float:
        """Return the slipped execution price.

        Buys get worse (higher) price; sells get worse (lower) price.
        """
        rng = np.random.RandomState(seed) if seed else np.random

        if self.slippage.mode == "random":
            slip_bps = max(0, rng.normal(self.slippage.bps_mean, self.slippage.bps_std))
        elif self.slippage.mode == "fixed":
            slip_bps = self.slippage.bps_mean
        else:
            slip_bps = 0.0

        slip_pct = slip_bps / 10_000
        if side == "BUY":
            return price * (1 + slip_pct)
        else:
            return price * (1 - slip_pct)

    def total_bps(self) -> float:
        """Return the approximate total cost in BPS (excluding slippage)."""
        return (
            self.costs.brokerage_bps
            + self.costs.stt_bps
            + self.costs.stamp_bps
            + self.costs.sebi_bps
        )
