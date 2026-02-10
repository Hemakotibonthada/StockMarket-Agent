"""Position sizing based on ATR, variance, and risk budgets."""

from __future__ import annotations

from src.core.logging_utils import get_logger
from src.core.utils import round_to_lot, round_to_tick

logger = get_logger("risk.sizing")


def atr_position_size(
    capital: float,
    risk_pct: float,
    atr_value: float,
    atr_multiplier: float = 2.0,
    tick_size: float = 0.05,
    lot_size: int = 1,
    max_position_value: float | None = None,
) -> int:
    """Compute position size using ATR-based risk.

    Risk per trade = capital * risk_pct
    Shares = risk_amount / (ATR * multiplier)

    Args:
        capital: Current portfolio value.
        risk_pct: Fraction of capital to risk (e.g., 0.005 for 0.5%).
        atr_value: Current ATR value.
        atr_multiplier: Multiplier for risk distance.
        tick_size: Minimum price increment.
        lot_size: Minimum lot size.
        max_position_value: Maximum notional value for a single position.

    Returns:
        Number of shares to trade.
    """
    if atr_value <= 0 or capital <= 0:
        return 0

    risk_amount = capital * risk_pct
    risk_per_share = atr_value * atr_multiplier

    quantity = int(risk_amount / risk_per_share)
    quantity = round_to_lot(quantity, lot_size)

    # Apply max position value constraint
    if max_position_value and quantity > 0:
        # Estimate price as midpoint, but we'll clip by value
        pass  # Caller should enforce this with actual price

    return max(0, quantity)


def variance_position_size(
    capital: float,
    risk_pct: float,
    volatility: float,
    price: float,
    lot_size: int = 1,
) -> int:
    """Compute position size based on realized volatility.

    Args:
        capital: Current portfolio value.
        risk_pct: Fraction of capital to risk.
        volatility: Annualized realized volatility (e.g., 0.25 for 25%).
        price: Current price per share.
        lot_size: Minimum lot size.

    Returns:
        Number of shares.
    """
    if volatility <= 0 or price <= 0 or capital <= 0:
        return 0

    # Daily vol from annualized
    daily_vol = volatility / (252 ** 0.5)
    risk_amount = capital * risk_pct
    price_risk = price * daily_vol

    quantity = int(risk_amount / price_risk)
    return round_to_lot(max(0, quantity), lot_size)


def fixed_fraction_size(
    capital: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
    lot_size: int = 1,
) -> int:
    """Fixed-fraction sizing based on entry and stop-loss distance.

    Args:
        capital: Portfolio value.
        risk_pct: Fraction of capital to risk.
        entry_price: Planned entry price.
        stop_loss: Stop-loss price.
        lot_size: Minimum lot size.

    Returns:
        Number of shares.
    """
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share <= 0 or capital <= 0:
        return 0

    risk_amount = capital * risk_pct
    quantity = int(risk_amount / risk_per_share)
    return round_to_lot(max(0, quantity), lot_size)


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Kelly criterion position fraction.

    Args:
        win_rate: Probability of winning (0-1).
        avg_win: Average winning trade return.
        avg_loss: Average losing trade return (positive number).

    Returns:
        Optimal fraction of capital to risk (capped at 0.25).
    """
    if avg_loss <= 0:
        return 0.0

    b = avg_win / avg_loss  # Win/loss ratio
    p = win_rate
    q = 1 - p

    kelly = (b * p - q) / b
    # Cap at 25% and floor at 0
    return max(0.0, min(0.25, kelly))
