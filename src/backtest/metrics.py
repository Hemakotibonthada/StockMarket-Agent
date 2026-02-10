"""Performance metrics: CAGR, drawdown, Calmar, Sortino, Profit Factor, VaR/ES."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Summary of backtest performance."""

    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    win_rate_pct: float = 0.0
    total_trades: int = 0
    avg_trade_return_pct: float = 0.0
    max_consecutive_losses: int = 0
    var_95_pct: float = 0.0
    es_95_pct: float = 0.0
    exposure_pct: float = 0.0
    avg_holding_bars: float = 0.0


def compute_returns(equity_curve: pd.Series) -> pd.Series:
    """Compute percentage returns from an equity curve."""
    return equity_curve.pct_change().fillna(0)


def compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    """Compute drawdown series (as negative fractions)."""
    peak = equity_curve.cummax()
    return (equity_curve - peak) / peak


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown as a positive percentage."""
    dd = compute_drawdown(equity_curve)
    return abs(dd.min()) * 100


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """Compound Annual Growth Rate."""
    if len(equity_curve) < 2 or equity_curve.iloc[0] <= 0:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    n_years = len(equity_curve) / periods_per_year
    if n_years <= 0 or total_return <= 0:
        return 0.0
    return (total_return ** (1 / n_years) - 1) * 100


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.06,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio."""
    if returns.std() == 0:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    return float(excess.mean() / returns.std() * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.06,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio (downside deviation)."""
    excess = returns - risk_free_rate / periods_per_year
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(periods_per_year))


def calmar_ratio(cagr_val: float, max_dd_pct: float) -> float:
    """Calmar ratio = CAGR / Max Drawdown."""
    if max_dd_pct == 0:
        return 0.0
    return cagr_val / max_dd_pct


def profit_factor(trade_returns: pd.Series) -> float:
    """Profit factor = gross profit / gross loss."""
    gross_profit = trade_returns[trade_returns > 0].sum()
    gross_loss = abs(trade_returns[trade_returns < 0].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def win_rate(trade_returns: pd.Series) -> float:
    """Win rate as percentage."""
    if len(trade_returns) == 0:
        return 0.0
    return float((trade_returns > 0).sum() / len(trade_returns) * 100)


def max_consecutive_losses(trade_returns: pd.Series) -> int:
    """Longest streak of consecutive losing trades."""
    max_streak = 0
    current_streak = 0
    for r in trade_returns:
        if r < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value at Risk (VaR) at given confidence level."""
    if len(returns) == 0:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100)) * 100


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall (CVaR) at given confidence level."""
    var = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return 0.0
    return float(tail.mean()) * 100


def compute_all_metrics(
    equity_curve: pd.Series,
    trade_returns: pd.Series | None = None,
    periods_per_year: int = 252,
) -> PerformanceMetrics:
    """Compute comprehensive performance metrics.

    Args:
        equity_curve: Series of portfolio values over time.
        trade_returns: Series of per-trade returns (optional).
        periods_per_year: Number of trading periods per year.

    Returns:
        PerformanceMetrics dataclass.
    """
    rets = compute_returns(equity_curve)

    cagr_val = cagr(equity_curve, periods_per_year)
    max_dd = max_drawdown(equity_curve)

    metrics = PerformanceMetrics(
        total_return_pct=(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100 if len(equity_curve) > 1 else 0.0,
        cagr_pct=cagr_val,
        max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe_ratio(rets, periods_per_year=periods_per_year),
        sortino_ratio=sortino_ratio(rets, periods_per_year=periods_per_year),
        calmar_ratio=calmar_ratio(cagr_val, max_dd),
        var_95_pct=value_at_risk(rets),
        es_95_pct=expected_shortfall(rets),
    )

    if trade_returns is not None and len(trade_returns) > 0:
        metrics.total_trades = len(trade_returns)
        metrics.profit_factor = profit_factor(trade_returns)
        metrics.win_rate_pct = win_rate(trade_returns)
        metrics.avg_trade_return_pct = float(trade_returns.mean() * 100)
        metrics.max_consecutive_losses = max_consecutive_losses(trade_returns)

    return metrics
