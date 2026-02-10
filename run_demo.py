"""Demo: Run end-to-end backtest with synthetic data."""

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.core.config import AppConfig, RiskConfig, SlippageConfig
from src.core.utils import set_seed
from src.backtest.engine import BacktestEngine
from src.strategies.mean_reversion import MeanReversion
from src.features.feature_sets import compute_base_features

console = Console()

# --- Generate synthetic NIFTY50 stock data (500 bars) ---
set_seed(42)
n = 500
rng = np.random.default_rng(42)
dates = pd.bdate_range("2024-01-01", periods=n, freq="D")
price = 2000.0
closes = []
for _ in range(n):
    price *= np.exp(rng.normal(0.0003, 0.015))
    closes.append(price)
closes = np.array(closes)

df = pd.DataFrame({
    "date": dates,
    "symbol": "RELIANCE",
    "open": closes * (1 + rng.uniform(-0.005, 0.005, n)),
    "high": closes * (1 + rng.uniform(0.005, 0.02, n)),
    "low": closes * (1 - rng.uniform(0.005, 0.02, n)),
    "close": closes,
    "volume": rng.integers(500_000, 5_000_000, n),
})

# Compute all technical features
df = compute_base_features(df)

# --- Configure backtest ---
cfg = AppConfig(
    strategy="mean_reversion",
    strategy_params={"zscore_entry": 1.5, "zscore_exit": 0.5},
    slippage=SlippageConfig(mode="fixed", bps_mean=3.0, bps_std=0.0),
    risk=RiskConfig(initial_capital=1_000_000),
)
strategy = MeanReversion(config={"zscore_entry": 1.5, "zscore_exit": 0.5})

# --- Run backtest ---
console.print("\n[bold blue]============================================[/]")
console.print("[bold blue]    Stock Agent - Backtest Engine[/]")
console.print("[bold blue]============================================[/]\n")
console.print(f"  Symbol:    RELIANCE (synthetic)")
console.print(f"  Strategy:  Mean Reversion (z-score)")
console.print(f"  Capital:   INR {cfg.risk.initial_capital:,.0f}")
console.print(f"  Bars:      {len(df)}")

date_start = df["date"].iloc[0].date()
date_end = df["date"].iloc[-1].date()
console.print(f"  Period:    {date_start} to {date_end}\n")

engine = BacktestEngine(config=cfg, strategy=strategy)
result = engine.run(df)

# --- Display results ---
summary = result.summary()

table = Table(title="Backtest Results", show_header=True)
table.add_column("Metric", style="cyan", width=25)
table.add_column("Value", style="green", justify="right")

for k, v in summary.items():
    label = k.replace("_", " ").title()
    table.add_row(label, str(v))

console.print(table)

console.print(f"\n  Total trades:  {len(result.trades)}")
if result.trades:
    winners = [t for t in result.trades if t.net_pnl > 0]
    losers = [t for t in result.trades if t.net_pnl <= 0]
    console.print(f"  Winners:       {len(winners)}")
    console.print(f"  Losers:        {len(losers)}")
    total_pnl = sum(t.net_pnl for t in result.trades)
    console.print(f"  Net P&L:       INR {total_pnl:,.2f}")

final_equity = result.equity_curve.iloc[-1] if len(result.equity_curve) > 0 else cfg.risk.initial_capital
console.print(f"  Final equity:  INR {final_equity:,.2f}")
console.print(f"  Total costs:   INR {result.total_costs:,.2f}")
console.print("\n[bold green]Backtest completed successfully.[/]\n")
