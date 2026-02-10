# Stock Agent 🇮🇳

A **local-first**, modular stock trading agent for Indian equities (NSE).  
Paper trading by default. Optional live adapters for Zerodha / Upstox.

> **Disclaimer**: This software is for **educational and research purposes only**.  
> The authors accept **no liability** for any trading losses.  
> Always consult a SEBI-registered advisor before trading.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Layer                             │
│   sa-data │ sa-backtest │ sa-paper │ sa-live                 │
├─────────────────────────────────────────────────────────────┤
│                     Live Event Loop                          │
│   event_loop │ aggregator │ health monitor                   │
├──────────────────┬──────────────────┬───────────────────────┤
│  Execution Layer │   Risk Layer     │  Training Pipeline     │
│  paper_broker    │   sizing         │  dataset               │
│  zerodha_adapter │   limits         │  train / eval          │
│  upstox_adapter  │   portfolio      │  model registry        │
│  order_router    │   tripwires      │                        │
│  reconciler      │                  │                        │
├──────────────────┴──────────────────┴───────────────────────┤
│                    Backtest Engine                            │
│   costs │ metrics │ engine │ walkforward │ reports            │
├─────────────────────────────────────────────────────────────┤
│                      Strategies                              │
│   orb_momentum │ mean_reversion │ pairs_trading              │
├─────────────────────────────────────────────────────────────┤
│                   Features Library                           │
│   indicators │ feature_sets (30+ features)                   │
├─────────────────────────────────────────────────────────────┤
│                      Data Layer                              │
│   loaders │ adjustments │ resample │ calendar │ universe     │
├─────────────────────────────────────────────────────────────┤
│                    Core Utilities                             │
│   config (YAML+Pydantic) │ clocks (IST) │ logging │ io      │
├─────────────────────────────────────────────────────────────┤
│                      Storage                                 │
│   Parquet (time series) │ DuckDB/SQLite (metadata)           │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd stock-agent
make install        # production deps
make dev            # + dev/test deps

# 2. Prepare data
cp -r your_data/ data/raw/
sa-data ingest --input-dir data/raw --output-dir data/processed

# 3. Run a backtest
sa-backtest run --config configs/backtest.yaml

# 4. Paper trade
sa-paper --config configs/paper_trade.yaml

# 5. Live trade (requires broker setup + explicit confirmation)
sa-live start --config configs/live_trade.yaml --confirm-live true
```

See [RUN.md](RUN.md) for detailed copy-paste commands.

---

## Project Structure

```
stock-agent/
├── configs/                 # YAML configs with inheritance
│   ├── base.yaml            # Shared defaults
│   ├── backtest.yaml        # Backtest parameters
│   ├── paper_trade.yaml     # Paper trading setup
│   ├── live_trade.yaml      # Live trading (Zerodha)
│   └── universe_nifty50.yaml # Symbol universe
├── src/
│   ├── core/                # Config, logging, IO, clocks, utils
│   ├── data/                # Ingestion, adjustments, resampling, calendar
│   ├── features/            # Technical indicators, feature engineering
│   ├── strategies/          # Trading strategies (ORB, MR, Pairs)
│   ├── models/              # ML models (classic + deep learning)
│   ├── backtest/            # Engine, costs, metrics, walk-forward
│   ├── training/            # Dataset prep, training, evaluation
│   ├── risk/                # Position sizing, limits, portfolio, tripwires
│   ├── exec/                # Broker adapters, order routing, reconciliation
│   ├── live/                # Event loop, bar aggregation, health monitor
│   └── cli/                 # Typer CLI entry points
├── tests/                   # Unit tests (pytest)
├── data/                    # Data directory (gitignored)
├── results/                 # Backtest results
├── logs/                    # Runtime logs
├── notebooks/               # Jupyter notebooks
├── pyproject.toml           # Project config
├── Makefile                 # Common tasks
└── .env.example             # Broker API credentials template
```

---

## Configuration

All configs are **YAML** with **inheritance** via `inherits: base.yaml`.  
Validated at load time using **Pydantic** models.

```yaml
# configs/backtest.yaml
inherits: base.yaml
strategy: orb_momentum
strategy_params:
  lookback: 15
  atr_multiplier: 1.5
initial_capital: 1000000
costs:
  brokerage_bps: 3
  stt_bps: 2.5
```

See `src/core/config.py` for all available fields.

---

## Strategies

| Strategy | Description | Timeframe |
|----------|------------|-----------|
| **ORB Momentum** | Opening range breakout with ATR filter + volume confirmation | Intraday |
| **Mean Reversion** | Z-score + RSI dual filter with dynamic thresholds | Swing / Intraday |
| **Pairs Trading** | Engle-Granger cointegration, spread z-score entry/exit | Multi-day |

Custom strategies inherit from `BaseStrategy` and implement `generate_signals()`.

---

## Risk Controls

- **Per-trade loss limit** (default: ₹5,000)
- **Daily loss limit** (default: ₹20,000)
- **Weekly loss limit** (default: ₹50,000)
- **Max drawdown** kill switch (default: 10%)
- **Position sizing**: ATR-based, variance-based, fixed-fraction, Kelly
- **Portfolio limits**: Single stock cap, sector cap, correlation cap
- **Tripwires**: Consecutive rejects, latency, feed timeout, exception count

See [RISK_POLICY.md](RISK_POLICY.md) for details.

---

## Testing

```bash
make test           # Run all tests (≤ 5 min)
make lint           # Ruff + type checks
make format         # Black formatter
```

---

## Tech Stack

- **Python 3.11+** with type hints throughout
- **pandas / numpy / scipy** — data & math
- **scikit-learn / XGBoost / LightGBM** — ML models
- **PyTorch** (optional) — LSTM/GRU deep models
- **Parquet** — time series storage
- **DuckDB / SQLite** — metadata store
- **Pydantic** — config validation
- **Typer + Rich** — CLI interface
- **matplotlib / plotly** — charting

---

## Important Notes

1. **Paper trading is the default.** Live trading requires `--confirm-live true`.
2. **All timestamps are IST** (Asia/Kolkata). NSE trading hours: 9:15–15:30.
3. **Transaction costs** model Indian equity charges: brokerage, STT, GST, stamp duty, SEBI fees.
4. **No data is included.** Supply your own bhavcopy / intraday CSVs.
5. **Broker APIs require credentials** in `.env`. See `.env.example`.

---

## License

MIT — See LICENSE file.

## Contributing

PRs welcome. Please run `make lint && make test` before submitting.
