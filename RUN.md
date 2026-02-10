# RUN.md — Copy-Paste Commands

Step-by-step instructions to get stock-agent running. Copy and paste each block.

---

## 1. Initial Setup

```bash
# Clone the repository
git clone <repo-url>
cd stock-agent

# Create Python virtual environment (Python 3.11+)
python -m venv .venv

# Activate the environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -e .

# Install dev dependencies (for testing/linting)
pip install -e ".[dev]"

# Optional: deep learning support
pip install -e ".[deep]"
```

---

## 2. Configure Environment

```bash
# Copy the environment template
cp .env.example .env

# Edit .env with your broker credentials (only needed for live trading)
# ZERODHA_API_KEY=your_key
# ZERODHA_API_SECRET=your_secret
# ZERODHA_ACCESS_TOKEN=your_token
```

---

## 3. Prepare Data

```bash
# Create data directories
mkdir -p data/raw data/processed

# Copy your raw CSV files into data/raw/
# Expected format: date,open,high,low,close,volume columns
# One file per symbol, or bhavcopy format

# Ingest and process data
sa-data ingest --input-dir data/raw --output-dir data/processed

# Verify data
sa-data info data/processed/RELIANCE.parquet

# View trading universe
sa-data universe --config configs/universe_nifty50.yaml
```

---

## 4. Run Backtest

```bash
# Single backtest with ORB Momentum strategy
sa-backtest run --config configs/backtest.yaml

# Walk-forward analysis
sa-backtest walkforward --config configs/backtest.yaml

# Parameter sweep
sa-backtest sweep --config configs/backtest.yaml --param lookback --values 10,15,20,30

# View saved results
sa-backtest report results/backtest_latest.json

# HTML report is auto-generated at results/backtest_report.html
```

---

## 5. Paper Trading

```bash
# Run paper trading with simulated broker
sa-paper run --config configs/paper_trade.yaml

# With speed multiplier (10x faster replay)
sa-paper run --config configs/paper_trade.yaml --speed 10

# Limit to first 1000 bars
sa-paper run --config configs/paper_trade.yaml --max-bars 1000

# Check configuration
sa-paper status --config configs/paper_trade.yaml
```

---

## 6. Live Trading (CAUTION)

```bash
# ⚠ LIVE TRADING WITH REAL MONEY — READ RISK_POLICY.md FIRST

# Prerequisite: Configure broker API in .env
# Prerequisite: Test thoroughly in paper mode

# Start live trading (requires explicit confirmation)
sa-live start --config configs/live_trade.yaml --confirm-live true

# Dry run (connects but doesn't place orders)
sa-live start --config configs/live_trade.yaml --confirm-live true --dry-run

# Check system health
sa-live health --config configs/live_trade.yaml

# Reconcile positions with broker
sa-live reconcile --config configs/live_trade.yaml
```

---

## 7. Train ML Models

```bash
# Train a classic ML model (RandomForest, XGBoost, etc.)
python -m src.training.train --config configs/backtest.yaml --model-type xgboost

# Evaluate a trained model
python -m src.training.eval --config configs/backtest.yaml --model-name xgboost_latest

# List saved models
python -c "from src.models.selection import ModelRegistry; r = ModelRegistry(); print(r.list_models())"
```

---

## 8. Development

```bash
# Run tests
make test
# or: pytest tests/ -v --tb=short

# Lint code
make lint
# or: ruff check src/ tests/

# Format code
make format
# or: black src/ tests/

# Clean build artifacts
make clean
```

---

## 9. Common Workflows

### A. Full Backtest Pipeline
```bash
sa-data ingest --input-dir data/raw --output-dir data/processed
sa-backtest run --config configs/backtest.yaml
# Open results/backtest_report.html in browser
```

### B. Strategy Comparison
```bash
# Backtest with ORB Momentum
sa-backtest run --config configs/backtest.yaml --output-dir results/orb

# Create a mean_reversion config and backtest
sa-backtest run --config configs/paper_trade.yaml --output-dir results/mr

# Compare results
sa-backtest report results/orb/backtest_latest.json
sa-backtest report results/mr/backtest_latest.json
```

### C. Data Resampling
```bash
# Resample 1-min data to 5-min bars
sa-data resample data/processed/RELIANCE.parquet --freq 5min

# Resample to 15-min bars
sa-data resample data/processed/RELIANCE.parquet --freq 15min
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -e .` |
| `No data found` | Run `sa-data ingest` first |
| `kiteconnect not installed` | `pip install kiteconnect` (for Zerodha) |
| `Permission denied` on .env | Check file permissions |
| Tests fail with seed issues | Ensure `set_seed(42)` in test setup |
