# RISK_POLICY.md — Risk Management Policy

## Overview

This document describes the default risk controls and how to customize them.
All risk parameters are configurable via YAML config files.

> **Golden Rule**: The system is designed to **prevent catastrophic losses**.
> Individual trade P&L will vary, but risk controls ensure survival.

---

## Risk Hierarchy

```
Level 1: Per-Trade Risk
    └─ Max loss per trade
    └─ Stop-loss (ATR-based or fixed)
    └─ Position sizing (ATR / variance / Kelly)

Level 2: Daily Risk
    └─ Max daily loss limit
    └─ Max trades per day
    └─ EOD position squareoff

Level 3: Weekly Risk
    └─ Max weekly loss limit
    └─ Auto-reset on week boundaries

Level 4: Portfolio Risk
    └─ Max drawdown kill switch
    └─ Single stock concentration limit
    └─ Sector concentration limit
    └─ Correlation-based limits

Level 5: Operational Risk (Tripwires)
    └─ Consecutive order rejects
    └─ Feed latency timeout
    └─ Exception count threshold
    └─ Slippage deviation alerts
```

---

## Default Limits

| Parameter | Default | Config Key | Description |
|-----------|---------|-----------|-------------|
| Max loss per trade | ₹5,000 | `risk.max_loss_per_trade` | Hard stop on any single trade |
| Max daily loss | ₹20,000 | `risk.max_daily_loss` | Halt trading for the day |
| Max weekly loss | ₹50,000 | `risk.max_weekly_loss` | Halt trading for the week |
| Max drawdown | 10% | `risk.max_drawdown` | Kill switch — halts all trading |
| Max position (% of capital) | 20% | `risk.max_position_pct` | Per-stock allocation cap |
| Max sector (% of capital) | 40% | `risk.max_sector_pct` | Per-sector allocation cap |
| Max correlation | 0.7 | `risk.max_correlation` | Correlation diversification cap |

---

## Position Sizing Methods

### 1. ATR-Based (Default)
```
position_size = (risk_per_trade) / (ATR × multiplier)
```
- Risks a fixed rupee amount per trade
- Adjusts position size inversely with volatility
- Config: `risk.sizing_method: atr`

### 2. Variance-Based
```
position_size = (risk_per_trade) / (price × realized_vol × multiplier)
```
- Uses realized volatility for sizing
- Config: `risk.sizing_method: variance`

### 3. Fixed Fraction
```
position_size = (capital × fraction) / price
```
- Simple percentage of capital
- Config: `risk.sizing_method: fixed_fraction`

### 4. Kelly Criterion
```
kelly_fraction = (win_prob × avg_win - (1 - win_prob) × avg_loss) / avg_win
position_size = capital × kelly_fraction × kelly_multiplier
```
- Optimal growth rate sizing
- Typically use half-Kelly (multiplier = 0.5)
- Config: `risk.sizing_method: kelly`

---

## Tripwire Monitors

Tripwires are **operational safety checks** that detect system anomalies:

| Tripwire | Default Threshold | Action |
|----------|------------------|--------|
| Consecutive order rejects | 5 | Halt trading |
| Feed latency | 60 seconds | Warning, then halt |
| Exceptions per session | 10 | Halt trading |
| Slippage deviation | 3 std deviations | Warning |

### Configuration
```yaml
tripwire:
  max_consecutive_rejects: 5
  feed_timeout_seconds: 60
  max_exceptions: 10
  slippage_std_threshold: 3.0
```

---

## Kill Switch Behavior

When a risk limit is breached:

1. **All pending orders are cancelled**
2. **No new orders are placed**
3. **Open positions are NOT auto-liquidated** (manual intervention required)
4. **An alert is logged** (CRITICAL level)
5. **The audit log records the breach**

To resume trading after a kill switch:
1. Manually assess the situation
2. Restart the application
3. Limits auto-reset at day/week boundaries

---

## Customizing Risk Limits

### For Conservative Trading
```yaml
risk:
  max_loss_per_trade: 2000
  max_daily_loss: 10000
  max_weekly_loss: 25000
  max_drawdown: 0.05
  max_position_pct: 0.10
```

### For Aggressive Trading
```yaml
risk:
  max_loss_per_trade: 10000
  max_daily_loss: 50000
  max_weekly_loss: 100000
  max_drawdown: 0.15
  max_position_pct: 0.25
```

### For Paper Trading (Relaxed)
```yaml
risk:
  max_loss_per_trade: 50000
  max_daily_loss: 200000
  max_weekly_loss: 500000
  max_drawdown: 0.30
```

---

## Transaction Cost Model

The backtest and paper trading engines model Indian equity costs:

| Component | Rate | Applied On |
|-----------|------|-----------|
| Brokerage | 3 bps (₹20 cap) | Both sides |
| STT | 2.5 bps | Sell side (intraday) |
| GST | 18% of brokerage | Both sides |
| Stamp Duty | 1.5 bps | Buy side |
| SEBI Fees | 0.1 bps | Both sides |
| Slippage | Random 0–5 bps | Per trade |

Total estimated cost: ~10–15 bps round-trip.

---

## Emergency Procedures

### During Live Trading
1. **Press Ctrl+C** to stop the event loop gracefully
2. Check the health dashboard output
3. Log into your broker terminal to verify positions
4. Run `sa-live reconcile` to compare positions

### After Abnormal Exit
1. Check `logs/` directory for error details
2. Check `logs/audit.jsonl` for last recorded orders
3. Run `sa-live reconcile` to detect discrepancies
4. Manually close any orphaned positions via broker terminal

### Contact Points
- Broker support desk for order/position issues
- Exchange helpline for settlement queries
- SEBI SCORES portal for formal complaints

---

## Monitoring Checklist

Daily checks for live trading:

- [ ] Verify API token is fresh (Zerodha tokens expire daily)
- [ ] Check system health before market open
- [ ] Monitor feed latency during first 15 minutes
- [ ] Review audit log for anomalies at EOD
- [ ] Reconcile positions with broker at EOD
- [ ] Back up audit logs weekly
