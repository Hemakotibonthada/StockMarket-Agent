# COMPLIANCE.md — Regulatory & Exchange Rules

## Scope

This document outlines the regulatory and exchange compliance considerations for
automated trading on Indian stock exchanges (NSE/BSE).

> **Disclaimer**: This is NOT legal advice. Consult a SEBI-registered advisor
> and your broker for current regulations.

---

## SEBI Regulations

### Algorithmic Trading (Algo Trading)

1. **SEBI Circular**: SEBI/HO/MRD2/PoD-1/P/CIR/2024/171 and related circulars
   govern algorithmic trading in India.

2. **Registration**: Retail algo trading through broker APIs typically falls under
   the broker's algo framework. Check with your broker whether your strategy
   requires separate registration.

3. **Order-to-Trade Ratio (OTR)**: Exchanges impose OTR limits. Excessive order
   modifications/cancellations relative to trades can trigger penalties.
   - This software includes rate limiting in the OrderRouter.
   - Default: max 10 orders/second, 100 orders/minute.

4. **Co-location**: This software is designed for retail use and does NOT use
   co-location services.

### Risk Management

SEBI mandates risk controls for all trading systems:
- **Pre-trade risk checks**: Implemented via `RiskLimiter`
- **Position limits**: Implemented via `PortfolioRisk`
- **Kill switch**: Implemented via daily/weekly loss limits with auto-halt

---

## Exchange Rules (NSE)

### Trading Hours
- **Pre-open session**: 9:00–9:15 IST
- **Normal trading**: 9:15–15:30 IST
- **Closing session**: 15:40–16:00 IST
- This software only trades during normal hours (9:15–15:30).

### Circuit Breakers
- NSE implements index-level and stock-level circuit breakers.
- If a circuit breaker is hit, the software's health monitor will detect
  feed timeout and halt operations.

### Transaction Charges
- **STT (Securities Transaction Tax)**: 0.025% (delivery buy+sell), 0.0125% (intraday sell)
- **Exchange Transaction Charges**: ~0.00345% (NSE)
- **GST**: 18% on brokerage + transaction charges
- **SEBI Turnover Fee**: 0.0001%
- **Stamp Duty**: Varies by state (typically 0.003%–0.015%)

All of these are modeled in `src/backtest/costs.py`.

---

## Broker-Specific Rules

### Zerodha
- API rate limit: 3 requests/second for order placement
- Access tokens expire daily — must re-authenticate each morning
- Margin requirements per SPAN + exposure framework
- Intraday positions auto-squared off at 3:20 PM

### Upstox
- Similar API rate limits
- OAuth2 token-based authentication

---

## Software Controls

This software implements the following compliance-relevant controls:

| Control | Implementation | Location |
|---------|---------------|----------|
| Rate limiting | Max orders/sec, orders/min | `src/exec/router.py` |
| Pre-trade risk | Loss limits per trade/day/week | `src/risk/limits.py` |
| Position limits | Single stock and sector caps | `src/risk/portfolio.py` |
| Kill switch | Auto-halt on max drawdown | `src/risk/limits.py` |
| Price protection | Reject orders > threshold from LTP | `src/exec/router.py` |
| Audit trail | JSONL logs of all orders/signals | `src/core/logging_utils.py` |
| Live trade gate | Requires `--confirm-live true` | `src/cli/live_cli.py` |
| Tripwire monitors | Auto-halt on anomalies | `src/risk/tripwires.py` |
| Reconciliation | Position/cash verify with broker | `src/exec/reconcile.py` |

---

## Data Privacy

- All data is stored **locally** (local-first design).
- No telemetry or external data transmission.
- Broker API credentials stored in `.env` (gitignored).
- Audit logs contain trade details — secure appropriately.

---

## Tax Considerations

- **Intraday**: Profits taxed as speculative business income.
- **Short-term (< 12 months)**: 15% STCG on equities.
- **Long-term (> 12 months)**: 10% LTCG above ₹1 lakh on equities.
- **F&O**: Treated as non-speculative business income.

Maintain audit logs for tax filing purposes. This software generates JSONL
audit logs that can be used for record-keeping.

---

## Acknowledgment

By using this software for live trading, you acknowledge that:
1. You understand the risks of algorithmic trading
2. You have read and understood applicable SEBI regulations
3. You will comply with your broker's terms of service
4. You accept full responsibility for all trading activity
5. The software authors accept NO LIABILITY for trading losses
