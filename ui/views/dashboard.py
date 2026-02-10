"""Dashboard page – portfolio overview with key metrics, equity curve, and market snapshot."""

from __future__ import annotations

import sys
from pathlib import Path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from ui.utils import (
    COLORS, load_real_data, run_backtest_from_ui,
    metric_card_html,
    create_equity_curve, create_returns_distribution,
    create_monthly_returns_heatmap, create_trade_scatter,
    create_cumulative_pnl,
)
from src.data.market_data import (
    fetch_watchlist_quotes, fetch_multiple_stocks,
    get_default_watchlist, INDIAN_STOCKS,
)
from src.backtest.metrics import compute_returns, compute_drawdown


def render():
    st.markdown("## 📊 Dashboard")
    st.markdown(
        f'<p style="color: {COLORS["text_muted"]}; margin-top: -10px;">Portfolio overview &amp; live market snapshot</p>',
        unsafe_allow_html=True,
    )

    # ── Watchlist config ───────────────────────────────────────────────────
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = get_default_watchlist()

    # ── Run a backtest on real data for the primary symbol ─────────────────
    primary_symbol = st.session_state["watchlist"][0] if st.session_state["watchlist"] else "RELIANCE"

    if "dashboard_result" not in st.session_state:
        with st.spinner(f"Loading real market data for {primary_symbol}..."):
            df = load_real_data(primary_symbol, period="2y")
            result = run_backtest_from_ui(
                df, "Mean Reversion",
                {"zscore_entry": 2.0, "zscore_exit": 0.5},
            )
            st.session_state["dashboard_result"] = result
            st.session_state["dashboard_df"] = df
            st.session_state["dashboard_symbol"] = primary_symbol

    result = st.session_state["dashboard_result"]
    metrics = result.metrics

    # ── Top KPI Row ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        final_eq = result.equity_curve.iloc[-1] if len(result.equity_curve) > 0 else 1_000_000
        st.metric("Portfolio Value", f"₹{final_eq:,.0f}",
                  delta=f"{metrics.total_return_pct:+.2f}%")
    with c2:
        st.metric("Total Return", f"{metrics.total_return_pct:.2f}%",
                  delta=f"CAGR {metrics.cagr_pct:.1f}%")
    with c3:
        st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}",
                  delta="above 1.0" if metrics.sharpe_ratio > 1 else "below 1.0")
    with c4:
        st.metric("Max Drawdown", f"{metrics.max_drawdown_pct:.1f}%",
                  delta=f"{metrics.max_drawdown_pct:.1f}%", delta_color="inverse")
    with c5:
        st.metric("Win Rate", f"{metrics.win_rate_pct:.0f}%",
                  delta=f"{metrics.total_trades} trades")
    with c6:
        st.metric("Profit Factor", f"{metrics.profit_factor:.2f}",
                  delta="profitable" if metrics.profit_factor > 1 else "unprofitable")

    st.markdown("---")

    # ── Row 2: Equity Curve + Drawdown ─────────────────────────────────────
    col_left, col_right = st.columns([2, 1])
    with col_left:
        dd = compute_drawdown(result.equity_curve)
        fig = create_equity_curve(result.equity_curve, dd,
                                  title=f"Portfolio Equity – {st.session_state.get('dashboard_symbol', 'N/A')}")
        st.plotly_chart(fig, width='stretch', key="dash_equity")

    with col_right:
        rets = compute_returns(result.equity_curve)
        fig_dist = create_returns_distribution(rets, title="Daily Returns")
        st.plotly_chart(fig_dist, width='stretch', key="dash_dist")

    # ── Row 3: Monthly Heatmap + Trade Scatter ─────────────────────────────
    col_hm, col_sc = st.columns(2)
    with col_hm:
        fig_hm = create_monthly_returns_heatmap(result.equity_curve)
        st.plotly_chart(fig_hm, width='stretch', key="dash_hm")

    with col_sc:
        fig_scatter = create_trade_scatter(result.trades, title="Trade P&L Distribution")
        st.plotly_chart(fig_scatter, width='stretch', key="dash_scatter")

    # ── Row 4: Live Market Watchlist ───────────────────────────────────────
    st.markdown("### 🏪 Live Market Watchlist")

    wl_col1, wl_col2 = st.columns([3, 1])
    with wl_col1:
        new_sym = st.selectbox(
            "Add stock to watchlist",
            options=[s for s in sorted(INDIAN_STOCKS.keys()) if s not in st.session_state["watchlist"]],
            index=None,
            placeholder="Select a stock to add...",
            key="wl_add_select",
        )
    with wl_col2:
        st.markdown("")
        st.markdown("")
        if st.button("➕ Add", key="wl_add_btn") and new_sym:
            st.session_state["watchlist"].append(new_sym)
            st.rerun()

    watchlist = st.session_state["watchlist"]

    with st.spinner("Fetching live quotes..."):
        quotes = fetch_watchlist_quotes(watchlist)

    if quotes:
        watchlist_cols = st.columns(min(len(quotes), 5))
        for i, q in enumerate(quotes):
            with watchlist_cols[i % 5]:
                change_pct = q.get("change_pct", 0)
                price = q.get("price", 0)
                symbol = q.get("symbol", "")

                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, {COLORS['bg_card']}, {COLORS['bg_card_alt']});
                        border: 1px solid {COLORS['grid']};
                        border-radius: 12px;
                        padding: 16px;
                        text-align: center;
                    ">
                        <div style="color: {COLORS['text_muted']}; font-size: 0.8rem;">{symbol}</div>
                        <div style="color: {COLORS['text']}; font-size: 1.3rem; font-weight: 700;">
                            ₹{price:,.1f}
                        </div>
                        <div style="color: {COLORS['success'] if change_pct > 0 else COLORS['danger']};
                                    font-size: 0.85rem;">
                            {'▲' if change_pct > 0 else '▼'} {abs(change_pct):.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button("✕", key=f"wl_rm_{symbol}", help=f"Remove {symbol}"):
                    st.session_state["watchlist"].remove(symbol)
                    st.rerun()
    else:
        st.info("Unable to fetch live quotes. Check your internet connection.")

    # ── Mini sparklines (real data) ────────────────────────────────────────
    st.markdown("### 📈 30-Day Price Trends")
    with st.spinner("Loading price history..."):
        multi_data = fetch_multiple_stocks(watchlist[:5], period="1mo")

    if multi_data:
        spark_cols = st.columns(len(multi_data))
        for i, (sym, df) in enumerate(multi_data.items()):
            with spark_cols[i]:
                recent = df.tail(30)
                if len(recent) < 2:
                    st.caption(f"{sym}: insufficient data")
                    continue
                is_positive = recent["close"].iloc[-1] > recent["close"].iloc[0]
                color = COLORS["success"] if is_positive else COLORS["danger"]
                fill_color = "rgba(52,211,153,0.1)" if is_positive else "rgba(248,113,113,0.1)"
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=recent["close"].values,
                    mode="lines",
                    line=dict(color=color, width=2),
                    fill="tozeroy",
                    fillcolor=fill_color,
                ))
                fig.update_layout(
                    height=100, margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    showlegend=False,
                )
                st.plotly_chart(fig, width='stretch', key=f"spark_{sym}")
                pct = (recent["close"].iloc[-1] / recent["close"].iloc[0] - 1) * 100
                st.caption(f"{sym}: {pct:+.1f}%")

    # ── Performance Summary Table ──────────────────────────────────────────
    st.markdown("### 📋 Detailed Metrics")
    left_m, right_m = st.columns(2)

    with left_m:
        st.markdown("**Returns & Risk**")
        perf_data = {
            "Total Return": f"{metrics.total_return_pct:.2f}%",
            "CAGR": f"{metrics.cagr_pct:.2f}%",
            "Sharpe Ratio": f"{metrics.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{metrics.sortino_ratio:.2f}",
            "Calmar Ratio": f"{metrics.calmar_ratio:.2f}",
            "Max Drawdown": f"{metrics.max_drawdown_pct:.2f}%",
            "VaR (95%)": f"{metrics.var_95_pct:.2f}%",
            "Expected Shortfall": f"{metrics.es_95_pct:.2f}%",
        }
        st.dataframe(pd.DataFrame(perf_data.items(), columns=["Metric", "Value"]),
                      width='stretch', hide_index=True)

    with right_m:
        st.markdown("**Trade Statistics**")
        trade_data = {
            "Total Trades": str(metrics.total_trades),
            "Win Rate": f"{metrics.win_rate_pct:.1f}%",
            "Profit Factor": f"{metrics.profit_factor:.2f}",
            "Avg Trade Return": f"{metrics.avg_trade_return_pct:.2f}%",
            "Max Consecutive Losses": str(metrics.max_consecutive_losses),
            "Avg Holding Period": f"{metrics.avg_holding_bars:.0f} bars",
            "Total Costs": f"₹{result.total_costs:,.2f}",
            "Exposure": f"{metrics.exposure_pct:.1f}%",
        }
        st.dataframe(pd.DataFrame(trade_data.items(), columns=["Metric", "Value"]),
                      width='stretch', hide_index=True)

    # ── Refresh ────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔄 Refresh Dashboard Data", width='stretch'):
        for k in ["dashboard_result", "dashboard_df", "dashboard_symbol"]:
            if k in st.session_state:
                del st.session_state[k]
        st.cache_data.clear()
        st.rerun()
