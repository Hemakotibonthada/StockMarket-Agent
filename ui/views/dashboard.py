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
from plotly.subplots import make_subplots

from ui.utils import (
    COLORS, generate_synthetic_data, generate_multi_symbol_data,
    compute_base_features, run_backtest_from_ui,
    create_equity_curve, create_returns_distribution,
    create_monthly_returns_heatmap, create_trade_scatter,
    create_cumulative_pnl, metric_card_html,
)
from src.backtest.metrics import compute_returns, compute_drawdown


def render():
    st.markdown("## 📊 Dashboard")
    st.markdown(
        f'<p style="color: {COLORS["text_muted"]}; margin-top: -10px;">Portfolio overview and market snapshot</p>',
        unsafe_allow_html=True,
    )

    # ── Run a quick backtest if no cached result ───────────────────────────
    if "dashboard_result" not in st.session_state:
        with st.spinner("Running initial backtest on synthetic data..."):
            df = generate_synthetic_data(n_bars=500, seed=42)
            result = run_backtest_from_ui(
                df, "Mean Reversion",
                {"zscore_entry": 2.0, "zscore_exit": 0.5},
            )
            st.session_state["dashboard_result"] = result
            st.session_state["dashboard_df"] = df

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
        fig = create_equity_curve(result.equity_curve, dd, title="Portfolio Equity Curve")
        st.plotly_chart(fig, width='stretch')

    with col_right:
        # Returns Distribution
        rets = compute_returns(result.equity_curve)
        fig_dist = create_returns_distribution(rets, title="Daily Returns")
        st.plotly_chart(fig_dist, width='stretch')

    # ── Row 3: Monthly Heatmap + Trade Scatter ─────────────────────────────
    col_hm, col_sc = st.columns(2)
    with col_hm:
        fig_hm = create_monthly_returns_heatmap(result.equity_curve)
        st.plotly_chart(fig_hm, width='stretch')

    with col_sc:
        fig_scatter = create_trade_scatter(result.trades, title="Trade P&L Distribution")
        st.plotly_chart(fig_scatter, width='stretch')

    # ── Row 4: Market Watchlist ────────────────────────────────────────────
    st.markdown("### 🏪 Market Watchlist")
    multi_data = generate_multi_symbol_data(n_bars=60, seed=42)

    watchlist_cols = st.columns(len(multi_data))
    for i, (sym, df) in enumerate(multi_data.items()):
        with watchlist_cols[i]:
            last_close = df["close"].iloc[-1]
            prev_close = df["close"].iloc[-2]
            change_pct = (last_close - prev_close) / prev_close * 100

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {COLORS['bg_card']}, {COLORS['bg_card_alt']});
                    border: 1px solid {COLORS['grid']};
                    border-radius: 12px;
                    padding: 16px;
                    text-align: center;
                ">
                    <div style="color: {COLORS['text_muted']}; font-size: 0.8rem;">{sym}</div>
                    <div style="color: {COLORS['text']}; font-size: 1.3rem; font-weight: 700;">
                        ₹{last_close:,.1f}
                    </div>
                    <div style="color: {COLORS['success'] if change_pct > 0 else COLORS['danger']};
                                font-size: 0.85rem;">
                        {'▲' if change_pct > 0 else '▼'} {abs(change_pct):.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Mini sparklines ────────────────────────────────────────────────────
    st.markdown("### 📈 30-Day Sparklines")
    spark_cols = st.columns(len(multi_data))
    for i, (sym, df) in enumerate(multi_data.items()):
        with spark_cols[i]:
            recent = df.tail(30)
            color = COLORS["success"] if recent["close"].iloc[-1] > recent["close"].iloc[0] else COLORS["danger"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=recent["close"].values,
                mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=color.replace(")", ", 0.1)").replace("rgb", "rgba") if "rgb" in color else f"rgba(16,185,129,0.1)" if color == COLORS["success"] else "rgba(239,68,68,0.1)",
            ))
            fig.update_layout(
                height=100, margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                showlegend=False,
            )
            st.plotly_chart(fig, width='stretch')

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
