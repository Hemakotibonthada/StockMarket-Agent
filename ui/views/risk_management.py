"""Risk Management page – position sizing, risk limits, and portfolio risk analysis."""

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
    COLORS, generate_synthetic_data, run_backtest_from_ui,
    create_equity_curve,
)
from src.risk.sizing import atr_position_size, fixed_fraction_size, kelly_fraction, variance_position_size
from src.risk.limits import RiskLimiter, RiskState
from src.core.config import RiskConfig
from src.backtest.metrics import compute_returns, compute_drawdown


def render():
    st.markdown("## 🛡️ Risk Management")
    st.markdown(
        f'<p style="color: {COLORS["text_muted"]}; margin-top: -10px;">'
        'Position sizing calculators, risk limits, and portfolio risk analytics</p>',
        unsafe_allow_html=True,
    )

    tab_sizing, tab_limits, tab_analysis, tab_monte = st.tabs([
        "📐 Position Sizing", "🚦 Risk Limits", "📊 Risk Analysis", "🎲 Monte Carlo",
    ])

    # ═══════════════════════════════════════════════════════════════════════
    #  Position Sizing Calculator
    # ═══════════════════════════════════════════════════════════════════════
    with tab_sizing:
        st.markdown("### Position Size Calculator")

        method = st.selectbox("Sizing Method", [
            "ATR-Based", "Fixed Fraction (Risk/Reward)", "Kelly Criterion", "Volatility-Based",
        ])

        col_input, col_result = st.columns([1, 1])

        with col_input:
            capital = st.number_input("Capital (₹)", 100_000, 100_000_000, 1_000_000, step=100_000, key="rs_cap")
            risk_pct = st.slider("Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1, key="rs_risk")

            if method == "ATR-Based":
                atr_value = st.number_input("ATR Value", 1.0, 500.0, 25.0, step=1.0)
                atr_mult = st.slider("ATR Multiplier", 0.5, 5.0, 2.0, 0.1)
                lot_size = st.number_input("Lot Size", 1, 1000, 1)
                max_pos = st.number_input("Max Position Value (₹)", 0, 50_000_000, 0,
                                          help="0 = no limit")

                shares = atr_position_size(
                    capital, risk_pct, atr_value, atr_mult,
                    lot_size=lot_size,
                    max_position_value=max_pos if max_pos > 0 else None,
                )

            elif method == "Fixed Fraction (Risk/Reward)":
                entry_price = st.number_input("Entry Price (₹)", 1.0, 100_000.0, 2500.0, step=10.0)
                stop_loss = st.number_input("Stop Loss (₹)", 1.0, 100_000.0, 2450.0, step=10.0)
                lot_size = st.number_input("Lot Size", 1, 1000, 1, key="ff_lot")

                shares = fixed_fraction_size(capital, risk_pct, entry_price, stop_loss, lot_size=lot_size)

            elif method == "Kelly Criterion":
                win_rate = st.slider("Win Rate (%)", 10.0, 90.0, 55.0, 1.0) / 100
                avg_win = st.number_input("Avg Win (₹)", 100.0, 1_000_000.0, 5000.0)
                avg_loss_val = st.number_input("Avg Loss (₹)", 100.0, 1_000_000.0, 3000.0)

                kelly = kelly_fraction(win_rate, avg_win, avg_loss_val)
                shares = int(capital * kelly / 2500)  # Assume ₹2500 stock

            else:  # Volatility-Based
                volatility = st.slider("Annual Volatility (%)", 5.0, 100.0, 25.0, 1.0) / 100
                price = st.number_input("Stock Price (₹)", 1.0, 100_000.0, 2500.0, step=10.0, key="vb_price")
                lot_size = st.number_input("Lot Size", 1, 1000, 1, key="vb_lot")

                shares = variance_position_size(capital, risk_pct, volatility, price, lot_size=lot_size)

        with col_result:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {COLORS['bg_card']}, {COLORS['bg_card_alt']});
                    border: 1px solid {COLORS['primary']};
                    border-radius: 16px;
                    padding: 32px;
                    text-align: center;
                    margin-top: 20px;
                ">
                    <div style="color: {COLORS['text_muted']}; font-size: 1rem; margin-bottom: 8px;">
                        Recommended Position Size
                    </div>
                    <div style="color: {COLORS['primary']}; font-size: 3rem; font-weight: 800;">
                        {shares:,}
                    </div>
                    <div style="color: {COLORS['text_muted']}; font-size: 0.9rem;">shares</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if method == "ATR-Based":
                position_val = shares * (capital / 100)  # approximate
                risk_amount = capital * risk_pct / 100
                st.markdown(f"**Risk Amount:** ₹{risk_amount:,.0f}")
                st.markdown(f"**Stop Distance:** ₹{atr_value * atr_mult:.2f} (ATR × multiplier)")

            elif method == "Fixed Fraction (Risk/Reward)":
                risk_per_share = abs(entry_price - stop_loss)
                position_val = shares * entry_price
                st.markdown(f"**Position Value:** ₹{position_val:,.0f}")
                st.markdown(f"**Risk per Share:** ₹{risk_per_share:.2f}")
                st.markdown(f"**Total Risk:** ₹{risk_per_share * shares:,.0f}")
                rr_ratio = 2.0  # assumed
                st.markdown(f"**% of Capital:** {position_val/capital*100:.1f}%")

            elif method == "Kelly Criterion":
                st.markdown(f"**Kelly Fraction:** {kelly*100:.1f}%")
                st.markdown(f"**Half-Kelly (recommended):** {kelly*50:.1f}%")
                st.markdown(f"**Optimal Bet Size:** ₹{capital * kelly:,.0f}")

            else:
                position_val = shares * price
                st.markdown(f"**Position Value:** ₹{position_val:,.0f}")
                st.markdown(f"**% of Capital:** {position_val/capital*100:.1f}%")

        # Position sizing comparison chart
        st.markdown("---")
        st.markdown("### 📊 Method Comparison")
        methods_compare = {
            "ATR-Based": atr_position_size(capital, risk_pct, 25.0, 2.0),
            "Fixed Fraction": fixed_fraction_size(capital, risk_pct, 2500.0, 2450.0),
            "Kelly (half)": int(capital * kelly_fraction(0.55, 5000, 3000) / 2 / 2500),
            "Volatility": variance_position_size(capital, risk_pct, 0.25, 2500.0),
        }

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            x=list(methods_compare.keys()),
            y=list(methods_compare.values()),
            marker_color=[COLORS["primary"], COLORS["info"], COLORS["warning"], COLORS["secondary"]],
            text=[f"{v:,}" for v in methods_compare.values()],
            textposition="auto",
        ))
        fig_compare.update_layout(
            title="Position Size by Method (shares)",
            yaxis_title="Shares",
            template=COLORS["plotly_template"], height=350,
            paper_bgcolor=COLORS["bg_dark"], plot_bgcolor=COLORS["bg_dark"],
            font=dict(color=COLORS["text"]),
        )
        st.plotly_chart(fig_compare, width='stretch')

    # ═══════════════════════════════════════════════════════════════════════
    #  Risk Limits Panel
    # ═══════════════════════════════════════════════════════════════════════
    with tab_limits:
        st.markdown("### 🚦 Risk Limit Configuration")

        lc1, lc2 = st.columns(2)
        with lc1:
            st.markdown("**Per-Trade Limits**")
            risk_per_trade = st.slider("Max Risk per Trade (%)", 0.1, 5.0, 0.5, 0.1)
            max_pos_value = st.number_input("Max Position Value (₹)", 0, 50_000_000, 5_000_000, step=500_000)

        with lc2:
            st.markdown("**Portfolio Limits**")
            daily_max = st.slider("Daily Max Loss (%)", 0.5, 10.0, 1.0, 0.5)
            weekly_max = st.slider("Weekly Max Loss (%)", 1.0, 20.0, 2.0, 0.5)
            max_dd = st.slider("Max Drawdown Kill Switch (%)", 1.0, 30.0, 5.0, 0.5)

        risk_config = RiskConfig(
            risk_per_trade_pct=risk_per_trade,
            daily_max_loss_pct=daily_max,
            weekly_max_loss_pct=weekly_max,
            strategy_max_dd_pct=max_dd,
            initial_capital=1_000_000,
        )

        # Simulate risk limiter with trades
        limiter = RiskLimiter(config=risk_config)
        limiter.initialize(equity=1_000_000)

        # Show current state
        summary = limiter.summary()
        st.markdown("---")
        st.markdown("### Current Risk State")

        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            st.metric("Daily P&L", f"₹{summary['daily_pnl']:,.0f}")
        with rc2:
            st.metric("Weekly P&L", f"₹{summary['weekly_pnl']:,.0f}")
        with rc3:
            dd = summary.get("current_drawdown_pct", 0)
            st.metric("Current Drawdown", f"{dd:.2f}%", delta_color="inverse")
        with rc4:
            status = "🟢 Active" if not summary.get("is_killed", False) else "🔴 Killed"
            st.metric("Status", status)

        # Limit visualization
        st.markdown("### 📊 Risk Budget Usage")
        budget_data = {
            "Daily Loss": {"used": 0, "limit": daily_max},
            "Weekly Loss": {"used": 0, "limit": weekly_max},
            "Drawdown": {"used": 0, "limit": max_dd},
        }

        for name, data in budget_data.items():
            pct_used = min(data["used"] / data["limit"] * 100, 100) if data["limit"] > 0 else 0
            color = COLORS["success"] if pct_used < 50 else COLORS["warning"] if pct_used < 80 else COLORS["danger"]
            st.markdown(
                f"""
                <div style="margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between; color: {COLORS['text']};">
                        <span>{name}</span>
                        <span>{data['used']:.1f}% / {data['limit']:.1f}%</span>
                    </div>
                    <div style="background: {COLORS['grid']}; border-radius: 4px; height: 8px; margin-top: 4px;">
                        <div style="background: {color}; height: 100%; border-radius: 4px; width: {pct_used}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════════════════════
    #  Risk Analysis
    # ═══════════════════════════════════════════════════════════════════════
    with tab_analysis:
        st.markdown("### 📊 Portfolio Risk Analytics")

        if "backtest_result" in st.session_state:
            result = st.session_state["backtest_result"]
        else:
            df = generate_synthetic_data(n_bars=500, seed=42)
            result = run_backtest_from_ui(df, "Mean Reversion", {"zscore_entry": 2.0, "zscore_exit": 0.5})

        rets = compute_returns(result.equity_curve)
        dd = compute_drawdown(result.equity_curve)

        # VaR/ES visualization
        fig_var = go.Figure()
        sorted_rets = rets.dropna().sort_values() * 100
        fig_var.add_trace(go.Histogram(
            x=sorted_rets, nbinsx=50,
            marker_color=COLORS["primary"], opacity=0.6,
            name="Returns",
        ))

        var_95 = np.percentile(sorted_rets.dropna(), 5)
        fig_var.add_vline(x=var_95, line_dash="dash", line_color=COLORS["danger"],
                          annotation_text=f"VaR 95%: {var_95:.2f}%")

        es_95 = sorted_rets[sorted_rets <= var_95].mean()
        fig_var.add_vline(x=es_95, line_dash="dot", line_color=COLORS["warning"],
                          annotation_text=f"ES 95%: {es_95:.2f}%")

        fig_var.update_layout(
            title="Value at Risk & Expected Shortfall",
            xaxis_title="Daily Return (%)",
            template=COLORS["plotly_template"], height=400,
            paper_bgcolor=COLORS["bg_dark"], plot_bgcolor=COLORS["bg_dark"],
            font=dict(color=COLORS["text"]),
        )
        st.plotly_chart(fig_var, width='stretch')

        # Drawdown underwater chart
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            y=dd.values * 100,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.2)",
            line=dict(color=COLORS["danger"], width=1.5),
        ))
        fig_dd.update_layout(
            title="Drawdown Underwater Chart",
            yaxis_title="Drawdown %",
            template=COLORS["plotly_template"], height=350,
            paper_bgcolor=COLORS["bg_dark"], plot_bgcolor=COLORS["bg_dark"],
            font=dict(color=COLORS["text"]),
        )
        st.plotly_chart(fig_dd, width='stretch')

        # Rolling metrics
        st.markdown("### 📈 Rolling Risk Metrics")
        window = st.slider("Rolling Window", 10, 100, 30, key="risk_window")

        rolling_vol = rets.rolling(window).std() * np.sqrt(252) * 100
        rolling_sharpe = (rets.rolling(window).mean() * 252) / (rets.rolling(window).std() * np.sqrt(252))

        fig_rolling = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                     subplot_titles=["Rolling Volatility (%)", "Rolling Sharpe Ratio"],
                                     vertical_spacing=0.08)
        fig_rolling.add_trace(go.Scatter(
            y=rolling_vol, mode="lines",
            line=dict(color=COLORS["secondary"], width=1.5),
            fill="tozeroy", fillcolor="rgba(236,72,153,0.1)",
        ), row=1, col=1)
        fig_rolling.add_trace(go.Scatter(
            y=rolling_sharpe, mode="lines",
            line=dict(color=COLORS["info"], width=1.5),
        ), row=2, col=1)
        fig_rolling.add_hline(y=1, line_dash="dot", line_color=COLORS["success"], row=2, col=1)
        fig_rolling.add_hline(y=0, line_dash="solid", line_color=COLORS["text_muted"], row=2, col=1, opacity=0.3)

        fig_rolling.update_layout(
            height=500, template=COLORS["plotly_template"], showlegend=False,
            paper_bgcolor=COLORS["bg_dark"], plot_bgcolor=COLORS["bg_dark"],
            font=dict(color=COLORS["text"]),
        )
        st.plotly_chart(fig_rolling, width='stretch')

    # ═══════════════════════════════════════════════════════════════════════
    #  Monte Carlo Simulation
    # ═══════════════════════════════════════════════════════════════════════
    with tab_monte:
        st.markdown("### 🎲 Monte Carlo Simulation")
        st.markdown("Reshuffle trade outcomes to estimate outcome distribution.")

        if "backtest_result" in st.session_state:
            result = st.session_state["backtest_result"]
        else:
            df = generate_synthetic_data(n_bars=500, seed=42)
            result = run_backtest_from_ui(df, "Mean Reversion", {"zscore_entry": 2.0, "zscore_exit": 0.5})

        mc1, mc2 = st.columns(2)
        with mc1:
            n_sims = st.slider("Number of Simulations", 100, 5000, 1000, 100)
        with mc2:
            mc_seed = st.number_input("Seed", 1, 9999, 42, key="mc_seed")

        if st.button("🎲 Run Monte Carlo", width='stretch'):
            with st.spinner("Running simulations..."):
                trade_pnls = [t.net_pnl for t in result.trades]
                if not trade_pnls:
                    st.warning("No trades available for simulation.")
                else:
                    rng = np.random.default_rng(mc_seed)
                    final_pnls = []
                    equity_paths = []

                    for _ in range(n_sims):
                        shuffled = rng.permutation(trade_pnls)
                        cum = np.cumsum(shuffled) + 1_000_000
                        equity_paths.append(cum)
                        final_pnls.append(cum[-1])

                    # Plot paths
                    fig_mc = go.Figure()
                    for i, path in enumerate(equity_paths[:100]):
                        fig_mc.add_trace(go.Scatter(
                            y=path, mode="lines",
                            line=dict(width=0.5, color=COLORS["primary"]),
                            opacity=0.15, showlegend=False,
                        ))

                    # Percentile lines
                    paths_arr = np.array(equity_paths)
                    for pct, color, label in [
                        (5, COLORS["danger"], "P5"),
                        (50, COLORS["warning"], "Median"),
                        (95, COLORS["success"], "P95"),
                    ]:
                        pct_line = np.percentile(paths_arr, pct, axis=0)
                        fig_mc.add_trace(go.Scatter(
                            y=pct_line, mode="lines",
                            line=dict(width=2.5, color=color),
                            name=label,
                        ))

                    fig_mc.update_layout(
                        title=f"Monte Carlo Equity Paths ({n_sims} simulations)",
                        yaxis_title="Portfolio Value (₹)",
                        template=COLORS["plotly_template"], height=500,
                        paper_bgcolor=COLORS["bg_dark"], plot_bgcolor=COLORS["bg_dark"],
                        font=dict(color=COLORS["text"]),
                    )
                    st.plotly_chart(fig_mc, width='stretch')

                    # Final P&L distribution
                    fig_final = go.Figure()
                    fig_final.add_trace(go.Histogram(
                        x=final_pnls, nbinsx=50,
                        marker_color=COLORS["primary"], opacity=0.7,
                    ))
                    fig_final.update_layout(
                        title="Final Portfolio Value Distribution",
                        xaxis_title="Final Value (₹)",
                        template=COLORS["plotly_template"], height=350,
                        paper_bgcolor=COLORS["bg_dark"], plot_bgcolor=COLORS["bg_dark"],
                        font=dict(color=COLORS["text"]),
                    )
                    st.plotly_chart(fig_final, width='stretch')

                    # Stats
                    mc_s1, mc_s2, mc_s3, mc_s4, mc_s5 = st.columns(5)
                    with mc_s1:
                        st.metric("P5 (Worst)", f"₹{np.percentile(final_pnls, 5):,.0f}")
                    with mc_s2:
                        st.metric("P25", f"₹{np.percentile(final_pnls, 25):,.0f}")
                    with mc_s3:
                        st.metric("Median", f"₹{np.percentile(final_pnls, 50):,.0f}")
                    with mc_s4:
                        st.metric("P75", f"₹{np.percentile(final_pnls, 75):,.0f}")
                    with mc_s5:
                        st.metric("P95 (Best)", f"₹{np.percentile(final_pnls, 95):,.0f}")
