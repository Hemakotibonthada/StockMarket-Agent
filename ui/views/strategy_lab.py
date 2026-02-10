"""Strategy Lab page – compare strategies, parameter sweep, walk-forward analysis."""

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
    COLORS, STRATEGY_MAP, generate_synthetic_data,
    run_backtest_from_ui, create_equity_curve,
)
from src.backtest.metrics import compute_returns, compute_drawdown


def render():
    st.markdown("## ⚡ Strategy Lab")
    st.markdown(
        f'<p style="color: {COLORS["text_muted"]}; margin-top: -10px;">'
        'Compare strategies, optimize parameters, and validate robustness</p>',
        unsafe_allow_html=True,
    )

    tab_compare, tab_sweep, tab_wf = st.tabs([
        "⚔️ Strategy Comparison", "🔍 Parameter Sweep", "📅 Walk-Forward",
    ])

    # ═══════════════════════════════════════════════════════════════════════
    #  Strategy Comparison
    # ═══════════════════════════════════════════════════════════════════════
    with tab_compare:
        st.markdown("### Compare Multiple Strategies")

        strategies_to_compare = st.multiselect(
            "Select Strategies",
            list(STRATEGY_MAP.keys()),
            default=["Mean Reversion"],
        )

        comp_c1, comp_c2 = st.columns(2)
        with comp_c1:
            comp_bars = st.slider("Data Length (bars)", 200, 2000, 500, 50, key="comp_bars")
        with comp_c2:
            comp_capital = st.number_input("Capital (₹)", 100_000, 100_000_000, 1_000_000,
                                           step=100_000, key="comp_cap")

        if st.button("🚀 Run Comparison", width='stretch'):
            if not strategies_to_compare:
                st.warning("Please select at least one strategy.")
            else:
                df = generate_synthetic_data(n_bars=comp_bars, seed=42)
                results = {}

                progress = st.progress(0)
                for i, strat_name in enumerate(strategies_to_compare):
                    with st.spinner(f"Running {strat_name}..."):
                        params = STRATEGY_MAP[strat_name]["default_params"].copy()
                        result = run_backtest_from_ui(
                            df, strat_name, params,
                            initial_capital=comp_capital,
                        )
                        results[strat_name] = result
                    progress.progress((i + 1) / len(strategies_to_compare))

                st.session_state["comparison_results"] = results
                progress.empty()
                st.success("Comparison complete!")

        if "comparison_results" in st.session_state:
            results = st.session_state["comparison_results"]

            # Equity curves overlay
            fig_eq = go.Figure()
            colors_list = [COLORS["primary"], COLORS["success"], COLORS["warning"],
                           COLORS["secondary"], COLORS["info"]]

            for i, (name, result) in enumerate(results.items()):
                color = colors_list[i % len(colors_list)]
                fig_eq.add_trace(go.Scatter(
                    y=result.equity_curve.values,
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=2),
                ))

            fig_eq.update_layout(
                title="Equity Curve Comparison",
                yaxis_title="Portfolio Value (₹)",
                template=COLORS["plotly_template"], height=500,
                paper_bgcolor=COLORS["bg_dark"], plot_bgcolor=COLORS["bg_dark"],
                font=dict(color=COLORS["text"]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_eq, width='stretch')

            # Metrics comparison table
            st.markdown("### 📊 Metrics Comparison")
            comparison_data = []
            for name, result in results.items():
                m = result.metrics
                comparison_data.append({
                    "Strategy": name,
                    "Total Return %": f"{m.total_return_pct:.2f}",
                    "CAGR %": f"{m.cagr_pct:.2f}",
                    "Sharpe": f"{m.sharpe_ratio:.2f}",
                    "Sortino": f"{m.sortino_ratio:.2f}",
                    "Max DD %": f"{m.max_drawdown_pct:.1f}",
                    "Win Rate %": f"{m.win_rate_pct:.0f}",
                    "Profit Factor": f"{m.profit_factor:.2f}",
                    "Total Trades": m.total_trades,
                    "VaR 95%": f"{m.var_95_pct:.2f}",
                })
            st.dataframe(pd.DataFrame(comparison_data), width='stretch', hide_index=True)

            # Radar chart
            if len(results) >= 2:
                st.markdown("### 🕸️ Strategy Radar")
                categories = ["Return", "Sharpe", "Win Rate", "Profit Factor", "Low DD"]

                fig_radar = go.Figure()
                for i, (name, result) in enumerate(results.items()):
                    m = result.metrics
                    # Normalize values to 0-100 scale
                    vals = [
                        min(max(m.total_return_pct, -50), 100),
                        min(max(m.sharpe_ratio * 30, 0), 100),
                        m.win_rate_pct,
                        min(m.profit_factor * 25, 100),
                        max(100 - m.max_drawdown_pct * 3, 0),
                    ]
                    color = colors_list[i % len(colors_list)]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=categories + [categories[0]],
                        fill="toself",
                        name=name,
                        fillcolor=color.replace(")", ", 0.1)").replace("rgb", "rgba")
                                        if "rgb" in color else f"{color}1A",
                        line=dict(color=color, width=2),
                    ))

                fig_radar.update_layout(
                    polar=dict(
                        bgcolor=COLORS["bg_dark"],
                        radialaxis=dict(visible=True, range=[0, 100],
                                        gridcolor=COLORS["grid"]),
                        angularaxis=dict(gridcolor=COLORS["grid"]),
                    ),
                    template=COLORS["plotly_template"], height=450,
                    paper_bgcolor=COLORS["bg_dark"],
                    font=dict(color=COLORS["text"]),
                )
                st.plotly_chart(fig_radar, width='stretch')

    # ═══════════════════════════════════════════════════════════════════════
    #  Parameter Sweep
    # ═══════════════════════════════════════════════════════════════════════
    with tab_sweep:
        st.markdown("### 🔍 Parameter Sweep")
        st.markdown("Test how different parameter values affect strategy performance.")

        sweep_strat = st.selectbox("Strategy", list(STRATEGY_MAP.keys()), key="sweep_strat")

        if sweep_strat == "Mean Reversion":
            param_name = st.selectbox("Parameter to Sweep", [
                "zscore_entry", "zscore_exit", "zscore_window", "rsi_period",
            ])
            if param_name == "zscore_entry":
                values = st.slider("Range", 0.5, 4.0, (1.0, 3.0), 0.25, key="sweep_range")
                param_values = np.arange(values[0], values[1] + 0.25, 0.25).tolist()
            elif param_name == "zscore_exit":
                values = st.slider("Range", 0.1, 2.0, (0.2, 1.5), 0.1, key="sweep_range2")
                param_values = np.arange(values[0], values[1] + 0.1, 0.1).tolist()
            elif param_name == "zscore_window":
                values = st.slider("Range", 5, 50, (10, 40), 5, key="sweep_range3")
                param_values = list(range(values[0], values[1] + 5, 5))
            else:
                values = st.slider("Range", 5, 30, (7, 21), 7, key="sweep_range4")
                param_values = list(range(values[0], values[1] + 7, 7))
        else:
            st.info("Parameter sweep is currently optimized for Mean Reversion strategy.")
            param_name = "zscore_entry"
            param_values = [1.0, 1.5, 2.0, 2.5, 3.0]

        sweep_bars = st.slider("Data Length", 200, 1000, 500, 50, key="sweep_bars")

        if st.button("🔍 Run Parameter Sweep", width='stretch'):
            df = generate_synthetic_data(n_bars=sweep_bars, seed=42)
            sweep_results = []

            progress = st.progress(0)
            for i, val in enumerate(param_values):
                params = STRATEGY_MAP[sweep_strat]["default_params"].copy()
                params[param_name] = val

                try:
                    result = run_backtest_from_ui(df, sweep_strat, params)
                    m = result.metrics
                    sweep_results.append({
                        param_name: val,
                        "Total Return %": m.total_return_pct,
                        "Sharpe": m.sharpe_ratio,
                        "Max DD %": m.max_drawdown_pct,
                        "Win Rate %": m.win_rate_pct,
                        "Profit Factor": m.profit_factor,
                        "Trades": m.total_trades,
                    })
                except Exception:
                    sweep_results.append({param_name: val, "Total Return %": 0, "Sharpe": 0,
                                          "Max DD %": 0, "Win Rate %": 0, "Profit Factor": 0, "Trades": 0})
                progress.progress((i + 1) / len(param_values))

            progress.empty()
            sweep_df = pd.DataFrame(sweep_results)
            st.session_state["sweep_results"] = sweep_df

        if "sweep_results" in st.session_state:
            sweep_df = st.session_state["sweep_results"]

            # Results table
            st.dataframe(sweep_df, width='stretch', hide_index=True)

            # Charts
            fig_sweep = make_subplots(
                rows=2, cols=2,
                subplot_titles=["Total Return %", "Sharpe Ratio", "Max Drawdown %", "Win Rate %"],
                vertical_spacing=0.12,
            )

            x_vals = sweep_df[sweep_df.columns[0]]

            fig_sweep.add_trace(go.Bar(x=x_vals, y=sweep_df["Total Return %"],
                                        marker_color=COLORS["primary"]), row=1, col=1)
            fig_sweep.add_trace(go.Bar(x=x_vals, y=sweep_df["Sharpe"],
                                        marker_color=COLORS["info"]), row=1, col=2)
            fig_sweep.add_trace(go.Bar(x=x_vals, y=sweep_df["Max DD %"],
                                        marker_color=COLORS["danger"]), row=2, col=1)
            fig_sweep.add_trace(go.Bar(x=x_vals, y=sweep_df["Win Rate %"],
                                        marker_color=COLORS["success"]), row=2, col=2)

            fig_sweep.update_layout(
                height=600, template=COLORS["plotly_template"], showlegend=False,
                paper_bgcolor=COLORS["bg_dark"], plot_bgcolor=COLORS["bg_dark"],
                font=dict(color=COLORS["text"]),
            )
            st.plotly_chart(fig_sweep, width='stretch')

    # ═══════════════════════════════════════════════════════════════════════
    #  Walk-Forward Analysis
    # ═══════════════════════════════════════════════════════════════════════
    with tab_wf:
        st.markdown("### 📅 Walk-Forward Analysis")
        st.markdown("Split data into train, validate, and test periods to check for overfitting.")

        wf_c1, wf_c2 = st.columns(2)
        with wf_c1:
            total_bars = st.slider("Total Data Bars", 300, 2000, 750, 50, key="wf_bars")
            train_pct = st.slider("Training %", 40, 80, 60, 5)
            val_pct = st.slider("Validation %", 10, 30, 20, 5)
        with wf_c2:
            wf_strat = st.selectbox("Strategy", list(STRATEGY_MAP.keys()), key="wf_strat")
            st.markdown(f"**Test %**: {100 - train_pct - val_pct}%")

        if st.button("📅 Run Walk-Forward", width='stretch'):
            df = generate_synthetic_data(n_bars=total_bars, seed=42)

            train_end = int(len(df) * train_pct / 100)
            val_end = train_end + int(len(df) * val_pct / 100)

            periods = {
                "Train": df.iloc[:train_end],
                "Validate": df.iloc[train_end:val_end],
                "Test": df.iloc[val_end:],
            }

            wf_results = {}
            params = STRATEGY_MAP[wf_strat]["default_params"].copy()

            for period_name, period_df in periods.items():
                if len(period_df) < 50:
                    continue
                period_df = period_df.reset_index(drop=True)
                try:
                    result = run_backtest_from_ui(period_df, wf_strat, params)
                    wf_results[period_name] = result
                except Exception as e:
                    st.error(f"Error in {period_name}: {e}")

            st.session_state["wf_results"] = wf_results
            st.session_state["wf_periods"] = {k: len(v) for k, v in periods.items()}

        if "wf_results" in st.session_state:
            wf_results = st.session_state["wf_results"]

            # Period metrics comparison
            st.markdown("### Results by Period")
            wf_data = []
            for period, result in wf_results.items():
                m = result.metrics
                wf_data.append({
                    "Period": period,
                    "Bars": st.session_state["wf_periods"][period],
                    "Return %": f"{m.total_return_pct:.2f}",
                    "Sharpe": f"{m.sharpe_ratio:.2f}",
                    "Max DD %": f"{m.max_drawdown_pct:.1f}",
                    "Win Rate %": f"{m.win_rate_pct:.0f}",
                    "Trades": m.total_trades,
                })
            st.dataframe(pd.DataFrame(wf_data), width='stretch', hide_index=True)

            # Equity curves by period
            fig_wf = go.Figure()
            period_colors = {"Train": COLORS["info"], "Validate": COLORS["warning"], "Test": COLORS["success"]}
            for period, result in wf_results.items():
                fig_wf.add_trace(go.Scatter(
                    y=result.equity_curve.values,
                    mode="lines",
                    name=period,
                    line=dict(color=period_colors.get(period, COLORS["primary"]), width=2),
                ))

            fig_wf.update_layout(
                title="Equity Curves by Period",
                yaxis_title="Portfolio Value (₹)",
                template=COLORS["plotly_template"], height=450,
                paper_bgcolor=COLORS["bg_dark"], plot_bgcolor=COLORS["bg_dark"],
                font=dict(color=COLORS["text"]),
            )
            st.plotly_chart(fig_wf, width='stretch')

            # Overfitting indicator
            if "Train" in wf_results and "Test" in wf_results:
                train_ret = wf_results["Train"].metrics.total_return_pct
                test_ret = wf_results["Test"].metrics.total_return_pct

                overfit_ratio = test_ret / train_ret if train_ret != 0 else 0

                if overfit_ratio > 0.7:
                    st.success(f"✅ Low overfitting risk — Test/Train ratio: {overfit_ratio:.2f}")
                elif overfit_ratio > 0.3:
                    st.warning(f"⚠️ Moderate overfitting risk — Test/Train ratio: {overfit_ratio:.2f}")
                else:
                    st.error(f"🔴 High overfitting risk — Test/Train ratio: {overfit_ratio:.2f}")
