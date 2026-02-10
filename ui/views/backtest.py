"""Backtest Runner page – configure strategies, run backtests, view results."""

from __future__ import annotations

import sys
from pathlib import Path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import pandas as pd
import numpy as np

from ui.utils import (
    COLORS, STRATEGY_MAP,
    generate_synthetic_data, run_backtest_from_ui, compute_base_features,
    create_equity_curve, create_returns_distribution,
    create_monthly_returns_heatmap, create_trade_scatter,
    create_cumulative_pnl, create_candlestick_chart,
    trades_to_dataframe,
)
from src.backtest.metrics import compute_returns, compute_drawdown


def render():
    st.markdown("## ▶️ Backtest Runner")
    st.markdown(
        f'<p style="color: {COLORS["text_muted"]}; margin-top: -10px;">'
        'Configure and run backtests with full strategy customization</p>',
        unsafe_allow_html=True,
    )

    # ── Sidebar-style config in an expander ────────────────────────────────
    with st.expander("⚙️ Backtest Configuration", expanded=True):
        cfg_c1, cfg_c2, cfg_c3 = st.columns(3)

        with cfg_c1:
            st.markdown("**Data Settings**")
            data_source = st.selectbox(
                "Data Source",
                ["Synthetic Data", "Upload CSV", "Sample: RELIANCE", "Sample: TCS"],
            )
            n_bars = st.slider("Number of Bars", 100, 2000, 500, 50)
            start_price = st.number_input("Start Price (₹)", 500, 10000, 2500, step=100)
            seed = st.number_input("Random Seed", 1, 9999, 42)

        with cfg_c2:
            st.markdown("**Strategy Settings**")
            strategy_name = st.selectbox("Strategy", list(STRATEGY_MAP.keys()))
            defaults = STRATEGY_MAP[strategy_name]["default_params"]

            st.markdown("**Parameters**")
            params = {}
            if strategy_name == "Mean Reversion":
                params["zscore_window"] = st.slider("Z-Score Window", 5, 50, defaults["zscore_window"])
                params["zscore_entry"] = st.slider("Z-Score Entry", 0.5, 4.0, defaults["zscore_entry"], 0.1)
                params["zscore_exit"] = st.slider("Z-Score Exit", 0.1, 2.0, defaults["zscore_exit"], 0.1)
                params["rsi_period"] = st.slider("RSI Period", 5, 30, defaults["rsi_period"])
                params["rsi_oversold"] = st.slider("RSI Oversold", 10, 40, defaults["rsi_oversold"])
                params["rsi_overbought"] = st.slider("RSI Overbought", 60, 90, defaults["rsi_overbought"])
                params["max_positions"] = st.slider("Max Positions", 1, 10, defaults["max_positions"])
            elif strategy_name == "ORB Momentum":
                params["orb_window_minutes"] = st.slider("ORB Window (min)", 5, 60, defaults["orb_window_minutes"])
                params["atr_period"] = st.slider("ATR Period", 5, 30, defaults["atr_period"])
                params["atr_multiplier"] = st.slider("ATR Multiplier", 0.5, 4.0, defaults["atr_multiplier"], 0.1)
                params["time_stop_minutes"] = st.slider("Time Stop (min)", 30, 360, defaults["time_stop_minutes"])
                params["max_positions"] = st.slider("Max Positions", 1, 10, defaults["max_positions"])
            elif strategy_name == "Pairs Trading":
                params["zscore_entry"] = st.slider("Z-Score Entry", 0.5, 4.0, defaults["zscore_entry"], 0.1)
                params["zscore_exit"] = st.slider("Z-Score Exit", 0.1, 2.0, defaults["zscore_exit"], 0.1)
                params["lookback"] = st.slider("Lookback", 20, 120, defaults["lookback"])
                params["max_positions"] = st.slider("Max Positions", 1, 5, defaults["max_positions"])

        with cfg_c3:
            st.markdown("**Capital & Costs**")
            initial_capital = st.number_input("Initial Capital (₹)", 100_000, 100_000_000, 1_000_000, step=100_000)
            slippage_mode = st.selectbox("Slippage Mode", ["fixed", "random", "none"])
            slippage_bps = st.slider("Slippage (bps)", 0.0, 20.0, 3.0, 0.5)

    # ── Data loading ───────────────────────────────────────────────────────
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload OHLCV CSV (columns: date, open, high, low, close, volume)",
            type=["csv"],
        )

    # ── Run button ─────────────────────────────────────────────────────────
    run_col1, run_col2, _ = st.columns([1, 1, 4])
    with run_col1:
        run_clicked = st.button("🚀 Run Backtest", width='stretch')
    with run_col2:
        clear_clicked = st.button("🗑️ Clear Results", width='stretch')
        if clear_clicked and "backtest_result" in st.session_state:
            del st.session_state["backtest_result"]

    if run_clicked:
        with st.spinner("Running backtest... ⏳"):
            # Load data
            if data_source == "Upload CSV" and uploaded_file is not None:
                df = pd.read_csv(uploaded_file, parse_dates=["date"])
            elif data_source == "Sample: RELIANCE":
                from ui.utils import load_sample_data
                df = load_sample_data("RELIANCE")
            elif data_source == "Sample: TCS":
                from ui.utils import load_sample_data
                df = load_sample_data("TCS")
            else:
                df = generate_synthetic_data(n_bars=n_bars, start_price=start_price, seed=seed)

            if "symbol" not in df.columns:
                df["symbol"] = "STOCK"

            result = run_backtest_from_ui(
                df, strategy_name, params,
                initial_capital=initial_capital,
                slippage_mode=slippage_mode,
                slippage_bps=slippage_bps,
                seed=seed,
            )
            st.session_state["backtest_result"] = result
            st.session_state["backtest_df"] = df
            st.session_state["backtest_config"] = {
                "strategy": strategy_name,
                "params": params,
                "capital": initial_capital,
            }

        st.success("Backtest complete! ✅")

    # ── Display results ────────────────────────────────────────────────────
    if "backtest_result" in st.session_state:
        result = st.session_state["backtest_result"]
        metrics = result.metrics
        config = st.session_state.get("backtest_config", {})

        st.markdown("---")
        st.markdown(f"### 📊 Results: {config.get('strategy', 'N/A')}")

        # KPI strip
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            st.metric("Total Return", f"{metrics.total_return_pct:.2f}%")
        with k2:
            st.metric("CAGR", f"{metrics.cagr_pct:.2f}%")
        with k3:
            st.metric("Sharpe", f"{metrics.sharpe_ratio:.2f}")
        with k4:
            st.metric("Max DD", f"{metrics.max_drawdown_pct:.1f}%", delta_color="inverse")
        with k5:
            st.metric("Win Rate", f"{metrics.win_rate_pct:.0f}%")
        with k6:
            st.metric("Trades", f"{metrics.total_trades}")

        # Tabs for different views
        tab_eq, tab_trades, tab_dist, tab_monthly = st.tabs([
            "📈 Equity Curve", "💰 Trades", "📊 Distribution", "📅 Monthly Returns",
        ])

        with tab_eq:
            dd = compute_drawdown(result.equity_curve)
            fig = create_equity_curve(result.equity_curve, dd)
            st.plotly_chart(fig, width='stretch')

            # Cumulative P&L
            fig_cum = create_cumulative_pnl(result.trades)
            st.plotly_chart(fig_cum, width='stretch')

        with tab_trades:
            trade_df = trades_to_dataframe(result.trades)
            if not trade_df.empty:
                st.dataframe(trade_df, width='stretch', hide_index=True, height=400)
                fig_sc = create_trade_scatter(result.trades)
                st.plotly_chart(fig_sc, width='stretch')
            else:
                st.info("No trades were generated during this backtest.")

        with tab_dist:
            rets = compute_returns(result.equity_curve)
            fig_dist = create_returns_distribution(rets)
            st.plotly_chart(fig_dist, width='stretch')

            # Statistics
            st.markdown("**Return Statistics**")
            ret_stats = {
                "Mean Daily Return": f"{rets.mean()*100:.4f}%",
                "Std Dev": f"{rets.std()*100:.4f}%",
                "Skewness": f"{rets.skew():.3f}",
                "Kurtosis": f"{rets.kurtosis():.3f}",
                "VaR 95%": f"{metrics.var_95_pct:.3f}%",
                "Expected Shortfall 95%": f"{metrics.es_95_pct:.3f}%",
            }
            st.dataframe(pd.DataFrame(ret_stats.items(), columns=["Statistic", "Value"]),
                          width='stretch', hide_index=True)

        with tab_monthly:
            fig_hm = create_monthly_returns_heatmap(result.equity_curve)
            st.plotly_chart(fig_hm, width='stretch')

        # Detailed metrics expander
        with st.expander("📋 Full Performance Report"):
            summary = result.summary()
            left_s, right_s = st.columns(2)
            items = list(summary.items())
            mid = len(items) // 2
            with left_s:
                for k, v in items[:mid]:
                    st.markdown(f"**{k.replace('_', ' ').title()}**: {v}")
            with right_s:
                for k, v in items[mid:]:
                    st.markdown(f"**{k.replace('_', ' ').title()}**: {v}")
