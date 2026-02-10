"""Settings page – application configuration and preferences."""

from __future__ import annotations

import sys
from pathlib import Path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import pandas as pd

from ui.utils import COLORS
from src.core.config import (
    AppConfig, CostsConfig, SlippageConfig, RiskConfig,
    TripwireConfig, WalkforwardConfig,
)


def render():
    st.markdown("## ⚙️ Settings")
    st.markdown(
        f'<p style="color: {COLORS["text_muted"]}; margin-top: -10px;">'
        'Configure application defaults and preferences</p>',
        unsafe_allow_html=True,
    )

    tab_theme, tab_general, tab_costs, tab_risk, tab_data, tab_about = st.tabs([
        "🎨 Theme", "🔧 General", "💰 Costs & Slippage", "🛡️ Risk Defaults", "📁 Data", "ℹ️ About",
    ])

    # ═══════════════════════════════════════════════════════════════════════
    #  Theme Settings
    # ═══════════════════════════════════════════════════════════════════════
    with tab_theme:
        st.markdown("### 🎨 Appearance")
        st.markdown(
            f'<p style="color: {COLORS["text_muted"]}; margin-top: -10px;">'
            "Choose your preferred theme. Changes take effect immediately.</p>",
            unsafe_allow_html=True,
        )

        current_theme = st.session_state.get("app_theme", "system")

        theme_options = {
            "system": "🖥️ System (Auto-detect)",
            "dark": "🌙 Dark",
            "light": "☀️ Light",
        }

        theme_index = list(theme_options.keys()).index(current_theme) if current_theme in theme_options else 0

        selected_theme = st.radio(
            "Theme",
            options=list(theme_options.keys()),
            format_func=lambda x: theme_options[x],
            index=theme_index,
            key="theme_radio",
            horizontal=True,
        )

        if selected_theme != current_theme:
            st.session_state["app_theme"] = selected_theme
            # Reset system detection if switching away from system
            if selected_theme != "system" and "system_theme_resolved" in st.session_state:
                del st.session_state["system_theme_resolved"]
            st.rerun()

        # Theme preview
        st.markdown("---")
        st.markdown("#### Preview")

        from ui.utils import DARK_COLORS, LIGHT_COLORS
        preview_colors = DARK_COLORS if (selected_theme == "dark" or
                         (selected_theme == "system" and
                          st.session_state.get("system_theme_resolved", "dark") == "dark")) else LIGHT_COLORS

        prev_cols = st.columns(4)
        preview_items = [
            ("Background", "bg_dark"),
            ("Cards", "bg_card"),
            ("Text", "text"),
            ("Primary", "primary"),
        ]
        for col, (label, key) in zip(prev_cols, preview_items):
            with col:
                hex_color = preview_colors[key]
                border = f"2px solid {preview_colors['grid']}"
                st.markdown(
                    f'<div style="background: {hex_color}; border: {border}; '
                    f'border-radius: 8px; height: 60px; display: flex; '
                    f'align-items: center; justify-content: center;">'
                    f'<span style="color: {"#000" if key in ("bg_dark", "bg_card", "text") and selected_theme == "light" else "#FFF"}; '
                    f'font-size: 0.75rem; font-weight: 600;">{label}<br/>{hex_color}</span></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("")
        st.info(
            "**System** mode detects your OS dark/light preference on first load. "
            "**Dark** and **Light** force a specific theme regardless of OS settings."
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  General Settings
    # ═══════════════════════════════════════════════════════════════════════
    with tab_general:
        st.markdown("### General Configuration")

        gc1, gc2 = st.columns(2)
        with gc1:
            default_strategy = st.selectbox(
                "Default Strategy",
                ["Mean Reversion", "ORB Momentum", "Pairs Trading"],
                key="set_strat",
            )
            default_capital = st.number_input(
                "Default Capital (₹)", 100_000, 100_000_000, 1_000_000,
                step=100_000, key="set_cap",
            )
            seed = st.number_input("Default Seed", 1, 9999, 42, key="set_seed")

        with gc2:
            bar_interval = st.selectbox("Default Bar Interval", ["1min", "5min", "15min", "1h", "1D"], index=1)
            timezone = st.selectbox("Timezone", ["Asia/Kolkata", "UTC"], index=0)
            log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)

        if st.button("💾 Save General Settings", key="save_gen"):
            st.session_state["settings_general"] = {
                "strategy": default_strategy,
                "capital": default_capital,
                "seed": seed,
                "bar_interval": bar_interval,
                "timezone": timezone,
                "log_level": log_level,
            }
            st.success("General settings saved!")

    # ═══════════════════════════════════════════════════════════════════════
    #  Costs & Slippage
    # ═══════════════════════════════════════════════════════════════════════
    with tab_costs:
        st.markdown("### Transaction Cost Model (Indian Market)")

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Regulatory Charges**")
            brokerage = st.number_input("Brokerage (bps)", 0.0, 50.0, 5.0, 0.5, key="c_brok")
            stt = st.number_input("STT (bps)", 0.0, 50.0, 10.0, 0.5, key="c_stt")
            gst = st.number_input("GST (bps)", 0.0, 10.0, 1.8, 0.1, key="c_gst")
            stamp = st.number_input("Stamp Duty (bps)", 0.0, 1.0, 0.003, 0.001, format="%.3f", key="c_stamp")
            sebi = st.number_input("SEBI Charges (bps)", 0.0, 0.01, 0.0001, 0.0001, format="%.4f", key="c_sebi")

        with cc2:
            st.markdown("**Slippage Model**")
            slip_mode = st.selectbox("Slippage Mode", ["fixed", "random", "none"], key="c_smode")
            slip_bps = st.number_input("Slippage Mean (bps)", 0.0, 30.0, 4.0, 0.5, key="c_sbps")
            slip_std = st.number_input("Slippage Std (bps)", 0.0, 20.0, 3.0, 0.5, key="c_sstd")

            # Preview total cost
            total_bps = brokerage + stt + gst + stamp + sebi + slip_bps
            st.markdown("---")
            st.markdown(
                f"""
                <div style="
                    background: {COLORS['bg_card']};
                    border: 1px solid {COLORS['primary']};
                    border-radius: 12px;
                    padding: 20px;
                    text-align: center;
                ">
                    <div style="color: {COLORS['text_muted']};">Approx. Total Cost per Trade</div>
                    <div style="color: {COLORS['primary']}; font-size: 2rem; font-weight: 700;">
                        {total_bps:.2f} bps
                    </div>
                    <div style="color: {COLORS['text_muted']}; font-size: 0.85rem;">
                        ₹{total_bps * 100:.0f} per ₹10L turnover
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("💾 Save Cost Settings", key="save_costs"):
            st.session_state["settings_costs"] = {
                "brokerage_bps": brokerage, "stt_bps": stt, "gst_bps": gst,
                "stamp_bps": stamp, "sebi_bps": sebi,
                "slippage_mode": slip_mode, "slippage_bps": slip_bps, "slippage_std": slip_std,
            }
            st.success("Cost settings saved!")

    # ═══════════════════════════════════════════════════════════════════════
    #  Risk Defaults
    # ═══════════════════════════════════════════════════════════════════════
    with tab_risk:
        st.markdown("### Default Risk Parameters")

        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("**Per-Trade Risk**")
            risk_per_trade = st.slider("Risk per Trade (%)", 0.1, 5.0, 0.5, 0.1, key="r_ptrade")
            max_pos_pct = st.slider("Max Position (% of Capital)", 1, 50, 20, key="r_maxpos")

        with rc2:
            st.markdown("**Portfolio Risk**")
            daily_max_loss = st.slider("Daily Max Loss (%)", 0.5, 10.0, 1.0, 0.5, key="r_daily")
            weekly_max_loss = st.slider("Weekly Max Loss (%)", 1.0, 20.0, 2.0, 0.5, key="r_weekly")
            max_drawdown = st.slider("Max Drawdown Kill Switch (%)", 1.0, 30.0, 5.0, 0.5, key="r_dd")

        st.markdown("**Tripwires**")
        tw1, tw2 = st.columns(2)
        with tw1:
            max_consec_rejects = st.number_input("Max Consecutive Rejects", 1, 20, 3, key="r_rejects")
            max_latency = st.number_input("Max Latency (ms)", 100, 10_000, 2000, key="r_lat")
        with tw2:
            feed_timeout = st.number_input("Feed Timeout (s)", 10, 300, 60, key="r_feed")

        if st.button("💾 Save Risk Settings", key="save_risk"):
            st.session_state["settings_risk"] = {
                "risk_per_trade_pct": risk_per_trade,
                "daily_max_loss_pct": daily_max_loss,
                "weekly_max_loss_pct": weekly_max_loss,
                "max_drawdown_pct": max_drawdown,
            }
            st.success("Risk settings saved!")

    # ═══════════════════════════════════════════════════════════════════════
    #  Data Settings
    # ═══════════════════════════════════════════════════════════════════════
    with tab_data:
        st.markdown("### Data Configuration")

        data_dir = st.text_input("Data Directory", "./data", key="d_dir")
        models_dir = st.text_input("Models Registry", "./models_registry", key="d_models")
        reports_dir = st.text_input("Reports Directory", "./reports", key="d_reports")

        st.markdown("---")
        st.markdown("### 📁 Available Data Files")

        sample_dir = Path("data/sample")
        if sample_dir.exists():
            files = list(sample_dir.glob("*.csv"))
            if files:
                file_data = []
                for f in files:
                    df = pd.read_csv(f, nrows=1)
                    full_df = pd.read_csv(f)
                    file_data.append({
                        "File": f.name,
                        "Rows": len(full_df),
                        "Columns": ", ".join(df.columns),
                        "Size": f"{f.stat().st_size / 1024:.1f} KB",
                    })
                st.dataframe(pd.DataFrame(file_data), width='stretch', hide_index=True)
            else:
                st.info("No CSV files found in data/sample/")
        else:
            st.info("data/sample directory not found")

        # Config file viewer
        st.markdown("### 📄 Configuration Files")
        config_dir = Path("configs")
        if config_dir.exists():
            config_files = list(config_dir.glob("*.yaml"))
            if config_files:
                selected_config = st.selectbox("View Config", [f.name for f in config_files])
                config_path = config_dir / selected_config
                with open(config_path) as f:
                    st.code(f.read(), language="yaml")

    # ═══════════════════════════════════════════════════════════════════════
    #  About
    # ═══════════════════════════════════════════════════════════════════════
    with tab_about:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {COLORS['bg_card']}, {COLORS['bg_card_alt']});
                border: 1px solid {COLORS['grid']};
                border-radius: 16px;
                padding: 40px;
                text-align: center;
                margin: 20px 0;
            ">
                <h1 style="
                    background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-size: 2.5rem;
                    font-weight: 800;
                ">📈 Stock Agent</h1>
                <p style="color: {COLORS['text_muted']}; font-size: 1.1rem; margin-top: 8px;">
                    Local-First Indian Stock Trading Agent
                </p>
                <p style="color: {COLORS['text_muted']}; font-size: 0.9rem; margin-top: 16px;">
                    Version 1.0.0
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### ✨ Features")
        features = [
            "🔄 **Backtest Engine** — Vectorized + event-driven hybrid with realistic cost model",
            "📊 **3 Strategies** — ORB Momentum, Mean Reversion, Pairs Trading",
            "📈 **16+ Indicators** — ATR, RSI, MACD, Bollinger, VWAP, OBV, Z-Score, and more",
            "🛡️ **Risk Management** — Position sizing (ATR, Kelly, Fixed Fraction), kill switches",
            "💰 **Indian Cost Model** — Brokerage, STT, GST, stamp duty, SEBI charges, slippage",
            "📅 **Walk-Forward Analysis** — Train/validate/test splits for robustness",
            "🎲 **Monte Carlo Simulation** — Trade reshuffling for outcome distribution",
            "📊 **Performance Metrics** — Sharpe, Sortino, Calmar, VaR, Expected Shortfall",
            "🗄️ **DuckDB Storage** — Parquet I/O for fast data handling",
            "🖥️ **CLI Tools** — sa-backtest, sa-data, sa-paper, sa-live",
        ]
        for feat in features:
            st.markdown(feat)

        st.markdown("### 🛠️ Tech Stack")
        tech_cols = st.columns(4)
        techs = [
            ("Python", "3.13+"),
            ("Pandas", "Data handling"),
            ("NumPy / SciPy", "Computation"),
            ("Plotly", "Visualization"),
            ("Streamlit", "Dashboard UI"),
            ("DuckDB", "Storage"),
            ("XGBoost / LightGBM", "ML Models"),
            ("Pydantic", "Configuration"),
        ]
        for i, (name, desc) in enumerate(techs):
            with tech_cols[i % 4]:
                st.markdown(
                    f"""
                    <div style="
                        background: {COLORS['bg_card']};
                        border: 1px solid {COLORS['grid']};
                        border-radius: 8px;
                        padding: 12px;
                        text-align: center;
                        margin: 4px 0;
                    ">
                        <div style="color: {COLORS['text']}; font-weight: 600;">{name}</div>
                        <div style="color: {COLORS['text_muted']}; font-size: 0.8rem;">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
