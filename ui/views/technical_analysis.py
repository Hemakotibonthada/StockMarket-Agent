"""Technical Analysis page – interactive charting with multiple indicators."""

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
    COLORS, generate_synthetic_data, load_sample_data, compute_base_features,
    create_candlestick_chart,
)
from src.features.indicators import (
    atr, rsi, zscore, vwap, bollinger_bands, ema, macd,
    realized_volatility, returns, log_returns, rate_of_change,
    on_balance_volume, rolling_mean, rolling_std,
)


def render():
    st.markdown("## 📈 Technical Analysis")
    st.markdown(
        f'<p style="color: {COLORS["text_muted"]}; margin-top: -10px;">'
        'Interactive price charts with overlays and technical indicators</p>',
        unsafe_allow_html=True,
    )

    # ── Data source ────────────────────────────────────────────────────────
    col_src, col_sym, col_bars = st.columns([2, 1, 1])
    with col_src:
        source = st.selectbox("Data Source", [
            "Synthetic Data", "Sample: RELIANCE", "Sample: TCS", "Upload CSV",
        ], key="ta_source")
    with col_sym:
        custom_symbol = st.text_input("Symbol Label", "RELIANCE", key="ta_sym")
    with col_bars:
        n_bars = st.slider("Bars", 60, 2000, 500, 20, key="ta_bars")

    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload OHLCV CSV", type=["csv"], key="ta_upload")

    # Load data
    if source == "Upload CSV" and uploaded:
        df = pd.read_csv(uploaded, parse_dates=["date"])
    elif source == "Sample: RELIANCE":
        df = load_sample_data("RELIANCE")
    elif source == "Sample: TCS":
        df = load_sample_data("TCS")
    else:
        df = generate_synthetic_data(symbol=custom_symbol, n_bars=n_bars, seed=42)

    date_col = "date" if "date" in df.columns else "datetime"

    # ── Indicator selection ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔧 Indicator Panel")

    overlay_col, sub_col, params_col = st.columns([1, 1, 1])

    with overlay_col:
        st.markdown("**Price Overlays**")
        show_sma = st.checkbox("SMA (Simple Moving Average)", True)
        sma_period = st.slider("SMA Period", 5, 100, 20, key="sma_p") if show_sma else 20
        show_ema_line = st.checkbox("EMA (Exponential Moving Average)", False)
        ema_period = st.slider("EMA Span", 5, 100, 20, key="ema_p") if show_ema_line else 20
        show_bb = st.checkbox("Bollinger Bands", True)
        bb_period = st.slider("BB Period", 5, 50, 20, key="bb_p") if show_bb else 20
        bb_std = st.slider("BB Std Dev", 1.0, 3.5, 2.0, 0.1, key="bb_s") if show_bb else 2.0
        show_vwap_line = st.checkbox("VWAP", False)

    with sub_col:
        st.markdown("**Sub-indicators**")
        show_rsi = st.checkbox("RSI", True)
        rsi_period = st.slider("RSI Period", 5, 30, 14, key="rsi_p") if show_rsi else 14
        show_macd = st.checkbox("MACD", True)
        show_volume = st.checkbox("Volume", True)
        show_atr = st.checkbox("ATR", False)
        atr_period = st.slider("ATR Period", 5, 30, 14, key="atr_p") if show_atr else 14
        show_obv = st.checkbox("OBV", False)

    with params_col:
        st.markdown("**Analysis**")
        show_zscore = st.checkbox("Z-Score", False)
        zscore_window = st.slider("Z-Score Window", 5, 60, 20, key="zs_w") if show_zscore else 20
        show_rvol = st.checkbox("Realized Volatility", False)
        show_roc = st.checkbox("Rate of Change", False)
        roc_period = st.slider("ROC Period", 5, 30, 10, key="roc_p") if show_roc else 10

    # ── Build chart ────────────────────────────────────────────────────────
    st.markdown("---")

    # Count sub-indicator panels
    sub_panels = []
    if show_volume:
        sub_panels.append("Volume")
    if show_rsi:
        sub_panels.append("RSI")
    if show_macd:
        sub_panels.append("MACD")
    if show_atr:
        sub_panels.append("ATR")
    if show_obv:
        sub_panels.append("OBV")
    if show_zscore:
        sub_panels.append("Z-Score")
    if show_rvol:
        sub_panels.append("Volatility")
    if show_roc:
        sub_panels.append("ROC")

    n_rows = 1 + len(sub_panels)
    heights = [0.45] + [0.55 / max(len(sub_panels), 1)] * len(sub_panels) if sub_panels else [1.0]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=heights,
        subplot_titles=["Price"] + sub_panels,
    )

    # ── Price candlestick ──────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df[date_col], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="OHLC",
        increasing_line_color=COLORS["success"],
        decreasing_line_color=COLORS["danger"],
    ), row=1, col=1)

    # Overlays
    if show_sma:
        sma_vals = rolling_mean(df["close"], sma_period)
        fig.add_trace(go.Scatter(
            x=df[date_col], y=sma_vals, mode="lines",
            name=f"SMA({sma_period})", line=dict(color="#F59E0B", width=1.5),
        ), row=1, col=1)

    if show_ema_line:
        ema_vals = ema(df["close"], ema_period)
        fig.add_trace(go.Scatter(
            x=df[date_col], y=ema_vals, mode="lines",
            name=f"EMA({ema_period})", line=dict(color="#EC4899", width=1.5),
        ), row=1, col=1)

    if show_bb:
        bb_mid, bb_upper, bb_lower = bollinger_bands(df["close"], bb_period, bb_std)
        fig.add_trace(go.Scatter(
            x=df[date_col], y=bb_upper, mode="lines",
            name="BB Upper", line=dict(color="#3B82F6", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df[date_col], y=bb_lower, mode="lines",
            name="BB Lower", line=dict(color="#3B82F6", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(59,130,246,0.05)",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df[date_col], y=bb_mid, mode="lines",
            name="BB Mid", line=dict(color="#3B82F6", width=1),
        ), row=1, col=1)

    if show_vwap_line:
        vwap_vals = vwap(df["high"], df["low"], df["close"], df["volume"])
        fig.add_trace(go.Scatter(
            x=df[date_col], y=vwap_vals, mode="lines",
            name="VWAP", line=dict(color="#8B5CF6", width=1.5, dash="dash"),
        ), row=1, col=1)

    # ── Sub-panels ─────────────────────────────────────────────────────────
    current_row = 2

    if show_volume:
        colors = [COLORS["success"] if c >= o else COLORS["danger"]
                  for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Bar(
            x=df[date_col], y=df["volume"], marker_color=colors,
            opacity=0.6, name="Volume",
        ), row=current_row, col=1)
        current_row += 1

    if show_rsi:
        rsi_vals = rsi(df["close"], rsi_period)
        fig.add_trace(go.Scatter(
            x=df[date_col], y=rsi_vals, mode="lines",
            name="RSI", line=dict(color="#8B5CF6", width=1.5),
        ), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color=COLORS["danger"],
                      row=current_row, col=1, opacity=0.5)
        fig.add_hline(y=30, line_dash="dot", line_color=COLORS["success"],
                      row=current_row, col=1, opacity=0.5)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(99,102,241,0.05)",
                      line_width=0, row=current_row, col=1)
        current_row += 1

    if show_macd:
        macd_line, signal_line, histogram = macd(df["close"])
        hist_colors = [COLORS["success"] if h >= 0 else COLORS["danger"]
                       for h in histogram.fillna(0)]
        fig.add_trace(go.Bar(
            x=df[date_col], y=histogram, marker_color=hist_colors,
            opacity=0.5, name="MACD Hist",
        ), row=current_row, col=1)
        fig.add_trace(go.Scatter(
            x=df[date_col], y=macd_line, mode="lines",
            name="MACD", line=dict(color="#3B82F6", width=1.5),
        ), row=current_row, col=1)
        fig.add_trace(go.Scatter(
            x=df[date_col], y=signal_line, mode="lines",
            name="Signal", line=dict(color="#F59E0B", width=1.5),
        ), row=current_row, col=1)
        current_row += 1

    if show_atr:
        atr_vals = atr(df["high"], df["low"], df["close"], atr_period)
        fig.add_trace(go.Scatter(
            x=df[date_col], y=atr_vals, mode="lines",
            name="ATR", line=dict(color="#F97316", width=1.5),
            fill="tozeroy", fillcolor="rgba(249,115,22,0.1)",
        ), row=current_row, col=1)
        current_row += 1

    if show_obv:
        obv_vals = on_balance_volume(df["close"], df["volume"])
        fig.add_trace(go.Scatter(
            x=df[date_col], y=obv_vals, mode="lines",
            name="OBV", line=dict(color="#14B8A6", width=1.5),
        ), row=current_row, col=1)
        current_row += 1

    if show_zscore:
        zs_vals = zscore(df["close"], zscore_window)
        fig.add_trace(go.Scatter(
            x=df[date_col], y=zs_vals, mode="lines",
            name="Z-Score", line=dict(color="#A855F7", width=1.5),
        ), row=current_row, col=1)
        fig.add_hline(y=2, line_dash="dot", line_color=COLORS["danger"],
                      row=current_row, col=1, opacity=0.5)
        fig.add_hline(y=-2, line_dash="dot", line_color=COLORS["success"],
                      row=current_row, col=1, opacity=0.5)
        fig.add_hline(y=0, line_dash="solid", line_color=COLORS["text_muted"],
                      row=current_row, col=1, opacity=0.3)
        current_row += 1

    if show_rvol:
        rvol_vals = realized_volatility(df["close"])
        fig.add_trace(go.Scatter(
            x=df[date_col], y=rvol_vals * 100, mode="lines",
            name="RVol %", line=dict(color="#EC4899", width=1.5),
            fill="tozeroy", fillcolor="rgba(236,72,153,0.1)",
        ), row=current_row, col=1)
        current_row += 1

    if show_roc:
        roc_vals = rate_of_change(df["close"], roc_period)
        fig.add_trace(go.Scatter(
            x=df[date_col], y=roc_vals, mode="lines",
            name="ROC", line=dict(color="#06B6D4", width=1.5),
        ), row=current_row, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color=COLORS["text_muted"],
                      row=current_row, col=1, opacity=0.3)
        current_row += 1

    # Layout
    chart_height = 500 + len(sub_panels) * 150
    fig.update_layout(
        title=f"{custom_symbol} — Technical Analysis",
        template=COLORS["plotly_template"],
        height=chart_height,
        xaxis_rangeslider_visible=False,
        paper_bgcolor=COLORS["bg_dark"],
        plot_bgcolor=COLORS["bg_dark"],
        font=dict(color=COLORS["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, width='stretch')

    # ── Price Statistics ───────────────────────────────────────────────────
    st.markdown("### 📊 Price Statistics")
    stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
    with stat_c1:
        st.metric("Current Price", f"₹{df['close'].iloc[-1]:,.2f}")
    with stat_c2:
        range_pct = (df["high"].max() - df["low"].min()) / df["low"].min() * 100
        st.metric("52W Range", f"{range_pct:.1f}%")
    with stat_c3:
        avg_vol = df["volume"].mean()
        st.metric("Avg Volume", f"{avg_vol:,.0f}")
    with stat_c4:
        daily_ret = returns(df["close"])
        st.metric("Volatility (ann.)", f"{daily_ret.std() * np.sqrt(252) * 100:.1f}%")

    # Feature correlation
    with st.expander("📐 Feature Correlation Matrix"):
        df_feat = compute_base_features(df.copy())
        numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
        # Pick a subset to avoid overwhelming
        key_cols = [c for c in numeric_cols if c in [
            "close", "returns", "log_returns", "rsi", "zscore", "atr",
            "macd", "signal_line", "ema_12", "sma_20", "obv", "rvol",
        ]]
        if len(key_cols) >= 3:
            corr = df_feat[key_cols].corr()
            import plotly.express as px
            fig_corr = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale=[COLORS["danger"], COLORS["bg_card"], COLORS["success"]],
                aspect="auto",
            )
            fig_corr.update_layout(
                height=500,
                template=COLORS["plotly_template"],
                paper_bgcolor=COLORS["bg_dark"],
                plot_bgcolor=COLORS["bg_dark"],
                font=dict(color=COLORS["text"]),
            )
            st.plotly_chart(fig_corr, width='stretch')
        else:
            st.info("Not enough numeric feature columns for a correlation matrix.")
