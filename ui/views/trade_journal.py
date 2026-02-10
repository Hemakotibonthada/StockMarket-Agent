"""Trade Journal page – detailed trade log with filtering, analytics, and export."""

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
import plotly.express as px

from ui.utils import (
    COLORS, generate_synthetic_data, run_backtest_from_ui,
    create_trade_scatter, create_cumulative_pnl, trades_to_dataframe,
)


def render():
    st.markdown("## 📒 Trade Journal")
    st.markdown(
        f'<p style="color: {COLORS["text_muted"]}; margin-top: -10px;">'
        'Detailed trade log with filtering and export</p>',
        unsafe_allow_html=True,
    )

    # Check for existing backtest result
    if "backtest_result" not in st.session_state:
        st.info("No backtest results found. Running a demo backtest...")
        df = generate_synthetic_data(n_bars=500, seed=42)
        result = run_backtest_from_ui(df, "Mean Reversion", {"zscore_entry": 2.0, "zscore_exit": 0.5})
        st.session_state["backtest_result"] = result

    result = st.session_state["backtest_result"]
    trades = result.trades

    if not trades:
        st.warning("No trades were generated. Try adjusting strategy parameters.")
        return

    # Build full trade dataframe
    records = []
    for i, t in enumerate(trades, 1):
        records.append({
            "Trade #": i,
            "Symbol": t.symbol,
            "Side": t.side,
            "Entry Price": t.entry_price,
            "Exit Price": t.exit_price,
            "Quantity": t.quantity,
            "Gross P&L": t.pnl,
            "Costs": t.costs,
            "Net P&L": t.net_pnl,
            "Return %": (t.exit_price - t.entry_price) / t.entry_price * 100 if t.side == "BUY"
                        else (t.entry_price - t.exit_price) / t.entry_price * 100,
            "Holding Bars": t.holding_bars,
            "Winner": "✅ Win" if t.net_pnl > 0 else "❌ Loss",
        })
    trade_df = pd.DataFrame(records)

    # ── KPIs ───────────────────────────────────────────────────────────────
    total_trades = len(trades)
    winners = [t for t in trades if t.net_pnl > 0]
    losers = [t for t in trades if t.net_pnl <= 0]
    total_pnl = sum(t.net_pnl for t in trades)
    avg_win = np.mean([t.net_pnl for t in winners]) if winners else 0
    avg_loss = np.mean([t.net_pnl for t in losers]) if losers else 0
    largest_win = max([t.net_pnl for t in trades]) if trades else 0
    largest_loss = min([t.net_pnl for t in trades]) if trades else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Total Trades", total_trades)
    with k2:
        st.metric("Win Rate", f"{len(winners)/total_trades*100:.0f}%")
    with k3:
        st.metric("Net P&L", f"₹{total_pnl:,.0f}",
                  delta="profit" if total_pnl > 0 else "loss")
    with k4:
        st.metric("Avg Win", f"₹{avg_win:,.0f}")
    with k5:
        st.metric("Avg Loss", f"₹{avg_loss:,.0f}")

    st.markdown("---")

    # ── Filters ────────────────────────────────────────────────────────────
    with st.expander("🔍 Filters", expanded=False):
        f1, f2, f3 = st.columns(3)
        with f1:
            side_filter = st.multiselect("Side", ["BUY", "SELL"], default=["BUY", "SELL"])
        with f2:
            outcome_filter = st.multiselect("Outcome", ["✅ Win", "❌ Loss"], default=["✅ Win", "❌ Loss"])
        with f3:
            min_pnl = st.number_input("Min Net P&L", value=float(trade_df["Net P&L"].min()), key="min_pnl")
            max_pnl = st.number_input("Max Net P&L", value=float(trade_df["Net P&L"].max()), key="max_pnl")

    # Apply filters
    filtered = trade_df[
        (trade_df["Side"].isin(side_filter)) &
        (trade_df["Winner"].isin(outcome_filter)) &
        (trade_df["Net P&L"] >= min_pnl) &
        (trade_df["Net P&L"] <= max_pnl)
    ]

    # ── Trade Table ────────────────────────────────────────────────────────
    st.markdown(f"### 📋 Trade Log ({len(filtered)} / {total_trades} trades)")

    # Format for display
    display_df = filtered.copy()
    for col in ["Entry Price", "Exit Price", "Gross P&L", "Costs", "Net P&L"]:
        display_df[col] = display_df[col].apply(lambda x: f"₹{x:,.2f}")
    display_df["Return %"] = filtered["Return %"].apply(lambda x: f"{x:.2f}%")

    st.dataframe(display_df, width='stretch', hide_index=True, height=400)

    # CSV download
    csv = filtered.to_csv(index=False)
    st.download_button(
        "📥 Download Trades CSV",
        csv,
        "trade_journal.csv",
        "text/csv",
        width='content',
    )

    st.markdown("---")

    # ── Visualizations ─────────────────────────────────────────────────────
    tab_pnl, tab_dist, tab_streak, tab_analysis = st.tabs([
        "💰 P&L Chart", "📊 Distribution", "🔥 Streaks", "🔬 Analysis",
    ])

    with tab_pnl:
        col_sc, col_cum = st.columns(2)
        with col_sc:
            fig_scatter = create_trade_scatter(trades, "Trade P&L Scatter")
            st.plotly_chart(fig_scatter, width='stretch')
        with col_cum:
            fig_cum = create_cumulative_pnl(trades, "Cumulative P&L")
            st.plotly_chart(fig_cum, width='stretch')

    with tab_dist:
        # P&L distribution
        fig_hist = go.Figure()
        pnls = [t.net_pnl for t in trades]
        fig_hist.add_trace(go.Histogram(
            x=pnls, nbinsx=20,
            marker_color=[COLORS["success"] if p > 0 else COLORS["danger"] for p in sorted(pnls)],
            opacity=0.7,
        ))
        fig_hist.update_layout(
            title="P&L Distribution",
            xaxis_title="Net P&L (₹)", yaxis_title="Count",
            template=COLORS["plotly_template"], height=400,
            paper_bgcolor=COLORS["bg_dark"], plot_bgcolor=COLORS["bg_dark"],
        )
        st.plotly_chart(fig_hist, width='stretch')

        # Return % distribution
        fig_ret = go.Figure()
        ret_pcts = trade_df["Return %"].values
        fig_ret.add_trace(go.Histogram(
            x=ret_pcts, nbinsx=20,
            marker_color=COLORS["primary"],
            opacity=0.7,
        ))
        fig_ret.update_layout(
            title="Return % Distribution",
            xaxis_title="Return %", yaxis_title="Count",
            template=COLORS["plotly_template"], height=400,
            paper_bgcolor=COLORS["bg_dark"], plot_bgcolor=COLORS["bg_dark"],
        )
        st.plotly_chart(fig_ret, width='stretch')

    with tab_streak:
        # Win/Loss streak analysis
        outcomes = ["W" if t.net_pnl > 0 else "L" for t in trades]
        streaks = []
        current_streak = 1
        for i in range(1, len(outcomes)):
            if outcomes[i] == outcomes[i-1]:
                current_streak += 1
            else:
                streaks.append({"type": outcomes[i-1], "length": current_streak})
                current_streak = 1
        streaks.append({"type": outcomes[-1], "length": current_streak})

        streak_df = pd.DataFrame(streaks)

        fig_streak = go.Figure()
        for _, row in streak_df.iterrows():
            fig_streak.add_trace(go.Bar(
                x=[row.name],
                y=[row["length"]],
                marker_color=COLORS["success"] if row["type"] == "W" else COLORS["danger"],
                name=f"{'Win' if row['type'] == 'W' else 'Loss'} Streak",
                showlegend=False,
            ))
        fig_streak.update_layout(
            title="Win/Loss Streaks",
            yaxis_title="Streak Length",
            template=COLORS["plotly_template"], height=350,
            paper_bgcolor=COLORS["bg_dark"], plot_bgcolor=COLORS["bg_dark"],
        )
        st.plotly_chart(fig_streak, width='stretch')

        # Streak stats
        s1, s2, s3 = st.columns(3)
        win_streaks = [s["length"] for s in streaks if s["type"] == "W"]
        loss_streaks = [s["length"] for s in streaks if s["type"] == "L"]
        with s1:
            st.metric("Longest Win Streak", max(win_streaks) if win_streaks else 0)
        with s2:
            st.metric("Longest Loss Streak", max(loss_streaks) if loss_streaks else 0)
        with s3:
            st.metric("Avg Win Streak", f"{np.mean(win_streaks):.1f}" if win_streaks else "0")

    with tab_analysis:
        # Holding period analysis
        fig_hold = go.Figure()
        fig_hold.add_trace(go.Scatter(
            x=trade_df["Holding Bars"],
            y=trade_df["Net P&L"],
            mode="markers",
            marker=dict(
                color=[COLORS["success"] if p > 0 else COLORS["danger"] for p in trade_df["Net P&L"]],
                size=10,
            ),
            text=[f"Trade #{i}" for i in trade_df["Trade #"]],
        ))
        fig_hold.update_layout(
            title="P&L vs Holding Period",
            xaxis_title="Holding Bars",
            yaxis_title="Net P&L (₹)",
            template=COLORS["plotly_template"], height=400,
            paper_bgcolor=COLORS["bg_dark"], plot_bgcolor=COLORS["bg_dark"],
        )
        st.plotly_chart(fig_hold, width='stretch')

        # Win % by side
        side_stats = trade_df.groupby("Side").agg(
            total=("Trade #", "count"),
            wins=("Net P&L", lambda x: (x > 0).sum()),
            avg_pnl=("Net P&L", "mean"),
            total_pnl=("Net P&L", "sum"),
        ).reset_index()
        side_stats["win_rate"] = side_stats["wins"] / side_stats["total"] * 100
        st.markdown("**Performance by Side**")
        st.dataframe(side_stats, width='stretch', hide_index=True)
