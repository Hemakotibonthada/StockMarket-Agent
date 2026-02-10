"""Reports page – generate and download HTML/PDF performance reports."""

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
from datetime import datetime

from ui.utils import (
    COLORS, generate_synthetic_data, run_backtest_from_ui,
    create_equity_curve, create_returns_distribution,
    create_monthly_returns_heatmap, create_trade_scatter,
    create_cumulative_pnl, trades_to_dataframe,
)
from src.backtest.metrics import compute_returns, compute_drawdown


def _generate_html_report(result, config_info: dict) -> str:
    """Generate a standalone HTML report."""
    metrics = result.metrics
    summary = result.summary()
    trades = result.trades

    trade_rows = ""
    for i, t in enumerate(trades[:50], 1):
        pnl_color = "#10B981" if t.net_pnl > 0 else "#EF4444"
        trade_rows += f"""
        <tr>
            <td>{i}</td>
            <td>{t.symbol}</td>
            <td>{t.side}</td>
            <td>₹{t.entry_price:,.2f}</td>
            <td>₹{t.exit_price:,.2f}</td>
            <td>{t.quantity}</td>
            <td style="color: {pnl_color};">₹{t.net_pnl:,.2f}</td>
            <td>{t.holding_bars}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Stock Agent — Backtest Report</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                background: #0F172A; color: #F8FAFC;
                padding: 40px; line-height: 1.6;
            }}
            .header {{
                text-align: center; padding: 40px 0;
                border-bottom: 2px solid #334155; margin-bottom: 40px;
            }}
            .header h1 {{
                background: linear-gradient(135deg, #6366F1, #EC4899);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                font-size: 2.5rem; font-weight: 800;
            }}
            .header p {{ color: #94A3B8; margin-top: 8px; }}
            .kpi-grid {{
                display: grid; grid-template-columns: repeat(4, 1fr);
                gap: 16px; margin-bottom: 40px;
            }}
            .kpi-card {{
                background: linear-gradient(135deg, #1E293B, #253048);
                border: 1px solid #334155; border-radius: 12px;
                padding: 24px; text-align: center;
            }}
            .kpi-label {{ color: #94A3B8; font-size: 0.85rem; }}
            .kpi-value {{ font-size: 2rem; font-weight: 700; margin: 8px 0; }}
            .section {{ margin-bottom: 40px; }}
            .section h2 {{ color: #818CF8; margin-bottom: 16px; font-size: 1.4rem; }}
            table {{
                width: 100%; border-collapse: collapse;
                background: #1E293B; border-radius: 8px; overflow: hidden;
            }}
            th {{ background: #334155; padding: 12px 16px; text-align: left; font-weight: 600; }}
            td {{ padding: 10px 16px; border-bottom: 1px solid #334155; }}
            tr:hover {{ background: #253048; }}
            .footer {{
                text-align: center; padding: 20px; margin-top: 40px;
                border-top: 1px solid #334155; color: #64748B; font-size: 0.8rem;
            }}
            .metrics-grid {{
                display: grid; grid-template-columns: repeat(2, 1fr);
                gap: 12px;
            }}
            .metric-row {{
                display: flex; justify-content: space-between;
                padding: 8px 16px; background: #1E293B;
                border-radius: 8px; border: 1px solid #334155;
            }}
            .metric-row span:first-child {{ color: #94A3B8; }}
            .metric-row span:last-child {{ font-weight: 600; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📈 Stock Agent — Backtest Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
               Strategy: {config_info.get('strategy', 'N/A')} |
               Capital: ₹{config_info.get('capital', 1_000_000):,.0f}</p>
        </div>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Total Return</div>
                <div class="kpi-value" style="color: {'#10B981' if metrics.total_return_pct > 0 else '#EF4444'};">
                    {metrics.total_return_pct:.2f}%
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Sharpe Ratio</div>
                <div class="kpi-value" style="color: #6366F1;">{metrics.sharpe_ratio:.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Max Drawdown</div>
                <div class="kpi-value" style="color: #EF4444;">{metrics.max_drawdown_pct:.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Win Rate</div>
                <div class="kpi-value" style="color: #F59E0B;">{metrics.win_rate_pct:.0f}%</div>
            </div>
        </div>

        <div class="section">
            <h2>📋 Performance Metrics</h2>
            <div class="metrics-grid">
                {''.join(f'<div class="metric-row"><span>{k.replace("_", " ").title()}</span><span>{v}</span></div>' for k, v in summary.items())}
            </div>
        </div>

        <div class="section">
            <h2>📒 Trade Log (Top 50)</h2>
            <table>
                <thead>
                    <tr>
                        <th>#</th><th>Symbol</th><th>Side</th>
                        <th>Entry</th><th>Exit</th><th>Qty</th>
                        <th>Net P&L</th><th>Bars</th>
                    </tr>
                </thead>
                <tbody>{trade_rows}</tbody>
            </table>
        </div>

        <div class="footer">
            <p>Stock Agent v1.0.0 — Local-First Indian Market Trading Agent</p>
        </div>
    </body>
    </html>
    """
    return html


def render():
    st.markdown("## 📊 Reports")
    st.markdown(
        '<p style="color: #94A3B8; margin-top: -10px;">'
        'Generate comprehensive performance reports</p>',
        unsafe_allow_html=True,
    )

    # Check for backtest results
    has_result = "backtest_result" in st.session_state

    if not has_result:
        st.info("No backtest results found. Running a demo backtest...")
        df = generate_synthetic_data(n_bars=500, seed=42)
        result = run_backtest_from_ui(df, "Mean Reversion", {"zscore_entry": 2.0, "zscore_exit": 0.5})
        st.session_state["backtest_result"] = result
        st.session_state["backtest_config"] = {"strategy": "Mean Reversion", "capital": 1_000_000}

    result = st.session_state["backtest_result"]
    config_info = st.session_state.get("backtest_config", {"strategy": "N/A", "capital": 1_000_000})
    metrics = result.metrics

    # ── Report Type Selection ──────────────────────────────────────────────
    st.markdown("### 📄 Generate Report")

    report_col1, report_col2 = st.columns([2, 1])
    with report_col1:
        report_type = st.selectbox("Report Format", ["HTML Report", "CSV Export", "Metrics Summary"])
    with report_col2:
        include_trades = st.checkbox("Include Trade Log", True)
        include_charts_data = st.checkbox("Include Equity Data", True)

    if report_type == "HTML Report":
        html_content = _generate_html_report(result, config_info)
        st.download_button(
            "📥 Download HTML Report",
            html_content,
            f"stock_agent_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            "text/html",
            width='stretch',
        )
        # Preview
        with st.expander("👁️ Report Preview"):
            st.components.v1.html(html_content, height=600, scrolling=True)

    elif report_type == "CSV Export":
        csv_parts = {}

        # Metrics CSV
        summary = result.summary()
        metrics_df = pd.DataFrame(summary.items(), columns=["Metric", "Value"])
        csv_parts["metrics"] = metrics_df.to_csv(index=False)

        # Trades CSV
        if include_trades and result.trades:
            trades_df = trades_to_dataframe(result.trades)
            csv_parts["trades"] = trades_df.to_csv(index=False)

        # Equity CSV
        if include_charts_data:
            eq_df = pd.DataFrame({"bar": range(len(result.equity_curve)),
                                   "equity": result.equity_curve.values})
            csv_parts["equity"] = eq_df.to_csv(index=False)

        for name, csv_data in csv_parts.items():
            st.download_button(
                f"📥 Download {name.title()}.csv",
                csv_data,
                f"{name}_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                key=f"dl_{name}",
            )

    else:  # Metrics Summary
        summary = result.summary()
        st.json(summary)

    st.markdown("---")

    # ── Live Report Preview ────────────────────────────────────────────────
    st.markdown("### 📊 Report Dashboard")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Return", f"{metrics.total_return_pct:.2f}%")
    with k2:
        st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
    with k3:
        st.metric("Max Drawdown", f"{metrics.max_drawdown_pct:.1f}%")
    with k4:
        st.metric("Total Trades", metrics.total_trades)

    # Charts
    chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
        "📈 Equity", "📊 Returns", "📅 Monthly", "💰 Trades",
    ])

    with chart_tab1:
        dd = compute_drawdown(result.equity_curve)
        fig = create_equity_curve(result.equity_curve, dd)
        st.plotly_chart(fig, width='stretch')

    with chart_tab2:
        rets = compute_returns(result.equity_curve)
        fig_dist = create_returns_distribution(rets, "Daily Returns Distribution")
        st.plotly_chart(fig_dist, width='stretch')

    with chart_tab3:
        fig_hm = create_monthly_returns_heatmap(result.equity_curve)
        st.plotly_chart(fig_hm, width='stretch')

    with chart_tab4:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_sc = create_trade_scatter(result.trades)
            st.plotly_chart(fig_sc, width='stretch')
        with col_b:
            fig_cum = create_cumulative_pnl(result.trades)
            st.plotly_chart(fig_cum, width='stretch')

    # Full metrics table
    with st.expander("📋 Complete Metrics Table"):
        summary = result.summary()
        st.dataframe(
            pd.DataFrame(summary.items(), columns=["Metric", "Value"]),
            width='stretch', hide_index=True,
        )
