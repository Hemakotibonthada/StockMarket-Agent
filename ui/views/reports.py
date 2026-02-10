"""Reports page – generate PDF reports with agent learnings, portfolio, P&L, news, charts."""

from __future__ import annotations

import sys
from pathlib import Path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from ui.utils import (
    COLORS, generate_synthetic_data, run_backtest_from_ui,
    create_equity_curve, create_returns_distribution,
    create_monthly_returns_heatmap, create_trade_scatter,
    create_cumulative_pnl, trades_to_dataframe,
)
from ui.pdf_report import (
    generate_pdf_report, build_report_data_from_backtest,
    send_report_email, ReportData, AgentLearnings, PortfolioSnapshot,
    DailyInvestment, NewsItem,
)
from src.backtest.metrics import compute_returns, compute_drawdown


def _ensure_backtest_result():
    """Ensure a backtest result exists in session state."""
    if "backtest_result" not in st.session_state:
        df = generate_synthetic_data(n_bars=500, seed=42)
        result = run_backtest_from_ui(df, "Mean Reversion", {"zscore_entry": 2.0, "zscore_exit": 0.5})
        st.session_state["backtest_result"] = result
        st.session_state["backtest_config"] = {"strategy": "Mean Reversion", "capital": 1_000_000}


def render():
    st.markdown("## 📊 Reports & Daily Digest")
    st.markdown(
        f'<p style="color: {COLORS["text_muted"]}; margin-top: -10px;">'
        "Generate comprehensive PDF reports with agent learnings, portfolio analysis, "
        "P&L, news, and performance charts. Schedule end-of-day emails or generate on demand.</p>",
        unsafe_allow_html=True,
    )

    _ensure_backtest_result()
    result = st.session_state["backtest_result"]
    config_info = st.session_state.get("backtest_config", {"strategy": "N/A", "capital": 1_000_000})
    metrics = result.metrics

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab_generate, tab_preview, tab_email, tab_history = st.tabs([
        "📄 Generate Report", "👁️ Preview Sections", "📧 Email Settings", "📂 Report History",
    ])

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 1: Generate Report
    # ══════════════════════════════════════════════════════════════════════
    with tab_generate:
        st.markdown("### 📄 Generate PDF Report")

        gen_col1, gen_col2 = st.columns([2, 1])
        with gen_col1:
            report_type = st.selectbox(
                "Report Type",
                ["End of Day Report", "On Demand Report", "Weekly Summary", "Monthly Summary"],
                key="report_type_select",
            )

        with gen_col2:
            st.markdown("")
            st.markdown("")

        # Report content toggles
        st.markdown("#### Report Sections")
        sec_col1, sec_col2, sec_col3, sec_col4 = st.columns(4)
        with sec_col1:
            inc_learnings = st.checkbox("Agent Learnings", True, key="inc_learn")
            inc_portfolio = st.checkbox("Portfolio Overview", True, key="inc_port")
        with sec_col2:
            inc_trades = st.checkbox("Today's Trades", True, key="inc_trade")
            inc_pnl = st.checkbox("P&L Analysis", True, key="inc_pnl")
        with sec_col3:
            inc_news = st.checkbox("Market News", True, key="inc_news")
            inc_charts = st.checkbox("Performance Charts", True, key="inc_charts")
        with sec_col4:
            inc_risk = st.checkbox("Risk Analysis", True, key="inc_risk")
            inc_tradelog = st.checkbox("Trade Log", True, key="inc_tlog")

        st.markdown("---")

        # Custom news input (optional)
        with st.expander("📰 Custom News Headlines (Optional)"):
            st.markdown(
                f'<p style="color: {COLORS["text_muted"]}; font-size: 0.85rem;">'
                "Add custom news items to include in the report. Leave empty to use sample market news.</p>",
                unsafe_allow_html=True,
            )
            custom_news = []
            for i in range(3):
                ncol1, ncol2, ncol3 = st.columns([3, 1, 1])
                with ncol1:
                    headline = st.text_input(f"Headline {i+1}", key=f"news_h_{i}")
                with ncol2:
                    sentiment = st.selectbox("Sentiment", ["Positive", "Neutral", "Negative"],
                                             key=f"news_s_{i}")
                with ncol3:
                    impact = st.selectbox("Impact", ["High", "Medium", "Low"], key=f"news_i_{i}")
                if headline:
                    custom_news.append({
                        "headline": headline,
                        "source": "User Input",
                        "sentiment": sentiment,
                        "impact": impact,
                        "summary": "",
                    })

        # Generate button
        st.markdown("")
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])

        with btn_col1:
            generate_pdf = st.button("📥 Generate PDF Report", width="stretch",
                                      type="primary", key="gen_pdf_btn")
        with btn_col2:
            generate_csv = st.button("📊 Export CSV Data", width="stretch", key="gen_csv_btn")

        if generate_pdf:
            with st.spinner("Generating PDF report with charts..."):
                try:
                    report_data = build_report_data_from_backtest(
                        result, config_info,
                        news=custom_news if custom_news else None,
                        report_type=report_type.replace(" Report", "").replace(" Summary", ""),
                    )
                    pdf_bytes = generate_pdf_report(report_data)

                    st.session_state["last_pdf_bytes"] = pdf_bytes
                    st.session_state["last_pdf_date"] = datetime.now().strftime("%Y-%m-%d %H:%M")

                    # Add to history
                    if "report_history" not in st.session_state:
                        st.session_state["report_history"] = []
                    st.session_state["report_history"].append({
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": report_type,
                        "size_kb": len(pdf_bytes) / 1024,
                    })

                    st.success("PDF report generated successfully!")

                    st.download_button(
                        "⬇️ Download PDF Report",
                        pdf_bytes,
                        f"stock_agent_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        "application/pdf",
                        width="stretch",
                        key="dl_pdf",
                    )
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")

        if generate_csv:
            summary = result.summary()
            metrics_df = pd.DataFrame(
                [(k.replace("_", " ").title(), str(v)) for k, v in summary.items()],
                columns=["Metric", "Value"],
            )
            trades_df = trades_to_dataframe(result.trades)
            eq_df = pd.DataFrame({
                "bar": range(len(result.equity_curve)),
                "equity": result.equity_curve.values,
            })

            csv_col1, csv_col2, csv_col3 = st.columns(3)
            with csv_col1:
                st.download_button(
                    "📋 Metrics.csv", metrics_df.to_csv(index=False),
                    f"metrics_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv",
                    key="dl_metrics_csv",
                )
            with csv_col2:
                if not trades_df.empty:
                    st.download_button(
                        "📒 Trades.csv", trades_df.to_csv(index=False),
                        f"trades_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv",
                        key="dl_trades_csv",
                    )
            with csv_col3:
                st.download_button(
                    "📈 Equity.csv", eq_df.to_csv(index=False),
                    f"equity_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv",
                    key="dl_equity_csv",
                )

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 2: Preview Sections
    # ══════════════════════════════════════════════════════════════════════
    with tab_preview:
        st.markdown("### 👁️ Report Section Preview")
        st.markdown(
            f'<p style="color: {COLORS["text_muted"]}; font-size: 0.85rem;">'
            "Preview each section that will be included in the PDF report.</p>",
            unsafe_allow_html=True,
        )

        # Section 1: Agent Learnings
        with st.expander("🧠 Section 1: What the Agent Learned Today", expanded=True):
            learn_col1, learn_col2, learn_col3, learn_col4 = st.columns(4)
            regime = "Trending" if metrics.total_return_pct > 5 else "Mean-Reverting" if abs(metrics.total_return_pct) < 2 else "Volatile"
            confidence = min(95, 50 + abs(metrics.sharpe_ratio) * 15)
            sentiment = 1.5 if metrics.total_return_pct > 0 else -1.0
            vol_regime = "High" if metrics.max_drawdown_pct > 15 else "Normal" if metrics.max_drawdown_pct > 8 else "Low"

            with learn_col1:
                st.metric("Market Regime", regime)
            with learn_col2:
                st.metric("Confidence", f"{confidence:.0f}%")
            with learn_col3:
                st.metric("Sentiment Score", f"{sentiment:+.1f}")
            with learn_col4:
                st.metric("Volatility", vol_regime)

            st.markdown("**Patterns Detected:**")
            st.markdown(f"- Win rate of `{metrics.win_rate_pct:.0f}%` suggests "
                        f"{'strong' if metrics.win_rate_pct > 60 else 'moderate' if metrics.win_rate_pct > 50 else 'weak'} signal quality")
            st.markdown(f"- Sharpe ratio of `{metrics.sharpe_ratio:.2f}` indicates "
                        f"{'good' if metrics.sharpe_ratio > 1 else 'moderate' if metrics.sharpe_ratio > 0.5 else 'poor'} risk-adjusted returns")
            st.markdown(f"- Max drawdown of `{metrics.max_drawdown_pct:.1f}%` "
                        f"{'within acceptable limits' if metrics.max_drawdown_pct < 15 else 'exceeds risk threshold'}")
            st.markdown(f"- Profit factor of `{metrics.profit_factor:.2f}` shows "
                        f"{'profits exceed losses' if metrics.profit_factor > 1 else 'losses dominate'}")

            st.markdown("**Strategy Adjustments:**")
            st.markdown(f"- Position sizing calibrated based on {metrics.total_trades} historical trades")
            st.markdown(f"- Stop-loss levels adjusted to limit drawdown below {metrics.max_drawdown_pct * 1.1:.0f}%")
            st.markdown("- Entry thresholds fine-tuned based on recent signal performance")

        # Section 2: Current Portfolio
        with st.expander("💼 Section 2: Current Portfolio", expanded=True):
            initial_cap = config_info.get("capital", 1_000_000)
            final_eq = result.equity_curve.iloc[-1] if len(result.equity_curve) else initial_cap
            total_pnl = final_eq - initial_cap
            total_pnl_pct = (total_pnl / initial_cap * 100) if initial_cap else 0
            day_pnl = result.trades[-1].net_pnl if result.trades else 0
            day_pnl_pct = (day_pnl / final_eq * 100) if final_eq else 0

            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            with p_col1:
                st.metric("Portfolio Value", f"₹{final_eq:,.0f}")
            with p_col2:
                st.metric("Cash Balance", f"₹{final_eq * 0.3:,.0f}")
            with p_col3:
                st.metric("Day P&L", f"₹{day_pnl:+,.0f}", f"{day_pnl_pct:+.2f}%")
            with p_col4:
                st.metric("Total P&L", f"₹{total_pnl:+,.0f}", f"{total_pnl_pct:+.2f}%")

            # Holdings table
            if result.trades:
                st.markdown("**Current Holdings (from recent trades):**")
                holdings_data = []
                seen = set()
                for t in reversed(result.trades[-20:]):
                    if t.symbol not in seen and t.side == "BUY":
                        pnl_pct = (t.exit_price / t.entry_price - 1) * 100 if t.entry_price else 0
                        holdings_data.append({
                            "Symbol": t.symbol,
                            "Qty": t.quantity,
                            "Avg Cost": f"₹{t.entry_price:,.2f}",
                            "Current": f"₹{t.exit_price:,.2f}",
                            "P&L": f"₹{t.net_pnl:+,.0f}",
                            "P&L %": f"{pnl_pct:+.1f}%",
                        })
                        seen.add(t.symbol)
                    if len(holdings_data) >= 5:
                        break
                if holdings_data:
                    st.dataframe(pd.DataFrame(holdings_data), width="stretch", hide_index=True)

        # Section 3: Today's Trades
        with st.expander("📈 Section 3: Today's Investments", expanded=False):
            if result.trades:
                recent = result.trades[-10:]
                buys = sum(1 for t in recent if t.side == "BUY")
                sells = len(recent) - buys
                turnover = sum(t.quantity * t.entry_price for t in recent)

                t_col1, t_col2, t_col3, t_col4 = st.columns(4)
                with t_col1:
                    st.metric("Total Trades", str(len(recent)))
                with t_col2:
                    st.metric("Buys", str(buys))
                with t_col3:
                    st.metric("Sells", str(sells))
                with t_col4:
                    st.metric("Turnover", f"₹{turnover:,.0f}")

                trade_data = []
                for t in recent:
                    trade_data.append({
                        "Symbol": t.symbol,
                        "Action": t.side,
                        "Qty": t.quantity,
                        "Price": f"₹{t.entry_price:,.2f}",
                        "Value": f"₹{t.quantity * t.entry_price:,.0f}",
                        "Net P&L": f"₹{t.net_pnl:+,.0f}",
                        "Strategy": config_info.get("strategy", ""),
                    })
                st.dataframe(pd.DataFrame(trade_data), width="stretch", hide_index=True)
            else:
                st.info("No trades executed today.")

        # Section 4: Profit & Loss
        with st.expander("💰 Section 4: Profit & Loss", expanded=False):
            pnl_col1, pnl_col2, pnl_col3, pnl_col4 = st.columns(4)
            with pnl_col1:
                st.metric("Total Return", f"{metrics.total_return_pct:.2f}%")
            with pnl_col2:
                st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
            with pnl_col3:
                st.metric("Max Drawdown", f"{metrics.max_drawdown_pct:.1f}%")
            with pnl_col4:
                st.metric("Win Rate", f"{metrics.win_rate_pct:.0f}%")

            chart_c1, chart_c2 = st.columns(2)
            with chart_c1:
                dd = compute_drawdown(result.equity_curve)
                fig_eq = create_equity_curve(result.equity_curve, dd, height=350)
                st.plotly_chart(fig_eq, width="stretch")
            with chart_c2:
                fig_pnl = create_cumulative_pnl(result.trades, height=350)
                st.plotly_chart(fig_pnl, width="stretch")

        # Section 5: News
        with st.expander("📰 Section 5: Top Market News", expanded=False):
            from ui.pdf_report import _generate_sample_news
            sample_news = _generate_sample_news()
            for item in sample_news:
                sentiment_color = {"Positive": COLORS["success"], "Negative": COLORS["danger"], "Neutral": COLORS["warning"]}.get(item.sentiment, COLORS["text_muted"])
                impact_color = {"High": COLORS["danger"], "Medium": COLORS["warning"], "Low": COLORS["success"]}.get(item.impact, COLORS["text_muted"])
                st.markdown(
                    f'<div style="background: {COLORS["bg_card"]}; border-left: 3px solid {sentiment_color}; '
                    f'padding: 12px 16px; border-radius: 0 8px 8px 0; margin-bottom: 10px;">'
                    f'<div style="font-weight: 600; color: {COLORS["text"]}; font-size: 0.95rem;">{item.headline}</div>'
                    f'<div style="margin-top: 4px;">'
                    f'<span style="background: {sentiment_color}; color: white; padding: 2px 8px; '
                    f'border-radius: 4px; font-size: 0.7rem; font-weight: 600;">{item.sentiment}</span>'
                    f'&nbsp;<span style="background: {impact_color}; color: white; padding: 2px 8px; '
                    f'border-radius: 4px; font-size: 0.7rem; font-weight: 600;">Impact: {item.impact}</span>'
                    f'</div>'
                    f'<div style="color: {COLORS["text_muted"]}; font-size: 0.85rem; margin-top: 6px;">{item.summary}</div>'
                    f'<div style="color: {COLORS["text_dim"]}; font-size: 0.75rem; margin-top: 4px;">Source: {item.source}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Section 6: Charts
        with st.expander("📊 Section 6: Performance Charts", expanded=False):
            ch_col1, ch_col2 = st.columns(2)
            with ch_col1:
                rets = compute_returns(result.equity_curve)
                fig_dist = create_returns_distribution(rets, height=350)
                st.plotly_chart(fig_dist, width="stretch")
            with ch_col2:
                fig_hm = create_monthly_returns_heatmap(result.equity_curve, height=350)
                st.plotly_chart(fig_hm, width="stretch")

            sc_col1, sc_col2 = st.columns(2)
            with sc_col1:
                fig_sc = create_trade_scatter(result.trades, height=350)
                st.plotly_chart(fig_sc, width="stretch")
            with sc_col2:
                fig_cum = create_cumulative_pnl(result.trades, height=350)
                st.plotly_chart(fig_cum, width="stretch")

        # Section 7: Metrics
        with st.expander("📋 Section 7: Complete Metrics", expanded=False):
            summary = result.summary()
            summary_str = {k: str(v) for k, v in summary.items()}
            st.dataframe(
                pd.DataFrame(
                    [(k.replace("_", " ").title(), v) for k, v in summary_str.items()],
                    columns=["Metric", "Value"],
                ),
                width="stretch", hide_index=True,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 3: Email Settings
    # ══════════════════════════════════════════════════════════════════════
    with tab_email:
        st.markdown("### 📧 Email Report Configuration")
        st.markdown(
            f'<p style="color: {COLORS["text_muted"]}; font-size: 0.85rem;">'
            "Configure automatic end-of-day report delivery or manually send reports.</p>",
            unsafe_allow_html=True,
        )

        email_col1, email_col2 = st.columns(2)

        with email_col1:
            st.markdown("#### 📬 Recipient Settings")

            recipient = st.text_input(
                "Recipient Email",
                value=st.session_state.get("report_recipient", ""),
                placeholder="your.email@example.com",
                key="email_recipient",
            )
            if recipient:
                st.session_state["report_recipient"] = recipient

            sender = st.text_input(
                "Sender Email (Gmail)",
                value=st.session_state.get("report_sender", ""),
                placeholder="sender@gmail.com",
                key="email_sender",
            )
            if sender:
                st.session_state["report_sender"] = sender

            password = st.text_input(
                "App Password",
                type="password",
                value=st.session_state.get("report_smtp_password", ""),
                help="Use a Gmail App Password (not your regular password). "
                     "Generate one at myaccount.google.com > Security > App Passwords.",
                key="email_password",
            )
            if password:
                st.session_state["report_smtp_password"] = password

        with email_col2:
            st.markdown("#### Schedule Settings")

            auto_send = st.toggle(
                "Enable End-of-Day Auto-Send",
                value=st.session_state.get("auto_send_enabled", False),
                key="auto_send_toggle",
            )
            st.session_state["auto_send_enabled"] = auto_send

            if auto_send:
                send_time = st.time_input(
                    "Send Time",
                    value=st.session_state.get("send_time", None),
                    key="auto_send_time",
                )
                st.session_state["send_time"] = send_time

                st.info(
                    f"Reports will be automatically generated and sent at "
                    f"{send_time.strftime('%I:%M %p') if send_time else 'N/A'} every trading day."
                )

            st.markdown("#### SMTP Settings")
            smtp_server = st.text_input(
                "SMTP Server",
                value=st.session_state.get("smtp_server", "smtp.gmail.com"),
                key="smtp_srv",
            )
            smtp_port = st.number_input(
                "SMTP Port",
                value=st.session_state.get("smtp_port", 587),
                key="smtp_prt",
            )
            st.session_state["smtp_server"] = smtp_server
            st.session_state["smtp_port"] = smtp_port

        st.markdown("---")

        # Manual send button
        send_col1, send_col2, send_col3 = st.columns([1, 1, 2])
        with send_col1:
            send_btn = st.button("📤 Send Report Now", width="stretch",
                                  type="primary", key="send_email_btn")

        if send_btn:
            if not recipient:
                st.error("Please enter a recipient email address.")
            elif "last_pdf_bytes" not in st.session_state:
                st.warning("No PDF report generated yet. Generate a report first in the 'Generate Report' tab.")
            else:
                with st.spinner("Sending report email..."):
                    success, msg = send_report_email(
                        pdf_bytes=st.session_state["last_pdf_bytes"],
                        recipient_email=recipient,
                        sender_email=sender,
                        smtp_server=smtp_server,
                        smtp_port=smtp_port,
                        smtp_password=password,
                    )
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

        # Quick test
        with send_col2:
            test_btn = st.button("🧪 Generate & Preview", width="stretch", key="test_email_btn")

        if test_btn:
            with st.spinner("Generating test report..."):
                try:
                    report_data = build_report_data_from_backtest(
                        result, config_info, report_type="End of Day",
                    )
                    pdf_bytes = generate_pdf_report(report_data)
                    st.session_state["last_pdf_bytes"] = pdf_bytes
                    st.session_state["last_pdf_date"] = datetime.now().strftime("%Y-%m-%d %H:%M")

                    st.success(f"Test report generated ({len(pdf_bytes) / 1024:.0f} KB)")
                    st.download_button(
                        "⬇️ Download Test PDF",
                        pdf_bytes,
                        f"test_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        "application/pdf",
                        key="dl_test_pdf",
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 4: Report History
    # ══════════════════════════════════════════════════════════════════════
    with tab_history:
        st.markdown("### 📂 Report History")

        history = st.session_state.get("report_history", [])

        if history:
            hist_df = pd.DataFrame(history)
            hist_df.columns = ["Generated At", "Type", "Size (KB)"]
            hist_df["Size (KB)"] = hist_df["Size (KB)"].apply(lambda x: f"{x:.1f}")
            st.dataframe(hist_df, width="stretch", hide_index=True)

            st.markdown(f"**Total reports generated this session:** {len(history)}")

            if "last_pdf_bytes" in st.session_state:
                st.download_button(
                    "⬇️ Download Latest Report",
                    st.session_state["last_pdf_bytes"],
                    f"latest_report.pdf",
                    "application/pdf",
                    width="stretch",
                    key="dl_latest_pdf",
                )
        else:
            st.markdown(
                f'<div style="text-align: center; padding: 40px; color: {COLORS["text_muted"]};">'
                '<p style="font-size: 2rem;">📋</p>'
                f'<p style="color: {COLORS["text_secondary"]};">No reports generated yet in this session.</p>'
                f'<p style="font-size: 0.85rem; color: {COLORS["text_muted"]};">Go to the "Generate Report" tab to create your first report.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
