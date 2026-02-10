"""Backtest report generation (HTML + Markdown)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.engine import BacktestResult
from src.core.logging_utils import get_logger

logger = get_logger("backtest.reports")


def generate_html_report(result: BacktestResult, output_path: str | Path) -> Path:
    """Generate an HTML backtest report with charts and metrics.

    Args:
        result: BacktestResult from the backtest engine.
        output_path: Output file path.

    Returns:
        Path to the generated report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = result.summary()

    # Build trade table
    trade_rows = []
    for t in result.trades:
        trade_rows.append({
            "Symbol": t.symbol,
            "Side": t.side,
            "Entry": f"{t.entry_price:.2f}",
            "Exit": f"{t.exit_price:.2f}",
            "Qty": t.quantity,
            "PnL": f"₹{t.net_pnl:,.2f}",
            "Costs": f"₹{t.costs:,.2f}",
            "Bars": t.holding_bars,
        })

    trades_html = ""
    if trade_rows:
        trades_df = pd.DataFrame(trade_rows)
        trades_html = trades_df.to_html(index=False, classes="table", border=0)

    # Equity curve as inline SVG (simple line chart using matplotlib)
    equity_svg = _render_equity_svg(result.equity_curve)
    dd_svg = _render_drawdown_svg(result.drawdown)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report - {result.config.strategy}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                    gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: white; padding: 15px; border-radius: 8px;
                       box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-card .label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
        .metric-card .value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .table {{ width: 100%; border-collapse: collapse; background: white;
                  border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .table th {{ background: #3498db; color: white; padding: 12px; text-align: left; }}
        .table td {{ padding: 10px; border-bottom: 1px solid #ecf0f1; }}
        .chart {{ background: white; padding: 20px; border-radius: 8px;
                  box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0; }}
        .disclaimer {{ background: #fff3cd; padding: 15px; border-radius: 8px;
                       border-left: 4px solid #ffc107; margin-top: 30px; }}
    </style>
</head>
<body>
    <h1>📊 Backtest Report</h1>
    <p>Strategy: <strong>{result.config.strategy}</strong></p>

    <h2>Performance Metrics</h2>
    <div class="metrics">
        {"".join(f'<div class="metric-card"><div class="label">{k}</div><div class="value">{v}</div></div>' for k, v in summary.items())}
    </div>

    <h2>Equity Curve</h2>
    <div class="chart">{equity_svg}</div>

    <h2>Drawdown</h2>
    <div class="chart">{dd_svg}</div>

    <h2>Trade Log ({len(result.trades)} trades)</h2>
    {trades_html}

    <div class="disclaimer">
        <strong>⚠️ Disclaimer:</strong> This is a historical backtest for educational purposes only.
        Past performance does not guarantee future results. Not investment advice.
    </div>
</body>
</html>"""

    output_path.write_text(html)
    logger.info(f"HTML report saved to {output_path}")
    return output_path


def _render_equity_svg(equity: pd.Series) -> str:
    """Render equity curve as inline SVG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io
        import base64

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(equity.values, color="#3498db", linewidth=1.5)
        ax.set_title("Equity Curve")
        ax.set_ylabel("Portfolio Value (₹)")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="plain", axis="y")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        return f'<img src="data:image/png;base64,{img_b64}" style="width:100%"/>'
    except Exception:
        return "<p>Chart rendering requires matplotlib</p>"


def _render_drawdown_svg(drawdown: pd.Series) -> str:
    """Render drawdown chart as inline SVG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io
        import base64

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.fill_between(range(len(drawdown)), drawdown.values * 100, color="#e74c3c", alpha=0.5)
        ax.set_title("Drawdown (%)")
        ax.set_ylabel("Drawdown %")
        ax.grid(True, alpha=0.3)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        return f'<img src="data:image/png;base64,{img_b64}" style="width:100%"/>'
    except Exception:
        return "<p>Chart rendering requires matplotlib</p>"


def generate_markdown_report(result: BacktestResult, output_path: str | Path) -> Path:
    """Generate a Markdown backtest report.

    Args:
        result: BacktestResult from the backtest engine.
        output_path: Output file path.

    Returns:
        Path to the generated report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = result.summary()

    lines = [
        f"# Backtest Report: {result.config.strategy}",
        "",
        "## Performance Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    for k, v in summary.items():
        lines.append(f"| {k} | {v} |")

    lines.extend([
        "",
        "## Trade Log",
        "",
        "| Symbol | Side | Entry | Exit | Qty | Net PnL | Bars |",
        "|--------|------|-------|------|-----|---------|------|",
    ])

    for t in result.trades[:50]:  # First 50 trades
        lines.append(
            f"| {t.symbol} | {t.side} | {t.entry_price:.2f} | {t.exit_price:.2f} | "
            f"{t.quantity} | ₹{t.net_pnl:,.2f} | {t.holding_bars} |"
        )

    if len(result.trades) > 50:
        lines.append(f"\n_...and {len(result.trades) - 50} more trades_")

    lines.extend([
        "",
        "---",
        "",
        "> **Disclaimer:** Historical backtest for educational purposes only. Not investment advice.",
    ])

    output_path.write_text("\n".join(lines))
    logger.info(f"Markdown report saved to {output_path}")
    return output_path
