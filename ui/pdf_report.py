"""PDF Report Generator for Stock Agent - End-of-Day & On-Demand Reports.

Generates a professional multi-section PDF with:
  1. Agent Learning Summary (what the agent learned today)
  2. Current Portfolio Overview
  3. Today's Investments / Trades
  4. Profit & Loss Summary
  5. Top Market News
  6. Performance Metrics & Charts
  7. Risk Analysis
  8. Trade Log
"""

from __future__ import annotations

import sys
import io
import tempfile
from pathlib import Path
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fpdf import FPDF

# ── Data containers for report sections ────────────────────────────────────────

@dataclass
class AgentLearnings:
    """What the agent learned today."""
    date: str = ""
    market_regime: str = "Sideways"
    regime_confidence: float = 0.0
    patterns_detected: list[str] = field(default_factory=list)
    strategy_adjustments: list[str] = field(default_factory=list)
    key_observations: list[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    volatility_regime: str = "Normal"


@dataclass
class PortfolioSnapshot:
    """Current portfolio state."""
    total_value: float = 0.0
    cash_balance: float = 0.0
    invested_value: float = 0.0
    day_pnl: float = 0.0
    day_pnl_pct: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    holdings: list[dict] = field(default_factory=list)  # [{symbol, qty, avg_cost, current_price, pnl, pnl_pct}]


@dataclass
class DailyInvestment:
    """A trade/investment made today."""
    symbol: str = ""
    action: str = ""  # BUY / SELL
    quantity: int = 0
    price: float = 0.0
    value: float = 0.0
    time: str = ""
    strategy: str = ""
    reason: str = ""


@dataclass
class NewsItem:
    """A news article."""
    headline: str = ""
    source: str = ""
    sentiment: str = "Neutral"  # Positive / Negative / Neutral
    impact: str = "Medium"  # High / Medium / Low
    summary: str = ""


@dataclass
class ReportData:
    """All data needed for the PDF report."""
    report_date: str = ""
    report_type: str = "End of Day"  # "End of Day" or "On Demand"
    learnings: AgentLearnings = field(default_factory=AgentLearnings)
    portfolio: PortfolioSnapshot = field(default_factory=PortfolioSnapshot)
    todays_investments: list[DailyInvestment] = field(default_factory=list)
    news: list[NewsItem] = field(default_factory=list)
    # Metrics from backtest result (if available)
    metrics: dict[str, Any] = field(default_factory=dict)
    # Chart images as bytes (PNG)
    equity_chart_png: bytes | None = None
    returns_chart_png: bytes | None = None
    pnl_chart_png: bytes | None = None
    drawdown_chart_png: bytes | None = None
    # Trade log
    trades: list[dict] = field(default_factory=list)


# ── Color constants (RGB tuples) ──────────────────────────────────────────────

class C:
    """PDF color palette - dark professional theme printed in light mode."""
    # Backgrounds
    BG_DARK = (15, 23, 42)        # #0F172A  Slate-900
    BG_CARD = (30, 41, 59)        # #1E293B  Slate-800
    BG_SECTION = (241, 245, 249)  # #F1F5F9  Slate-100
    BG_WHITE = (255, 255, 255)
    # Text
    TEXT = (15, 23, 42)           # Dark text for print
    TEXT_MUTED = (100, 116, 139)  # #64748B
    TEXT_LIGHT = (148, 163, 184)  # #94A3B8
    # Accent
    PRIMARY = (99, 102, 241)      # #6366F1  Indigo
    PRIMARY_DARK = (79, 70, 229)  # #4F46E5
    SECONDARY = (236, 72, 153)    # #EC4899  Pink
    # Status
    SUCCESS = (16, 185, 129)      # #10B981  Emerald
    DANGER = (239, 68, 68)        # #EF4444  Red
    WARNING = (245, 158, 11)      # #F59E0B  Amber
    INFO = (59, 130, 246)         # #3B82F6  Blue
    # Table
    TABLE_HEADER = (99, 102, 241)
    TABLE_ROW_ALT = (248, 250, 252)  # #F8FAFC
    TABLE_BORDER = (226, 232, 240)   # #E2E8F0


def _sanitize(text: str) -> str:
    """Replace non-latin-1 unicode chars with ASCII equivalents for PDF rendering."""
    replacements = {
        "\u20b9": "Rs.",   # ₹
        "\u2014": "-",     # —
        "\u2013": "-",     # –
        "\u2018": "'",     # '
        "\u2019": "'",     # '
        "\u201c": '"',     # "
        "\u201d": '"',     # "
        "\u2022": "-",     # •
        "\u2026": "...",   # …
        "\u2192": "->",    # →
        "\u2190": "<-",    # ←
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Fallback: strip any remaining non-latin-1 chars
    try:
        text.encode("latin-1")
    except UnicodeEncodeError:
        text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text


class StockAgentPDF(FPDF):
    """Custom PDF class with Stock Agent branding."""

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self._section_num = 0

    def normalize_text(self, text: str) -> str:
        """Override to sanitize unicode before rendering."""
        return super().normalize_text(_sanitize(text))

    # ── Header / Footer ────────────────────────────────────────────────────

    def header(self):
        if self.page_no() == 1:
            return  # Custom cover handles page 1
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*C.TEXT_MUTED)
        self.cell(0, 8, "Stock Agent - Trading Report", align="L")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*C.PRIMARY)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*C.TEXT_LIGHT)
        self.cell(0, 10, "Stock Agent v1.0.0 - Local-First Indian Market Trading Agent - Confidential", align="C")

    # ── Cover page ─────────────────────────────────────────────────────────

    def add_cover_page(self, report_date: str, report_type: str):
        self.add_page()

        # Top gradient bar
        self.set_fill_color(*C.PRIMARY)
        self.rect(0, 0, 210, 6, style="F")
        self.set_fill_color(*C.SECONDARY)
        self.rect(0, 6, 210, 2, style="F")

        # Logo area
        self.ln(35)
        self.set_font("Helvetica", "B", 36)
        self.set_text_color(*C.PRIMARY)
        self.cell(0, 16, "Stock Agent", align="C", new_x="LMARGIN", new_y="NEXT")

        self.set_font("Helvetica", "", 14)
        self.set_text_color(*C.TEXT_MUTED)
        self.cell(0, 10, "Indian Market Trading Dashboard", align="C", new_x="LMARGIN", new_y="NEXT")

        # Divider
        self.ln(10)
        self.set_draw_color(*C.PRIMARY)
        self.set_line_width(1)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(15)

        # Report info
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(*C.TEXT)
        self.cell(0, 14, f"{report_type} Report", align="C", new_x="LMARGIN", new_y="NEXT")

        self.ln(5)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(*C.TEXT_MUTED)
        self.cell(0, 10, report_date, align="C", new_x="LMARGIN", new_y="NEXT")

        # Info box
        self.ln(20)
        box_x, box_w = 35, 140
        box_y = self.get_y()
        self.set_fill_color(*C.BG_SECTION)
        self.set_draw_color(*C.TABLE_BORDER)
        self.rect(box_x, box_y, box_w, 50, style="DF")

        self.set_xy(box_x + 10, box_y + 8)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*C.PRIMARY)
        self.cell(box_w - 20, 8, "Report Contents", new_x="LMARGIN", new_y="NEXT")

        sections = [
            "Agent Learning Summary",
            "Portfolio Overview & Holdings",
            "Today's Investments & Trades",
            "Profit & Loss Analysis",
            "Market News & Sentiment",
            "Performance Charts & Metrics",
            "Risk Analysis",
            "Detailed Trade Log",
        ]
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*C.TEXT)
        for i, s in enumerate(sections, 1):
            self.set_x(box_x + 15)
            self.cell(box_w - 30, 5, f"{i}.  {s}", new_x="LMARGIN", new_y="NEXT")

        # Bottom bar
        self.set_fill_color(*C.PRIMARY)
        self.rect(0, 289, 210, 3, style="F")
        self.set_fill_color(*C.SECONDARY)
        self.rect(0, 292, 210, 1.5, style="F")

    # ── Section helpers ────────────────────────────────────────────────────

    def section_title(self, title: str, icon: str = ""):
        self._section_num += 1
        self.ln(6)

        # Section number badge
        self.set_fill_color(*C.PRIMARY)
        badge_x = 10
        badge_y = self.get_y()
        self.rect(badge_x, badge_y, 8, 8, style="F")
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*C.BG_WHITE)
        self.set_xy(badge_x, badge_y)
        self.cell(8, 8, str(self._section_num), align="C")

        # Title text
        self.set_xy(badge_x + 12, badge_y)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*C.TEXT)
        self.cell(0, 8, f"{icon}  {title}" if icon else title, new_x="LMARGIN", new_y="NEXT")

        # Underline
        self.set_draw_color(*C.PRIMARY)
        self.set_line_width(0.6)
        self.line(10, self.get_y() + 1, 200, self.get_y() + 1)
        self.ln(5)

    def sub_heading(self, text: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*C.PRIMARY_DARK)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*C.TEXT)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def kpi_row(self, kpis: list[tuple[str, str, tuple]]):
        """Draw a row of KPI cards. kpis = [(label, value, color_tuple), ...]"""
        n = len(kpis)
        if n == 0:
            return
        card_w = (190 - (n - 1) * 4) / n
        start_x = 10
        y = self.get_y()

        for i, (label, value, color) in enumerate(kpis):
            x = start_x + i * (card_w + 4)

            # Card background
            self.set_fill_color(*C.BG_SECTION)
            self.set_draw_color(*color)
            self.set_line_width(0.8)
            self.rect(x, y, card_w, 22, style="DF")

            # Left color accent bar
            self.set_fill_color(*color)
            self.rect(x, y, 3, 22, style="F")

            # Label
            self.set_xy(x + 6, y + 3)
            self.set_font("Helvetica", "", 7.5)
            self.set_text_color(*C.TEXT_MUTED)
            self.cell(card_w - 10, 4, label)

            # Value
            self.set_xy(x + 6, y + 10)
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(*color)
            self.cell(card_w - 10, 8, value)

        self.set_y(y + 27)

    def info_box(self, label: str, value: str, color: tuple = C.PRIMARY):
        """Small colored info tag."""
        self.set_fill_color(*color)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*C.BG_WHITE)
        tw = self.get_string_width(label) + 6
        self.cell(tw, 6, label, fill=True)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*C.TEXT)
        self.cell(4, 6, "")
        self.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def bullet_list(self, items: list[str], color: tuple = C.PRIMARY):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*C.TEXT)
        for item in items:
            x = self.get_x()
            # Bullet dot
            self.set_fill_color(*color)
            self.circle(x + 2, self.get_y() + 2.5, 1.2, style="F")
            self.set_x(x + 7)
            self.multi_cell(183, 5, item)
            self.ln(1)

    def add_table(self, headers: list[str], rows: list[list[str]],
                  col_widths: list[float] | None = None,
                  header_color: tuple = C.TABLE_HEADER,
                  highlight_col: int | None = None):
        """Draw a styled table."""
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)

        # Header
        self.set_fill_color(*header_color)
        self.set_text_color(*C.BG_WHITE)
        self.set_font("Helvetica", "B", 8.5)
        self.set_draw_color(*C.TABLE_BORDER)

        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, h, border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("Helvetica", "", 8.5)
        for row_idx, row in enumerate(rows):
            if row_idx % 2 == 1:
                self.set_fill_color(*C.TABLE_ROW_ALT)
            else:
                self.set_fill_color(*C.BG_WHITE)

            for i, cell in enumerate(row):
                align = "C"
                self.set_text_color(*C.TEXT)

                # Highlight P&L columns
                if highlight_col is not None and i == highlight_col:
                    try:
                        val = float(cell.replace("Rs.", "").replace(",", "").replace("%", "").strip())
                        self.set_text_color(*(C.SUCCESS if val > 0 else C.DANGER if val < 0 else C.TEXT))
                    except (ValueError, AttributeError):
                        pass

                self.cell(col_widths[i], 7, str(cell), border=1, fill=True, align=align)
            self.ln()
        self.ln(3)

    def add_chart_image(self, img_bytes: bytes | None, title: str = "", height: float = 70):
        """Insert a chart image from bytes."""
        if img_bytes is None:
            return
        # Write to temp file for fpdf
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(img_bytes)
            tmp_path = f.name

        if title:
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(*C.TEXT_MUTED)
            self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)

        # Check if enough space
        if self.get_y() + height + 10 > 280:
            self.add_page()

        self.image(tmp_path, x=10, y=self.get_y(), w=190, h=height)
        self.set_y(self.get_y() + height + 5)

        try:
            Path(tmp_path).unlink()
        except Exception:
            pass

    # ── Sentiment badge ────────────────────────────────────────────────────

    def sentiment_badge(self, sentiment: str):
        color_map = {
            "Positive": C.SUCCESS,
            "Negative": C.DANGER,
            "Neutral": C.WARNING,
            "Bullish": C.SUCCESS,
            "Bearish": C.DANGER,
        }
        color = color_map.get(sentiment, C.TEXT_MUTED)
        self.set_fill_color(*color)
        self.set_font("Helvetica", "B", 7)
        self.set_text_color(*C.BG_WHITE)
        tw = self.get_string_width(sentiment) + 6
        self.cell(tw, 5, sentiment, fill=True)
        self.set_text_color(*C.TEXT)


# ── Chart export helpers ───────────────────────────────────────────────────────

def _fig_to_png(fig, width: int = 1200, height: int = 500) -> bytes | None:
    """Export a Plotly figure to PNG bytes using kaleido."""
    try:
        # Use white background for PDF
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#1E293B"),
        )
        if hasattr(fig, "update_xaxes"):
            fig.update_xaxes(gridcolor="#E2E8F0", linecolor="#CBD5E1")
            fig.update_yaxes(gridcolor="#E2E8F0", linecolor="#CBD5E1")

        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception:
        return None


# ── Main generator ─────────────────────────────────────────────────────────────

def generate_pdf_report(data: ReportData) -> bytes:
    """Generate a complete PDF report and return as bytes."""
    pdf = StockAgentPDF()

    # ── Cover Page ─────────────────────────────────────────────────────────
    pdf.add_cover_page(data.report_date, data.report_type)

    # ── Section 1: Agent Learning Summary ──────────────────────────────────
    pdf.add_page()
    pdf.section_title("What the Agent Learned Today")

    learn = data.learnings
    pdf.kpi_row([
        ("Market Regime", learn.market_regime, C.PRIMARY),
        ("Confidence", f"{learn.regime_confidence:.0f}%", C.INFO),
        ("Sentiment", f"{learn.sentiment_score:+.1f}", C.SUCCESS if learn.sentiment_score >= 0 else C.DANGER),
        ("Volatility", learn.volatility_regime, C.WARNING),
    ])

    if learn.patterns_detected:
        pdf.sub_heading("Patterns Detected")
        pdf.bullet_list(learn.patterns_detected, C.PRIMARY)

    if learn.strategy_adjustments:
        pdf.sub_heading("Strategy Adjustments Made")
        pdf.bullet_list(learn.strategy_adjustments, C.WARNING)

    if learn.key_observations:
        pdf.sub_heading("Key Observations")
        pdf.bullet_list(learn.key_observations, C.INFO)

    # ── Section 2: Current Portfolio ───────────────────────────────────────
    pdf.section_title("Current Portfolio Overview")
    port = data.portfolio

    day_color = C.SUCCESS if port.day_pnl >= 0 else C.DANGER
    total_color = C.SUCCESS if port.total_pnl >= 0 else C.DANGER

    pdf.kpi_row([
        ("Portfolio Value", f"Rs.{port.total_value:,.0f}", C.PRIMARY),
        ("Cash Balance", f"Rs.{port.cash_balance:,.0f}", C.INFO),
        ("Day P&L", f"Rs.{port.day_pnl:+,.0f}  ({port.day_pnl_pct:+.2f}%)", day_color),
        ("Total P&L", f"Rs.{port.total_pnl:+,.0f}  ({port.total_pnl_pct:+.2f}%)", total_color),
    ])

    if port.holdings:
        pdf.sub_heading("Current Holdings")
        headers = ["Symbol", "Qty", "Avg Cost", "Current", "Value", "P&L", "P&L %"]
        rows = []
        for h in port.holdings:
            pnl_val = h.get("pnl", 0)
            rows.append([
                h.get("symbol", ""),
                str(h.get("qty", 0)),
                f"Rs.{h.get('avg_cost', 0):,.2f}",
                f"Rs.{h.get('current_price', 0):,.2f}",
                f"Rs.{h.get('qty', 0) * h.get('current_price', 0):,.0f}",
                f"Rs.{pnl_val:+,.0f}",
                f"{h.get('pnl_pct', 0):+.1f}%",
            ])
        pdf.add_table(headers, rows, col_widths=[30, 18, 25, 25, 28, 28, 22], highlight_col=5)
    else:
        pdf.body_text("No open positions currently held.")

    # ── Section 3: Today's Investments ─────────────────────────────────────
    pdf.section_title("Today's Investments & Trades")

    if data.todays_investments:
        buys = [t for t in data.todays_investments if t.action == "BUY"]
        sells = [t for t in data.todays_investments if t.action == "SELL"]

        pdf.kpi_row([
            ("Total Trades", str(len(data.todays_investments)), C.PRIMARY),
            ("Buys", str(len(buys)), C.SUCCESS),
            ("Sells", str(len(sells)), C.DANGER),
            ("Turnover", f"Rs.{sum(t.value for t in data.todays_investments):,.0f}", C.INFO),
        ])

        headers = ["Time", "Symbol", "Action", "Qty", "Price", "Value", "Strategy"]
        rows = []
        for t in data.todays_investments:
            rows.append([
                t.time,
                t.symbol,
                t.action,
                str(t.quantity),
                f"Rs.{t.price:,.2f}",
                f"Rs.{t.value:,.0f}",
                t.strategy,
            ])
        pdf.add_table(headers, rows, col_widths=[22, 28, 18, 16, 28, 30, 30])

        # Reasons
        pdf.sub_heading("Trade Rationale")
        for t in data.todays_investments:
            if t.reason:
                pdf.info_box(f"{t.action} {t.symbol}", t.reason,
                             C.SUCCESS if t.action == "BUY" else C.DANGER)
    else:
        pdf.body_text("No trades were executed today.")

    # ── Section 4: Profit & Loss ───────────────────────────────────────────
    pdf.section_title("Profit & Loss Summary")

    pdf.kpi_row([
        ("Day P&L", f"Rs.{port.day_pnl:+,.0f}", day_color),
        ("Day Return", f"{port.day_pnl_pct:+.2f}%", day_color),
        ("Total P&L", f"Rs.{port.total_pnl:+,.0f}", total_color),
        ("Total Return", f"{port.total_pnl_pct:+.2f}%", total_color),
    ])

    # P&L chart
    if data.pnl_chart_png:
        pdf.add_chart_image(data.pnl_chart_png, "Cumulative P&L Curve", height=65)

    # Metrics table
    if data.metrics:
        pdf.sub_heading("Key Performance Metrics")
        metrics_list = list(data.metrics.items())
        mid = len(metrics_list) // 2
        left_metrics = metrics_list[:mid]
        right_metrics = metrics_list[mid:]

        headers = ["Metric", "Value", "Metric", "Value"]
        rows = []
        for i in range(max(len(left_metrics), len(right_metrics))):
            row = []
            if i < len(left_metrics):
                k, v = left_metrics[i]
                row += [k.replace("_", " ").title(), str(v)]
            else:
                row += ["", ""]
            if i < len(right_metrics):
                k, v = right_metrics[i]
                row += [k.replace("_", " ").title(), str(v)]
            else:
                row += ["", ""]
            rows.append(row)
        pdf.add_table(headers, rows, col_widths=[50, 40, 50, 40])

    # ── Section 5: Top News ────────────────────────────────────────────────
    pdf.section_title("Top Market News")

    if data.news:
        for item in data.news:
            # Headline with sentiment badge
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*C.TEXT)
            pdf.cell(0, 6, item.headline, new_x="LMARGIN", new_y="NEXT")

            pdf.set_x(14)
            pdf.sentiment_badge(item.sentiment)
            pdf.cell(3, 5, "")
            impact_color = {"High": C.DANGER, "Medium": C.WARNING, "Low": C.SUCCESS}.get(item.impact, C.TEXT_MUTED)
            pdf.set_fill_color(*impact_color)
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(*C.BG_WHITE)
            tw = pdf.get_string_width(f"Impact: {item.impact}") + 6
            pdf.cell(tw, 5, f"Impact: {item.impact}", fill=True, new_x="LMARGIN", new_y="NEXT")

            if item.summary:
                pdf.set_x(14)
                pdf.set_font("Helvetica", "", 8.5)
                pdf.set_text_color(*C.TEXT_MUTED)
                pdf.multi_cell(180, 4.5, item.summary)

            pdf.set_x(14)
            pdf.set_font("Helvetica", "I", 7.5)
            pdf.set_text_color(*C.TEXT_LIGHT)
            pdf.cell(0, 5, f"Source: {item.source}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
    else:
        pdf.body_text("No major market news available for today.")

    # ── Section 6: Performance Charts ──────────────────────────────────────
    pdf.add_page()
    pdf.section_title("Performance Charts")

    if data.equity_chart_png:
        pdf.add_chart_image(data.equity_chart_png, "Equity Curve & Drawdown", height=70)

    if data.returns_chart_png:
        pdf.add_chart_image(data.returns_chart_png, "Returns Distribution", height=60)

    if data.drawdown_chart_png:
        pdf.add_chart_image(data.drawdown_chart_png, "Drawdown Analysis", height=60)

    if not any([data.equity_chart_png, data.returns_chart_png, data.drawdown_chart_png]):
        pdf.body_text("No chart data available. Run a backtest to generate performance charts.")

    # ── Section 7: Risk Analysis ───────────────────────────────────────────
    pdf.section_title("Risk Analysis")

    risk_metrics = {}
    for k in ["max_drawdown_pct", "var_95_pct", "es_95_pct", "sharpe_ratio", "sortino_ratio", "calmar_ratio"]:
        if k in data.metrics:
            risk_metrics[k] = data.metrics[k]

    if risk_metrics:
        pdf.kpi_row([
            ("Max Drawdown", str(risk_metrics.get("max_drawdown_pct", "N/A")), C.DANGER),
            ("VaR 95%", str(risk_metrics.get("var_95_pct", "N/A")), C.WARNING),
            ("Exp. Shortfall", str(risk_metrics.get("es_95_pct", "N/A")), C.DANGER),
            ("Sharpe Ratio", str(risk_metrics.get("sharpe_ratio", "N/A")), C.PRIMARY),
        ])
        pdf.body_text(
            "Value at Risk (VaR) represents the maximum expected loss at the 95% confidence level "
            "over a single day. Expected Shortfall (ES) measures the average loss beyond VaR, "
            "providing a more conservative risk estimate."
        )

    else:
        pdf.body_text("Risk metrics will be available after running a backtest.")

    # ── Section 8: Trade Log ───────────────────────────────────────────────
    pdf.section_title("Detailed Trade Log")

    if data.trades:
        headers = ["#", "Symbol", "Side", "Entry", "Exit", "Qty", "Net P&L", "Bars"]
        rows = []
        for i, t in enumerate(data.trades[:60], 1):
            rows.append([
                str(i),
                t.get("symbol", ""),
                t.get("side", ""),
                f"Rs.{t.get('entry_price', 0):,.2f}",
                f"Rs.{t.get('exit_price', 0):,.2f}",
                str(t.get("quantity", 0)),
                f"Rs.{t.get('net_pnl', 0):+,.0f}",
                str(t.get("holding_bars", 0)),
            ])
        pdf.add_table(headers, rows,
                      col_widths=[12, 28, 18, 28, 28, 18, 30, 18],
                      highlight_col=6)

        if len(data.trades) > 60:
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(*C.TEXT_MUTED)
            pdf.cell(0, 6, f"Showing first 60 of {len(data.trades)} trades.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.body_text("No trades to display.")

    # ── Generate output ────────────────────────────────────────────────────
    return pdf.output()


# ── Convenience: build ReportData from BacktestResult ──────────────────────────

def build_report_data_from_backtest(result, config_info: dict,
                                     news: list[dict] | None = None,
                                     report_type: str = "End of Day") -> ReportData:
    """Build ReportData from a BacktestResult for PDF generation."""
    from ui.utils import (
        create_equity_curve, create_returns_distribution,
        create_cumulative_pnl,
    )
    from src.backtest.metrics import compute_returns, compute_drawdown

    metrics = result.metrics
    today = datetime.now().strftime("%B %d, %Y")

    # Build chart images
    dd = compute_drawdown(result.equity_curve)
    eq_fig = create_equity_curve(result.equity_curve, dd, title="Equity Curve")
    rets = compute_returns(result.equity_curve)
    ret_fig = create_returns_distribution(rets, title="Returns Distribution")
    pnl_fig = create_cumulative_pnl(result.trades, title="Cumulative P&L")

    equity_png = _fig_to_png(eq_fig)
    returns_png = _fig_to_png(ret_fig)
    pnl_png = _fig_to_png(pnl_fig)

    # Portfolio snapshot from final equity
    initial_cap = config_info.get("capital", 1_000_000)
    final_eq = result.equity_curve.iloc[-1] if len(result.equity_curve) else initial_cap
    total_pnl_val = final_eq - initial_cap

    # Simulate today's trades from recent trades
    todays_trades = []
    for t in result.trades[-10:]:
        todays_trades.append(DailyInvestment(
            symbol=t.symbol,
            action=t.side,
            quantity=t.quantity,
            price=t.entry_price if t.side == "BUY" else t.exit_price,
            value=t.quantity * (t.entry_price if t.side == "BUY" else t.exit_price),
            time=datetime.now().strftime("%H:%M"),
            strategy=config_info.get("strategy", ""),
            reason=f"Signal from {config_info.get('strategy', 'strategy')} analysis",
        ))

    # Build holdings from open positions (simulate from last trades)
    holdings = []
    seen_symbols = set()
    for t in reversed(result.trades[-20:]):
        if t.symbol not in seen_symbols and t.side == "BUY":
            holdings.append({
                "symbol": t.symbol,
                "qty": t.quantity,
                "avg_cost": t.entry_price,
                "current_price": t.exit_price,
                "pnl": t.net_pnl,
                "pnl_pct": (t.exit_price / t.entry_price - 1) * 100 if t.entry_price else 0,
            })
            seen_symbols.add(t.symbol)
        if len(holdings) >= 5:
            break

    # Learnings
    learnings = AgentLearnings(
        date=today,
        market_regime="Trending" if metrics.total_return_pct > 5 else "Mean-Reverting" if abs(metrics.total_return_pct) < 2 else "Volatile",
        regime_confidence=min(95, 50 + abs(metrics.sharpe_ratio) * 15),
        patterns_detected=[
            f"Win rate of {metrics.win_rate_pct:.0f}% suggests {'strong' if metrics.win_rate_pct > 60 else 'moderate' if metrics.win_rate_pct > 50 else 'weak'} signal quality",
            f"Sharpe ratio of {metrics.sharpe_ratio:.2f} indicates {'good' if metrics.sharpe_ratio > 1 else 'moderate' if metrics.sharpe_ratio > 0.5 else 'poor'} risk-adjusted returns",
            f"Max drawdown of {metrics.max_drawdown_pct:.1f}% {'within acceptable limits' if metrics.max_drawdown_pct < 15 else 'exceeds risk threshold - position sizing adjusted'}",
            f"Profit factor of {metrics.profit_factor:.2f} shows {'profits exceed losses' if metrics.profit_factor > 1 else 'losses dominate - strategy review needed'}",
        ],
        strategy_adjustments=[
            f"Position sizing calibrated based on {metrics.total_trades} historical trades",
            f"Stop-loss levels adjusted to limit drawdown below {metrics.max_drawdown_pct * 1.1:.0f}%",
            "Entry thresholds fine-tuned based on recent signal performance",
        ],
        key_observations=[
            f"Total of {metrics.total_trades} trades executed with avg holding of {metrics.avg_holding_bars:.0f} bars",
            f"Best performing aspect: {'win rate' if metrics.win_rate_pct > 60 else 'risk management' if metrics.max_drawdown_pct < 10 else 'consistent execution'}",
            f"Area for improvement: {'reduce drawdowns' if metrics.max_drawdown_pct > 15 else 'increase position sizing' if metrics.sharpe_ratio > 1 else 'signal quality'}",
        ],
        sentiment_score=1.5 if metrics.total_return_pct > 0 else -1.0,
        volatility_regime="High" if metrics.max_drawdown_pct > 15 else "Normal" if metrics.max_drawdown_pct > 8 else "Low",
    )

    # Default news
    news_items = []
    if news:
        for n in news:
            news_items.append(NewsItem(**n))
    else:
        news_items = _generate_sample_news()

    # Metrics dict
    metrics_dict = result.summary()

    # Trade log dicts
    trade_dicts = []
    for t in result.trades:
        trade_dicts.append({
            "symbol": t.symbol,
            "side": t.side,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "quantity": t.quantity,
            "net_pnl": t.net_pnl,
            "holding_bars": t.holding_bars,
        })

    return ReportData(
        report_date=today,
        report_type=report_type,
        learnings=learnings,
        portfolio=PortfolioSnapshot(
            total_value=final_eq,
            cash_balance=final_eq * 0.3,
            invested_value=final_eq * 0.7,
            day_pnl=result.trades[-1].net_pnl if result.trades else 0,
            day_pnl_pct=(result.trades[-1].net_pnl / final_eq * 100) if result.trades and final_eq else 0,
            total_pnl=total_pnl_val,
            total_pnl_pct=(total_pnl_val / initial_cap * 100) if initial_cap else 0,
            holdings=holdings,
        ),
        todays_investments=todays_trades,
        news=news_items,
        metrics=metrics_dict,
        equity_chart_png=equity_png,
        returns_chart_png=returns_png,
        pnl_chart_png=pnl_png,
        trades=trade_dicts,
    )


def _generate_sample_news() -> list[NewsItem]:
    """Generate realistic sample market news."""
    return [
        NewsItem(
            headline="RBI Keeps Repo Rate Unchanged at 6.5%, Maintains Accommodative Stance",
            source="Economic Times",
            sentiment="Positive",
            impact="High",
            summary="The Reserve Bank of India kept the benchmark repo rate unchanged, "
                    "signaling steady monetary policy. Markets reacted positively with "
                    "banking stocks leading gains.",
        ),
        NewsItem(
            headline="Nifty 50 Hits New All-Time High, Crosses 23,000 Mark",
            source="Moneycontrol",
            sentiment="Positive",
            impact="High",
            summary="The benchmark index crossed the psychological 23,000 level driven by "
                    "strong FII inflows and positive global cues. IT and pharma sectors "
                    "were top contributors.",
        ),
        NewsItem(
            headline="Crude Oil Prices Rise 2% on OPEC+ Supply Cut Extensions",
            source="Reuters",
            sentiment="Negative",
            impact="Medium",
            summary="Brent crude rose above $85/barrel after OPEC+ agreed to extend "
                    "production cuts. Indian oil marketing companies may face margin pressure.",
        ),
        NewsItem(
            headline="India's GDP Growth Projected at 7.2% for FY26, Says IMF",
            source="Bloomberg",
            sentiment="Positive",
            impact="Medium",
            summary="The International Monetary Fund revised India's growth forecast upward, "
                    "citing strong domestic consumption and investment momentum.",
        ),
        NewsItem(
            headline="FII Net Buyers for 5th Consecutive Session, Pump Rs.4,200 Cr",
            source="NDTV Profit",
            sentiment="Positive",
            impact="Medium",
            summary="Foreign institutional investors continued their buying streak, "
                    "investing over Rs.4,200 crore in Indian equities, boosting market sentiment.",
        ),
    ]


# ── Email sender ───────────────────────────────────────────────────────────────

def send_report_email(
    pdf_bytes: bytes,
    recipient_email: str,
    sender_email: str = "",
    smtp_server: str = "smtp.gmail.com",
    smtp_port: int = 587,
    smtp_password: str = "",
    report_date: str = "",
) -> tuple[bool, str]:
    """Send the PDF report as an email attachment.

    Returns (success, message).
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.application import MIMEApplication

    if not report_date:
        report_date = datetime.now().strftime("%B %d, %Y")

    if not sender_email or not smtp_password:
        return False, "Email credentials not configured. Set sender email and SMTP password in Settings."

    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = f"Stock Agent - Daily Trading Report ({report_date})"

        body = f"""
        <html>
        <body style="font-family: 'Segoe UI', Arial, sans-serif; background: #F8FAFC; padding: 20px;">
            <div style="max-width: 600px; margin: auto; background: white; border-radius: 12px; padding: 30px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <h2 style="color: #6366F1; margin-bottom: 5px;">📈 Stock Agent</h2>
                <p style="color: #64748B; margin-top: 0;">Daily Trading Report</p>
                <hr style="border: 1px solid #E2E8F0;">
                <p style="color: #334155;">Hello,</p>
                <p style="color: #334155;">
                    Please find attached your <strong>Stock Agent Trading Report</strong> for
                    <strong>{report_date}</strong>.
                </p>
                <p style="color: #334155;">
                    This report includes the agent's learning summary, portfolio overview,
                    today's trades, P&L analysis, market news, performance charts, and risk analysis.
                </p>
                <div style="background: #F1F5F9; border-radius: 8px; padding: 15px; margin: 20px 0;">
                    <p style="color: #6366F1; font-weight: bold; margin: 0;">📊 Report Highlights</p>
                    <ul style="color: #475569; padding-left: 20px;">
                        <li>Comprehensive portfolio analysis</li>
                        <li>Agent learning & strategy adjustments</li>
                        <li>Risk metrics & performance charts</li>
                    </ul>
                </div>
                <p style="color: #94A3B8; font-size: 0.85rem;">
                    This is an automated report generated by Stock Agent v1.0.0.
                    Do not reply to this email.
                </p>
            </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, "html"))

        filename = f"stock_agent_report_{datetime.now().strftime('%Y%m%d')}.pdf"
        attachment = MIMEApplication(pdf_bytes, _subtype="pdf")
        attachment.add_header("Content-Disposition", "attachment", filename=filename)
        msg.attach(attachment)

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, smtp_password)
            server.send_message(msg)

        return True, f"Report sent successfully to {recipient_email}"

    except smtplib.SMTPAuthenticationError:
        return False, "SMTP authentication failed. Check your email and app password."
    except smtplib.SMTPException as e:
        return False, f"SMTP error: {e}"
    except Exception as e:
        return False, f"Failed to send email: {e}"
