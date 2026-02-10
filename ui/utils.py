"""Shared UI utilities, synthetic data generation, and styling for the Stock Agent dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.config import AppConfig, RiskConfig, SlippageConfig, CostsConfig
from src.core.utils import set_seed
from src.features.feature_sets import compute_base_features
from src.features.indicators import (
    atr, rsi, zscore, vwap, bollinger_bands, ema, macd,
    realized_volatility, returns, log_returns, rate_of_change, on_balance_volume,
    rolling_mean, rolling_std,
)
from src.backtest.engine import BacktestEngine, BacktestResult, Trade, Portfolio
from src.backtest.metrics import (
    PerformanceMetrics, compute_all_metrics, compute_drawdown, compute_returns,
)
from src.backtest.costs import CostModel
from src.strategies.mean_reversion import MeanReversion
from src.strategies.orb_momentum import ORBMomentum
from src.strategies.pairs_trading import PairsTrading
from src.risk.sizing import atr_position_size, fixed_fraction_size, kelly_fraction, variance_position_size
from src.risk.limits import RiskLimiter

# ── Color palette ──────────────────────────────────────────────────────────────
DARK_COLORS = {
    "primary": "#6366F1",        # Indigo
    "primary_light": "#818CF8",
    "secondary": "#EC4899",      # Pink
    "success": "#10B981",        # Emerald
    "danger": "#EF4444",         # Red
    "warning": "#F59E0B",        # Amber
    "info": "#3B82F6",           # Blue
    "bg_dark": "#0F172A",        # Slate-900
    "bg_card": "#1E293B",        # Slate-800
    "bg_card_alt": "#253048",    # Gradient destination
    "text": "#F8FAFC",           # Slate-50
    "text_muted": "#94A3B8",     # Slate-400
    "text_secondary": "#CBD5E1", # Slate-300
    "text_dim": "#64748B",       # Slate-500
    "grid": "#334155",           # Slate-700
    "profit": "#10B981",
    "loss": "#EF4444",
    "neutral": "#6B7280",
    "plotly_template": "plotly_dark",
    "nav_hover": "#1E293B",
    "sidebar_bg": "#1E293B",
    "sidebar_border": "#334155",
}

LIGHT_COLORS = {
    "primary": "#4F46E5",        # Indigo-600
    "primary_light": "#6366F1",
    "secondary": "#DB2777",      # Pink-600
    "success": "#059669",        # Emerald-600
    "danger": "#DC2626",         # Red-600
    "warning": "#D97706",        # Amber-600
    "info": "#2563EB",           # Blue-600
    "bg_dark": "#F8FAFC",        # Slate-50
    "bg_card": "#FFFFFF",        # White
    "bg_card_alt": "#F1F5F9",    # Slate-100
    "text": "#0F172A",           # Slate-900
    "text_muted": "#64748B",     # Slate-500
    "text_secondary": "#475569", # Slate-600
    "text_dim": "#94A3B8",       # Slate-400
    "grid": "#E2E8F0",           # Slate-200
    "profit": "#059669",
    "loss": "#DC2626",
    "neutral": "#6B7280",
    "plotly_template": "plotly_white",
    "nav_hover": "#F1F5F9",
    "sidebar_bg": "#FFFFFF",
    "sidebar_border": "#E2E8F0",
}

# Mutable dict – updated in-place by apply_theme()
COLORS = dict(DARK_COLORS)


def apply_theme(theme: str = "dark") -> None:
    """Update COLORS dict in-place to match the requested theme.

    Since every module that does ``from ui.utils import COLORS`` holds a
    reference to the *same* dict object, mutating it here propagates
    automatically to all pages.
    """
    source = LIGHT_COLORS if theme == "light" else DARK_COLORS
    COLORS.update(source)

STRATEGY_MAP = {
    "Mean Reversion": {
        "class": MeanReversion,
        "default_params": {
            "zscore_window": 20,
            "zscore_entry": 2.0,
            "zscore_exit": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "max_positions": 5,
        },
    },
    "ORB Momentum": {
        "class": ORBMomentum,
        "default_params": {
            "orb_window_minutes": 15,
            "atr_period": 14,
            "atr_multiplier": 1.5,
            "time_stop_minutes": 180,
            "max_positions": 5,
        },
    },
    "Pairs Trading": {
        "class": PairsTrading,
        "default_params": {
            "pairs": [["RELIANCE", "TCS"]],
            "zscore_entry": 2.0,
            "zscore_exit": 0.5,
            "lookback": 60,
            "max_positions": 3,
        },
    },
}


def generate_synthetic_data(
    symbol: str = "RELIANCE",
    n_bars: int = 500,
    start_price: float = 2500.0,
    seed: int = 42,
    start_date: str = "2023-01-01",
    trend: float = 0.0003,
    volatility: float = 0.015,
) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start_date, periods=n_bars, freq="D")

    price = start_price
    closes = []
    for _ in range(n_bars):
        price *= np.exp(rng.normal(trend, volatility))
        closes.append(price)
    closes = np.array(closes)

    df = pd.DataFrame({
        "date": dates,
        "symbol": symbol,
        "open": closes * (1 + rng.uniform(-0.005, 0.005, n_bars)),
        "high": closes * (1 + rng.uniform(0.005, 0.02, n_bars)),
        "low": closes * (1 - rng.uniform(0.005, 0.02, n_bars)),
        "close": closes,
        "volume": rng.integers(500_000, 5_000_000, n_bars),
    })
    return df


def load_sample_data(symbol: str = "RELIANCE") -> pd.DataFrame:
    """Load sample CSV data or generate synthetic data."""
    from pathlib import Path
    sample_dir = Path("data/sample")

    file_map = {
        "RELIANCE": sample_dir / "eod_reliance.csv",
        "TCS": sample_dir / "eod_tcs.csv",
    }

    csv_path = file_map.get(symbol)
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["date"])
        return df
    return generate_synthetic_data(symbol=symbol)


def generate_multi_symbol_data(
    symbols: list[str] | None = None,
    n_bars: int = 500,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate synthetic data for multiple symbols."""
    if symbols is None:
        symbols = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK"]

    data = {}
    for i, sym in enumerate(symbols):
        df = generate_synthetic_data(
            symbol=sym,
            n_bars=n_bars,
            start_price=1500 + i * 500,
            seed=seed + i,
            trend=0.0002 + i * 0.0001,
            volatility=0.012 + i * 0.002,
        )
        data[sym] = df
    return data


def run_backtest_from_ui(
    df: pd.DataFrame,
    strategy_name: str,
    strategy_params: dict,
    initial_capital: float = 1_000_000,
    slippage_mode: str = "fixed",
    slippage_bps: float = 3.0,
    seed: int = 42,
) -> BacktestResult:
    """Run a backtest with UI-provided parameters."""
    set_seed(seed)
    df = compute_base_features(df.copy())

    cfg = AppConfig(
        strategy=strategy_name.lower().replace(" ", "_"),
        strategy_params=strategy_params,
        slippage=SlippageConfig(mode=slippage_mode, bps_mean=slippage_bps, bps_std=0.0),
        risk=RiskConfig(initial_capital=initial_capital),
    )

    strategy_info = STRATEGY_MAP[strategy_name]
    strategy = strategy_info["class"](config=strategy_params)

    engine = BacktestEngine(config=cfg, strategy=strategy)
    return engine.run(df)


# ── Plotly chart builders ──────────────────────────────────────────────────────

def create_candlestick_chart(
    df: pd.DataFrame,
    title: str = "Price Chart",
    show_volume: bool = True,
    height: int = 600,
) -> go.Figure:
    """Create an interactive candlestick chart with volume."""
    rows = 2 if show_volume else 1
    row_heights = [0.7, 0.3] if show_volume else [1.0]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    date_col = "date" if "date" in df.columns else "datetime"

    fig.add_trace(go.Candlestick(
        x=df[date_col],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="OHLC",
        increasing_line_color=COLORS["success"],
        decreasing_line_color=COLORS["danger"],
    ), row=1, col=1)

    if show_volume and "volume" in df.columns:
        colors = [COLORS["success"] if c >= o else COLORS["danger"]
                  for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Bar(
            x=df[date_col],
            y=df["volume"],
            marker_color=colors,
            opacity=0.5,
            name="Volume",
        ), row=2, col=1)

    fig.update_layout(
        title=title,
        template=COLORS["plotly_template"],
        height=height,
        xaxis_rangeslider_visible=False,
        showlegend=False,
        paper_bgcolor=COLORS["bg_dark"],
        plot_bgcolor=COLORS["bg_dark"],
        font=dict(color=COLORS["text"]),
    )
    return fig


def create_equity_curve(
    equity: pd.Series,
    drawdown: pd.Series | None = None,
    title: str = "Equity Curve",
    height: int = 500,
) -> go.Figure:
    """Create equity curve with optional drawdown subplot."""
    rows = 2 if drawdown is not None else 1
    row_heights = [0.65, 0.35] if drawdown is not None else [1.0]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=("Equity Curve", "Drawdown") if drawdown is not None else ("Equity Curve",),
    )

    fig.add_trace(go.Scatter(
        x=equity.index if not isinstance(equity.index, pd.RangeIndex) else list(range(len(equity))),
        y=equity.values,
        mode="lines",
        name="Equity",
        line=dict(color=COLORS["primary"], width=2),
        fill="tozeroy",
        fillcolor="rgba(99, 102, 241, 0.1)",
    ), row=1, col=1)

    if drawdown is not None:
        fig.add_trace(go.Scatter(
            x=drawdown.index if not isinstance(drawdown.index, pd.RangeIndex) else list(range(len(drawdown))),
            y=drawdown.values * 100,
            mode="lines",
            name="Drawdown %",
            line=dict(color=COLORS["danger"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.15)",
        ), row=2, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

    fig.update_layout(
        title=title,
        template=COLORS["plotly_template"],
        height=height,
        paper_bgcolor=COLORS["bg_dark"],
        plot_bgcolor=COLORS["bg_dark"],
        font=dict(color=COLORS["text"]),
        showlegend=False,
    )
    fig.update_yaxes(title_text="Portfolio Value (₹)", row=1, col=1)
    return fig


def create_returns_distribution(
    returns_series: pd.Series,
    title: str = "Returns Distribution",
    height: int = 400,
) -> go.Figure:
    """Create a histogram of returns."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns_series.dropna() * 100,
        nbinsx=50,
        marker_color=COLORS["primary"],
        opacity=0.75,
        name="Returns %",
    ))

    mean_ret = returns_series.dropna().mean() * 100
    fig.add_vline(x=mean_ret, line_dash="dash", line_color=COLORS["warning"],
                  annotation_text=f"Mean: {mean_ret:.3f}%")
    fig.add_vline(x=0, line_dash="solid", line_color=COLORS["text_muted"])

    fig.update_layout(
        title=title,
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        template=COLORS["plotly_template"],
        height=height,
        paper_bgcolor=COLORS["bg_dark"],
        plot_bgcolor=COLORS["bg_dark"],
        font=dict(color=COLORS["text"]),
    )
    return fig


def create_monthly_returns_heatmap(
    equity: pd.Series,
    title: str = "Monthly Returns Heatmap",
    height: int = 400,
) -> go.Figure:
    """Create a monthly returns heatmap."""
    rets = compute_returns(equity)
    if isinstance(equity.index, pd.RangeIndex):
        dates = pd.bdate_range("2023-01-01", periods=len(equity), freq="D")
        rets.index = dates[:len(rets)]

    monthly = rets.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
    if len(monthly) == 0:
        fig = go.Figure()
        fig.update_layout(title=title, template=COLORS["plotly_template"], height=height,
                          paper_bgcolor=COLORS["bg_dark"], plot_bgcolor=COLORS["bg_dark"])
        return fig

    pivot = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot_table = pivot.pivot_table(values="return", index="year", columns="month", aggfunc="sum")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot_table.columns = [month_names[m - 1] for m in pivot_table.columns]

    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns.tolist(),
        y=[str(y) for y in pivot_table.index],
        colorscale=[[0, COLORS["danger"]], [0.5, COLORS["bg_card"]], [1, COLORS["success"]]],
        zmid=0,
        text=np.round(pivot_table.values, 2),
        texttemplate="%{text:.1f}%",
        textfont={"size": 11},
    ))

    fig.update_layout(
        title=title,
        template=COLORS["plotly_template"],
        height=height,
        paper_bgcolor=COLORS["bg_dark"],
        plot_bgcolor=COLORS["bg_dark"],
        font=dict(color=COLORS["text"]),
    )
    return fig


def create_trade_scatter(
    trades: list[Trade],
    title: str = "Trade P&L Scatter",
    height: int = 400,
) -> go.Figure:
    """Scatter plot of trade P&L."""
    if not trades:
        fig = go.Figure()
        fig.update_layout(title=title, template=COLORS["plotly_template"], height=height,
                          paper_bgcolor=COLORS["bg_dark"], plot_bgcolor=COLORS["bg_dark"])
        return fig

    pnls = [t.net_pnl for t in trades]
    colors = [COLORS["success"] if p > 0 else COLORS["danger"] for p in pnls]
    sizes = [max(6, min(20, abs(p) / max(abs(min(pnls)), abs(max(pnls))) * 20)) for p in pnls]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(trades) + 1)),
        y=pnls,
        mode="markers",
        marker=dict(color=colors, size=sizes, line=dict(width=1, color=COLORS["text_muted"])),
        text=[f"#{i+1}: ₹{p:,.0f}" for i, p in enumerate(pnls)],
        hoverinfo="text",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["text_muted"])

    fig.update_layout(
        title=title,
        xaxis_title="Trade #",
        yaxis_title="Net P&L (₹)",
        template=COLORS["plotly_template"],
        height=height,
        paper_bgcolor=COLORS["bg_dark"],
        plot_bgcolor=COLORS["bg_dark"],
        font=dict(color=COLORS["text"]),
    )
    return fig


def create_cumulative_pnl(
    trades: list[Trade],
    title: str = "Cumulative P&L",
    height: int = 400,
) -> go.Figure:
    """Cumulative P&L line chart."""
    if not trades:
        fig = go.Figure()
        fig.update_layout(title=title, template=COLORS["plotly_template"], height=height)
        return fig

    cum_pnl = np.cumsum([t.net_pnl for t in trades])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(trades) + 1)),
        y=cum_pnl,
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=5),
        fill="tozeroy",
        fillcolor="rgba(99, 102, 241, 0.1)",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["text_muted"])

    fig.update_layout(
        title=title,
        xaxis_title="Trade #",
        yaxis_title="Cumulative P&L (₹)",
        template=COLORS["plotly_template"],
        height=height,
        paper_bgcolor=COLORS["bg_dark"],
        plot_bgcolor=COLORS["bg_dark"],
        font=dict(color=COLORS["text"]),
    )
    return fig


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    """Convert trades to a styled DataFrame."""
    if not trades:
        return pd.DataFrame()
    records = []
    for i, t in enumerate(trades, 1):
        records.append({
            "#": i,
            "Symbol": t.symbol,
            "Side": t.side,
            "Entry Price": f"₹{t.entry_price:,.2f}",
            "Exit Price": f"₹{t.exit_price:,.2f}",
            "Qty": t.quantity,
            "Gross P&L": f"₹{t.pnl:,.2f}",
            "Costs": f"₹{t.costs:,.2f}",
            "Net P&L": f"₹{t.net_pnl:,.2f}",
            "Holding Bars": t.holding_bars,
        })
    return pd.DataFrame(records)


# ── Custom CSS ─────────────────────────────────────────────────────────────────

def get_custom_css() -> str:
    """Generate theme-aware CSS using current COLORS values."""
    return f"""
<style>
    /* Main background */
    .stApp {{
        background-color: {COLORS["bg_dark"]};
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {COLORS["sidebar_bg"]};
        border-right: 1px solid {COLORS["sidebar_border"]};
    }}

    /* Metric cards */
    [data-testid="stMetric"] {{
        background-color: {COLORS["bg_card"]};
        border: 1px solid {COLORS["grid"]};
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }}
    [data-testid="stMetricLabel"] {{
        color: {COLORS["text_muted"]} !important;
        font-size: 0.85rem !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {COLORS["text"]} !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricDelta"] > div {{
        font-size: 0.9rem !important;
    }}

    /* Headers */
    h1, h2, h3 {{
        color: {COLORS["text"]} !important;
    }}

    /* Paragraph text */
    p, span, label, .stMarkdown {{
        color: {COLORS["text_secondary"]};
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {COLORS["bg_card"]};
        border-radius: 8px;
        color: {COLORS["text_muted"]};
        border: 1px solid {COLORS["grid"]};
        padding: 8px 16px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS["primary"]} !important;
        color: #FFFFFF !important;
        border-color: {COLORS["primary"]} !important;
    }}

    /* Cards / containers */
    [data-testid="stExpander"] {{
        background-color: {COLORS["bg_card"]};
        border: 1px solid {COLORS["grid"]};
        border-radius: 12px;
    }}

    /* Dataframes */
    [data-testid="stDataFrame"] {{
        border-radius: 8px;
        overflow: hidden;
    }}

    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["primary_light"]});
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: 600;
        transition: all 0.2s;
    }}
    .stButton > button:hover {{
        background: linear-gradient(135deg, {COLORS["primary_light"]}, {COLORS["primary"]});
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }}

    /* Selectbox / inputs */
    [data-testid="stSelectbox"] label,
    [data-testid="stNumberInput"] label,
    [data-testid="stSlider"] label,
    [data-testid="stTextInput"] label,
    [data-testid="stTextArea"] label {{
        color: {COLORS["text_secondary"]} !important;
    }}

    /* Input fields */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"] {{
        background-color: {COLORS["bg_card"]} !important;
        color: {COLORS["text"]} !important;
        border-color: {COLORS["grid"]} !important;
    }}

    /* Slider */
    .stSlider [data-testid="stThumbValue"] {{
        color: {COLORS["text"]} !important;
    }}

    /* Checkbox */
    .stCheckbox label span {{
        color: {COLORS["text_secondary"]} !important;
    }}

    /* Toggle */
    [data-testid="stToggle"] label span {{
        color: {COLORS["text_secondary"]} !important;
    }}

    /* Divider */
    hr {{
        border-color: {COLORS["grid"]} !important;
    }}

    /* Success/Error/Info/Warning boxes */
    [data-testid="stAlert"] {{
        border-radius: 8px;
    }}

    /* Multiselect tags */
    [data-baseweb="tag"] {{
        background-color: {COLORS["primary"]} !important;
    }}

    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
</style>
"""


# Keep backward compatibility – static string alias pointing to dark theme CSS
CUSTOM_CSS = get_custom_css()


def metric_card_html(label: str, value: str, delta: str = "", delta_color: str = "normal") -> str:
    """Generate styled metric card HTML using theme-aware COLORS."""
    delta_style = ""
    if delta:
        color = COLORS["success"] if delta_color == "normal" else COLORS["danger"]
        delta_style = f'<div style="color: {color}; font-size: 0.85rem; margin-top: 4px;">{delta}</div>'

    return f"""
    <div style="
        background: linear-gradient(135deg, {COLORS['bg_card']}, {COLORS['bg_card_alt']});
        border: 1px solid {COLORS['grid']};
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    ">
        <div style="color: {COLORS['text_muted']}; font-size: 0.85rem; margin-bottom: 6px;">{label}</div>
        <div style="color: {COLORS['text']}; font-size: 1.8rem; font-weight: 700;">{value}</div>
        {delta_style}
    </div>
    """
