"""Stock Agent Dashboard - Main Application Entry Point."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of CWD
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
from streamlit_option_menu import option_menu

# Must be first Streamlit command
st.set_page_config(
    page_title="Stock Agent | Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.utils import CUSTOM_CSS

# Inject custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="
                background: linear-gradient(135deg, #6366F1, #EC4899);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 1.8rem;
                font-weight: 800;
                margin: 0;
            ">📈 Stock Agent</h1>
            <p style="color: #94A3B8; font-size: 0.8rem; margin-top: 4px;">
                Indian Market Trading Dashboard
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = option_menu(
        menu_title=None,
        options=[
            "Dashboard",
            "Backtest",
            "Technical Analysis",
            "Trade Journal",
            "Risk Management",
            "Strategy Lab",
            "Reports",
            "Settings",
        ],
        icons=[
            "speedometer2",
            "play-circle",
            "graph-up",
            "journal-text",
            "shield-check",
            "lightning",
            "file-earmark-bar-graph",
            "gear",
        ],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"color": "#818CF8", "font-size": "18px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "2px 0",
                "padding": "10px 16px",
                "border-radius": "8px",
                "color": "#CBD5E1",
                "--hover-color": "#1E293B",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, #6366F1, #4F46E5)",
                "color": "#F8FAFC",
                "font-weight": "600",
            },
        },
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 10px;">
            <p style="color: #64748B; font-size: 0.7rem;">
                v1.0.0 • Local-First • NSE/BSE<br/>
                Built with Streamlit
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Page routing ───────────────────────────────────────────────────────────────
if selected == "Dashboard":
    from ui.pages.dashboard import render
    render()
elif selected == "Backtest":
    from ui.pages.backtest import render
    render()
elif selected == "Technical Analysis":
    from ui.pages.technical_analysis import render
    render()
elif selected == "Trade Journal":
    from ui.pages.trade_journal import render
    render()
elif selected == "Risk Management":
    from ui.pages.risk_management import render
    render()
elif selected == "Strategy Lab":
    from ui.pages.strategy_lab import render
    render()
elif selected == "Reports":
    from ui.pages.reports import render
    render()
elif selected == "Settings":
    from ui.pages.settings import render
    render()
