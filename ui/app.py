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

from ui.utils import COLORS, apply_theme, get_custom_css

# ── Apply theme ────────────────────────────────────────────────────────────────
# Read user preference; default "system" detects OS dark/light via JS
_theme_pref = st.session_state.get("app_theme", "system")

# For "system" mode, detect OS preference once per session via JavaScript
if _theme_pref == "system":
    if "system_theme_resolved" not in st.session_state:
        # Inject a tiny JS component that detects OS preference
        import streamlit.components.v1 as components
        components.html(
            """
            <script>
                const isDark = window.matchMedia &&
                               window.matchMedia('(prefers-color-scheme: dark)').matches;
                const theme = isDark ? 'dark' : 'light';
                // Store in sessionStorage so it persists across reruns
                window.sessionStorage.setItem('stock_agent_os_theme', theme);
                // Communicate back via query string (triggers rerun)
                const url = new URL(window.parent.location.href);
                if (url.searchParams.get('_os_theme') !== theme) {
                    url.searchParams.set('_os_theme', theme);
                    window.parent.location.replace(url.toString());
                }
            </script>
            """,
            height=0,
        )
        # Read the detected value from query params (available after redirect)
        _detected = st.query_params.get("_os_theme", "dark")
        st.session_state["system_theme_resolved"] = _detected
        _effective_theme = _detected
    else:
        _effective_theme = st.session_state["system_theme_resolved"]
else:
    _effective_theme = _theme_pref

apply_theme(_effective_theme)

# Inject theme-aware CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="
                background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 1.8rem;
                font-weight: 800;
                margin: 0;
            ">📈 Stock Agent</h1>
            <p style="color: {COLORS['text_muted']}; font-size: 0.8rem; margin-top: 4px;">
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
            "icon": {"color": COLORS["primary_light"], "font-size": "18px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "2px 0",
                "padding": "10px 16px",
                "border-radius": "8px",
                "color": COLORS["text_secondary"],
                "--hover-color": COLORS["nav_hover"],
            },
            "nav-link-selected": {
                "background": f"linear-gradient(135deg, {COLORS['primary']}, {COLORS['primary_light']})",
                "color": "#FFFFFF",
                "font-weight": "600",
            },
        },
    )

    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align: center; padding: 10px;">
            <p style="color: {COLORS['text_dim']}; font-size: 0.7rem;">
                v1.0.0 · Local-First · NSE/BSE<br/>
                Built with Streamlit
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Page routing ───────────────────────────────────────────────────────────────
if selected == "Dashboard":
    from ui.views.dashboard import render
    render()
elif selected == "Backtest":
    from ui.views.backtest import render
    render()
elif selected == "Technical Analysis":
    from ui.views.technical_analysis import render
    render()
elif selected == "Trade Journal":
    from ui.views.trade_journal import render
    render()
elif selected == "Risk Management":
    from ui.views.risk_management import render
    render()
elif selected == "Strategy Lab":
    from ui.views.strategy_lab import render
    render()
elif selected == "Reports":
    from ui.views.reports import render
    render()
elif selected == "Settings":
    from ui.views.settings import render
    render()
