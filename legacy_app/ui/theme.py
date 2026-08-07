"""
ui/theme.py
===========
Themes (presets selectable in the sidebar), polish CSS and the app header.

Streamlit has no public API to switch themes at runtime; the community pattern
is `st._config.set_option("theme.*", ...)` followed by `st.rerun()`.
Note: config options are global to the PROCESS (not the session) — fine for
local/single-user use; in a multi-user deployment the switch would affect everyone.
"""

from __future__ import annotations

import streamlit as st

THEMES: dict[str, dict[str, str]] = {
    "🌙 Midnight (default)": {
        "base": "dark",
        "primaryColor": "#FBBF24",
        "backgroundColor": "#0A0A0A",
        "secondaryBackgroundColor": "#181818",
        "textColor": "#FAFAFA",
        "sidebar.backgroundColor": "#181818",
        "sidebar.secondaryBackgroundColor": "#262626",
        "sidebar.textColor": "#FAFAFA",
    },
    "🏁 PUCPR Racing": {
        "base": "light",
        "primaryColor": "#E10600",
        "backgroundColor": "#F5F5F5",
        "secondaryBackgroundColor": "#F3F4F6",
        "textColor": "#111827",
        "sidebar.backgroundColor": "#15151E",
        "sidebar.secondaryBackgroundColor": "#262633",
        "sidebar.textColor": "#FAFAFA",
    },
    "🔴 Racing Dark": {
        "base": "dark",
        "primaryColor": "#E10600",
        "backgroundColor": "#0F0F0F",
        "secondaryBackgroundColor": "#1C1C1C",
        "textColor": "#FAFAFA",
        "sidebar.backgroundColor": "#161616",
        "sidebar.secondaryBackgroundColor": "#242424",
        "sidebar.textColor": "#FAFAFA",
    },
    "🌊 Petrol Blue": {
        "base": "dark",
        "primaryColor": "#38BDF8",
        "backgroundColor": "#0B1220",
        "secondaryBackgroundColor": "#142033",
        "textColor": "#E2E8F0",
        "sidebar.backgroundColor": "#0E1626",
        "sidebar.secondaryBackgroundColor": "#1A2940",
        "sidebar.textColor": "#E2E8F0",
    },
    "☀️ Classic Light": {
        "base": "light",
        "primaryColor": "#2563EB",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F1F5F9",
        "textColor": "#0F172A",
        "sidebar.backgroundColor": "#F8FAFC",
        "sidebar.secondaryBackgroundColor": "#E2E8F0",
        "sidebar.textColor": "#0F172A",
    },
}

# Must mirror the theme defined in .streamlit/config.toml (boot state)
_DEFAULT_THEME = "🌙 Midnight (default)"


def _apply_theme(name: str) -> None:
    """Apply all preset keys via config (takes effect from the next rerun)."""
    for option, value in THEMES[name].items():
        st._config.set_option(f"theme.{option}", value)


def init_theme() -> None:
    """Initialize the theme state in the session and apply the chosen preset."""
    st.session_state.setdefault("ui_theme", _DEFAULT_THEME)
    st.session_state.setdefault("_theme_applied", _DEFAULT_THEME)
    _apply_theme(st.session_state["ui_theme"])


def inject_css() -> None:
    """Polish CSS: more compact paddings and more readable tabs."""
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2.2rem; padding-bottom: 4rem; }
        h1 { font-size: 2.1rem !important; }
        button[data-baseweb="tab"] { font-size: 1.02rem; font-weight: 600; }
        [data-testid="stMetricValue"] { font-size: 1.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """App title + badges with the status of the loaded geometry."""
    header_left, header_right = st.columns([3, 2], vertical_alignment="bottom")
    with header_left:
        st.title("🏎️ FSAE Suspension Geometry Engine")
        st.caption("Analysis · Synthesis · Visual suspension-geometry inputs")
    with header_right:
        _df_hdr = st.session_state.get("hardpoints_df")
        if _df_hdr is not None:
            _src_hdr = st.session_state.get("hardpoints_source", "?")
            _corners_hdr = sorted(_df_hdr["corner"].unique().to_list())
            st.markdown(
                f":green-badge[✅ {_src_hdr}] "
                f":blue-badge[📍 {_df_hdr.height} points] "
                f":gray-badge[{' · '.join(_corners_hdr)}]"
            )
        else:
            st.markdown(":orange-badge[⚠️ No geometry loaded]")
