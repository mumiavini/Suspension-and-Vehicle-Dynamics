"""
ui/theme.py
===========
Temas (presets selecionáveis na sidebar), CSS de polimento e header do app.

O Streamlit não tem API pública para trocar tema em runtime; o padrão da
comunidade é `st._config.set_option("theme.*", ...)` seguido de `st.rerun()`.
Atenção: opções de config são globais ao PROCESSO (não à sessão) — ok para
uso local/single-user; em deploy multi-usuário a troca afetaria todos.
"""

from __future__ import annotations

import streamlit as st

THEMES: dict[str, dict[str, str]] = {
    "🌙 Midnight (padrão)": {
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
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F3F4F6",
        "textColor": "#111827",
        "sidebar.backgroundColor": "#15151E",
        "sidebar.secondaryBackgroundColor": "#262633",
        "sidebar.textColor": "#FAFAFA",
    },
    "🔴 Racing Escuro": {
        "base": "dark",
        "primaryColor": "#E10600",
        "backgroundColor": "#0F0F0F",
        "secondaryBackgroundColor": "#1C1C1C",
        "textColor": "#FAFAFA",
        "sidebar.backgroundColor": "#161616",
        "sidebar.secondaryBackgroundColor": "#242424",
        "sidebar.textColor": "#FAFAFA",
    },
    "🌊 Azul Petróleo": {
        "base": "dark",
        "primaryColor": "#38BDF8",
        "backgroundColor": "#0B1220",
        "secondaryBackgroundColor": "#142033",
        "textColor": "#E2E8F0",
        "sidebar.backgroundColor": "#0E1626",
        "sidebar.secondaryBackgroundColor": "#1A2940",
        "sidebar.textColor": "#E2E8F0",
    },
    "☀️ Claro Clássico": {
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

# Deve espelhar o tema definido em .streamlit/config.toml (estado de boot)
_DEFAULT_THEME = "🌙 Midnight (padrão)"


def _apply_theme(name: str) -> None:
    """Aplica todas as chaves do preset via config (vale a partir do próximo rerun)."""
    for option, value in THEMES[name].items():
        st._config.set_option(f"theme.{option}", value)


def init_theme() -> None:
    """Inicializa o estado de tema na sessão e aplica o preset escolhido."""
    st.session_state.setdefault("ui_theme", _DEFAULT_THEME)
    st.session_state.setdefault("_theme_applied", _DEFAULT_THEME)
    _apply_theme(st.session_state["ui_theme"])


def inject_css() -> None:
    """CSS de polimento: paddings mais compactos e abas mais legíveis."""
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
    """Título do app + badges com o status da geometria carregada."""
    header_left, header_right = st.columns([3, 2], vertical_alignment="bottom")
    with header_left:
        st.title("🏎️ FSAE Suspension Geometry Engine")
        st.caption("Análise · Síntese · Inputs visuais de geometria de suspensão")
    with header_right:
        _df_hdr = st.session_state.get("hardpoints_df")
        if _df_hdr is not None:
            _src_hdr = st.session_state.get("hardpoints_source", "?")
            _corners_hdr = sorted(_df_hdr["corner"].unique().to_list())
            st.markdown(
                f":green-badge[✅ {_src_hdr}] "
                f":blue-badge[📍 {_df_hdr.height} pontos] "
                f":gray-badge[{' · '.join(_corners_hdr)}]"
            )
        else:
            st.markdown(":orange-badge[⚠️ Nenhuma geometria carregada]")
