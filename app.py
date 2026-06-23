"""
app.py
======
Streamlit app — Interface gráfica do motor FSAE Suspension Geometry.

Este arquivo é só orquestração: config da página, tema, header, sidebar e as
abas. O código de cada aba vive em `ui/` (um módulo por aba, expondo `render()`).

ESTRUTURA EM ABAS:
    ✏️  Inputs       : Cria/edita hardpoints manualmente, com visualização
                       2D em vistas YZ (frontal), XZ (lateral), XY (superior).
    📊 Análise       : Carrega hardpoints, roda sweeps, mostra KPIs e gráficos.
    🌐 Vista 3D      : Visualização 3D interativa (veículo, corner, animação).
    🎯 Síntese       : Otimização global a partir de targets (engenharia reversa).
    🔄 Comparação    : Compara duas geometrias lado a lado.

COMO RODAR:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from ui import tab_analysis, tab_compare, tab_inputs, tab_synthesis, tab_view3d
from ui.sidebar import render_sidebar
from ui.theme import init_theme, inject_css, render_header

st.set_page_config(
    page_title="FSAE Suspension Geometry",
    layout="wide",
    page_icon="🏎️",
    initial_sidebar_state="expanded",
)

init_theme()
inject_css()
render_header()
render_sidebar()

t_inputs, t_analysis, t_3d, t_synthesis, t_compare = st.tabs([
    "✏️ Inputs", "📊 Análise", "🌐 Vista 3D", "🎯 Síntese / Otimização", "🔄 Comparação",
])

with t_inputs:
    tab_inputs.render()

with t_analysis:
    tab_analysis.render()

with t_3d:
    tab_view3d.render()

with t_synthesis:
    tab_synthesis.render()

with t_compare:
    tab_compare.render()
