"""
app.py
======
Streamlit app — Graphical interface for the FSAE Suspension Geometry engine.

This file is orchestration only: page config, theme, header, sidebar and the
tabs. The code for each tab lives in `ui/` (one module per tab, exposing `render()`).

TAB STRUCTURE:
    ✏️  Inputs       : Create/edit hardpoints manually, with 2D visualization
                       in YZ (front), XZ (side), XY (top) views.
    📊 Analysis      : Load hardpoints, run sweeps, show KPIs and charts.
    🌐 View 3D       : Interactive 3D visualization (vehicle, corner, animation).
    🎯 Synthesis     : Global optimization from targets (reverse engineering).
    🔄 Comparison    : Compare two geometries side by side.

HOW TO RUN:
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

st.error(
    "**Dynamic KPIs on this app are known to be wrong. Do not quote them.**\n\n"
    "The 3D solver models each wishbone as a single strut to the midpoint "
    "between the two chassis pivots, so the ball joint rides a sphere instead "
    "of a circle about the pivot axis. That leaves 3 of 9 degrees of freedom "
    "closed by a numerical regularisation rather than by the linkage. Confirmed "
    "wrong: anti-dive (reports +200%, actually 0%), anti-squat (+83.74%, "
    "actually 0%), Ackermann (+173%, actually ~70% — the formula is the "
    "reciprocal of its own docstring), roll-centre migration under roll "
    "(reports ~1 mm, actually 110 mm front), rear camber gain (27% low) and "
    "mechanical trail (sign inverted).\n\n"
    "Static values — KPI, caster, scrub, roll-centre height — are correct.\n\n"
    "For anything dynamic use `sla_geometry.py` and `steering_geometry.py`, "
    "or section 3b of `scripts/geometry_summary.py`, which solve in 3D via "
    "`vdcore.analysis.axle` and are covered by the test suite.",
    icon="🚨",
)

render_sidebar()

t_inputs, t_analysis, t_3d, t_synthesis, t_compare = st.tabs([
    "✏️ Inputs", "📊 Analysis", "🌐 View 3D", "🎯 Synthesis / Optimization", "🔄 Comparison",
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
