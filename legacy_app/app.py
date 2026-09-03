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

import sys
from pathlib import Path

# `streamlit run` puts THIS file's directory on sys.path, which finds `ui/`,
# `analysis/` and `geometry/` but not the repo root -- so `import vdcore` fails
# and the Analysis and vdcore tabs cannot load. Put the repo root on the path
# before importing anything that reaches for it.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import streamlit as st

from ui import (
    tab_compare,
    tab_inputs,
    tab_synthesis,
    tab_vdcore,
    tab_view3d,
)
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

# The old "Analysis" and "vdcore (validated)" tabs were merged. They showed the
# same KPIs from two solvers, and the Analysis side was wrong on several: static
# camber read 0 instead of -1.5, right-side scrub and mechanical trail had
# inverted signs, and every sweep chart ran on the strut-to-midpoint solver
# (camber gain came out with the wrong SIGN). One tab now, all of it on DWSolver.
t_inputs, t_analysis, t_3d, t_synthesis, t_compare = st.tabs([
    "✏️ Inputs", "📊 Analysis", "🌐 View 3D",
    "🎯 Synthesis / Optimization", "🔄 Comparison",
])

with t_inputs:
    tab_inputs.render()

with t_analysis:
    tab_vdcore.render()

with t_3d:
    tab_view3d.render()

with t_synthesis:
    tab_synthesis.render()

with t_compare:
    tab_compare.render()
