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

st.warning(
    "**The Analysis tab's dynamic KPIs came from a solver that is wrong.** "
    "It modelled each wishbone as a single strut to the midpoint between the two "
    "chassis pivots, so the ball joint rode a sphere instead of a circle about "
    "the pivot axis, leaving 3 of 9 degrees of freedom closed by a numerical "
    "regularisation rather than by the linkage.\n\n"
    "**Now fixed by delegation.** The swept dynamic KPIs — camber gain, "
    "roll-centre migration and height, roll cambers — are computed by the "
    "validated `vdcore` 3D solver (`vdcore.analysis.axle`, covered by the test "
    "suite). Open the new **🔬 vdcore (validated)** tab to run the same loaded "
    "geometry through it and compare side by side.\n\n"
    "**Still flagged.** Anti-dive / anti-squat, Ackermann, scrub and mechanical "
    "trail cannot be recomputed from loaded hardpoints alone — they need a "
    "synthesised corner. Do not quote those from the Analysis tab; use "
    "`sla_geometry.py` / `steering_geometry.py` or section 3b of "
    "`scripts/geometry_summary.py`.\n\n"
    "Static values — KPI, caster, scrub, roll-centre height — were always "
    "correct.",
    icon="🔧",
)

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
