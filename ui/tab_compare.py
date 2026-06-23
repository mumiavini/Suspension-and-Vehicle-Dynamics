"""
ui/tab_compare.py
=================
🔄 Compare tab — two geometries side by side: corners from the file, the seed
of the last optimization, or the optimized result (via `last_optimization`).
"""

from __future__ import annotations

import polars as pl
import streamlit as st
import plotly.graph_objects as go

from analysis.io_hardpoints import VALID_CORNERS
from analysis.sweeps import camber_gain_per_mm, bump_steer_per_mm
from ui.shared import (
    load_hardpoints_from_state,
    render_empty_state,
    build_corner_safe,
    run_sweep_cached,
)


def render() -> None:
    st.header("Comparison between two geometries")

    df = load_hardpoints_from_state()
    has_optimization = "last_optimization" in st.session_state

    if df is None and not has_optimization:
        render_empty_state(
            "The comparison places **two geometries side by side**: corners from "
            "the file, the seed of the last optimization, or the optimized result.",
            key="empty_compare",
        )
        st.stop()

    def resolve_geometry(source, side):
        if source == "File corner":
            if df is None:
                st.warning(f"⚠️ Load a file for side {side}.")
                return None
            cid = st.selectbox(f"Corner {side}", VALID_CORNERS, key=f"cmp_{side}")
            return build_corner_safe(df, cid)
        elif source == "Last SEED":
            if not has_optimization:
                st.warning("⚠️ Run an optimization first.")
                return None
            lo = st.session_state["last_optimization"]
            return lo["seed_corner"], lo["seed_tie_rod"]
        else:
            lo = st.session_state["last_optimization"]
            return lo["opt_corner"], lo["opt_tie_rod"]

    col_src_a, col_src_b = st.columns(2)
    with col_src_a.container(border=True):
        st.markdown("**🅰️ Geometry A**")
        sa_opts = ["File corner", "Last SEED"]
        if has_optimization: sa_opts.append("Last OPTIMIZED")
        source_a = st.radio("A", sa_opts, key="src_a",
                            label_visibility="collapsed", horizontal=True)
        geom_a = resolve_geometry(source_a, "A")
    with col_src_b.container(border=True):
        st.markdown("**🅱️ Geometry B**")
        sb_opts = ["File corner", "Last SEED"]
        if has_optimization: sb_opts.append("Last OPTIMIZED")
        default_idx = 2 if has_optimization else 0
        source_b = st.radio("B", sb_opts, index=default_idx, key="src_b",
                            label_visibility="collapsed", horizontal=True)
        geom_b = resolve_geometry(source_b, "B")

    if geom_a is None or geom_b is None:
        st.stop()
    corner_a, tie_rod_a = geom_a
    corner_b, tie_rod_b = geom_b

    st.markdown("---")
    st.markdown("### Static KPIs")

    metrics = [
        ("Caster (°)",          corner_a.static_caster_deg(),         corner_b.static_caster_deg()),
        ("KPI (°)",             corner_a.static_kpi_deg(),            corner_b.static_kpi_deg()),
        ("Static camber (°)",   corner_a.static_camber_deg(),         corner_b.static_camber_deg()),
        ("Scrub (mm)",          corner_a.static_scrub_radius_mm(),    corner_b.static_scrub_radius_mm()),
        ("Trail (mm)",          corner_a.static_mechanical_trail_mm(),corner_b.static_mechanical_trail_mm()),
        ("Kingpin Offset (mm)", corner_a.static_kingpin_offset_mm(),  corner_b.static_kingpin_offset_mm()),
        ("Steer Arm (mm)",      corner_a.steer_arm_length_mm(tie_rod_a.outboard),
                                 corner_b.steer_arm_length_mm(tie_rod_b.outboard)),
        ("RC Height (mm)",      corner_a.roll_center_height_mm(),     corner_b.roll_center_height_mm()),
    ]
    static_cmp = pl.DataFrame([
        {"Parameter": n, "A": f"{a:+.3f}", "B": f"{b:+.3f}",
         "Δ (B−A)": f"{b-a:+.3f}"} for n, a, b in metrics
    ])
    st.dataframe(static_cmp, width="stretch", hide_index=True)

    st.markdown("### Heave Sweep — Overlay")
    hsc1, hsc2, hsc3 = st.columns(3)
    with hsc1: cmp_h_min  = st.number_input("Min", value=-25.0, key="cmp_hmin")
    with hsc2: cmp_h_max  = st.number_input("Max", value= 25.0, key="cmp_hmax")
    with hsc3: cmp_h_step = st.number_input("Step",value=  1.0, key="cmp_hstep")

    with st.spinner("Running sweeps..."):
        sweep_a = run_sweep_cached(corner_a, tie_rod_a, "Heave",
                                    (cmp_h_min, cmp_h_max, cmp_h_step))
        sweep_b = run_sweep_cached(corner_b, tie_rod_b, "Heave",
                                    (cmp_h_min, cmp_h_max, cmp_h_step))

    kc = st.columns(4)
    cg_a, cg_b = camber_gain_per_mm(sweep_a), camber_gain_per_mm(sweep_b)
    bs_a, bs_b = bump_steer_per_mm(sweep_a),  bump_steer_per_mm(sweep_b)
    kc[0].metric("Camber gain A (°/mm)", f"{cg_a:+.5f}",
                 delta=f"Δ {cg_b-cg_a:+.5f}", border=True)
    kc[1].metric("Camber gain B (°/mm)", f"{cg_b:+.5f}", border=True)
    kc[2].metric("Bump steer A (°/mm)", f"{bs_a:+.5f}",
                 delta=f"Δ {bs_b-bs_a:+.5f}", border=True)
    kc[3].metric("Bump steer B (°/mm)", f"{bs_b:+.5f}", border=True)

    def overlay(field, title, ylab):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sweep_a["heave_mm"], y=sweep_a[field],
                                   mode="lines+markers", name="A",
                                   line=dict(width=2, color="#1f77b4")))
        fig.add_trace(go.Scatter(x=sweep_b["heave_mm"], y=sweep_b[field],
                                   mode="lines+markers", name="B",
                                   line=dict(width=2, color="#d62728", dash="dash")))
        fig.update_layout(title=title, xaxis_title="Heave (mm)",
                           yaxis_title=ylab, template="plotly_white",
                           hovermode="x unified")
        return fig

    pc1, pc2 = st.columns(2)
    with pc1: st.plotly_chart(overlay("camber_deg", "Camber vs Heave", "Camber (°)"),
                                width="stretch")
    with pc2: st.plotly_chart(overlay("toe_deg", "Δ Toe vs Heave", "Δ Toe (°)"),
                                width="stretch")

    fig_rc = go.Figure()
    fig_rc.add_trace(go.Scatter(x=sweep_a["rc_y_mm"], y=sweep_a["rc_z_mm"],
                                  mode="lines+markers", name="RC A",
                                  line=dict(width=2, color="#1f77b4")))
    fig_rc.add_trace(go.Scatter(x=sweep_b["rc_y_mm"], y=sweep_b["rc_z_mm"],
                                  mode="lines+markers", name="RC B",
                                  line=dict(width=2, color="#d62728", dash="dash")))
    fig_rc.update_layout(title="Roll Center (Y × Z)",
                          xaxis_title="RC Y", yaxis_title="RC Z",
                          template="plotly_white")
    fig_rc.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig_rc, width="stretch")
