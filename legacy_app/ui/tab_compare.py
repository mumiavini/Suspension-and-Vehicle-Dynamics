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
from analysis.vdcore_bridge import (
    BridgeConversionError,
    CornerInputs,
    legacy_corner_to_vdcore,
    vdcore_sweep,
)
from vdcore.geometry.derived import mechanical_trail_mm, scrub_radius_mm
from vdcore.geometry.solver import DWSolver
from ui.shared import (
    load_hardpoints_from_state,
    render_empty_state,
    build_corner_safe,
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

    # Static KPIs come from DWSolver, not the legacy Corner methods. Those
    # reported static camber as 0.000 whatever the design (the legacy model
    # cannot infer it from hardpoints), did not fold right-side scrub, and had
    # mechanical trail sign-inverted for this frame — so a side-by-side built on
    # them could show two geometries as identical when they were not.
    inputs = CornerInputs.from_vehicle_setup(st.session_state.get("vehicle_setup", {}))
    try:
        vd_a = legacy_corner_to_vdcore(corner_a, tie_rod_a, corner_a.corner_id, inputs)
        vd_b = legacy_corner_to_vdcore(corner_b, tie_rod_b, corner_b.corner_id, inputs)
        res_a, res_b = DWSolver(vd_a).solve(), DWSolver(vd_b).solve()
    except (BridgeConversionError, RuntimeError, ValueError) as exc:
        st.error(f"❌ Could not solve one of the geometries: {exc}")
        st.stop()

    def _derived(result, fn):
        try:
            return fn(result)
        except ValueError:
            return float("nan")

    metrics = [
        ("Caster (°)",          res_a.caster_deg,            res_b.caster_deg),
        ("KPI (°)",             res_a.kpi_deg,               res_b.kpi_deg),
        ("Static camber (°)",   res_a.camber_deg,            res_b.camber_deg),
        ("Toe per side (°)",    res_a.toe_deg_per_side,      res_b.toe_deg_per_side),
        ("Scrub (mm)",          _derived(res_a, scrub_radius_mm),
                                _derived(res_b, scrub_radius_mm)),
        ("Trail (mm)",          _derived(res_a, mechanical_trail_mm),
                                _derived(res_b, mechanical_trail_mm)),
        ("Steer Arm (mm)",      corner_a.steer_arm_length_mm(tie_rod_a.outboard),
                                 corner_b.steer_arm_length_mm(tie_rod_b.outboard)),
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

    # Sweeps on DWSolver. The legacy sweep returned camber gain with the SIGN
    # INVERTED (+0.0388 where the truth is -0.0384 deg/mm), so an A/B comparison
    # built on it could rank two geometries backwards.
    with st.spinner("Running sweeps..."):
        params = (cmp_h_min, cmp_h_max, cmp_h_step)
        sweep_a = vdcore_sweep(vd_a, "Heave", params)
        sweep_b = vdcore_sweep(vd_b, "Heave", params)

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
