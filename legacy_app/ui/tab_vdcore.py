"""
ui/tab_vdcore.py
================
The "vdcore (validated)" tab — a second option that runs the SAME loaded
geometry through the validated ``vdcore`` 3D solver instead of the legacy
strut-to-midpoint solver.

This tab is the honest counterpart to the Analysis tab. It shows the dynamic
KPIs the legacy solver gets wrong (camber gain, roll-centre migration, RC
height, roll cambers) computed by ``vdcore.geometry.solver.DWSolver``, which
constrains all six DOF with the real linkage and is covered by the test suite.
A side-by-side delta table makes the correction visible.

Everything comes from ``analysis/vdcore_bridge.py``; this module is presentation
only (Streamlit + plotly). Plotly is allowed here — this is ``legacy_app/``, an
application layer, not the pure ``vdcore/`` library.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from analysis.io_hardpoints import build_vehicle_from_dataframe
from analysis.sweeps import SweepRunner, camber_gain_per_mm, rc_migration_range
from analysis.vdcore_bridge import (
    AxleVdcoreKPIs,
    BridgeConversionError,
    CornerInputs,
    VdcoreKPIs,
    compute_vdcore_kpis_cached,
    compute_vdcore_setup_sheet_cached,
)
from geometry import KinematicSolver3D
from ui.shared import load_hardpoints_from_state, render_empty_state

# KPIs that vdcore does NOT cover from loaded hardpoints alone. Kept in the UI
# so the tab tells the designer why they are absent rather than silently omitting
# them (they need a synthesised corner from sla_geometry / steering_geometry).
# Scrub radius and mechanical trail were moved OUT of this list: they are now
# computed by ``vdcore.geometry.derived`` and appear on the setup sheet below.
_NOT_COVERED = (
    "anti-dive", "anti-squat", "Ackermann",
)

# Source label used for every geometry row the validated solver produces.
_VDCORE_SRC = "📐 vdcore (validated)"
# Source label for the rows that genuinely need a synthesised corner.
_NOT_VDCORE_SRC = "⚠️ not vdcore — needs synthesised corner (steering_geometry.py)"


def _fmt(value: Optional[float], digits: int = 4, unit: str = "") -> str:
    """Format a metric, showing an em dash for None / NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    suffix = f" {unit}" if unit else ""
    return f"{value:.{digits}f}{suffix}"


def _legacy_dynamic_kpis(df: pl.DataFrame) -> dict[str, dict[str, float]]:
    """Recompute the legacy dynamic KPIs for the delta table.

    Mirrors the heave-sweep block of ``tab_analysis._compute_axle_cached`` so the
    numbers shown as "legacy" are exactly what the Analysis tab produces. Returns
    ``{"front": {...}, "rear": {...}}`` with camber gain (deg/mm) and RC
    migration Z range (mm). Any failure degrades to NaN, never a crash.
    """
    out: dict[str, dict[str, float]] = {}
    vehicle, tie_rods = build_vehicle_from_dataframe(df)
    axles = {
        "front": (vehicle.front_left, tie_rods["FL"]),
        "rear": (vehicle.rear_left, tie_rods["RL"]),
    }
    for name, (corner, tie_rod) in axles.items():
        try:
            runner = SweepRunner(solver=KinematicSolver3D(corner, tie_rod))
            heave = runner.heave_sweep(-25.0, 25.0, 2.5)
            _, rc_dz = rc_migration_range(heave)
            out[name] = {
                "camber_gain_deg_per_mm": camber_gain_per_mm(heave),
                "rc_migration_z_mm": rc_dz,
            }
        except Exception:
            out[name] = {
                "camber_gain_deg_per_mm": float("nan"),
                "rc_migration_z_mm": float("nan"),
            }
    return out


def _render_axle_cards(kpi: AxleVdcoreKPIs) -> None:
    """Metric cards for one axle. Greys out non-converged computations."""
    st.markdown(f"#### {kpi.label} axle")

    if kpi.error is not None:
        st.warning(f"⚠️ {kpi.label} axle did not solve: {kpi.error}")
        return

    rates = kpi.rates
    roll = kpi.roll

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Camber gain",
        _fmt(rates.camber_gain_deg_per_mm if rates else None, 4, "°/mm"),
        help="d(camber)/d(bump). Negative gains negative camber in bump.",
    )
    c2.metric(
        "RC migration",
        _fmt(rates.rc_migration_mm_per_mm if rates else None, 4, "mm/mm"),
        help="Chassis-referenced roll-centre height change per mm of parallel "
             "bump.",
    )
    c3.metric(
        "Half-track change",
        _fmt(rates.half_track_change_mm_per_mm if rates else None, 4, "mm/mm"),
        help="Contact-patch lateral movement per mm of bump (scrub in bump).",
    )

    c4, c5, c6 = st.columns(3)
    c4.metric(
        "Camber @ full bump",
        _fmt(rates.camber_full_bump_deg if rates else None, 3, "°"),
    )
    c5.metric(
        "Camber @ full droop",
        _fmt(rates.camber_full_droop_deg if rates else None, 3, "°"),
    )
    c6.metric(
        "RC height range",
        _fmt(
            (rates.rc_max_mm - rates.rc_min_mm) if rates else None, 2, "mm"
        ),
        help="RC height span over the bump/droop sweep (chassis-referenced).",
    )

    st.markdown(f"##### Roll — chassis rolled {kpi.roll_deg:.2f}°, both wheels on road")
    if roll is None:
        st.info("Roll solve did not converge for this axle.")
        return
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Outer camber", _fmt(roll.outer_camber_deg, 3, "°"),
              help="Road-relative. Outer wheel loses the roll angle.")
    r2.metric("Inner camber", _fmt(roll.inner_camber_deg, 3, "°"))
    r3.metric("RC height", _fmt(roll.rc_height_mm, 2, "mm"),
              help="Ground-referenced roll-centre height at this roll angle.")
    r4.metric("RC lateral", _fmt(roll.rc_lateral_mm, 2, "mm"),
              help="Sideways RC shift. The legacy app reports ~0 here because it "
                   "averages the two sides and the lateral terms cancel.")


def _render_delta_table(vd: VdcoreKPIs, legacy: dict[str, dict[str, float]]) -> None:
    """Side-by-side legacy-vs-vdcore table for the two overlapping KPIs."""
    st.markdown("#### Legacy vs vdcore — the correction, made visible")

    def rows_for(axle_name: str, vd_axle: AxleVdcoreKPIs) -> list[dict]:
        leg = legacy.get(axle_name, {})
        rates = vd_axle.rates
        vd_cg = rates.camber_gain_deg_per_mm if rates else float("nan")
        vd_rc = (rates.rc_max_mm - rates.rc_min_mm) if rates else float("nan")
        return [
            {
                "Axle": vd_axle.label,
                "KPI": "Camber gain (°/mm)",
                "Legacy": _fmt(leg.get("camber_gain_deg_per_mm"), 4),
                "vdcore": _fmt(vd_cg, 4),
            },
            {
                "Axle": vd_axle.label,
                "KPI": "RC migration Z range (mm)",
                "Legacy": _fmt(leg.get("rc_migration_z_mm"), 2),
                "vdcore": _fmt(vd_rc, 2),
            },
        ]

    rows = rows_for("front", vd.front) + rows_for("rear", vd.rear)
    st.dataframe(pl.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "The legacy strut-to-midpoint solver barely rotates the arms under "
        "travel, so its RC migration collapses toward ~1 mm and its camber gain "
        "reads low. vdcore constrains the real linkage."
    )


def _render_camber_plot(vd: VdcoreKPIs) -> None:
    """Camber vs wheel-travel sweep, front and rear left corners."""
    fig = go.Figure()
    for axle in (vd.front, vd.rear):
        sweep = axle.camber_sweep_left
        if sweep is None:
            continue
        travel = np.array(sweep.wheel_travel_mm)
        camber = np.array(sweep.camber_deg)
        fig.add_trace(go.Scatter(
            x=travel, y=camber, mode="lines",
            name=f"{axle.label} ({sweep.corner_id})",
        ))
    fig.update_layout(
        title="Camber vs wheel travel (vdcore)",
        xaxis_title="Wheel travel (mm, + = bump)",
        yaxis_title="Camber (°)",
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, width="stretch")


def _render_rc_migration_plot(vd: VdcoreKPIs) -> None:
    """RC height vs parallel wheel travel, front and rear.

    Plots the CHASSIS-referenced ``rc_height_sweep`` (built from ``sample_corner``
    / ``axle_rates``), not ``roll_centre_migration``. The latter constructs the
    instant centre from world-frame ball joints against chassis-fixed pivots, so
    its RC barely moves (~1 mm) — the same artefact the legacy solver produced —
    and would visually contradict the migration range in the delta table.
    """
    fig = go.Figure()
    for axle in (vd.front, vd.rear):
        sweep = axle.rc_height_sweep
        if sweep is None:
            continue
        travel = np.array(sweep.wheel_travel_mm)
        rc_h = np.array(sweep.rc_height_mm)
        fig.add_trace(go.Scatter(
            x=travel, y=rc_h, mode="lines", name=axle.label,
        ))
    fig.update_layout(
        title="Roll-centre height vs parallel travel (vdcore)",
        xaxis_title="Wheel travel (mm, + = bump)",
        yaxis_title="RC height (mm, chassis-referenced)",
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, width="stretch")


def _wheel_rate(spring_rate: float, mr: float) -> float:
    """Wheel rate (N/mm) = spring_rate × MR². NaN if inputs missing."""
    if spring_rate <= 0 or mr <= 0:
        return float("nan")
    return spring_rate * mr * mr


def _roll_rate(wheel_rate: float, track_mm: float) -> float:
    """Roll rate per wheel (Nm/°) from wheel rate and track. NaN if missing.

    K_roll = (1/2) × K_wheel × T² × (π/180), K_wheel in N/m, T in m.
    Mirrors ``tab_analysis._roll_rate`` so the two sheets agree.
    """
    if math.isnan(wheel_rate) or track_mm <= 0:
        return float("nan")
    t_m = track_mm / 1000.0
    return 0.5 * (wheel_rate * 1000.0) * t_m * t_m * math.pi / 180.0


def _natural_freq(wheel_rate: float, sprung_per_corner: float) -> float:
    """Sprung-mass natural frequency (Hz) = (1/2π)·√(K/M). NaN if missing."""
    if math.isnan(wheel_rate) or sprung_per_corner <= 0:
        return float("nan")
    k = wheel_rate * 1000.0  # N/m
    return (1.0 / (2.0 * math.pi)) * math.sqrt(k / sprung_per_corner)


def _fmt_pair(v_l: Optional[float], v_r: Optional[float], digits: int = 3) -> str:
    """Format 'L / R' for a per-wheel quantity, em dash for None/NaN."""
    return f"{_fmt(v_l, digits)} / {_fmt(v_r, digits)}"


def _render_setup_sheet(
    df: pl.DataFrame,
    inputs: CornerInputs,
    *,
    roll_deg: float,
    travel_mm: float,
) -> None:
    """Full documentation setup sheet, geometry rows sourced from vdcore.

    Mirrors the Analysis tab's "Complete Setup Sheet" (same categories and row
    labels for easy cross-reference) but every geometry row is computed by the
    validated ``DWSolver`` (Source ``📐 vdcore (validated)``). User-typed rows
    (tyre/wheel/suspension/mass/damper) pass through unchanged. Anti-dive /
    anti-squat / Ackermann are shown but flagged — they need a synthesised
    corner, not loadable hardpoints. A CSV of the full table is downloadable for
    documentation.

    The input widgets use ``vd_sheet_*`` keys, namespaced away from the Analysis
    tab's bare keys so the two tabs never collide in one session.
    """
    st.markdown("### 📋 Complete Setup Sheet (vdcore-validated)")
    st.caption(
        "The same sheet as the Analysis tab, but **every geometry row comes "
        "from the validated `DWSolver`**, not the legacy strut-to-midpoint "
        "solver. Fill the inputs below for the rows the hardpoints file does "
        "not carry (masses, springs, dampers)."
    )

    # ─── User inputs (namespaced keys — never the Analysis tab's) ─────────────
    with st.expander("🔧 Additional inputs (masses, springs, dampers)", expanded=False):
        t_tire, t_susp, t_mass, t_damp, t_other = st.tabs(
            ["🛞 Tires & Wheels", "🔩 Suspension & Spring", "⚖️ Masses",
             "🌊 Damper", "📝 Other"]
        )
        with t_tire:
            c1, c2 = st.columns(2)
            with c1:
                tire_size = st.text_input("Tire size, compound, make", value="",
                                          key="vd_sheet_tire")
                wheel_diam = st.number_input("Wheel diameter (inch)", 0.0, value=10.0,
                                             step=0.5, key="vd_sheet_wdiam")
            with c2:
                wheel_mat = st.text_input("Wheel material / construction", value="",
                                          key="vd_sheet_wmat")
                wheel_wid = st.number_input("Wheel width (inch)", 0.0, value=7.0,
                                            step=0.5, key="vd_sheet_wwid")
        with t_susp:
            c1, c2 = st.columns(2)
            with c1:
                susp_type = st.text_input("Suspension type",
                                          value="Double wishbone push/pull-rod",
                                          key="vd_sheet_susp_type")
                travel_f = st.number_input("Design travel — FRONT (mm)", 0.0,
                                           value=0.0, step=1.0, key="vd_sheet_travel_f")
                spring_f = st.number_input("Spring rate — FRONT (N/mm)", 0.0,
                                           value=0.0, step=1.0, key="vd_sheet_spring_f")
                mr_f = st.number_input("Motion Ratio — FRONT", 0.0, value=0.0,
                                       step=0.05, format="%.3f", key="vd_sheet_mr_f")
            with c2:
                susp_adj = st.text_input("Static camber adjustment method", value="",
                                         key="vd_sheet_susp_adj")
                travel_r = st.number_input("Design travel — REAR (mm)", 0.0,
                                           value=0.0, step=1.0, key="vd_sheet_travel_r")
                spring_r = st.number_input("Spring rate — REAR (N/mm)", 0.0,
                                           value=0.0, step=1.0, key="vd_sheet_spring_r")
                mr_r = st.number_input("Motion Ratio — REAR", 0.0, value=0.0,
                                       step=0.05, format="%.3f", key="vd_sheet_mr_r")
        with t_mass:
            c1, c2, c3 = st.columns(3)
            with c1:
                total_mass = st.number_input("Total mass w/ driver (kg)", 0.0,
                                             value=0.0, step=5.0, key="vd_sheet_mass")
            with c2:
                weight_dist_f = st.number_input("Weight distribution — FRONT (%)",
                                                0.0, 100.0, value=45.0, step=0.5,
                                                key="vd_sheet_wdist")
            with c3:
                unsprung = st.number_input("Unsprung mass per corner (kg)", 0.0,
                                           value=0.0, step=0.5, key="vd_sheet_unsprung")
        with t_damp:
            c1, c2 = st.columns(2)
            with c1:
                jounce_f = st.number_input("Jounce damping — FRONT (% crit)", 0.0,
                                           200.0, value=0.0, step=5.0, key="vd_sheet_jf")
                rebound_f = st.number_input("Rebound damping — FRONT (% crit)", 0.0,
                                            200.0, value=0.0, step=5.0, key="vd_sheet_rf")
            with c2:
                jounce_r = st.number_input("Jounce damping — REAR (% crit)", 0.0,
                                           200.0, value=0.0, step=5.0, key="vd_sheet_jr")
                rebound_r = st.number_input("Rebound damping — REAR (% crit)", 0.0,
                                            200.0, value=0.0, step=5.0, key="vd_sheet_rr")
        with t_other:
            ackermann_adj = st.selectbox(
                "Adjustable Ackermann?",
                ["No", "Yes (multiple positions)", "Yes (continuous)"],
                key="vd_sheet_ack_adj",
            )

    # ─── Geometry rows from the validated solver ──────────────────────────────
    try:
        sheet = compute_vdcore_setup_sheet_cached(
            df, inputs, roll_deg=roll_deg, travel_mm=travel_mm,
        )
    except BridgeConversionError as exc:
        st.error(
            f"❌ Could not compute the vdcore setup sheet: {exc}. "
            "The geometry rows are unavailable."
        )
        return
    front, rear = sheet["front"], sheet["rear"]

    # ─── Derived rows that depend on the user inputs above ────────────────────
    from analysis.io_hardpoints import build_vehicle_from_dataframe
    vehicle, _ = build_vehicle_from_dataframe(df)
    wr_f, wr_r = _wheel_rate(spring_f, mr_f), _wheel_rate(spring_r, mr_r)
    rr_f = _roll_rate(wr_f, vehicle.track_front_mm)
    rr_r = _roll_rate(wr_r, vehicle.track_rear_mm)
    sprung_f_pc = sprung_r_pc = float("nan")
    if total_mass > 0 and unsprung > 0:
        sprung_total = total_mass - 4.0 * unsprung
        if sprung_total > 0:
            wd = weight_dist_f / 100.0
            sprung_f_pc = sprung_total * wd / 2.0
            sprung_r_pc = sprung_total * (1.0 - wd) / 2.0
    nf_f, nf_r = _natural_freq(wr_f, sprung_f_pc), _natural_freq(wr_r, sprung_r_pc)

    # ─── Build the table (mirrors tab_analysis category order and labels) ─────
    rows: list[dict[str, str]] = []
    category = "General"

    def add(param: str, unit: str, f_val: str, r_val: str, origin: str) -> None:
        rows.append({"Category": category, "Parameter": param, "Unit": unit,
                     "Front": f_val, "Rear": r_val, "Source": origin})

    category = "🛞 Tires & Wheels"
    add("Tire size, compound, make", "", tire_size or "—", tire_size or "—",
        "⌨️ input")
    add("Wheel (diameter × width)", "inch",
        f"{wheel_diam:.1f} × {wheel_wid:.1f}" if wheel_diam else "—",
        f"{wheel_diam:.1f} × {wheel_wid:.1f}" if wheel_diam else "—", "⌨️ input")
    add("Wheel material / construction", "", wheel_mat or "—", wheel_mat or "—",
        "⌨️ input")

    category = "🔩 Suspension & Rates"
    add("Suspension type", "", susp_type or "—", susp_type or "—", "⌨️ input")
    add("Suspension design travel", "mm",
        _fmt(travel_f if travel_f > 0 else None, 1),
        _fmt(travel_r if travel_r > 0 else None, 1), "⌨️ input")
    add("Wheel rate (chassis → wheel center)", "N/mm",
        _fmt(wr_f, 2), _fmt(wr_r, 2), "🧮 derived from spring + MR")
    add("Roll rate (chassis → wheel center)", "Nm/deg",
        _fmt(rr_f, 1), _fmt(rr_r, 1), "🧮 derived from wheel rate + track")
    add("Sprung mass natural frequency", "Hz",
        _fmt(nf_f, 2), _fmt(nf_r, 2), "🧮 derived from wheel rate + mass")
    add("Jounce damping", "% critical",
        _fmt(jounce_f if jounce_f > 0 else None, 0),
        _fmt(jounce_r if jounce_r > 0 else None, 0), "⌨️ input")
    add("Rebound damping", "% critical",
        _fmt(rebound_f if rebound_f > 0 else None, 0),
        _fmt(rebound_r if rebound_r > 0 else None, 0), "⌨️ input")
    add("Motion ratio", "x:1",
        _fmt(mr_f if mr_f > 0 else None, 3),
        _fmt(mr_r if mr_r > 0 else None, 3), "⌨️ input")

    category = "🎢 Kinematics"
    add("Ride Camber (rate of change)", "deg/m",
        _fmt(front.get("ride_camber_dpm"), 2), _fmt(rear.get("ride_camber_dpm"), 2),
        _VDCORE_SRC)
    add("Roll Camber", "deg/deg",
        _fmt(front.get("roll_camber"), 4), _fmt(rear.get("roll_camber"), 4),
        _VDCORE_SRC)
    add("Anti dive / Anti squat", "%", "—", "—", _NOT_VDCORE_SRC)
    add("Roll center height above ground, static", "mm",
        _fmt(front.get("rc_static"), 2), _fmt(rear.get("rc_static"), 2), _VDCORE_SRC)
    add("Roll center @ roll — height", "mm",
        _fmt(front.get("rc_1g_z"), 2), _fmt(rear.get("rc_1g_z"), 2),
        f"{_VDCORE_SRC} @ {roll_deg:.2f}° roll")
    add("Roll center @ roll — lateral", "mm",
        _fmt(front.get("rc_1g_y"), 2), _fmt(rear.get("rc_1g_y"), 2),
        f"{_VDCORE_SRC} @ {roll_deg:.2f}° roll")

    category = "📐 Static alignment"
    add("Kingpin Inclination (L / R)", "deg",
        _fmt_pair(front.get("kpi_l"), front.get("kpi_r")),
        _fmt_pair(rear.get("kpi_l"), rear.get("kpi_r")), _VDCORE_SRC)
    add("Caster (L / R)", "deg",
        _fmt_pair(front.get("caster_l"), front.get("caster_r")),
        _fmt_pair(rear.get("caster_l"), rear.get("caster_r")), _VDCORE_SRC)
    add("Scrub radius (L / R)", "mm",
        _fmt_pair(front.get("scrub_l"), front.get("scrub_r"), 2),
        _fmt_pair(rear.get("scrub_l"), rear.get("scrub_r"), 2), _VDCORE_SRC)
    add("Mechanical trail (L / R)", "mm",
        _fmt_pair(front.get("trail_l"), front.get("trail_r"), 2),
        _fmt_pair(rear.get("trail_l"), rear.get("trail_r"), 2), _VDCORE_SRC)
    add("Static Sum Toe (− out, + in)", "deg",
        _fmt(front.get("sum_toe"), 4), _fmt(rear.get("sum_toe"), 4), _VDCORE_SRC)
    add("Static camber (L / R)", "deg",
        _fmt_pair(front.get("camber_l"), front.get("camber_r")),
        _fmt_pair(rear.get("camber_l"), rear.get("camber_r")), _VDCORE_SRC)
    add("Static camber adjustment method", "", susp_adj or "—", susp_adj or "—",
        "⌨️ input")

    category = "🕹️ Steering"
    add("Static Ackermann", "%", "—", "N/A", _NOT_VDCORE_SRC)
    add("Adjustable Ackermann?", "", ackermann_adj, "—", "⌨️ input")

    category = "⚖️ Masses"
    if total_mass > 0:
        add("Total mass w/ driver", "kg", f"{total_mass:.1f}", f"{total_mass:.1f}",
            "⌨️ input")
        if not math.isnan(sprung_f_pc):
            add("Sprung mass per corner", "kg",
                _fmt(sprung_f_pc, 1), _fmt(sprung_r_pc, 1), "🧮 derived")
        add("Unsprung mass per corner", "kg", f"{unsprung:.1f}", f"{unsprung:.1f}",
            "⌨️ input")
        add("Weight distribution", "%",
            f"{weight_dist_f:.1f}", f"{100 - weight_dist_f:.1f}", "⌨️ input")

    # ─── Category filter + render + download ──────────────────────────────────
    categories = list(dict.fromkeys(r["Category"] for r in rows))
    cat_sel = st.pills("Filter by category", ["All"] + categories, default="All",
                       key="vd_sheet_category")
    rows_view = rows if (not cat_sel or cat_sel == "All") else \
        [r for r in rows if r["Category"] == cat_sel]

    st.dataframe(pl.DataFrame(rows_view), width="stretch", hide_index=True,
                 height=min(80 + 35 * len(rows_view), 900))
    st.caption(
        "**Legend:** "
        f"{_VDCORE_SRC} — geometry from the real linkage · "
        "🧮 derived (needs the inputs above) · ⌨️ user input · "
        "⚠️ not vdcore — needs a synthesised corner from `steering_geometry.py`"
    )
    st.download_button(
        "⬇️ Download setup sheet (CSV)",
        data=pl.DataFrame(rows).write_csv().encode(),
        file_name="setup_sheet_vdcore.csv",
        mime="text/csv",
        key="vd_sheet_download",
    )


def render() -> None:
    st.header("vdcore (validated) — dynamic KPIs on the real linkage")
    st.markdown(
        "This tab runs the **same loaded geometry** through the validated "
        "`vdcore` 3D solver (`DWSolver`), which constrains all six degrees of "
        "freedom with the real wishbone linkage. Unlike the **Analysis** tab, "
        "the dynamic KPIs here are trustworthy — they are covered by the test "
        "suite (`tests/benchmarks/test_fsae2027_design.py`)."
    )

    df = load_hardpoints_from_state()
    if df is None:
        render_empty_state(
            "The vdcore tab computes the **dynamic** KPIs (camber gain, "
            "roll-centre migration, roll cambers) correctly from the loaded "
            "hardpoints.",
            key="empty_vdcore",
        )
        return

    # Static camber / loaded radius come from the sidebar's vehicle-setup
    # (single source of truth); roll and travel are tab-local sweep parameters.
    vs = st.session_state.get("vehicle_setup", {})
    inputs = CornerInputs.from_vehicle_setup(vs)

    with st.expander("🔧 vdcore sweep parameters", expanded=False):
        st.caption(
            f"Static camber **{inputs.static_camber_deg:.2f}°** and loaded "
            f"radius **{inputs.loaded_radius_mm:.0f} mm** come from the sidebar "
            "(tagged `estimate` in provenance). Change them there."
        )
        col1, col2 = st.columns(2)
        with col1:
            roll_deg = st.slider(
                "Roll angle (°)", 0.0, 3.0, 1.5, step=0.1, key="vd_roll",
            )
        with col2:
            travel_mm = st.slider(
                "Bump/droop travel (± mm)", 5.0, 40.0, 25.0, step=1.0,
                key="vd_travel",
            )

    try:
        vd = compute_vdcore_kpis_cached(
            df, inputs, roll_deg=float(roll_deg), travel_mm=float(travel_mm),
        )
    except BridgeConversionError as exc:
        st.error(
            f"❌ Could not convert the loaded geometry for vdcore: {exc}"
        )
        return

    # Any non-converged solves? Warn once at the top.
    if any(a.error is not None for a in (vd.front, vd.rear)):
        st.warning(
            "⚠️ One or more axles did not fully converge. Non-converged values "
            "are shown as `—`, never as a plausible-looking number."
        )

    _render_axle_cards(vd.front)
    st.divider()
    _render_axle_cards(vd.rear)
    st.divider()

    legacy = _legacy_dynamic_kpis(df)
    _render_delta_table(vd, legacy)
    st.divider()

    c_left, c_right = st.columns(2)
    with c_left:
        _render_camber_plot(vd)
    with c_right:
        _render_rc_migration_plot(vd)

    st.divider()
    _render_setup_sheet(
        df, inputs, roll_deg=float(roll_deg), travel_mm=float(travel_mm),
    )

    st.info(
        "**Still flagged.** vdcore does not compute "
        + ", ".join(_NOT_COVERED)
        + " from loaded hardpoints — those need a synthesised corner from "
        "`sla_geometry.py` / `steering_geometry.py`, so the setup sheet above "
        "marks them ⚠️. Scrub radius and mechanical trail, by contrast, are now "
        "computed by `vdcore` and appear on the sheet."
    )
