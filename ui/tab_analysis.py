"""
ui/tab_analysis.py
==================
📊 Analysis tab — complete vehicle setup sheet: static and dynamic KPIs
side by side for front/rear, additional physical inputs and optional
detailed sweeps.
"""

from __future__ import annotations

import io
import math

import numpy as np
import polars as pl
import streamlit as st
import plotly.graph_objects as go

from analysis.io_hardpoints import (
    build_vehicle_from_dataframe,
    VALID_CORNERS,
)
from geometry import KinematicSolver3D
from analysis.sweeps import (
    SweepRunner,
    camber_gain_per_mm,
    bump_steer_per_mm,
    rc_migration_range,
    plot_camber_vs_heave,
    plot_bump_steer,
    plot_rc_migration,
    plot_caster_kpi_vs_steer,
)
from analysis.kpis import (
    static_sum_toe_deg,
    ackermann_geometry,
    steer_ratio_and_cfactor,
    steer_ratio_from_pinion,
    roll_center_at_1g_lat,
)
from ui.shared import (
    load_hardpoints_from_state,
    render_empty_state,
    build_corner_safe,
    build_vehicle_safe,
    run_sweep_cached,
)


@st.cache_data(show_spinner=False, max_entries=16)
def _compute_axle_cached(df_csv: str, axle: str, brake_bias: float,
                         c_factor_mm: float, roll_stiff: float) -> dict:
    """
    All KPIs and sweeps for one axle ("front"/"rear"), cached.

    Without the cache, the 2 heave sweeps + 2 roll sweeps would run on EVERY
    interaction with ANY widget in the app (Streamlit re-runs the whole script).
    The key is the hardpoints CSV + the inputs that affect the result.
    """
    df = pl.read_csv(io.StringIO(df_csv))
    vehicle, tie_rods = build_vehicle_from_dataframe(df)
    is_front = axle == "front"
    if is_front:
        left_corner, right_corner = vehicle.front_left, vehicle.front_right
        left_tr, right_tr = tie_rods["FL"], tie_rods["FR"]
    else:
        left_corner, right_corner = vehicle.rear_left, vehicle.rear_right
        left_tr, right_tr = tie_rods["RL"], tie_rods["RR"]

    res: dict[str, object] = {}

    # ── Static kinematics ──
    res["caster_l"] = left_corner.static_caster_deg()
    res["caster_r"] = right_corner.static_caster_deg()
    res["kpi_l"]    = left_corner.static_kpi_deg()
    res["kpi_r"]    = right_corner.static_kpi_deg()
    res["camber_l"] = left_corner.static_camber_deg()
    res["camber_r"] = right_corner.static_camber_deg()
    res["scrub_l"]  = left_corner.static_scrub_radius_mm()
    res["scrub_r"]  = right_corner.static_scrub_radius_mm()
    res["trail_l"]  = left_corner.static_mechanical_trail_mm()
    res["trail_r"]  = right_corner.static_mechanical_trail_mm()
    res["rc_static"]= 0.5 * (left_corner.roll_center_height_mm()
                              + right_corner.roll_center_height_mm())
    res["sum_toe"]  = static_sum_toe_deg(left_corner, left_tr,
                                          right_corner, right_tr)

    # ── Heave sweep (left is enough; geometry is symmetric) ──
    solver_l = KinematicSolver3D(left_corner, left_tr)
    runner_l = SweepRunner(solver=solver_l)
    heave_sweep = runner_l.heave_sweep(-25.0, 25.0, 2.5)
    res["ride_camber_dpm"] = camber_gain_per_mm(heave_sweep) * 1000.0
    res["camber_gain"]     = camber_gain_per_mm(heave_sweep)
    res["bump_steer"]      = bump_steer_per_mm(heave_sweep)
    res["rc_dy"], res["rc_dz"] = rc_migration_range(heave_sweep)

    # ── Roll sweep for roll camber ──
    solver_l.reset_seed()
    roll_sweep = runner_l.roll_sweep(-2.0, 2.0, 0.25)
    if len(roll_sweep) > 1:
        rolls   = roll_sweep["roll_deg"]
        cambers = roll_sweep["camber_deg"]
        if (rolls.max() - rolls.min()) > 1e-6:
            res["roll_camber"] = float(np.polyfit(rolls, cambers, 1)[0])
        else:
            res["roll_camber"] = float("nan")
    else:
        res["roll_camber"] = float("nan")

    # ── RC @ 1g lateral ──
    try:
        rc1g = roll_center_at_1g_lat(
            left_corner, left_tr, right_corner, right_tr,
            roll_stiffness_deg_per_g=roll_stiff,
        )
        res["rc_1g_y"] = rc1g["rc_y_mm"]
        res["rc_1g_z"] = rc1g["rc_z_mm"]
    except Exception:
        res["rc_1g_y"] = float("nan")
        res["rc_1g_z"] = float("nan")

    # ── Anti-features ──
    if is_front:
        res["anti_dive"]  = left_corner.anti_dive_percent(brake_bias=brake_bias)
        res["anti_squat"] = float("nan")
    else:
        res["anti_dive"]  = float("nan")
        res["anti_squat"] = left_corner.anti_squat_percent(drive_fraction=1.0)

    # ── Ackermann (only meaningful at the front) ──
    if is_front:
        ack_info = ackermann_geometry(left_corner, left_tr,
                                       right_corner, right_tr,
                                       vehicle.rear_left)
        res["ackermann"] = ack_info["ackermann_percent"]
        res["steer_arm_l"] = ack_info["steer_arm_length_left"]
        res["steer_arm_r"] = ack_info["steer_arm_length_right"]

        sr_info = steer_ratio_and_cfactor(left_corner, left_tr)
        if c_factor_mm > 0:
            res["steer_ratio"] = steer_ratio_from_pinion(
                sr_info["rack_per_wheel_deg_mm_per_deg"], c_factor_mm)
        else:
            res["steer_ratio"] = float("nan")
        res["c_factor"] = c_factor_mm
    else:
        res["ackermann"]   = float("nan")
        res["steer_arm_l"] = float("nan")
        res["steer_arm_r"] = float("nan")
        res["steer_ratio"] = float("nan")
        res["c_factor"]    = float("nan")

    return res


def render() -> None:
    st.header("Analysis — Complete setup sheet")
    st.markdown(
        "Table with **all the vehicle parameters**, side by side for "
        "front and rear. Values are computed automatically from the "
        "hardpoints when possible; whatever needs additional input appears "
        "below."
    )

    df = load_hardpoints_from_state()
    if df is None:
        render_empty_state(
            "The analysis computes **all the vehicle KPIs** (camber, caster, "
            "roll center, bump steer…) from the loaded hardpoints.",
            key="empty_analysis",
        )
    else:
        vehicle, all_tie_rods = build_vehicle_safe(df)
        if vehicle is None or all_tie_rods is None:
            st.error("Could not build the complete vehicle from the file.")
            st.stop()

        # ─── USER INPUTS (physical parameters that cannot be computed) ───────
        with st.expander("🔧 **Additional inputs** — settings that do not come from the hardpoints", expanded=False):
            st.markdown(
                "These values are needed to compute wheel rate, roll rate, "
                "natural frequency, motion ratio and damping. Leave blank (0) "
                "for those KPIs to appear as `—` in the table."
            )

            tab_tire, tab_susp, tab_mass, tab_damper, tab_other = st.tabs([
                "🛞 Tires & Wheels",
                "🔩 Suspension & Spring",
                "⚖️ Masses",
                "🌊 Damper",
                "📝 Other",
            ])

            with tab_tire:
                c1, c2 = st.columns(2)
                with c1:
                    tire_size  = st.text_input("Tire size, compound, make",
                                                 value="", placeholder="e.g. 18.0×7.5-10 Hoosier R25B",
                                                 key="in_tire")
                    wheel_diam = st.number_input("Wheel diameter (inch)",
                                                   min_value=0.0, value=10.0, step=0.5,
                                                   key="in_wheel_diam")
                with c2:
                    wheel_mat  = st.text_input("Wheel material / construction",
                                                 value="", placeholder="e.g. forged aluminum 2-piece",
                                                 key="in_wheel_mat")
                    wheel_wid  = st.number_input("Wheel width (inch)",
                                                   min_value=0.0, value=7.0, step=0.5,
                                                   key="in_wheel_wid")

            with tab_susp:
                c1, c2 = st.columns(2)
                with c1:
                    susp_type  = st.text_input("Suspension type",
                                                 value="Double wishbone push/pull-rod",
                                                 key="in_susp_type")
                    susp_travel_f = st.number_input("Design travel — FRONT (mm)",
                                                      min_value=0.0, value=0.0, step=1.0,
                                                      key="in_travel_f",
                                                      help="Usable front heave travel (bump+rebound)")
                    spring_rate_f = st.number_input("Spring rate — FRONT (N/mm)",
                                                      min_value=0.0, value=0.0, step=1.0,
                                                      key="in_spring_f",
                                                      help="Front spring stiffness (at the spring axis, not the wheel)")
                    mr_front = st.number_input("Motion Ratio — FRONT",
                                                 min_value=0.0, value=0.0, step=0.05, format="%.3f",
                                                 key="in_mr_f",
                                                 help="MR = Δ(spring) / Δ(wheel). Typical FSAE: 0.7–1.1")
                with c2:
                    susp_adj    = st.text_input("Static camber adjustment method",
                                                  value="2 mm plates between upright and upper arm fixation",
                                                  key="in_susp_adj")
                    susp_travel_r = st.number_input("Design travel — REAR (mm)",
                                                      min_value=0.0, value=0.0, step=1.0,
                                                      key="in_travel_r")
                    spring_rate_r = st.number_input("Spring rate — REAR (N/mm)",
                                                      min_value=0.0, value=0.0, step=1.0,
                                                      key="in_spring_r")
                    mr_rear = st.number_input("Motion Ratio — REAR",
                                                 min_value=0.0, value=0.0, step=0.05, format="%.3f",
                                                 key="in_mr_r")
                arb_adj = st.text_input("Suspension adjustment methods (other)",
                                          value="",
                                          placeholder="e.g. ARB with 3 positions, variable preload",
                                          key="in_susp_methods")

            with tab_mass:
                c1, c2, c3 = st.columns(3)
                with c1:
                    total_mass = st.number_input("Total mass w/ driver (kg)",
                                                   min_value=0.0, value=0.0, step=5.0,
                                                   key="in_mass_total",
                                                   help="Car + driver + fuel")
                with c2:
                    weight_dist_f = st.number_input("Weight distribution — FRONT (%)",
                                                     min_value=0.0, max_value=100.0,
                                                     value=45.0, step=0.5,
                                                     key="in_weight_dist_f",
                                                     help="% of the weight on the front axle")
                with c3:
                    unsprung_per_corner = st.number_input("Unsprung mass per corner (kg)",
                                                            min_value=0.0, value=0.0, step=0.5,
                                                            key="in_unsprung",
                                                            help="Wheel + tire + upright + brake + ~50% of the arms")

            with tab_damper:
                c1, c2 = st.columns(2)
                with c1:
                    jounce_pct_f = st.number_input("Jounce damping — FRONT (% critical)",
                                                     min_value=0.0, max_value=200.0,
                                                     value=0.0, step=5.0,
                                                     key="in_jounce_f")
                    rebound_pct_f = st.number_input("Rebound damping — FRONT (% critical)",
                                                      min_value=0.0, max_value=200.0,
                                                      value=0.0, step=5.0,
                                                      key="in_rebound_f")
                with c2:
                    jounce_pct_r = st.number_input("Jounce damping — REAR (% critical)",
                                                     min_value=0.0, max_value=200.0,
                                                     value=0.0, step=5.0,
                                                     key="in_jounce_r")
                    rebound_pct_r = st.number_input("Rebound damping — REAR (% critical)",
                                                      min_value=0.0, max_value=200.0,
                                                      value=0.0, step=5.0,
                                                      key="in_rebound_r")

            with tab_other:
                roll_stiff = st.number_input("Roll stiffness (°/g) — used for RC@1g",
                                               min_value=0.1, value=1.5, step=0.1,
                                               key="in_roll_stiff")
                ackermann_adj = st.selectbox(
                    "Adjustable Ackermann?",
                    ["No", "Yes (multiple positions)", "Yes (continuous)"],
                    key="in_ack_adj",
                )

        # ─── INFRASTRUCTURE: compute F and R sweeps at once (cached) ──────────
        vs = st.session_state["vehicle_setup"]

        # Single textual representation of the df = cache invalidation key
        df_signature = df.write_csv()

        with st.spinner("Computing KPIs for both axles..."):
            front_data = _compute_axle_cached(df_signature, "front",
                                              vs["brake_bias"],
                                              vs["c_factor_mm"], roll_stiff)
            rear_data  = _compute_axle_cached(df_signature, "rear",
                                              vs["brake_bias"],
                                              vs["c_factor_mm"], roll_stiff)

        # ─── Derived calculations that depend on USER inputs ─────────────────
        def _wheel_rate(spring_rate: float, mr: float) -> float:
            """Wheel rate (N/mm) = spring_rate × MR²."""
            if spring_rate <= 0 or mr <= 0:
                return float("nan")
            return spring_rate * mr * mr

        def _roll_rate(wheel_rate: float, track_mm: float) -> float:
            """Roll rate per wheel (Nm/°) = wheel_rate × track² / 2 × π/180 / 1000.

            Formula: K_roll = (1/2) × K_wheel × T² × (π/180) [Nm/°]
            where K_wheel is in N/mm and T in mm. The factor 1000 converts mm² to m².
            """
            if math.isnan(wheel_rate) or track_mm <= 0:
                return float("nan")
            # wheel_rate N/mm = wheel_rate × 1000 N/m
            # roll rate in Nm/rad = (1/2) × K × T² (T in m)
            # convert to Nm/°: × π/180
            T_m = track_mm / 1000.0
            return 0.5 * (wheel_rate * 1000.0) * T_m * T_m * math.pi / 180.0

        def _natural_freq(wheel_rate: float, sprung_per_corner: float) -> float:
            """Natural frequency (Hz) = (1/2π) × √(K/M).

            K in N/m, M in kg → ω in rad/s → / 2π = Hz.
            """
            if math.isnan(wheel_rate) or sprung_per_corner <= 0:
                return float("nan")
            K = wheel_rate * 1000.0   # N/m
            return (1.0 / (2.0 * math.pi)) * math.sqrt(K / sprung_per_corner)

        # Compute masses per corner
        sprung_total = float("nan")
        sprung_front_per_corner = float("nan")
        sprung_rear_per_corner  = float("nan")
        if total_mass > 0 and unsprung_per_corner > 0:
            unsprung_total = 4.0 * unsprung_per_corner
            sprung_total   = total_mass - unsprung_total
            if sprung_total > 0:
                wd = weight_dist_f / 100.0
                sprung_front_per_corner = sprung_total * wd / 2.0
                sprung_rear_per_corner  = sprung_total * (1.0 - wd) / 2.0

        # Apply to each axle's data
        front_data["wheel_rate"] = _wheel_rate(spring_rate_f, mr_front)
        rear_data["wheel_rate"]  = _wheel_rate(spring_rate_r, mr_rear)
        front_data["roll_rate"]  = _roll_rate(front_data["wheel_rate"],
                                                vehicle.track_front_mm)
        rear_data["roll_rate"]   = _roll_rate(rear_data["wheel_rate"],
                                                vehicle.track_rear_mm)
        front_data["nat_freq"]   = _natural_freq(front_data["wheel_rate"],
                                                   sprung_front_per_corner)
        rear_data["nat_freq"]    = _natural_freq(rear_data["wheel_rate"],
                                                   sprung_rear_per_corner)
        front_data["motion_ratio"] = mr_front if mr_front > 0 else float("nan")
        rear_data["motion_ratio"]  = mr_rear  if mr_rear  > 0 else float("nan")
        front_data["jounce_pct"]   = jounce_pct_f if jounce_pct_f > 0 else float("nan")
        front_data["rebound_pct"]  = rebound_pct_f if rebound_pct_f > 0 else float("nan")
        rear_data["jounce_pct"]    = jounce_pct_r if jounce_pct_r > 0 else float("nan")
        rear_data["rebound_pct"]   = rebound_pct_r if rebound_pct_r > 0 else float("nan")
        front_data["travel"]       = susp_travel_f if susp_travel_f > 0 else float("nan")
        rear_data["travel"]        = susp_travel_r if susp_travel_r > 0 else float("nan")

        # ─── Formatting helper ───────────────────────────────────────────────
        def fmt(v, fmt_str="+.3f") -> str:
            """Format a number; returns '—' if NaN or missing input."""
            if v is None:
                return "—"
            try:
                if math.isnan(float(v)):
                    return "—"
                return format(float(v), fmt_str)
            except (TypeError, ValueError):
                return str(v)

        def fmt_pair(v_l, v_r, fmt_str="+.3f") -> str:
            """Format 'L / R' for parameters that differ per wheel."""
            return f"{fmt(v_l, fmt_str)} / {fmt(v_r, fmt_str)}"

        # ─── QUICK SUMMARY (cards with the most consulted KPIs) ──────────────
        st.markdown("---")
        st.markdown("### ⚡ Quick summary")

        st.caption("**Front**")
        fm = st.columns(5)
        fm[0].metric("Camber L/R (°)",
                     fmt_pair(front_data["camber_l"], front_data["camber_r"], "+.2f"),
                     border=True)
        fm[1].metric("Caster L/R (°)",
                     fmt_pair(front_data["caster_l"], front_data["caster_r"], "+.2f"),
                     border=True)
        fm[2].metric("Static RC (mm)",
                     fmt(front_data["rc_static"], "+.1f"), border=True)
        fm[3].metric("Σ Toe (°)",
                     fmt(front_data["sum_toe"], "+.3f"), border=True)
        fm[4].metric("Ackermann (%)",
                     fmt(front_data["ackermann"], "+.1f"), border=True)

        st.caption("**Rear**")
        rm = st.columns(5)
        rm[0].metric("Camber L/R (°)",
                     fmt_pair(rear_data["camber_l"], rear_data["camber_r"], "+.2f"),
                     border=True)
        rm[1].metric("KPI L/R (°)",
                     fmt_pair(rear_data["kpi_l"], rear_data["kpi_r"], "+.2f"),
                     border=True)
        rm[2].metric("Static RC (mm)",
                     fmt(rear_data["rc_static"], "+.1f"), border=True)
        rm[3].metric("Σ Toe (°)",
                     fmt(rear_data["sum_toe"], "+.3f"), border=True)
        rm[4].metric("Anti-squat (%)",
                     fmt(rear_data["anti_squat"], "+.1f"), border=True)

        # ─── MAIN TABLE ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📋 Complete Setup Sheet")

        # Build the table row by row
        # Each row: [Category, Parameter, Unit, Front, Rear, Source]
        # Source: 📐 calculated / ⌨️ input / 🧮 derived

        rows: list[dict[str, str]] = []
        category = "General"  # reassigned before each add() group

        def add(param, unit, f_val, r_val, origin):
            rows.append({
                "Category":  category,
                "Parameter": param,
                "Unit":      unit,
                "Front":     str(f_val),
                "Rear":      str(r_val),
                "Source":    origin,
            })

        # Tire/wheel inputs (same for both axles)
        category = "🛞 Tires & Wheels"
        add("Tire size, compound, make",          "",
            tire_size or "—",  tire_size or "—",  "⌨️ input")
        add("Wheel (diameter × width)",           "inch",
            f"{wheel_diam:.1f} × {wheel_wid:.1f}" if wheel_diam else "—",
            f"{wheel_diam:.1f} × {wheel_wid:.1f}" if wheel_diam else "—",
            "⌨️ input")
        add("Wheel material / construction",      "",
            wheel_mat or "—", wheel_mat or "—", "⌨️ input")
        category = "🔩 Suspension & Rates"
        add("Suspension type",                    "",
            susp_type or "—", susp_type or "—", "⌨️ input")
        add("Suspension design travel",           "mm",
            fmt(front_data["travel"], ".1f"),
            fmt(rear_data["travel"],  ".1f"),
            "⌨️ input")

        # Derived (need inputs)
        add("Wheel rate (chassis → wheel center)", "N/mm",
            fmt(front_data["wheel_rate"], ".2f"),
            fmt(rear_data["wheel_rate"],  ".2f"),
            "🧮 derived from spring + MR")
        add("Roll rate (chassis → wheel center)",  "Nm/deg",
            fmt(front_data["roll_rate"], ".1f"),
            fmt(rear_data["roll_rate"],  ".1f"),
            "🧮 derived from wheel rate + track")
        add("Sprung mass natural frequency",       "Hz",
            fmt(front_data["nat_freq"], ".2f"),
            fmt(rear_data["nat_freq"],  ".2f"),
            "🧮 derived from wheel rate + mass")
        add("Jounce damping",                       "% critical",
            fmt(front_data["jounce_pct"],  ".0f"),
            fmt(rear_data["jounce_pct"],   ".0f"),
            "⌨️ input")
        add("Rebound damping",                      "% critical",
            fmt(front_data["rebound_pct"], ".0f"),
            fmt(rear_data["rebound_pct"],  ".0f"),
            "⌨️ input")
        add("Motion ratio",                         "x:1",
            fmt(front_data["motion_ratio"], ".3f"),
            fmt(rear_data["motion_ratio"],  ".3f"),
            "⌨️ input")

        # Calculated (geometry)
        category = "🎢 Kinematics"
        add("Ride Camber (rate of change)",        "deg/m",
            fmt(front_data["ride_camber_dpm"], "+.2f"),
            fmt(rear_data["ride_camber_dpm"],  "+.2f"),
            "📐 calculated")
        add("Roll Camber",                          "deg/deg",
            fmt(front_data["roll_camber"], "+.4f"),
            fmt(rear_data["roll_camber"],  "+.4f"),
            "📐 calculated")
        category = "📐 Static alignment"
        add("Static Sum Toe (− out, + in)",         "deg",
            fmt(front_data["sum_toe"], "+.4f"),
            fmt(rear_data["sum_toe"],  "+.4f"),
            "📐 calculated")
        add("Static camber (L / R)",                "deg",
            fmt_pair(front_data["camber_l"], front_data["camber_r"], "+.3f"),
            fmt_pair(rear_data["camber_l"],  rear_data["camber_r"],  "+.3f"),
            "📐 calculated")
        add("Static camber adjustment method",      "",
            susp_adj or "—", susp_adj or "—", "⌨️ input")
        category = "🎢 Kinematics"
        add("Anti dive / Anti squat",               "%",
            fmt(front_data["anti_dive"],  "+.2f"),
            fmt(rear_data["anti_squat"], "+.2f"),
            "📐 calculated (needs CG, brake bias)")
        add("Roll center height above ground, static", "mm",
            fmt(front_data["rc_static"], "+.2f"),
            fmt(rear_data["rc_static"],  "+.2f"),
            "📐 calculated")
        add("Roll center @ 1g lateral acc — height",   "mm",
            fmt(front_data["rc_1g_z"], "+.2f"),
            fmt(rear_data["rc_1g_z"],  "+.2f"),
            f"📐 calculated (roll stiffness {roll_stiff}°/g)")
        add("Roll center @ 1g lateral acc — lateral",  "mm",
            fmt(front_data["rc_1g_y"], "+.2f"),
            fmt(rear_data["rc_1g_y"],  "+.2f"),
            f"📐 calculated (roll stiffness {roll_stiff}°/g)")
        category = "📐 Static alignment"
        add("Caster (L / R)",                         "deg",
            fmt_pair(front_data["caster_l"], front_data["caster_r"], "+.3f"),
            "N/A (no relevant rear caster)",
            "📐 calculated")
        add("Kingpin trail (L / R)",                  "mm",
            fmt_pair(front_data["trail_l"], front_data["trail_r"], "+.2f"),
            fmt_pair(rear_data["trail_l"],  rear_data["trail_r"],  "+.2f"),
            "📐 calculated")
        add("Scrub radius (L / R)",                   "mm",
            fmt_pair(front_data["scrub_l"], front_data["scrub_r"], "+.2f"),
            fmt_pair(rear_data["scrub_l"],  rear_data["scrub_r"],  "+.2f"),
            "📐 calculated")
        add("Kingpin Inclination (L / R)",             "deg",
            fmt_pair(front_data["kpi_l"], front_data["kpi_r"], "+.3f"),
            fmt_pair(rear_data["kpi_l"],  rear_data["kpi_r"],  "+.3f"),
            "📐 calculated")
        category = "🕹️ Steering"
        add("Static Ackermann",                        "%",
            fmt(front_data["ackermann"], "+.2f"),
            "N/A",
            "📐 calculated")
        add("Adjustable Ackermann?",                   "",
            ackermann_adj, "—",
            "⌨️ input")
        category = "🔩 Suspension & Rates"
        add("Suspension adjustment methods",           "",
            arb_adj or "—", arb_adj or "—",
            "⌨️ input")
        category = "🕹️ Steering"
        add("Steer Ratio",                             "x:1",
            fmt(front_data["steer_ratio"], ".2f"),
            "N/A",
            f"🧮 derived from c-factor={vs['c_factor_mm']:.0f} mm/rev")
        add("C-factor",                                "mm/rev",
            fmt(front_data["c_factor"], ".1f"),
            "N/A",
            "⌨️ input (sidebar)")
        add("Steer Arm Length (L / R)",                "mm",
            fmt_pair(front_data["steer_arm_l"], front_data["steer_arm_r"], ".2f"),
            "N/A",
            "📐 calculated")

        # Masses / distribution
        category = "⚖️ Masses"
        if total_mass > 0:
            add("Total mass w/ driver",                "kg",
                f"{total_mass:.1f}", f"{total_mass:.1f}", "⌨️ input")
            if not math.isnan(sprung_total):
                add("Total sprung mass",               "kg",
                    f"{sprung_total:.1f}", f"{sprung_total:.1f}", "🧮 derived")
                add("Sprung mass per corner",          "kg",
                    fmt(sprung_front_per_corner, ".1f"),
                    fmt(sprung_rear_per_corner,  ".1f"),
                    "🧮 derived")
            add("Unsprung mass per corner",            "kg",
                f"{unsprung_per_corner:.1f}", f"{unsprung_per_corner:.1f}",
                "⌨️ input")
            add("Weight distribution",                 "%",
                f"{weight_dist_f:.1f}", f"{100-weight_dist_f:.1f}",
                "⌨️ input")

        # Category filter + rendering
        categories = list(dict.fromkeys(r["Category"] for r in rows))
        cat_sel = st.pills("Filter by category", ["All"] + categories,
                           default="All", key="sheet_category")
        if cat_sel and cat_sel != "All":
            rows_view = [r for r in rows if r["Category"] == cat_sel]
        else:
            rows_view = rows

        table_df = pl.DataFrame(rows_view)
        st.dataframe(table_df, width="stretch", hide_index=True,
                     height=min(80 + 35 * len(rows_view), 900))

        # Legend
        st.caption(
            "**Legend:** "
            "📐 calculated from the hardpoints · "
            "⌨️ user input · "
            "🧮 derived (needs inputs in the expanders above)"
        )

        # Download the table (always complete, regardless of the filter)
        csv_table = pl.DataFrame(rows).write_csv().encode()
        st.download_button(
            "⬇️ Download setup sheet (CSV)",
            data=csv_table,
            file_name="setup_sheet.csv",
            mime="text/csv",
        )

        # ─── SWEEPS WITH CHARTS (kept, optional) ─────────────────────────────
        st.markdown("---")
        st.markdown("### 📈 Detailed sweeps (optional)")

        with st.expander("Show sweep charts for a specific corner", expanded=False):
            col_a, col_b = st.columns([1, 3])
            with col_a:
                corner_choice = st.selectbox("Corner", VALID_CORNERS, key="analysis_corner")
                sweep_type = st.radio("Sweep", ["Heave", "Roll", "Steer"],
                                       horizontal=False, key="analysis_sweep_type")

            with col_b:
                sc1, sc2, sc3 = st.columns(3)
                if sweep_type == "Heave":
                    with sc1: h_min  = st.number_input("Min (mm)",  value=-25.0, key="hmin")
                    with sc2: h_max  = st.number_input("Max (mm)",  value= 25.0, key="hmax")
                    with sc3: h_step = st.number_input("Step (mm)", value= 1.0,  key="hstep")
                    sweep_params = (h_min, h_max, h_step)
                elif sweep_type == "Roll":
                    with sc1: r_min  = st.number_input("Min (°)",  value=-3.0, key="rmin")
                    with sc2: r_max  = st.number_input("Max (°)",  value= 3.0, key="rmax")
                    with sc3: r_step = st.number_input("Step (°)", value= 0.2, key="rstep")
                    sweep_params = (r_min, r_max, r_step)
                else:
                    with sc1: s_min  = st.number_input("Min (mm)",  value=-30.0, key="smin")
                    with sc2: s_max  = st.number_input("Max (mm)",  value= 30.0, key="smax")
                    with sc3: s_step = st.number_input("Step (mm)", value= 1.0,  key="sstep")
                    sweep_params = (s_min, s_max, s_step)

            built = build_corner_safe(df, corner_choice)
            if built is not None:
                corner, tie_rod = built
                with st.spinner(f"{sweep_type} sweep..."):
                    sweep = run_sweep_cached(corner, tie_rod, sweep_type, sweep_params)

                if sweep_type == "Heave":
                    pc1, pc2 = st.columns(2)
                    with pc1:
                        st.plotly_chart(plot_camber_vs_heave(sweep), width="stretch")
                    with pc2:
                        st.plotly_chart(plot_bump_steer(sweep), width="stretch")
                    st.plotly_chart(plot_rc_migration(sweep), width="stretch")
                elif sweep_type == "Steer":
                    st.plotly_chart(plot_caster_kpi_vs_steer(sweep), width="stretch")
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=sweep["roll_deg"], y=sweep["camber_deg"],
                                               mode="lines+markers"))
                    fig.update_layout(title="Camber vs Roll", xaxis_title="Roll (°)",
                                       yaxis_title="Camber (°)", template="plotly_white")
                    st.plotly_chart(fig, width="stretch")

                with st.expander("📋 Sweep data"):
                    sweep_df = pl.DataFrame({n: sweep[n] for n in sweep.dtype.names})
                    st.dataframe(sweep_df, width="stretch")
