"""
ui/tab_vdcore.py
================
The 📊 **Analysis** tab — setup sheet, sweeps, and the Altair cross-check.

Every geometry row runs the loaded hardpoints through
``vdcore.geometry.solver.DWSolver``, which constrains all six DOF with the real
linkage, is covered by the test suite, and agrees with Altair MotionSolve to
~1e-7 mm.

HISTORY — this absorbed the old ``tab_analysis.py`` (deleted 2026-08-27).
    The app used to carry two tabs showing the same KPIs from two solvers. The
    Analysis side was wrong on six of them: static camber read 0.000 instead of
    -1.500 (the legacy model cannot infer camber from hardpoints, and that error
    cascaded into the contact patch, scrub and RC), right-side scrub radius and
    mechanical trail had inverted signs, and every sweep chart ran on the
    strut-to-midpoint solver — which produced camber gain with the SIGN
    INVERTED (+0.0388 vs the true -0.0384 deg/mm) and a roll-centre swing of
    70 mm against a true 19.6 mm.

    Anti-dive/anti-squat and Ackermann were also wrong at the source (+200 % and
    +173 %); both are fixed and now appear here with real values.

The legacy-vs-vdcore delta table is kept deliberately: it is the evidence for
why the old solver was retired, and it is quotable in Design Event.

Everything comes from ``analysis/vdcore_bridge.py``; this module is presentation
only (Streamlit + plotly). Plotly is allowed here — this is ``legacy_app/``, an
application layer, not the pure ``vdcore/`` library.
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from analysis.altair_bridge import (
    AltairKPIs,
    AltairRunError,
    AltairUnavailableError,
    run_altair,
)
from analysis.altair_bridge import (
    availability as altair_availability,
)
from analysis.altair_bridge import (
    geometry_signature as altair_signature,
)
from analysis.altair_bridge import (
    load_cached as load_altair_cached,
)
from analysis.io_hardpoints import VALID_CORNERS, build_vehicle_from_dataframe
from analysis.kpis import ackermann_geometry, steer_ratio_from_pinion
from analysis.sweeps import (
    SweepRunner,
    camber_gain_per_mm,
    plot_bump_steer,
    plot_camber_vs_heave,
    plot_caster_kpi_vs_steer,
    plot_rc_migration,
    rc_migration_range,
)
from analysis.vdcore_bridge import (
    AxleVdcoreKPIs,
    BridgeConversionError,
    CornerInputs,
    VdcoreKPIs,
    compute_vdcore_kpis_cached,
    compute_vdcore_setup_sheet_cached,
    df_to_vdcore_axles,
    rack_mm_per_wheel_deg,
    solved_ackermann_pct,
    vdcore_roll_sweep,
    vdcore_sweep,
)
from geometry import KinematicSolver3D

from ui.shared import load_hardpoints_from_state, render_empty_state

# Source label used for every geometry row the validated solver produces.
_VDCORE_SRC = "📐 vdcore (validated)"

# Shown when the Altair column has no number for a row. Distinguishes "Altair
# cannot produce this" from "Altair produced zero", which matters on rows like
# rear mechanical trail where zero is the correct answer.
_NO_ALTAIR = "—"


def _fmt(value: float | None, digits: int = 4, unit: str = "") -> str:
    """Format a metric, showing an em dash for None / NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    suffix = f" {unit}" if unit else ""
    return f"{value:.{digits}f}{suffix}"


def _legacy_dynamic_kpis(df: pl.DataFrame) -> dict[str, dict[str, float]]:
    """Recompute the legacy dynamic KPIs for the delta table.

    Mirrors the heave-sweep block of the deleted ``tab_analysis`` so the
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


def _render_altair_controls(
    df: pl.DataFrame,
    inputs: CornerInputs,
    *,
    roll_deg: float,
    travel_mm: float,
    sweep_steps: int = 41,
) -> AltairKPIs | None:
    """Status, run button and cache handling for the Altair cross-check column.

    Returns the KPIs when a run matching the CURRENT geometry and inputs is
    cached, otherwise None. A run made against different hardpoints is never
    returned: a second opinion on a different geometry is worse than none.
    """
    usable, reason = altair_availability()
    signature = altair_signature(
        df,
        static_camber_deg=inputs.static_camber_deg,
        loaded_radius_mm=inputs.loaded_radius_mm,
        static_toe_deg_per_side=inputs.static_toe_deg_per_side,
        roll_deg=roll_deg,
        travel_mm=travel_mm,
        sweep_steps=sweep_steps,
    )
    cached = load_altair_cached(signature) if usable else None

    with st.expander(
        "🅰️ Altair MotionSolve cross-check"
        + ("" if cached is None else " — ✅ current"),
        expanded=cached is None and usable,
    ):
        st.caption(
            "An independent second opinion on the same hardpoints. MotionSolve "
            "assembles revolute/spherical/universal joints into an index-3 DAE "
            "and integrates it with DASPK; `vdcore` writes nine distance "
            "residuals and drives them to zero with `least_squares`. The two "
            "share **nothing but the geometry**, so agreement is real evidence. "
            "Both columns are then reduced to KPIs by *vdcore's own* formulas, "
            "so a difference is always a kinematics difference, never a "
            "definition difference."
        )

        if not usable:
            st.info(f"ℹ️ {reason}")
            return None

        if cached is not None:
            st.success(
                f"Showing a MotionSolve run for exactly this geometry and these "
                f"inputs (took {cached.elapsed_s:.0f} s). Worst roll-travel "
                f"patch residual **{cached.roll_patch_residual_mm:.1e} mm**."
            )
        else:
            st.warning(
                "No MotionSolve run for the current geometry. The Altair column "
                "is hidden rather than showing numbers from a different "
                "geometry."
            )

        st.caption(
            "A full four-corner pass runs MotionSolve about a dozen times and "
            "takes ~150 s. The result is cached against the geometry **and** "
            "the design inputs, so it reappears instantly until one of them "
            "changes."
        )
        if st.button(
            "▶️ Run MotionSolve cross-check"
            + (" again" if cached is not None else ""),
            key="vd_altair_run",
            type="primary" if cached is None else "secondary",
        ):
            with st.spinner("Running Altair MotionSolve on all four corners…"):
                try:
                    run_altair(
                        df,
                        static_camber_deg=inputs.static_camber_deg,
                        loaded_radius_mm=inputs.loaded_radius_mm,
                        static_toe_deg_per_side=inputs.static_toe_deg_per_side,
                        roll_deg=roll_deg,
                        travel_mm=travel_mm,
                        sweep_steps=sweep_steps,
                    )
                except AltairUnavailableError as exc:
                    st.error(f"❌ {exc}")
                    return None
                except AltairRunError as exc:
                    st.error(f"❌ MotionSolve failed: {exc}")
                    return None
            st.rerun()

    return cached


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


def _render_delta_table(
    vd: VdcoreKPIs,
    legacy: dict[str, dict[str, float]],
    altair: AltairKPIs | None = None,
) -> None:
    """Side-by-side legacy-vs-vdcore table for the two overlapping KPIs.

    When a current MotionSolve run exists, an Altair column and a
    vdcore-minus-Altair delta are appended — the independent check on the two
    KPIs the legacy solver got most wrong.
    """
    st.markdown("#### Legacy vs vdcore — the correction, made visible")

    def rows_for(axle_name: str, vd_axle: AxleVdcoreKPIs) -> list[dict]:
        leg = legacy.get(axle_name, {})
        rates = vd_axle.rates
        vd_cg = rates.camber_gain_deg_per_mm if rates else float("nan")
        vd_rc = (rates.rc_max_mm - rates.rc_min_mm) if rates else float("nan")

        spec = (
            ("Camber gain (°/mm)", "camber_gain_deg_per_mm", "camber_gain", vd_cg, 4),
            ("RC migration Z range (mm)", "rc_migration_z_mm", "rc_dz", vd_rc, 2),
        )
        out: list[dict] = []
        for label, legacy_key, altair_key, vd_value, digits in spec:
            row = {
                "Axle": vd_axle.label,
                "KPI": label,
                "Legacy": _fmt(leg.get(legacy_key), digits),
                "vdcore": _fmt(vd_value, digits),
            }
            if altair is not None:
                alt = altair.get(axle_name, altair_key)
                row["Altair"] = _fmt(alt, digits)
                row["vdcore − Altair"] = (
                    _NO_ALTAIR if alt is None or math.isnan(vd_value)
                    else f"{vd_value - alt:.2e}"
                )
            out.append(row)
        return out

    rows = rows_for("front", vd.front) + rows_for("rear", vd.rear)
    st.dataframe(pl.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "The legacy strut-to-midpoint solver barely rotates the arms under "
        "travel, so its RC migration collapses toward ~1 mm and its camber gain "
        "reads low. vdcore constrains the real linkage."
        + (
            "  **Altair** is Altair MotionSolve on the same hardpoints — an "
            "independent DAE solver, not a re-run of vdcore."
            if altair is not None else ""
        )
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


def _render_detailed_sweeps(df: pl.DataFrame, inputs: CornerInputs) -> None:
    """Per-corner heave / steer sweeps and the axle roll sweep, on DWSolver.

    These charts used to run on ``KinematicSolver3D``, whose camber gain came
    out with the SIGN INVERTED (+0.0388 vs the true -0.0384 deg/mm) and whose
    roll-centre swing read 70 mm against a true 19.6 mm. They now run the real
    linkage.

    Roll is handled at AXLE level rather than per corner. Solving one corner at
    fixed travel under roll is degenerate -- ``DWSolver``'s travel constraint is
    chassis-referenced, so the wheel just rolls with the car and camber tracks
    roll at exactly -1.000 deg/deg. ``vdcore_roll_sweep`` instead solves the
    travel that keeps both patches on the tilted road, which is the real corner
    case and matches the setup sheet's roll-camber figure.
    """
    st.markdown("### 📈 Detailed sweeps")

    with st.expander("Show sweep charts", expanded=False):
        sweep_type = st.radio(
            "Sweep", ["Heave", "Steer", "Roll (axle)"],
            horizontal=True, key="vd_sweep_type",
        )

        if sweep_type == "Roll (axle)":
            axle_name = st.selectbox("Axle", ["Front", "Rear"], key="vd_sweep_axle")
            c1, c2, c3 = st.columns(3)
            with c1:
                lo = st.number_input("Min (°)", value=-3.0, key="vd_rmin")
            with c2:
                hi = st.number_input("Max (°)", value=3.0, key="vd_rmax")
            with c3:
                step = st.number_input("Step (°)", value=0.25, min_value=0.01,
                                       key="vd_rstep")
            try:
                front_axle, rear_axle = df_to_vdcore_axles(df, inputs)
            except BridgeConversionError as exc:
                st.error(f"❌ {exc}")
                return
            axle = front_axle if axle_name == "Front" else rear_axle
            with st.spinner("Roll sweep…"):
                roll = vdcore_roll_sweep(axle, float(lo), float(hi), float(step))
            if not roll.roll_deg:
                st.warning(
                    "No roll angle in this range could be solved with both "
                    "contact patches on the road."
                )
                return

            cc1, cc2 = st.columns(2)
            with cc1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=roll.roll_deg, y=roll.outer_camber_deg,
                                         mode="lines", name="Outer wheel"))
                fig.add_trace(go.Scatter(x=roll.roll_deg, y=roll.inner_camber_deg,
                                         mode="lines", name="Inner wheel"))
                fig.update_layout(
                    title=f"{axle_name} camber vs chassis roll (road-relative)",
                    xaxis_title="Chassis roll (°)", yaxis_title="Camber (°)",
                    height=360, margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, width="stretch")
            with cc2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=roll.roll_deg, y=roll.rc_height_mm,
                                         mode="lines", name="RC height"))
                fig.add_trace(go.Scatter(x=roll.roll_deg, y=roll.rc_lateral_mm,
                                         mode="lines", name="RC lateral"))
                fig.update_layout(
                    title=f"{axle_name} roll centre vs chassis roll (ground-ref)",
                    xaxis_title="Chassis roll (°)", yaxis_title="Position (mm)",
                    height=360, margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, width="stretch")

            st.caption(
                "Both wheels are held on the tilted road, so the wheel travel "
                "per side is solved, not assumed — the outer/inner split and "
                "the sideways roll-centre migration both come out of that."
            )
            with st.expander("📋 Sweep data"):
                st.dataframe(pl.DataFrame({
                    "roll_deg": roll.roll_deg,
                    "wheel_travel_mm": roll.wheel_travel_mm,
                    "outer_camber_deg": roll.outer_camber_deg,
                    "inner_camber_deg": roll.inner_camber_deg,
                    "rc_height_mm": roll.rc_height_mm,
                    "rc_lateral_mm": roll.rc_lateral_mm,
                }), width="stretch", hide_index=True)
            return

        # ---- per-corner heave / steer ------------------------------------- #
        corner_choice = st.selectbox("Corner", list(VALID_CORNERS),
                                     key="vd_sweep_corner")
        c1, c2, c3 = st.columns(3)
        if sweep_type == "Heave":
            with c1:
                lo = st.number_input("Min (mm)", value=-25.0, key="vd_hmin")
            with c2:
                hi = st.number_input("Max (mm)", value=25.0, key="vd_hmax")
            with c3:
                step = st.number_input("Step (mm)", value=1.0, min_value=0.05,
                                       key="vd_hstep")
        else:
            with c1:
                lo = st.number_input("Min (mm)", value=-30.0, key="vd_smin")
            with c2:
                hi = st.number_input("Max (mm)", value=30.0, key="vd_smax")
            with c3:
                step = st.number_input("Step (mm)", value=1.0, min_value=0.05,
                                       key="vd_sstep")

        try:
            corner = _corner_from_df(df, corner_choice, inputs)
        except BridgeConversionError as exc:
            st.error(f"❌ {exc}")
            return

        with st.spinner(f"{sweep_type} sweep…"):
            sweep = vdcore_sweep(corner, sweep_type, (float(lo), float(hi), float(step)))

        if not bool(sweep["converged"].all()):
            st.warning(
                f"⚠️ {int((~sweep['converged']).sum())} of {len(sweep)} points did "
                "not converge and are shown as gaps, never as interpolated values."
            )

        if sweep_type == "Heave":
            pc1, pc2 = st.columns(2)
            with pc1:
                st.plotly_chart(plot_camber_vs_heave(sweep), width="stretch")
            with pc2:
                st.plotly_chart(plot_bump_steer(sweep), width="stretch")
            st.plotly_chart(plot_rc_migration(sweep), width="stretch")
            st.caption(
                "Roll-centre height here is **chassis-referenced**, the same "
                "frame as the rate table. Ground-referenced values differ by "
                "exactly 1 mm per mm of travel. Lateral RC migration is zero by "
                "construction for a single corner — the axle-level figure is on "
                "the roll sweep."
            )
        else:
            st.plotly_chart(plot_caster_kpi_vs_steer(sweep), width="stretch")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sweep["rack_mm"], y=sweep["toe_deg"],
                                     mode="lines", name="Toe"))
            fig.update_layout(
                title=f"{corner_choice} toe vs rack travel",
                xaxis_title="Rack travel (mm)", yaxis_title="Toe per side (°)",
                height=360, margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, width="stretch")

        with st.expander("📋 Sweep data"):
            st.dataframe(
                pl.DataFrame({n: sweep[n] for n in sweep.dtype.names}),
                width="stretch",
            )


def _corner_from_df(
    df: pl.DataFrame, corner_id: str, inputs: CornerInputs
):
    """One vdcore Corner from the loaded geometry."""
    from analysis.vdcore_bridge import df_to_vdcore_corner

    return df_to_vdcore_corner(df, corner_id, inputs)


def _wheel_rate(spring_rate: float, mr: float) -> float:
    """Wheel rate (N/mm) = spring_rate × MR². NaN if inputs missing."""
    if spring_rate <= 0 or mr <= 0:
        return float("nan")
    return spring_rate * mr * mr


def _roll_rate(wheel_rate: float, track_mm: float) -> float:
    """Roll rate per wheel (Nm/°) from wheel rate and track. NaN if missing.

    K_roll = (1/2) × K_wheel × T² × (π/180), K_wheel in N/m, T in m.
    Kept from the deleted ``tab_analysis`` so exported sheets stay comparable.
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


def _fmt_pair(v_l: float | None, v_r: float | None, digits: int = 3) -> str:
    """Format 'L / R' for a per-wheel quantity, em dash for None/NaN."""
    return f"{_fmt(v_l, digits)} / {_fmt(v_r, digits)}"


def _render_setup_sheet(
    df: pl.DataFrame,
    inputs: CornerInputs,
    *,
    roll_deg: float,
    travel_mm: float,
    altair: AltairKPIs | None = None,
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
            arb_adj = st.text_input(
                "Suspension adjustment methods (other)", value="",
                placeholder="e.g. ARB with 3 positions, variable preload",
                key="vd_sheet_susp_methods",
            )
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
            cg_height_mm = st.number_input(
                "CG height (mm)", 0.0, value=320.0, step=5.0, key="vd_sheet_cgh",
                help="Sets the anti-dive / anti-squat scale: %anti is "
                     "proportional to wheelbase / CG height.",
            )
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
    vehicle, tie_rods = build_vehicle_from_dataframe(df)

    # Anti-geometry, Ackermann and the steering-arm rows. These used to live
    # only on the old Analysis tab; both anti-dive and Ackermann were wrong
    # there (+200 % and +173 %) and are now fixed at the source. Ackermann comes
    # from an actual rack sweep on DWSolver, not the plan-view construction --
    # see analysis.vdcore_bridge.solved_ackermann_pct for why that construction
    # is unusable on a car with 10 deg KPI.
    vs = st.session_state.get("vehicle_setup", {})
    brake_bias = float(vs.get("brake_bias", 0.6))
    c_factor_mm = float(vs.get("c_factor_mm", 0.0))
    try:
        anti_dive = vehicle.front_left.anti_dive_percent(
            brake_bias=brake_bias, wheelbase_mm=vehicle.wheelbase_mm,
            cg_height_mm=cg_height_mm,
        )
        anti_squat = vehicle.rear_left.anti_dive_percent(
            brake_bias=1.0, wheelbase_mm=vehicle.wheelbase_mm,
            cg_height_mm=cg_height_mm,
        )
    except Exception:
        anti_dive = anti_squat = float("nan")

    try:
        front_axle, _ = df_to_vdcore_axles(df, inputs)
        ackermann_pct = solved_ackermann_pct(front_axle, vehicle.wheelbase_mm)
    except Exception:
        ackermann_pct = float("nan")

    try:
        ack_info = ackermann_geometry(
            vehicle.front_left, tie_rods["FL"],
            vehicle.front_right, tie_rods["FR"], vehicle.rear_left,
        )
        steer_arm_l = ack_info["steer_arm_length_left"]
        steer_arm_r = ack_info["steer_arm_length_right"]
        # Steer ratio on DWSolver, not the legacy solver: the latter reads
        # 10.2 % high because it mis-places the outboard ball joints and so
        # swings the wrong steering-arm length.
        rack_per_deg = rack_mm_per_wheel_deg(front_axle.left)
        steer_ratio = (
            steer_ratio_from_pinion(rack_per_deg, c_factor_mm)
            if c_factor_mm > 0 else float("nan")
        )
    except Exception:
        steer_arm_l = steer_arm_r = steer_ratio = float("nan")

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

    # ─── Build the table (category order and labels from the old Analysis tab) ─
    rows: list[dict[str, str]] = []
    category = "General"
    show_altair = altair is not None

    def add(param: str, unit: str, f_val: str, r_val: str, origin: str,
            *, f_alt: str = _NO_ALTAIR, r_alt: str = _NO_ALTAIR) -> None:
        """One sheet row. Altair columns are omitted entirely when no run exists.

        Keeping them out (rather than filling a column with em dashes) means the
        sheet looks exactly as it did before the cross-check was added until
        there is something real to show.
        """
        row = {"Category": category, "Parameter": param, "Unit": unit,
               "Front": f_val}
        if show_altair:
            row["Front (Altair)"] = f_alt
        row["Rear"] = r_val
        if show_altair:
            row["Rear (Altair)"] = r_alt
        row["Source"] = origin
        rows.append(row)

    def alt(axle_key: str, sheet_key: str, digits: int = 4) -> str:
        """Altair's value for one setup-sheet key, or an em dash."""
        if altair is None:
            return _NO_ALTAIR
        return _fmt(altair.get(axle_key, sheet_key), digits)

    def alt_pair(axle_key: str, key_l: str, key_r: str, digits: int = 3) -> str:
        """Altair's 'L / R' pair for one setup-sheet key, or an em dash."""
        if altair is None:
            return _NO_ALTAIR
        return _fmt_pair(
            altair.get(axle_key, key_l), altair.get(axle_key, key_r), digits
        )

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
    add("Suspension adjustment methods", "",
        arb_adj or "—", arb_adj or "—", "⌨️ input")

    category = "🎢 Kinematics"
    add("Ride Camber (rate of change)", "deg/m",
        _fmt(front.get("ride_camber_dpm"), 2), _fmt(rear.get("ride_camber_dpm"), 2),
        _VDCORE_SRC,
        f_alt=alt("front", "ride_camber_dpm", 2),
        r_alt=alt("rear", "ride_camber_dpm", 2))
    add("Roll Camber", "deg/deg",
        _fmt(front.get("roll_camber"), 4), _fmt(rear.get("roll_camber"), 4),
        _VDCORE_SRC,
        f_alt=alt("front", "roll_camber", 4), r_alt=alt("rear", "roll_camber", 4))
    add("Anti dive / Anti squat", "%",
        _fmt(anti_dive, 2), _fmt(anti_squat, 2),
        "📐 pivot-axis rake (cross-checked vs 3D linkage)")
    add("Roll center height above ground, static", "mm",
        _fmt(front.get("rc_static"), 2), _fmt(rear.get("rc_static"), 2), _VDCORE_SRC,
        f_alt=alt("front", "rc_static", 2), r_alt=alt("rear", "rc_static", 2))
    add("Roll center @ roll — height", "mm",
        _fmt(front.get("rc_1g_z"), 2), _fmt(rear.get("rc_1g_z"), 2),
        f"{_VDCORE_SRC} @ {roll_deg:.2f}° roll",
        f_alt=alt("front", "rc_1g_z", 2), r_alt=alt("rear", "rc_1g_z", 2))
    add("Roll center @ roll — lateral", "mm",
        _fmt(front.get("rc_1g_y"), 2), _fmt(rear.get("rc_1g_y"), 2),
        f"{_VDCORE_SRC} @ {roll_deg:.2f}° roll",
        f_alt=alt("front", "rc_1g_y", 2), r_alt=alt("rear", "rc_1g_y", 2))

    category = "📐 Static alignment"
    add("Kingpin Inclination (L / R)", "deg",
        _fmt_pair(front.get("kpi_l"), front.get("kpi_r")),
        _fmt_pair(rear.get("kpi_l"), rear.get("kpi_r")), _VDCORE_SRC,
        f_alt=alt_pair("front", "kpi_l", "kpi_r"),
        r_alt=alt_pair("rear", "kpi_l", "kpi_r"))
    add("Caster (L / R)", "deg",
        _fmt_pair(front.get("caster_l"), front.get("caster_r")),
        _fmt_pair(rear.get("caster_l"), rear.get("caster_r")), _VDCORE_SRC,
        f_alt=alt_pair("front", "caster_l", "caster_r"),
        r_alt=alt_pair("rear", "caster_l", "caster_r"))
    add("Scrub radius (L / R)", "mm",
        _fmt_pair(front.get("scrub_l"), front.get("scrub_r"), 2),
        _fmt_pair(rear.get("scrub_l"), rear.get("scrub_r"), 2), _VDCORE_SRC,
        f_alt=alt_pair("front", "scrub_l", "scrub_r", 2),
        r_alt=alt_pair("rear", "scrub_l", "scrub_r", 2))
    add("Mechanical trail (L / R)", "mm",
        _fmt_pair(front.get("trail_l"), front.get("trail_r"), 2),
        _fmt_pair(rear.get("trail_l"), rear.get("trail_r"), 2), _VDCORE_SRC,
        f_alt=alt_pair("front", "trail_l", "trail_r", 2),
        r_alt=alt_pair("rear", "trail_l", "trail_r", 2))
    add("Static Sum Toe (− out, + in)", "deg",
        _fmt(front.get("sum_toe"), 4), _fmt(rear.get("sum_toe"), 4), _VDCORE_SRC,
        f_alt=alt("front", "sum_toe", 4), r_alt=alt("rear", "sum_toe", 4))
    add("Static camber (L / R)", "deg",
        _fmt_pair(front.get("camber_l"), front.get("camber_r")),
        _fmt_pair(rear.get("camber_l"), rear.get("camber_r")), _VDCORE_SRC,
        f_alt=alt_pair("front", "camber_l", "camber_r"),
        r_alt=alt_pair("rear", "camber_l", "camber_r"))
    add("Static camber adjustment method", "", susp_adj or "—", susp_adj or "—",
        "⌨️ input")

    category = "🕹️ Steering"
    add("Static Ackermann", "%", _fmt(ackermann_pct, 2), "N/A",
        f"{_VDCORE_SRC} — rack sweep @ 10° outer steer")
    add("Adjustable Ackermann?", "", ackermann_adj, "—", "⌨️ input")
    add("Steer Ratio", "x:1", _fmt(steer_ratio, 2), "N/A",
        f"🧮 derived from c-factor = {c_factor_mm:.0f} mm/rev")
    add("C-factor", "mm/rev", _fmt(c_factor_mm if c_factor_mm > 0 else None, 1),
        "N/A", "⌨️ input (sidebar)")
    add("Steer Arm Length (L / R)", "mm",
        _fmt_pair(steer_arm_l, steer_arm_r, 2), "N/A", "📐 calculated")

    category = "⚖️ Masses"
    if total_mass > 0:
        add("Total mass w/ driver", "kg", f"{total_mass:.1f}", f"{total_mass:.1f}",
            "⌨️ input")
        if not math.isnan(sprung_f_pc):
            add("Total sprung mass", "kg",
                f"{total_mass - 4.0 * unsprung:.1f}",
                f"{total_mass - 4.0 * unsprung:.1f}", "🧮 derived")
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
        + (
            "  ·  **(Altair)** — the same row from Altair MotionSolve on the "
            "same hardpoints. Rows that are not geometry have no Altair value; "
            "so do anti-dive/anti-squat and Ackermann, which MotionSolve is "
            "not being asked for here."
            if show_altair else ""
        )
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

    altair = _render_altair_controls(
        df, inputs, roll_deg=float(roll_deg), travel_mm=float(travel_mm),
    )
    st.divider()

    _render_axle_cards(vd.front)
    st.divider()
    _render_axle_cards(vd.rear)
    st.divider()

    legacy = _legacy_dynamic_kpis(df)
    _render_delta_table(vd, legacy, altair)
    st.divider()

    c_left, c_right = st.columns(2)
    with c_left:
        _render_camber_plot(vd)
    with c_right:
        _render_rc_migration_plot(vd)

    st.divider()
    _render_detailed_sweeps(df, inputs)

    st.divider()
    _render_setup_sheet(
        df, inputs, roll_deg=float(roll_deg), travel_mm=float(travel_mm),
        altair=altair,
    )

    st.info(
        "**What this tab does and does not claim.** Every geometry row is "
        "solved on the real linkage by `vdcore`'s `DWSolver`, which agrees with "
        "Altair MotionSolve to ~1e-7 mm and whose sign conventions are pinned "
        "by known-answer tests. Anti-dive/anti-squat come from the pivot-axis "
        "rake, cross-checked against the same 3D linkage; Ackermann comes from "
        "an actual rack sweep, **not** the plan-view construction — that "
        "construction assumes a vertical kingpin and swings ~150 points across "
        "defensible reference heights on a car with 10° KPI.\n\n"
        "Not derivable from loaded hardpoints: wheel rate, motion ratio, "
        "frequencies and damping (they need the pushrod/spring package), and "
        "anything requiring tyre data. Those rows are inputs or blank."
    )
