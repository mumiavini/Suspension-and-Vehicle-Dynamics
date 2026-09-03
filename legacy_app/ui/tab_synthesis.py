"""
ui/tab_synthesis.py
===================
🎯 Synthesis tab — seed-corner KPI snapshot. Geometry synthesis (automated
hardpoint optimization from targets) has been retired: `analysis/optimizer.py`
scored candidates with the legacy strut-to-pivot-midpoint solver
(`KinematicSolver3D`), which does not close the real linkage — see
CLAUDE.md. There is no drop-in replacement yet: a `vdcore`/`DWSolver`-based
optimizer needs its cost function reworked to use far fewer solves, since
`DWSolver` is far slower per sweep than the legacy solver was.
"""

from __future__ import annotations

from analysis.io_hardpoints import VALID_CORNERS
from analysis.sweeps import camber_gain_per_mm, bump_steer_per_mm
from analysis.vdcore_bridge import (
    BridgeConversionError,
    CornerInputs,
    df_to_vdcore_corner,
    vdcore_sweep,
)
from ui.shared import (
    load_hardpoints_from_state,
    render_empty_state,
    build_corner_safe,
)

import streamlit as st


@st.fragment
def render() -> None:
    """
    Synthesis tab isolated in an st.fragment: interacting with any widget here
    re-runs ONLY this tab, not the whole app (in particular, it does not
    recompute the Analysis tab) — much faster interaction.
    """
    st.header("Geometry synthesis — Reverse engineering")

    df = load_hardpoints_from_state()
    if df is None:
        render_empty_state(
            "Synthesis starts from an existing geometry (the **seed**) and "
            "shows its KPIs as a reference for a manual redesign.",
            key="empty_synthesis",
        )
        return

    # ── Seed corner + snapshot of the current KPIs ───────────────────────────
    sel_col, kpi_col = st.columns([1, 4])
    with sel_col:
        seed_corner_id = st.selectbox("Seed corner", VALID_CORNERS,
                                       key="synth_seed_corner")
    built = build_corner_safe(df, seed_corner_id)
    if built is None:
        return
    seed_corner, seed_tie_rod = built

    with kpi_col:
        st.caption("**Current seed values**")
        # Dynamic seed KPIs run on DWSolver, not the legacy strut-to-midpoint
        # solver. Static values below (caster/KPI/camber/RC) are unaffected:
        # the legacy solver's STATIC numbers are correct, only its sweeps
        # are not.
        seed_sweep = None
        try:
            vd_seed = df_to_vdcore_corner(
                df, seed_corner_id,
                CornerInputs.from_vehicle_setup(
                    st.session_state.get("vehicle_setup", {})
                ),
            )
            seed_sweep = vdcore_sweep(vd_seed, "Heave", (-25.0, 25.0, 5.0))
        except (BridgeConversionError, ValueError, KeyError) as exc:
            st.warning(
                f"⚠️ Dynamic seed KPIs unavailable ({exc}). Static values below "
                "are still valid."
            )

        def _dyn(fn) -> str:
            """Format a dynamic KPI, or an em dash if the solve was unavailable."""
            return "—" if seed_sweep is None else f"{fn(seed_sweep):+.4f} °/mm"

        m = st.columns(6)
        m[0].metric("Caster",      f"{seed_corner.static_caster_deg():+.2f}°",
                    border=True)
        m[1].metric("KPI",         f"{seed_corner.static_kpi_deg():+.2f}°",
                    border=True)
        m[2].metric("Camber",      f"{seed_corner.static_camber_deg():+.2f}°",
                    border=True)
        m[3].metric("Camber gain", _dyn(camber_gain_per_mm), border=True)
        m[4].metric("Bump steer",  _dyn(bump_steer_per_mm), border=True)
        m[5].metric("RC height",   f"{seed_corner.roll_center_height_mm():+.1f} mm",
                    border=True)

    st.markdown("---")
    st.info(
        "🚧 **Automated synthesis is retired.** The former optimizer "
        "(`differential_evolution` over hardpoint bounds) scored candidates "
        "with the legacy strut-to-pivot-midpoint solver, which reported "
        "camber gain and roll-centre migration with the wrong sign and "
        "magnitude — see CLAUDE.md. Use the seed KPIs above as a reference "
        "and iterate on hardpoints manually, checking results on the "
        "Analysis tab (vdcore/DWSolver-based)."
    )
