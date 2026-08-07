"""
ui/tab_view3d.py
================
🌐 View 3D tab — interactive visualization of the suspension: complete vehicle,
single corner or sweep animation (heave/roll/steer).
"""

from __future__ import annotations

import streamlit as st

from analysis.io_hardpoints import (
    build_vehicle_from_dataframe,
    HardpointValidationError,
    VALID_CORNERS,
)
from ui.shared import (
    load_hardpoints_from_state,
    render_empty_state,
    build_corner_safe,
)


def render() -> None:
    st.header("3D visualization of the hardpoints")
    st.markdown(
        "See the suspension in **interactive 3D**: rotate, zoom, and watch how "
        "the hardpoints relate to each other in space. Use the animated mode to "
        "see the motion during heave, roll or steer."
    )

    df = load_hardpoints_from_state()
    if df is None:
        render_empty_state(
            "The 3D view shows the complete suspension interactively: "
            "rotation, zoom and heave/roll/steer animation.",
            key="empty_3d",
        )
    else:
        from analysis.viz3d import (plot_corner_3d, plot_vehicle_3d,
                                     plot_corner_animated)

        # ── Main controls ────────────────────────────────────────────────────
        view_mode = st.segmented_control(
            "Visualization mode",
            ["🏎️ Complete vehicle", "🔍 Single corner",
             "🎬 Sweep animation"],
            default="🏎️ Complete vehicle",
            key="view3d_mode",
        )
        if view_mode is None:  # clicking the selected item deselects it
            view_mode = "🏎️ Complete vehicle"

        st.markdown("---")

        # ─── MODE 1: COMPLETE VEHICLE ────────────────────────────────────────
        if view_mode == "🏎️ Complete vehicle":
            try:
                vehicle, tie_rods = build_vehicle_from_dataframe(df)

                opt1, opt2, _ = st.columns([1, 1.4, 1.6])
                with opt1:
                    show_tires = st.toggle("Show tires", value=True,
                                           key="veh_show_tires")
                with opt2:
                    show_chassis = st.toggle("Show chassis wireframe", value=True,
                                             key="veh_show_chassis")

                with st.spinner("Rendering..."):
                    fig = plot_vehicle_3d(
                        vehicle, tie_rods,
                        show_tires=show_tires,
                        show_chassis_box=show_chassis,
                        title="FSAE suspension — Complete 3D view",
                    )
                st.plotly_chart(fig, width="stretch")

                st.caption(
                    "💡 **Tip:** click and drag to rotate, scroll to "
                    "zoom, double-click to reset the camera."
                )
            except HardpointValidationError as exc:
                st.error(f"❌ {exc}")

        # ─── MODE 2: SINGLE CORNER ───────────────────────────────────────────
        elif view_mode == "🔍 Single corner":
            col_a, col_b = st.columns([1, 3])
            with col_a:
                corner_choice = st.selectbox("Corner", VALID_CORNERS,
                                              key="view3d_corner")
                show_tire = st.checkbox("Show tire", value=True,
                                         key="corner_show_tire")

            built = build_corner_safe(df, corner_choice)
            if built is not None:
                corner, tie_rod = built
                with st.spinner("Rendering..."):
                    fig = plot_corner_3d(corner, tie_rod, show_tire=show_tire)
                st.plotly_chart(fig, width="stretch")

                # KPIs next to the visualization for context
                with st.expander("📊 KPIs for this corner"):
                    k = st.columns(3)
                    k[0].metric("Caster (°)",    f"{corner.static_caster_deg():+.3f}")
                    k[1].metric("KPI (°)",       f"{corner.static_kpi_deg():+.3f}")
                    k[2].metric("Scrub (mm)",    f"{corner.static_scrub_radius_mm():+.2f}")

        # ─── MODE 3: SWEEP ANIMATION ─────────────────────────────────────────
        else:  # 🎬 Sweep animation
            ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
            with ctrl1:
                corner_choice = st.selectbox("Corner", VALID_CORNERS,
                                              key="anim_corner")
            with ctrl2:
                sweep_axis = st.radio("Sweep axis",
                                      ["heave", "roll", "steer"],
                                      key="anim_axis")
            with ctrl3:
                if sweep_axis == "heave":
                    rng = st.slider("Heave range (mm)", -50.0, 50.0,
                                     (-20.0, 20.0), step=2.5, key="anim_h_range")
                elif sweep_axis == "roll":
                    rng = st.slider("Roll range (°)", -5.0, 5.0,
                                     (-3.0, 3.0), step=0.5, key="anim_r_range")
                else:
                    rng = st.slider("Rack range (mm)", -50.0, 50.0,
                                     (-25.0, 25.0), step=2.5, key="anim_s_range")
                n_frames = st.slider("Number of frames", 5, 30, 15,
                                      key="anim_n_frames")

            built = build_corner_safe(df, corner_choice)
            if built is not None:
                corner, tie_rod = built
                with st.spinner(f"Computing {n_frames} frames..."):
                    fig = plot_corner_animated(
                        corner, tie_rod,
                        sweep_axis=sweep_axis,
                        sweep_min=rng[0], sweep_max=rng[1],
                        n_frames=n_frames,
                        show_tire=True,
                    )
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    "💡 **Tip:** drag the slider to see the geometry at each "
                    "position, or click ▶ Play to animate automatically."
                )
