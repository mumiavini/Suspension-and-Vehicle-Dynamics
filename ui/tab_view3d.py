"""
ui/tab_view3d.py
================
🌐 Aba Vista 3D — visualização interativa da suspensão: veículo completo,
corner individual ou animação de sweep (heave/roll/steer).
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
    st.header("Visualização 3D dos hardpoints")
    st.markdown(
        "Veja a suspensão em **3D interativo**: rotacione, dê zoom, e veja como "
        "os hardpoints se relacionam no espaço. Use o modo animado para ver o "
        "movimento durante heave, roll ou steer."
    )

    df = load_hardpoints_from_state()
    if df is None:
        render_empty_state(
            "A vista 3D mostra a suspensão completa de forma interativa: "
            "rotação, zoom e animação de heave/roll/steer.",
            key="empty_3d",
        )
    else:
        from analysis.viz3d import (plot_corner_3d, plot_vehicle_3d,
                                     plot_corner_animated)

        # ── Controles principais ─────────────────────────────────────────────
        view_mode = st.segmented_control(
            "Modo de visualização",
            ["🏎️ Veículo completo", "🔍 Corner individual",
             "🎬 Animação de sweep"],
            default="🏎️ Veículo completo",
            key="view3d_mode",
        )
        if view_mode is None:  # clicar no item selecionado desseleciona
            view_mode = "🏎️ Veículo completo"

        st.markdown("---")

        # ─── MODO 1: VEÍCULO COMPLETO ────────────────────────────────────────
        if view_mode == "🏎️ Veículo completo":
            try:
                vehicle, tie_rods = build_vehicle_from_dataframe(df)

                opt1, opt2, _ = st.columns([1, 1.4, 1.6])
                with opt1:
                    show_tires = st.toggle("Mostrar pneus", value=True,
                                           key="veh_show_tires")
                with opt2:
                    show_chassis = st.toggle("Mostrar wireframe do chassi", value=True,
                                             key="veh_show_chassis")

                with st.spinner("Renderizando..."):
                    fig = plot_vehicle_3d(
                        vehicle, tie_rods,
                        show_tires=show_tires,
                        show_chassis_box=show_chassis,
                        title="Suspensão FSAE — Vista 3D completa",
                    )
                st.plotly_chart(fig, width="stretch")

                st.caption(
                    "💡 **Dica:** clique e arraste para rotacionar, scroll para "
                    "dar zoom, duplo-clique para resetar a câmera."
                )
            except HardpointValidationError as exc:
                st.error(f"❌ {exc}")

        # ─── MODO 2: CORNER INDIVIDUAL ───────────────────────────────────────
        elif view_mode == "🔍 Corner individual":
            col_a, col_b = st.columns([1, 3])
            with col_a:
                corner_choice = st.selectbox("Corner", VALID_CORNERS,
                                              key="view3d_corner")
                show_tire = st.checkbox("Mostrar pneu", value=True,
                                         key="corner_show_tire")

            built = build_corner_safe(df, corner_choice)
            if built is not None:
                corner, tie_rod = built
                with st.spinner("Renderizando..."):
                    fig = plot_corner_3d(corner, tie_rod, show_tire=show_tire)
                st.plotly_chart(fig, width="stretch")

                # KPIs ao lado da visualização para contexto
                with st.expander("📊 KPIs deste corner"):
                    k = st.columns(3)
                    k[0].metric("Caster (°)",    f"{corner.static_caster_deg():+.3f}")
                    k[1].metric("KPI (°)",       f"{corner.static_kpi_deg():+.3f}")
                    k[2].metric("Scrub (mm)",    f"{corner.static_scrub_radius_mm():+.2f}")

        # ─── MODO 3: ANIMAÇÃO DE SWEEP ───────────────────────────────────────
        else:  # 🎬 Animação de sweep
            ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
            with ctrl1:
                corner_choice = st.selectbox("Corner", VALID_CORNERS,
                                              key="anim_corner")
            with ctrl2:
                sweep_axis = st.radio("Eixo do sweep",
                                      ["heave", "roll", "steer"],
                                      key="anim_axis")
            with ctrl3:
                if sweep_axis == "heave":
                    rng = st.slider("Faixa heave (mm)", -50.0, 50.0,
                                     (-20.0, 20.0), step=2.5, key="anim_h_range")
                elif sweep_axis == "roll":
                    rng = st.slider("Faixa roll (°)", -5.0, 5.0,
                                     (-3.0, 3.0), step=0.5, key="anim_r_range")
                else:
                    rng = st.slider("Faixa rack (mm)", -50.0, 50.0,
                                     (-25.0, 25.0), step=2.5, key="anim_s_range")
                n_frames = st.slider("Número de frames", 5, 30, 15,
                                      key="anim_n_frames")

            built = build_corner_safe(df, corner_choice)
            if built is not None:
                corner, tie_rod = built
                with st.spinner(f"Calculando {n_frames} frames..."):
                    fig = plot_corner_animated(
                        corner, tie_rod,
                        sweep_axis=sweep_axis,
                        sweep_min=rng[0], sweep_max=rng[1],
                        n_frames=n_frames,
                        show_tire=True,
                    )
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    "💡 **Dica:** arraste o slider para ver a geometria em cada "
                    "posição, ou clique em ▶ Play para animar automaticamente."
                )
