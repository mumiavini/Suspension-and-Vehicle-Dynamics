"""
app.py
======
Aplicativo Streamlit para uso do motor FSAE Suspension Geometry.

Como rodar
----------
    pip install streamlit polars plotly scipy numpy openpyxl xlsx2csv fastexcel
    streamlit run app.py

Fluxo
-----
  1. Usuário faz upload de arquivo .xlsx/.csv/.json com hardpoints.
  2. App valida e exibe os hardpoints em tabela.
  3. Usuário escolhe corner, configura range de sweep.
  4. App executa sweep 3D, calcula KPIs e renderiza gráficos Plotly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st
import numpy as np

from analysis.io_hardpoints import (
    read_hardpoints,
    build_corner_from_dataframe,
    build_vehicle_from_dataframe,
    generate_template_dataframe,
    HardpointValidationError,
    VALID_CORNERS,
)
from geometry.solver_3d import KinematicSolver3D
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


# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FSAE Suspension Geometry",
    layout="wide",
    page_icon="🏎️",
)

st.title("🏎️ FSAE Suspension Geometry Engine")
st.caption("Motor de análise cinemática 3D para suspensão double-wishbone")


# ---------------------------------------------------------------------------
# Sidebar — Upload e seleção
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📂 Hardpoints")

    uploaded = st.file_uploader(
        "Faça upload do arquivo de hardpoints",
        type=["xlsx", "csv", "json"],
        help="Estrutura: corner, point, x_mm, y_mm, z_mm",
    )

    use_template = st.button("📋 Usar template demo")

    st.markdown("---")
    st.subheader("⚙️ Sweep config")

    sweep_type = st.radio(
        "Tipo de varredura",
        options=["Heave", "Roll", "Steer"],
        index=0,
    )

    if sweep_type == "Heave":
        h_min = st.number_input("Heave min (mm)", value=-25.0, step=1.0)
        h_max = st.number_input("Heave max (mm)", value= 25.0, step=1.0)
        h_step = st.number_input("Step (mm)", value=1.0, step=0.5)
    elif sweep_type == "Roll":
        r_min = st.number_input("Roll min (°)", value=-3.0, step=0.5)
        r_max = st.number_input("Roll max (°)", value= 3.0, step=0.5)
        r_step = st.number_input("Step (°)", value=0.2, step=0.1)
    else:
        s_min = st.number_input("Rack min (mm)", value=-30.0, step=1.0)
        s_max = st.number_input("Rack max (mm)", value= 30.0, step=1.0)
        s_step = st.number_input("Step (mm)", value=1.0, step=0.5)

    st.markdown("---")
    corner_choice = st.selectbox("Corner", VALID_CORNERS, index=0)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

df = None

if use_template:
    df = generate_template_dataframe()
    st.success("Template demo carregado.")
elif uploaded is not None:
    # Salva temporariamente o upload e lê
    suffix = Path(uploaded.name).suffix
    tmp_path = Path("/tmp") / f"_upload{suffix}"
    tmp_path.write_bytes(uploaded.read())
    try:
        df = read_hardpoints(tmp_path)
        st.success(f"Arquivo '{uploaded.name}' validado com sucesso.")
    except HardpointValidationError as exc:
        st.error(f"❌ Validação falhou: {exc}")
        st.stop()


# ---------------------------------------------------------------------------
# Visualização dos hardpoints
# ---------------------------------------------------------------------------

if df is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Hardpoints carregados")
        st.dataframe(df, width='content', height=300)

    # ---------------------------------------------------------------------
    # Constrói o corner selecionado e executa sweep
    # ---------------------------------------------------------------------
    try:
        corner, tie_rod = build_corner_from_dataframe(df, corner_choice)
    except HardpointValidationError as exc:
        st.error(f"Erro ao construir corner: {exc}")
        st.stop()

    solver = KinematicSolver3D(corner, tie_rod)
    runner = SweepRunner(solver=solver)

    with col2:
        st.subheader("Parâmetros estáticos")
        s0 = runner.static_state
        st.metric("Camber estático (°)", f"{s0.camber_deg:+.3f}")
        st.metric("Caster (°)",          f"{s0.caster_deg:+.3f}")
        st.metric("KPI (°)",             f"{s0.kpi_deg:+.3f}")
        st.metric("Comprimento UCA (mm)", f"{corner.upper_arm.arm_length():.1f}")
        st.metric("Comprimento LCA (mm)", f"{corner.lower_arm.arm_length():.1f}")
        st.metric("Tie-rod length (mm)",  f"{tie_rod.length:.1f}")

    # ---------------------------------------------------------------------
    # Executa o sweep selecionado
    # ---------------------------------------------------------------------
    st.markdown("---")
    st.subheader(f"📊 {sweep_type} Sweep")

    with st.spinner(f"Executando {sweep_type.lower()} sweep..."):
        if sweep_type == "Heave":
            sweep = runner.heave_sweep(h_min, h_max, h_step)
        elif sweep_type == "Roll":
            sweep = runner.roll_sweep(r_min, r_max, r_step)
        else:
            sweep = runner.steer_sweep(s_min, s_max, s_step)

    # KPIs derivados
    kpi_cols = st.columns(4)
    if sweep_type == "Heave":
        kpi_cols[0].metric("Camber gain (°/mm)", f"{camber_gain_per_mm(sweep):+.5f}")
        kpi_cols[1].metric("Bump steer (°/mm)",  f"{bump_steer_per_mm(sweep):+.5f}")
        dy, dz = rc_migration_range(sweep)
        kpi_cols[2].metric("RC ΔY (mm)", f"{dy:.2f}")
        kpi_cols[3].metric("RC ΔZ (mm)", f"{dz:.2f}")

    # Plots Plotly
    st.markdown("### Gráficos")
    plot_cols = st.columns(2)

    if sweep_type == "Heave":
        with plot_cols[0]:
            st.plotly_chart(plot_camber_vs_heave(sweep), width='content')
        with plot_cols[1]:
            st.plotly_chart(plot_bump_steer(sweep), width='content')
        st.plotly_chart(plot_rc_migration(sweep), width='content')

    elif sweep_type == "Steer":
        st.plotly_chart(plot_caster_kpi_vs_steer(sweep), width='content')

    else:  # Roll
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sweep["roll_deg"], y=sweep["camber_deg"],
            mode="lines+markers", name="Camber",
        ))
        fig.update_layout(
            title="Camber vs Roll",
            xaxis_title="Roll (°)",
            yaxis_title="Camber (°)",
            template="plotly_white",
        )
        st.plotly_chart(fig, width='content')

    # Tabela completa do sweep
    with st.expander("📋 Dados completos do sweep"):
        import polars as pl
        sweep_df = pl.DataFrame({name: sweep[name] for name in sweep.dtype.names})
        st.dataframe(sweep_df, width='content')

else:
    st.info(
        "👋 Faça upload de um arquivo de hardpoints na barra lateral ou "
        "clique em **'Usar template demo'** para começar."
    )
    with st.expander("📖 Formato do arquivo"):
        st.markdown("""
        O arquivo deve conter as colunas:
        - **corner** : `FL`, `FR`, `RL`, `RR`
        - **point**  : `UCA_IN_FRONT`, `UCA_IN_REAR`, `UCA_OUT`,
                        `LCA_IN_FRONT`, `LCA_IN_REAR`, `LCA_OUT`,
                        `TIE_ROD_IN`, `TIE_ROD_OUT`,
                        `WHEEL_CENTER`, `CONTACT_PATCH`
        - **x_mm**, **y_mm**, **z_mm** : coordenadas em mm (convenção SAE)

        Cada corner deve ter exatamente 10 pontos.
        """)