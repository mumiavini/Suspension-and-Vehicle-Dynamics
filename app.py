"""
app.py
======
Streamlit app — Interface gráfica completa do motor FSAE Suspension Geometry.

ESTRUTURA EM ABAS:
    ✏️  Inputs       : Cria/edita hardpoints manualmente, com visualização
                       2D em vistas YZ (frontal), XZ (lateral), XY (superior).
    📊 Análise       : Carrega hardpoints, roda sweeps, mostra KPIs e gráficos.
    🎯 Síntese       : Otimização global a partir de targets (engenharia reversa).
    🔄 Comparação    : Compara duas geometrias lado a lado.

COMO RODAR:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import streamlit as st
import plotly.graph_objects as go

from analysis.io_hardpoints import (
    read_hardpoints,
    build_corner_from_dataframe,
    generate_template_dataframe,
    dataframe_from_corner,
    HardpointValidationError,
    VALID_CORNERS,
    REQUIRED_POINTS_PER_CORNER,
)
from geometry import (
    Point3D, ControlArm, SuspensionCorner, TieRod, KinematicSolver3D,
)
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
from analysis.optimizer import (
    SuspensionOptimizer,
    DesignTargets,
    HardpointBounds,
    validate_against_targets,
)


# =============================================================================
# Configuração da página
# =============================================================================

st.set_page_config(
    page_title="FSAE Suspension Geometry",
    layout="wide",
    page_icon="🏎️",
)

st.title("🏎️ FSAE Suspension Geometry Engine")
st.caption("Análise + Síntese + Inputs visuais de geometria de suspensão")


# =============================================================================
# Helpers
# =============================================================================

def _save_uploaded_to_tmp(uploaded_file) -> Path:
    """
    Salva um arquivo upload do Streamlit em diretório temporário e retorna o path.

    Usa `tempfile.gettempdir()` para portabilidade entre SOs:
        - Linux/macOS : /tmp
        - Windows     : C:\\Users\\<user>\\AppData\\Local\\Temp
    """
    import tempfile
    suffix = Path(uploaded_file.name).suffix
    tmp_path = Path(tempfile.gettempdir()) / f"_fsae_upload{suffix}"
    tmp_path.write_bytes(uploaded_file.read())
    return tmp_path


def _load_hardpoints_from_state() -> Optional[pl.DataFrame]:
    return st.session_state.get("hardpoints_df", None)


def _build_corner_safe(df, corner_id):
    try:
        return build_corner_from_dataframe(df, corner_id)
    except HardpointValidationError as exc:
        st.error(f"❌ Erro corner '{corner_id}': {exc}")
        return None


def _build_vehicle_safe(df):
    from analysis.io_hardpoints import build_vehicle_from_dataframe
    try:
        return build_vehicle_from_dataframe(df)
    except HardpointValidationError as exc:
        st.warning(f"⚠️ Veículo incompleto: {exc}")
        return None, None


def _run_sweep_cached(corner, tie_rod, sweep_type, params):
    solver = KinematicSolver3D(corner, tie_rod)
    runner = SweepRunner(solver=solver)
    if sweep_type == "Heave":
        return runner.heave_sweep(*params)
    elif sweep_type == "Roll":
        return runner.roll_sweep(*params)
    else:
        return runner.steer_sweep(*params)


def _format_kpi(value: float, unit: str = "", fmt: str = "+.3f") -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{format(value, fmt)} {unit}".strip()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("📂 Hardpoints")

    # ─── ETAPA 1: UPLOAD (apenas armazena, NÃO aplica ainda) ─────────────────
    uploaded = st.file_uploader(
        "1. Carregue arquivo",
        type=["xlsx", "csv", "json"],
        help="Colunas: corner, point, x_mm, y_mm, z_mm",
    )

    # Faz parse do arquivo só pra preview/validação, mas não aplica ainda
    pending_df: Optional[pl.DataFrame] = None
    pending_error: Optional[str] = None

    if uploaded is not None:
        try:
            tmp = _save_uploaded_to_tmp(uploaded)
            pending_df = read_hardpoints(tmp)
        except HardpointValidationError as exc:
            pending_error = f"Validação: {exc}"
        except Exception as exc:
            pending_error = str(exc)

    # ─── ETAPA 2: PREVIEW + BOTÃO APLICAR ────────────────────────────────────
    if pending_error is not None:
        st.error(f"❌ {pending_error}")
    elif pending_df is not None:
        # Mostra um mini-preview do que foi carregado
        n_rows  = pending_df.height
        corners = sorted(pending_df["corner"].unique().to_list())
        st.success(f"✅ '{uploaded.name}' — {n_rows} pontos · corners: {', '.join(corners)}")

        # Botão explícito que aplica o arquivo (recalcula tudo)
        if st.button("🔄 **Aplicar arquivo**", type="primary", width='content',
                      help="Carrega esse arquivo no app e recalcula todos os KPIs e gráficos"):
            st.session_state["hardpoints_df"]     = pending_df
            st.session_state["hardpoints_source"] = uploaded.name
            st.rerun()   # força re-render imediato com o novo arquivo

    # ─── DEMO + TEMPLATE (atalhos) ───────────────────────────────────────────
    st.markdown("**Ou use:**")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📋 Demo", width='content',
                      help="Carrega geometria FSAE realista de exemplo"):
            st.session_state["hardpoints_df"]     = generate_template_dataframe()
            st.session_state["hardpoints_source"] = "Template demo"
            st.rerun()

    with col_b:
        template_df = generate_template_dataframe()
        st.download_button("⬇️ Template", data=template_df.write_csv().encode(),
                            file_name="hardpoints_template.csv", mime="text/csv",
                            width='content',
                            help="Baixa o template em CSV para edição manual")

    # ─── ESTADO ATUAL ────────────────────────────────────────────────────────
    st.markdown("---")
    if "hardpoints_df" in st.session_state:
        st.info(f"📊 Em uso: **{st.session_state.get('hardpoints_source', '?')}**")
        if st.button("🗑️ Limpar", width='content',
                      help="Remove o arquivo atual da sessão"):
            for key in ["hardpoints_df", "hardpoints_source", "last_optimization",
                        "manual_hardpoints", "manual_synced_source"]:
                st.session_state.pop(key, None)
            # Limpa também as chaves dos data_editors
            for cid in VALID_CORNERS:
                st.session_state.pop(f"editor_{cid}", None)
            st.rerun()
    else:
        st.warning("⚠️ Nenhum hardpoint carregado")

    st.markdown("---")
    st.subheader("⚙️ Setup do veículo")
    st.session_state.setdefault("vehicle_setup", {
        "brake_bias": 0.60,
        "c_factor_mm": 100.0,
        "steering_wheel_lock_deg": 270.0,
    })
    vs = st.session_state["vehicle_setup"]
    vs["brake_bias"] = st.slider("Brake bias front", 0.0, 1.0, vs["brake_bias"],
                                   step=0.05, help="Fração na frente")
    with st.expander("🔧 Direção"):
        vs["c_factor_mm"] = st.number_input("c-factor (mm/rev)",
                                              value=vs["c_factor_mm"], step=1.0,
                                              help="2π × raio do pinhão")
        vs["steering_wheel_lock_deg"] = st.number_input("Lock total volante (°)",
                                                          value=vs["steering_wheel_lock_deg"],
                                                          step=10.0)


# =============================================================================
# ABAS
# =============================================================================

tab_inputs, tab_analysis, tab_synthesis, tab_compare = st.tabs([
    "✏️ Inputs", "📊 Análise", "🎯 Síntese / Otimização", "🔄 Comparação",
])


# ─────────────────────────────────────────────────────────────────────────────
# ABA 1 — INPUTS MANUAIS COM VISUALIZAÇÃO 2D
# ─────────────────────────────────────────────────────────────────────────────

with tab_inputs:
    st.header("Editor de hardpoints com visualização")
    st.markdown(
        "Insira/edite os hardpoints manualmente e veja as projeções em "
        "**YZ (frontal)**, **XZ (lateral)** e **XY (superior)** em tempo real. "
        "Útil para criar geometrias do zero ou inspecionar visualmente."
    )

    # Inicialização / sincronização do estado manual
    #
    # O editor mantém seu próprio dict `manual_hardpoints[corner][point] = (x,y,z)`.
    # Para que o botão "Aplicar arquivo" da sidebar também atualize o editor,
    # rastreamos qual fonte foi a última carregada. Se mudou, recarregamos.
    def _empty_corner_dict():
        return {p: (0.0, 0.0, 0.0) for p in REQUIRED_POINTS_PER_CORNER}

    def _load_manual_from_df(df_loaded: pl.DataFrame) -> dict:
        """Converte o DataFrame de hardpoints no formato dict do editor."""
        result = {}
        for corner_id in VALID_CORNERS:
            sub = df_loaded.filter(pl.col("corner") == corner_id)
            pts = {row["point"]: (float(row["x_mm"]),
                                   float(row["y_mm"]),
                                   float(row["z_mm"]))
                   for row in sub.iter_rows(named=True)}
            for p in REQUIRED_POINTS_PER_CORNER:
                pts.setdefault(p, (0.0, 0.0, 0.0))
            result[corner_id] = pts
        return result

    current_source = st.session_state.get("hardpoints_source", None)
    last_synced    = st.session_state.get("manual_synced_source", None)

    # Re-sincroniza se: (a) ainda não tem estado manual, ou
    # (b) a fonte mudou desde a última sincronização
    needs_sync = ("manual_hardpoints" not in st.session_state) or \
                 (current_source != last_synced)

    if needs_sync:
        df_loaded = _load_hardpoints_from_state()
        if df_loaded is not None:
            st.session_state["manual_hardpoints"] = _load_manual_from_df(df_loaded)
        else:
            st.session_state["manual_hardpoints"] = {
                cid: _empty_corner_dict() for cid in VALID_CORNERS
            }
        st.session_state["manual_synced_source"] = current_source

    # Botão explícito para recarregar do arquivo atual a qualquer momento
    if current_source is not None:
        bcol1, bcol2 = st.columns([1, 3])
        with bcol1:
            if st.button("🔁 Recarregar do arquivo",
                          help=f"Descarta edições manuais e recarrega do arquivo atual ({current_source})",
                          width='content'):
                df_loaded = _load_hardpoints_from_state()
                if df_loaded is not None:
                    st.session_state["manual_hardpoints"] = _load_manual_from_df(df_loaded)
                    # Limpa as chaves dos data_editors para forçar redesenho
                    for cid in VALID_CORNERS:
                        st.session_state.pop(f"editor_{cid}", None)
                    st.rerun()
        with bcol2:
            st.caption(f"📊 Sincronizado com: **{current_source}**")

    # Controles superiores
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1.5, 2])
    with ctrl1:
        edit_corner = st.selectbox("Corner editado", VALID_CORNERS, key="edit_corner")
    with ctrl2:
        if st.button("📋 Carregar template neste corner",
                      width='content'):
            tmpl = generate_template_dataframe()
            sub = tmpl.filter(pl.col("corner") == edit_corner)
            pts = {row["point"]: (float(row["x_mm"]),
                                   float(row["y_mm"]),
                                   float(row["z_mm"]))
                   for row in sub.iter_rows(named=True)}
            st.session_state["manual_hardpoints"][edit_corner] = pts
            st.rerun()
    with ctrl3:
        if st.button("🪞 Espelhar Esquerdo → Direito (Y → −Y)",
                      width='content',
                      help="FL → FR e RL → RR invertendo Y"):
            for left, right in [("FL", "FR"), ("RL", "RR")]:
                left_pts = st.session_state["manual_hardpoints"][left]
                st.session_state["manual_hardpoints"][right] = {
                    p: (x, -y, z) for p, (x, y, z) in left_pts.items()
                }
            st.rerun()

    st.markdown("---")

    # Layout: tabela à esquerda, plots à direita
    col_inputs, col_plots = st.columns([1, 1])

    # COLUNA ESQUERDA: tabela editável
    with col_inputs:
        st.subheader(f"Coordenadas — corner **{edit_corner}**")

        current_pts = st.session_state["manual_hardpoints"][edit_corner]
        edit_data = [
            {"point": p, "x_mm": current_pts.get(p, (0,0,0))[0],
                          "y_mm": current_pts.get(p, (0,0,0))[1],
                          "z_mm": current_pts.get(p, (0,0,0))[2]}
            for p in REQUIRED_POINTS_PER_CORNER
        ]

        edited = st.data_editor(
            edit_data, num_rows="fixed", hide_index=True,
            width='content', disabled=["point"],
            column_config={
                "point": st.column_config.TextColumn("Ponto", width="medium"),
                "x_mm": st.column_config.NumberColumn("X (mm)", format="%.2f"),
                "y_mm": st.column_config.NumberColumn("Y (mm)", format="%.2f"),
                "z_mm": st.column_config.NumberColumn("Z (mm)", format="%.2f"),
            },
            key=f"editor_{edit_corner}",
        )
        for row in edited:
            st.session_state["manual_hardpoints"][edit_corner][row["point"]] = (
                float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"]),
            )

        st.markdown("#### Ações")
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("✅ Aplicar como hardpoints carregados",
                          type="primary", width='content'):
                rows = []
                for cid in VALID_CORNERS:
                    for pn, (x, y, z) in st.session_state["manual_hardpoints"][cid].items():
                        rows.append({"corner": cid, "point": pn,
                                      "x_mm": x, "y_mm": y, "z_mm": z})
                df_built = pl.DataFrame(rows)
                try:
                    from analysis.io_hardpoints import _validate_dataframe
                    _validate_dataframe(df_built)
                    st.session_state["hardpoints_df"] = df_built
                    st.session_state["hardpoints_source"] = "Inputs manuais"
                    st.success("✅ Aplicado! Vá em '📊 Análise'.")
                except HardpointValidationError as exc:
                    st.error(f"❌ {exc}")

        with action_col2:
            rows = []
            for cid in VALID_CORNERS:
                for pn, (x, y, z) in st.session_state["manual_hardpoints"][cid].items():
                    rows.append({"corner": cid, "point": pn,
                                  "x_mm": x, "y_mm": y, "z_mm": z})
            df_out = pl.DataFrame(rows)
            st.download_button(
                "⬇️ Baixar tudo (CSV)",
                data=df_out.write_csv().encode(),
                file_name="hardpoints_manual.csv",
                mime="text/csv", width='content',
            )

    # COLUNA DIREITA: 3 vistas 2D
    with col_plots:
        st.subheader("Vistas 2D")

        POINT_GROUPS = {
            "UCA":     ["UCA_IN_FRONT", "UCA_IN_REAR", "UCA_OUT"],
            "LCA":     ["LCA_IN_FRONT", "LCA_IN_REAR", "LCA_OUT"],
            "Tie-rod": ["TIE_ROD_IN", "TIE_ROD_OUT"],
            "Wheel":   ["WHEEL_CENTER", "CONTACT_PATCH"],
        }
        GROUP_COLORS = {
            "UCA": "#1f77b4", "LCA": "#d62728",
            "Tie-rod": "#2ca02c", "Wheel": "#ff7f0e",
        }
        CONNECTIONS = [
            ("UCA_IN_FRONT", "UCA_OUT", "UCA"),
            ("UCA_IN_REAR",  "UCA_OUT", "UCA"),
            ("UCA_IN_FRONT", "UCA_IN_REAR", "UCA"),
            ("LCA_IN_FRONT", "LCA_OUT", "LCA"),
            ("LCA_IN_REAR",  "LCA_OUT", "LCA"),
            ("LCA_IN_FRONT", "LCA_IN_REAR", "LCA"),
            ("TIE_ROD_IN",   "TIE_ROD_OUT", "Tie-rod"),
            ("UCA_OUT",      "LCA_OUT",     "UCA"),  # manga
            ("WHEEL_CENTER", "CONTACT_PATCH", "Wheel"),
        ]

        def make_2d_view(view: str) -> go.Figure:
            pts = st.session_state["manual_hardpoints"][edit_corner]

            def coords(name):
                x, y, z = pts.get(name, (0.0, 0.0, 0.0))
                if view == "YZ":   return y, z, "Y", "Z"
                elif view == "XZ": return x, z, "X", "Z"
                else:              return x, y, "X", "Y"

            fig = go.Figure()

            # Linhas de conexão
            for p1, p2, group in CONNECTIONS:
                u1, v1, _, _ = coords(p1)
                u2, v2, _, _ = coords(p2)
                color = GROUP_COLORS.get(group, "#888")
                fig.add_trace(go.Scatter(
                    x=[u1, u2], y=[v1, v2], mode="lines",
                    line=dict(color=color, width=2), showlegend=False,
                    hoverinfo="skip",
                ))

            # Pontos com label
            for group, point_names in POINT_GROUPS.items():
                xs, ys, labels = [], [], []
                for pn in point_names:
                    u, v, _, _ = coords(pn)
                    xs.append(u); ys.append(v); labels.append(pn)
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers+text",
                    marker=dict(size=10, color=GROUP_COLORS[group],
                                 line=dict(width=1, color="white")),
                    text=labels, textposition="top center",
                    textfont=dict(size=9),
                    name=group,
                    hovertemplate="<b>%{text}</b><br>%{x:.1f}, %{y:.1f}<extra></extra>",
                ))

            # Linha do chão (YZ e XZ)
            if view in ("YZ", "XZ"):
                fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=1))

            # Plano de simetria (apenas em XY se Y=0 estiver visível)
            if view == "XY":
                fig.add_hline(y=0, line=dict(color="lightgray", dash="dot", width=1))

            _, _, lab_u, lab_v = coords(REQUIRED_POINTS_PER_CORNER[0])
            fig.update_layout(
                title=f"Vista {view} — {edit_corner}",
                xaxis_title=f"{lab_u} (mm)",
                yaxis_title=f"{lab_v} (mm)",
                template="plotly_white", height=320,
                margin=dict(l=40, r=20, t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                             xanchor="right", x=1),
            )
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            return fig

        st.plotly_chart(make_2d_view("YZ"), width='content',
                         key=f"yz_{edit_corner}")
        st.plotly_chart(make_2d_view("XZ"), width='content',
                         key=f"xz_{edit_corner}")
        st.plotly_chart(make_2d_view("XY"), width='content',
                         key=f"xy_{edit_corner}")


# ─────────────────────────────────────────────────────────────────────────────
# ABA 2 — ANÁLISE
# ─────────────────────────────────────────────────────────────────────────────

with tab_analysis:
    st.header("Análise cinemática completa")

    df = _load_hardpoints_from_state()
    if df is None:
        st.info("👈 Carregue ou edite hardpoints primeiro.")
    else:
        vehicle, all_tie_rods = _build_vehicle_safe(df)

        col1, col2 = st.columns([1, 3])
        with col1:
            corner_choice = st.selectbox("Corner", VALID_CORNERS,
                                          key="analysis_corner")

        built = _build_corner_safe(df, corner_choice)
        if built is None:
            st.stop()
        corner, tie_rod = built
        vs = st.session_state["vehicle_setup"]

        with col2:
            sweep_type = st.radio("Sweep", ["Heave", "Roll", "Steer"],
                                    horizontal=True, key="analysis_sweep_type")

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

        # ── KPIs ESTÁTICOS ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📐 KPIs Estáticos")

        st.markdown("**Pino Mestre & Geometria de Roda**")
        kkp = st.columns(6)
        kkp[0].metric("Caster (°)",       _format_kpi(corner.static_caster_deg(),     "", "+.3f"))
        kkp[1].metric("KPI (°)",          _format_kpi(corner.static_kpi_deg(),        "", "+.3f"))
        kkp[2].metric("Kin Trail (mm)",   _format_kpi(corner.static_mechanical_trail_mm(), "", "+.2f"))
        kkp[3].metric("Scrub Rad (mm)",   _format_kpi(corner.static_scrub_radius_mm(),     "", "+.2f"))
        kkp[4].metric("Kingpin Offset (mm)",
                       _format_kpi(corner.static_kingpin_offset_mm(), "", "+.2f"),
                       help="Distância do eixo do pino ao WC, no nível do WC")
        kkp[5].metric("Steer Arm (mm)",
                       _format_kpi(corner.steer_arm_length_mm(tie_rod.outboard), "", "+.2f"),
                       help="Distância perpendicular do TRO ao eixo do pino mestre")

        st.markdown("**Camber & Roll Center**")
        krc = st.columns(4)
        krc[0].metric("Static Camber (°)",     _format_kpi(corner.static_camber_deg(), "", "+.3f"))
        krc[1].metric("RC Height (mm)",        _format_kpi(corner.roll_center_height_mm(), "", "+.2f"))
        if vehicle is not None:
            rc_f, rc_r = vehicle.roll_axis()
            krc[2].metric("RC Front avg (mm)", _format_kpi(rc_f, "", "+.2f"))
            krc[3].metric("Roll Axis incl (°)", _format_kpi(vehicle.roll_axis_inclination_deg(), "", "+.4f"))
        else:
            krc[2].metric("RC Front avg (mm)", "N/A")
            krc[3].metric("Roll Axis incl (°)", "N/A")

        st.markdown("**Anti-features (vista lateral)**")
        kanti = st.columns(2)
        is_front = corner_choice in ("FL", "FR")
        if is_front:
            anti_dive = corner.anti_dive_percent(brake_bias=vs["brake_bias"])
            kanti[0].metric("Anti-dive (%)",  _format_kpi(anti_dive, "", "+.2f"),
                             help=f"Brake bias front = {vs['brake_bias']:.2f}")
            kanti[1].metric("Anti-squat (%)", "N/A (eixo dianteiro)")
        else:
            anti_squat = corner.anti_squat_percent(drive_fraction=1.0)
            kanti[0].metric("Anti-dive (%)", "N/A (eixo traseiro)")
            kanti[1].metric("Anti-squat (%)", _format_kpi(anti_squat, "", "+.2f"),
                             help="Tração 100% no eixo traseiro")

        st.markdown("**Direção (veículo completo)**")
        kstr = st.columns(4)
        if vehicle is not None and all_tie_rods is not None:
            ack = vehicle.static_ackermann_percent(
                all_tie_rods["FL"].outboard, all_tie_rods["FR"].outboard,
            )
            kstr[0].metric("Static Ackermann (%)", _format_kpi(ack, "", "+.2f"))
        else:
            kstr[0].metric("Static Ackermann (%)", "N/A")

        # Steer Ratio aproximado
        steer_ratio_str = "N/A"
        if vs["c_factor_mm"] > 0:
            sa = corner.steer_arm_length_mm(tie_rod.outboard)
            if sa > 1e-6:
                max_rack = vs["steering_wheel_lock_deg"] * vs["c_factor_mm"] / 360.0
                max_road_wheel = np.degrees(max_rack / sa)
                if max_road_wheel > 1e-3:
                    sr = vs["steering_wheel_lock_deg"] / (2 * max_road_wheel)
                    steer_ratio_str = f"{sr:.2f}:1"
        kstr[1].metric("Steer Ratio (≈)",       steer_ratio_str,
                        help="lock_volante / (2·max_road_angle)")
        kstr[2].metric("c-factor (mm/rev)",     f"{vs['c_factor_mm']:.1f}")
        kstr[3].metric("Tie-rod length (mm)",   f"{tie_rod.length:.1f}")

        # ── SWEEP DINÂMICO ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"### 📈 {sweep_type} Sweep")

        with st.spinner(f"Executando {sweep_type.lower()} sweep..."):
            sweep = _run_sweep_cached(corner, tie_rod, sweep_type, sweep_params)

        if sweep_type == "Heave":
            kd = st.columns(5)
            cg = camber_gain_per_mm(sweep)
            kd[0].metric("Ride Camber (°/m)",
                          _format_kpi(cg * 1000.0, "", "+.2f"),
                          help="Camber Gain × 1000 (mm→m)")
            kd[1].metric("Camber Gain (°/mm)", _format_kpi(cg, "", "+.5f"))
            kd[2].metric("Bump Steer (°/mm)",  _format_kpi(bump_steer_per_mm(sweep), "", "+.5f"))
            dy, dz = rc_migration_range(sweep)
            kd[3].metric("RC ΔY (mm)", f"{dy:.2f}")
            kd[4].metric("RC ΔZ (mm)", f"{dz:.2f}")

        elif sweep_type == "Roll":
            kd = st.columns(3)
            roll_data = sweep["roll_deg"]; camber_data = sweep["camber_deg"]
            if len(roll_data) > 1 and (roll_data.max() - roll_data.min()) > 1e-6:
                roll_camber = float(np.polyfit(roll_data, camber_data, 1)[0])
            else:
                roll_camber = float("nan")
            kd[0].metric("Roll Camber (°/°)", _format_kpi(roll_camber, "", "+.4f"))
            kd[1].metric("Camber min (°)",    _format_kpi(camber_data.min(), "", "+.3f"))
            kd[2].metric("Camber max (°)",    _format_kpi(camber_data.max(), "", "+.3f"))

        else:
            kd = st.columns(3)
            kd[0].metric("Toe range (°)",
                          f"{sweep['toe_deg'].min():+.2f} a {sweep['toe_deg'].max():+.2f}")
            kd[1].metric("Caster range (°)",
                          f"{sweep['caster_deg'].min():+.3f} a {sweep['caster_deg'].max():+.3f}")
            kd[2].metric("KPI range (°)",
                          f"{sweep['kpi_deg'].min():+.3f} a {sweep['kpi_deg'].max():+.3f}")

        # Plots
        st.markdown("#### Gráficos")
        if sweep_type == "Heave":
            pc1, pc2 = st.columns(2)
            with pc1:
                st.plotly_chart(plot_camber_vs_heave(sweep), width='content')
            with pc2:
                st.plotly_chart(plot_bump_steer(sweep), width='content')
            st.plotly_chart(plot_rc_migration(sweep), width='content')
        elif sweep_type == "Steer":
            st.plotly_chart(plot_caster_kpi_vs_steer(sweep), width='content')
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sweep["roll_deg"], y=sweep["camber_deg"],
                                       mode="lines+markers", name="Camber"))
            fig.update_layout(title="Camber vs Roll", xaxis_title="Roll (°)",
                               yaxis_title="Camber (°)", template="plotly_white")
            st.plotly_chart(fig, width='content')

        with st.expander("📋 Dados do sweep"):
            sweep_df = pl.DataFrame({n: sweep[n] for n in sweep.dtype.names})
            st.dataframe(sweep_df, width='content')


# ─────────────────────────────────────────────────────────────────────────────
# ABA 3 — SÍNTESE
# ─────────────────────────────────────────────────────────────────────────────

with tab_synthesis:
    st.header("Síntese de geometria — Engenharia reversa")

    df = _load_hardpoints_from_state()
    if df is None:
        st.info("👈 Carregue ou edite hardpoints primeiro (são o SEED).")
    else:
        seed_corner_id = st.selectbox("Corner-seed", VALID_CORNERS,
                                       key="synth_seed_corner")
        built = _build_corner_safe(df, seed_corner_id)
        if built is None:
            st.stop()
        seed_corner, seed_tie_rod = built

        st.markdown("---")
        st.subheader("🎯 Targets")
        col_static, col_dynamic = st.columns(2)

        with col_static:
            st.markdown("**Estáticos**")
            use_caster = st.checkbox("Caster", value=True, key="use_caster")
            tgt_caster = st.number_input("Caster (°)", value=4.0, step=0.5,
                                          disabled=not use_caster, key="tgt_caster")
            use_kpi = st.checkbox("KPI", value=True, key="use_kpi")
            tgt_kpi = st.number_input("KPI (°)", value=7.0, step=0.5,
                                       disabled=not use_kpi, key="tgt_kpi")
            use_camber = st.checkbox("Camber estático", value=True, key="use_camber")
            tgt_camber = st.number_input("Camber (°)", value=-1.5, step=0.25,
                                          disabled=not use_camber, key="tgt_camber")
            use_scrub = st.checkbox("Scrub", value=False, key="use_scrub")
            tgt_scrub = st.number_input("Scrub (mm)", value=15.0, step=1.0,
                                         disabled=not use_scrub, key="tgt_scrub")
            use_trail = st.checkbox("Trail", value=False, key="use_trail")
            tgt_trail = st.number_input("Trail (mm)", value=20.0, step=1.0,
                                         disabled=not use_trail, key="tgt_trail")

        with col_dynamic:
            st.markdown("**Dinâmicos**")
            tgt_cg = st.number_input("Camber Gain (°/mm)", value=-0.020,
                                      step=0.005, format="%.3f", key="tgt_cg")
            tgt_bs = st.number_input("Bump Steer máx (°/mm)", value=0.005,
                                      step=0.001, format="%.3f", key="tgt_bs")
            tgt_rch = st.number_input("RC Height (mm)", value=45.0, key="tgt_rch")
            tgt_rcm = st.number_input("RC ΔY máx (mm)", value=25.0, key="tgt_rcm")
            st.markdown("**Heave sweep range**")
            hc1, hc2, hc3 = st.columns(3)
            with hc1: opt_h_min  = st.number_input("min", value=-25.0, key="opt_hmin")
            with hc2: opt_h_max  = st.number_input("max", value= 25.0, key="opt_hmax")
            with hc3: opt_h_step = st.number_input("step", value= 5.0, key="opt_hstep")

        with st.expander("⚙️ Pesos"):
            wc1, wc2 = st.columns(2)
            with wc1:
                w_caster = st.number_input("w_caster",        value=1.0,  key="w_caster")
                w_kpi    = st.number_input("w_kpi",           value=1.0,  key="w_kpi")
                w_camber = st.number_input("w_static_camber", value=5.0,  key="w_camber")
                w_scrub  = st.number_input("w_scrub",  value=0.01, format="%.3f", key="w_scrub")
                w_trail  = st.number_input("w_trail",  value=0.01, format="%.3f", key="w_trail")
            with wc2:
                w_cg     = st.number_input("w_camber_gain",   value=1.0,  key="w_cg")
                w_bs     = st.number_input("w_bump_steer",    value=10.0, key="w_bs")
                w_rch    = st.number_input("w_rc_height",     value=0.01, format="%.3f", key="w_rch")
                w_rcm    = st.number_input("w_rc_migration",  value=0.05, format="%.3f", key="w_rcm")

        with st.expander("📦 Bounds"):
            bc1, bc2 = st.columns(2)
            with bc1:
                margin_uca = st.slider("UCA out (±mm)", 10, 100, 50, key="m_uca")
                margin_lca = st.slider("LCA out (±mm)", 10, 100, 50, key="m_lca")
            with bc2:
                margin_tri = st.slider("TR in (±mm)",  5, 50, 25, key="m_tri")
                margin_tro = st.slider("TR out (±mm)", 5, 50, 25, key="m_tro")

        with st.expander("🔧 Solver"):
            oc1, oc2 = st.columns(2)
            with oc1:
                pop_size = st.slider("População (×n_dims)", 5, 30, 12, key="pop")
                max_iter = st.slider("Iterações", 10, 200, 40, key="iter")
            with oc2:
                seed_rng = st.number_input("seed", value=42, key="seed_rng")
                workers  = st.selectbox("Cores", [1, -1],
                                          format_func=lambda x: "1" if x == 1 else "Todos",
                                          key="workers")

        st.markdown("---")
        run_opt = st.button("🚀 Rodar Otimização", type="primary")

        if run_opt:
            targets = DesignTargets(
                camber_gain_target_deg_per_mm=tgt_cg,
                bump_steer_max_abs_deg_per_mm=tgt_bs,
                rc_height_target_mm=tgt_rch, rc_y_migration_max_mm=tgt_rcm,
                caster_target_deg          = tgt_caster if use_caster else None,
                kpi_target_deg             = tgt_kpi    if use_kpi    else None,
                static_camber_target_deg   = tgt_camber if use_camber else None,
                scrub_radius_target_mm     = tgt_scrub  if use_scrub  else None,
                mechanical_trail_target_mm = tgt_trail  if use_trail  else None,
                heave_min_mm=opt_h_min, heave_max_mm=opt_h_max, heave_step_mm=opt_h_step,
                w_camber_gain=w_cg, w_bump_steer=w_bs,
                w_rc_height=w_rch, w_rc_migration=w_rcm,
                w_caster=w_caster, w_kpi=w_kpi,
                w_static_camber=w_camber, w_scrub=w_scrub, w_trail=w_trail,
            )

            def box_around(p: Point3D, m: float) -> HardpointBounds:
                return HardpointBounds(p.x-m, p.x+m, p.y-m, p.y+m, p.z-m, p.z+m)

            optimizer = SuspensionOptimizer(
                seed_corner=seed_corner, seed_tie_rod=seed_tie_rod, targets=targets,
                bounds_uca_outboard=box_around(seed_corner.upper_arm.outboard, margin_uca),
                bounds_lca_outboard=box_around(seed_corner.lower_arm.outboard, margin_lca),
                bounds_tie_rod_in  =box_around(seed_tie_rod.inboard,           margin_tri),
                bounds_tie_rod_out =box_around(seed_tie_rod.outboard,          margin_tro),
                population_size=pop_size, max_iterations=max_iter,
                seed=seed_rng, workers=workers,
            )

            seed_validation = validate_against_targets(seed_corner, seed_tie_rod, targets)
            seed_cost = optimizer.objective(optimizer._initial_guess_vector())

            with st.spinner(f"Otimizando ({pop_size}×{max_iter} avals)..."):
                result = optimizer.run()

            opt_validation = validate_against_targets(
                result.optimal_corner, result.optimal_tie_rod, targets,
            )

            st.success(f"✅ {result.scipy_result.nit} gerações. "
                        f"Custo: {seed_cost:.3e} → {result.cost:.3e}")

            st.markdown("### Target × Seed × Otimizado")
            seed_rows = seed_validation.as_dict_list()
            opt_rows = opt_validation.as_dict_list()
            comparison = pl.DataFrame([
                {"Parâmetro": s["name"], "Target": s["target_str"],
                 "Seed": s["obtained_str"], "Otimizado": o["obtained_str"],
                 "Erro Seed": s["error_str"], "Erro Otimizado": o["error_str"],
                 "OK Seed": "✅" if s["ok"] else "❌",
                 "OK Otimizado": "✅" if o["ok"] else "❌"}
                for s, o in zip(seed_rows, opt_rows)
            ])
            st.dataframe(comparison, width='content', hide_index=True)

            st.markdown("### 🎯 Hardpoints otimizados")
            opt_df = dataframe_from_corner(result.optimal_corner, result.optimal_tie_rod)
            st.dataframe(opt_df, width='content', hide_index=True)

            st.download_button(
                "⬇️ Baixar CSV otimizado",
                data=opt_df.write_csv().encode(),
                file_name=f"hardpoints_optimized_{seed_corner_id}.csv",
                mime="text/csv", type="primary",
            )

            st.session_state["last_optimization"] = {
                "seed_corner": seed_corner, "seed_tie_rod": seed_tie_rod,
                "opt_corner": result.optimal_corner, "opt_tie_rod": result.optimal_tie_rod,
                "targets": targets, "corner_id": seed_corner_id,
            }
            st.info("💡 Vá em '🔄 Comparação' para ver os gráficos.")


# ─────────────────────────────────────────────────────────────────────────────
# ABA 4 — COMPARAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

with tab_compare:
    st.header("Comparação entre duas geometrias")

    df = _load_hardpoints_from_state()
    has_optimization = "last_optimization" in st.session_state

    col_src_a, col_src_b = st.columns(2)
    with col_src_a:
        st.markdown("**Geometria A**")
        sa_opts = ["Corner do arquivo", "Última SEED"]
        if has_optimization: sa_opts.append("Última OTIMIZADA")
        source_a = st.radio("A", sa_opts, key="src_a", label_visibility="collapsed")
    with col_src_b:
        st.markdown("**Geometria B**")
        sb_opts = ["Corner do arquivo", "Última SEED"]
        if has_optimization: sb_opts.append("Última OTIMIZADA")
        default_idx = 2 if has_optimization else 1
        source_b = st.radio("B", sb_opts, index=default_idx,
                              key="src_b", label_visibility="collapsed")

    def resolve_geometry(source, side):
        if source == "Corner do arquivo":
            if df is None:
                st.warning(f"⚠️ Carregue arquivo p/ lado {side}.")
                return None
            cid = st.selectbox(f"Corner {side}", VALID_CORNERS, key=f"cmp_{side}")
            return _build_corner_safe(df, cid)
        elif source == "Última SEED":
            if not has_optimization:
                st.warning("⚠️ Rode otimização primeiro.")
                return None
            lo = st.session_state["last_optimization"]
            return lo["seed_corner"], lo["seed_tie_rod"]
        else:
            lo = st.session_state["last_optimization"]
            return lo["opt_corner"], lo["opt_tie_rod"]

    cga, cgb = st.columns(2)
    with cga: geom_a = resolve_geometry(source_a, "A")
    with cgb: geom_b = resolve_geometry(source_b, "B")
    if geom_a is None or geom_b is None:
        st.stop()
    corner_a, tie_rod_a = geom_a
    corner_b, tie_rod_b = geom_b

    st.markdown("---")
    st.markdown("### KPIs estáticos")

    metrics = [
        ("Caster (°)",          corner_a.static_caster_deg(),         corner_b.static_caster_deg()),
        ("KPI (°)",             corner_a.static_kpi_deg(),            corner_b.static_kpi_deg()),
        ("Camber estático (°)", corner_a.static_camber_deg(),         corner_b.static_camber_deg()),
        ("Scrub (mm)",          corner_a.static_scrub_radius_mm(),    corner_b.static_scrub_radius_mm()),
        ("Trail (mm)",          corner_a.static_mechanical_trail_mm(),corner_b.static_mechanical_trail_mm()),
        ("Kingpin Offset (mm)", corner_a.static_kingpin_offset_mm(),  corner_b.static_kingpin_offset_mm()),
        ("Steer Arm (mm)",      corner_a.steer_arm_length_mm(tie_rod_a.outboard),
                                 corner_b.steer_arm_length_mm(tie_rod_b.outboard)),
        ("RC Height (mm)",      corner_a.roll_center_height_mm(),     corner_b.roll_center_height_mm()),
    ]
    static_cmp = pl.DataFrame([
        {"Parâmetro": n, "A": f"{a:+.3f}", "B": f"{b:+.3f}",
         "Δ (B−A)": f"{b-a:+.3f}"} for n, a, b in metrics
    ])
    st.dataframe(static_cmp, width='content', hide_index=True)

    st.markdown("### Heave Sweep — Sobreposição")
    hsc1, hsc2, hsc3 = st.columns(3)
    with hsc1: cmp_h_min  = st.number_input("Min", value=-25.0, key="cmp_hmin")
    with hsc2: cmp_h_max  = st.number_input("Max", value= 25.0, key="cmp_hmax")
    with hsc3: cmp_h_step = st.number_input("Step",value=  1.0, key="cmp_hstep")

    with st.spinner("Rodando sweeps..."):
        sweep_a = _run_sweep_cached(corner_a, tie_rod_a, "Heave",
                                     (cmp_h_min, cmp_h_max, cmp_h_step))
        sweep_b = _run_sweep_cached(corner_b, tie_rod_b, "Heave",
                                     (cmp_h_min, cmp_h_max, cmp_h_step))

    kc = st.columns(4)
    cg_a, cg_b = camber_gain_per_mm(sweep_a), camber_gain_per_mm(sweep_b)
    bs_a, bs_b = bump_steer_per_mm(sweep_a),  bump_steer_per_mm(sweep_b)
    kc[0].metric("CG A", f"{cg_a:+.5f}", delta=f"Δ {cg_b-cg_a:+.5f}")
    kc[1].metric("CG B", f"{cg_b:+.5f}")
    kc[2].metric("BS A", f"{bs_a:+.5f}", delta=f"Δ {bs_b-bs_a:+.5f}")
    kc[3].metric("BS B", f"{bs_b:+.5f}")

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
                                width='content')
    with pc2: st.plotly_chart(overlay("toe_deg", "Δ Toe vs Heave", "Δ Toe (°)"),
                                width='content')

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
    st.plotly_chart(fig_rc, width='content')