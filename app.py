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

import math
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import streamlit as st
import plotly.graph_objects as go

from analysis.io_hardpoints import (
    read_hardpoints,
    build_corner_from_dataframe,
    build_vehicle_from_dataframe,
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
from analysis.kpis import (
    static_sum_toe_deg,
    ackermann_geometry,
    steer_ratio_and_cfactor,
    steer_ratio_from_pinion,
    roll_center_at_1g_lat,
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
        if st.button("🔄 **Aplicar arquivo**", type="primary", use_container_width=True,
                      help="Carrega esse arquivo no app e recalcula todos os KPIs e gráficos"):
            st.session_state["hardpoints_df"]     = pending_df
            st.session_state["hardpoints_source"] = uploaded.name
            st.rerun()   # força re-render imediato com o novo arquivo

    # ─── DEMO + TEMPLATE (atalhos) ───────────────────────────────────────────
    st.markdown("**Ou use:**")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📋 Demo", use_container_width=True,
                      help="Carrega geometria FSAE realista de exemplo"):
            st.session_state["hardpoints_df"]     = generate_template_dataframe()
            st.session_state["hardpoints_source"] = "Template demo"
            st.rerun()

    with col_b:
        template_df = generate_template_dataframe()
        st.download_button("⬇️ Template", data=template_df.write_csv().encode(),
                            file_name="hardpoints_template.csv", mime="text/csv",
                            use_container_width=True,
                            help="Baixa o template em CSV para edição manual")

    # ─── ESTADO ATUAL ────────────────────────────────────────────────────────
    st.markdown("---")
    if "hardpoints_df" in st.session_state:
        st.info(f"📊 Em uso: **{st.session_state.get('hardpoints_source', '?')}**")
        if st.button("🗑️ Limpar", use_container_width=True,
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

tab_inputs, tab_analysis, tab_3d, tab_synthesis, tab_compare = st.tabs([
    "✏️ Inputs", "📊 Análise", "🌐 Vista 3D", "🎯 Síntese / Otimização", "🔄 Comparação",
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
                          use_container_width=True):
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
                      use_container_width=True):
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
                      use_container_width=True,
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
            use_container_width=True, disabled=["point"],
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
                          type="primary", use_container_width=True):
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
                    # CRÍTICO: atualizar synced_source também, senão o próximo
                    # rerun vai descartar os dados manuais por achar que a fonte
                    # mudou.
                    st.session_state["manual_synced_source"] = "Inputs manuais"
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
                mime="text/csv", use_container_width=True,
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

        st.plotly_chart(make_2d_view("YZ"), use_container_width=True,
                         key=f"yz_{edit_corner}")
        st.plotly_chart(make_2d_view("XZ"), use_container_width=True,
                         key=f"xz_{edit_corner}")
        st.plotly_chart(make_2d_view("XY"), use_container_width=True,
                         key=f"xy_{edit_corner}")


# ─────────────────────────────────────────────────────────────────────────────
# ABA 2 — ANÁLISE
# ─────────────────────────────────────────────────────────────────────────────

with tab_analysis:
    st.header("Análise — Ficha de setup completa")
    st.markdown(
        "Tabela com **todos os parâmetros do veículo**, lado a lado para "
        "dianteiro e traseiro. Valores são calculados automaticamente dos "
        "hardpoints quando possível; o que precisar de input adicional aparece "
        "abaixo."
    )

    df = _load_hardpoints_from_state()
    if df is None:
        st.info("👈 Carregue ou edite hardpoints primeiro.")
    else:
        vehicle, all_tie_rods = _build_vehicle_safe(df)
        if vehicle is None or all_tie_rods is None:
            st.error("Não foi possível construir o veículo completo a partir do arquivo.")
            st.stop()

        # ─── INPUTS DO USUÁRIO (parâmetros físicos não-calculáveis) ──────────
        with st.expander("🔧 **Inputs adicionais** — configurações que não vêm dos hardpoints", expanded=False):
            st.markdown(
                "Esses valores são necessários para calcular wheel rate, roll rate, "
                "frequência natural, motion ratio e damping. Deixe em branco (0) "
                "para que esses KPIs apareçam como `—` na tabela."
            )

            tab_tire, tab_susp, tab_mass, tab_damper, tab_other = st.tabs([
                "🛞 Pneus & Rodas",
                "🔩 Suspensão & Mola",
                "⚖️ Massas",
                "🌊 Amortecedor",
                "📝 Outros",
            ])

            with tab_tire:
                c1, c2 = st.columns(2)
                with c1:
                    tire_size  = st.text_input("Tire size, compound, make",
                                                 value="", placeholder="ex: 18.0×7.5-10 Hoosier R25B",
                                                 key="in_tire")
                    wheel_diam = st.number_input("Wheel diameter (inch)",
                                                   min_value=0.0, value=10.0, step=0.5,
                                                   key="in_wheel_diam")
                with c2:
                    wheel_mat  = st.text_input("Wheel material / construction",
                                                 value="", placeholder="ex: alumínio forjado 2-piece",
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
                                                      help="Curso útil de heave dianteiro (bump+rebound)")
                    spring_rate_f = st.number_input("Spring rate — FRONT (N/mm)",
                                                      min_value=0.0, value=0.0, step=1.0,
                                                      key="in_spring_f",
                                                      help="Rigidez da mola dianteira (no eixo da mola, não da roda)")
                    mr_front = st.number_input("Motion Ratio — FRONT",
                                                 min_value=0.0, value=0.0, step=0.05, format="%.3f",
                                                 key="in_mr_f",
                                                 help="MR = Δ(mola) / Δ(roda). Típico FSAE: 0.7–1.1")
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
                arb_adj = st.text_input("Suspension adjustment methods (outros)",
                                          value="",
                                          placeholder="ex: ARB com 3 posições, pré-carga variável",
                                          key="in_susp_methods")

            with tab_mass:
                c1, c2, c3 = st.columns(3)
                with c1:
                    total_mass = st.number_input("Massa total c/ piloto (kg)",
                                                   min_value=0.0, value=0.0, step=5.0,
                                                   key="in_mass_total",
                                                   help="Carro + piloto + combustível")
                with c2:
                    weight_dist_f = st.number_input("Distribuição de peso — FRONT (%)",
                                                     min_value=0.0, max_value=100.0,
                                                     value=45.0, step=0.5,
                                                     key="in_weight_dist_f",
                                                     help="% do peso no eixo dianteiro")
                with c3:
                    unsprung_per_corner = st.number_input("Massa não-suspensa por corner (kg)",
                                                            min_value=0.0, value=0.0, step=0.5,
                                                            key="in_unsprung",
                                                            help="Roda + pneu + manga + freio + ~50% braços")

            with tab_damper:
                c1, c2 = st.columns(2)
                with c1:
                    jounce_pct_f = st.number_input("Jounce damping — FRONT (% crítico)",
                                                     min_value=0.0, max_value=200.0,
                                                     value=0.0, step=5.0,
                                                     key="in_jounce_f")
                    rebound_pct_f = st.number_input("Rebound damping — FRONT (% crítico)",
                                                      min_value=0.0, max_value=200.0,
                                                      value=0.0, step=5.0,
                                                      key="in_rebound_f")
                with c2:
                    jounce_pct_r = st.number_input("Jounce damping — REAR (% crítico)",
                                                     min_value=0.0, max_value=200.0,
                                                     value=0.0, step=5.0,
                                                     key="in_jounce_r")
                    rebound_pct_r = st.number_input("Rebound damping — REAR (% crítico)",
                                                      min_value=0.0, max_value=200.0,
                                                      value=0.0, step=5.0,
                                                      key="in_rebound_r")

            with tab_other:
                roll_stiff = st.number_input("Roll stiffness (°/g) — usado para RC@1g",
                                               min_value=0.1, value=1.5, step=0.1,
                                               key="in_roll_stiff")
                ackermann_adj = st.selectbox(
                    "Ackermann ajustável?",
                    ["No", "Yes (multiple positions)", "Yes (continuous)"],
                    key="in_ack_adj",
                )

        # ─── INFRASTRUTURA: calcula sweeps F e R de uma vez ───────────────────
        vs = st.session_state["vehicle_setup"]

        @st.cache_data(show_spinner=False)
        def _compute_axle_data_cached(df_hash, axle_side, brake_bias, c_factor, sw_lock):
            """Computa todos os KPIs e sweeps para um eixo. Cache via hash do df."""
            # Esse wrapper existe só pra usar st.cache_data; o df_hash garante
            # invalidação quando os hardpoints mudam.
            return None

        # Quando df muda, force re-compute manualmente (cache invalidation)
        df_signature = df.write_csv()  # representação textual única do df

        def compute_axle(left_corner, left_tr, right_corner, right_tr,
                          is_front: bool):
            """Calcula todos os KPIs de um eixo (média esquerda+direita onde faz sentido)."""
            res: dict[str, object] = {}

            # ── Cinemática estática (mesma F e R) ──
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

            # ── Sweep de heave (esquerdo basta; geometria é simétrica) ──
            solver_l = KinematicSolver3D(left_corner, left_tr)
            runner_l = SweepRunner(solver=solver_l)
            heave_sweep = runner_l.heave_sweep(-25.0, 25.0, 2.5)
            res["ride_camber_dpm"] = camber_gain_per_mm(heave_sweep) * 1000.0
            res["camber_gain"]     = camber_gain_per_mm(heave_sweep)
            res["bump_steer"]      = bump_steer_per_mm(heave_sweep)
            res["rc_dy"], res["rc_dz"] = rc_migration_range(heave_sweep)

            # ── Roll sweep para roll camber ──
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
                res["anti_dive"]  = left_corner.anti_dive_percent(brake_bias=vs["brake_bias"])
                res["anti_squat"] = float("nan")
            else:
                res["anti_dive"]  = float("nan")
                res["anti_squat"] = left_corner.anti_squat_percent(drive_fraction=1.0)

            # ── Ackermann (só faz sentido na dianteira) ──
            if is_front:
                ack_info = ackermann_geometry(left_corner, left_tr,
                                               right_corner, right_tr,
                                               vehicle.rear_left)
                res["ackermann"] = ack_info["ackermann_percent"]
                res["steer_arm_l"] = ack_info["steer_arm_length_left"]
                res["steer_arm_r"] = ack_info["steer_arm_length_right"]

                # Steer ratio
                sr_info = steer_ratio_and_cfactor(left_corner, left_tr)
                if vs["c_factor_mm"] > 0:
                    res["steer_ratio"] = steer_ratio_from_pinion(
                        sr_info["rack_per_wheel_deg_mm_per_deg"], vs["c_factor_mm"])
                else:
                    res["steer_ratio"] = float("nan")
                res["c_factor"] = vs["c_factor_mm"]
            else:
                res["ackermann"]   = float("nan")
                res["steer_arm_l"] = float("nan")
                res["steer_arm_r"] = float("nan")
                res["steer_ratio"] = float("nan")
                res["c_factor"]    = float("nan")

            return res

        with st.spinner("Calculando KPIs dos dois eixos..."):
            front_data = compute_axle(vehicle.front_left, all_tie_rods["FL"],
                                       vehicle.front_right, all_tie_rods["FR"],
                                       is_front=True)
            rear_data  = compute_axle(vehicle.rear_left, all_tie_rods["RL"],
                                       vehicle.rear_right, all_tie_rods["RR"],
                                       is_front=False)

        # ─── Cálculos derivados que dependem de INPUTS do usuário ────────────
        def _wheel_rate(spring_rate: float, mr: float) -> float:
            """Wheel rate (N/mm) = spring_rate × MR²."""
            if spring_rate <= 0 or mr <= 0:
                return float("nan")
            return spring_rate * mr * mr

        def _roll_rate(wheel_rate: float, track_mm: float) -> float:
            """Roll rate por roda (Nm/°) = wheel_rate × track² / 2 × π/180 / 1000.

            Fórmula: K_roll = (1/2) × K_wheel × T² × (π/180) [Nm/°]
            onde K_wheel está em N/mm e T em mm. O fator 1000 converte mm² para m².
            """
            if math.isnan(wheel_rate) or track_mm <= 0:
                return float("nan")
            # wheel_rate N/mm = wheel_rate × 1000 N/m
            # roll rate em Nm/rad = (1/2) × K × T² (T em m)
            # converte para Nm/°: × π/180
            T_m = track_mm / 1000.0
            return 0.5 * (wheel_rate * 1000.0) * T_m * T_m * math.pi / 180.0

        def _natural_freq(wheel_rate: float, sprung_per_corner: float) -> float:
            """Frequência natural (Hz) = (1/2π) × √(K/M).

            K em N/m, M em kg → ω em rad/s → / 2π = Hz.
            """
            if math.isnan(wheel_rate) or sprung_per_corner <= 0:
                return float("nan")
            K = wheel_rate * 1000.0   # N/m
            return (1.0 / (2.0 * math.pi)) * math.sqrt(K / sprung_per_corner)

        # Calcula massas por corner
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

        # Aplica nos dados de cada eixo
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

        # ─── Helper de formatação ────────────────────────────────────────────
        def fmt(v, fmt_str="+.3f") -> str:
            """Formata um número; retorna '—' se NaN ou input ausente."""
            if v is None:
                return "—"
            try:
                if math.isnan(float(v)):
                    return "—"
                return format(float(v), fmt_str)
            except (TypeError, ValueError):
                return str(v)

        def fmt_pair(v_l, v_r, fmt_str="+.3f") -> str:
            """Formata 'L / R' para parâmetros que diferem por roda."""
            return f"{fmt(v_l, fmt_str)} / {fmt(v_r, fmt_str)}"

        # ─── TABELA PRINCIPAL ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📋 Ficha de Setup Completa")

        # Construção da tabela linha-por-linha
        # Cada linha: [Parâmetro, Unidade, Front, Rear, Origem]
        # Origem: 📐 calculado / ⌨️ input / 🧮 derivado

        rows: list[dict[str, str]] = []

        def add(param, unit, f_val, r_val, origin):
            rows.append({
                "Parâmetro": param,
                "Unidade":   unit,
                "Front":     str(f_val),
                "Rear":      str(r_val),
                "Origem":    origin,
            })

        # Inputs de pneu/roda (mesmo para os dois eixos)
        add("Tire size, compound, make",          "",
            tire_size or "—",  tire_size or "—",  "⌨️ input")
        add("Wheel (diameter × width)",           "inch",
            f"{wheel_diam:.1f} × {wheel_wid:.1f}" if wheel_diam else "—",
            f"{wheel_diam:.1f} × {wheel_wid:.1f}" if wheel_diam else "—",
            "⌨️ input")
        add("Wheel material / construction",      "",
            wheel_mat or "—", wheel_mat or "—", "⌨️ input")
        add("Suspension type",                    "",
            susp_type or "—", susp_type or "—", "⌨️ input")
        add("Suspension design travel",           "mm",
            fmt(front_data["travel"], ".1f"),
            fmt(rear_data["travel"],  ".1f"),
            "⌨️ input")

        # Derivados (precisam de inputs)
        add("Wheel rate (chassis → wheel center)", "N/mm",
            fmt(front_data["wheel_rate"], ".2f"),
            fmt(rear_data["wheel_rate"],  ".2f"),
            "🧮 derivado de mola + MR")
        add("Roll rate (chassis → wheel center)",  "Nm/deg",
            fmt(front_data["roll_rate"], ".1f"),
            fmt(rear_data["roll_rate"],  ".1f"),
            "🧮 derivado de wheel rate + track")
        add("Sprung mass natural frequency",       "Hz",
            fmt(front_data["nat_freq"], ".2f"),
            fmt(rear_data["nat_freq"],  ".2f"),
            "🧮 derivado de wheel rate + massa")
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

        # Calculados (geometria)
        add("Ride Camber (rate of change)",        "deg/m",
            fmt(front_data["ride_camber_dpm"], "+.2f"),
            fmt(rear_data["ride_camber_dpm"],  "+.2f"),
            "📐 calculado")
        add("Roll Camber",                          "deg/deg",
            fmt(front_data["roll_camber"], "+.4f"),
            fmt(rear_data["roll_camber"],  "+.4f"),
            "📐 calculado")
        add("Static Sum Toe (− out, + in)",         "deg",
            fmt(front_data["sum_toe"], "+.4f"),
            fmt(rear_data["sum_toe"],  "+.4f"),
            "📐 calculado")
        add("Static camber (L / R)",                "deg",
            fmt_pair(front_data["camber_l"], front_data["camber_r"], "+.3f"),
            fmt_pair(rear_data["camber_l"],  rear_data["camber_r"],  "+.3f"),
            "📐 calculado")
        add("Static camber adjustment method",      "",
            susp_adj or "—", susp_adj or "—", "⌨️ input")
        add("Anti dive / Anti squat",               "%",
            fmt(front_data["anti_dive"],  "+.2f"),
            fmt(rear_data["anti_squat"], "+.2f"),
            "📐 calculado (precisa CG, brake bias)")
        add("Roll center height above ground, static", "mm",
            fmt(front_data["rc_static"], "+.2f"),
            fmt(rear_data["rc_static"],  "+.2f"),
            "📐 calculado")
        add("Roll center @ 1g lateral acc — height",   "mm",
            fmt(front_data["rc_1g_z"], "+.2f"),
            fmt(rear_data["rc_1g_z"],  "+.2f"),
            f"📐 calculado (roll stiffness {roll_stiff}°/g)")
        add("Roll center @ 1g lateral acc — lateral",  "mm",
            fmt(front_data["rc_1g_y"], "+.2f"),
            fmt(rear_data["rc_1g_y"],  "+.2f"),
            f"📐 calculado (roll stiffness {roll_stiff}°/g)")
        add("Caster (L / R)",                         "deg",
            fmt_pair(front_data["caster_l"], front_data["caster_r"], "+.3f"),
            "N/A (sem caster traseiro relevante)",
            "📐 calculado")
        add("Kingpin trail (L / R)",                  "mm",
            fmt_pair(front_data["trail_l"], front_data["trail_r"], "+.2f"),
            fmt_pair(rear_data["trail_l"],  rear_data["trail_r"],  "+.2f"),
            "📐 calculado")
        add("Scrub radius (L / R)",                   "mm",
            fmt_pair(front_data["scrub_l"], front_data["scrub_r"], "+.2f"),
            fmt_pair(rear_data["scrub_l"],  rear_data["scrub_r"],  "+.2f"),
            "📐 calculado")
        add("Kingpin Inclination (L / R)",             "deg",
            fmt_pair(front_data["kpi_l"], front_data["kpi_r"], "+.3f"),
            fmt_pair(rear_data["kpi_l"],  rear_data["kpi_r"],  "+.3f"),
            "📐 calculado")
        add("Static Ackermann",                        "%",
            fmt(front_data["ackermann"], "+.2f"),
            "N/A",
            "📐 calculado")
        add("Ackermann ajustável?",                    "",
            ackermann_adj, "—",
            "⌨️ input")
        add("Suspension adjustment methods",           "",
            arb_adj or "—", arb_adj or "—",
            "⌨️ input")
        add("Steer Ratio",                             "x:1",
            fmt(front_data["steer_ratio"], ".2f"),
            "N/A",
            f"🧮 derivado de c-factor={vs['c_factor_mm']:.0f} mm/rev")
        add("C-factor",                                "mm/rev",
            fmt(front_data["c_factor"], ".1f"),
            "N/A",
            "⌨️ input (sidebar)")
        add("Steer Arm Length (L / R)",                "mm",
            fmt_pair(front_data["steer_arm_l"], front_data["steer_arm_r"], ".2f"),
            "N/A",
            "📐 calculado")

        # Massas / distribuição
        if total_mass > 0:
            add("Massa total c/ piloto",               "kg",
                f"{total_mass:.1f}", f"{total_mass:.1f}", "⌨️ input")
            if not math.isnan(sprung_total):
                add("Massa suspensa total",            "kg",
                    f"{sprung_total:.1f}", f"{sprung_total:.1f}", "🧮 derivado")
                add("Massa suspensa por corner",       "kg",
                    fmt(sprung_front_per_corner, ".1f"),
                    fmt(sprung_rear_per_corner,  ".1f"),
                    "🧮 derivado")
            add("Massa não-suspensa por corner",       "kg",
                f"{unsprung_per_corner:.1f}", f"{unsprung_per_corner:.1f}",
                "⌨️ input")
            add("Distribuição de peso",                "%",
                f"{weight_dist_f:.1f}", f"{100-weight_dist_f:.1f}",
                "⌨️ input")

        # Renderiza tabela
        table_df = pl.DataFrame(rows)
        st.dataframe(table_df, use_container_width=True, hide_index=True, height=900)

        # Legenda
        st.caption(
            "**Legenda:** "
            "📐 calculado dos hardpoints · "
            "⌨️ input do usuário · "
            "🧮 derivado (precisa de inputs nos expanders acima)"
        )

        # Download da tabela
        csv_table = table_df.write_csv().encode()
        st.download_button(
            "⬇️ Baixar ficha de setup (CSV)",
            data=csv_table,
            file_name="setup_sheet.csv",
            mime="text/csv",
        )

        # ─── SWEEPS COM GRÁFICOS (mantido, opcional) ─────────────────────────
        st.markdown("---")
        st.markdown("### 📈 Sweeps detalhados (opcional)")

        with st.expander("Mostrar gráficos de sweep para um corner específico", expanded=False):
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

            built = _build_corner_safe(df, corner_choice)
            if built is not None:
                corner, tie_rod = built
                with st.spinner(f"{sweep_type} sweep..."):
                    sweep = _run_sweep_cached(corner, tie_rod, sweep_type, sweep_params)

                if sweep_type == "Heave":
                    pc1, pc2 = st.columns(2)
                    with pc1:
                        st.plotly_chart(plot_camber_vs_heave(sweep), use_container_width=True)
                    with pc2:
                        st.plotly_chart(plot_bump_steer(sweep), use_container_width=True)
                    st.plotly_chart(plot_rc_migration(sweep), use_container_width=True)
                elif sweep_type == "Steer":
                    st.plotly_chart(plot_caster_kpi_vs_steer(sweep), use_container_width=True)
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=sweep["roll_deg"], y=sweep["camber_deg"],
                                               mode="lines+markers"))
                    fig.update_layout(title="Camber vs Roll", xaxis_title="Roll (°)",
                                       yaxis_title="Camber (°)", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

                with st.expander("📋 Dados do sweep"):
                    sweep_df = pl.DataFrame({n: sweep[n] for n in sweep.dtype.names})
                    st.dataframe(sweep_df, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ABA 3 — VISTA 3D
# ─────────────────────────────────────────────────────────────────────────────

with tab_3d:
    st.header("Visualização 3D dos hardpoints")
    st.markdown(
        "Veja a suspensão em **3D interativo**: rotacione, dê zoom, e veja como "
        "os hardpoints se relacionam no espaço. Use o modo animado para ver o "
        "movimento durante heave, roll ou steer."
    )

    df = _load_hardpoints_from_state()
    if df is None:
        st.info("👈 Carregue hardpoints na barra lateral primeiro.")
    else:
        from analysis.viz3d import (plot_corner_3d, plot_vehicle_3d,
                                     plot_corner_animated)

        # ── Controles principais ─────────────────────────────────────────────
        view_mode = st.radio(
            "Modo de visualização",
            ["🏎️ Veículo completo", "🔍 Corner individual",
             "🎬 Animação de sweep"],
            horizontal=True,
            key="view3d_mode",
        )

        st.markdown("---")

        # ─── MODO 1: VEÍCULO COMPLETO ────────────────────────────────────────
        if view_mode == "🏎️ Veículo completo":
            try:
                vehicle, tie_rods = build_vehicle_from_dataframe(df)

                show_tires = st.checkbox("Mostrar pneus", value=True,
                                         key="veh_show_tires")
                show_chassis = st.checkbox("Mostrar wireframe do chassi", value=True,
                                            key="veh_show_chassis")

                with st.spinner("Renderizando..."):
                    fig = plot_vehicle_3d(
                        vehicle, tie_rods,
                        show_tires=show_tires,
                        show_chassis_box=show_chassis,
                        title="Suspensão FSAE — Vista 3D completa",
                    )
                st.plotly_chart(fig, use_container_width=True)

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

            built = _build_corner_safe(df, corner_choice)
            if built is not None:
                corner, tie_rod = built
                with st.spinner("Renderizando..."):
                    fig = plot_corner_3d(corner, tie_rod, show_tire=show_tire)
                st.plotly_chart(fig, use_container_width=True)

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

            built = _build_corner_safe(df, corner_choice)
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
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "💡 **Dica:** arraste o slider para ver a geometria em cada "
                    "posição, ou clique em ▶ Play para animar automaticamente."
                )


# ─────────────────────────────────────────────────────────────────────────────
# ABA 4 — SÍNTESE
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
            st.dataframe(comparison, use_container_width=True, hide_index=True)

            st.markdown("### 🎯 Hardpoints otimizados")
            opt_df = dataframe_from_corner(result.optimal_corner, result.optimal_tie_rod)
            st.dataframe(opt_df, use_container_width=True, hide_index=True)

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
    st.dataframe(static_cmp, use_container_width=True, hide_index=True)

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
                                use_container_width=True)
    with pc2: st.plotly_chart(overlay("toe_deg", "Δ Toe vs Heave", "Δ Toe (°)"),
                                use_container_width=True)

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
    st.plotly_chart(fig_rc, use_container_width=True)
