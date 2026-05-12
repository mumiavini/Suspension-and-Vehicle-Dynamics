"""
app.py
======
Streamlit app — Interface gráfica completa do motor FSAE Suspension Geometry.

ESTRUTURA EM ABAS:

    📊 Análise       : Carrega hardpoints existentes (xlsx/csv/json), seleciona
                       um corner, roda sweeps de heave/roll/steer, e mostra
                       KPIs + gráficos interativos.

    🎯 Síntese       : Engenharia REVERSA. O usuário define targets estáticos
                       (caster, KPI, camber, scrub) e dinâmicos (camber gain,
                       bump steer, RC height), opcionalmente define bounding
                       boxes, e o otimizador encontra hardpoints que atendem
                       às metas. Resultado pode ser baixado em .xlsx.

    🔄 Comparação    : Compara DUAS geometrias lado a lado (KPIs em tabela e
                       sobreposição de gráficos). Útil para validar mudanças
                       ou comparar geometria atual com proposta do otimizador.

COMO RODAR:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import streamlit as st

from analysis.io_hardpoints import (
    read_hardpoints,
    build_corner_from_dataframe,
    generate_template_dataframe,
    dataframe_from_corner,
    save_dataframe,
    HardpointValidationError,
    VALID_CORNERS,
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
st.caption("Motor de análise cinemática 3D + síntese de geometria de suspensão")


# =============================================================================
# Helpers gerais
# =============================================================================

def _save_uploaded_to_tmp(uploaded_file) -> Path:
    """Salva um arquivo upload do Streamlit em /tmp e retorna o path."""
    suffix = Path(uploaded_file.name).suffix
    tmp_path = Path("/tmp") / f"_fsae_upload{suffix}"
    tmp_path.write_bytes(uploaded_file.read())
    return tmp_path


def _load_hardpoints_from_state() -> Optional[pl.DataFrame]:
    """Recupera o DataFrame de hardpoints do session_state, se houver."""
    return st.session_state.get("hardpoints_df", None)


def _build_corner_safe(
    df: pl.DataFrame, corner_id: str,
) -> Optional[tuple[SuspensionCorner, TieRod]]:
    """Wrapper que mostra erro Streamlit se a construção falhar."""
    try:
        return build_corner_from_dataframe(df, corner_id)
    except HardpointValidationError as exc:
        st.error(f"❌ Erro ao construir corner '{corner_id}': {exc}")
        return None


def _run_sweep_cached(
    corner: SuspensionCorner,
    tie_rod: TieRod,
    sweep_type: str,
    params: tuple,
) -> np.ndarray:
    """
    Roda um sweep. Não usamos @st.cache_data porque os objetos não são hashable
    de forma trivial; o custo é baixo (<2s para sweep típico).
    """
    solver = KinematicSolver3D(corner, tie_rod)
    runner = SweepRunner(solver=solver)
    if sweep_type == "Heave":
        return runner.heave_sweep(*params)
    elif sweep_type == "Roll":
        return runner.roll_sweep(*params)
    else:
        return runner.steer_sweep(*params)


# =============================================================================
# SIDEBAR — Upload global de hardpoints (compartilhado entre as abas)
# =============================================================================

with st.sidebar:
    st.header("📂 Hardpoints do veículo")

    uploaded = st.file_uploader(
        "Carregue um arquivo de hardpoints",
        type=["xlsx", "csv", "json"],
        help=(
            "O arquivo deve conter colunas: corner, point, x_mm, y_mm, z_mm.\n"
            "Cada corner (FL/FR/RL/RR) precisa de 10 pontos."
        ),
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📋 Demo", width='content',
                     help="Carrega geometria FSAE realista de demonstração"):
            st.session_state["hardpoints_df"] = generate_template_dataframe()
            st.session_state["hardpoints_source"] = "Template demo"

    with col_b:
        # Download do template para o usuário começar do zero
        template_df = generate_template_dataframe()
        csv_bytes = template_df.write_csv().encode()
        st.download_button(
            "⬇️ Template",
            data=csv_bytes,
            file_name="hardpoints_template.csv",
            mime="text/csv",
            width='content',
            help="Baixa o template em CSV para edição manual",
        )

    # Processa upload
    if uploaded is not None:
        try:
            tmp = _save_uploaded_to_tmp(uploaded)
            df = read_hardpoints(tmp)
            st.session_state["hardpoints_df"] = df
            st.session_state["hardpoints_source"] = uploaded.name
            st.success(f"✅ Arquivo '{uploaded.name}' validado")
        except HardpointValidationError as exc:
            st.error(f"❌ Validação falhou: {exc}")
        except Exception as exc:
            st.error(f"❌ Erro ao ler arquivo: {exc}")

    # Indica o estado atual
    if "hardpoints_df" in st.session_state:
        st.info(f"📊 Carregado: **{st.session_state.get('hardpoints_source', '?')}**")
    else:
        st.warning("⚠️ Nenhum hardpoint carregado")

    st.markdown("---")
    st.caption("ℹ️ As 3 abas usam os hardpoints carregados acima.")


# =============================================================================
# Abas principais
# =============================================================================

tab_analysis, tab_synthesis, tab_compare = st.tabs([
    "📊 Análise",
    "🎯 Síntese / Otimização",
    "🔄 Comparação",
])


# ─────────────────────────────────────────────────────────────────────────────
# ABA 1 — ANÁLISE
# ─────────────────────────────────────────────────────────────────────────────

with tab_analysis:
    st.header("Análise cinemática")
    st.markdown(
        "Selecione um corner do veículo carregado e rode varreduras de **heave**, "
        "**roll** ou **steer** para extrair KPIs e gráficos."
    )

    df = _load_hardpoints_from_state()
    if df is None:
        st.info("👈 Carregue hardpoints na barra lateral para começar.")
    else:
        # ── Controles ─────────────────────────────────────────────────────────
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            corner_choice = st.selectbox("Corner", VALID_CORNERS, index=0,
                                          key="analysis_corner")

        with col2:
            sweep_type = st.radio("Tipo de sweep",
                                   ["Heave", "Roll", "Steer"],
                                   key="analysis_sweep_type")

        with col3:
            st.markdown("**Parâmetros do sweep**")
            if sweep_type == "Heave":
                c1, c2, c3 = st.columns(3)
                with c1: h_min  = st.number_input("Min (mm)",  value=-25.0, step=1.0, key="hmin")
                with c2: h_max  = st.number_input("Max (mm)",  value= 25.0, step=1.0, key="hmax")
                with c3: h_step = st.number_input("Step (mm)", value= 1.0, step=0.5, key="hstep")
                sweep_params = (h_min, h_max, h_step)
            elif sweep_type == "Roll":
                c1, c2, c3 = st.columns(3)
                with c1: r_min  = st.number_input("Min (°)",  value=-3.0, step=0.5, key="rmin")
                with c2: r_max  = st.number_input("Max (°)",  value= 3.0, step=0.5, key="rmax")
                with c3: r_step = st.number_input("Step (°)", value= 0.2, step=0.1, key="rstep")
                sweep_params = (r_min, r_max, r_step)
            else:  # Steer
                c1, c2, c3 = st.columns(3)
                with c1: s_min  = st.number_input("Min (mm)",  value=-30.0, step=1.0, key="smin")
                with c2: s_max  = st.number_input("Max (mm)",  value= 30.0, step=1.0, key="smax")
                with c3: s_step = st.number_input("Step (mm)", value= 1.0, step=0.5, key="sstep")
                sweep_params = (s_min, s_max, s_step)

        # ── Constrói o corner e mostra KPIs estáticos ─────────────────────────
        built = _build_corner_safe(df, corner_choice)
        if built is not None:
            corner, tie_rod = built

            st.markdown("### KPIs estáticos do corner selecionado")
            kc = st.columns(6)
            kc[0].metric("Caster (°)",         f"{corner.static_caster_deg():+.3f}")
            kc[1].metric("KPI (°)",            f"{corner.static_kpi_deg():+.3f}")
            kc[2].metric("Camber estático (°)",f"{corner.static_camber_deg():+.3f}")
            kc[3].metric("Scrub (mm)",         f"{corner.static_scrub_radius_mm():+.2f}")
            kc[4].metric("Trail (mm)",         f"{corner.static_mechanical_trail_mm():+.2f}")
            kc[5].metric("RC Height (mm)",     f"{corner.roll_center_height_mm():+.2f}")

            # ── Roda o sweep ──────────────────────────────────────────────────
            st.markdown(f"### {sweep_type} Sweep")
            with st.spinner(f"Executando {sweep_type.lower()} sweep..."):
                sweep = _run_sweep_cached(corner, tie_rod, sweep_type, sweep_params)

            # ── KPIs derivados do sweep ───────────────────────────────────────
            if sweep_type == "Heave":
                kd = st.columns(4)
                kd[0].metric("Camber Gain (°/mm)", f"{camber_gain_per_mm(sweep):+.5f}")
                kd[1].metric("Bump Steer (°/mm)",  f"{bump_steer_per_mm(sweep):+.5f}")
                dy, dz = rc_migration_range(sweep)
                kd[2].metric("RC ΔY (mm)", f"{dy:.2f}")
                kd[3].metric("RC ΔZ (mm)", f"{dz:.2f}")

            # ── Plots ─────────────────────────────────────────────────────────
            st.markdown("#### Gráficos")
            if sweep_type == "Heave":
                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    st.plotly_chart(plot_camber_vs_heave(sweep),
                                     width='content')
                with pcol2:
                    st.plotly_chart(plot_bump_steer(sweep),
                                     width='content')
                st.plotly_chart(plot_rc_migration(sweep),
                                 width='content')
            elif sweep_type == "Steer":
                st.plotly_chart(plot_caster_kpi_vs_steer(sweep),
                                 width='content')
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

            # ── Tabela completa ───────────────────────────────────────────────
            with st.expander("📋 Dados completos do sweep"):
                sweep_df = pl.DataFrame({n: sweep[n] for n in sweep.dtype.names})
                st.dataframe(sweep_df, width='content')


# ─────────────────────────────────────────────────────────────────────────────
# ABA 2 — SÍNTESE / OTIMIZAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

with tab_synthesis:
    st.header("Síntese de geometria — Engenharia reversa")
    st.markdown(
        "Defina os **targets** (estáticos e dinâmicos) e o otimizador busca "
        "os hardpoints que melhor os atendem. O ponto de partida (seed) é o "
        "corner selecionado abaixo."
    )

    df = _load_hardpoints_from_state()
    if df is None:
        st.info("👈 Carregue um arquivo de hardpoints na barra lateral primeiro "
                "(será usado como SEED inicial da otimização).")
    else:
        # ── Seleção do corner-seed ────────────────────────────────────────────
        seed_corner_id = st.selectbox("Corner-seed (ponto de partida)",
                                       VALID_CORNERS, index=0,
                                       key="synth_seed_corner")

        built = _build_corner_safe(df, seed_corner_id)
        if built is None:
            st.stop()
        seed_corner, seed_tie_rod = built

        st.markdown("---")

        # ── TARGETS ───────────────────────────────────────────────────────────
        st.subheader("🎯 Definição dos Targets")

        col_static, col_dynamic = st.columns(2)

        # ── Targets estáticos ─────────────────────────────────────────────────
        with col_static:
            st.markdown("**Estáticos** (medidos na posição neutra)")
            use_caster = st.checkbox("Caster", value=True, key="use_caster")
            tgt_caster = st.number_input("Caster alvo (°)",
                                          value=4.0, step=0.5,
                                          disabled=not use_caster,
                                          key="tgt_caster")

            use_kpi = st.checkbox("KPI", value=True, key="use_kpi")
            tgt_kpi = st.number_input("KPI alvo (°)",
                                       value=7.0, step=0.5,
                                       disabled=not use_kpi,
                                       key="tgt_kpi")

            use_camber = st.checkbox("Camber estático", value=True,
                                      key="use_camber")
            tgt_camber = st.number_input("Camber estático alvo (°)",
                                          value=-1.5, step=0.25,
                                          disabled=not use_camber,
                                          key="tgt_camber")

            use_scrub = st.checkbox("Scrub Radius", value=False,
                                     key="use_scrub")
            tgt_scrub = st.number_input("Scrub alvo (mm)",
                                         value=15.0, step=1.0,
                                         disabled=not use_scrub,
                                         key="tgt_scrub")

            use_trail = st.checkbox("Mechanical Trail", value=False,
                                     key="use_trail")
            tgt_trail = st.number_input("Trail alvo (mm)",
                                         value=20.0, step=1.0,
                                         disabled=not use_trail,
                                         key="tgt_trail")

        # ── Targets dinâmicos ─────────────────────────────────────────────────
        with col_dynamic:
            st.markdown("**Dinâmicos** (medidos no heave sweep)")
            tgt_cg = st.number_input("Camber Gain alvo (°/mm)",
                                      value=-0.020, step=0.005, format="%.3f",
                                      key="tgt_cg")
            tgt_bs = st.number_input("Bump Steer máx absoluto (°/mm)",
                                      value=0.005, step=0.001, format="%.3f",
                                      key="tgt_bs")
            tgt_rch = st.number_input("RC Height alvo (mm)",
                                       value=45.0, step=5.0,
                                       key="tgt_rch")
            tgt_rcm = st.number_input("RC ΔY máx (mm)",
                                       value=25.0, step=5.0,
                                       key="tgt_rcm")

            st.markdown("**Faixa do heave sweep**")
            hc1, hc2, hc3 = st.columns(3)
            with hc1: opt_h_min  = st.number_input("min", value=-25.0, step=1.0, key="opt_hmin")
            with hc2: opt_h_max  = st.number_input("max", value= 25.0, step=1.0, key="opt_hmax")
            with hc3: opt_h_step = st.number_input("step",value=  5.0, step=1.0, key="opt_hstep",
                                                    help="Passo maior = otimização mais rápida")

        st.markdown("---")

        # ── PESOS ─────────────────────────────────────────────────────────────
        with st.expander("⚙️ Pesos da função objetivo (avançado)"):
            st.markdown(
                "Os pesos controlam a IMPORTÂNCIA RELATIVA de cada termo. "
                "Aumente o peso de um target que está sendo violado para forçar "
                "o otimizador a respeitá-lo (em troca de pior performance em "
                "outros)."
            )
            wc1, wc2 = st.columns(2)
            with wc1:
                st.caption("Estáticos")
                w_caster = st.number_input("w_caster",        value=1.0, key="w_caster")
                w_kpi    = st.number_input("w_kpi",           value=1.0, key="w_kpi")
                w_camber = st.number_input("w_static_camber", value=5.0, key="w_camber")
                w_scrub  = st.number_input("w_scrub",         value=0.01, format="%.3f", key="w_scrub")
                w_trail  = st.number_input("w_trail",         value=0.01, format="%.3f", key="w_trail")
            with wc2:
                st.caption("Dinâmicos")
                w_cg     = st.number_input("w_camber_gain",   value=1.0, key="w_cg")
                w_bs     = st.number_input("w_bump_steer",    value=10.0, key="w_bs")
                w_rch    = st.number_input("w_rc_height",     value=0.01, format="%.3f", key="w_rch")
                w_rcm    = st.number_input("w_rc_migration",  value=0.05, format="%.3f", key="w_rcm")

        # ── BOUNDS ────────────────────────────────────────────────────────────
        with st.expander("📦 Bounding Boxes / Keep-out zones (avançado)"):
            st.markdown(
                "Restrinja o espaço de busca para cada hardpoint variável. "
                "Use para representar regiões interditadas pelo chassi ou "
                "packaging. Os valores são MARGENS em mm em torno do hardpoint do seed."
            )
            bc1, bc2 = st.columns(2)
            with bc1:
                margin_uca = st.slider("Margem UCA out (±mm)",  10, 100, 50, key="m_uca")
                margin_lca = st.slider("Margem LCA out (±mm)",  10, 100, 50, key="m_lca")
            with bc2:
                margin_tri = st.slider("Margem TR inboard (±mm)",  5, 50, 25, key="m_tri")
                margin_tro = st.slider("Margem TR outboard (±mm)", 5, 50, 25, key="m_tro")

        # ── CONTROLES DO OTIMIZADOR ───────────────────────────────────────────
        with st.expander("🔧 Configuração do solver evolutivo"):
            oc1, oc2 = st.columns(2)
            with oc1:
                pop_size = st.slider("População (×n_dims)", 5, 30, 12, key="pop")
                max_iter = st.slider("Iterações máx", 10, 200, 40, key="iter")
            with oc2:
                seed_rng = st.number_input("Random seed", value=42, key="seed_rng")
                workers  = st.selectbox("Paralelismo", [1, -1],
                                         format_func=lambda x: "1 core" if x == 1 else "Todos os cores",
                                         key="workers")

        st.markdown("---")

        # ── BOTÃO RUN ─────────────────────────────────────────────────────────
        run_col1, run_col2 = st.columns([1, 3])
        with run_col1:
            run_optimization = st.button("🚀 Rodar Otimização",
                                          type="primary",
                                          width='content')

        if run_optimization:
            # ── Constrói o objeto DesignTargets ──────────────────────────────
            targets = DesignTargets(
                # Dinâmicos
                camber_gain_target_deg_per_mm=tgt_cg,
                bump_steer_max_abs_deg_per_mm=tgt_bs,
                rc_height_target_mm=tgt_rch,
                rc_y_migration_max_mm=tgt_rcm,
                # Estáticos (None = desligado)
                caster_target_deg          = tgt_caster if use_caster else None,
                kpi_target_deg             = tgt_kpi    if use_kpi    else None,
                static_camber_target_deg   = tgt_camber if use_camber else None,
                scrub_radius_target_mm     = tgt_scrub  if use_scrub  else None,
                mechanical_trail_target_mm = tgt_trail  if use_trail  else None,
                # Sweep range
                heave_min_mm=opt_h_min,
                heave_max_mm=opt_h_max,
                heave_step_mm=opt_h_step,
                # Pesos
                w_camber_gain=w_cg, w_bump_steer=w_bs,
                w_rc_height=w_rch, w_rc_migration=w_rcm,
                w_caster=w_caster, w_kpi=w_kpi,
                w_static_camber=w_camber, w_scrub=w_scrub, w_trail=w_trail,
            )

            # ── Constrói os bounds ───────────────────────────────────────────
            def box_around(p: Point3D, m: float) -> HardpointBounds:
                return HardpointBounds(p.x-m, p.x+m, p.y-m, p.y+m, p.z-m, p.z+m)

            optimizer = SuspensionOptimizer(
                seed_corner=seed_corner,
                seed_tie_rod=seed_tie_rod,
                targets=targets,
                bounds_uca_outboard=box_around(seed_corner.upper_arm.outboard, margin_uca),
                bounds_lca_outboard=box_around(seed_corner.lower_arm.outboard, margin_lca),
                bounds_tie_rod_in  =box_around(seed_tie_rod.inboard,           margin_tri),
                bounds_tie_rod_out =box_around(seed_tie_rod.outboard,          margin_tro),
                population_size=pop_size,
                max_iterations=max_iter,
                seed=seed_rng,
                workers=workers,
            )

            # ── Validação inicial (seed) ─────────────────────────────────────
            seed_validation = validate_against_targets(seed_corner, seed_tie_rod, targets)
            seed_cost = optimizer.objective(optimizer._initial_guess_vector())

            with st.spinner(
                f"Rodando differential_evolution "
                f"({pop_size}×{max_iter} ≈ {pop_size*12*max_iter} avaliações)..."
            ):
                result = optimizer.run()

            # ── Validação pós-otimização ─────────────────────────────────────
            opt_validation = validate_against_targets(
                result.optimal_corner, result.optimal_tie_rod, targets,
            )

            # ── Mostra resultados ────────────────────────────────────────────
            st.success(f"✅ Otimização concluída em {result.scipy_result.nit} gerações "
                       f"({result.scipy_result.nfev} avaliações). "
                       f"Custo: {seed_cost:.3e} → {result.cost:.3e}")

            st.markdown("### Comparação Target × Seed × Otimizado")

            # Tabela combinada
            seed_rows = seed_validation.as_dict_list()
            opt_rows = opt_validation.as_dict_list()
            comparison = pl.DataFrame([
                {
                    "Parâmetro":      s["name"],
                    "Target":         s["target_str"],
                    "Seed":           s["obtained_str"],
                    "Otimizado":      o["obtained_str"],
                    "Erro Seed":      s["error_str"],
                    "Erro Otimizado": o["error_str"],
                    "OK Seed":        "✅" if s["ok"] else "❌",
                    "OK Otimizado":   "✅" if o["ok"] else "❌",
                }
                for s, o in zip(seed_rows, opt_rows)
            ])
            st.dataframe(comparison, width='content', hide_index=True)

            # ── Hardpoints resultantes ───────────────────────────────────────
            st.markdown("### 🎯 Hardpoints otimizados")
            opt_df = dataframe_from_corner(result.optimal_corner, result.optimal_tie_rod)
            st.dataframe(opt_df, width='content', hide_index=True)

            # ── Download ─────────────────────────────────────────────────────
            csv_bytes = opt_df.write_csv().encode()
            st.download_button(
                "⬇️ Baixar hardpoints otimizados (CSV)",
                data=csv_bytes,
                file_name=f"hardpoints_optimized_{seed_corner_id}.csv",
                mime="text/csv",
                type="primary",
            )

            # ── Guarda no session_state para a aba de Comparação ─────────────
            st.session_state["last_optimization"] = {
                "seed_corner":    seed_corner,
                "seed_tie_rod":   seed_tie_rod,
                "opt_corner":     result.optimal_corner,
                "opt_tie_rod":    result.optimal_tie_rod,
                "targets":        targets,
                "corner_id":      seed_corner_id,
            }
            st.info("💡 A geometria otimizada foi salva — vá na aba "
                    "**'🔄 Comparação'** para ver os gráficos lado a lado.")


# ─────────────────────────────────────────────────────────────────────────────
# ABA 3 — COMPARAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

with tab_compare:
    st.header("Comparação entre duas geometrias")
    st.markdown(
        "Compara sweeps de duas geometrias **A** e **B** lado a lado. Útil para:\n"
        "- Validar mudanças propostas pelo otimizador (Seed vs Otimizado)\n"
        "- Comparar geometria atual vs alternativa em estudo\n"
        "- A/B de diferentes cantos do mesmo carro"
    )

    df = _load_hardpoints_from_state()

    # ── Fonte de cada geometria ───────────────────────────────────────────────
    has_optimization = "last_optimization" in st.session_state

    col_src_a, col_src_b = st.columns(2)
    with col_src_a:
        st.markdown("**Geometria A**")
        source_a = st.radio(
            "Fonte A",
            ["Corner do arquivo", "Última geometria SEED (Aba 2)"]
            + (["Última geometria OTIMIZADA (Aba 2)"] if has_optimization else []),
            key="src_a",
            label_visibility="collapsed",
        )

    with col_src_b:
        st.markdown("**Geometria B**")
        source_b = st.radio(
            "Fonte B",
            ["Corner do arquivo", "Última geometria SEED (Aba 2)"]
            + (["Última geometria OTIMIZADA (Aba 2)"] if has_optimization else []),
            index=2 if has_optimization else 1,
            key="src_b",
            label_visibility="collapsed",
        )

    # ── Resolução das geometrias ──────────────────────────────────────────────
    def resolve_geometry(source: str, side: str,
                         ) -> Optional[tuple[SuspensionCorner, TieRod]]:
        if source == "Corner do arquivo":
            if df is None:
                st.warning(f"⚠️ Carregue um arquivo na sidebar para usar a fonte do lado {side}.")
                return None
            corner_id = st.selectbox(f"Corner {side}", VALID_CORNERS,
                                      key=f"compare_corner_{side}")
            return _build_corner_safe(df, corner_id)
        elif source == "Última geometria SEED (Aba 2)":
            if not has_optimization:
                st.warning("⚠️ Rode uma otimização na aba 2 primeiro.")
                return None
            lo = st.session_state["last_optimization"]
            return lo["seed_corner"], lo["seed_tie_rod"]
        else:  # OTIMIZADA
            lo = st.session_state["last_optimization"]
            return lo["opt_corner"], lo["opt_tie_rod"]

    cga, cgb = st.columns(2)
    with cga:
        geom_a = resolve_geometry(source_a, "A")
    with cgb:
        geom_b = resolve_geometry(source_b, "B")

    if geom_a is None or geom_b is None:
        st.stop()

    corner_a, tie_rod_a = geom_a
    corner_b, tie_rod_b = geom_b

    st.markdown("---")

    # ── Tabela comparativa de KPIs estáticos ──────────────────────────────────
    st.markdown("### KPIs estáticos")
    static_comparison = pl.DataFrame([
        {"Parâmetro": "Caster (°)",
         "A": f"{corner_a.static_caster_deg():+.3f}",
         "B": f"{corner_b.static_caster_deg():+.3f}",
         "Δ (B−A)": f"{corner_b.static_caster_deg() - corner_a.static_caster_deg():+.3f}"},
        {"Parâmetro": "KPI (°)",
         "A": f"{corner_a.static_kpi_deg():+.3f}",
         "B": f"{corner_b.static_kpi_deg():+.3f}",
         "Δ (B−A)": f"{corner_b.static_kpi_deg() - corner_a.static_kpi_deg():+.3f}"},
        {"Parâmetro": "Camber estático (°)",
         "A": f"{corner_a.static_camber_deg():+.3f}",
         "B": f"{corner_b.static_camber_deg():+.3f}",
         "Δ (B−A)": f"{corner_b.static_camber_deg() - corner_a.static_camber_deg():+.3f}"},
        {"Parâmetro": "Scrub Radius (mm)",
         "A": f"{corner_a.static_scrub_radius_mm():+.2f}",
         "B": f"{corner_b.static_scrub_radius_mm():+.2f}",
         "Δ (B−A)": f"{corner_b.static_scrub_radius_mm() - corner_a.static_scrub_radius_mm():+.2f}"},
        {"Parâmetro": "Trail Mecânico (mm)",
         "A": f"{corner_a.static_mechanical_trail_mm():+.2f}",
         "B": f"{corner_b.static_mechanical_trail_mm():+.2f}",
         "Δ (B−A)": f"{corner_b.static_mechanical_trail_mm() - corner_a.static_mechanical_trail_mm():+.2f}"},
        {"Parâmetro": "RC Height (mm)",
         "A": f"{corner_a.roll_center_height_mm():+.2f}",
         "B": f"{corner_b.roll_center_height_mm():+.2f}",
         "Δ (B−A)": f"{corner_b.roll_center_height_mm() - corner_a.roll_center_height_mm():+.2f}"},
    ])
    st.dataframe(static_comparison, width='content', hide_index=True)

    # ── Heave sweep nos dois ──────────────────────────────────────────────────
    st.markdown("### Heave Sweep — Sobreposição")
    hsc1, hsc2, hsc3 = st.columns(3)
    with hsc1: cmp_h_min  = st.number_input("Min (mm)", value=-25.0, key="cmp_hmin")
    with hsc2: cmp_h_max  = st.number_input("Max (mm)", value= 25.0, key="cmp_hmax")
    with hsc3: cmp_h_step = st.number_input("Step (mm)",value=  1.0, key="cmp_hstep")

    with st.spinner("Rodando sweeps das duas geometrias..."):
        sweep_a = _run_sweep_cached(corner_a, tie_rod_a, "Heave",
                                     (cmp_h_min, cmp_h_max, cmp_h_step))
        sweep_b = _run_sweep_cached(corner_b, tie_rod_b, "Heave",
                                     (cmp_h_min, cmp_h_max, cmp_h_step))

    # KPIs dinâmicos lado a lado
    kpi_cols = st.columns(4)
    cg_a, cg_b = camber_gain_per_mm(sweep_a), camber_gain_per_mm(sweep_b)
    bs_a, bs_b = bump_steer_per_mm(sweep_a),  bump_steer_per_mm(sweep_b)
    dy_a, dz_a = rc_migration_range(sweep_a)
    dy_b, dz_b = rc_migration_range(sweep_b)

    kpi_cols[0].metric("Camber Gain A (°/mm)", f"{cg_a:+.5f}",
                        delta=f"Δ {cg_b - cg_a:+.5f}")
    kpi_cols[1].metric("Camber Gain B (°/mm)", f"{cg_b:+.5f}")
    kpi_cols[2].metric("Bump Steer A (°/mm)",  f"{bs_a:+.5f}",
                        delta=f"Δ {bs_b - bs_a:+.5f}")
    kpi_cols[3].metric("Bump Steer B (°/mm)",  f"{bs_b:+.5f}")

    # ── Gráficos sobrepostos ──────────────────────────────────────────────────
    import plotly.graph_objects as go

    def overlay_plot(field: str, title: str, ylab: str) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sweep_a["heave_mm"], y=sweep_a[field],
            mode="lines+markers", name="A",
            line=dict(width=2, color="#1f77b4"),
        ))
        fig.add_trace(go.Scatter(
            x=sweep_b["heave_mm"], y=sweep_b[field],
            mode="lines+markers", name="B",
            line=dict(width=2, color="#d62728", dash="dash"),
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Heave (mm)",
            yaxis_title=ylab,
            template="plotly_white",
            hovermode="x unified",
        )
        return fig

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.plotly_chart(overlay_plot("camber_deg", "Camber vs Heave",
                                       "Camber (°)"),
                         width='content')
    with pcol2:
        st.plotly_chart(overlay_plot("toe_deg", "Δ Toe vs Heave",
                                       "Δ Toe (°)"),
                         width='content')

    # Migração do RC (trajetórias YxZ) em um único plot
    fig_rc = go.Figure()
    fig_rc.add_trace(go.Scatter(
        x=sweep_a["rc_y_mm"], y=sweep_a["rc_z_mm"],
        mode="lines+markers", name="RC A",
        line=dict(width=2, color="#1f77b4"),
    ))
    fig_rc.add_trace(go.Scatter(
        x=sweep_b["rc_y_mm"], y=sweep_b["rc_z_mm"],
        mode="lines+markers", name="RC B",
        line=dict(width=2, color="#d62728", dash="dash"),
    ))
    fig_rc.update_layout(
        title="Trajetória do Roll Center (Y × Z)",
        xaxis_title="RC Y (mm)",
        yaxis_title="RC Z (mm)",
        template="plotly_white",
    )
    fig_rc.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig_rc, width='content')