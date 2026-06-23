"""
ui/tab_synthesis.py
===================
🎯 Aba Síntese — otimização global de hardpoints a partir de targets
(engenharia reversa), via `analysis.optimizer`. Os resultados ficam em
`st.session_state["last_optimization"]`, consumido também pela aba Comparação.
"""

from __future__ import annotations

import math
import time

import polars as pl
import streamlit as st
import plotly.graph_objects as go

from analysis.io_hardpoints import (
    dataframe_from_corner,
    VALID_CORNERS,
)
from geometry import Point3D
from analysis.sweeps import camber_gain_per_mm, bump_steer_per_mm
from analysis.optimizer import (
    SuspensionOptimizer,
    DesignTargets,
    HardpointBounds,
    validate_against_targets,
)
from ui.shared import (
    load_hardpoints_from_state,
    render_empty_state,
    build_corner_safe,
    run_sweep_cached,
)

MIRROR_CORNER = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}


def _movable_points(corner, tie_rod) -> dict[str, Point3D]:
    """Os 4 hardpoints que o otimizador pode mover."""
    return {
        "UCA_OUT":     corner.upper_arm.outboard,
        "LCA_OUT":     corner.lower_arm.outboard,
        "TIE_ROD_IN":  tie_rod.inboard,
        "TIE_ROD_OUT": tie_rod.outboard,
    }


def _overlay_2d_figure(seed_corner, seed_tr, opt_corner, opt_tr,
                       bounds: dict, view: str) -> go.Figure:
    """Sobreposição 2D seed × otimizado, com as caixas de busca (bounds)."""

    def uv(p):
        return (p.y, p.z) if view == "YZ" else (p.x, p.z)

    def add_geometry(fig, corner, tr, color, name, dash=None):
        ua, la = corner.upper_arm, corner.lower_arm
        segments = [
            (ua.inboard_front, ua.outboard), (ua.inboard_rear, ua.outboard),
            (la.inboard_front, la.outboard), (la.inboard_rear, la.outboard),
            (tr.inboard, tr.outboard),
            (ua.outboard, la.outboard),                    # manga
            (corner.wheel_center, corner.contact_patch),   # roda
        ]
        xs, ys = [], []
        for p1, p2 in segments:
            u1, v1 = uv(p1)
            u2, v2 = uv(p2)
            xs += [u1, u2, None]
            ys += [v1, v2, None]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=name,
            line=dict(color=color, width=2, dash=dash),
            hoverinfo="skip",
        ))
        mv = _movable_points(corner, tr)
        fig.add_trace(go.Scatter(
            x=[uv(p)[0] for p in mv.values()],
            y=[uv(p)[1] for p in mv.values()],
            mode="markers", showlegend=False,
            marker=dict(size=9, color=color, line=dict(width=1, color="white")),
            text=list(mv.keys()),
            hovertemplate="<b>%{text}</b> (" + name + ")<br>%{x:.1f}, %{y:.1f}<extra></extra>",
        ))

    fig = go.Figure()

    # Caixas de busca dos pontos móveis
    for b in bounds.values():
        if view == "YZ":
            u0, u1, v0, v1 = b.y_min, b.y_max, b.z_min, b.z_max
        else:
            u0, u1, v0, v1 = b.x_min, b.x_max, b.z_min, b.z_max
        fig.add_shape(type="rect", x0=u0, x1=u1, y0=v0, y1=v1,
                      line=dict(color="rgba(120,120,120,0.45)", width=1, dash="dot"),
                      fillcolor="rgba(120,120,120,0.06)")

    add_geometry(fig, seed_corner, seed_tr, "#1f77b4", "Seed", dash="dash")
    add_geometry(fig, opt_corner, opt_tr, "#d62728", "Otimizado")
    fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=1))

    axis_u = "Y" if view == "YZ" else "X"
    fig.update_layout(
        title=f"Vista {view}",
        xaxis_title=f"{axis_u} (mm)", yaxis_title="Z (mm)",
        template="plotly_white", height=420,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _apply_optimized_to_session(res: dict, mirror: bool) -> None:
    """Substitui no df da sessão os hardpoints do corner otimizado
    (e do corner oposto espelhado, se pedido) e re-executa o app."""
    cid = res["corner_id"]
    base = st.session_state.get("hardpoints_df")
    if base is None:
        st.error("❌ Nenhum hardpoint carregado na sessão.")
        return

    opt_rows = dataframe_from_corner(res["opt_corner"], res["opt_tie_rod"])
    replaced = [cid] + ([MIRROR_CORNER[cid]] if mirror else [])
    pieces = [
        base.filter(~pl.col("corner").is_in(replaced)),
        opt_rows.select(base.columns),
    ]
    if mirror:
        pieces.append(
            opt_rows.with_columns(
                pl.lit(MIRROR_CORNER[cid]).alias("corner"),
                (-pl.col("y_mm")).alias("y_mm"),
            ).select(base.columns)
        )
    label = f"Otimizado ({' + '.join(replaced)})"
    st.session_state["hardpoints_df"] = pl.concat(pieces, how="vertical_relaxed")
    st.session_state["hardpoints_source"] = label
    st.toast(f"Hardpoints aplicados: {label}", icon="✅")
    st.rerun(scope="app")


@st.fragment
def render() -> None:
    """
    Aba de síntese isolada em um st.fragment: interagir com qualquer widget
    daqui re-executa SÓ esta aba, não o app inteiro (em particular, não
    recalcula a aba de Análise) — interação muito mais rápida.
    """
    st.header("Síntese de geometria — Engenharia reversa")

    df = load_hardpoints_from_state()
    if df is None:
        render_empty_state(
            "A síntese parte de uma geometria existente (o **seed**) e move "
            "hardpoints automaticamente para atingir os targets definidos.",
            key="empty_synthesis",
        )
        return

    # ── Corner-seed + fotografia dos KPIs atuais ─────────────────────────────
    sel_col, kpi_col = st.columns([1, 4])
    with sel_col:
        seed_corner_id = st.selectbox("Corner-seed", VALID_CORNERS,
                                       key="synth_seed_corner")
    built = build_corner_safe(df, seed_corner_id)
    if built is None:
        return
    seed_corner, seed_tie_rod = built

    with kpi_col:
        st.caption("**Valores atuais do seed** — use como referência ao definir os targets")
        seed_sweep = run_sweep_cached(seed_corner, seed_tie_rod, "Heave",
                                       (-25.0, 25.0, 5.0))
        m = st.columns(6)
        m[0].metric("Caster",      f"{seed_corner.static_caster_deg():+.2f}°",
                    border=True)
        m[1].metric("KPI",         f"{seed_corner.static_kpi_deg():+.2f}°",
                    border=True)
        m[2].metric("Camber",      f"{seed_corner.static_camber_deg():+.2f}°",
                    border=True)
        m[3].metric("Camber gain", f"{camber_gain_per_mm(seed_sweep):+.4f} °/mm",
                    border=True)
        m[4].metric("Bump steer",  f"{bump_steer_per_mm(seed_sweep):+.4f} °/mm",
                    border=True)
        m[5].metric("RC height",   f"{seed_corner.roll_center_height_mm():+.1f} mm",
                    border=True)

    st.markdown("---")
    st.subheader("🎯 Targets")

    def _target_row(label, use_key, val_key, default, step, on_default, fmt=None):
        """Toggle + valor numérico na mesma linha."""
        c_use, c_val = st.columns([1.3, 1], vertical_alignment="center")
        use = c_use.toggle(label, value=on_default, key=use_key)
        val = c_val.number_input(label, value=default, step=step,
                                 disabled=not use, key=val_key,
                                 label_visibility="collapsed", format=fmt)
        return use, val

    col_static, col_dynamic = st.columns(2)

    with col_static.container(border=True):
        st.markdown("**Estáticos** — ative os que o otimizador deve perseguir")

        use_caster, tgt_caster = _target_row("Caster (°)", "use_caster",
                                             "tgt_caster", 4.0, 0.5, True)
        use_kpi, tgt_kpi       = _target_row("KPI (°)", "use_kpi",
                                             "tgt_kpi", 7.0, 0.5, True)
        use_camber, tgt_camber = _target_row("Camber estático (°)", "use_camber",
                                             "tgt_camber", -1.5, 0.25, True)
        use_scrub, tgt_scrub   = _target_row("Scrub (mm)", "use_scrub",
                                             "tgt_scrub", 15.0, 1.0, False)
        use_trail, tgt_trail   = _target_row("Trail (mm)", "use_trail",
                                             "tgt_trail", 20.0, 1.0, False)

    with col_dynamic.container(border=True):
        st.markdown("**Dinâmicos** — desativar zera o peso do termo no custo")

        use_cg, tgt_cg   = _target_row("Camber Gain (°/mm)", "use_cg",
                                       "tgt_cg", -0.020, 0.005, True, fmt="%.3f")
        use_bs, tgt_bs   = _target_row("Bump Steer máx (°/mm)", "use_bs",
                                       "tgt_bs", 0.005, 0.001, True, fmt="%.3f")
        use_rch, tgt_rch = _target_row("RC Height (mm)", "use_rch",
                                       "tgt_rch", 45.0, 1.0, True)
        use_rcm, tgt_rcm = _target_row("RC ΔY máx (mm)", "use_rcm",
                                       "tgt_rcm", 25.0, 1.0, True)
        st.markdown("**Heave sweep range**")
        hc1, hc2, hc3 = st.columns(3)
        with hc1: opt_h_min  = st.number_input("min", value=-25.0, key="opt_hmin")
        with hc2: opt_h_max  = st.number_input("max", value= 25.0, key="opt_hmax")
        with hc3: opt_h_step = st.number_input("step", value= 5.0, key="opt_hstep")

    exp_w, exp_b, exp_s = st.columns(3)
    with exp_w:
        with st.expander("⚙️ Pesos"):
            w_caster = st.number_input("w_caster",        value=1.0,  key="w_caster")
            w_kpi    = st.number_input("w_kpi",           value=1.0,  key="w_kpi")
            w_camber = st.number_input("w_static_camber", value=5.0,  key="w_camber")
            w_scrub  = st.number_input("w_scrub",  value=0.01, format="%.3f", key="w_scrub")
            w_trail  = st.number_input("w_trail",  value=0.01, format="%.3f", key="w_trail")
            w_cg     = st.number_input("w_camber_gain",   value=1.0,  key="w_cg")
            w_bs     = st.number_input("w_bump_steer",    value=10.0, key="w_bs")
            w_rch    = st.number_input("w_rc_height",     value=0.01, format="%.3f", key="w_rch")
            w_rcm    = st.number_input("w_rc_migration",  value=0.05, format="%.3f", key="w_rcm")

    with exp_b:
        with st.expander("📦 Bounds"):
            margin_uca = st.slider("UCA out (±mm)", 10, 100, 50, key="m_uca")
            margin_lca = st.slider("LCA out (±mm)", 10, 100, 50, key="m_lca")
            margin_tri = st.slider("TR in (±mm)",  5, 50, 25, key="m_tri")
            margin_tro = st.slider("TR out (±mm)", 5, 50, 25, key="m_tro")

    with exp_s:
        with st.expander("🔧 Solver"):
            pop_size = st.slider("População (×n_dims)", 5, 30, 12, key="pop")
            max_iter = st.slider("Iterações", 10, 200, 40, key="iter")
            seed_rng = st.number_input("seed", value=42, key="seed_rng")
            workers  = st.selectbox("Cores", [1, -1],
                                      format_func=lambda x: "1" if x == 1 else "Todos",
                                      key="workers")
            polish_opt = st.checkbox("Polish (refino local ao final)", value=True,
                                      key="polish_opt",
                                      help="Roda L-BFGS-B após o DE. Melhora o "
                                           "resultado, mas adiciona avaliações extras.")
            max_seconds = st.number_input("Tempo máx (s) — 0 = sem limite",
                                           min_value=0, value=0, step=10,
                                           key="max_seconds",
                                           help="Interrompe ao exceder, mantendo o "
                                                "melhor resultado até então.")

    st.markdown("---")
    run_col, clear_col, info_col = st.columns([1, 1, 2])
    with run_col:
        run_opt = st.button("🚀 Rodar Otimização", type="primary",
                             width="stretch")
    with clear_col:
        if st.session_state.get("last_optimization") is not None:
            if st.button("🗑️ Limpar resultados", width="stretch"):
                st.session_state.pop("last_optimization", None)
                st.rerun(scope="fragment")
    with info_col:
        n_evals = pop_size * 12 * (max_iter + 1)
        st.caption(f"≈ até {n_evals:,} avaliações da função objetivo "
                   f"({pop_size}×12 indivíduos × {max_iter} gerações)")

    # ── Execução ─────────────────────────────────────────────────────────────
    if run_opt:
        if not any([use_caster, use_kpi, use_camber, use_scrub, use_trail,
                    use_cg, use_bs, use_rch, use_rcm]):
            st.error("❌ Ative pelo menos um target (estático ou dinâmico) — "
                     "com todos desligados a função objetivo é constante e o "
                     "otimizador não tem o que perseguir.")
            return

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
            # Toggle desligado ⇒ peso 0 ⇒ termo sai da função objetivo
            w_camber_gain  = w_cg  if use_cg  else 0.0,
            w_bump_steer   = w_bs  if use_bs  else 0.0,
            w_rc_height    = w_rch if use_rch else 0.0,
            w_rc_migration = w_rcm if use_rcm else 0.0,
            w_caster=w_caster, w_kpi=w_kpi,
            w_static_camber=w_camber, w_scrub=w_scrub, w_trail=w_trail,
        )

        def box_around(p: Point3D, m: float) -> HardpointBounds:
            return HardpointBounds(p.x-m, p.x+m, p.y-m, p.y+m, p.z-m, p.z+m)

        bounds_map = {
            "UCA_OUT":     box_around(seed_corner.upper_arm.outboard, margin_uca),
            "LCA_OUT":     box_around(seed_corner.lower_arm.outboard, margin_lca),
            "TIE_ROD_IN":  box_around(seed_tie_rod.inboard,           margin_tri),
            "TIE_ROD_OUT": box_around(seed_tie_rod.outboard,          margin_tro),
        }

        progress = st.progress(0.0, text="Inicializando população…")
        t0 = time.monotonic()

        def on_generation(gen: int, best_cost: float, convergence: float) -> bool:
            elapsed = time.monotonic() - t0
            progress.progress(
                min(gen / max_iter, 1.0),
                text=f"Geração {gen}/{max_iter} · melhor custo {best_cost:.3e} "
                     f"· {elapsed:.0f}s",
            )
            return max_seconds > 0 and elapsed > max_seconds

        optimizer = SuspensionOptimizer(
            seed_corner=seed_corner, seed_tie_rod=seed_tie_rod, targets=targets,
            bounds_uca_outboard=bounds_map["UCA_OUT"],
            bounds_lca_outboard=bounds_map["LCA_OUT"],
            bounds_tie_rod_in  =bounds_map["TIE_ROD_IN"],
            bounds_tie_rod_out =bounds_map["TIE_ROD_OUT"],
            population_size=pop_size, max_iterations=max_iter,
            seed=seed_rng, workers=workers, polish=polish_opt,
            on_generation=on_generation,
        )

        seed_validation = validate_against_targets(seed_corner, seed_tie_rod, targets)
        seed_cost = optimizer.objective(optimizer._initial_guess_vector())

        try:
            result = optimizer.run()
        except Exception as exc:
            progress.empty()
            st.error(f"❌ Otimização falhou: {exc}")
            return
        elapsed = time.monotonic() - t0
        progress.empty()

        opt_validation = validate_against_targets(
            result.optimal_corner, result.optimal_tie_rod, targets,
        )

        st.session_state["last_optimization"] = {
            # consumido pela aba 🔄 Comparação
            "seed_corner": seed_corner, "seed_tie_rod": seed_tie_rod,
            "opt_corner": result.optimal_corner, "opt_tie_rod": result.optimal_tie_rod,
            "targets": targets, "corner_id": seed_corner_id,
            # consumido pela seção de resultados abaixo
            "seed_cost": float(seed_cost), "cost": float(result.cost),
            "nit": int(result.scipy_result.nit),
            "nfev": int(result.scipy_result.nfev),
            "message": str(result.scipy_result.message),
            "history": list(result.convergence_history),
            "elapsed_s": elapsed,
            "seed_rows": seed_validation.as_dict_list(),
            "opt_rows": opt_validation.as_dict_list(),
            "bounds": bounds_map,
        }
        st.toast(f"Otimização concluída em {elapsed:.1f}s", icon="🏁")

    # ── Resultados (persistem entre interações — fora do bloco do botão) ─────
    res = st.session_state.get("last_optimization")
    if res is None:
        st.info("Configure os targets e clique em **🚀 Rodar Otimização**. "
                "Os resultados aparecem aqui e ficam salvos enquanto você "
                "ajusta os controles.")
        return

    st.markdown("---")
    st.subheader(f"📈 Resultados — corner {res['corner_id']}")
    if res["corner_id"] != seed_corner_id:
        st.caption(f"⚠️ Resultados abaixo são do corner **{res['corner_id']}**, "
                   f"não do corner-seed selecionado ({seed_corner_id}).")

    seed_cost, final_cost = res["seed_cost"], res["cost"]
    improvement = (1.0 - final_cost / seed_cost) * 100.0 if seed_cost > 0 else 0.0
    n_ok  = sum(1 for r in res["opt_rows"] if r["ok"])
    n_tot = len(res["opt_rows"])

    rm = st.columns(5)
    rm[0].metric("Custo seed",  f"{seed_cost:.3e}", border=True)
    rm[1].metric("Custo final", f"{final_cost:.3e}",
                 delta=f"{-improvement:.1f}%", delta_color="inverse", border=True)
    rm[2].metric("Targets OK",  f"{n_ok}/{n_tot}", border=True)
    rm[3].metric("Gerações",    f"{res['nit']} ({res['nfev']} avals)", border=True)
    rm[4].metric("Tempo",       f"{res['elapsed_s']:.1f}s", border=True)

    t_targets, t_points, t_visual, t_conv = st.tabs([
        "✅ Targets", "📍 Hardpoints", "👁️ Comparação visual", "📉 Convergência",
    ])

    with t_targets:
        comparison = pl.DataFrame([
            {"Parâmetro": s["name"], "Target": s["target_str"],
             "Seed": s["obtained_str"], "Otimizado": o["obtained_str"],
             "Erro Seed": s["error_str"], "Erro Otimizado": o["error_str"],
             "OK Seed": "✅" if s["ok"] else "❌",
             "OK Otimizado": "✅" if o["ok"] else "❌"}
            for s, o in zip(res["seed_rows"], res["opt_rows"])
        ])
        st.dataframe(comparison, width="stretch", hide_index=True)

    with t_points:
        seed_pts = _movable_points(res["seed_corner"], res["seed_tie_rod"])
        opt_pts  = _movable_points(res["opt_corner"],  res["opt_tie_rod"])
        disp_rows = []
        for name, s in seed_pts.items():
            o = opt_pts[name]
            dx, dy, dz = o.x - s.x, o.y - s.y, o.z - s.z
            disp_rows.append({
                "Ponto":     name,
                "Seed":      f"({s.x:.1f}, {s.y:.1f}, {s.z:.1f})",
                "Otimizado": f"({o.x:.1f}, {o.y:.1f}, {o.z:.1f})",
                "Δx (mm)":   f"{dx:+.1f}",
                "Δy (mm)":   f"{dy:+.1f}",
                "Δz (mm)":   f"{dz:+.1f}",
                "|Δ| (mm)":  f"{math.sqrt(dx*dx + dy*dy + dz*dz):.1f}",
            })
        st.markdown("**Deslocamento dos pontos móveis**")
        st.dataframe(pl.DataFrame(disp_rows), width="stretch",
                     hide_index=True)

        opt_df = dataframe_from_corner(res["opt_corner"], res["opt_tie_rod"])
        with st.expander("Tabela completa de hardpoints otimizados"):
            st.dataframe(opt_df, width="stretch", hide_index=True)

        ac1, ac2, ac3 = st.columns([1.2, 1.2, 1.6])
        with ac1:
            st.download_button(
                "⬇️ Baixar CSV otimizado",
                data=opt_df.write_csv().encode(),
                file_name=f"hardpoints_optimized_{res['corner_id']}.csv",
                mime="text/csv", width="stretch",
            )
        with ac3:
            mirror = st.checkbox(
                f"Espelhar p/ {MIRROR_CORNER[res['corner_id']]} (Y → −Y)",
                value=True, key="synth_apply_mirror",
            )
        with ac2:
            if st.button("✅ Aplicar à sessão", type="primary",
                          width="stretch",
                          help="Substitui os hardpoints deste corner na sessão "
                               "— todas as abas passam a usar a geometria otimizada."):
                _apply_optimized_to_session(res, mirror)

    with t_visual:
        vc1, vc2 = st.columns(2)
        with vc1:
            st.plotly_chart(
                _overlay_2d_figure(res["seed_corner"], res["seed_tie_rod"],
                                   res["opt_corner"], res["opt_tie_rod"],
                                   res["bounds"], "YZ"),
                width="stretch", key="synth_overlay_yz")
        with vc2:
            st.plotly_chart(
                _overlay_2d_figure(res["seed_corner"], res["seed_tie_rod"],
                                   res["opt_corner"], res["opt_tie_rod"],
                                   res["bounds"], "XZ"),
                width="stretch", key="synth_overlay_xz")
        st.caption("Azul tracejado = seed · vermelho = otimizado · caixas "
                   "pontilhadas = região de busca (bounds) dos pontos móveis.")

    with t_conv:
        history = res.get("history") or []
        if len(history) < 2:
            st.info("Histórico de convergência indisponível "
                    "(otimização terminou em menos de 2 gerações).")
        else:
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=list(range(1, len(history) + 1)), y=history,
                mode="lines+markers", name="Melhor custo",
                line=dict(width=2, color="#d62728"),
            ))
            fig_conv.add_hline(y=seed_cost,
                               line=dict(color="gray", dash="dash", width=1),
                               annotation_text="custo do seed")
            fig_conv.update_layout(
                title="Convergência do differential evolution",
                xaxis_title="Geração", yaxis_title="Custo (escala log)",
                yaxis_type="log", template="plotly_white", height=420,
            )
            st.plotly_chart(fig_conv, width="stretch",
                            key="synth_convergence")
        st.caption(f"Solver: {res['message']}")
