"""
ui/tab_synthesis.py
===================
🎯 Synthesis tab — global hardpoint optimization from targets
(reverse engineering), via `analysis.optimizer`. The results are stored in
`st.session_state["last_optimization"]`, also consumed by the Comparison tab.
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
    """The 4 hardpoints the optimizer is allowed to move."""
    return {
        "UCA_OUT":     corner.upper_arm.outboard,
        "LCA_OUT":     corner.lower_arm.outboard,
        "TIE_ROD_IN":  tie_rod.inboard,
        "TIE_ROD_OUT": tie_rod.outboard,
    }


def _overlay_2d_figure(seed_corner, seed_tr, opt_corner, opt_tr,
                       bounds: dict, view: str) -> go.Figure:
    """2D overlay of seed × optimized, with the search boxes (bounds)."""

    def uv(p):
        return (p.y, p.z) if view == "YZ" else (p.x, p.z)

    def add_geometry(fig, corner, tr, color, name, dash=None):
        ua, la = corner.upper_arm, corner.lower_arm
        segments = [
            (ua.inboard_front, ua.outboard), (ua.inboard_rear, ua.outboard),
            (la.inboard_front, la.outboard), (la.inboard_rear, la.outboard),
            (tr.inboard, tr.outboard),
            (ua.outboard, la.outboard),                    # upright
            (corner.wheel_center, corner.contact_patch),   # wheel
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

    # Search boxes of the movable points
    for b in bounds.values():
        if view == "YZ":
            u0, u1, v0, v1 = b.y_min, b.y_max, b.z_min, b.z_max
        else:
            u0, u1, v0, v1 = b.x_min, b.x_max, b.z_min, b.z_max
        fig.add_shape(type="rect", x0=u0, x1=u1, y0=v0, y1=v1,
                      line=dict(color="rgba(120,120,120,0.45)", width=1, dash="dot"),
                      fillcolor="rgba(120,120,120,0.06)")

    add_geometry(fig, seed_corner, seed_tr, "#1f77b4", "Seed", dash="dash")
    add_geometry(fig, opt_corner, opt_tr, "#d62728", "Optimized")
    fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=1))

    axis_u = "Y" if view == "YZ" else "X"
    fig.update_layout(
        title=f"{view} view",
        xaxis_title=f"{axis_u} (mm)", yaxis_title="Z (mm)",
        template="plotly_white", height=420,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _apply_optimized_to_session(res: dict, mirror: bool) -> None:
    """Replace, in the session df, the hardpoints of the optimized corner
    (and of the mirrored opposite corner, if requested) and re-run the app."""
    cid = res["corner_id"]
    base = st.session_state.get("hardpoints_df")
    if base is None:
        st.error("❌ No hardpoint loaded in the session.")
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
    label = f"Optimized ({' + '.join(replaced)})"
    st.session_state["hardpoints_df"] = pl.concat(pieces, how="vertical_relaxed")
    st.session_state["hardpoints_source"] = label
    st.toast(f"Hardpoints applied: {label}", icon="✅")
    st.rerun(scope="app")


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
            "Synthesis starts from an existing geometry (the **seed**) and moves "
            "hardpoints automatically to reach the defined targets.",
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
        st.caption("**Current seed values** — use as a reference when defining the targets")
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
        """Toggle + numeric value on the same line."""
        c_use, c_val = st.columns([1.3, 1], vertical_alignment="center")
        use = c_use.toggle(label, value=on_default, key=use_key)
        val = c_val.number_input(label, value=default, step=step,
                                 disabled=not use, key=val_key,
                                 label_visibility="collapsed", format=fmt)
        return use, val

    col_static, col_dynamic = st.columns(2)

    with col_static.container(border=True):
        st.markdown("**Static** — enable the ones the optimizer should pursue")

        use_caster, tgt_caster = _target_row("Caster (°)", "use_caster",
                                             "tgt_caster", 4.0, 0.5, True)
        use_kpi, tgt_kpi       = _target_row("KPI (°)", "use_kpi",
                                             "tgt_kpi", 7.0, 0.5, True)
        use_camber, tgt_camber = _target_row("Static camber (°)", "use_camber",
                                             "tgt_camber", -1.5, 0.25, True)
        use_scrub, tgt_scrub   = _target_row("Scrub (mm)", "use_scrub",
                                             "tgt_scrub", 15.0, 1.0, False)
        use_trail, tgt_trail   = _target_row("Trail (mm)", "use_trail",
                                             "tgt_trail", 20.0, 1.0, False)

    with col_dynamic.container(border=True):
        st.markdown("**Dynamic** — disabling zeroes the term's weight in the cost")

        use_cg, tgt_cg   = _target_row("Camber Gain (°/mm)", "use_cg",
                                       "tgt_cg", -0.020, 0.005, True, fmt="%.3f")
        use_bs, tgt_bs   = _target_row("Max Bump Steer (°/mm)", "use_bs",
                                       "tgt_bs", 0.005, 0.001, True, fmt="%.3f")
        use_rch, tgt_rch = _target_row("RC Height (mm)", "use_rch",
                                       "tgt_rch", 45.0, 1.0, True)
        use_rcm, tgt_rcm = _target_row("Max RC ΔY (mm)", "use_rcm",
                                       "tgt_rcm", 25.0, 1.0, True)
        st.markdown("**Heave sweep range**")
        hc1, hc2, hc3 = st.columns(3)
        with hc1: opt_h_min  = st.number_input("min", value=-25.0, key="opt_hmin")
        with hc2: opt_h_max  = st.number_input("max", value= 25.0, key="opt_hmax")
        with hc3: opt_h_step = st.number_input("step", value= 5.0, key="opt_hstep")

    exp_w, exp_b, exp_s = st.columns(3)
    with exp_w:
        with st.expander("⚙️ Weights"):
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
            pop_size = st.slider("Population (×n_dims)", 5, 30, 12, key="pop")
            max_iter = st.slider("Iterations", 10, 200, 40, key="iter")
            seed_rng = st.number_input("seed", value=42, key="seed_rng")
            workers  = st.selectbox("Cores", [1, -1],
                                      format_func=lambda x: "1" if x == 1 else "All",
                                      key="workers")
            polish_opt = st.checkbox("Polish (local refinement at the end)", value=True,
                                      key="polish_opt",
                                      help="Runs L-BFGS-B after the DE. Improves the "
                                           "result, but adds extra evaluations.")
            max_seconds = st.number_input("Max time (s) — 0 = no limit",
                                           min_value=0, value=0, step=10,
                                           key="max_seconds",
                                           help="Stops when exceeded, keeping the "
                                                "best result so far.")

    st.markdown("---")
    run_col, clear_col, info_col = st.columns([1, 1, 2])
    with run_col:
        run_opt = st.button("🚀 Run Optimization", type="primary",
                             width="stretch")
    with clear_col:
        if st.session_state.get("last_optimization") is not None:
            if st.button("🗑️ Clear results", width="stretch"):
                st.session_state.pop("last_optimization", None)
                st.rerun(scope="fragment")
    with info_col:
        n_evals = pop_size * 12 * (max_iter + 1)
        st.caption(f"≈ up to {n_evals:,} objective-function evaluations "
                   f"({pop_size}×12 individuals × {max_iter} generations)")

    # ── Execution ────────────────────────────────────────────────────────────
    if run_opt:
        if not any([use_caster, use_kpi, use_camber, use_scrub, use_trail,
                    use_cg, use_bs, use_rch, use_rcm]):
            st.error("❌ Enable at least one target (static or dynamic) — "
                     "with all of them off the objective function is constant and the "
                     "optimizer has nothing to pursue.")
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
            # Toggle off ⇒ weight 0 ⇒ term drops out of the objective function
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

        progress = st.progress(0.0, text="Initializing population…")
        t0 = time.monotonic()

        def on_generation(gen: int, best_cost: float, convergence: float) -> bool:
            elapsed = time.monotonic() - t0
            progress.progress(
                min(gen / max_iter, 1.0),
                text=f"Generation {gen}/{max_iter} · best cost {best_cost:.3e} "
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
            st.error(f"❌ Optimization failed: {exc}")
            return
        elapsed = time.monotonic() - t0
        progress.empty()

        opt_validation = validate_against_targets(
            result.optimal_corner, result.optimal_tie_rod, targets,
        )

        st.session_state["last_optimization"] = {
            # consumed by the 🔄 Comparison tab
            "seed_corner": seed_corner, "seed_tie_rod": seed_tie_rod,
            "opt_corner": result.optimal_corner, "opt_tie_rod": result.optimal_tie_rod,
            "targets": targets, "corner_id": seed_corner_id,
            # consumed by the results section below
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
        st.toast(f"Optimization finished in {elapsed:.1f}s", icon="🏁")

    # ── Results (persist across interactions — outside the button block) ─────
    res = st.session_state.get("last_optimization")
    if res is None:
        st.info("Configure the targets and click **🚀 Run Optimization**. "
                "The results appear here and stay saved while you "
                "adjust the controls.")
        return

    st.markdown("---")
    st.subheader(f"📈 Results — corner {res['corner_id']}")
    if res["corner_id"] != seed_corner_id:
        st.caption(f"⚠️ The results below are for corner **{res['corner_id']}**, "
                   f"not the selected seed corner ({seed_corner_id}).")

    seed_cost, final_cost = res["seed_cost"], res["cost"]
    improvement = (1.0 - final_cost / seed_cost) * 100.0 if seed_cost > 0 else 0.0
    n_ok  = sum(1 for r in res["opt_rows"] if r["ok"])
    n_tot = len(res["opt_rows"])

    rm = st.columns(5)
    rm[0].metric("Seed cost",   f"{seed_cost:.3e}", border=True)
    rm[1].metric("Final cost",  f"{final_cost:.3e}",
                 delta=f"{-improvement:.1f}%", delta_color="inverse", border=True)
    rm[2].metric("Targets OK",  f"{n_ok}/{n_tot}", border=True)
    rm[3].metric("Generations", f"{res['nit']} ({res['nfev']} evals)", border=True)
    rm[4].metric("Time",        f"{res['elapsed_s']:.1f}s", border=True)

    t_targets, t_points, t_visual, t_conv = st.tabs([
        "✅ Targets", "📍 Hardpoints", "👁️ Visual comparison", "📉 Convergence",
    ])

    with t_targets:
        comparison = pl.DataFrame([
            {"Parameter": s["name"], "Target": s["target_str"],
             "Seed": s["obtained_str"], "Optimized": o["obtained_str"],
             "Seed Error": s["error_str"], "Optimized Error": o["error_str"],
             "Seed OK": "✅" if s["ok"] else "❌",
             "Optimized OK": "✅" if o["ok"] else "❌"}
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
                "Point":     name,
                "Seed":      f"({s.x:.1f}, {s.y:.1f}, {s.z:.1f})",
                "Optimized": f"({o.x:.1f}, {o.y:.1f}, {o.z:.1f})",
                "Δx (mm)":   f"{dx:+.1f}",
                "Δy (mm)":   f"{dy:+.1f}",
                "Δz (mm)":   f"{dz:+.1f}",
                "|Δ| (mm)":  f"{math.sqrt(dx*dx + dy*dy + dz*dz):.1f}",
            })
        st.markdown("**Displacement of the movable points**")
        st.dataframe(pl.DataFrame(disp_rows), width="stretch",
                     hide_index=True)

        opt_df = dataframe_from_corner(res["opt_corner"], res["opt_tie_rod"])
        with st.expander("Full table of optimized hardpoints"):
            st.dataframe(opt_df, width="stretch", hide_index=True)

        ac1, ac2, ac3 = st.columns([1.2, 1.2, 1.6])
        with ac1:
            st.download_button(
                "⬇️ Download optimized CSV",
                data=opt_df.write_csv().encode(),
                file_name=f"hardpoints_optimized_{res['corner_id']}.csv",
                mime="text/csv", width="stretch",
            )
        with ac3:
            mirror = st.checkbox(
                f"Mirror to {MIRROR_CORNER[res['corner_id']]} (Y → −Y)",
                value=True, key="synth_apply_mirror",
            )
        with ac2:
            if st.button("✅ Apply to session", type="primary",
                          width="stretch",
                          help="Replaces this corner's hardpoints in the session "
                               "— all tabs then use the optimized geometry."):
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
        st.caption("Dashed blue = seed · red = optimized · dotted "
                   "boxes = search region (bounds) of the movable points.")

    with t_conv:
        history = res.get("history") or []
        if len(history) < 2:
            st.info("Convergence history unavailable "
                    "(optimization finished in fewer than 2 generations).")
        else:
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=list(range(1, len(history) + 1)), y=history,
                mode="lines+markers", name="Best cost",
                line=dict(width=2, color="#d62728"),
            ))
            fig_conv.add_hline(y=seed_cost,
                               line=dict(color="gray", dash="dash", width=1),
                               annotation_text="seed cost")
            fig_conv.update_layout(
                title="Differential evolution convergence",
                xaxis_title="Generation", yaxis_title="Cost (log scale)",
                yaxis_type="log", template="plotly_white", height=420,
            )
            st.plotly_chart(fig_conv, width="stretch",
                            key="synth_convergence")
        st.caption(f"Solver: {res['message']}")
