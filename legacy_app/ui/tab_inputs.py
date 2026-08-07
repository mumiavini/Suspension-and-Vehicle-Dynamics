"""
ui/tab_inputs.py
================
✏️ Inputs tab — manual hardpoint editor with 2D visualization in the
YZ (front), XZ (side) and XY (top) views.
"""

from __future__ import annotations

import polars as pl
import streamlit as st
import plotly.graph_objects as go

from analysis.io_hardpoints import (
    generate_template_dataframe,
    HardpointValidationError,
    VALID_CORNERS,
    REQUIRED_POINTS_PER_CORNER,
)
from ui.shared import load_hardpoints_from_state


def render() -> None:
    st.header("Hardpoint editor with visualization")
    st.markdown(
        "Enter/edit the hardpoints manually and see the projections in "
        "**YZ (front)**, **XZ (side)** and **XY (top)** in real time. "
        "Useful for creating geometries from scratch or inspecting them visually."
    )

    # Initialization / synchronization of the manual state
    #
    # The editor keeps its own dict `manual_hardpoints[corner][point] = (x,y,z)`.
    # So that the sidebar's "Apply file" button also updates the editor, we
    # track which source was loaded last. If it changed, we reload.
    def _empty_corner_dict():
        return {p: (0.0, 0.0, 0.0) for p in REQUIRED_POINTS_PER_CORNER}

    def _load_manual_from_df(df_loaded: pl.DataFrame) -> dict:
        """Convert the hardpoints DataFrame into the editor's dict format."""
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

    # Re-sync if: (a) there is no manual state yet, or
    # (b) the source changed since the last sync
    needs_sync = ("manual_hardpoints" not in st.session_state) or \
                 (current_source != last_synced)

    if needs_sync:
        df_loaded = load_hardpoints_from_state()
        if df_loaded is not None:
            st.session_state["manual_hardpoints"] = _load_manual_from_df(df_loaded)
        else:
            st.session_state["manual_hardpoints"] = {
                cid: _empty_corner_dict() for cid in VALID_CORNERS
            }
        st.session_state["manual_synced_source"] = current_source

    # Explicit button to reload from the current file at any time
    if current_source is not None:
        bcol1, bcol2 = st.columns([1, 3])
        with bcol1:
            if st.button("🔁 Reload from file",
                          help=f"Discards manual edits and reloads from the current file ({current_source})",
                          width="stretch"):
                df_loaded = load_hardpoints_from_state()
                if df_loaded is not None:
                    st.session_state["manual_hardpoints"] = _load_manual_from_df(df_loaded)
                    # Clear the data_editor keys to force a redraw
                    for cid in VALID_CORNERS:
                        st.session_state.pop(f"editor_{cid}", None)
                    st.rerun()
        with bcol2:
            st.caption(f"📊 Synced with: **{current_source}**")

    # Top controls
    ctrl1, ctrl2, ctrl3 = st.columns([1.4, 1.4, 1.8], vertical_alignment="bottom")
    with ctrl1:
        edit_corner = st.segmented_control("Corner being edited", VALID_CORNERS,
                                           default=VALID_CORNERS[0],
                                           key="edit_corner")
        if edit_corner is None:  # clicking the selected item deselects it
            edit_corner = VALID_CORNERS[0]
    with ctrl2:
        if st.button("📋 Load template into this corner",
                      width="stretch"):
            tmpl = generate_template_dataframe()
            sub = tmpl.filter(pl.col("corner") == edit_corner)
            pts = {row["point"]: (float(row["x_mm"]),
                                   float(row["y_mm"]),
                                   float(row["z_mm"]))
                   for row in sub.iter_rows(named=True)}
            st.session_state["manual_hardpoints"][edit_corner] = pts
            st.rerun()
    with ctrl3:
        if st.button("🪞 Mirror Left → Right (Y → −Y)",
                      width="stretch",
                      help="FL → FR and RL → RR inverting Y"):
            for left, right in [("FL", "FR"), ("RL", "RR")]:
                left_pts = st.session_state["manual_hardpoints"][left]
                st.session_state["manual_hardpoints"][right] = {
                    p: (x, -y, z) for p, (x, y, z) in left_pts.items()
                }
            st.rerun()

    st.markdown("---")

    # Layout: table on the left, plots on the right
    col_inputs, col_plots = st.columns([1, 1])

    # LEFT COLUMN: editable table
    with col_inputs:
        st.subheader(f"Coordinates — corner **{edit_corner}**")

        current_pts = st.session_state["manual_hardpoints"][edit_corner]
        edit_data = [
            {"point": p, "x_mm": current_pts.get(p, (0,0,0))[0],
                          "y_mm": current_pts.get(p, (0,0,0))[1],
                          "z_mm": current_pts.get(p, (0,0,0))[2]}
            for p in REQUIRED_POINTS_PER_CORNER
        ]

        edited = st.data_editor(
            edit_data, num_rows="fixed", hide_index=True,
            width="stretch", disabled=["point"],
            column_config={
                "point": st.column_config.TextColumn("Point", width="medium"),
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

        st.markdown("#### Actions")
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("✅ Apply as loaded hardpoints",
                          type="primary", width="stretch"):
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
                    st.session_state["hardpoints_source"] = "Manual inputs"
                    # CRITICAL: also update synced_source, otherwise the next
                    # rerun would discard the manual data thinking the source
                    # changed.
                    st.session_state["manual_synced_source"] = "Manual inputs"
                    st.success("✅ Applied! Go to '📊 Analysis'.")
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
                "⬇️ Download all (CSV)",
                data=df_out.write_csv().encode(),
                file_name="hardpoints_manual.csv",
                mime="text/csv", width="stretch",
            )

    # RIGHT COLUMN: 3 2D views
    with col_plots:
        st.subheader("2D views")

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
            ("UCA_OUT",      "LCA_OUT",     "UCA"),  # upright
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

            # Connection lines
            for p1, p2, group in CONNECTIONS:
                u1, v1, _, _ = coords(p1)
                u2, v2, _, _ = coords(p2)
                color = GROUP_COLORS.get(group, "#888")
                fig.add_trace(go.Scatter(
                    x=[u1, u2], y=[v1, v2], mode="lines",
                    line=dict(color=color, width=2), showlegend=False,
                    hoverinfo="skip",
                ))

            # Points with labels
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

            # Ground line (YZ and XZ)
            if view in ("YZ", "XZ"):
                fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=1))

            # Symmetry plane (only in XY if Y=0 is visible)
            if view == "XY":
                fig.add_hline(y=0, line=dict(color="lightgray", dash="dot", width=1))

            _, _, lab_u, lab_v = coords(REQUIRED_POINTS_PER_CORNER[0])
            fig.update_layout(
                title=f"{view} view — {edit_corner}",
                xaxis_title=f"{lab_u} (mm)",
                yaxis_title=f"{lab_v} (mm)",
                template="plotly_white", height=460,
                margin=dict(l=40, r=20, t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                             xanchor="right", x=1),
            )
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            return fig

        view_tabs = st.tabs(["⬅️ Front (YZ)", "↔️ Side (XZ)", "⬆️ Top (XY)"])
        with view_tabs[0]:
            st.plotly_chart(make_2d_view("YZ"), width="stretch",
                             key=f"yz_{edit_corner}")
        with view_tabs[1]:
            st.plotly_chart(make_2d_view("XZ"), width="stretch",
                             key=f"xz_{edit_corner}")
        with view_tabs[2]:
            st.plotly_chart(make_2d_view("XY"), width="stretch",
                             key=f"xy_{edit_corner}")
