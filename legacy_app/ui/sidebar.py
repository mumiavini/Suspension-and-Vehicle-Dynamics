"""
ui/sidebar.py
=============
App sidebar: hardpoint loading (upload / demo / template), session clearing,
vehicle setup and the theme selector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl
import streamlit as st

from analysis.io_hardpoints import (
    read_hardpoints,
    generate_template_dataframe,
    HardpointValidationError,
    VALID_CORNERS,
)
from ui.shared import load_demo_into_session
from ui.theme import THEMES


def _save_uploaded_to_tmp(uploaded_file) -> Path:
    """
    Save a Streamlit upload to a temporary directory and return the path.

    Uses `tempfile.gettempdir()` for cross-OS portability:
        - Linux/macOS : /tmp
        - Windows     : C:\\Users\\<user>\\AppData\\Local\\Temp
    """
    import tempfile
    suffix = Path(uploaded_file.name).suffix
    tmp_path = Path(tempfile.gettempdir()) / f"_fsae_upload{suffix}"
    tmp_path.write_bytes(uploaded_file.read())
    return tmp_path


def render_sidebar() -> None:
    with st.sidebar:
        # ─── SESSION STATUS (always visible at the top) ──────────────────────
        if "hardpoints_df" in st.session_state:
            st.success(f"In use: **{st.session_state.get('hardpoints_source', '?')}**",
                       icon="📊")
        else:
            st.warning("No hardpoints loaded", icon="⚠️")

        st.markdown("### 1️⃣ Load data")

        # ─── STEP 1: UPLOAD (only stores, does NOT apply yet) ────────────────
        uploaded = st.file_uploader(
            "Hardpoints file",
            type=["xlsx", "csv", "json"],
            help="Columns: corner, point, x_mm, y_mm, z_mm",
        )

        # Parse the file just for preview/validation, but do not apply yet
        pending_df: Optional[pl.DataFrame] = None
        pending_error: Optional[str] = None

        if uploaded is not None:
            try:
                tmp = _save_uploaded_to_tmp(uploaded)
                pending_df = read_hardpoints(tmp)
            except HardpointValidationError as exc:
                pending_error = f"Validation: {exc}"
            except Exception as exc:
                pending_error = str(exc)

        # ─── STEP 2: PREVIEW + APPLY BUTTON ──────────────────────────────────
        if pending_error is not None:
            st.error(f"❌ {pending_error}")
        elif pending_df is not None:
            # Show a mini-preview of what was loaded
            n_rows  = pending_df.height
            corners = sorted(pending_df["corner"].unique().to_list())
            st.success(f"✅ '{uploaded.name}' — {n_rows} points · corners: {', '.join(corners)}")

            # Explicit button that applies the file (recomputes everything)
            if st.button("🔄 **Apply file**", type="primary", width="stretch",
                          help="Loads this file into the app and recomputes all KPIs and charts"):
                st.session_state["hardpoints_df"]     = pending_df
                st.session_state["hardpoints_source"] = uploaded.name
                st.rerun()   # force an immediate re-render with the new file

        # ─── DEMO + TEMPLATE (shortcuts) ─────────────────────────────────────
        st.caption("Or use a shortcut:")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📋 Demo", width="stretch",
                          help="Loads a realistic example FSAE geometry"):
                load_demo_into_session()
                st.rerun()

        with col_b:
            template_df = generate_template_dataframe()
            st.download_button("⬇️ Template", data=template_df.write_csv().encode(),
                                file_name="hardpoints_template.csv", mime="text/csv",
                                width="stretch",
                                help="Downloads the CSV template for manual editing")

        if "hardpoints_df" in st.session_state:
            if st.button("🗑️ Clear session", width="stretch",
                          help="Removes the current file from the session"):
                for key in ["hardpoints_df", "hardpoints_source", "last_optimization",
                            "manual_hardpoints", "manual_synced_source"]:
                    st.session_state.pop(key, None)
                # Also clear the data_editor keys
                for cid in VALID_CORNERS:
                    st.session_state.pop(f"editor_{cid}", None)
                st.rerun()

        st.divider()
        st.markdown("### 2️⃣ Vehicle setup")
        st.session_state.setdefault("vehicle_setup", {
            "brake_bias": 0.60,
            "c_factor_mm": 100.0,
            "steering_wheel_lock_deg": 270.0,
        })
        vs = st.session_state["vehicle_setup"]
        vs["brake_bias"] = st.slider("Brake bias front", 0.0, 1.0, vs["brake_bias"],
                                       step=0.05, help="Fraction at the front")
        with st.expander("🔧 Steering"):
            vs["c_factor_mm"] = st.number_input("c-factor (mm/rev)",
                                                  value=vs["c_factor_mm"], step=1.0,
                                                  help="2π × pinion radius")
            vs["steering_wheel_lock_deg"] = st.number_input("Total steering-wheel lock (°)",
                                                              value=vs["steering_wheel_lock_deg"],
                                                              step=10.0)

        st.divider()

        # ─── THEME ───────────────────────────────────────────────────────────
        st.selectbox("🎨 App theme", list(THEMES), key="ui_theme",
                     help="Applies to this session; the boot default comes from "
                          ".streamlit/config.toml")
        # The changed config only reaches the browser on the NEXT rerun, so we
        # force one when the choice changes (standard set_option + rerun pattern).
        if st.session_state["_theme_applied"] != st.session_state["ui_theme"]:
            st.session_state["_theme_applied"] = st.session_state["ui_theme"]
            st.rerun()
