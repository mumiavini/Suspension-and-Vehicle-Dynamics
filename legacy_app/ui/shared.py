"""
ui/shared.py
============
Helpers shared across the tabs: access to the session hardpoints, the standard
empty-state, safe construction of corners/vehicle and the sweep cache.
"""

from __future__ import annotations

from typing import Optional

import polars as pl
import streamlit as st

from analysis.io_hardpoints import (
    build_corner_from_dataframe,
    build_vehicle_from_dataframe,
    generate_template_dataframe,
    HardpointValidationError,
)

# Front brake bias used for anti-dive. MUST track
# ``sla_geometry.VEHICLE_2027.brake_bias_front`` -- the app used to default to
# 0.60 against a design value of 0.65, so the anti-dive it showed (6.91 %)
# disagreed with the published summary (7.50 %) purely through this constant.
# tests/unit/test_app_agrees_with_summary.py fails if the two drift apart
# again; sla_geometry is not imported here because legacy_app/ is frozen and
# must not gain a dependency on the design scripts.
DESIGN_BRAKE_BIAS_FRONT = 0.65


def load_hardpoints_from_state() -> Optional[pl.DataFrame]:
    return st.session_state.get("hardpoints_df", None)


def load_demo_into_session() -> None:
    st.session_state["hardpoints_df"] = generate_template_dataframe()
    st.session_state["hardpoints_source"] = "Demo template"


def render_empty_state(message: str, key: str) -> None:
    """Standard call-to-action shown when there are no hardpoints in the session."""
    with st.container(border=True):
        st.markdown("#### 📂 No geometry loaded")
        st.markdown(message)
        c1, c2 = st.columns([1, 2], vertical_alignment="center")
        with c1:
            if st.button("🏎️ Load demo geometry", type="primary",
                          key=key, width="stretch"):
                load_demo_into_session()
                st.rerun(scope="app")
        with c2:
            st.caption("Or load your file (.xlsx / .csv / .json) "
                       "in the **sidebar** ⬅️")


def build_corner_safe(df, corner_id):
    try:
        return build_corner_from_dataframe(df, corner_id)
    except HardpointValidationError as exc:
        st.error(f"❌ Error in corner '{corner_id}': {exc}")
        return None


def build_vehicle_safe(df):
    try:
        return build_vehicle_from_dataframe(df)
    except HardpointValidationError as exc:
        st.warning(f"⚠️ Incomplete vehicle: {exc}")
        return None, None
