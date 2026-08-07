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
from analysis.sweeps import SweepRunner
from geometry import KinematicSolver3D


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


def _geometry_signature(corner, tie_rod) -> tuple:
    """Hashable tuple with all hardpoints — cache key for sweeps."""
    pts = (
        corner.upper_arm.inboard_front, corner.upper_arm.inboard_rear,
        corner.upper_arm.outboard,
        corner.lower_arm.inboard_front, corner.lower_arm.inboard_rear,
        corner.lower_arm.outboard,
        tie_rod.inboard, tie_rod.outboard,
        corner.wheel_center, corner.contact_patch,
    )
    return tuple((p.x, p.y, p.z) for p in pts)


@st.cache_data(show_spinner=False, max_entries=128)
def _sweep_cache(geom_sig: tuple, sweep_type: str, params: tuple,
                 _corner=None, _tie_rod=None):
    # geom_sig identifies the geometry in the cache; _corner/_tie_rod (prefix "_"
    # = not hashed by Streamlit) are only used on a cache miss.
    solver = KinematicSolver3D(_corner, _tie_rod)
    runner = SweepRunner(solver=solver)
    if sweep_type == "Heave":
        return runner.heave_sweep(*params)
    elif sweep_type == "Roll":
        return runner.roll_sweep(*params)
    else:
        return runner.steer_sweep(*params)


def run_sweep_cached(corner, tie_rod, sweep_type, params):
    return _sweep_cache(
        _geometry_signature(corner, tie_rod), sweep_type,
        tuple(float(p) for p in params),
        _corner=corner, _tie_rod=tie_rod,
    )
