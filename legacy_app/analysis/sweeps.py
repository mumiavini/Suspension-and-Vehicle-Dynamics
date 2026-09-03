"""
analysis/sweeps.py
==================
Solver-agnostic helpers for parametric sweep results (heave/roll/rack) of the
suspension kinematics.

Sweeps themselves are produced by `analysis.vdcore_bridge.vdcore_sweep`
(vdcore/DWSolver). This module only defines the shared result layout and
derives metrics/plots from it.

OUTPUT:
    np.ndarray with a structured dtype (defined in SWEEP_DTYPE). Access by
    column name: `sweep["camber_deg"]`, `sweep["heave_mm"]`, etc.

DERIVED METRICS:
    - camber_gain_per_mm  : rate of camber change with heave
    - bump_steer_per_mm   : rate of toe change with heave
    - rc_migration_range  : Roll Center migration amplitude

PLOTTING (Plotly):
    The `plot_*` functions return `plotly.graph_objects.Figure` objects for
    rendering in Streamlit or Jupyter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    # plotly is imported ONLY for type hints — at runtime it is a lazy import
    import plotly.graph_objects as go


# =============================================================================
# Structured dtype for the sweep results
# =============================================================================

SWEEP_DTYPE: np.dtype = np.dtype([
    # Applied inputs
    ("heave_mm",   "f8"),
    ("roll_deg",   "f8"),
    ("rack_mm",    "f8"),
    # Derived angles
    ("camber_deg", "f8"),
    ("toe_deg",    "f8"),
    ("caster_deg", "f8"),
    ("kpi_deg",    "f8"),
    # Roll Center position in the front view
    ("rc_y_mm",    "f8"),
    ("rc_z_mm",    "f8"),
    # Wheel-center position
    ("wc_x_mm",    "f8"),
    ("wc_y_mm",    "f8"),
    ("wc_z_mm",    "f8"),
    # Diagnostics
    ("residual",   "f8"),
    ("converged",  "?"),     # bool
])


# =============================================================================
# Derived metrics from the sweeps
# =============================================================================

def camber_gain_per_mm(sweep: NDArray) -> float:
    """
    Camber gain (°/mm) — linear regression of camber vs heave.

    For a heave sweep, returns the SLOPE of the line that best fits
    the points (camber_deg) as a function of (heave_mm).

    TYPICAL FSAE: −0.005 to −0.025 °/mm
    """
    mask = sweep["converged"]
    if mask.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(sweep["heave_mm"][mask], sweep["camber_deg"][mask], 1)
    return float(slope)


def bump_steer_per_mm(sweep: NDArray) -> float:
    """
    Bump steer (°/mm) — linear regression of toe vs heave.

    How much the wheel steers (involuntarily) during bump/rebound.
    Should be MINIMIZED (ideally < 0.005°/mm).
    """
    mask = sweep["converged"]
    if mask.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(sweep["heave_mm"][mask], sweep["toe_deg"][mask], 1)
    return float(slope)


def rc_migration_range(sweep: NDArray) -> tuple[float, float]:
    """
    Roll Center migration amplitude: (ΔY, ΔZ) in mm.

    ΔY = how much the RC migrates laterally
    ΔZ = how much the RC migrates vertically

    Ideally close to zero (stable RC).
    """
    mask = sweep["converged"]
    if mask.sum() < 2:
        return (float("nan"), float("nan"))
    dy = float(sweep["rc_y_mm"][mask].max() - sweep["rc_y_mm"][mask].min())
    dz = float(sweep["rc_z_mm"][mask].max() - sweep["rc_z_mm"][mask].min())
    return (dy, dz)


# =============================================================================
# Plotting (Plotly) — lazy imports so plotly is not required outside the plots
# =============================================================================

def plot_camber_vs_heave(sweep: NDArray, title: str = "Camber vs Heave") -> "go.Figure":
    """Chart: Camber (°) versus Heave (mm)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep["heave_mm"],
        y=sweep["camber_deg"],
        mode="lines+markers",
        name="Camber",
        line=dict(width=2, color="#1f77b4"),
        marker=dict(size=5),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
    fig.update_layout(
        title=title,
        xaxis_title="Heave (mm)  [+ = bump]",
        yaxis_title="Camber (°)  [− = top inward]",
        template="plotly_white",
        height=380,
        hovermode="x unified",
    )
    return fig


def plot_bump_steer(sweep: NDArray, title: str = "Bump Steer") -> "go.Figure":
    """
    Chart: toe variation (°) versus Heave (mm).

    NOTE: the solver already returns toe as a DELTA relative to the static
    state, so it can be plotted directly.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep["heave_mm"],
        y=sweep["toe_deg"],
        mode="lines+markers",
        name="Δ Toe",
        line=dict(width=2, color="darkorange"),
        marker=dict(size=5),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
    fig.update_layout(
        title=title,
        xaxis_title="Heave (mm)",
        yaxis_title="Δ Toe (°)  [+ = toe-in]",
        template="plotly_white",
        height=380,
        hovermode="x unified",
    )
    return fig


def plot_rc_migration(sweep: NDArray, title: str = "Roll Center Migration") -> "go.Figure":
    """
    Chart: Roll Center trajectory in the Y-Z plane, colored by heave.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep["rc_y_mm"],
        y=sweep["rc_z_mm"],
        mode="lines+markers",
        marker=dict(
            size=6,
            color=sweep["heave_mm"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Heave (mm)"),
        ),
        line=dict(width=1, color="gray"),
        name="RC trajectory",
    ))
    # Mark the static position (heave closest to 0)
    i_static = int(np.argmin(np.abs(sweep["heave_mm"])))
    fig.add_trace(go.Scatter(
        x=[sweep["rc_y_mm"][i_static]],
        y=[sweep["rc_z_mm"][i_static]],
        mode="markers",
        marker=dict(size=12, symbol="star", color="#d62728",
                    line=dict(width=1, color="white")),
        name="Static position",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", line_width=1)
    fig.update_layout(
        title=title,
        xaxis_title="RC Y (mm)",
        yaxis_title="RC Z (mm)",
        template="plotly_white",
        height=380,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def plot_caster_kpi_vs_steer(
    sweep: NDArray,
    title: str = "Caster & KPI vs Steer",
) -> "go.Figure":
    """Chart: Caster and KPI (°) versus rack displacement (mm)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep["rack_mm"], y=sweep["caster_deg"],
        mode="lines+markers", name="Caster",
        line=dict(width=2, color="#1f77b4"),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=sweep["rack_mm"], y=sweep["kpi_deg"],
        mode="lines+markers", name="KPI",
        line=dict(width=2, color="#d62728", dash="dash"),
        marker=dict(size=5),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Rack (mm)",
        yaxis_title="Angle (°)",
        template="plotly_white",
        height=380,
        hovermode="x unified",
    )
    return fig
