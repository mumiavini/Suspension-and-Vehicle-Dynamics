"""
analysis/sweeps.py
==================
Parametric sweeps of the suspension kinematics.

A SWEEP is a sequence of configurations (heave, roll, rack) run in order,
where each point uses the previous one as the solver seed — this guarantees
physical continuity of the motion.

SWEEP TYPES:
    - Heave Sweep : varies heave, with roll=0 and rack=0
    - Roll Sweep  : varies roll,  with heave=0 and rack=0
    - Steer Sweep : varies rack,  with heave=0 and roll=0

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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from geometry.solver_3d import KinematicSolver3D, KinematicState3D

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
# SweepRunner — Sweep executor
# =============================================================================

@dataclass
class SweepRunner:
    """
    Runs parametric sweeps using a KinematicSolver3D.

    Usage:
        runner = SweepRunner(solver=my_solver)
        heave_data = runner.heave_sweep(-25.0, 25.0, 1.0)
        roll_data  = runner.roll_sweep(-3.0, 3.0, 0.2)
        steer_data = runner.steer_sweep(-30.0, 30.0, 1.0)

    Attributes:
        solver       : already-initialized 3D solver
        static_state : static state (solved on demand and cached —
                       avoids 1 extra solve per SweepRunner when nobody uses it,
                       which matters in the optimizer, which creates one per
                       evaluation)
    """
    solver:        KinematicSolver3D
    _static_cache: Optional[KinematicState3D] = field(default=None, repr=False)

    @property
    def static_state(self) -> KinematicState3D:
        """Static state (heave=roll=rack=0), computed on first read."""
        if self._static_cache is None:
            self.solver.reset_seed()
            self._static_cache = self.solver.solve(0.0, 0.0, 0.0)
        return self._static_cache

    # -------------------------------------------------------------------------
    # Standard sweeps
    # -------------------------------------------------------------------------

    def heave_sweep(
        self,
        heave_min_mm: float = -25.0,
        heave_max_mm: float =  25.0,
        step_mm:      float =   1.0,
    ) -> NDArray:
        """
        Pure heave sweep (bump/rebound), with roll=0 and rack=0.
        """
        if step_mm <= 0:
            raise ValueError(f"step_mm must be positive, got: {step_mm}")
        if heave_max_mm < heave_min_mm:
            raise ValueError(
                f"heave_max ({heave_max_mm}) must be >= heave_min ({heave_min_mm})"
            )
        values = np.arange(heave_min_mm, heave_max_mm + step_mm * 0.5, step_mm)
        configurations = [(float(h), 0.0, 0.0) for h in values]
        return self._run_sweep(configurations)

    def roll_sweep(
        self,
        roll_min_deg: float = -3.0,
        roll_max_deg: float =  3.0,
        step_deg:     float =  0.1,
    ) -> NDArray:
        """
        Chassis roll sweep, with heave=0 and rack=0.
        """
        if step_deg <= 0:
            raise ValueError(f"step_deg must be positive, got: {step_deg}")
        if roll_max_deg < roll_min_deg:
            raise ValueError(
                f"roll_max ({roll_max_deg}) must be >= roll_min ({roll_min_deg})"
            )
        values = np.arange(roll_min_deg, roll_max_deg + step_deg * 0.5, step_deg)
        configurations = [(0.0, float(r), 0.0) for r in values]
        return self._run_sweep(configurations)

    def steer_sweep(
        self,
        rack_min_mm: float = -30.0,
        rack_max_mm: float =  30.0,
        step_mm:     float =   1.0,
    ) -> NDArray:
        """
        Steering sweep (rack displacement), with heave=0 and roll=0.
        """
        if step_mm <= 0:
            raise ValueError(f"step_mm must be positive, got: {step_mm}")
        if rack_max_mm < rack_min_mm:
            raise ValueError(
                f"rack_max ({rack_max_mm}) must be >= rack_min ({rack_min_mm})"
            )
        values = np.arange(rack_min_mm, rack_max_mm + step_mm * 0.5, step_mm)
        configurations = [(0.0, 0.0, float(r)) for r in values]
        return self._run_sweep(configurations)

    def combined_sweep(
        self,
        configurations: list[tuple[float, float, float]],
    ) -> NDArray:
        """
        Arbitrary sweep. Use for combinations such as simultaneous heave+roll.

        IMPORTANT: order the configurations so that adjacent points are close
        in phase space (the solver uses the previous one as the seed).
        """
        return self._run_sweep(configurations)

    # -------------------------------------------------------------------------
    # Main execution loop
    # -------------------------------------------------------------------------

    def _run_sweep(
        self,
        configurations: list[tuple[float, float, float]],
    ) -> NDArray:
        """
        Run the solver for each configuration and fill the result array.
        """
        n = len(configurations)
        result = np.empty(n, dtype=SWEEP_DTYPE)

        # Reset the seed: the sweep starts from the static position
        self.solver.reset_seed()

        for i, (heave, roll, rack) in enumerate(configurations):
            try:
                state = self.solver.solve(heave_mm=heave, roll_deg=roll, rack_mm=rack)
                self._fill_record(result, i, state)
            except Exception:
                # Solver failed: mark the record as non-converged
                result[i] = self._make_failed_record(heave, roll, rack)

        return result

    # -------------------------------------------------------------------------
    # Fill an array record with the state data
    # -------------------------------------------------------------------------

    def _fill_record(
        self,
        arr:   NDArray,
        idx:   int,
        state: KinematicState3D,
    ) -> None:
        """Copy the KinematicState3D fields into the array record."""
        rc_y, rc_z = self._estimate_roll_center_yz(state)

        arr[idx]["heave_mm"]   = state.heave_mm
        arr[idx]["roll_deg"]   = state.roll_deg
        arr[idx]["rack_mm"]    = state.rack_mm
        arr[idx]["camber_deg"] = state.camber_deg
        arr[idx]["toe_deg"]    = state.toe_deg
        arr[idx]["caster_deg"] = state.caster_deg
        arr[idx]["kpi_deg"]    = state.kpi_deg
        arr[idx]["rc_y_mm"]    = rc_y
        arr[idx]["rc_z_mm"]    = rc_z
        arr[idx]["wc_x_mm"]    = state.wheel_center.x
        arr[idx]["wc_y_mm"]    = state.wheel_center.y
        arr[idx]["wc_z_mm"]    = state.wheel_center.z
        arr[idx]["residual"]   = state.residual_norm
        arr[idx]["converged"]  = state.converged

    @staticmethod
    def _make_failed_record(heave: float, roll: float, rack: float) -> NDArray:
        """Default record for a solver failure."""
        rec = np.zeros(1, dtype=SWEEP_DTYPE)[0]
        rec["heave_mm"]  = heave
        rec["roll_deg"]  = roll
        rec["rack_mm"]   = rack
        rec["residual"]  = np.nan
        rec["converged"] = False
        return rec

    # -------------------------------------------------------------------------
    # Roll Center estimate from the 3D state
    # -------------------------------------------------------------------------

    def _estimate_roll_center_yz(
        self,
        state: KinematicState3D,
    ) -> tuple[float, float]:
        """
        Estimate the Roll Center in the front Y-Z view using the standard 2D
        method (Instant Center → line to contact patch → symmetry plane).

        IMPORTANT: the vehicle's "symmetry plane" rotates with the chassis
        during roll. To compute the RC correctly under roll, we need to:

            1. Find the IC in the WORLD frame
            2. Find where the IC→CP line crosses the CHASSIS symmetry plane
               (which is the vertical plane at Y=0 of the rolled chassis frame)
            3. The result is the RC position in the world (in Y and Z)

        For roll=0, the chassis plane is the world's Y=0 plane (previous
        behavior). For roll≠0, the plane is tilted.
        """
        import math
        from geometry.primitives import Point2D, line_intersection_2d

        corner = self.solver.corner
        uca_in_eff = corner.upper_arm.effective_inboard
        lca_in_eff = corner.lower_arm.effective_inboard

        # Apply the chassis motion to the effective inboards
        uca_in_arr = self.solver._move_chassis_point(
            uca_in_eff, state.heave_mm, state.roll_deg
        )
        lca_in_arr = self.solver._move_chassis_point(
            lca_in_eff, state.heave_mm, state.roll_deg
        )

        # Projections onto the Y-Z plane (u=Y, v=Z)
        uca_in_2d  = Point2D(float(uca_in_arr[1]),  float(uca_in_arr[2]))
        uca_out_2d = Point2D(state.uca_outboard.y,  state.uca_outboard.z)
        lca_in_2d  = Point2D(float(lca_in_arr[1]),  float(lca_in_arr[2]))
        lca_out_2d = Point2D(state.lca_outboard.y,  state.lca_outboard.z)

        # Instant Center: intersection of the extended arm lines
        try:
            ic = line_intersection_2d(lca_in_2d, lca_out_2d, uca_in_2d, uca_out_2d)
        except ValueError:
            return (0.0, 0.0)   # parallel arms: RC at ground level

        cp = Point2D(state.contact_patch.y, state.contact_patch.z)

        # Vector of the IC→CP line in the YZ plane
        du = cp.u - ic.u
        dv = cp.v - ic.v

        if abs(du) < 1e-12 and abs(dv) < 1e-12:
            return (float(ic.u), float(ic.v))

        # ─── CHASSIS symmetry plane in the YZ plane ──────────────────────────
        # At roll=0: vertical plane Y=0 (line Y=0, any dz)
        # At roll>0 (chassis rolls to the right = chassis top toward −Y):
        #   the plane is a line through the chassis origin with direction
        #   rotated by -roll about X.
        roll_rad = math.radians(state.roll_deg)
        # Vector of the "chassis vertical" in the YZ plane (Z rotated by -roll)
        # Chassis origin assumed at (Y=0, Z=0)
        cy = -math.sin(roll_rad)   # Y component
        cz =  math.cos(roll_rad)   # Z component

        # Intersection of the parametric line IC + t*(CP-IC) with the parametric
        # line of the chassis plane: (0,0) + s*(cy, cz)
        # System:
        #   ic.u + t*du = s*cy
        #   ic.v + t*dv = s*cz
        # → t*du - s*cy = -ic.u
        #   t*dv - s*cz = -ic.v
        det = du * (-cz) - dv * (-cy)   # det = -du*cz + dv*cy
        if abs(det) < 1e-12:
            return (0.0, float(ic.v))

        t = ((-ic.u) * (-cz) - (-ic.v) * (-cy)) / det
        rc_y = ic.u + t * du
        rc_z = ic.v + t * dv
        return (float(rc_y), float(rc_z))


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
        line=dict(width=2),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Heave (mm)  [+ = bump]",
        yaxis_title="Camber (°)  [− = top inward]",
        template="plotly_white",
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
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Heave (mm)",
        yaxis_title="Δ Toe (°)  [+ = toe-in]",
        template="plotly_white",
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
    fig.update_layout(
        title=title,
        xaxis_title="RC Y (mm)",
        yaxis_title="RC Z (mm)",
        template="plotly_white",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)   # isometric axes
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
        mode="lines+markers", name="Caster (°)",
        line=dict(width=2),
    ))
    fig.add_trace(go.Scatter(
        x=sweep["rack_mm"], y=sweep["kpi_deg"],
        mode="lines+markers", name="KPI (°)",
        line=dict(width=2, dash="dash"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Rack (mm)",
        yaxis_title="Angle (°)",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig
