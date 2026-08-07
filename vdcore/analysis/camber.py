"""Camber analysis — static camber, camber gain, and wheel-travel sweeps.

Coordinate system: ISO 8855 — X+ forward, Y+ LEFT, Z+ up.
Camber sign: negative = top of wheel inboard (both sides).
Wheel travel: positive = bump (wheel up relative to chassis).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel

from vdcore.geometry.solver import DWSolver, SolverResult
from vdcore.models.hardpoint import Corner


class CamberSweepResult(BaseModel, frozen=True):
    """Result of a camber vs wheel-travel sweep."""

    wheel_travel_mm: list[float]
    camber_deg: list[float]
    converged: list[bool]
    corner_id: Literal["FL", "FR", "RL", "RR"]


def static_camber_deg(corner: Corner) -> float:
    """Static camber from hardpoint geometry (zero wheel travel/roll/rack).

    ISO 8855: negative = top of wheel inboard (both sides).
    Handles left/right corner sign difference internally via the solver.
    """
    solver = DWSolver(corner)
    result = solver.solve(wheel_travel_mm=0.0, roll_deg=0.0, rack_mm=0.0)
    if not result.converged:
        raise RuntimeError(
            f"Solver did not converge for static position of corner {corner.corner_id}. "
            f"Residual norm: {result.residual_norm:.2e}"
        )
    return result.camber_deg


def camber_gain_deg_per_mm(
    corner: Corner,
    wheel_travel_range_mm: float = 25.0,
    steps: int = 50,
) -> float:
    """Linear camber gain: d(camber_deg) / d(wheel_travel_mm).

    Computed via least-squares regression over a symmetric wheel-travel sweep.
    wheel_travel_mm is positive in bump (wheel up relative to chassis).

    For a typical double wishbone with UCA shorter than LCA, this value
    is negative: bump produces more negative camber (top of wheel moves
    further inboard).
    """
    travel_vals = np.linspace(-wheel_travel_range_mm, wheel_travel_range_mm, steps)
    solver = DWSolver(corner)

    travel_good: list[float] = []
    camber_good: list[float] = []

    for wt in travel_vals:
        result = solver.solve(wheel_travel_mm=float(wt))
        if result.converged:
            travel_good.append(float(wt))
            camber_good.append(result.camber_deg)

    if len(travel_good) < 2:
        raise RuntimeError(
            f"Too few converged points ({len(travel_good)}) to compute camber gain "
            f"for corner {corner.corner_id}"
        )

    coeffs = np.polyfit(travel_good, camber_good, 1)
    return float(coeffs[0])


def camber_sweep(
    corner: Corner,
    wheel_travel_min_mm: float = -25.0,
    wheel_travel_max_mm: float = 25.0,
    steps: int = 50,
) -> CamberSweepResult:
    """Full camber vs wheel-travel sweep with convergence info.

    wheel_travel_mm is positive in bump (wheel up relative to chassis).
    Non-converged points have camber set to NaN.
    """
    travel_vals = np.linspace(wheel_travel_min_mm, wheel_travel_max_mm, steps)
    solver = DWSolver(corner)

    travel_list: list[float] = []
    camber_list: list[float] = []
    conv_list: list[bool] = []

    for wt in travel_vals:
        result = solver.solve(wheel_travel_mm=float(wt))
        travel_list.append(float(wt))
        conv_list.append(result.converged)
        camber_list.append(result.camber_deg if result.converged else float("nan"))

    return CamberSweepResult(
        wheel_travel_mm=travel_list,
        camber_deg=camber_list,
        converged=conv_list,
        corner_id=corner.corner_id,
    )
