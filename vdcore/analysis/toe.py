"""Toe analysis — bump steer and toe vs wheel-travel sweeps.

Coordinate system: ISO 8855 — X+ forward, Y+ LEFT, Z+ up.
Toe sign: positive = toe-IN (both sides).
Wheel travel: positive = bump (wheel up relative to chassis).

PER-SIDE VS TOTAL
    Every quantity here is named for which one it is. ``toe_deg_per_side`` is
    one wheel; ``total_toe_deg`` is the pair, i.e. twice the per-side value for
    a symmetric axle. This has caused confusion on the real car, so nothing in
    this module reports a bare "toe".

BUMP STEER IS NOT ONE NUMBER
    The linear rate through ride height and the peak excursion over travel are
    different quantities, and a geometry can be excellent at one and poor at the
    other. A tie rod whose linear term is nulled still has a quadratic term, so
    the toe curve is a parabola about ride height: both bump and droop steer the
    same way. ``BumpSteerResult`` carries both, because reporting only the
    linear rate would call that geometry "zero bump steer".
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel

from vdcore.geometry.solver import DWSolver
from vdcore.models.hardpoint import Corner


class ToeSweepResult(BaseModel, frozen=True):
    """Result of a toe vs wheel-travel sweep, referenced to ride height."""

    wheel_travel_mm: list[float]
    toe_deg_per_side: list[float]
    converged: list[bool]
    corner_id: Literal["FL", "FR", "RL", "RR"]

    @property
    def total_toe_deg(self) -> list[float]:
        """Axle total toe, assuming the opposite corner is a mirror image."""
        return [2.0 * t for t in self.toe_deg_per_side]


class BumpSteerResult(BaseModel, frozen=True):
    """Bump steer summarised as a linear rate plus a peak excursion."""

    corner_id: Literal["FL", "FR", "RL", "RR"]
    linear_deg_per_mm_per_side: float
    peak_abs_deg_per_side: float
    toe_at_full_bump_deg_per_side: float
    toe_at_full_droop_deg_per_side: float
    travel_range_mm: float

    @property
    def peak_abs_total_toe_deg(self) -> float:
        return 2.0 * self.peak_abs_deg_per_side


def _solve_toe(
    corner: Corner, travel_vals: np.ndarray
) -> tuple[list[float], list[float], list[bool]]:
    """Solve toe at each travel, referenced to the ride-height value.

    The reference subtraction is what makes this BUMP STEER rather than toe:
    static toe is a design variable set by the tie rod length (or shims), and
    is not what the linkage does over travel.
    """
    solver = DWSolver(corner)

    static = solver.solve(wheel_travel_mm=0.0)
    if not static.converged:
        raise RuntimeError(
            f"Solver did not converge at ride height for corner {corner.corner_id}. "
            f"Residual norm: {static.residual_norm:.2e}"
        )
    reference = static.toe_deg_per_side

    travel_list: list[float] = []
    toe_list: list[float] = []
    conv_list: list[bool] = []

    for wt in travel_vals:
        result = solver.solve(wheel_travel_mm=float(wt))
        travel_list.append(float(wt))
        conv_list.append(result.converged)
        toe_list.append(
            result.toe_deg_per_side - reference if result.converged else float("nan")
        )

    return travel_list, toe_list, conv_list


def toe_sweep(
    corner: Corner,
    wheel_travel_min_mm: float = -25.0,
    wheel_travel_max_mm: float = 25.0,
    steps: int = 50,
) -> ToeSweepResult:
    """Full toe vs wheel-travel sweep, referenced to ride height.

    Non-converged points have toe set to NaN.
    """
    travel_vals = np.linspace(wheel_travel_min_mm, wheel_travel_max_mm, steps)
    travel_list, toe_list, conv_list = _solve_toe(corner, travel_vals)

    return ToeSweepResult(
        wheel_travel_mm=travel_list,
        toe_deg_per_side=toe_list,
        converged=conv_list,
        corner_id=corner.corner_id,
    )


def bump_steer(
    corner: Corner,
    wheel_travel_range_mm: float = 25.0,
    steps: int = 21,
) -> BumpSteerResult:
    """Bump steer over a symmetric wheel-travel sweep.

    The linear rate is a least-squares fit over the whole sweep, matching how
    ``camber_gain_deg_per_mm`` is defined so the two read the same way. The
    peak is the largest absolute departure from the ride-height value, which is
    the number that actually bounds toe change on track.

    Positive linear rate = the wheel toes IN as it moves into bump.
    """
    if steps < 3:
        raise ValueError(f"steps must be at least 3 to fit a rate, got {steps}")

    travel_vals = np.linspace(-wheel_travel_range_mm, wheel_travel_range_mm, steps)
    travel_list, toe_list, conv_list = _solve_toe(corner, travel_vals)

    good = [
        (t, v)
        for t, v, c in zip(travel_list, toe_list, conv_list, strict=True)
        if c
    ]
    if len(good) < 3:
        raise RuntimeError(
            f"Too few converged points ({len(good)}) to compute bump steer "
            f"for corner {corner.corner_id}"
        )

    travels = np.array([t for t, _ in good])
    toes = np.array([v for _, v in good])
    slope = float(np.polyfit(travels, toes, 1)[0])

    return BumpSteerResult(
        corner_id=corner.corner_id,
        linear_deg_per_mm_per_side=slope,
        peak_abs_deg_per_side=float(np.max(np.abs(toes))),
        toe_at_full_bump_deg_per_side=float(toes[-1]),
        toe_at_full_droop_deg_per_side=float(toes[0]),
        travel_range_mm=wheel_travel_range_mm,
    )
