"""Camber trade surface: consequences of each (roll gradient, static camber) pair.

This is the central table of the project.  It shows explicitly that stiffer
roll needs less camber gain, which allows a longer FVSA, which reduces RC
migration and jacking — and what each of those costs in tire force.

The tool computes consequences.  The designer reads the table and picks a row.

Coordinate system: ISO 8855 — X+ forward, Y+ left, Z+ up.
Camber sign: negative = top of wheel inboard (both sides).

Geometry approximations (stated, not hidden):
  - Roll-to-camber coupling is 1:1 at zero camber gain (body roll directly
    adds to inclination angle).
  - Camber gain ≈ −1/FVSA  (first order, radians/mm → FVSA in mm).
  - RC height from FVSA + assumed IC height via line-from-CP-through-IC
    geometry.  The assumed IC height is a parameter with a default typical
    of FSAE DW front suspension.
  - Wheel travel in roll = roll_angle_rad × half_track.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from vdcore.tire.metrics import BinMetrics, CamberSensitivity


def _mean_camber_sensitivity(
    camber_sensitivity: list[CamberSensitivity],
) -> float:
    """Mean dFy/dIA across all bins, in N/deg."""
    if not camber_sensitivity:
        return 0.0
    return sum(cs.dfy_dia_n_per_deg for cs in camber_sensitivity) / len(camber_sensitivity)


def _mean_fy_at_peak(
    bin_metrics: list[BinMetrics],
) -> float:
    """Mean peak lateral force across bins (Fy = mu * Fz), in N."""
    if not bin_metrics:
        return 0.0
    return sum(b.peak_mu_lateral * b.fz_nominal_n for b in bin_metrics) / len(bin_metrics)


def camber_trade_surface(
    bin_metrics: list[BinMetrics],
    camber_sensitivity: list[CamberSensitivity],
    *,
    roll_gradient_range_deg_per_g: tuple[float, float],
    roll_gradient_steps: int,
    static_camber_range_deg: tuple[float, float],
    static_camber_steps: int,
    target_ay_g: float,
    front_track_mm: float,
    tire_best_ia_deg: float,
    assumed_ic_height_mm: float = 250.0,
) -> pl.DataFrame:
    """Compute the camber trade surface.

    For each (roll_gradient, static_camber) pair in the supplied ranges,
    computes the consequences for the outside-front tire at peak cornering.

    Parameters
    ----------
    bin_metrics:
        Tire bin metrics for peak Fy reference.
    camber_sensitivity:
        Camber sensitivity data from tire metrics.
    roll_gradient_range_deg_per_g:
        (min, max) roll gradient in deg/g to sweep.
    roll_gradient_steps:
        Number of roll gradient values in the sweep.
    static_camber_range_deg:
        (min, max) static camber in deg to sweep.
    static_camber_steps:
        Number of static camber values in the sweep.
    target_ay_g:
        Target lateral acceleration in g (from achievable_ay or designer).
    front_track_mm:
        Front track width in mm.
    tire_best_ia_deg:
        Inclination angle at which the tire produces maximum Fy.
        Read from the tire data — the IA at which peak_mu_lateral
        is highest across IA bins.
    assumed_ic_height_mm:
        Assumed instant-centre height in mm for RC height estimation.
        Default 250 mm is typical for an FSAE double-wishbone front.
        RC height estimates are only as good as this assumption.

    Returns
    -------
    pl.DataFrame
        One row per (roll_gradient, static_camber) combination.

    Assumptions
    -----------
    - Roll-to-camber coupling is 1:1 at zero camber gain
    - Camber gain ≈ −1/FVSA (first order)
    - Wheel travel in roll = roll_angle_rad × half_track
    - RC height from FVSA + assumed IC height (first-order geometry)
    """
    rg_values = np.linspace(
        roll_gradient_range_deg_per_g[0],
        roll_gradient_range_deg_per_g[1],
        roll_gradient_steps,
    )
    sc_values = np.linspace(
        static_camber_range_deg[0],
        static_camber_range_deg[1],
        static_camber_steps,
    )

    half_track_mm = front_track_mm / 2.0
    dfy_dia = _mean_camber_sensitivity(camber_sensitivity)
    fy_peak = _mean_fy_at_peak(bin_metrics)

    rows: list[dict[str, float]] = []

    for rg in rg_values:
        for sc in sc_values:
            roll_angle_deg = rg * target_ay_g

            # Outside tire IA with zero camber compensation from geometry.
            # In a left turn (positive Ay), the body rolls right (positive
            # roll angle).  The outside (right) tire's top tilts further
            # outboard → IA becomes less negative (more positive).
            # outside_ia = static_camber + roll_angle.
            outside_ia_no_gain = sc + roll_angle_deg

            # Fy penalty of doing nothing (no camber compensation).
            ia_error_no_gain = outside_ia_no_gain - tire_best_ia_deg
            if fy_peak > 0 and dfy_dia != 0.0:
                fy_loss_no_gain_n = abs(ia_error_no_gain * dfy_dia)
                fy_penalty_no_gain_pct = fy_loss_no_gain_n / fy_peak * 100.0
            else:
                fy_penalty_no_gain_pct = 0.0

            # Wheel travel at the outside corner in roll (bump direction).
            roll_angle_rad = math.radians(roll_angle_deg)
            wheel_travel_mm = roll_angle_rad * half_track_mm

            # Required camber gain to hold tire at best IA.
            delta_camber_needed = tire_best_ia_deg - sc - roll_angle_deg

            if abs(wheel_travel_mm) > 1e-6:
                required_gain_deg_per_mm = delta_camber_needed / wheel_travel_mm
            else:
                required_gain_deg_per_mm = 0.0

            # Implied FVSA from required camber gain.
            # camber_gain_rad_per_mm = required_gain_deg_per_mm * pi/180
            # fvsa_mm = -1 / camber_gain_rad_per_mm
            gain_rad_per_mm = math.radians(required_gain_deg_per_mm)
            if abs(gain_rad_per_mm) > 1e-10:
                implied_fvsa_mm = -1.0 / gain_rad_per_mm
            else:
                implied_fvsa_mm = float("inf")

            # Implied RC height from FVSA + assumed IC height.
            # Line from CP at (half_track, 0) through IC at
            # (half_track + fvsa, ic_z) crosses Y=0 at:
            # rc_z = -half_track * ic_z / fvsa
            if abs(implied_fvsa_mm) > 1e-3 and math.isfinite(implied_fvsa_mm):
                implied_rc_height_mm = -half_track_mm * assumed_ic_height_mm / implied_fvsa_mm
            else:
                implied_rc_height_mm = 0.0

            # RC height sensitivity to bump (first-order).
            # d(rc_h)/d(wt) ≈ rc_h / fvsa  (geometric sensitivity).
            if abs(implied_fvsa_mm) > 1e-3 and math.isfinite(implied_fvsa_mm):
                rc_sensitivity = implied_rc_height_mm / implied_fvsa_mm
            else:
                rc_sensitivity = 0.0

            # Reference Fy at best IA (from tire data).
            fy_at_best = fy_peak

            rows.append(
                {
                    "roll_gradient_deg_per_g": float(rg),
                    "static_camber_deg": float(sc),
                    "roll_angle_at_target_ay_deg": roll_angle_deg,
                    "outside_ia_no_gain_deg": outside_ia_no_gain,
                    "fy_penalty_no_gain_pct": fy_penalty_no_gain_pct,
                    "required_camber_gain_deg_per_mm": required_gain_deg_per_mm,
                    "implied_fvsa_mm": implied_fvsa_mm,
                    "implied_rc_height_mm": implied_rc_height_mm,
                    "rc_height_sensitivity_mm_per_mm_bump": rc_sensitivity,
                    "fy_at_best_ia_n": fy_at_best,
                }
            )

    return pl.DataFrame(rows)
