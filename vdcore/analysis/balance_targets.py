"""Balance trade table: LLTD versus predicted vehicle balance.

Using front and rear load sensitivity from the tire metrics, computes
front/rear axle lateral force capacity as a function of LLTD.  Returns a
table the designer reads to choose a roll-stiffness split.

The tool computes consequences.  The designer picks the LLTD.

Assumptions (stated, not hidden):
  - Steady-state cornering (no transients)
  - No aerodynamic downforce
  - No transient weight transfer (quasi-static only)
  - No suspension compliance
  - Tire model is binned raw data with linear interpolation between FZ bins
  - Same tire front and rear

Coordinate system: ISO 8855 — X+ forward, Y+ left, Z+ up.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from vdcore.analysis.load_transfer import lateral_load_transfer
from vdcore.analysis.operating_envelope import mu_at_fz
from vdcore.models.mass import MassProperties, UnsprungMassSet
from vdcore.tire.metrics import BinMetrics

_G = 9.81


def balance_trade_table(
    mass: MassProperties,
    unsprung: UnsprungMassSet,
    bin_metrics: list[BinMetrics],
    *,
    ay_g: float,
    front_rc_height_mm: float,
    rear_rc_height_mm: float,
    front_track_mm: float,
    rear_track_mm: float,
    wheelbase_mm: float,
    total_roll_stiffness_nm_per_deg: float,
    lltd_range: tuple[float, float] = (0.3, 0.7),
    lltd_steps: int = 21,
    ia_nominal_deg: float | None = None,
    p_nominal_kpa: float | None = None,
) -> pl.DataFrame:
    """Compute the balance trade table: LLTD versus vehicle balance.

    Sweeps LLTD by varying the front/rear roll-stiffness split at a
    fixed total stiffness.  For each LLTD, computes per-corner Fz from
    load transfer, looks up μ from the tire bins, and determines which
    axle limits.

    Parameters
    ----------
    mass:
        Vehicle mass properties.
    unsprung:
        Per-corner unsprung masses.
    bin_metrics:
        Tire bin metrics (same tire all four corners).
    ay_g:
        Lateral acceleration to evaluate at (g).
    front_rc_height_mm, rear_rc_height_mm:
        Roll-centre heights (mm).
    front_track_mm, rear_track_mm:
        Track widths (mm).
    wheelbase_mm:
        Wheelbase (mm).
    total_roll_stiffness_nm_per_deg:
        Total roll stiffness (Nm/deg).  Held constant; the split
        varies with LLTD.
    lltd_range:
        (min, max) LLTD to sweep.
    lltd_steps:
        Number of LLTD values.
    ia_nominal_deg, p_nominal_kpa:
        Optional filters for bin lookup.

    Returns
    -------
    pl.DataFrame
        One row per LLTD value with balance assessment.
    """
    lltd_values = np.linspace(lltd_range[0], lltd_range[1], lltd_steps)

    m_total = mass.total_mass_kg.value
    fmf = mass.front_mass_fraction.value
    w_total = m_total * _G

    static_front_per_corner = w_total * fmf / 2.0
    static_rear_per_corner = w_total * (1.0 - fmf) / 2.0

    rows: list[dict[str, object]] = []

    for lltd in lltd_values:
        k_front = float(lltd) * total_roll_stiffness_nm_per_deg
        k_rear = (1.0 - float(lltd)) * total_roll_stiffness_nm_per_deg

        lt = lateral_load_transfer(
            mass,
            unsprung,
            ay_g=ay_g,
            front_rc_height_mm=front_rc_height_mm,
            rear_rc_height_mm=rear_rc_height_mm,
            front_track_mm=front_track_mm,
            rear_track_mm=rear_track_mm,
            front_roll_stiffness_nm_per_deg=k_front,
            rear_roll_stiffness_nm_per_deg=k_rear,
            wheelbase_mm=wheelbase_mm,
        )

        front_delta = lt.front.total_delta_fz_n
        rear_delta = lt.rear.total_delta_fz_n

        # Positive ay → right side loaded (outside).
        fl_fz = max(static_front_per_corner - front_delta, 0.0)
        fr_fz = max(static_front_per_corner + front_delta, 0.0)
        rl_fz = max(static_rear_per_corner - rear_delta, 0.0)
        rr_fz = max(static_rear_per_corner + rear_delta, 0.0)

        fl_mu, fl_clamp = mu_at_fz(
            fl_fz,
            bin_metrics,
            ia_nominal_deg=ia_nominal_deg,
            p_nominal_kpa=p_nominal_kpa,
        )
        fr_mu, fr_clamp = mu_at_fz(
            fr_fz,
            bin_metrics,
            ia_nominal_deg=ia_nominal_deg,
            p_nominal_kpa=p_nominal_kpa,
        )
        rl_mu, rl_clamp = mu_at_fz(
            rl_fz,
            bin_metrics,
            ia_nominal_deg=ia_nominal_deg,
            p_nominal_kpa=p_nominal_kpa,
        )
        rr_mu, rr_clamp = mu_at_fz(
            rr_fz,
            bin_metrics,
            ia_nominal_deg=ia_nominal_deg,
            p_nominal_kpa=p_nominal_kpa,
        )

        front_fy = fl_mu * fl_fz + fr_mu * fr_fz
        rear_fy = rl_mu * rl_fz + rr_mu * rr_fz

        margin_n = abs(front_fy - rear_fy)
        ref_fy = min(front_fy, rear_fy)
        margin_pct = (margin_n / ref_fy * 100.0) if ref_fy > 1e-6 else 0.0

        if margin_pct < 1.0:
            balance = "neutral"
        elif front_fy < rear_fy:
            balance = "front_limited"
        else:
            balance = "rear_limited"

        any_clamped = fl_clamp or fr_clamp or rl_clamp or rr_clamp

        rows.append(
            {
                "lltd": float(lltd),
                "front_roll_stiffness_nm_per_deg": k_front,
                "rear_roll_stiffness_nm_per_deg": k_rear,
                "front_fy_n": front_fy,
                "rear_fy_n": rear_fy,
                "balance": balance,
                "margin_n": margin_n,
                "margin_pct": margin_pct,
                "front_roll_stiffness_fraction": float(lltd),
                "fz_was_clamped": any_clamped,
            }
        )

    return pl.DataFrame(rows)
