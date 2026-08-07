"""Roll gradient: body roll per unit lateral acceleration.

Assumptions (each matters more than any hardpoint tolerance):
  1. Rigid chassis — no torsional flex distributing moment
     between axles. Real FSAE space frames have finite torsional
     stiffness; this model overestimates roll if the chassis is
     soft relative to the suspension.
  2. Small angles — sin(φ) ≈ φ, cos(φ) ≈ 1. Valid below ~5 deg,
     which covers normal FSAE operation.
  3. Roll axis fixed at static position — the axis actually
     migrates with roll, but the effect is second-order at
     small angles.
  4. Tire vertical stiffness neglected — real tires add
     compliance that increases roll. This means the predicted
     roll gradient is a lower bound.

Coordinate system: ISO 8855 — X+ forward, Y+ left, Z+ up.
"""

from __future__ import annotations

import math

from vdcore.analysis.load_transfer import sprung_mass_kg
from vdcore.explain import Explained
from vdcore.models.mass import MassProperties, UnsprungMassSet


def roll_gradient_deg_per_g(
    mass: MassProperties,
    unsprung: UnsprungMassSet,
    *,
    front_rc_height_mm: float,
    rear_rc_height_mm: float,
    front_roll_stiffness_nm_per_deg: float,
    rear_roll_stiffness_nm_per_deg: float,
) -> float:
    """Body roll angle per g of lateral acceleration.

    Roll gradient φ/Ay = (m_s · g · d) / (K_roll - m_s · g · d)

    where:
      m_s = sprung mass (kg)
      g = 9.81 m/s²
      d = distance from sprung-mass CG to roll axis (m)
      K_roll = total roll stiffness (Nm/rad)

    The denominator subtracts the gravity-induced roll moment
    (pendulum effect). If K_roll < m_s·g·d the car would roll
    over — this function raises ValueError in that case.

    Assumptions: rigid chassis, small angle, roll axis fixed at
    static, tire vertical stiffness neglected. See module docstring.

    Parameters
    ----------
    mass:
        Vehicle mass properties.
    unsprung:
        Per-corner unsprung masses.
    front_rc_height_mm, rear_rc_height_mm:
        Roll centre heights from solved kinematic state (mm).
    front_roll_stiffness_nm_per_deg, rear_roll_stiffness_nm_per_deg:
        Axle roll stiffness including springs and ARB (Nm/deg).

    Returns
    -------
    float
        Roll gradient in deg/g. Positive = body rolls away from
        the turn (normal behaviour).
    """
    g = 9.81

    m_s = sprung_mass_kg(mass, unsprung)
    fmf = mass.front_mass_fraction.value

    ra_height_mm = front_rc_height_mm * (1.0 - fmf) + rear_rc_height_mm * fmf

    h_cg_mm = mass.cg_height_mm.value
    d_mm = h_cg_mm - ra_height_mm
    d_m = d_mm / 1000.0

    k_front_nm_per_rad = front_roll_stiffness_nm_per_deg * (180.0 / math.pi)
    k_rear_nm_per_rad = rear_roll_stiffness_nm_per_deg * (180.0 / math.pi)
    k_total_nm_per_rad = k_front_nm_per_rad + k_rear_nm_per_rad

    gravity_moment = m_s * g * d_m

    effective_stiffness = k_total_nm_per_rad - gravity_moment
    if effective_stiffness <= 0:
        raise ValueError(
            f"Roll stiffness ({k_total_nm_per_rad:.1f} Nm/rad) is less than "
            f"gravity roll moment ({gravity_moment:.1f} Nm/rad). "
            f"The vehicle would roll over."
        )

    roll_rad_per_ms2 = (m_s * g * d_m) / effective_stiffness
    roll_deg_per_g = math.degrees(roll_rad_per_ms2 * g)

    return roll_deg_per_g


def roll_gradient_deg_per_g_explained(
    mass: MassProperties,
    unsprung: UnsprungMassSet,
    *,
    front_rc_height_mm: float,
    rear_rc_height_mm: float,
    front_roll_stiffness_nm_per_deg: float,
    rear_roll_stiffness_nm_per_deg: float,
) -> Explained[float]:
    """Roll gradient with full derivation audit trail.

    Calls :func:`roll_gradient_deg_per_g` internally and wraps the
    result in an :class:`Explained` object.  See that function for
    the physics and assumptions.
    """
    value = roll_gradient_deg_per_g(
        mass,
        unsprung,
        front_rc_height_mm=front_rc_height_mm,
        rear_rc_height_mm=rear_rc_height_mm,
        front_roll_stiffness_nm_per_deg=front_roll_stiffness_nm_per_deg,
        rear_roll_stiffness_nm_per_deg=rear_roll_stiffness_nm_per_deg,
    )

    g = 9.81
    m_s = sprung_mass_kg(mass, unsprung)
    fmf = mass.front_mass_fraction.value

    ra_height_mm = front_rc_height_mm * (1.0 - fmf) + rear_rc_height_mm * fmf
    d_mm = mass.cg_height_mm.value - ra_height_mm
    d_m = d_mm / 1000.0

    k_total_nm_per_rad = (front_roll_stiffness_nm_per_deg + rear_roll_stiffness_nm_per_deg) * (
        180.0 / math.pi
    )
    gravity_moment = m_s * g * d_m

    def _src(field_name: str) -> str:
        pf = getattr(mass, field_name, None)
        if pf is not None:
            return str(pf.source)
        return "computed"

    return Explained(
        value=value,
        formula="phi_per_g = (m_s * g * d) / (K_roll - m_s * g * d)",
        inputs={
            "m_s": (m_s, "kg", "computed"),
            "g": (g, "m/s^2", "computed"),
            "d": (d_m, "m", "computed"),
            "K_roll": (k_total_nm_per_rad, "Nm/rad", "computed"),
            "cg_height_mm": (
                mass.cg_height_mm.value,
                "mm",
                _src("cg_height_mm"),
            ),
            "front_rc_height_mm": (front_rc_height_mm, "mm", "computed"),
            "rear_rc_height_mm": (rear_rc_height_mm, "mm", "computed"),
            "front_roll_stiffness_nm_per_deg": (
                front_roll_stiffness_nm_per_deg,
                "Nm/deg",
                "computed",
            ),
            "rear_roll_stiffness_nm_per_deg": (
                rear_roll_stiffness_nm_per_deg,
                "Nm/deg",
                "computed",
            ),
        },
        intermediates={
            "sprung_mass_kg": m_s,
            "roll_axis_height_mm": ra_height_mm,
            "sprung_cg_above_ra_mm": d_mm,
            "k_total_nm_per_rad": k_total_nm_per_rad,
            "gravity_moment_nm_per_rad": gravity_moment,
            "effective_stiffness_nm_per_rad": k_total_nm_per_rad - gravity_moment,
        },
        reference="Milliken RCVD Ch. 16 — roll gradient with pendulum correction",
        assumptions=[
            "Rigid chassis — no torsional flex",
            "Small angles — sin(phi) ~ phi, cos(phi) ~ 1",
            "Roll axis fixed at static position",
            "Tire vertical stiffness neglected (lower bound on roll)",
        ],
    )
