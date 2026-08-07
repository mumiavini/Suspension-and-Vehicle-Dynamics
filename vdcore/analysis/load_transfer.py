"""Quasi-static load transfer, decomposed by mechanism.

Coordinate system: ISO 8855 — X+ forward, Y+ left, Z+ up.

The decomposition makes geometry decisions visible:
  1. **Geometric (jacking)** — through the roll centre. Depends on
     RC height, which comes from the solved kinematic state.
  2. **Elastic** — through springs and ARB, distributed by roll
     stiffness front-to-rear.
  3. **Unsprung** — direct, proportional to unsprung mass and its
     CG height.

The total lateral load transfer for one axle at lateral acceleration
Ay is:

    ΔFz = m · Ay · h_cg / t

regardless of how it splits between mechanisms. This is the conservation
check: geometric + elastic + unsprung must sum to the total.

Assumptions:
  - Quasi-static (no transients, dampers irrelevant)
  - Small angles (sin φ ≈ φ, cos φ ≈ 1)
  - Roll axis fixed at static position
  - Rigid chassis (no torsional flex)
  - Tire vertical stiffness neglected (ride height change ignored)
"""

from __future__ import annotations

from dataclasses import dataclass

from vdcore.models.mass import MassProperties, UnsprungMassSet


@dataclass(frozen=True)
class LateralLoadTransferResult:
    """Per-axle lateral load transfer at a given Ay.

    All ΔFz values are in Newtons. Positive ΔFz means load transferred
    to the outside wheel (away from the turn centre).

    The three components sum to total_delta_fz_n (conservation).
    """

    axle: str
    ay_g: float

    geometric_delta_fz_n: float
    elastic_delta_fz_n: float
    unsprung_delta_fz_n: float
    total_delta_fz_n: float

    estimate_inputs: list[str]


@dataclass(frozen=True)
class VehicleLoadTransferResult:
    """Full vehicle lateral load transfer."""

    front: LateralLoadTransferResult
    rear: LateralLoadTransferResult
    lltd: float

    @property
    def total_delta_fz_n(self) -> float:
        return self.front.total_delta_fz_n + self.rear.total_delta_fz_n


@dataclass(frozen=True)
class LongitudinalLoadTransferResult:
    """Longitudinal load transfer under braking or acceleration.

    Positive delta_fz_front_n means load transferred forward (braking).
    Negative means load transferred rearward (acceleration).
    """

    ax_g: float
    delta_fz_front_n: float
    delta_fz_rear_n: float
    estimate_inputs: list[str]


def lateral_load_transfer(
    mass: MassProperties,
    unsprung: UnsprungMassSet,
    *,
    ay_g: float,
    front_rc_height_mm: float,
    rear_rc_height_mm: float,
    front_track_mm: float,
    rear_track_mm: float,
    front_roll_stiffness_nm_per_deg: float,
    rear_roll_stiffness_nm_per_deg: float,
    wheelbase_mm: float,
) -> VehicleLoadTransferResult:
    """Compute decomposed lateral load transfer for the full vehicle.

    Parameters
    ----------
    mass:
        Vehicle mass properties (total, CG, inertia).
    unsprung:
        Per-corner unsprung masses.
    ay_g:
        Lateral acceleration in g (positive = leftward in ISO 8855).
    front_rc_height_mm, rear_rc_height_mm:
        Roll centre heights from the solved kinematic state (mm).
    front_track_mm, rear_track_mm:
        Contact-patch-to-contact-patch track widths (mm).
    front_roll_stiffness_nm_per_deg, rear_roll_stiffness_nm_per_deg:
        Axle roll stiffness including springs and ARB (Nm/deg).
    wheelbase_mm:
        Wheelbase (mm).

    Returns
    -------
    VehicleLoadTransferResult
        Decomposed front and rear load transfer plus LLTD.
    """
    g = 9.81
    ay_ms2 = ay_g * g

    m_total = mass.total_mass_kg.value
    h_cg = mass.cg_height_mm.value

    usm_fl = unsprung.fl.mass_kg.value
    usm_fr = unsprung.fr.mass_kg.value
    usm_rl = unsprung.rl.mass_kg.value
    usm_rr = unsprung.rr.mass_kg.value
    usm_front = usm_fl + usm_fr
    usm_rear = usm_rl + usm_rr
    usm_total = usm_front + usm_rear

    m_sprung = m_total - usm_total

    fmf = mass.front_mass_fraction.value
    m_front_total = m_total * fmf
    m_rear_total = m_total * (1.0 - fmf)

    ra_height_mm = (
        front_rc_height_mm * (1.0 - fmf)
        + rear_rc_height_mm * fmf
    )

    h_sprung_above_ra = h_cg - ra_height_mm

    k_front = front_roll_stiffness_nm_per_deg
    k_rear = rear_roll_stiffness_nm_per_deg
    k_total = k_front + k_rear

    estimates: list[str] = []
    if mass.has_estimates():
        estimates.extend(mass.estimate_fields())
    if unsprung.fl.has_estimates():
        estimates.append("unsprung_fl")
    if unsprung.fr.has_estimates():
        estimates.append("unsprung_fr")
    if unsprung.rl.has_estimates():
        estimates.append("unsprung_rl")
    if unsprung.rr.has_estimates():
        estimates.append("unsprung_rr")

    front_geom = (
        m_front_total * ay_ms2 * front_rc_height_mm / 1000.0
    ) / front_track_mm * 1000.0

    rear_geom = (
        m_rear_total * ay_ms2 * rear_rc_height_mm / 1000.0
    ) / rear_track_mm * 1000.0

    if k_total > 0:
        elastic_moment_nm = m_sprung * ay_ms2 * h_sprung_above_ra / 1000.0
        front_elastic = (
            elastic_moment_nm * (k_front / k_total) / front_track_mm * 1000.0
        )
        rear_elastic = (
            elastic_moment_nm * (k_rear / k_total) / rear_track_mm * 1000.0
        )
    else:
        front_elastic = 0.0
        rear_elastic = 0.0

    usm_h_front = (
        unsprung.fl.cg_height_mm.value + unsprung.fr.cg_height_mm.value
    ) / 2.0
    usm_h_rear = (
        unsprung.rl.cg_height_mm.value + unsprung.rr.cg_height_mm.value
    ) / 2.0

    front_unsprung = (
        usm_front * ay_ms2 * usm_h_front / 1000.0
    ) / front_track_mm * 1000.0

    rear_unsprung = (
        usm_rear * ay_ms2 * usm_h_rear / 1000.0
    ) / rear_track_mm * 1000.0

    front_total = front_geom + front_elastic + front_unsprung
    rear_total = rear_geom + rear_elastic + rear_unsprung

    total_lt = front_total + rear_total
    lltd = front_total / total_lt if abs(total_lt) > 1e-10 else 0.5

    front_result = LateralLoadTransferResult(
        axle="front",
        ay_g=ay_g,
        geometric_delta_fz_n=front_geom,
        elastic_delta_fz_n=front_elastic,
        unsprung_delta_fz_n=front_unsprung,
        total_delta_fz_n=front_total,
        estimate_inputs=estimates,
    )
    rear_result = LateralLoadTransferResult(
        axle="rear",
        ay_g=ay_g,
        geometric_delta_fz_n=rear_geom,
        elastic_delta_fz_n=rear_elastic,
        unsprung_delta_fz_n=rear_unsprung,
        total_delta_fz_n=rear_total,
        estimate_inputs=estimates,
    )

    return VehicleLoadTransferResult(
        front=front_result,
        rear=rear_result,
        lltd=lltd,
    )


def longitudinal_load_transfer(
    mass: MassProperties,
    *,
    ax_g: float,
    wheelbase_mm: float,
) -> LongitudinalLoadTransferResult:
    """Compute longitudinal load transfer under braking or acceleration.

    Parameters
    ----------
    mass:
        Vehicle mass properties.
    ax_g:
        Longitudinal acceleration in g. Positive = forward (acceleration),
        negative = braking. ISO 8855: X+ forward.
    wheelbase_mm:
        Wheelbase in mm.

    Returns
    -------
    LongitudinalLoadTransferResult
        Load change at front and rear axles.
    """
    g = 9.81
    ax_ms2 = ax_g * g
    m = mass.total_mass_kg.value
    h = mass.cg_height_mm.value

    delta_fz = m * ax_ms2 * (h / 1000.0) / (wheelbase_mm / 1000.0)

    estimates: list[str] = []
    if mass.has_estimates():
        estimates.extend(mass.estimate_fields())

    return LongitudinalLoadTransferResult(
        ax_g=ax_g,
        delta_fz_front_n=-delta_fz,
        delta_fz_rear_n=delta_fz,
        estimate_inputs=estimates,
    )


def sprung_mass_kg(
    mass: MassProperties,
    unsprung: UnsprungMassSet,
) -> float:
    """Compute sprung mass from total mass minus total unsprung."""
    return mass.total_mass_kg.value - unsprung.total_kg()


def sprung_mass_front_fraction(
    mass: MassProperties,
    unsprung: UnsprungMassSet,
) -> float:
    """Fraction of sprung mass on the front axle.

    Uses total front mass (from front_mass_fraction) minus front
    unsprung mass.
    """
    m_total = mass.total_mass_kg.value
    m_front = m_total * mass.front_mass_fraction.value
    usm_front = unsprung.fl.mass_kg.value + unsprung.fr.mass_kg.value
    m_sprung = sprung_mass_kg(mass, unsprung)
    if m_sprung <= 0:
        return 0.5
    return (m_front - usm_front) / m_sprung
