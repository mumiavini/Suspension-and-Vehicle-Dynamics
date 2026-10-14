"""Sprung / unsprung split shared by the roll budget, the concept page and the maneuver.

``VehicleData.cg_height_mm`` and ``cg_from_front_axle_mm`` describe the WHOLE
car (the CG estimator includes the unsprung rows, the tilt test and the load
cases use the total weight). The roll moment the springs react, however, is
the SPRUNG mass about the roll axis, so it needs the sprung CG -- not the
whole-car CG paired with the sprung weight (which under-states the arm and
breaks the moment balance ``sum(dFz * t) = m * ay * h``).

``VehicleData`` carries only the total unsprung mass, so two assumptions are
made and stated wherever the result is shown:

* the unsprung mass is split equally between the axles;
* each axle's unsprung CG sits at its wheel-centre height (the loaded radius).

Frames: heights above the ground plane, stations rearward from the front axle
(the ``VehicleData`` convention), mm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

UNSPRUNG_ASSUMPTION = ("unsprung mass split equally front/rear, its CG at each axle's "
                       "wheel-centre height (loaded radius)")


@dataclass(frozen=True)
class SprungMass:
    """The sprung body alone, derived from whole-car data."""

    mass_kg: float
    cg_height_mm: float
    cg_from_front_axle_mm: float      # station, rearward of the front axle
    front_share: float                # share of the SPRUNG weight on the front axle
    unsprung_front_kg: float
    unsprung_rear_kg: float
    unsprung_cg_height_front_mm: float
    unsprung_cg_height_rear_mm: float


def sprung_mass(total_mass_kg: float, unsprung_mass_kg: float, cg_height_mm: float,
                cg_from_front_axle_mm: float, wheelbase_mm: float,
                loaded_radius_front_mm: float, loaded_radius_rear_mm: float) -> SprungMass:
    """Remove the unsprung masses from the whole-car CG.

    Raises:
        ValueError: if the sprung mass or the wheelbase is not positive.
    """
    m, m_u = float(total_mass_kg), float(unsprung_mass_kg)
    m_s = m - m_u
    if not (m_s > 0.0):
        raise ValueError(f"sprung mass is {m_s:g} kg (total {m:g}, unsprung {m_u:g})")
    if not (wheelbase_mm > 0.0):
        raise ValueError(f"wheelbase must be positive, got {wheelbase_mm:g} mm")
    m_uf = m_ur = 0.5 * m_u
    h_s = (m * cg_height_mm - m_uf * loaded_radius_front_mm
           - m_ur * loaded_radius_rear_mm) / m_s
    a_s = (m * cg_from_front_axle_mm - m_ur * wheelbase_mm) / m_s
    front_share = 1.0 - a_s / wheelbase_mm
    if not all(math.isfinite(v) for v in (h_s, a_s, front_share)):
        raise ValueError("sprung CG is not finite")
    return SprungMass(m_s, h_s, a_s, front_share, m_uf, m_ur,
                      float(loaded_radius_front_mm), float(loaded_radius_rear_mm))
