"""Concept-stage numbers: what the car must be before any hardpoint exists.

Closed-form checks from the team's concept notebook (``random_calcs.ipynb``)
and ``sla_geometry.solve_vehicle``: CG estimate from subsystem masses,
rollover / tilt margins, track and wheelbase rules, the roll-stiffness budget a
roll-gradient target implies, the chassis torsional stiffness that keeps the
front/rear split meaningful, and how much of the roll the FVSA's camber gain
recovers.

Every target used here (roll gradient, RC heights, FVSA, design lateral g) is
``design_intent`` from literature -- the tool cannot derive the targets it
serves (CLAUDE.md). The results are consequences of those targets, not
recommendations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import sla_geometry as sla

from . import mass
from .kpis import (
    FSAE_MIN_TRACK_RATIO,
    FSAE_MIN_WHEELBASE_MM,
    GRAVITY,
    KpiSet,
    _Builder,
)
from .project import MassItem, Project


@dataclass(frozen=True)
class CgEstimate:
    total_mass_kg: float
    cg_height_mm: float
    cg_x_mm: float | None            # behind the front axle, None if any x is missing
    front_mass_fraction: float | None
    estimate_share: float            # fraction of mass whose source is "estimate"


def cg_estimate(items: list[MassItem], wheelbase_mm: float) -> CgEstimate:
    """Mass-weighted CG: h = Σ m_i h_i / Σ m_i (same for x when every x is known).

    Raises:
        ValueError: if an item's mass is negative (or not a number): a negative
            mass puts the CG outside the span of the item heights.
    """
    for i in items:
        if not i.mass_kg >= 0.0:
            raise ValueError(f"'{i.name}' has mass {i.mass_kg:g} kg; masses must be ≥ 0")
    m = sum(i.mass_kg for i in items)
    if m <= 0:
        return CgEstimate(0.0, float("nan"), None, None, 0.0)
    h = sum(i.mass_kg * i.z_mm for i in items) / m
    x = None
    frac = None
    if items and all(i.x_mm is not None for i in items):
        x = sum(i.mass_kg * float(i.x_mm) for i in items) / m  # type: ignore[arg-type]
        frac = 1.0 - x / wheelbase_mm if wheelbase_mm > 0 else None
    est = sum(i.mass_kg for i in items if i.source == "estimate") / m
    return CgEstimate(m, h, x, frac, est)


def apply_cg_estimate(project: Project, est: CgEstimate) -> Project:
    """Copy the estimate into the vehicle data -- an explicit designer action."""
    from dataclasses import replace

    p = project.copy()
    kw: dict[str, float] = {"total_mass_kg": est.total_mass_kg, "cg_height_mm": est.cg_height_mm}
    if est.cg_x_mm is not None:
        kw["cg_from_front_axle_mm"] = est.cg_x_mm
    p.vehicle = replace(p.vehicle, **kw)
    return p


def concept_kpis(project: Project, track_front_mm: float | None = None,
                 track_rear_mm: float | None = None, wheelbase_mm: float | None = None,
                 rc_front_mm: float | None = None, rc_rear_mm: float | None = None) -> KpiSet:
    """Concept checks. Defaults read the synthesis inputs (design-intent targets)."""
    veh: sla.VehicleData = project.vehicle
    tf = track_front_mm if track_front_mm is not None else project.front.track_mm
    tr = track_rear_mm if track_rear_mm is not None else project.rear.track_mm
    wb = wheelbase_mm if wheelbase_mm is not None else veh.wheelbase_mm
    rcf = rc_front_mm if rc_front_mm is not None else project.front.rc_height_mm
    rcr = rc_rear_mm if rc_rear_mm is not None else project.rear.rc_height_mm
    h = veh.cg_height_mm
    b = _Builder()

    g = "Layout & rules"
    narrow, wide = min(tf, tr), max(tf, tr)
    b.add("concept.wheelbase", "Wheelbase", g, "vehicle", wb, "mm", ".1f", "vehicle data",
          (FSAE_MIN_WHEELBASE_MM, math.inf), "FSAE rules")
    b.add("concept.track_ratio", "Narrow / wide track", g, "vehicle", 100 * narrow / wide, "%",
          ".1f", "axle inputs", (100 * FSAE_MIN_TRACK_RATIO, 100.0), "FSAE rules")
    b.add("concept.t_over_l", "Mean track / wheelbase", g, "vehicle", 0.5 * (tf + tr) / wb, "-",
          ".3f", "axle inputs",
          note="0.75-0.80 is the FSAE consensus band quoted in the concept notes; at fixed "
               "track a higher T/L (shorter wheelbase) raises the yaw gain V/L (more "
               "responsive), a lower one is calmer (information).")
    ssf = narrow / (2.0 * h)
    b.add("concept.ssf", "Static stability factor T/(2h)", g, "vehicle", ssf, "-", ".3f",
          "narrow track, CG height", (math.tan(math.radians(veh.tilt_test_angle_deg)), math.inf),
          f"tan({veh.tilt_test_angle_deg:g}°) tilt-test equivalent",
          note=f"sliding before rollover needs T/(2h) > μ (rigid body); the tilt test asks "
               f"for tan({veh.tilt_test_angle_deg:g}°). Rigid body: on the table the normal "
               "load falls to W·cos θ (suspension extends, CG rises) and the body rolls by "
               "about gradient·sin θ, both reducing the margin")
    tilt_min = 2.0 * h * math.tan(math.radians(veh.tilt_test_angle_deg))
    b.add("concept.tilt_min_track", "Minimum track for the tilt test", g, "vehicle", tilt_min,
          "mm", ".0f", "2·h·tan(tilt)")

    g = "Mass & CG"
    b.add("concept.front_frac", "Static front mass fraction", g, "vehicle",
          100 * veh.front_mass_fraction, "%", ".1f", "vehicle data (design_intent)")
    b.add("concept.sprung_mass", "Sprung mass", g, "vehicle", veh.sprung_mass_kg, "kg", ".1f",
          "total − unsprung")
    W = veh.total_mass_kg * GRAVITY
    b.add("concept.weight", "Total weight", g, "vehicle", W, "N", ".0f", "m·g")

    g = "Roll budget (design-intent RCs)"
    grad = veh.target_roll_gradient_deg_per_g
    specs = [("concept.roll_axis", "Roll axis height at sprung CG", g, "vehicle", "mm"),
             ("concept.roll_arm", "Roll moment arm d (sprung CG)", g, "vehicle", "mm"),
             ("concept.k_roll", "Required roll stiffness", g, "vehicle", "N·m/deg"),
             ("concept.k_roll_pend", "Required roll stiffness, with pendulum term", g,
              "vehicle", "N·m/deg"),
             ("concept.torsion_min", "Chassis torsion, min factor", g, "vehicle", "N·m/deg"),
             ("concept.torsion_target", "Chassis torsion, target factor", g, "vehicle",
              "N·m/deg")]
    try:
        # the springs react the SPRUNG mass about the roll axis: sprung CG, not whole-car
        sm = mass.sprung_mass(veh.total_mass_kg, veh.unsprung_mass_kg, h,
                              veh.cg_from_front_axle_mm, wb, project.front.loaded_radius_mm,
                              project.rear.loaded_radius_mm)
    except ValueError as exc:
        b.fail(specs, f"sprung CG: {exc}")
        sm = None
    if sm is not None:
        h_axis = rcf + (rcr - rcf) * sm.cg_from_front_axle_mm / wb
        d = sm.cg_height_mm - h_axis
        w_s = sm.mass_kg * GRAVITY
        err = ""
        if d <= 0:
            err = "roll axis at or above the sprung CG: requirement undefined"
        elif grad <= 0:
            err = "target roll gradient must be positive"
        k = w_s * d / 1000.0 / grad if not err else math.nan
        k_p = w_s * d / 1000.0 * (1.0 + 1.0 / math.radians(grad)) * math.pi / 180.0 \
            if not err else math.nan
        # the whole-car CG paired with the sprung weight (sla_geometry.solve_vehicle)
        d_whole = h - (rcf + (rcr - rcf) * veh.cg_from_front_axle_mm / wb)
        parity = (f"; sla_geometry.solve_vehicle (whole-car CG) gives "
                  f"{w_s * d_whole / 1000.0 / grad:.1f} N·m/deg" if grad > 0 else "")
        b.add("concept.roll_axis", "Roll axis height at sprung CG", g, "vehicle", h_axis, "mm",
              ".1f", "RC targets, sprung CG station",
              note=f"sprung CG {sm.cg_from_front_axle_mm:.1f} mm behind the front axle "
                   f"(whole car {veh.cg_from_front_axle_mm:.1f} mm)")
        b.add("concept.roll_arm", "Roll moment arm d (sprung CG)", g, "vehicle", d, "mm", ".1f",
              "h_s − h_axis (engine.mass)",
              note=f"sprung CG height h_s {sm.cg_height_mm:.1f} mm; {mass.UNSPRUNG_ASSUMPTION}; "
                   f"the whole-car CG ({h:g} mm) gives d = {d_whole:.1f} mm")
        b.add("concept.k_roll", "Required roll stiffness", g, "vehicle", k, "N·m/deg", ".1f",
              "W_s·d / gradient on the sprung CG",
              note=f"for {grad:g} deg/g (design_intent){parity}", error=err)
        b.add("concept.k_roll_pend", "Required roll stiffness, with pendulum term", g,
              "vehicle", k_p, "N·m/deg", ".1f", "W_s·d·(1 + 1/φ) (Milliken Ch.16)",
              note="sprung CG; the Maneuver page's implied stiffness uses this definition",
              error=err)
        b.add("concept.torsion_min", "Chassis torsion, min factor", g, "vehicle",
              veh.chassis_factor_min * k, "N·m/deg", ".0f",
              f"{veh.chassis_factor_min:g} × K_roll", error=err)
        b.add("concept.torsion_target", "Chassis torsion, target factor", g, "vehicle",
              veh.chassis_factor_target * k, "N·m/deg", ".0f",
              f"{veh.chassis_factor_target:g} × K_roll", error=err)

    g = f"Lateral transfer at {veh.design_ay_g:g} g (LLTD {veh.lltd_front:.2f} front)"
    ay = veh.design_ay_g
    for axle, t, share, frac_ax in (("front", tf, veh.lltd_front, veh.front_mass_fraction),
                                    ("rear", tr, 1 - veh.lltd_front,
                                     1 - veh.front_mass_fraction)):
        fz_s = W * frac_ax / 2.0
        dW = share * (W * ay * h / 1000.0) / (t / 1000.0)
        b.add(f"concept.{axle}_fz_outer", f"{axle.title()} outer wheel load", g, axle,
              fz_s + dW, "N", ".0f", "sla_geometry.load_cases")
        b.add(f"concept.{axle}_fz_inner", f"{axle.title()} inner wheel load", g, axle,
              fz_s - dW, "N", ".0f", "sla_geometry.load_cases", (0.0, math.inf),
              "inner wheel must stay loaded", note="≤ 0 = the inner wheel has lifted")
    # no transfer at ay = 0: no wheel can lift, so the band has no edges
    no_ay = "" if ay > 0 else "undefined at design ay ≤ 0 (no lateral transfer)"
    lo = 1.0 - (1 - veh.front_mass_fraction) * tr / (2.0 * ay * h) if not no_ay else math.nan
    hi = veh.front_mass_fraction * tf / (2.0 * ay * h) if not no_ay else math.nan
    b.add("concept.lltd_lo", "LLTD band lower edge (rear inner lifts below)", g, "vehicle", lo,
          "-", ".3f", "geometry_summary.lltd_lift_threshold", error=no_ay)
    b.add("concept.lltd_hi", "LLTD band upper edge (front inner lifts above)", g, "vehicle", hi,
          "-", ".3f", "geometry_summary.lltd_lift_threshold", error=no_ay)

    g = "Camber recovery in roll (first order)"
    roll = grad * ay
    for axle, inp in (("front", project.front), ("rear", project.rear)):
        t = inp.track_mm
        travel = 0.5 * t * math.tan(math.radians(roll))
        gain = math.degrees(1.0) / inp.fvsa_length_mm
        rec = gain * travel
        b.add(f"concept.{axle}_roll_travel", f"{axle.title()} wheel travel at {roll:.2f}° roll",
              g, axle, travel, "mm", ".1f", "½·T·tan(φ)")
        b.add(f"concept.{axle}_camber_recovered", f"{axle.title()} camber recovered", g, axle,
              rec, "deg", ".2f", "57.3/FVSA × travel")
        b.add(f"concept.{axle}_recovery_pct", f"{axle.title()} share of roll recovered", g, axle,
              100.0 * rec / roll if roll > 0 else float("nan"), "%", ".0f",
              "recovered / roll", note="concept notes quote 60-80 % as the usual window")
        b.add(f"concept.{axle}_outer_camber", f"{axle.title()} outer camber vs road", g, axle,
              inp.static_camber_deg + roll - rec, "deg", ".2f",
              "static + roll − recovered (first order)",
              note="the solved value is on the Kinematics / Maneuver pages")
    return b.out
