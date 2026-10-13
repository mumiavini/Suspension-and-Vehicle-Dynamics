"""Resolve a project into one complete, solvable car.

Two geometry sources, one result type:

* ``synthesis`` -- the repository's design chain. ``sla_geometry`` places the
  wishbones from the front-view targets, ``steering_geometry`` applies caster,
  supersedes the zero-caster outboard ball joints and adds the front tie rod,
  and the rear toe link is the project's declared design input. This is the
  same merge ``scripts/geometry_summary.build_merged`` performs; the only
  difference is that the toe link comes from the project instead of a module
  constant (pinned equal on the defaults by the tests). The vehicle
  wheelbase is authoritative: the rear axle station is derived from it, so
  the point wheelbase and the mass split can never disagree.
* ``hardpoints`` -- points given directly (CAD read-back, measurement, or a
  frozen synthesis the designer is now editing point by point).

Either way the result carries the ISO 8855 point table, per-point provenance
and the four ``vdcore`` corners. Every KPI in the app runs on those corners --
the merged corner is the only complete one (CLAUDE.md, "Where numbers come
from").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

import sla_geometry as sla
import steering_geometry as stg
from vdcore.models.hardpoint import Axle, Corner, Hardpoint, TirePackage, Vehicle

from .project import (
    CORNERS,
    INPUT_POINTS,
    AxleAlignment,
    HardpointSet,
    Layout,
    Project,
    Pt,
)

POINT_ORDER: tuple[str, ...] = (*INPUT_POINTS, "CONTACT_PATCH")
_FIELD = {
    "UCA_IN_FRONT": "uca_inboard_front",
    "UCA_IN_REAR": "uca_inboard_rear",
    "UCA_OUT": "uca_outboard",
    "LCA_IN_FRONT": "lca_inboard_front",
    "LCA_IN_REAR": "lca_inboard_rear",
    "LCA_OUT": "lca_outboard",
    "TIE_ROD_IN": "tie_rod_inboard",
    "TIE_ROD_OUT": "tie_rod_outboard",
    "WHEEL_CENTER": "wheel_center",
}
AXLE_OF = {"FL": "front", "FR": "front", "RL": "rear", "RR": "rear"}


class GeometryError(ValueError):
    """The project cannot be turned into a complete, valid car."""


@dataclass(frozen=True)
class AxleSetup:
    """Per-axle inputs the corners need beyond the points."""

    static_camber_deg: float
    static_toe_deg_per_side: float
    loaded_radius_mm: float
    tire_source: str
    tire_tol_mm: float


@dataclass
class CarGeometry:
    """One complete car in ISO 8855, ready for ``DWSolver``."""

    mode: str
    points: dict[str, dict[str, Pt]]
    sources: dict[str, dict[str, str]]
    tols: dict[str, dict[str, float]]
    axles: dict[str, AxleSetup]
    vehicle: Vehicle
    vehicle_data: sla.VehicleData
    front_inputs: sla.AxleInputs
    rear_inputs: sla.AxleInputs
    steering_inputs: stg.SteeringInputs
    # synthesis-only intermediates (None in hardpoint mode)
    design: sla.DesignReport | None = None
    steering_geometry: stg.SteeringGeometry | None = None
    warnings: list[str] = field(default_factory=list)
    mass_source: str = "design_intent"      # provenance of the vehicle mass / CG
    # brake / drive layout: picks the anti-geometry reference point
    layout: Layout = field(default_factory=Layout)

    # -- access ---------------------------------------------------------------
    def corner(self, cid: str) -> Corner:
        v = self.vehicle
        return {"FL": v.front.left, "FR": v.front.right,
                "RL": v.rear.left, "RR": v.rear.right}[cid]

    def axle(self, name: str) -> Axle:
        return self.vehicle.front if name == "front" else self.vehicle.rear

    def axle_inputs(self, name: str) -> sla.AxleInputs:
        return self.front_inputs if name == "front" else self.rear_inputs

    def p(self, cid: str, name: str) -> np.ndarray:
        return np.array(self.points[cid][name], dtype=float)

    @property
    def wheelbase_mm(self) -> float:
        return float(self.points["FL"]["WHEEL_CENTER"][0] - self.points["RL"]["WHEEL_CENTER"][0])

    def track_mm(self, axle: str) -> float:
        """Track at the contact patches (the rules / tilt-test datum)."""
        lc, rc = ("FL", "FR") if axle == "front" else ("RL", "RR")
        return float(self.points[lc]["CONTACT_PATCH"][1] - self.points[rc]["CONTACT_PATCH"][1])

    def estimate_count(self) -> int:
        n = sum(1 for c in self.sources.values() for s in c.values() if s == "estimate")
        n += sum(1 for a in self.axles.values() if a.tire_source == "estimate")
        n += self.mass_source == "estimate"
        return n

    def rows(self) -> list[tuple[str, str, float, float, float, str, float]]:
        out = []
        for c in CORNERS:
            for n in POINT_ORDER:
                pt = self.points[c].get(n)
                if pt is None:
                    continue
                out.append((c, n, pt[0], pt[1], pt[2],
                            self.sources[c].get(n, "derived"),
                            self.tols[c].get(n, float("nan"))))
        return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def contact_patch_point(wc: Pt, loaded_radius_mm: float, camber_deg: float, left: bool) -> Pt:
    """Contact patch from the wheel centre; mirrors ``vdcore.geometry.derived``.

    Negative camber moves the patch OUTBOARD; the patch stays on z = wc_z - r.
    """
    shift = -loaded_radius_mm * math.tan(math.radians(camber_deg))
    y = wc[1] + shift if left else wc[1] - shift
    return (wc[0], y, wc[2] - loaded_radius_mm)


def _corner(cid: str, pts: dict[str, Pt], sources: dict[str, str], tols: dict[str, float],
            setup: AxleSetup) -> Corner:
    def hp(name: str) -> Hardpoint:
        x, y, z = pts[name]
        return Hardpoint(name=name, x_mm=float(x), y_mm=float(y), z_mm=float(z),
                         source=sources.get(name, "estimate"),  # type: ignore[arg-type]
                         tol_mm=float(tols.get(name, 1.0)))

    try:
        return Corner(
            corner_id=cid,  # type: ignore[arg-type]
            **{_FIELD[n]: hp(n) for n in INPUT_POINTS},
            tire=TirePackage(loaded_radius_mm=setup.loaded_radius_mm,
                             source=setup.tire_source,  # type: ignore[arg-type]
                             tol_mm=setup.tire_tol_mm),
            static_camber_deg=setup.static_camber_deg,
            static_toe_deg_per_side=setup.static_toe_deg_per_side,
        )
    except (ValueError, KeyError) as exc:
        raise GeometryError(f"Corner {cid}: {exc}") from exc


def _vehicle_for_wheelbase(veh: sla.VehicleData, wheelbase_mm: float) -> sla.VehicleData:
    """Scale the CG station so the front mass fraction is kept at a new wheelbase."""
    if abs(wheelbase_mm - veh.wheelbase_mm) < 1e-6 or wheelbase_mm <= 0:
        return veh
    frac_rear = veh.cg_from_front_axle_mm / veh.wheelbase_mm
    return replace(veh, wheelbase_mm=wheelbase_mm,
                   cg_from_front_axle_mm=frac_rear * wheelbase_mm)


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #

def synthesis_rear_axle_x_mm(project: Project) -> float:
    """Rear axle station synthesis uses: front station + vehicle wheelbase (design frame).

    ``vehicle.wheelbase_mm`` is authoritative in synthesis mode, so the stored
    ``rear.axle_x_mm`` is not read; the editor shows this derived value.
    """
    return project.front.axle_x_mm + project.vehicle.wheelbase_mm


def synthesize(project: Project) -> CarGeometry:
    """sla wishbones + steering caster/tie rod + rear toe link -> complete car.

    Instant (pure analytic construction, no solver), so the UI can call it on
    every keystroke.
    """
    veh, front_in, steer = project.vehicle, project.front, project.steering
    # one wheelbase: the rear axle follows the vehicle data (the CG station is
    # kept, so a wheelbase edit also moves the weight distribution)
    rear_in = replace(project.rear, axle_x_mm=synthesis_rear_axle_x_mm(project))
    try:
        # a degenerate input divides by zero inside the construction; the
        # non-finite check below turns that into a GeometryError, so the
        # numpy warning would only be noise
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            design = sla.run(veh, front_in, rear_in)
            sg = stg.synthesize_steering(design.front, steer, veh)
    except (ValueError, ZeroDivisionError, FloatingPointError, sla.KinematicError) as exc:
        raise GeometryError(f"Synthesis failed: {exc}") from exc
    shp = stg.build_steering_hardpoints(sg)

    pts: dict[str, dict[str, Pt]] = {c: {} for c in CORNERS}
    for c, name, x, y, z in design.model.rows():
        pts[c][name] = (x, y, z)
    # steering_geometry supersedes the zero-caster outboard ball joints
    for c, name, x, y, z in shp.rows():
        pts[c][name] = (x, y, z)
    # rear toe link: declared design input, mirrored onto the right side. Its X
    # moves with the rear axle station it was declared against (ISO x = -design
    # x); Y and Z stay absolute.
    dx_toe = rear_in.axle_x_mm - project.rear_toe_link_axle_x_mm
    for c, sy in (("RL", 1.0), ("RR", -1.0)):
        for name, (x, y, z) in (("TIE_ROD_IN", project.rear_toe_link_inboard),
                                ("TIE_ROD_OUT", project.rear_toe_link_outboard)):
            pts[c][name] = (float(x) - dx_toe, sy * float(y), float(z))

    for row in pts.values():
        for name in list(row):
            row[name] = tuple(float(v) + 0.0 for v in row[name])  # type: ignore[assignment]

    tol = project.synthesis_tol_mm
    sources = {c: {n: ("derived" if n == "CONTACT_PATCH" else "design_intent")
                   for n in pts[c]} for c in CORNERS}
    tols = {c: {n: tol for n in pts[c]} for c in CORNERS}
    axles = {
        "front": AxleSetup(front_in.static_camber_deg, steer.static_toe_deg_per_side,
                           front_in.loaded_radius_mm, project.tire_source, 1.0),
        "rear": AxleSetup(rear_in.static_camber_deg, project.rear_static_toe_deg_per_side,
                          rear_in.loaded_radius_mm, project.tire_source, 1.0),
    }
    vehicle = _build_vehicle(pts, sources, tols, axles)
    warnings = []
    if not all(np.isfinite(v) for c in pts.values() for p in c.values() for v in p):
        raise GeometryError("Synthesis produced a non-finite hardpoint -- check the "
                            "inputs (e.g. a pickup plane at the FVIC).")
    return CarGeometry(
        mode="synthesis", points=pts, sources=sources, tols=tols, axles=axles,
        vehicle=vehicle, vehicle_data=veh, front_inputs=front_in, rear_inputs=rear_in,
        steering_inputs=steer, design=design, steering_geometry=sg, warnings=warnings,
        layout=project.layout, mass_source=project.mass_source,
    )


def _build_vehicle(pts: dict[str, dict[str, Pt]], sources: dict[str, dict[str, str]],
                   tols: dict[str, dict[str, float]], axles: dict[str, AxleSetup]) -> Vehicle:
    corners = {c: _corner(c, pts[c], sources[c], tols[c], axles[AXLE_OF[c]]) for c in CORNERS}
    return Vehicle(front=Axle(left=corners["FL"], right=corners["FR"]),
                   rear=Axle(left=corners["RL"], right=corners["RR"]))


# --------------------------------------------------------------------------- #
# Hardpoint mode
# --------------------------------------------------------------------------- #

def from_hardpoints(project: Project) -> CarGeometry:
    hs = project.hardpoints
    if hs is None:
        raise GeometryError("Hardpoint mode selected but the project has no hardpoint set.")
    missing = hs.missing()
    if missing:
        raise GeometryError("Missing hardpoints: " + ", ".join(missing))

    def setup(a: AxleAlignment) -> AxleSetup:
        return AxleSetup(a.static_camber_deg, a.static_toe_deg_per_side, a.loaded_radius_mm,
                         a.tire_source, a.tire_tol_mm)

    axles = {"front": setup(hs.front), "rear": setup(hs.rear)}
    pts: dict[str, dict[str, Pt]] = {}
    sources: dict[str, dict[str, str]] = {}
    tols: dict[str, dict[str, float]] = {}
    for c in CORNERS:
        pts[c] = {n: tuple(float(v) for v in hs.points[c][n]) for n in INPUT_POINTS}  # type: ignore[misc]
        sources[c] = {n: hs.sources.get(c, {}).get(n, "estimate") for n in INPUT_POINTS}
        tols[c] = {n: float(hs.tols.get(c, {}).get(n, 1.0)) for n in INPUT_POINTS}
        _check_corner_points(c, pts[c])
        ax = axles[AXLE_OF[c]]
        _check_ground_datum(c, pts[c]["WHEEL_CENTER"], ax)
        pts[c]["CONTACT_PATCH"] = contact_patch_point(
            pts[c]["WHEEL_CENTER"], ax.loaded_radius_mm, ax.static_camber_deg, c in ("FL", "RL"))
        sources[c]["CONTACT_PATCH"] = "derived"
        tols[c]["CONTACT_PATCH"] = float("nan")

    wb = float(pts["FL"]["WHEEL_CENTER"][0] - pts["RL"]["WHEEL_CENTER"][0])
    if wb <= 0.0:
        raise GeometryError(
            f"Front wheel centre x = {pts['FL']['WHEEL_CENTER'][0]:.1f} mm is not ahead of the "
            f"rear wheel centre x = {pts['RL']['WHEEL_CENTER'][0]:.1f} mm (wheelbase {wb:.1f} "
            "mm): X looks reversed: check the import frame (ISO X+ forward).")
    vehicle = _build_vehicle(pts, sources, tols, axles)
    veh = _vehicle_for_wheelbase(project.vehicle, wb)
    warnings = []
    if abs(wb - project.vehicle.wheelbase_mm) > 1.0:
        # wb > 0 here, so _vehicle_for_wheelbase did rescale; quote what it did
        warnings.append(
            f"Wheelbase from the hardpoints is {wb:.1f} mm; vehicle data says "
            f"{project.vehicle.wheelbase_mm:.1f} mm. The hardpoint value is used and the CG "
            f"station is scaled from {project.vehicle.cg_from_front_axle_mm:.1f} to "
            f"{veh.cg_from_front_axle_mm:.1f} mm behind the front axle to keep the front mass "
            f"fraction ({100.0 * veh.front_mass_fraction:.1f} %).")
    front_in = replace(project.front, loaded_radius_mm=axles["front"].loaded_radius_mm,
                       static_camber_deg=axles["front"].static_camber_deg,
                       track_mm=_cp_track(pts, "FL", "FR"))
    rear_in = replace(project.rear, loaded_radius_mm=axles["rear"].loaded_radius_mm,
                      static_camber_deg=axles["rear"].static_camber_deg,
                      track_mm=_cp_track(pts, "RL", "RR"),
                      axle_x_mm=-float(pts["RL"]["WHEEL_CENTER"][0]))
    return CarGeometry(
        mode="hardpoints", points=pts, sources=sources, tols=tols, axles=axles,
        vehicle=vehicle, vehicle_data=veh, front_inputs=front_in, rear_inputs=rear_in,
        steering_inputs=project.steering, warnings=warnings, layout=project.layout,
        mass_source=project.mass_source,
    )


def _cp_track(pts: dict[str, dict[str, Pt]], lc: str, rc: str) -> float:
    return float(pts[lc]["CONTACT_PATCH"][1] - pts[rc]["CONTACT_PATCH"][1])


_DEGENERATE_MM = 1e-3


def _check_corner_points(cid: str, pts: dict[str, Pt]) -> None:
    """Name the degenerate inputs the solver would otherwise fail on without a cause."""
    p = {n: np.array(v, dtype=float) for n, v in pts.items()}
    for arm in ("UCA", "LCA"):
        if np.linalg.norm(p[f"{arm}_IN_FRONT"] - p[f"{arm}_IN_REAR"]) < _DEGENERATE_MM:
            raise GeometryError(
                f"Corner {cid}: the {arm} pivot axis has zero length ({arm}_IN_FRONT "
                f"coincides with {arm}_IN_REAR).")
    kp = p["UCA_OUT"] - p["LCA_OUT"]
    n = float(np.linalg.norm(kp))
    if n < _DEGENERATE_MM:
        raise GeometryError(
            f"Corner {cid}: UCA_OUT coincides with LCA_OUT (the upper and lower ball joints "
            "are one point, so there is no kingpin axis).")
    u = kp / n
    v = p["TIE_ROD_OUT"] - p["LCA_OUT"]
    if np.linalg.norm(v - np.dot(v, u) * u) < _DEGENERATE_MM:
        raise GeometryError(
            f"Corner {cid}: TIE_ROD_OUT lies on the kingpin axis (LCA_OUT-UCA_OUT line): the "
            "steering arm has zero length, so the tie rod cannot hold the steer angle.")


def _check_ground_datum(cid: str, wc: Pt, ax: AxleSetup) -> None:
    """The ISO origin is on the ground: the wheel centre must sit one loaded radius above z=0.

    Scrub, trail, the roll centre, FVSA and anti all take the ground as z = 0,
    while the contact patch is derived at ``wc_z - loaded_radius``; a datum off
    the ground would make the two disagree silently.
    """
    off = float(wc[2]) - ax.loaded_radius_mm
    tol = max(0.5, ax.tire_tol_mm)
    if abs(off) > tol:
        raise GeometryError(
            f"Corner {cid}: wheel-centre z {wc[2]:.1f} mm minus the loaded radius "
            f"{ax.loaded_radius_mm:.1f} mm is {off:+.1f} mm (allowed ± {tol:g} mm). The ISO 8855 "
            "origin must lie on the ground plane (z = 0 under the contact patches): translate "
            "the point set in Z or correct the loaded radius.")


def resolve(project: Project) -> CarGeometry:
    """The car the project describes, from whichever source it selects."""
    if project.geometry_mode == "hardpoints":
        return from_hardpoints(project)
    return synthesize(project)


def freeze_to_hardpoints(project: Project, origin: str = "") -> HardpointSet:
    """Snapshot the current synthesis as an editable hardpoint set.

    Provenance stays ``design_intent`` -- freezing does not make a chosen
    number measured.
    """
    geo = synthesize(project)
    pts = {c: {n: geo.points[c][n] for n in INPUT_POINTS} for c in CORNERS}
    hs = HardpointSet.from_points(pts, source="design_intent", tol_mm=project.synthesis_tol_mm,
                                  origin=origin or "frozen from synthesis")
    hs.front = AxleAlignment(geo.axles["front"].static_camber_deg,
                             geo.axles["front"].static_toe_deg_per_side,
                             geo.axles["front"].loaded_radius_mm, geo.axles["front"].tire_source,
                             1.0)
    hs.rear = AxleAlignment(geo.axles["rear"].static_camber_deg,
                            geo.axles["rear"].static_toe_deg_per_side,
                            geo.axles["rear"].loaded_radius_mm, geo.axles["rear"].tire_source,
                            1.0)
    return hs


def mirror_left_to_right(points: dict[str, dict[str, Pt]]) -> dict[str, dict[str, Pt]]:
    """Right corners as the Y-mirror of the left ones (ISO 8855: Y+ LEFT)."""
    out = {c: dict(d) for c, d in points.items()}
    for left, right in (("FL", "FR"), ("RL", "RR")):
        out[right] = {n: (x, -y + 0.0, z) for n, (x, y, z) in points[left].items()}
    return out
