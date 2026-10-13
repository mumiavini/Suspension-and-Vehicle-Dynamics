"""Place the whole car at a chassis pose for the live 3D view.

Inputs are chassis motions -- heave, roll, pitch -- and rack travel. Each
corner gets the wheel travel that keeps its contact patch on the road to first
order::

    travel_i = heave + (x_i - x_cg)·tan(pitch) - y_i·tan(roll)

(ISO 8855: roll positive about +X lifts the LEFT side, so left wheels droop;
pitch positive about +Y drops the nose, so front wheels bump; heave positive =
chassis down = bump everywhere.)

Each corner is then solved exactly by ``DWSolver`` at that travel and rack,
expressed in the CHASSIS frame, and finally carried into the world by the
chassis rotation about the CG station at ground level. The small-angle travel
mapping means a patch can sit a fraction of a millimetre off z = 0 at large
angles -- this is a visualisation pose. Quantitative roll results (camber
relative to the road, RC in roll) come from ``vdcore.analysis.axle.axle_roll``,
which solves both patches onto the road exactly.

``DWSolver`` assembles the linkage far beyond any physical articulation, so a
corner whose travel leaves the design window [-droop, +bump] is flagged
(``CornerPose.beyond_travel``) rather than shown as an ordinary pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from vdcore.analysis.roll_centre import front_view_instant_centre
from vdcore.geometry.solver import DWSolver, SolverResult
from vdcore.models.hardpoint import DerivedPoint

from .geometry import AXLE_OF, CarGeometry

LEFT = ("FL", "RL")


@dataclass
class CornerPose:
    cid: str
    travel_mm: float
    converged: bool
    camber_deg: float
    toe_deg_per_side: float
    caster_deg: float
    kpi_deg: float
    points: dict[str, np.ndarray]          # chassis frame
    spin_axis: np.ndarray                  # unit, outboard, chassis frame
    residual: float = 0.0
    inclination_deg: float = math.nan      # wheel plane to chassis vertical, inclination_deg()
    beyond_travel: bool = False            # travel outside [-droop, +bump] of its axle


@dataclass
class Pose:
    heave_mm: float
    roll_deg: float
    pitch_deg: float
    rack_mm: float
    corners: dict[str, CornerPose]
    rotation: np.ndarray                   # chassis -> world
    pivot: np.ndarray
    rc_chassis: dict[str, np.ndarray | None] = field(default_factory=dict)

    def to_world(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        moved = (self.rotation @ (p - self.pivot).T).T + self.pivot
        return moved + np.array([0.0, 0.0, -self.heave_mm])

    @property
    def all_converged(self) -> bool:
        return all(c.converged for c in self.corners.values())


def spin_axis_from_angles(camber_deg: float, toe_deg: float, left: bool) -> np.ndarray:
    """Outboard spin axis whose extraction gives back exactly (camber, toe).

    Exact inverse of ``DWSolver._extract_angles``: toe = atan2(x, |y|) and
    camber = -atan2(z, |y|), so the axis is proportional to
    (tan toe, ±1, -tan camber) with +1 on the left (Y+ is LEFT).
    """
    v = np.array([math.tan(math.radians(toe_deg)), 1.0 if left else -1.0,
                  -math.tan(math.radians(camber_deg))])
    return v / np.linalg.norm(v)


def inclination_deg(res: SolverResult) -> float:
    """Inclination: angle of the wheel plane to the vertical, deg (ISO 8855).

    ``-asin(s_z)`` of the outboard spin axis s, rebuilt from the solver's
    angles exactly as :func:`spin_axis_from_angles` does (independent of side).
    Negative = top inboard, like camber. ``DWSolver.camber_deg`` projects the
    spin axis onto the vehicle Y-Z plane first, which for a steered wheel
    reads about 1/cos(steer) too large; the two agree at zero toe and differ
    by < 1e-5 deg for |toe| <= 0.2 deg, so this is the camber to quote when
    the wheel is steered.
    """
    s = spin_axis_from_angles(res.camber_deg, res.toe_deg_per_side, True)
    return -math.degrees(math.asin(float(np.clip(s[2], -1.0, 1.0))))


class PoseSolver:
    """Keeps one DWSolver per corner so repeated poses (slider drags) are cheap."""

    def __init__(self, geo: CarGeometry) -> None:
        self.geo = geo
        self.solvers = {cid: DWSolver(geo.corner(cid)) for cid in ("FL", "FR", "RL", "RR")}
        veh = geo.vehicle_data
        self.x_cg = -veh.cg_from_front_axle_mm   # ISO: rear of the front axle is -X

    def solve(self, heave_mm: float = 0.0, roll_deg: float = 0.0, pitch_deg: float = 0.0,
              rack_mm: float = 0.0) -> Pose:
        geo = self.geo
        tr, tp = math.tan(math.radians(roll_deg)), math.tan(math.radians(pitch_deg))
        corners: dict[str, CornerPose] = {}
        for cid, solver in self.solvers.items():
            cp = geo.p(cid, "CONTACT_PATCH")
            travel = heave_mm + (cp[0] - self.x_cg) * tp - cp[1] * tr
            rk = rack_mm if AXLE_OF[cid] == "front" else 0.0
            r = solver.solve(wheel_travel_mm=float(travel), rack_mm=float(rk))
            inp = geo.axle_inputs(AXLE_OF[cid])
            beyond = not (-inp.travel_droop_mm <= travel <= inp.travel_bump_mm)
            corners[cid] = self._corner_pose(cid, r, travel, rk, beyond)

        phi, th = math.radians(roll_deg), math.radians(pitch_deg)
        rx = np.array([[1, 0, 0], [0, math.cos(phi), -math.sin(phi)],
                       [0, math.sin(phi), math.cos(phi)]])
        ry = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0],
                       [-math.sin(th), 0, math.cos(th)]])
        pose = Pose(heave_mm, roll_deg, pitch_deg, rack_mm, corners, ry @ rx,
                    np.array([self.x_cg, 0.0, 0.0]))
        for axle, (lc, rc) in (("front", ("FL", "FR")), ("rear", ("RL", "RR"))):
            pose.rc_chassis[axle] = self._roll_centre(corners[lc], corners[rc])
        return pose

    def _corner_pose(self, cid: str, r: SolverResult, travel: float, rack: float,
                     beyond_travel: bool = False) -> CornerPose:
        geo = self.geo
        up = np.array([0.0, 0.0, travel])   # world (chassis displaced by -travel) -> chassis

        def v(p: DerivedPoint) -> np.ndarray:
            return np.array([p.x_mm, p.y_mm, p.z_mm]) + up

        pts = {n: geo.p(cid, n) for n in ("UCA_IN_FRONT", "UCA_IN_REAR", "LCA_IN_FRONT",
                                          "LCA_IN_REAR")}
        pts["TIE_ROD_IN"] = geo.p(cid, "TIE_ROD_IN") + np.array([0.0, rack, 0.0])
        pts["UCA_OUT"] = v(r.ubj)
        pts["LCA_OUT"] = v(r.lbj)
        pts["TIE_ROD_OUT"] = v(r.tro)
        pts["WHEEL_CENTER"] = v(r.wheel_center)
        pts["CONTACT_PATCH"] = v(r.contact_patch)
        return CornerPose(
            cid=cid, travel_mm=float(travel), converged=bool(r.converged),
            camber_deg=r.camber_deg, toe_deg_per_side=r.toe_deg_per_side,
            caster_deg=r.caster_deg, kpi_deg=r.kpi_deg, points=pts,
            spin_axis=spin_axis_from_angles(r.camber_deg, r.toe_deg_per_side, cid in LEFT),
            residual=r.residual_norm, inclination_deg=inclination_deg(r),
            beyond_travel=beyond_travel,
        )

    def _roll_centre(self, lp: CornerPose, rp: CornerPose) -> np.ndarray | None:
        """Kinematic RC of the posed axle in the chassis frame (CP-FVIC lines)."""
        if not (lp.converged and rp.converged):
            return None
        lines = []
        for cp_ in (lp, rp):
            corner = self.geo.corner(cp_.cid)
            dp = {k: DerivedPoint(x_mm=float(p[0]), y_mm=float(p[1]), z_mm=float(p[2]))
                  for k, p in cp_.points.items()}
            # instant-axis FVIC at the patch station; the rack is read back off
            # the posed (rack-displaced) inboard tie-rod point
            rack = float(cp_.points["TIE_ROD_IN"][1]) - corner.tie_rod_inboard.y_mm
            try:
                fv = front_view_instant_centre(corner, ubj=dp["UCA_OUT"], lbj=dp["LCA_OUT"],
                                               contact_patch=dp["CONTACT_PATCH"],
                                               tro=dp["TIE_ROD_OUT"], rack_mm=rack)
            except (RuntimeError, ValueError):   # singular linkage at this pose: no RC
                return None
            if not fv.is_finite:
                return None
            c = cp_.points["CONTACT_PATCH"]
            lines.append((np.array([c[1], c[2]]), np.array([fv.fvic_y_mm, fv.fvic_z_mm])))
        (p1, p2), (p3, p4) = lines
        d1, d2 = p2 - p1, p4 - p3
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-12:
            return None
        t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / den
        y, z = p1 + t * d1
        x = 0.5 * (lp.points["WHEEL_CENTER"][0] + rp.points["WHEEL_CENTER"][0])
        return np.array([x, y, z])


def wheel_ring(centre: np.ndarray, axis: np.ndarray, radius: float, n: int = 48) -> np.ndarray:
    """Points on the tyre's outer circle (for drawing), in the frame of the inputs."""
    a = axis / np.linalg.norm(axis)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, ref)
    u /= np.linalg.norm(u)
    w = np.cross(a, u)
    th = np.linspace(0.0, 2.0 * math.pi, n)
    return centre + radius * (np.outer(np.cos(th), u) + np.outer(np.sin(th), w))
