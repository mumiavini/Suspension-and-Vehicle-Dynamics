"""Pushrod / pullrod and rocker kinematics on the solved corner.

The rod runs from a mount on a moving body of the corner (the LCA, the UCA or
the upright) to a pickup on a rocker. The rocker turns about an axis fixed in
the chassis and carries the damper pickup (and optionally an anti-roll bar
drop-link pickup); the damper's other end is fixed in the chassis.

For a wheel travel ``z`` (DWSolver convention: positive in bump, the wheel
centre's travel relative to the chassis):

1. :class:`~vdcore.geometry.solver.DWSolver` places the upright.
2. The mount body's rigid motion follows from it: a wishbone rotates about the
   line through its two chassis pivots by the angle that carries its ball
   joint to the solved position; the upright's rotation is the Kabsch fit of
   its four solved points.
3. The rocker angle ``psi`` is the root of
   ``|P_rocker(psi) - P_mount(z)| - L_rod = 0`` on the branch continuous with
   ``psi = 0`` at ride height (``L_rod`` = static rod length).
4. Damper length ``L_d(z) = |D_rocker(psi) - D_chassis|``.

Outputs and their definitions:

* ``motion_ratio = -dL_d/dz`` (damper over wheel; positive when the damper
  compresses in bump). Wheel rate ``= k_spring * MR^2`` follows by virtual work
  for a linear spring (the ``F * dMR/dz`` term from the spring force is not
  included).
* ``rod_per_damper = F_rod / F_damper``: rocker moment balance by virtual work,
  ``(dL_d/dpsi) / (dl/dpsi)`` with ``l`` the rod-pickup distance to the fixed
  mount. Cross-checked in the tests against the moment of each force about
  the rocker axis.
* Transmission angles: between the rod (and the damper) and the rocker arm's
  direction of motion at that pickup. 0 deg = the force acts along the motion
  (ideal); 90 deg = it acts through the axis (no lever, singular).

Every solve can fail; a failure raises :class:`RockerSolveError` rather than
returning a plausible number.

Frame: ISO 8855 (X+ forward, Y+ LEFT, Z+ up), chassis-fixed, mm and degrees.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import brentq

from vdcore.geometry.solver import DWSolver, SolverResult
from vdcore.models.hardpoint import Corner

Arr = NDArray[np.float64]
Vec3 = tuple[float, float, float]
MountBody = Literal["lca", "uca", "upright"]

__all__ = [
    "RockerLayout",
    "RockerSolveError",
    "RockerState",
    "RockerSweep",
    "body_transform",
    "rocker_state",
    "rocker_sweep",
]

# A rigid wishbone keeps its ball joint at a fixed distance from its pivot axis;
# the solver's residual tolerance is far below this.
RIGIDITY_TOL_MM = 1e-4
# Largest rocker rotation searched for, either way from the previous angle.
MAX_PSI_DEG = 120.0
FD_STEP_MM = 0.25           # central difference for the motion ratio
FD_STEP_DEG = 0.01          # central difference in the rocker angle


class RockerSolveError(RuntimeError):
    """The rocker position (or the corner under it) could not be solved."""


def _v(p: object) -> Arr:
    """A point as an array: Hardpoint / DerivedPoint (x_mm, y_mm, z_mm) or 3 numbers."""
    if hasattr(p, "x_mm"):
        return np.array([getattr(p, "x_mm"), getattr(p, "y_mm"), getattr(p, "z_mm")],  # noqa: B009
                        dtype=np.float64)
    return np.asarray(p, dtype=np.float64).reshape(3)


def _unit(v: Arr) -> Arr:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero-length direction")
    return v / n


def _rot(axis: Arr, angle: float) -> Arr:
    """Right-hand rotation matrix about a unit axis (Rodrigues)."""
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    out: Arr = np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)
    return out


class RockerLayout(BaseModel, frozen=True):
    """One corner's rod + rocker + damper, static (ride height) positions.

    All points are in the chassis frame (ISO 8855, mm) for THIS corner -- mirror
    them yourself for the other side. ``rocker_axis`` is a direction (it is
    normalised); its sign only sets the sign of ``psi``.
    """

    mount_body: MountBody
    rod_mount: Vec3                 # on the mount body (LCA / UCA / upright)
    rocker_pivot: Vec3              # a point on the rocker axis
    rocker_axis: Vec3
    rocker_rod: Vec3                # rod pickup on the rocker
    rocker_damper: Vec3             # damper pickup on the rocker
    damper_chassis: Vec3            # damper end fixed in the chassis
    rocker_arb: Vec3 | None = None  # anti-roll-bar drop-link pickup (optional)

    @model_validator(mode="after")
    def _drivable(self) -> RockerLayout:
        a = _unit(_v(self.rocker_axis))
        for name in ("rocker_rod", "rocker_damper"):
            r = _v(getattr(self, name)) - _v(self.rocker_pivot)
            if float(np.linalg.norm(r - (r @ a) * a)) < 1.0:
                raise ValueError(f"{name} lies on the rocker axis: it cannot be driven")
        if float(np.linalg.norm(_v(self.rocker_rod) - _v(self.rod_mount))) < 1.0:
            raise ValueError("rod mount and rocker rod pickup coincide (zero-length rod)")
        return self


def body_transform(corner: Corner, res: SolverResult, wheel_travel_mm: float,
                   body: MountBody) -> tuple[Arr, Arr]:
    """``(R, t)`` taking a static chassis-frame point on ``body`` to its solved position.

    DWSolver reports its points with the chassis displaced by ``-travel`` (the
    wheel stays near its static height); they are shifted back into the chassis
    frame here (pure heave: roll is not handled by this module).
    """
    lift = np.array([0.0, 0.0, wheel_travel_mm])
    if body in ("lca", "uca"):
        if body == "lca":
            p1, p2 = _v(corner.lca_inboard_front), _v(corner.lca_inboard_rear)
            j0, j1 = _v(corner.lca_outboard), _v(res.lbj) + lift
        else:
            p1, p2 = _v(corner.uca_inboard_front), _v(corner.uca_inboard_rear)
            j0, j1 = _v(corner.uca_outboard), _v(res.ubj) + lift
        a = _unit(p2 - p1)
        r0, r1 = j0 - p1, j1 - p1
        q0, q1 = r0 - (r0 @ a) * a, r1 - (r1 @ a) * a
        if abs(float(np.linalg.norm(q0) - np.linalg.norm(q1))) > RIGIDITY_TOL_MM or \
                abs(float(r0 @ a - r1 @ a)) > RIGIDITY_TOL_MM:
            raise RockerSolveError(f"{body.upper()} ball joint left its arc: the solved "
                                   "corner is not a rigid wishbone motion")
        ang = math.atan2(float(a @ np.cross(q0, q1)), float(q0 @ q1))
        rot = _rot(a, ang)
        return rot, p1 - rot @ p1
    if body == "upright":
        s = np.array([_v(corner.uca_outboard), _v(corner.lca_outboard),
                      _v(corner.tie_rod_outboard), _v(corner.wheel_center)])
        d = np.array([_v(res.ubj), _v(res.lbj), _v(res.tro), _v(res.wheel_center)]) + lift
        cs, cd = s.mean(axis=0), d.mean(axis=0)
        h = (s - cs).T @ (d - cd)
        u, _sv, vt = np.linalg.svd(h)
        sgn = np.sign(np.linalg.det(vt.T @ u.T))
        rot = vt.T @ np.diag([1.0, 1.0, sgn]) @ u.T
        return rot, cd - rot @ cs
    raise ValueError(f"unknown mount body {body!r}")


class RockerState(BaseModel):
    """The linkage at one wheel travel. Points: chassis frame, ISO 8855, mm."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    wheel_travel_mm: float
    psi_deg: float                    # rocker rotation from ride height, right-hand about axis
    rod_mount: Arr
    rocker_rod: Arr
    rocker_damper: Arr
    rocker_arb: Arr | None
    damper_length_mm: float
    rod_length_mm: float              # constant by construction; reported as a check
    rod_angle_deg: float              # transmission angle at the rocker rod pickup
    damper_angle_deg: float           # transmission angle at the rocker damper pickup
    rod_per_damper: float             # F_rod / F_damper


class _Rocker:
    def __init__(self, lay: RockerLayout) -> None:
        self.lay = lay
        self.o = _v(lay.rocker_pivot)
        self.a = _unit(_v(lay.rocker_axis))
        self.r = _v(lay.rocker_rod) - self.o
        self.d = _v(lay.rocker_damper) - self.o
        self.b = None if lay.rocker_arb is None else _v(lay.rocker_arb) - self.o
        self.dc = _v(lay.damper_chassis)
        self.l_rod = float(np.linalg.norm(_v(lay.rocker_rod) - _v(lay.rod_mount)))

    def point(self, arm: Arr, psi: float) -> Arr:
        return self.o + _rot(self.a, psi) @ arm

    def rod_gap(self, psi: float, mount: Arr) -> float:
        return float(np.linalg.norm(self.point(self.r, psi) - mount)) - self.l_rod

    def damper_len(self, psi: float) -> float:
        return float(np.linalg.norm(self.point(self.d, psi) - self.dc))

    def solve_psi(self, mount: Arr, guess: float) -> float:
        """Root of the rod constraint on the branch through ``guess``."""
        def f(s: float) -> float:
            return self.rod_gap(s, mount)

        f0 = f(guess)
        if f0 == 0.0:
            return guess
        step = math.radians(0.5)
        lim = math.radians(MAX_PSI_DEG)
        k = 1
        while k * step <= lim:
            for lo, hi in ((guess, guess + k * step), (guess - k * step, guess)):
                flo, fhi = f(lo), f(hi)
                if flo == 0.0:
                    return lo
                if fhi == 0.0:
                    return hi
                if flo * fhi < 0.0:
                    return float(brentq(f, lo, hi, xtol=1e-13, rtol=1e-13, maxiter=200))
            k += 1
        raise RockerSolveError(f"the rod cannot reach the rocker (closest gap "
                               f"{min(abs(f(guess + s)) for s in np.linspace(-lim, lim, 241)):.2f}"
                               " mm): the linkage locks or separates at this travel")


def _transmission(axis: Arr, pivot: Arr, pickup: Arr, line: Arr) -> float:
    """Angle (deg) between a force line and the pickup's direction of motion."""
    tangent = np.cross(axis, pickup - pivot)
    c = abs(float(_unit(tangent) @ _unit(line)))
    return math.degrees(math.acos(min(1.0, c)))


def _state(rk: _Rocker, corner: Corner, solver: DWSolver, z: float,
           guess: float) -> RockerState:
    res = solver.solve(wheel_travel_mm=z)
    if not res.converged:
        raise RockerSolveError(f"corner did not converge at {z:+.2f} mm travel")
    rot, t = body_transform(corner, res, z, rk.lay.mount_body)
    mount = rot @ _v(rk.lay.rod_mount) + t
    psi = rk.solve_psi(mount, guess)
    rr, dd = rk.point(rk.r, psi), rk.point(rk.d, psi)
    h = math.radians(FD_STEP_DEG)
    dld = (rk.damper_len(psi + h) - rk.damper_len(psi - h)) / (2 * h)
    dlr = (rk.rod_gap(psi + h, mount) - rk.rod_gap(psi - h, mount)) / (2 * h)
    if abs(dlr) < 1e-9:
        raise RockerSolveError(f"rod acts through the rocker axis at {z:+.2f} mm: no lever")
    return RockerState(
        wheel_travel_mm=z, psi_deg=math.degrees(psi), rod_mount=mount, rocker_rod=rr,
        rocker_damper=dd, rocker_arb=None if rk.b is None else rk.point(rk.b, psi),
        damper_length_mm=rk.damper_len(psi), rod_length_mm=float(np.linalg.norm(rr - mount)),
        rod_angle_deg=_transmission(rk.a, rk.o, rr, mount - rr),
        damper_angle_deg=_transmission(rk.a, rk.o, dd, rk.dc - dd),
        rod_per_damper=abs(dld / dlr))


def rocker_state(corner: Corner, layout: RockerLayout, wheel_travel_mm: float = 0.0
                 ) -> RockerState:
    """The linkage at one wheel travel, on the branch continuous with ride height."""
    rk, solver = _Rocker(layout), DWSolver(corner)
    guess = 0.0
    n = max(1, math.ceil(abs(wheel_travel_mm) / 2.0))      # walk out in <= 2 mm steps
    for z in np.linspace(0.0, wheel_travel_mm, n + 1)[1:]:
        guess = math.radians(_state(rk, corner, solver, float(z), guess).psi_deg)
    return _state(rk, corner, solver, wheel_travel_mm, guess)


class RockerSweep(BaseModel):
    """The linkage over a set of wheel travels (arrays in the order asked for)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    travel_mm: Arr
    psi_deg: Arr
    damper_length_mm: Arr
    motion_ratio: Arr                 # -dL_damper/dz (damper / wheel)
    rod_per_damper: Arr
    rod_angle_deg: Arr
    damper_angle_deg: Arr
    rod_length_error_mm: float        # max |rod length - static|: a solve check
    states: list[RockerState] = Field(default_factory=list, repr=False)


def rocker_sweep(corner: Corner, layout: RockerLayout, travels_mm: Arr | list[float]
                 ) -> RockerSweep:
    """The linkage over a set of wheel travels (any order; solved outward from 0)."""
    zs = np.asarray(travels_mm, dtype=np.float64)
    rk, solver = _Rocker(layout), DWSolver(corner)
    order = np.argsort(np.abs(zs), kind="stable")
    states: dict[int, RockerState] = {}
    psi_at: dict[float, float] = {0.0: 0.0}

    def solve_at(z: float) -> RockerState:
        near = min(psi_at, key=lambda k: abs(k - z))
        guess, start = psi_at[near], near
        n = max(1, math.ceil(abs(z - start) / 2.0))
        st = None
        for zz in np.linspace(start, z, n + 1)[1:]:
            st = _state(rk, corner, solver, float(zz), guess)
            guess = math.radians(st.psi_deg)
            psi_at[float(zz)] = guess
        assert st is not None
        return st

    mr = np.empty_like(zs)
    for i in order:
        z = float(zs[i])
        st = solve_at(z) if z != 0.0 else _state(rk, corner, solver, 0.0, 0.0)
        states[i] = st
        lo = _state(rk, corner, solver, z - FD_STEP_MM, math.radians(st.psi_deg))
        hi = _state(rk, corner, solver, z + FD_STEP_MM, math.radians(st.psi_deg))
        mr[i] = -(hi.damper_length_mm - lo.damper_length_mm) / (2 * FD_STEP_MM)
    ss = [states[i] for i in range(len(zs))]
    return RockerSweep(
        travel_mm=zs, psi_deg=np.array([s.psi_deg for s in ss]),
        damper_length_mm=np.array([s.damper_length_mm for s in ss]), motion_ratio=mr,
        rod_per_damper=np.array([s.rod_per_damper for s in ss]),
        rod_angle_deg=np.array([s.rod_angle_deg for s in ss]),
        damper_angle_deg=np.array([s.damper_angle_deg for s in ss]),
        rod_length_error_mm=max(abs(s.rod_length_mm - rk.l_rod) for s in ss), states=ss)
