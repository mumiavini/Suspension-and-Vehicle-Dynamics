"""Front-view instant centre from the upright's instantaneous 3D motion.

Coordinate system: ISO 8855 -- X+ forward, Y+ LEFT, Z+ up. Every Z here is
CHASSIS-referenced (DWSolver results are raised by their wheel travel).

``vdcore.analysis.roll_centre.instant_axis_fvic`` replaced the intersection of
the projected arm lines on 2026-10-12. The projected lines are each taken at
their own ball joint's X; with caster the two ball joints sit at different X,
and the front-view pole of a rigid upright moves with X by
``(w_y / w_x, w_z / w_x)``, so the two lines pass through different points and
their intersection is not the instant centre at the contact patch. What is
pinned here:

1. The analytic twist (positions only, no extra solves) equals central finite
   differences of DWSolver, on all four 2027 corners, an asymmetric corner, at
   several travels and with the rack displaced.
2. Where both ball joints and the patch share one X station the two
   constructions agree exactly, and both hit a hand-computed answer.
3. The 2027 rear (no caster) is unchanged by the switch; the 2027 front's RC
   migration is -0.1999 mm/mm (the projected lines gave -0.3106).
4. Failures are explicit: parallel arms put the FVIC at infinity, a singular
   linkage raises, and world-frame points are refused.
5. ISO 8855 signs: mirrored FVICs, RC on y = 0, opposite front-view rotation.
6. ``sample_corner`` needs no state beyond the travel it was asked for, so the
   Altair MotionSolve replay keeps working.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import geometry_summary as gs  # noqa: E402

import sla_geometry as sla  # noqa: E402
import steering_geometry as stg  # noqa: E402
from vdcore.analysis.axle import axle_rates, sample_corner  # noqa: E402
from vdcore.analysis.roll_centre import (  # noqa: E402
    InstantKinematicsError,
    _rc_height_from_cp_and_ic,
    front_view_instant_centre,
    instant_axis_fvic,
    projected_fvic,
    roll_centre_height,
    roll_centre_migration,
    upright_twist,
)
from vdcore.geometry.solver import DWSolver, SolverResult  # noqa: E402
from vdcore.models.hardpoint import (  # noqa: E402
    Axle,
    Corner,
    DerivedPoint,
    Hardpoint,
    TirePackage,
)

# --------------------------------------------------------------------------- #
# fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def car() -> dict[str, Axle]:
    """The merged 2027 car (the only complete one -- CLAUDE.md)."""
    design = sla.run()
    merged = gs.build_merged(design, stg.run())

    def axle(inp: sla.AxleInputs, left: str, right: str) -> Axle:
        return Axle(
            left=gs._vdcore_corner(merged, left, inp),
            right=gs._vdcore_corner(merged, right, inp),
        )

    return {
        "front": axle(design.front.inputs, "FL", "FR"),
        "rear": axle(design.rear.inputs, "RL", "RR"),
    }


def _corner(car: dict[str, Axle], cid: str) -> Corner:
    ax = car["front" if cid.startswith("F") else "rear"]
    return ax.left if cid.endswith("L") else ax.right


def _chassis(p: DerivedPoint, travel_mm: float) -> DerivedPoint:
    """DWSolver output (chassis displaced by -travel) -> chassis frame."""
    return DerivedPoint(x_mm=p.x_mm, y_mm=p.y_mm, z_mm=p.z_mm + travel_mm)


def _vec(p: DerivedPoint) -> np.ndarray:
    return np.array([p.x_mm, p.y_mm, p.z_mm])


def _moved(hp: Hardpoint, dx: float, dy: float, dz: float) -> Hardpoint:
    return hp.model_copy(
        update={"x_mm": hp.x_mm + dx, "y_mm": hp.y_mm + dy, "z_mm": hp.z_mm + dz}
    )


def _asymmetric(corner: Corner) -> Corner:
    """The corner with three chassis pickups moved by a few mm, so nothing is
    mirrored or aligned with an axis any more."""
    return corner.model_copy(
        update={
            "uca_inboard_front": _moved(corner.uca_inboard_front, 6.0, -3.0, 4.0),
            "lca_inboard_rear": _moved(corner.lca_inboard_rear, -4.0, 2.0, -5.0),
            "tie_rod_inboard": _moved(corner.tie_rod_inboard, 3.0, 0.0, -6.0),
        }
    )


def _hp(name: str, x: float, y: float, z: float) -> Hardpoint:
    return Hardpoint(name=name, x_mm=x, y_mm=y, z_mm=z, source="design_intent", tol_mm=0.5)


def _planar(
    corner_id: str,
    uca_out_z: float = 360.0,
    uca_in_y: float = 200.0,
    tro_x: float = -70.0,
) -> Corner:
    """Every pivot axis along X; both ball joints and the wheel centre at x = 0.

    Front view of the left corner: LCA horizontal at z = 120 from y = 200 to
    580, UCA from (200, 340) to (580, ``uca_out_z``). The steering arm (TRO)
    sits ``tro_x`` behind the axle, so the tie rod is out of the plane.
    """
    s = 1.0 if corner_id in ("FL", "RL") else -1.0
    return Corner(
        corner_id=corner_id,  # type: ignore[arg-type]
        uca_inboard_front=_hp("UCA_IF", 100.0, s * uca_in_y, 340.0),
        uca_inboard_rear=_hp("UCA_IR", -100.0, s * uca_in_y, 340.0),
        uca_outboard=_hp("UCA_O", 0.0, s * 580.0, uca_out_z),
        lca_inboard_front=_hp("LCA_IF", 120.0, s * 200.0, 120.0),
        lca_inboard_rear=_hp("LCA_IR", -120.0, s * 200.0, 120.0),
        lca_outboard=_hp("LCA_O", 0.0, s * 580.0, 120.0),
        tie_rod_inboard=_hp("TR_I", tro_x, s * 210.0, 200.0),
        tie_rod_outboard=_hp("TR_O", tro_x, s * 570.0, 205.0),
        wheel_center=_hp("WC", 0.0, s * 620.0, 250.0),
        tire=TirePackage(loaded_radius_mm=250.0, source="design_intent", tol_mm=1.0),
        static_camber_deg=0.0,
        static_toe_deg_per_side=0.0,
    )


# --------------------------------------------------------------------------- #
# 1. analytic twist == finite differences of DWSolver
# --------------------------------------------------------------------------- #

_H_MM = 0.05  # central-difference half step
# Measured worst case 2.4e-9 over these states; the gap to 1e-6 is the margin
# for solver tolerance, not for a formula error (a wrong sign or frame shows
# up at order 1).
_TWIST_REL_TOL = 1e-6

_TWIST_CASES = [
    (cid, travel, rack)
    for cid in ("FL", "FR", "RL", "RR", "FR-asym", "RL-asym")
    for travel in (-25.0, -10.0, 0.0, 12.5, 25.0)
    for rack in ((0.0, 8.0, -8.0) if cid.startswith("F") else (0.0,))
]


@pytest.mark.parametrize(("cid", "travel", "rack"), _TWIST_CASES)
def test_analytic_twist_matches_finite_differences(
    car: dict[str, Axle], cid: str, travel: float, rack: float
) -> None:
    corner = _corner(car, cid[:2])
    if cid.endswith("asym"):
        corner = _asymmetric(corner)
    solver = DWSolver(corner)
    r0 = solver.solve(wheel_travel_mm=travel, rack_mm=rack)
    rp = solver.solve(wheel_travel_mm=travel + _H_MM, rack_mm=rack)
    rm = solver.solve(wheel_travel_mm=travel - _H_MM, rack_mm=rack)
    assert r0.converged and rp.converged and rm.converged

    names = ("ubj", "lbj", "tro", "wheel_center")
    fd = {
        n: (
            _vec(_chassis(getattr(rp, n), travel + _H_MM))
            - _vec(_chassis(getattr(rm, n), travel - _H_MM))
        ) / (2.0 * _H_MM)
        for n in names
    }
    base = {n: _chassis(getattr(r0, n), travel) for n in names}
    tw = upright_twist(
        corner, base["ubj"], base["lbj"], base["tro"], base["wheel_center"], rack_mm=rack
    )

    scale = max(float(np.linalg.norm(v)) for v in fd.values())
    for n in names:
        analytic = np.array(tw.velocity_at(base[n]))
        err = float(np.linalg.norm(analytic - fd[n])) / scale
        assert err < _TWIST_REL_TOL, f"{cid} {n} at {travel:+} mm, rack {rack:+}: {err:.2e}"
    # Normalised per mm of travel: the wheel centre rises 1 mm per mm in the
    # chassis frame, by DWSolver's definition of travel.
    assert tw.velocity_at(base["wheel_center"])[2] == pytest.approx(1.0, abs=1e-12)


def test_twist_rotates_the_upright_inboard_at_the_top_in_bump(car: dict[str, Axle]) -> None:
    """ISO 8855 sign of the front-view rotation: about +X on the left (top
    moves toward -Y, inboard), about -X on the right -- negative camber gain."""
    for cid, sign in (("FL", 1.0), ("RL", 1.0), ("FR", -1.0), ("RR", -1.0)):
        corner = _corner(car, cid)
        r = DWSolver(corner).solve()
        tw = upright_twist(corner, r.ubj, r.lbj, r.tro, r.wheel_center)
        assert sign * tw.omega_rad_per_mm[0] > 1e-4, cid


# --------------------------------------------------------------------------- #
# 2. known answer: ball joints and patch on one X station
# --------------------------------------------------------------------------- #


class TestPlanarKnownAnswer:
    """The projected construction is exact when both ball joints and the patch
    share one X station, so there the two methods must agree -- and both must
    hit the hand computation.

    Lines: LCA z = 120; UCA z = 340 + (20/380)(y - 200). They meet at
    y = 200 - 220 * 380 / 20 = -3980, z = 120. Zero camber puts the patch at
    (620, 0), so the RC is 120 * 620 / (620 + 3980) = 16.174 mm.
    """

    def test_both_methods_hit_the_hand_computed_instant_centre(self) -> None:
        corner = _planar("FL")
        r = DWSolver(corner).solve()
        assert r.converged
        new = instant_axis_fvic(corner, r.ubj, r.lbj, r.tro, r.contact_patch)
        old = projected_fvic(corner, r.ubj, r.lbj, r.contact_patch)
        assert new.is_finite and old.is_finite
        assert new.fvic_y_mm == pytest.approx(-3980.0, abs=1e-6)
        assert new.fvic_z_mm == pytest.approx(120.0, abs=1e-6)
        assert new.fvic_y_mm == pytest.approx(old.fvic_y_mm, abs=1e-6)
        assert new.fvic_z_mm == pytest.approx(old.fvic_z_mm, abs=1e-6)
        assert new.fvsa_mm == pytest.approx(old.fvsa_mm, abs=1e-6)
        # FVSA: patch (620, 0) to IC (-3980, 120), inboard -> negative
        assert new.fvsa_mm == pytest.approx(-math.hypot(4600.0, 120.0), abs=1e-6)

    @pytest.mark.parametrize("tro_x", [-70.0, 60.0])
    def test_result_does_not_depend_on_the_steering_arm(self, tro_x: float) -> None:
        """With every point on one station the tie rod only sets the yaw rate,
        which does not move the pole of that station."""
        corner = _planar("FL", tro_x=tro_x)
        r = DWSolver(corner).solve()
        new = instant_axis_fvic(corner, r.ubj, r.lbj, r.tro, r.contact_patch)
        assert new.fvic_y_mm == pytest.approx(-3980.0, abs=1e-6)
        assert new.fvic_z_mm == pytest.approx(120.0, abs=1e-6)

    def test_roll_centre_matches_the_hand_computation(self) -> None:
        axle = Axle(left=_planar("FL"), right=_planar("FR"))
        rc = roll_centre_height(axle, DWSolver(axle.left).solve(), DWSolver(axle.right).solve())
        assert rc.rc_height_mm == pytest.approx(120.0 * 620.0 / 4600.0, abs=1e-6)
        assert rc.rc_y_mm == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 3. the 2027 car
# --------------------------------------------------------------------------- #


def test_rear_without_caster_equals_the_projected_construction(car: dict[str, Axle]) -> None:
    """Rear ball joints and patch all sit at x = -1540: the switch must not
    move the rear (measured worst 8.9e-5 mm of RC over +/-25 mm)."""
    for corner in (car["rear"].left, car["rear"].right):
        solver = DWSolver(corner)
        for t in (-25.0, -12.5, 0.0, 12.5, 25.0):
            r = solver.solve(wheel_travel_mm=t)
            assert r.converged
            cp = _chassis(r.contact_patch, t)
            new = instant_axis_fvic(
                corner, _chassis(r.ubj, t), _chassis(r.lbj, t), _chassis(r.tro, t), cp
            )
            old = projected_fvic(corner, _chassis(r.ubj, t), _chassis(r.lbj, t), cp)
            rc_new = _rc_height_from_cp_and_ic(cp.y_mm, cp.z_mm, new.fvic_y_mm, new.fvic_z_mm)
            rc_old = _rc_height_from_cp_and_ic(cp.y_mm, cp.z_mm, old.fvic_y_mm, old.fvic_z_mm)
            assert rc_new == pytest.approx(rc_old, abs=1e-3), (corner.corner_id, t)


def test_front_migration_is_the_instant_axis_value(car: dict[str, Axle]) -> None:
    """5 deg caster puts the front UBJ at x = -12.3 and the LBJ at +10.1.

    The instant-axis RC migrates -0.1999 mm/mm (chassis-referenced); the
    projected arm lines, read at those two stations, gave -0.3106. The
    auditor's independent finite-difference screw-axis RC gave -0.1999.
    """
    rates = axle_rates(car["front"])
    assert rates.rc_migration_mm_per_mm == pytest.approx(-0.1999, abs=0.0005)
    assert rates.rc_min_mm == pytest.approx(30.67, abs=0.02)
    assert rates.rc_max_mm == pytest.approx(40.70, abs=0.02)

    # The cross-check construction still reproduces the old number, so the
    # gap is a difference of definition, not of solver.
    corner = car["front"].left
    solver = DWSolver(corner)
    step = 25.0 / 20.0
    rc = []
    for t in (+step, -step):
        r = solver.solve(wheel_travel_mm=t)
        cp = _chassis(r.contact_patch, t)
        f = projected_fvic(corner, _chassis(r.ubj, t), _chassis(r.lbj, t), cp)
        rc.append(_rc_height_from_cp_and_ic(cp.y_mm, cp.z_mm, f.fvic_y_mm, f.fvic_z_mm))
    assert (rc[0] - rc[1]) / (2.0 * step) == pytest.approx(-0.3106, abs=0.0005)


def test_static_front_roll_centre(car: dict[str, Axle]) -> None:
    ax = car["front"]
    rc = roll_centre_height(ax, DWSolver(ax.left).solve(), DWSolver(ax.right).solve())
    assert rc.rc_height_mm == pytest.approx(35.373, abs=0.002)  # projected: 35.500


@pytest.mark.parametrize("travel", [-20.0, 0.0, 15.0])
def test_axle_and_single_corner_constructions_agree(
    car: dict[str, Axle], travel: float
) -> None:
    """roll_centre_height (two lines crossed) == sample_corner (one line to
    y = 0) == roll_centre_migration on a symmetric axle at symmetric travel."""
    ax = car["front"]
    rl = DWSolver(ax.left).solve(wheel_travel_mm=travel)
    rr = DWSolver(ax.right).solve(wheel_travel_mm=travel)
    both = roll_centre_height(ax, rl, rr, left_travel_mm=travel, right_travel_mm=travel)
    one = sample_corner(ax.left, DWSolver(ax.left), travel)
    sweep = roll_centre_migration(ax, travel, travel, steps=1)
    assert both.rc_height_mm == pytest.approx(one.rc_height_mm, abs=1e-6)
    assert sweep.rc_height_mm[0] == pytest.approx(one.rc_height_mm, abs=1e-6)
    assert both.left_fvic.fvic_y_mm == pytest.approx(one.fvic_y_mm, abs=1e-9)


# --------------------------------------------------------------------------- #
# 4. explicit failures
# --------------------------------------------------------------------------- #


def test_parallel_arms_put_the_instant_centre_at_infinity() -> None:
    """Equal parallel arms: the upright translates in front view (w_x = 0)."""
    corner = _planar("FL", uca_out_z=340.0)
    r = DWSolver(corner).solve()
    fvic = instant_axis_fvic(corner, r.ubj, r.lbj, r.tro, r.contact_patch)
    assert not fvic.is_finite
    assert math.isinf(fvic.fvsa_mm) and math.isinf(fvic.fvic_y_mm)

    axle = Axle(left=corner, right=_planar("FR", uca_out_z=340.0))
    with pytest.raises(RuntimeError, match="infinity"):
        roll_centre_height(axle, r, DWSolver(axle.right).solve())


def _static_points(corner: Corner) -> dict[str, DerivedPoint]:
    """The corner's own outboard points: the chassis frame at zero travel."""
    return {
        n: DerivedPoint(x_mm=p.x_mm, y_mm=p.y_mm, z_mm=p.z_mm)
        for n, p in (
            ("ubj", corner.uca_outboard),
            ("lbj", corner.lca_outboard),
            ("tro", corner.tie_rod_outboard),
            ("wc", corner.wheel_center),
        )
    }


def test_singular_linkage_raises() -> None:
    """A tie rod that duplicates the lower arm's front leg (outboard at the
    LBJ, inboard on the LCA pivot axis) constrains nothing the LCA does not:
    the upright can also steer freely, two independent motions, so no unique
    instant centre."""
    base = _planar("FL")
    corner = base.model_copy(
        update={
            "tie_rod_outboard": base.lca_outboard.model_copy(update={"name": "TR_O"}),
            "tie_rod_inboard": base.lca_inboard_front.model_copy(update={"name": "TR_I"}),
        }
    )
    p = _static_points(corner)
    with pytest.raises(InstantKinematicsError, match="singular"):
        instant_axis_fvic(corner, p["ubj"], p["lbj"], p["tro"], p["wc"])


def test_locked_suspension_has_no_instant_centre() -> None:
    """A tie rod aimed at the kingpin locks the wheel travel: the only motion
    left is steer about the (vertical) kingpin, which has no front-view
    rotation. The FVIC is at infinity and the twist cannot be expressed per
    mm of travel -- both say so instead of returning a number."""
    base = _planar("FL")
    mid = 0.5 * (_vec(base.uca_outboard) + _vec(base.lca_outboard))
    corner = base.model_copy(
        update={"tie_rod_outboard": _hp("TR_O", float(mid[0]), float(mid[1]), float(mid[2]))}
    )
    p = _static_points(corner)
    assert not instant_axis_fvic(corner, p["ubj"], p["lbj"], p["tro"], p["wc"]).is_finite
    with pytest.raises(ValueError, match="no vertical velocity"):
        upright_twist(corner, p["ubj"], p["lbj"], p["tro"], p["wc"])


def test_world_frame_points_are_refused(car: dict[str, Axle]) -> None:
    """GUARD FOR the frame mix the projected construction was fed for years:
    DWSolver points at 25 mm travel, not raised into the chassis frame."""
    ax = car["front"]
    rl = DWSolver(ax.left).solve(wheel_travel_mm=25.0)
    rr = DWSolver(ax.right).solve(wheel_travel_mm=25.0)
    with pytest.raises(ValueError, match="chassis-frame"):
        roll_centre_height(ax, rl, rr)
    with pytest.raises(ValueError, match="chassis-frame"):
        instant_axis_fvic(ax.left, rl.ubj, rl.lbj, rl.tro, rl.contact_patch)


def test_wrong_rack_is_refused(car: dict[str, Axle]) -> None:
    corner = car["front"].left
    r = DWSolver(corner).solve(rack_mm=8.0)
    ok = instant_axis_fvic(corner, r.ubj, r.lbj, r.tro, r.contact_patch, rack_mm=8.0)
    assert ok.is_finite
    with pytest.raises(ValueError, match="tie-rod length"):
        instant_axis_fvic(corner, r.ubj, r.lbj, r.tro, r.contact_patch)


# --------------------------------------------------------------------------- #
# 5. ISO 8855 signs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("axle_name", ["front", "rear"])
@pytest.mark.parametrize("travel", [-15.0, 0.0, 20.0])
def test_left_and_right_mirror_and_the_rc_is_on_the_centreline(
    car: dict[str, Axle], axle_name: str, travel: float
) -> None:
    ax = car[axle_name]
    rl = DWSolver(ax.left).solve(wheel_travel_mm=travel)
    rr = DWSolver(ax.right).solve(wheel_travel_mm=travel)
    rc = roll_centre_height(ax, rl, rr, left_travel_mm=travel, right_travel_mm=travel)
    # Y+ is LEFT: the left corner's instant centre is on the far (right, -Y)
    # side for an SLA, the right corner's on +Y.
    assert rc.left_fvic.fvic_y_mm < 0.0 < rc.right_fvic.fvic_y_mm
    assert rc.left_fvic.fvic_y_mm == pytest.approx(-rc.right_fvic.fvic_y_mm, abs=1e-6)
    assert rc.left_fvic.fvic_z_mm == pytest.approx(rc.right_fvic.fvic_z_mm, abs=1e-6)
    assert rc.left_fvic.fvsa_mm == pytest.approx(rc.right_fvic.fvsa_mm, abs=1e-6)
    assert rc.rc_y_mm == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# 6. API
# --------------------------------------------------------------------------- #


def test_front_view_instant_centre_dispatches_on_tro(car: dict[str, Axle]) -> None:
    corner = car["front"].left
    r = DWSolver(corner).solve()
    with_tro = front_view_instant_centre(corner, r.ubj, r.lbj, r.contact_patch, tro=r.tro)
    without = front_view_instant_centre(corner, r.ubj, r.lbj, r.contact_patch)
    assert with_tro == instant_axis_fvic(corner, r.ubj, r.lbj, r.tro, r.contact_patch)
    assert without == projected_fvic(corner, r.ubj, r.lbj, r.contact_patch)
    with pytest.raises(ValueError, match="rack_mm needs tro"):
        front_view_instant_centre(corner, r.ubj, r.lbj, r.contact_patch, rack_mm=5.0)


class _OneStateSolver:
    """Holds one solved state and refuses every other request -- the contract
    of the Altair MotionSolve replay, which cannot solve new travels."""

    def __init__(self, travel_mm: float, result: SolverResult) -> None:
        self.travel_mm = travel_mm
        self.result = result
        self.calls = 0

    def solve(
        self, wheel_travel_mm: float = 0.0, roll_deg: float = 0.0, rack_mm: float = 0.0
    ) -> SolverResult:
        self.calls += 1
        if wheel_travel_mm != self.travel_mm or roll_deg != 0.0 or rack_mm != 0.0:
            raise KeyError(f"no state at {wheel_travel_mm} mm")
        return self.result


def test_sample_corner_needs_only_the_requested_state(car: dict[str, Axle]) -> None:
    corner = car["front"].left
    travel = 13.75
    replay = _OneStateSolver(travel, DWSolver(corner).solve(wheel_travel_mm=travel))
    s = sample_corner(corner, replay, travel)
    assert replay.calls == 1
    ref = sample_corner(corner, DWSolver(corner), travel)
    assert s.rc_height_mm == pytest.approx(ref.rc_height_mm, abs=1e-9)
    assert s.fvsa_mm == pytest.approx(ref.fvsa_mm, abs=1e-9)
