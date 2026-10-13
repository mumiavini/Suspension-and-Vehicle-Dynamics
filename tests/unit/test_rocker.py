"""Rod + rocker + damper kinematics (vdcore.analysis.rocker).

Constraint closure, rigid-body checks, the motion ratio against an independent
finite difference, the force ratio against the moment balance about the rocker
axis, mirror symmetry, a mount that does not move, and explicit failures.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vdcore.analysis.rocker import (
    RockerLayout,
    RockerSolveError,
    body_transform,
    rocker_state,
    rocker_sweep,
)
from vdcore.geometry.solver import DWSolver
from vdcore.models.hardpoint import Corner, Hardpoint, TirePackage


def _hp(name: str, x: float, y: float, z: float) -> Hardpoint:
    return Hardpoint(name=name, x_mm=x, y_mm=y, z_mm=z, source="cad", tol_mm=0.5)


def _corner(cid: str = "FL", s: float = 1.0) -> Corner:
    return Corner(
        corner_id=cid,  # type: ignore[arg-type]
        uca_inboard_front=_hp("UCA_IF", 120, s * 210, 322),
        uca_inboard_rear=_hp("UCA_IR", -120, s * 210, 310),
        uca_outboard=_hp("UCA_O", -12, s * 537, 385),
        lca_inboard_front=_hp("LCA_IF", 130, s * 175, 117),
        lca_inboard_rear=_hp("LCA_IR", -130, s * 175, 117),
        lca_outboard=_hp("LCA_O", 10, s * 582, 130),
        tie_rod_inboard=_hp("TR_I", -30, s * 270, 157),
        tie_rod_outboard=_hp("TR_O", 85, s * 590, 179),
        wheel_center=_hp("WC", 0, s * 614, 245),
        tire=TirePackage(loaded_radius_mm=245.0, source="cad", tol_mm=1.0),
        static_camber_deg=-1.5,
        static_toe_deg_per_side=0.0,
    )


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _layout(s: float = 1.0, body: str = "lca", mount=(9.0, 542.0, 145.0)) -> RockerLayout:
    """Arms placed so each force acts along its pickup's motion at ride height."""
    m = np.array([mount[0], s * mount[1], mount[2]])
    o = np.array([20.0, s * 180.0, 535.0])
    ud = _unit([540.0, s * -5.0, -35.0])
    rp = o.copy()
    for _ in range(30):
        up = _unit(m - rp)
        a = _unit(np.cross(up, ud))
        rp = o + 50.0 * _unit(np.cross(up, a))
    rd = o + 60.0 * _unit(np.cross(a, ud))
    return RockerLayout(mount_body=body, rod_mount=tuple(m), rocker_pivot=tuple(o),  # type: ignore[arg-type]
                        rocker_axis=tuple(a), rocker_rod=tuple(rp), rocker_damper=tuple(rd),
                        damper_chassis=tuple(rd + 195.0 * ud), rocker_arb=tuple(o + [0, 0, 40]))


C = _corner()
LAY = _layout()
TRAVEL = np.linspace(-25.0, 25.0, 11)


@pytest.fixture(scope="module")
def sweep():
    return rocker_sweep(C, LAY, TRAVEL)


def test_ride_height_is_the_static_layout(sweep) -> None:
    i0 = int(np.argmin(np.abs(TRAVEL)))
    st = sweep.states[i0]
    assert st.psi_deg == pytest.approx(0.0, abs=1e-9)
    assert st.damper_length_mm == pytest.approx(195.0, abs=1e-9)
    assert np.allclose(st.rocker_rod, LAY.rocker_rod, atol=1e-9)
    assert st.rod_angle_deg == pytest.approx(0.0, abs=1e-6)        # built tangential
    assert st.damper_angle_deg == pytest.approx(0.0, abs=1e-6)


def test_closure_and_rigidity(sweep) -> None:
    assert sweep.rod_length_error_mm < 1e-9
    o, a = np.array(LAY.rocker_pivot), _unit(LAY.rocker_axis)
    for st in sweep.states:
        for p0, p in ((LAY.rocker_rod, st.rocker_rod), (LAY.rocker_damper, st.rocker_damper),
                      (LAY.rocker_arb, st.rocker_arb)):
            r0, r = np.array(p0) - o, np.asarray(p) - o
            assert np.linalg.norm(r) == pytest.approx(np.linalg.norm(r0), abs=1e-9)
            assert r @ a == pytest.approx(r0 @ a, abs=1e-9)        # stays in its plane
    # bump compresses the damper and turns the rocker monotonically
    assert np.all(np.diff(sweep.damper_length_mm) < 0) and np.all(np.diff(sweep.psi_deg) != 0)


def test_motion_ratio_matches_an_independent_difference(sweep) -> None:
    h = 0.5
    for z, mr in zip(TRAVEL[1:-1], sweep.motion_ratio[1:-1], strict=True):
        lo = rocker_state(C, LAY, float(z - h)).damper_length_mm
        hi = rocker_state(C, LAY, float(z + h)).damper_length_mm
        assert mr == pytest.approx(-(hi - lo) / (2 * h), abs=2e-5)
    assert 0.5 < float(sweep.motion_ratio[5]) < 1.2


def test_force_ratio_matches_the_moment_balance(sweep) -> None:
    """F_rod·((r_r × u_r)·a) = F_d·((r_d × u_d)·a) about the rocker axis."""
    o, a = np.array(LAY.rocker_pivot), _unit(LAY.rocker_axis)
    for st in sweep.states:
        ur = _unit(st.rod_mount - st.rocker_rod)
        ud = _unit(np.array(LAY.damper_chassis) - st.rocker_damper)
        arm_r = abs(float(np.cross(st.rocker_rod - o, ur) @ a))
        arm_d = abs(float(np.cross(st.rocker_damper - o, ud) @ a))
        assert st.rod_per_damper == pytest.approx(arm_d / arm_r, rel=1e-6)
    # tangential at ride: the force ratio is the arm ratio 60/50
    assert sweep.states[5].rod_per_damper == pytest.approx(60.0 / 50.0, rel=1e-9)


def test_body_transforms_carry_the_joints() -> None:
    s = DWSolver(C)
    for z in (-20.0, 15.0):
        r = s.solve(wheel_travel_mm=z)
        lift = np.array([0.0, 0.0, z])
        lbj = np.array([r.lbj.x_mm, r.lbj.y_mm, r.lbj.z_mm]) + lift
        ubj = np.array([r.ubj.x_mm, r.ubj.y_mm, r.ubj.z_mm]) + lift
        rot, t = body_transform(C, r, z, "lca")
        assert rot @ np.array([10.0, 582.0, 130.0]) + t == pytest.approx(lbj, abs=1e-6)
        assert rot @ np.array([130.0, 175.0, 117.0]) + t == pytest.approx([130, 175, 117],
                                                                           abs=1e-9)
        rot, t = body_transform(C, r, z, "uca")
        assert rot @ np.array([-12.0, 537.0, 385.0]) + t == pytest.approx(ubj, abs=1e-6)
        rot, t = body_transform(C, r, z, "upright")
        assert rot @ np.array([-12.0, 537.0, 385.0]) + t == pytest.approx(ubj, abs=1e-3)
        assert np.allclose(rot @ rot.T, np.eye(3)) and np.linalg.det(rot) == pytest.approx(1.0)


def test_mirror_symmetry() -> None:
    left = rocker_sweep(C, LAY, [-20.0, 0.0, 20.0])
    right = rocker_sweep(_corner("FR", -1.0), _layout(-1.0), [-20.0, 0.0, 20.0])
    assert right.motion_ratio == pytest.approx(left.motion_ratio, abs=1e-6)
    assert right.damper_length_mm == pytest.approx(left.damper_length_mm, abs=1e-6)


def test_a_mount_on_the_pivot_axis_does_not_drive_the_rocker() -> None:
    lay = _layout(mount=(0.0, 175.0, 117.0))          # on the LCA's pivot line
    sw = rocker_sweep(C, lay, [-20.0, 0.0, 20.0])
    assert np.allclose(sw.psi_deg, 0.0, atol=1e-9) and np.allclose(sw.motion_ratio, 0.0,
                                                                    atol=1e-9)


def test_explicit_failures() -> None:
    with pytest.raises(ValueError, match="on the rocker axis"):
        RockerLayout(mount_body="lca", rod_mount=(9, 542, 145), rocker_pivot=(0, 0, 0),
                     rocker_axis=(0, 0, 1), rocker_rod=(0, 0, 50), rocker_damper=(50, 0, 0),
                     damper_chassis=(200, 0, 0))
    with pytest.raises(ValueError, match="zero-length rod"):
        RockerLayout(mount_body="lca", rod_mount=(50, 0, 0), rocker_pivot=(0, 0, 0),
                     rocker_axis=(0, 0, 1), rocker_rod=(50, 0, 0), rocker_damper=(0, 50, 0),
                     damper_chassis=(0, 200, 0))
    # rod arm almost along the rod: the linkage reaches its toggle within the travel
    bad = RockerLayout(mount_body="lca", rod_mount=(9, 542, 145), rocker_pivot=(20, 180, 535),
                       rocker_axis=(0, 0.73, 0.68), rocker_rod=(20, 145, 573),
                       rocker_damper=(80, 180, 535), damper_chassis=(275, 186, 515))
    with pytest.raises(RockerSolveError, match="locks or separates"):
        rocker_sweep(C, bad, [0.0, 10.0])
    assert math.isfinite(rocker_state(C, LAY, 0.0).damper_length_mm)
