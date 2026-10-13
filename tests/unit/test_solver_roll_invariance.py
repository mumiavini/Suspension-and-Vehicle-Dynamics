"""DWSolver under chassis roll: a rigid chassis motion must not change the linkage.

Rolling the chassis rotates every inboard pickup about X. Expressed back in the
CHASSIS frame, the solved joints must be exactly the unrolled solution at the
same wheel travel -- roll moves the whole car, it does not reshape the
suspension. The axial-position residuals used to compare the moved ball joint
against the STATIC points projected onto the MOVED pivot axis, which is not the
static axial position once the axis has Y/Z components: an inclined/raked
pivot axis shifted the ball joints by ~0.4 mm per degree of roll.

Frame: ISO 8855 (X fwd, Y LEFT, Z up). DWSolver applies travel as a chassis
displacement of -wheel_travel_mm and roll as a rotation about +X through the
origin, so a chassis point c sits at R c + dz with dz = (0, 0, -travel). The
unrolled solve at the same travel puts it at c + dz, so a rolled joint w maps
onto the unrolled solution as R^T (w - dz) + dz.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vdcore.geometry.solver import DWSolver, SolverResult
from vdcore.models.hardpoint import Corner, Hardpoint, TirePackage


def _hp(name: str, x: float, y: float, z: float) -> Hardpoint:
    return Hardpoint(name=name, x_mm=x, y_mm=y, z_mm=z, source="cad", tol_mm=0.5)


def _inclined_corner(cid: str, y_sign: float) -> Corner:
    """Both pivot axes have Y and Z components (rake + plan-view angle)."""
    s = y_sign
    return Corner(
        corner_id=cid,  # type: ignore[arg-type]
        uca_inboard_front=_hp("UCA_IF", 90, s * 160, 300),
        uca_inboard_rear=_hp("UCA_IR", -110, s * 190, 270),
        uca_outboard=_hp("UCA_O", -10, s * 530, 290),
        lca_inboard_front=_hp("LCA_IF", 120, s * 120, 95),
        lca_inboard_rear=_hp("LCA_IR", -130, s * 150, 70),
        lca_outboard=_hp("LCA_O", 10, s * 580, 75),
        tie_rod_inboard=_hp("TR_I", -60, s * 160, 120),
        tie_rod_outboard=_hp("TR_O", -50, s * 540, 110),
        wheel_center=_hp("WC", 0, s * 600, 228),
        tire=TirePackage(loaded_radius_mm=228.0, source="cad", tol_mm=1.0),
        static_camber_deg=-1.5,
        static_toe_deg_per_side=0.1,
    )


def _joints_in_chassis(res: SolverResult, roll_deg: float, travel_mm: float) -> np.ndarray:
    rot = Rotation.from_rotvec(math.radians(roll_deg) * np.array([1.0, 0.0, 0.0])).as_matrix()
    dz = np.array([0.0, 0.0, -travel_mm])
    pts = []
    for p in (res.ubj, res.lbj, res.tro, res.wheel_center):
        w = np.array([p.x_mm, p.y_mm, p.z_mm])
        pts.append(rot.T @ (w - dz) + dz)
    return np.array(pts)


@pytest.mark.parametrize("cid,y_sign", [("FL", 1.0), ("FR", -1.0)])
@pytest.mark.parametrize("travel_mm", [-20.0, 0.0, 20.0])
@pytest.mark.parametrize("roll_deg", [-3.0, 1.5, 3.0])
def test_roll_is_a_rigid_motion(cid: str, y_sign: float, travel_mm: float,
                                roll_deg: float) -> None:
    solver = DWSolver(_inclined_corner(cid, y_sign))
    ref = solver.solve(wheel_travel_mm=travel_mm)
    rolled = solver.solve(wheel_travel_mm=travel_mm, roll_deg=roll_deg)
    assert ref.converged and rolled.converged
    ref_pts = np.array([[p.x_mm, p.y_mm, p.z_mm]
                        for p in (ref.ubj, ref.lbj, ref.tro, ref.wheel_center)])
    err = np.abs(_joints_in_chassis(rolled, roll_deg, travel_mm) - ref_pts).max()
    # Before the fix this was ~0.4 mm per degree of roll on this corner.
    assert err < 1e-6, f"rolled linkage differs from the unrolled one by {err:.3e} mm"


def test_axial_position_is_kept_under_roll() -> None:
    """Both ball joints keep their static distance along their pivot axes."""
    corner = _inclined_corner("FL", 1.0)
    solver = DWSolver(corner)
    res = solver.solve(wheel_travel_mm=10.0, roll_deg=2.5)
    rot = Rotation.from_rotvec(math.radians(2.5) * np.array([1.0, 0.0, 0.0])).as_matrix()
    dz = np.array([0.0, 0.0, -10.0])

    def axial(front: Hardpoint, rear: Hardpoint, joint: np.ndarray) -> float:
        f = rot @ np.array([front.x_mm, front.y_mm, front.z_mm]) + dz
        r = rot @ np.array([rear.x_mm, rear.y_mm, rear.z_mm]) + dz
        a = (r - f) / np.linalg.norm(r - f)
        return float(np.dot(joint - f, a))

    def axial0(front: Hardpoint, rear: Hardpoint, joint: Hardpoint) -> float:
        f = np.array([front.x_mm, front.y_mm, front.z_mm])
        r = np.array([rear.x_mm, rear.y_mm, rear.z_mm])
        a = (r - f) / np.linalg.norm(r - f)
        return float(np.dot(np.array([joint.x_mm, joint.y_mm, joint.z_mm]) - f, a))

    ubj = np.array([res.ubj.x_mm, res.ubj.y_mm, res.ubj.z_mm])
    lbj = np.array([res.lbj.x_mm, res.lbj.y_mm, res.lbj.z_mm])
    assert axial(corner.uca_inboard_front, corner.uca_inboard_rear, ubj) == pytest.approx(
        axial0(corner.uca_inboard_front, corner.uca_inboard_rear, corner.uca_outboard),
        abs=1e-8)
    assert axial(corner.lca_inboard_front, corner.lca_inboard_rear, lbj) == pytest.approx(
        axial0(corner.lca_inboard_front, corner.lca_inboard_rear, corner.lca_outboard),
        abs=1e-8)
