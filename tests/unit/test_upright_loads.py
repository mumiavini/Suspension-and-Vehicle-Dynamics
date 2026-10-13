"""Six-link upright equilibrium (vdcore.analysis.upright_loads).

Known answers, frame independence, left/right mirror, an explicit failure for a
mechanism, and an independent cross-check by virtual work on DWSolver: with the
four legs and the tie rod rigid the upright has one DOF (travel), the pushrod
alone resists it, so F . v(load point) + f_p * (u_p . v(pushrod end)) = 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from vdcore.analysis.upright_loads import (
    LINKS,
    SingularLinkLayoutError,
    upright_link_forces,
)
from vdcore.geometry.solver import DWSolver, SolverResult
from vdcore.models.hardpoint import Corner, Hardpoint, TirePackage


def _hp(name: str, x: float, y: float, z: float) -> Hardpoint:
    return Hardpoint(name=name, x_mm=x, y_mm=y, z_mm=z, source="cad", tol_mm=0.5)


def _corner(cid: str = "FL", s: float = 1.0) -> Corner:
    """A raked, swept, caster-ed corner (BJs at different X), wheel centre at r = 245."""
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


def _p(h: Hardpoint) -> np.ndarray:
    return np.array([h.x_mm, h.y_mm, h.z_mm])


C = _corner()
LBJ, UBJ = _p(C.lca_outboard), _p(C.uca_outboard)
PR_O, PR_I = LBJ, LBJ + np.array([0.0, -250.0, 300.0])
CP = np.array([0.0, 620.0, 0.0])


def test_a_load_along_one_link_loads_only_that_link() -> None:
    r = upright_link_forces(C, (0, 0, 1000), LBJ, LBJ, LBJ + [0, 0, 300])   # vertical pushrod
    assert r.pushrod_N == pytest.approx(-1000.0, abs=1e-9)                  # compression
    assert max(abs(v) for k, v in r.as_dict().items() if k != "pushrod") < 1e-9
    u = _p(C.uca_inboard_front) - UBJ
    r = upright_link_forces(C, 1000 * u / np.linalg.norm(u), UBJ, PR_O, PR_I)
    assert r.uca_front_N == pytest.approx(-1000.0, abs=1e-9)   # pushes the upright inboard
    assert max(abs(v) for k, v in r.as_dict().items() if k != "uca_front") < 1e-9


@pytest.mark.parametrize("load", [(0, -1851, 1234), (-1728, 0, 1234), (-1210, -1296, 1234)])
def test_equilibrium_closes_and_does_not_depend_on_the_moment_origin(load) -> None:
    r = upright_link_forces(C, load, CP, PR_O, PR_I)
    assert r.residual_force_N < 1e-8 and r.residual_moment_Nmm < 1e-6
    links = [(UBJ, _p(C.uca_inboard_front)), (UBJ, _p(C.uca_inboard_rear)),
             (LBJ, _p(C.lca_inboard_front)), (LBJ, _p(C.lca_inboard_rear)),
             (_p(C.tie_rod_outboard), _p(C.tie_rod_inboard)), (PR_O, PR_I)]
    f = np.array([r.as_dict()[k] for k in LINKS])
    F = np.array(load, float)
    tot_f = F + sum(fi * (q - p) / np.linalg.norm(q - p) for fi, (p, q) in zip(f, links,
                                                                               strict=True))
    o = np.array([500.0, -300.0, 900.0])                 # an arbitrary moment origin
    tot_m = np.cross(CP - o, F) + sum(
        np.cross(p - o, fi * (q - p) / np.linalg.norm(q - p))
        for fi, (p, q) in zip(f, links, strict=True))
    assert np.abs(tot_f).max() < 1e-8 and np.abs(tot_m).max() < 1e-5


def test_braking_now_has_an_answer() -> None:
    """The two-force UCA closure left Fx's moments unbalanced; six links react them."""
    r = upright_link_forces(C, (-1728, 0, 1234), CP, PR_O, PR_I)
    assert r.residual_moment_Nmm < 1e-6
    # braking pulls the LBJ rearward: LCA front leg in tension, rear in compression
    assert r.lca_front_N > 0 > r.lca_rear_N


def test_right_corner_is_the_mirror_of_the_left() -> None:
    m = np.array([1.0, -1.0, 1.0])
    left = upright_link_forces(C, (-1000, -500, 1200), CP, PR_O, PR_I)
    right = upright_link_forces(_corner("FR", -1.0), (-1000, 500, 1200), CP * m, PR_O * m,
                                PR_I * m)
    for k in LINKS:
        assert right.as_dict()[k] == pytest.approx(left.as_dict()[k], abs=1e-9)


def test_a_mechanism_is_an_explicit_failure() -> None:
    # pushrod along the LCA front leg: two links on one line cannot hold the upright
    with pytest.raises(SingularLinkLayoutError, match="cannot hold the upright"):
        upright_link_forces(C, (0, 0, 1000), CP, LBJ, _p(C.lca_inboard_front))
    with pytest.raises(SingularLinkLayoutError, match="zero length"):
        upright_link_forces(C, (0, 0, 1000), CP, LBJ, LBJ)


def _rates(solver: DWSolver, h: float = 0.01):
    """Chassis-frame velocity (per mm of travel) of any upright-fixed point."""
    def pts(r: SolverResult) -> np.ndarray:
        return np.array([[q.x_mm, q.y_mm, q.z_mm] for q in (r.ubj, r.lbj, r.tro, r.wheel_center)])

    def fit(a: np.ndarray, b: np.ndarray):
        ca, cb = a.mean(0), b.mean(0)
        u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
        d = np.sign(np.linalg.det(vt.T @ u.T))
        rot = vt.T @ np.diag([1, 1, d]) @ u.T
        return rot, cb - rot @ ca

    s0, su, sd = solver.solve(), solver.solve(wheel_travel_mm=h), solver.solve(wheel_travel_mm=-h)
    a = pts(s0)
    (ru, tu), (rd, td) = fit(a, pts(su)), fit(a, pts(sd))
    lift = np.array([0.0, 0.0, h])           # DWSolver moves the chassis down by the travel
    return lambda x: ((ru @ x + tu + lift) - (rd @ x + td - lift)) / (2 * h)


@pytest.mark.parametrize(("load", "point"), [((0, -1851, 1234), CP), ((-1728, 0, 1234), CP),
                                             ((-1728, 0, 1234), np.array([0.0, 614.0, 245.0]))])
def test_pushrod_force_agrees_with_virtual_work_on_dwsolver(load, point) -> None:
    vel = _rates(DWSolver(C))
    u = (PR_I - PR_O) / np.linalg.norm(PR_I - PR_O)
    f_vw = -np.asarray(load, float) @ vel(point) / (u @ vel(PR_O))
    f = upright_link_forces(C, load, point, PR_O, PR_I).pushrod_N
    assert f == pytest.approx(f_vw, rel=1e-7)
