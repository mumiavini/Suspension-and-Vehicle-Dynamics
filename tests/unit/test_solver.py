"""Tests for vdcore.geometry.solver — DWSolver kinematic solver."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vdcore.geometry.solver import DWSolver, SolverResult
from vdcore.models.hardpoint import Corner, Hardpoint, TirePackage


def _upright_yaw_deg(corner: Corner, result: SolverResult) -> float:
    """Yaw of the upright about Z, in degrees, independent of any toe convention.

    Recovers the upright's rigid rotation from the three ball joints (Kabsch)
    and reads the heading of the body-fixed +X axis. This is the ground truth
    the toe sign convention must agree with: for a LEFT wheel, toe-in means the
    front of the wheel turns toward the centreline (-Y), i.e. negative yaw; for
    a RIGHT wheel toe-in is positive yaw.
    """
    static = np.array([
        [corner.uca_outboard.x_mm, corner.uca_outboard.y_mm, corner.uca_outboard.z_mm],
        [corner.lca_outboard.x_mm, corner.lca_outboard.y_mm, corner.lca_outboard.z_mm],
        [corner.tie_rod_outboard.x_mm,
         corner.tie_rod_outboard.y_mm,
         corner.tie_rod_outboard.z_mm],
    ])
    moved = np.array([
        [result.ubj.x_mm, result.ubj.y_mm, result.ubj.z_mm],
        [result.lbj.x_mm, result.lbj.y_mm, result.lbj.z_mm],
        [result.tro.x_mm, result.tro.y_mm, result.tro.z_mm],
    ])
    p = static - static.mean(axis=0)
    q = moved - moved.mean(axis=0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = float(np.sign(np.linalg.det(vt.T @ u.T)))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    forward = rot @ np.array([1.0, 0.0, 0.0])
    return math.degrees(math.atan2(forward[1], forward[0]))


def _hp(name: str, x: float, y: float, z: float) -> Hardpoint:
    return Hardpoint(name=name, x_mm=x, y_mm=y, z_mm=z, source="cad", tol_mm=0.5)


def _tire(r: float = 228.0) -> TirePackage:
    return TirePackage(loaded_radius_mm=r, source="cad", tol_mm=1.0)


def _corner(
    cid: str,
    y_sign: float,
    x_offset: float = 0.0,
    static_camber_deg: float = -2.0,
    static_toe_deg_per_side: float = 0.0,
) -> Corner:
    s = y_sign
    return Corner(
        corner_id=cid,
        uca_inboard_front=_hp("UCA_IF", 80 + x_offset, s * 150, 280),
        uca_inboard_rear=_hp("UCA_IR", -80 + x_offset, s * 150, 280),
        uca_outboard=_hp("UCA_O", 0 + x_offset, s * 530, 290),
        lca_inboard_front=_hp("LCA_IF", 100 + x_offset, s * 130, 80),
        lca_inboard_rear=_hp("LCA_IR", -100 + x_offset, s * 130, 80),
        lca_outboard=_hp("LCA_O", 0 + x_offset, s * 580, 75),
        tie_rod_inboard=_hp("TR_I", -60 + x_offset, s * 160, 120),
        tie_rod_outboard=_hp("TR_O", -50 + x_offset, s * 540, 110),
        wheel_center=_hp("WC", 0 + x_offset, s * 600, 200),
        tire=_tire(),
        static_camber_deg=static_camber_deg,
        static_toe_deg_per_side=static_toe_deg_per_side,
    )


class TestStaticSolve:
    def test_static_converges(self) -> None:
        """Zero wheel_travel/roll/rack must converge."""
        solver = DWSolver(_corner("FL", 1.0))
        result = solver.solve()
        assert result.converged

    def test_static_residual_near_zero(self) -> None:
        """Static solve residual must be negligible."""
        solver = DWSolver(_corner("FL", 1.0))
        result = solver.solve()
        assert result.residual_norm < 1e-8

    def test_static_recovers_wheel_center(self) -> None:
        """At zero state, wheel centre must match the input."""
        solver = DWSolver(_corner("FL", 1.0))
        result = solver.solve()
        assert result.wheel_center.x_mm == pytest.approx(0.0, abs=0.01)
        assert result.wheel_center.y_mm == pytest.approx(600.0, abs=0.01)
        assert result.wheel_center.z_mm == pytest.approx(200.0, abs=0.01)

    def test_static_recovers_joint_positions(self) -> None:
        """At zero state, joint positions must match inputs."""
        solver = DWSolver(_corner("FL", 1.0))
        result = solver.solve()
        assert result.ubj.y_mm == pytest.approx(530.0, abs=0.01)
        assert result.lbj.y_mm == pytest.approx(580.0, abs=0.01)


class TestBumpSolve:
    def test_bump_changes_camber(self) -> None:
        """Non-zero wheel travel should change camber from static."""
        solver = DWSolver(_corner("FL", 1.0))
        r_static = solver.solve()
        r_bump = solver.solve(wheel_travel_mm=10.0)
        assert r_bump.converged
        assert r_bump.camber_deg != pytest.approx(r_static.camber_deg, abs=0.01)

    def test_bump_converges_full_range(self) -> None:
        """Solver should converge across ±25mm wheel travel range."""
        solver = DWSolver(_corner("FL", 1.0))
        for h in [-25, -15, -5, 0, 5, 15, 25]:
            result = solver.solve(wheel_travel_mm=float(h))
            assert result.converged, f"Did not converge at wheel_travel={h}mm"

    def test_bump_and_droop_differ(self) -> None:
        """Bump and droop should give different camber (non-linear)."""
        solver = DWSolver(_corner("FL", 1.0))
        r_bump = solver.solve(wheel_travel_mm=10.0)
        r_droop = solver.solve(wheel_travel_mm=-10.0)
        assert r_bump.converged and r_droop.converged
        assert r_bump.camber_deg != pytest.approx(r_droop.camber_deg, abs=0.01)


class TestRightCorner:
    def test_right_corner_converges(self) -> None:
        solver = DWSolver(_corner("FR", -1.0))
        result = solver.solve()
        assert result.converged

    def test_symmetric_camber(self) -> None:
        """Symmetric left/right geometry must give equal camber (same sign)."""
        solver_l = DWSolver(_corner("FL", 1.0))
        solver_r = DWSolver(_corner("FR", -1.0))
        rl = solver_l.solve()
        rr = solver_r.solve()
        assert rl.camber_deg == pytest.approx(rr.camber_deg, abs=0.01)

    def test_symmetric_kpi(self) -> None:
        """Symmetric left/right geometry must give equal KPI."""
        solver_l = DWSolver(_corner("FL", 1.0))
        solver_r = DWSolver(_corner("FR", -1.0))
        rl = solver_l.solve()
        rr = solver_r.solve()
        assert rl.kpi_deg == pytest.approx(rr.kpi_deg, abs=0.01)

    def test_symmetric_static_toe(self) -> None:
        """Mirrored geometry with the same design-intent toe must report the
        same toe on both sides — same sign, not opposite.

        Toe-in is defined per side relative to the centreline, so a symmetric
        car set to 0.15 deg of toe-in reads +0.15 on BOTH corners.
        """
        toe = 0.15
        rl = DWSolver(_corner("FL", 1.0, static_toe_deg_per_side=toe)).solve()
        rr = DWSolver(_corner("FR", -1.0, static_toe_deg_per_side=toe)).solve()
        assert rl.toe_deg_per_side == pytest.approx(toe, abs=1e-4)
        assert rr.toe_deg_per_side == pytest.approx(toe, abs=1e-4)

    def test_symmetric_bump_steer(self) -> None:
        """Mirrored geometry must toe the SAME way in bump on both sides.

        This is the regression guard for the left-corner toe sign inversion:
        the two branches of _extract_angles must agree in sign for mirrored
        input, or every left-corner toe change comes out backwards.
        """
        sl = DWSolver(_corner("FL", 1.0))
        sr = DWSolver(_corner("FR", -1.0))
        static_l = sl.solve().toe_deg_per_side
        static_r = sr.solve().toe_deg_per_side

        for travel in (-20.0, -10.0, 10.0, 20.0):
            rl = sl.solve(wheel_travel_mm=travel)
            rr = sr.solve(wheel_travel_mm=travel)
            assert rl.converged and rr.converged
            d_l = rl.toe_deg_per_side - static_l
            d_r = rr.toe_deg_per_side - static_r
            assert abs(d_l) > 1e-3, f"no bump steer to test at {travel}mm"
            assert d_l == pytest.approx(d_r, abs=1e-3), (
                f"Left/right toe change must match for mirrored geometry at "
                f"{travel}mm: left={d_l:.5f}, right={d_r:.5f}"
            )


class TestRackSteer:
    """Rack travel and toe.

    The rack is ONE RIGID BAR: steering it moves both tie-rod inboard points
    in the same direction along Y. That is precisely why both front wheels
    steer the same way, which per side reads as toe-in on one wheel and
    toe-out on the other.

    Feeding +rack to one corner and -rack to the other is therefore NOT a
    steering input — it is a mirror-image input, and it must produce a
    mirror-image (same-sign) toe change.
    """

    def test_real_rack_steers_both_wheels_the_same_way(self) -> None:
        """One rigid rack, both inboard points moving +Y: the wheels steer
        together, so the per-side toe changes have OPPOSITE signs."""
        solver_l = DWSolver(_corner("FL", 1.0))
        solver_r = DWSolver(_corner("FR", -1.0))

        rack_mm = 3.0
        static_l = solver_l.solve().toe_deg_per_side
        static_r = solver_r.solve().toe_deg_per_side
        rl = solver_l.solve(rack_mm=rack_mm)
        rr = solver_r.solve(rack_mm=rack_mm)

        assert rl.converged and rr.converged
        delta_l = rl.toe_deg_per_side - static_l
        delta_r = rr.toe_deg_per_side - static_r

        assert abs(delta_l) > 0.01, "Rack produced no toe change on left"
        assert abs(delta_r) > 0.01, "Rack produced no toe change on right"
        assert delta_l * delta_r < 0.0, (
            f"A real rack must steer the wheels together, giving opposite "
            f"per-side toe: left={delta_l:.4f}, right={delta_r:.4f}"
        )
        # Magnitudes differ by the Ackermann effect — same order, not equal.
        assert abs(delta_l) == pytest.approx(abs(delta_r), rel=0.25)

    def test_mirrored_rack_gives_mirrored_toe(self) -> None:
        """+rack on the left and -rack on the right is a mirror-image input,
        so a mirrored car must give the SAME toe change on both sides.

        Regression guard for the left-corner toe sign inversion: under the old
        convention these came out equal and opposite, which looked plausible.
        """
        solver_l = DWSolver(_corner("FL", 1.0))
        solver_r = DWSolver(_corner("FR", -1.0))

        rack_mm = 3.0
        static_l = solver_l.solve().toe_deg_per_side
        static_r = solver_r.solve().toe_deg_per_side
        delta_l = solver_l.solve(rack_mm=rack_mm).toe_deg_per_side - static_l
        delta_r = solver_r.solve(rack_mm=-rack_mm).toe_deg_per_side - static_r

        assert abs(delta_l) > 0.01
        assert delta_l == pytest.approx(delta_r, abs=1e-3), (
            f"Mirrored input on mirrored geometry must give mirrored toe: "
            f"left={delta_l:.4f}, right={delta_r:.4f}"
        )


class TestToeSignConvention:
    """Anchor the toe sign to geometry rather than to the solver's own model.

    _upright_yaw_deg recovers the upright's rigid rotation directly, so these
    tests fail if the spin-axis construction and the toe extraction are ever
    changed together in a way that stays self-consistent but is physically
    inverted — which is exactly how the left-corner bug survived.
    """

    def test_left_corner_toe_in_is_negative_yaw(self) -> None:
        corner = _corner("FL", 1.0)
        solver = DWSolver(corner)
        static = solver.solve()
        bumped = solver.solve(wheel_travel_mm=20.0)
        assert bumped.converged

        d_toe = bumped.toe_deg_per_side - static.toe_deg_per_side
        d_yaw = _upright_yaw_deg(corner, bumped) - _upright_yaw_deg(corner, static)

        assert abs(d_toe) > 1e-3, "no toe change to test"
        assert d_toe == pytest.approx(-d_yaw, rel=0.05), (
            f"Left corner: toe-in must correspond to negative upright yaw. "
            f"d_toe={d_toe:.5f}, d_yaw={d_yaw:.5f}"
        )

    def test_right_corner_toe_in_is_positive_yaw(self) -> None:
        corner = _corner("FR", -1.0)
        solver = DWSolver(corner)
        static = solver.solve()
        bumped = solver.solve(wheel_travel_mm=20.0)
        assert bumped.converged

        d_toe = bumped.toe_deg_per_side - static.toe_deg_per_side
        d_yaw = _upright_yaw_deg(corner, bumped) - _upright_yaw_deg(corner, static)

        assert abs(d_toe) > 1e-3, "no toe change to test"
        assert d_toe == pytest.approx(d_yaw, rel=0.05), (
            f"Right corner: toe-in must correspond to positive upright yaw. "
            f"d_toe={d_toe:.5f}, d_yaw={d_yaw:.5f}"
        )


class TestPerturbedRecovery:
    """Solve at small nonzero wheel travel and confirm the solver actually
    iterates (nfev > 1) and converges to a position different from static.
    This replaces the trivial zero-recovery test where x0 == solution."""

    def test_small_bump_exercises_solver(self) -> None:
        corner = _corner("FL", 1.0)
        solver = DWSolver(corner)

        result = solver.solve(wheel_travel_mm=5.0)

        assert result.converged, (
            f"Solver did not converge at 5mm bump. "
            f"Residual: {result.residual_norm:.2e}"
        )
        assert result.nfev > 1, (
            "nfev=1 means the solver never iterated — the initial guess was the solution"
        )
        assert result.residual_norm < 1e-8

        # Camber must differ from static — the solver actually did work
        r_static = solver.solve(wheel_travel_mm=0.0)
        assert result.camber_deg != pytest.approx(
            r_static.camber_deg, abs=0.001
        ), "Camber unchanged at 5mm bump — solver didn't produce any kinematic change"

    def test_solver_recovers_static_from_nearby_state(self) -> None:
        """Solve at 5mm bump, then at 0mm. The solver uses the same x0
        (static geometry) both times, so the 0mm solve has x0 == solution
        and is trivial. But if we could feed the 5mm result as x0 for the
        0mm solve, we'd exercise recovery. Instead, we verify round-trip:
        solve forward and back, check we return to static."""
        corner = _corner("FL", 1.0)
        solver = DWSolver(corner)

        r_static = solver.solve(wheel_travel_mm=0.0)
        r_bump = solver.solve(wheel_travel_mm=5.0)
        r_return = solver.solve(wheel_travel_mm=0.0)

        assert all(r.converged for r in [r_static, r_bump, r_return])
        assert r_return.wheel_center.y_mm == pytest.approx(
            r_static.wheel_center.y_mm, abs=0.01
        )
        assert r_return.wheel_center.z_mm == pytest.approx(
            r_static.wheel_center.z_mm, abs=0.01
        )
        assert r_return.camber_deg == pytest.approx(r_static.camber_deg, abs=0.001)


class TestConvergenceReporting:
    def test_result_reports_nfev(self) -> None:
        solver = DWSolver(_corner("FL", 1.0))
        result = solver.solve()
        assert result.nfev >= 1

    def test_result_reports_njev(self) -> None:
        solver = DWSolver(_corner("FL", 1.0))
        result = solver.solve()
        assert result.njev >= 0

    def test_result_has_contact_patch(self) -> None:
        solver = DWSolver(_corner("FL", 1.0))
        result = solver.solve()
        assert result.contact_patch.z_mm < result.wheel_center.z_mm
