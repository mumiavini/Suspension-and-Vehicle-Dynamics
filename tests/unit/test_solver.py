"""Tests for vdcore.geometry.solver — DWSolver kinematic solver."""

from __future__ import annotations

import numpy as np
import pytest

from vdcore.geometry.solver import DWSolver
from vdcore.models.hardpoint import Corner, Hardpoint, TirePackage


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


class TestAsymmetricRack:
    """Apply opposite rack travel to a mirrored pair: left and right toe
    must change in opposite senses with equal magnitude.

    A real steering rack pushes one tie-rod inboard in +Y while pulling
    the other in -Y. We simulate this by giving +rack to one side and
    -rack to the other.
    """

    def test_rack_produces_opposite_toe(self) -> None:
        fl = _corner("FL", 1.0)
        fr = _corner("FR", -1.0)
        solver_l = DWSolver(fl)
        solver_r = DWSolver(fr)

        rack_mm = 3.0
        rl = solver_l.solve(rack_mm=rack_mm)
        rr = solver_r.solve(rack_mm=-rack_mm)

        assert rl.converged and rr.converged

        rl_static = solver_l.solve()
        rr_static = solver_r.solve()
        delta_toe_l = rl.toe_deg_per_side - rl_static.toe_deg_per_side
        delta_toe_r = rr.toe_deg_per_side - rr_static.toe_deg_per_side

        assert abs(delta_toe_l) > 0.01, "Rack produced no toe change on left"
        assert abs(delta_toe_r) > 0.01, "Rack produced no toe change on right"
        assert delta_toe_l == pytest.approx(-delta_toe_r, abs=0.05), (
            f"Left/right toe change not antisymmetric: "
            f"left={delta_toe_l:.4f}, right={delta_toe_r:.4f}"
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
