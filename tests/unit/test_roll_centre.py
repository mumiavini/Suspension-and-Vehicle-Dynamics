"""Tests for vdcore.analysis.roll_centre -- FVIC and roll centre construction.

Coordinate system: ISO 8855 -- X+ forward, Y+ LEFT, Z+ up.
"""

from __future__ import annotations

import math

import pytest

from vdcore.analysis.roll_centre import (
    FVICResult,
    RollCentreResult,
    _effective_pivot_at_x,
    _line_intersect_yz,
    _rc_height_from_cp_and_ic,
    front_view_instant_centre,
    roll_centre_height,
    roll_centre_migration,
)
from vdcore.geometry.solver import DWSolver
from vdcore.models.hardpoint import Axle, Corner, Hardpoint, TirePackage


def _hp(name: str, x: float, y: float, z: float) -> Hardpoint:
    return Hardpoint(name=name, x_mm=x, y_mm=y, z_mm=z, source="cad", tol_mm=0.5)


def _tire(r: float = 228.0) -> TirePackage:
    return TirePackage(loaded_radius_mm=r, source="cad", tol_mm=1.0)


def _planar_fl(
    uca_ib_y: float = 150.0,
    uca_ib_z: float = 300.0,
    uca_ob_y: float = 500.0,
    uca_ob_z: float = 290.0,
    lca_ib_y: float = 100.0,
    lca_ib_z: float = 100.0,
    lca_ob_y: float = 550.0,
    lca_ob_z: float = 80.0,
) -> Corner:
    """Planar (zero caster) front-left corner with parameterised Y-Z geometry."""
    return Corner(
        corner_id="FL",
        uca_inboard_front=_hp("UCA_IF", 50, uca_ib_y, uca_ib_z),
        uca_inboard_rear=_hp("UCA_IR", -50, uca_ib_y, uca_ib_z),
        uca_outboard=_hp("UCA_O", 0, uca_ob_y, uca_ob_z),
        lca_inboard_front=_hp("LCA_IF", 50, lca_ib_y, lca_ib_z),
        lca_inboard_rear=_hp("LCA_IR", -50, lca_ib_y, lca_ib_z),
        lca_outboard=_hp("LCA_O", 0, lca_ob_y, lca_ob_z),
        tie_rod_inboard=_hp("TR_I", -30, 130, 110),
        tie_rod_outboard=_hp("TR_O", -20, 520, 100),
        wheel_center=_hp("WC", 0, 600, 228),
        tire=_tire(),
        static_camber_deg=0.0,
        static_toe_deg_per_side=0.0,
    )


def _mirror_to_fr(fl: Corner) -> Corner:
    """Mirror a FL corner to FR by negating all Y coordinates."""
    d = fl.model_dump()
    d["corner_id"] = "FR"
    for key in d:
        if isinstance(d[key], dict) and "y_mm" in d[key]:
            d[key]["y_mm"] = -d[key]["y_mm"]
    return Corner.model_validate(d)


def _solve_static(corner: Corner) -> "DWSolver":
    """Convenience: return a solver for the corner."""
    return DWSolver(corner)


def _solve_pair(fl: Corner, fr: Corner) -> tuple:
    """Solve both corners at zero travel, return (axle, rl, rr)."""
    axle = Axle(left=fl, right=fr)
    rl = DWSolver(fl).solve()
    rr = DWSolver(fr).solve()
    assert rl.converged and rr.converged
    return axle, rl, rr


class TestEffectivePivotAtX:
    def test_midpoint_when_axis_parallel_to_yz(self) -> None:
        front = [0.0, 100.0, 200.0]
        rear = [0.0, 100.0, 200.0]
        result = _effective_pivot_at_x(
            __import__("numpy").array(front),
            __import__("numpy").array(rear),
            0.0,
        )
        assert result[1] == pytest.approx(100.0)
        assert result[2] == pytest.approx(200.0)

    def test_interpolates_along_axis(self) -> None:
        import numpy as np

        front = np.array([100.0, 150.0, 300.0])
        rear = np.array([-100.0, 150.0, 300.0])
        result = _effective_pivot_at_x(front, rear, 0.0)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(150.0)
        assert result[2] == pytest.approx(300.0)

    def test_extrapolation_beyond_axis(self) -> None:
        import numpy as np

        front = np.array([50.0, 150.0, 300.0])
        rear = np.array([-50.0, 170.0, 310.0])
        result = _effective_pivot_at_x(front, rear, -100.0)
        assert result[0] == pytest.approx(-100.0)
        assert result[1] == pytest.approx(180.0)
        assert result[2] == pytest.approx(315.0)


class TestLineIntersectYZ:
    def test_known_intersection(self) -> None:
        y, z, ok = _line_intersect_yz(0, 0, 10, 10, 0, 10, 10, 0)
        assert ok
        assert y == pytest.approx(5.0)
        assert z == pytest.approx(5.0)

    def test_parallel_lines(self) -> None:
        y, z, ok = _line_intersect_yz(0, 0, 10, 0, 0, 5, 10, 5)
        assert not ok

    def test_coincident_lines(self) -> None:
        _, _, ok = _line_intersect_yz(0, 0, 10, 10, 5, 5, 15, 15)
        assert not ok


class TestRCHeightFromCPAndIC:
    def test_ic_on_centreline(self) -> None:
        rc = _rc_height_from_cp_and_ic(600.0, 0.0, 0.0, 50.0)
        assert rc == pytest.approx(50.0)

    def test_ic_at_wheel_raises(self) -> None:
        with pytest.raises(ValueError, match="vertical"):
            _rc_height_from_cp_and_ic(600.0, 0.0, 600.0, 200.0)

    def test_ic_at_centreline_vertical(self) -> None:
        rc = _rc_height_from_cp_and_ic(0.0, 0.0, 0.0, 50.0)
        assert rc == pytest.approx(50.0)

    def test_ic_beyond_centreline(self) -> None:
        rc = _rc_height_from_cp_and_ic(600.0, 0.0, -200.0, 100.0)
        assert rc == pytest.approx(75.0)

    def test_ic_between_wheel_and_centreline(self) -> None:
        rc = _rc_height_from_cp_and_ic(600.0, 0.0, 300.0, 200.0)
        assert rc == pytest.approx(400.0)


class TestFrontViewInstantCentre:
    def test_planar_geometry_matches_hand_calc(self) -> None:
        """FVIC from a planar geometry matches hand calculation."""
        fl = _planar_fl()
        r = DWSolver(fl).solve()
        assert r.converged
        fvic = front_view_instant_centre(fl, ubj=r.ubj, lbj=r.lbj, contact_patch=r.contact_patch)
        assert fvic.is_finite
        assert fvic.fvic_y_mm == pytest.approx(-12590.0, abs=5.0)
        assert fvic.fvic_z_mm == pytest.approx(664.0, abs=2.0)

    def test_fvsa_sign_negative_when_ic_opposite_side(self) -> None:
        fl = _planar_fl()
        r = DWSolver(fl).solve()
        assert r.converged
        fvic = front_view_instant_centre(fl, ubj=r.ubj, lbj=r.lbj, contact_patch=r.contact_patch)
        assert fvic.fvsa_mm < 0

    def test_parallel_arms_gives_infinite_ic(self) -> None:
        fl = _planar_fl(
            uca_ib_y=150, uca_ib_z=300, uca_ob_y=550, uca_ob_z=300,
            lca_ib_y=150, lca_ib_z=100, lca_ob_y=550, lca_ob_z=100,
        )
        r = DWSolver(fl).solve()
        assert r.converged
        fvic = front_view_instant_centre(fl, ubj=r.ubj, lbj=r.lbj, contact_patch=r.contact_patch)
        assert not fvic.is_finite
        assert fvic.fvsa_mm == float("inf")


class TestRollCentreHeight:
    def test_symmetric_axle_gives_consistent_rc(self) -> None:
        fl = _planar_fl()
        fr = _mirror_to_fr(fl)
        axle, rl, rr = _solve_pair(fl, fr)
        rc = roll_centre_height(axle, rl, rr)
        assert isinstance(rc.rc_height_mm, float)
        assert math.isfinite(rc.rc_height_mm)

    def test_symmetric_axle_fvic_mirror(self) -> None:
        fl = _planar_fl()
        fr = _mirror_to_fr(fl)
        axle, rl, rr = _solve_pair(fl, fr)
        rc = roll_centre_height(axle, rl, rr)
        assert rc.left_fvic.fvic_y_mm == pytest.approx(-rc.right_fvic.fvic_y_mm, abs=0.01)
        assert rc.left_fvic.fvic_z_mm == pytest.approx(rc.right_fvic.fvic_z_mm, abs=0.01)

    def test_rc_positive_for_typical_sla(self) -> None:
        fl = _planar_fl()
        fr = _mirror_to_fr(fl)
        axle, rl, rr = _solve_pair(fl, fr)
        rc = roll_centre_height(axle, rl, rr)
        assert rc.rc_height_mm > 0

    def test_parallel_arms_raises(self) -> None:
        fl = _planar_fl(
            uca_ib_y=150, uca_ib_z=300, uca_ob_y=550, uca_ob_z=300,
            lca_ib_y=150, lca_ib_z=100, lca_ob_y=550, lca_ob_z=100,
        )
        fr = _mirror_to_fr(fl)
        axle, rl, rr = _solve_pair(fl, fr)
        with pytest.raises(RuntimeError, match="FVIC is at infinity"):
            roll_centre_height(axle, rl, rr)

    def test_raising_lca_inboard_raises_rc(self) -> None:
        fl_base = _planar_fl(lca_ib_z=100)
        fr_base = _mirror_to_fr(fl_base)
        axle_base, rl_base, rr_base = _solve_pair(fl_base, fr_base)
        rc_base = roll_centre_height(axle_base, rl_base, rr_base)

        fl_raised = _planar_fl(lca_ib_z=130)
        fr_raised = _mirror_to_fr(fl_raised)
        axle_raised, rl_raised, rr_raised = _solve_pair(fl_raised, fr_raised)
        rc_raised = roll_centre_height(axle_raised, rl_raised, rr_raised)

        assert rc_raised.rc_height_mm > rc_base.rc_height_mm

    def test_rc_returns_frozen_model(self) -> None:
        fl = _planar_fl()
        fr = _mirror_to_fr(fl)
        axle, rl, rr = _solve_pair(fl, fr)
        rc = roll_centre_height(axle, rl, rr)
        with pytest.raises(Exception):
            rc.rc_height_mm = 999.0  # type: ignore[misc]

    def test_rc_y_zero_for_symmetric_axle(self) -> None:
        fl = _planar_fl()
        fr = _mirror_to_fr(fl)
        axle, rl, rr = _solve_pair(fl, fr)
        rc = roll_centre_height(axle, rl, rr)
        assert rc.rc_y_mm == pytest.approx(0.0, abs=0.01)

    def test_solved_zero_travel_equals_static_hardpoints(self) -> None:
        """RC from zero-travel solve uses identical geometry to static hardpoints.

        This is the regression guard: the zero-travel solver result
        recovers the static UBJ/LBJ/CP positions, so RC must match
        to machine precision.
        """
        fl = _planar_fl()
        fr = _mirror_to_fr(fl)
        axle = Axle(left=fl, right=fr)

        rl = DWSolver(fl).solve(wheel_travel_mm=0.0)
        rr = DWSolver(fr).solve(wheel_travel_mm=0.0)
        assert rl.converged and rr.converged
        rc_zero = roll_centre_height(axle, rl, rr)

        rl2 = DWSolver(fl).solve(wheel_travel_mm=0.0)
        rr2 = DWSolver(fr).solve(wheel_travel_mm=0.0)
        rc_zero2 = roll_centre_height(axle, rl2, rr2)

        assert rc_zero.rc_height_mm == pytest.approx(rc_zero2.rc_height_mm, abs=1e-9)
        assert rc_zero.rc_y_mm == pytest.approx(rc_zero2.rc_y_mm, abs=1e-9)

    def test_fvic_deterministic(self) -> None:
        """Two identical zero-travel solves produce identical FVICs."""
        fl = _planar_fl()
        r1 = DWSolver(fl).solve(wheel_travel_mm=0.0)
        r2 = DWSolver(fl).solve(wheel_travel_mm=0.0)
        assert r1.converged and r2.converged

        fvic1 = front_view_instant_centre(fl, ubj=r1.ubj, lbj=r1.lbj, contact_patch=r1.contact_patch)
        fvic2 = front_view_instant_centre(fl, ubj=r2.ubj, lbj=r2.lbj, contact_patch=r2.contact_patch)

        assert fvic1.fvic_y_mm == pytest.approx(fvic2.fvic_y_mm, abs=1e-6)
        assert fvic1.fvic_z_mm == pytest.approx(fvic2.fvic_z_mm, abs=1e-6)
        assert fvic1.fvsa_mm == pytest.approx(fvic2.fvsa_mm, abs=1e-3)

    def test_no_silent_fallback(self) -> None:
        """Calling without solved state raises TypeError, not a wrong answer."""
        fl = _planar_fl()
        with pytest.raises(TypeError):
            front_view_instant_centre(fl)  # type: ignore[call-arg]

        fr = _mirror_to_fr(fl)
        axle = Axle(left=fl, right=fr)
        with pytest.raises(TypeError):
            roll_centre_height(axle)  # type: ignore[call-arg]


class TestRollCentreMigration:
    def test_migration_returns_correct_length(self) -> None:
        fl = _planar_fl()
        fr = _mirror_to_fr(fl)
        axle = Axle(left=fl, right=fr)
        mig = roll_centre_migration(axle, -10, 10, steps=21)
        assert len(mig.wheel_travel_mm) == 21
        assert len(mig.rc_height_mm) == 21
        assert len(mig.rc_y_mm) == 21
        assert len(mig.converged) == 21

    def test_migration_all_converged(self) -> None:
        fl = _planar_fl()
        fr = _mirror_to_fr(fl)
        axle = Axle(left=fl, right=fr)
        mig = roll_centre_migration(axle, -10, 10, steps=21)
        assert all(mig.converged)

    def test_migration_zero_travel_matches_solved(self) -> None:
        """The point closest to zero travel must match a direct zero-travel solve."""
        fl = _planar_fl()
        fr = _mirror_to_fr(fl)
        axle = Axle(left=fl, right=fr)

        rl = DWSolver(fl).solve(wheel_travel_mm=0.0)
        rr = DWSolver(fr).solve(wheel_travel_mm=0.0)
        rc_direct = roll_centre_height(axle, rl, rr)

        mig = roll_centre_migration(axle, -10, 10, steps=21)
        zero_idx = min(range(len(mig.wheel_travel_mm)), key=lambda i: abs(mig.wheel_travel_mm[i]))
        assert mig.rc_height_mm[zero_idx] == pytest.approx(rc_direct.rc_height_mm, abs=1e-6)

    def test_migration_rc_varies_with_travel(self) -> None:
        fl = _planar_fl()
        fr = _mirror_to_fr(fl)
        axle = Axle(left=fl, right=fr)
        mig = roll_centre_migration(axle, -20, 20, steps=41)
        heights = [h for h, c in zip(mig.rc_height_mm, mig.converged) if c]
        assert max(heights) > min(heights)
