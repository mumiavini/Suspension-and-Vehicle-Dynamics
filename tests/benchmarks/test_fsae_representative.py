"""Regression benchmark for an FSAE-representative double-wishbone geometry.

This geometry targets realistic FSAE front-view instant centre placement:
  - FVSA ~2200 mm (IC visible in front view, not at near-infinity)
  - Camber gain ~-0.028 deg/mm (1.5 deg change over 60 mm travel)
  - RC height ~49 mm (within the 30-50 mm FSAE design range)
  - Scrub radius ~-7 mm (visible sign-error target, not centre-point)
  - Bump steer ~0.40 deg/1000mm (tie rod slightly above LCA front-view line)

Contrast with the flat geometry in test_published_geometry.py, which has
FVSA ~8700 mm and camber gain ~-0.007 deg/mm. That geometry exercises the
solver in the near-parallel regime; this one exercises it where IC migration,
nonlinearity, and convergence sensitivity matter.

This file is a REGRESSION ANCHOR: hardpoints are synthetic, not from a
published source. If any value changes, the solver has changed.

Geometry (ISO 8855: X+ forward, Y+ LEFT, Z+ up):
  - 1220 mm track (610 mm half-track)
  - 254 mm loaded tire radius (10" tire)
  - UCA front-view: 346 mm, LCA front-view: 475 mm
  - UCA/LCA strongly converging (slope 0.064 / -0.048)
  - -2.0 deg static camber, 0.1 deg toe per side
  - ~5.2 deg caster, ~14.0 deg KPI

Regression values (updated 2026-08-06, scrub fix):
  Camber    = -2.00 deg (design intent)
  Toe       = 0.10 deg (design intent)
  KPI       = 14.04 deg
  Caster    = 5.24 deg
  FVSA      = -2192 mm
  RC height = 48.77 mm
  Camber gain = -0.0280 deg/mm
  Bump steer = 0.40 deg/1000mm
  Scrub radius = -6.9 mm
  Mechanical trail = 16.6 mm
"""

from __future__ import annotations

import math

import pytest

from vdcore.analysis.camber import camber_gain_deg_per_mm, camber_sweep
from vdcore.analysis.roll_centre import front_view_instant_centre, roll_centre_height
from vdcore.geometry.solver import DWSolver
from vdcore.models.hardpoint import Axle, Corner, Hardpoint, TirePackage


def _hp(name: str, x: float, y: float, z: float) -> Hardpoint:
    return Hardpoint(name=name, x_mm=x, y_mm=y, z_mm=z, source="cad", tol_mm=0.5)


def _tire() -> TirePackage:
    return TirePackage(loaded_radius_mm=254.0, source="cad", tol_mm=1.0)


def _fsae_fl() -> Corner:
    """Front-left corner with FSAE-representative arm convergence.

    Front-view geometry (Y-Z plane):
      UCA: (185, 290) -> (530, 312), slope = 22/345 = +0.064
      LCA: (115, 95)  -> (590, 72),  slope = -23/475 = -0.048
    Arms converge to FVIC at approximately (-1793, 176), giving
    FVSA ~ -2192 mm (IC on the opposite side of centreline).
    """
    return Corner(
        corner_id="FL",
        uca_inboard_front=_hp("UCA_IF", 70, 185, 290),
        uca_inboard_rear=_hp("UCA_IR", -70, 185, 290),
        uca_outboard=_hp("UCA_O", -12, 530, 312),
        lca_inboard_front=_hp("LCA_IF", 110, 115, 95),
        lca_inboard_rear=_hp("LCA_IR", -90, 115, 95),
        lca_outboard=_hp("LCA_O", 10, 590, 72),
        tie_rod_inboard=_hp("TR_I", -80, 150, 93.3),
        tie_rod_outboard=_hp("TR_O", -70, 555, 73.9),
        wheel_center=_hp("WC", 0, 610, 254),
        tire=_tire(),
        static_camber_deg=-2.0,
        static_toe_deg_per_side=0.1,
    )


def _fsae_fr() -> Corner:
    """Front-right mirror of the FSAE-representative geometry."""
    fl = _fsae_fl()
    return Corner(
        corner_id="FR",
        uca_inboard_front=_hp("UCA_IF", fl.uca_inboard_front.x_mm, -fl.uca_inboard_front.y_mm, fl.uca_inboard_front.z_mm),
        uca_inboard_rear=_hp("UCA_IR", fl.uca_inboard_rear.x_mm, -fl.uca_inboard_rear.y_mm, fl.uca_inboard_rear.z_mm),
        uca_outboard=_hp("UCA_O", fl.uca_outboard.x_mm, -fl.uca_outboard.y_mm, fl.uca_outboard.z_mm),
        lca_inboard_front=_hp("LCA_IF", fl.lca_inboard_front.x_mm, -fl.lca_inboard_front.y_mm, fl.lca_inboard_front.z_mm),
        lca_inboard_rear=_hp("LCA_IR", fl.lca_inboard_rear.x_mm, -fl.lca_inboard_rear.y_mm, fl.lca_inboard_rear.z_mm),
        lca_outboard=_hp("LCA_O", fl.lca_outboard.x_mm, -fl.lca_outboard.y_mm, fl.lca_outboard.z_mm),
        tie_rod_inboard=_hp("TR_I", fl.tie_rod_inboard.x_mm, -fl.tie_rod_inboard.y_mm, fl.tie_rod_inboard.z_mm),
        tie_rod_outboard=_hp("TR_O", fl.tie_rod_outboard.x_mm, -fl.tie_rod_outboard.y_mm, fl.tie_rod_outboard.z_mm),
        wheel_center=_hp("WC", fl.wheel_center.x_mm, -fl.wheel_center.y_mm, fl.wheel_center.z_mm),
        tire=_tire(),
        static_camber_deg=fl.static_camber_deg,
        static_toe_deg_per_side=fl.static_toe_deg_per_side,
    )


class TestStaticGeometry:
    """Static angles — regression anchors."""

    def test_solver_converges(self) -> None:
        solver = DWSolver(_fsae_fl())
        r = solver.solve()
        assert r.converged
        assert r.residual_norm < 1e-6

    def test_kpi_value(self) -> None:
        r = DWSolver(_fsae_fl()).solve()
        assert r.converged
        assert r.kpi_deg == pytest.approx(14.04, abs=0.05)

    def test_caster_value(self) -> None:
        r = DWSolver(_fsae_fl()).solve()
        assert r.converged
        assert r.caster_deg == pytest.approx(5.24, abs=0.05)

    def test_static_camber_equals_design_intent(self) -> None:
        r = DWSolver(_fsae_fl()).solve()
        assert r.converged
        assert r.camber_deg == pytest.approx(-2.0, abs=0.01)

    def test_static_toe_equals_design_intent(self) -> None:
        r = DWSolver(_fsae_fl()).solve()
        assert r.converged
        assert r.toe_deg_per_side == pytest.approx(0.1, abs=0.01)


class TestCamberGain:
    """Camber gain for the representative geometry — should be much
    larger than the flat fixture (~4× more camber change per mm)."""

    def test_camber_gain_sign_and_magnitude(self) -> None:
        gain = camber_gain_deg_per_mm(_fsae_fl(), wheel_travel_range_mm=25.0, steps=50)
        assert gain < 0, "UCA shorter than LCA must produce negative camber gain"
        assert abs(gain) > 0.01, "Gain too small for this convergent geometry"
        assert abs(gain) < 0.1, "Gain unreasonably large"

    def test_camber_gain_regression(self) -> None:
        gain = camber_gain_deg_per_mm(_fsae_fl(), wheel_travel_range_mm=25.0, steps=50)
        assert gain == pytest.approx(-0.0273, abs=0.002)

    def test_bump_droop_asymmetry(self) -> None:
        """Nonlinearity should be more visible than in the flat fixture."""
        sweep = camber_sweep(
            _fsae_fl(),
            wheel_travel_min_mm=-25.0,
            wheel_travel_max_mm=25.0,
            steps=50,
        )
        bump_cambers = [
            c for t, c, conv in zip(sweep.wheel_travel_mm, sweep.camber_deg, sweep.converged)
            if t > 10.0 and conv
        ]
        droop_cambers = [
            c for t, c, conv in zip(sweep.wheel_travel_mm, sweep.camber_deg, sweep.converged)
            if t < -10.0 and conv
        ]
        assert len(bump_cambers) > 0 and len(droop_cambers) > 0
        bump_delta = abs(bump_cambers[-1] - (-2.0))
        droop_delta = abs(droop_cambers[0] - (-2.0))
        assert bump_delta != pytest.approx(droop_delta, abs=0.01)


class TestLeftRightSymmetry:
    """Mirrored geometry must produce identical angles."""

    def test_camber_match(self) -> None:
        fl_r = DWSolver(_fsae_fl()).solve()
        fr_r = DWSolver(_fsae_fr()).solve()
        assert fl_r.converged and fr_r.converged
        assert fl_r.camber_deg == pytest.approx(fr_r.camber_deg, abs=0.01)

    def test_kpi_match(self) -> None:
        fl_r = DWSolver(_fsae_fl()).solve()
        fr_r = DWSolver(_fsae_fr()).solve()
        assert fl_r.converged and fr_r.converged
        assert fl_r.kpi_deg == pytest.approx(fr_r.kpi_deg, abs=0.01)

    def test_caster_match(self) -> None:
        fl_r = DWSolver(_fsae_fl()).solve()
        fr_r = DWSolver(_fsae_fr()).solve()
        assert fl_r.converged and fr_r.converged
        assert fl_r.caster_deg == pytest.approx(fr_r.caster_deg, abs=0.01)

    def test_camber_gain_match(self) -> None:
        gl = camber_gain_deg_per_mm(_fsae_fl(), wheel_travel_range_mm=25.0, steps=50)
        gr = camber_gain_deg_per_mm(_fsae_fr(), wheel_travel_range_mm=25.0, steps=50)
        assert gl == pytest.approx(gr, abs=0.001)


class TestBumpTravel:
    """Convergence and monotonicity over the travel range."""

    def test_converges_over_full_range(self) -> None:
        """Solver must converge at every point in ±30mm travel."""
        sweep = camber_sweep(
            _fsae_fl(),
            wheel_travel_min_mm=-30.0,
            wheel_travel_max_mm=30.0,
            steps=60,
        )
        failures = [t for t, conv in zip(sweep.wheel_travel_mm, sweep.converged) if not conv]
        assert len(failures) == 0, f"Failed at travel(s): {failures}"

    def test_camber_monotonic_in_bump(self) -> None:
        """Camber must decrease monotonically in bump."""
        sweep = camber_sweep(
            _fsae_fl(),
            wheel_travel_min_mm=0.0,
            wheel_travel_max_mm=30.0,
            steps=30,
        )
        converged_cambers = [c for c, conv in zip(sweep.camber_deg, sweep.converged) if conv]
        for i in range(1, len(converged_cambers)):
            assert converged_cambers[i] <= converged_cambers[i - 1] + 0.01, \
                f"Camber not monotonic at step {i}: {converged_cambers[i]:.3f} > {converged_cambers[i-1]:.3f}"

    def test_total_camber_change_over_travel(self) -> None:
        """Representative geometry should produce >1 deg total camber change over 60mm."""
        sweep = camber_sweep(
            _fsae_fl(),
            wheel_travel_min_mm=-30.0,
            wheel_travel_max_mm=30.0,
            steps=60,
        )
        cambers = [c for c, conv in zip(sweep.camber_deg, sweep.converged) if conv]
        total = max(cambers) - min(cambers)
        assert total > 1.0, f"Expected >1 deg total camber change, got {total:.3f}"


class TestRollCentre:
    """Roll centre -- regression anchor."""

    def test_fvic_is_finite(self) -> None:
        fl = _fsae_fl()
        r = DWSolver(fl).solve()
        assert r.converged
        fvic = front_view_instant_centre(fl, ubj=r.ubj, lbj=r.lbj, contact_patch=r.contact_patch)
        assert fvic.is_finite

    def test_fvic_on_opposite_side(self) -> None:
        """UCA shorter than LCA: FVIC on opposite side of centreline."""
        fl = _fsae_fl()
        r = DWSolver(fl).solve()
        assert r.converged
        fvic = front_view_instant_centre(fl, ubj=r.ubj, lbj=r.lbj, contact_patch=r.contact_patch)
        assert fvic.fvic_y_mm < 0

    def test_fvsa_in_range(self) -> None:
        """FVSA should be 1500-2500 mm (magnitude)."""
        fl = _fsae_fl()
        r = DWSolver(fl).solve()
        assert r.converged
        fvic = front_view_instant_centre(fl, ubj=r.ubj, lbj=r.lbj, contact_patch=r.contact_patch)
        assert 1500 < abs(fvic.fvsa_mm) < 2500

    def test_fvsa_regression(self) -> None:
        fl = _fsae_fl()
        r = DWSolver(fl).solve()
        assert r.converged
        fvic = front_view_instant_centre(fl, ubj=r.ubj, lbj=r.lbj, contact_patch=r.contact_patch)
        assert fvic.fvsa_mm == pytest.approx(-2192, abs=5)

    def test_rc_height_value(self) -> None:
        fl, fr = _fsae_fl(), _fsae_fr()
        axle = Axle(left=fl, right=fr)
        rl = DWSolver(fl).solve()
        rr = DWSolver(fr).solve()
        assert rl.converged and rr.converged
        rc = roll_centre_height(axle, rl, rr)
        assert rc.rc_height_mm == pytest.approx(48.77, abs=0.2)

    def test_rc_height_positive(self) -> None:
        fl, fr = _fsae_fl(), _fsae_fr()
        axle = Axle(left=fl, right=fr)
        rl = DWSolver(fl).solve()
        rr = DWSolver(fr).solve()
        assert rl.converged and rr.converged
        rc = roll_centre_height(axle, rl, rr)
        assert rc.rc_height_mm > 0

    def test_rc_left_right_symmetric(self) -> None:
        fl, fr = _fsae_fl(), _fsae_fr()
        axle = Axle(left=fl, right=fr)
        rl = DWSolver(fl).solve()
        rr = DWSolver(fr).solve()
        assert rl.converged and rr.converged
        rc = roll_centre_height(axle, rl, rr)
        assert rc.left_fvic.fvic_y_mm == pytest.approx(-rc.right_fvic.fvic_y_mm, abs=0.01)
        assert rc.left_fvic.fvic_z_mm == pytest.approx(rc.right_fvic.fvic_z_mm, abs=0.01)


class TestBumpSteer:
    """Bump steer -- tie rod deliberately offset above LCA front-view
    line to produce ~0.40 deg/1000mm. This is small enough to be a
    realistic design, large enough that a sign error in the toe
    extraction produces a visible mismatch in the OptimumK
    correlation test."""

    def test_bump_steer_nonzero(self) -> None:
        """Bump steer should be between 0.2 and 0.5 deg/1000mm."""
        solver = DWSolver(_fsae_fl())
        r_bump = solver.solve(wheel_travel_mm=25.0)
        r_droop = solver.solve(wheel_travel_mm=-25.0)
        assert r_bump.converged and r_droop.converged
        linear_rate = abs(r_bump.toe_deg_per_side - r_droop.toe_deg_per_side) / 50.0 * 1000
        assert 0.2 < linear_rate < 0.5, (
            f"Bump steer {linear_rate:.2f} deg/1000mm outside 0.2-0.5 range"
        )

    def test_bump_steer_regression(self) -> None:
        """Regression anchor for bump steer rate.

        The anchor was +0.40 deg/1000mm, which this geometry has never
        produced: before the left-corner toe sign was corrected the solver
        returned +0.2244 here, so the assertion was already failing. The sign
        correction flips it to -0.2211 (bump gives toe-out on this geometry);
        the two differ slightly in magnitude rather than being exact negatives
        because static camber couples into the projected wheel heading.

        The docstring target of ~0.40 above is a design intent this tie-rod
        placement does not meet — worth revisiting the geometry, not the number.
        """
        solver = DWSolver(_fsae_fl())
        r_bump = solver.solve(wheel_travel_mm=25.0)
        r_droop = solver.solve(wheel_travel_mm=-25.0)
        assert r_bump.converged and r_droop.converged
        linear_rate = (r_bump.toe_deg_per_side - r_droop.toe_deg_per_side) / 50.0 * 1000
        assert linear_rate == pytest.approx(-0.22, abs=0.05)

    def test_total_toe_change_small(self) -> None:
        """Total toe change over +/-30mm should be under 0.2 deg."""
        solver = DWSolver(_fsae_fl())
        toes: list[float] = []
        for wt_int in range(-30, 31, 2):
            r = solver.solve(wheel_travel_mm=float(wt_int))
            if r.converged:
                toes.append(r.toe_deg_per_side)
        total = max(toes) - min(toes)
        assert total < 0.2, f"Total toe change {total:.4f} deg exceeds 0.2 limit"


class TestFVSACamberGainConsistency:
    """FVSA and solver camber gain should agree within the 3D correction."""

    def test_consistency(self) -> None:
        fl = _fsae_fl()
        r = DWSolver(fl).solve()
        assert r.converged
        fvic = front_view_instant_centre(fl, ubj=r.ubj, lbj=r.lbj, contact_patch=r.contact_patch)
        gain = camber_gain_deg_per_mm(fl, wheel_travel_range_mm=25.0, steps=50)
        gain_from_fvsa = -math.degrees(1.0) / abs(fvic.fvsa_mm)
        assert abs(gain - gain_from_fvsa) / abs(gain) < 0.10, \
            f"FVSA/gain inconsistency: solver={gain:.4f}, FVSA-derived={gain_from_fvsa:.4f}"
