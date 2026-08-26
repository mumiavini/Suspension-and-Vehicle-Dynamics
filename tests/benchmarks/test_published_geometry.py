"""Regression benchmark for a synthetic FSAE-scale double-wishbone geometry.

NOT validated against an externally published set of coordinates with
known kinematic results. The hardpoint values below are authored by
the project — not taken from a textbook table. Milliken RCVD Ch. 17
describes the construction methods used here but does not publish a
complete XYZ coordinate set with a known RC height to compare against.

This file is a REGRESSION ANCHOR: if any of these values change, the
solver behaviour has changed. It is not external validation.

The geometry is an unequal-length non-parallel arm (SLA) front
suspension with FSAE-scale dimensions:
  - 1220 mm track (610 mm half-track)
  - 254 mm loaded tire radius (10" tire)
  - UCA shorter than LCA (gives negative camber in bump)
  - ~15 deg KPI, ~5.6 deg caster
  - Tie rod behind the axle line (rear-steer)

Regression values (updated 2026-08-25, contact-patch sign fix):
  KPI       = 14.93 deg
  Caster    = 5.58 deg
  RC height = 12.95 mm
  FVSA      = -8761 mm
  Camber gain = -0.0068 deg/mm

All coordinates in ISO 8855: X+ forward, Y+ left, Z+ up.
"""

from __future__ import annotations

import pytest

from vdcore.analysis.camber import camber_gain_deg_per_mm, camber_sweep
from vdcore.analysis.roll_centre import front_view_instant_centre, roll_centre_height
from vdcore.geometry.solver import DWSolver
from vdcore.models.hardpoint import Axle, Corner, Hardpoint, TirePackage


def _hp(name: str, x: float, y: float, z: float) -> Hardpoint:
    return Hardpoint(name=name, x_mm=x, y_mm=y, z_mm=z, source="cad", tol_mm=0.5)


def _tire() -> TirePackage:
    return TirePackage(loaded_radius_mm=254.0, source="cad", tol_mm=1.0)


def _rcvd_fl() -> Corner:
    """Front-left corner based on RCVD Ch. 17 unequal-length SLA.

    Geometry rationale (ISO 8855, all mm):
      UCA: 200 mm long (front view), inboard at Y=175, outboard at Y=520, Z=305
      LCA: 340 mm long (front view), inboard at Y=140, outboard at Y=580, Z=90
      Kingpin axis from (Y=580, Z=90) to (Y=520, Z=305) gives ~7.9 deg KPI.
      Caster from front/rear spacing of UCA/LCA pivots gives ~5 deg.
      Wheel centre at (0, 610, 254).
    """
    return Corner(
        corner_id="FL",
        uca_inboard_front=_hp("UCA_IF", 70, 175, 305),
        uca_inboard_rear=_hp("UCA_IR", -70, 175, 305),
        uca_outboard=_hp("UCA_O", -12, 520, 310),
        lca_inboard_front=_hp("LCA_IF", 110, 140, 90),
        lca_inboard_rear=_hp("LCA_IR", -90, 140, 90),
        lca_outboard=_hp("LCA_O", 10, 580, 85),
        tie_rod_inboard=_hp("TR_I", -80, 175, 110),
        tie_rod_outboard=_hp("TR_O", -70, 555, 105),
        wheel_center=_hp("WC", 0, 610, 254),
        tire=_tire(),
        static_camber_deg=-1.5,
        static_toe_deg_per_side=0.1,
    )


def _rcvd_fr() -> Corner:
    """Front-right mirror of the RCVD geometry."""
    fl = _rcvd_fl()
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


class TestRCVDStaticGeometry:
    """Regression anchors for kingpin-derived angles.

    Caster and KPI come from the kingpin axis (UBJ–LBJ line),
    which is entirely geometry-derived. Not externally validated —
    these are regression anchors against solver self-consistency.
    """

    def test_solver_converges(self) -> None:
        solver = DWSolver(_rcvd_fl())
        r = solver.solve()
        assert r.converged
        assert r.residual_norm < 1e-6

    def test_kpi_value(self) -> None:
        """KPI from kingpin axis — regression anchor."""
        solver = DWSolver(_rcvd_fl())
        r = solver.solve()
        assert r.converged
        assert r.kpi_deg == pytest.approx(14.93, abs=0.05)

    def test_caster_value(self) -> None:
        """Caster from kingpin axis — regression anchor."""
        solver = DWSolver(_rcvd_fl())
        r = solver.solve()
        assert r.converged
        assert r.caster_deg == pytest.approx(5.58, abs=0.05)

    def test_static_camber_equals_design_intent(self) -> None:
        """At zero travel, extracted camber must equal the design-intent input."""
        solver = DWSolver(_rcvd_fl())
        r = solver.solve()
        assert r.converged
        assert r.camber_deg == pytest.approx(-1.5, abs=0.01)

    def test_static_toe_equals_design_intent(self) -> None:
        """At zero travel, extracted toe must equal the design-intent input."""
        solver = DWSolver(_rcvd_fl())
        r = solver.solve()
        assert r.converged
        assert r.toe_deg_per_side == pytest.approx(0.1, abs=0.01)


class TestRCVDCamberGain:
    """Camber gain through wheel travel — geometry-dependent.

    With UCA shorter than LCA, bump produces more negative camber.
    The gain should be negative and within a physically reasonable range
    for an FSAE-scale SLA suspension (~0.01 to 0.1 deg/mm).
    """

    def test_camber_gain_sign_and_magnitude(self) -> None:
        gain = camber_gain_deg_per_mm(_rcvd_fl(), wheel_travel_range_mm=25.0, steps=50)
        assert gain < 0, "UCA shorter than LCA must produce negative camber gain"
        assert abs(gain) < 0.5, "Camber gain magnitude is unreasonably large"
        assert abs(gain) > 0.001, "Camber gain is unreasonably small"

    def test_bump_droop_asymmetry(self) -> None:
        """Camber change in bump vs droop should differ (non-linear linkage)."""
        sweep = camber_sweep(_rcvd_fl(), wheel_travel_min_mm=-25.0, wheel_travel_max_mm=25.0, steps=50)
        bump_cambers = [c for t, c, conv in zip(sweep.wheel_travel_mm, sweep.camber_deg, sweep.converged) if t > 10.0 and conv]
        droop_cambers = [c for t, c, conv in zip(sweep.wheel_travel_mm, sweep.camber_deg, sweep.converged) if t < -10.0 and conv]
        assert len(bump_cambers) > 0 and len(droop_cambers) > 0
        bump_delta = abs(bump_cambers[-1] - (-1.5))
        droop_delta = abs(droop_cambers[0] - (-1.5))
        assert bump_delta != pytest.approx(droop_delta, abs=0.01), \
            "Bump and droop camber change should differ for a non-linear linkage"


class TestRCVDLeftRightSymmetry:
    """Mirrored geometry must produce identical angles."""

    def test_camber_match(self) -> None:
        fl_r = DWSolver(_rcvd_fl()).solve()
        fr_r = DWSolver(_rcvd_fr()).solve()
        assert fl_r.converged and fr_r.converged
        assert fl_r.camber_deg == pytest.approx(fr_r.camber_deg, abs=0.01)

    def test_kpi_match(self) -> None:
        fl_r = DWSolver(_rcvd_fl()).solve()
        fr_r = DWSolver(_rcvd_fr()).solve()
        assert fl_r.converged and fr_r.converged
        assert fl_r.kpi_deg == pytest.approx(fr_r.kpi_deg, abs=0.01)

    def test_caster_match(self) -> None:
        fl_r = DWSolver(_rcvd_fl()).solve()
        fr_r = DWSolver(_rcvd_fr()).solve()
        assert fl_r.converged and fr_r.converged
        assert fl_r.caster_deg == pytest.approx(fr_r.caster_deg, abs=0.01)

    def test_camber_gain_match(self) -> None:
        gl = camber_gain_deg_per_mm(_rcvd_fl(), wheel_travel_range_mm=25.0, steps=50)
        gr = camber_gain_deg_per_mm(_rcvd_fr(), wheel_travel_range_mm=25.0, steps=50)
        assert gl == pytest.approx(gr, abs=0.001)


class TestRCVDBumpTravel:
    """Verify solver behaviour through wheel travel range."""

    def test_converges_over_full_range(self) -> None:
        """Solver must converge at every point in ±25mm travel."""
        sweep = camber_sweep(_rcvd_fl(), wheel_travel_min_mm=-25.0, wheel_travel_max_mm=25.0, steps=50)
        failures = [t for t, conv in zip(sweep.wheel_travel_mm, sweep.converged) if not conv]
        assert len(failures) == 0, f"Failed at travel(s): {failures}"

    def test_camber_monotonic_in_bump(self) -> None:
        """Camber must decrease monotonically in bump (UCA shorter than LCA)."""
        sweep = camber_sweep(_rcvd_fl(), wheel_travel_min_mm=0.0, wheel_travel_max_mm=25.0, steps=25)
        converged_cambers = [c for c, conv in zip(sweep.camber_deg, sweep.converged) if conv]
        for i in range(1, len(converged_cambers)):
            assert converged_cambers[i] <= converged_cambers[i - 1] + 0.01, \
                f"Camber not monotonic at step {i}: {converged_cambers[i]:.3f} > {converged_cambers[i-1]:.3f}"


class TestRCVDRollCentre:
    """Roll centre — regression anchor only (no external published value).

    This geometry produces a long FVSA (-8761 mm) and low RC (12.69 mm).
    The FVSA and camber gain are consistent: gain ≈ -(1/FVSA)*(180/pi)
    gives -0.0065 deg/mm vs solver's -0.0068 (4% from 3D/caster effects).
    """

    def test_fvic_is_finite(self) -> None:
        fl = _rcvd_fl()
        r = DWSolver(fl).solve()
        assert r.converged
        fvic = front_view_instant_centre(fl, ubj=r.ubj, lbj=r.lbj, contact_patch=r.contact_patch)
        assert fvic.is_finite

    def test_fvic_on_opposite_side_of_centreline(self) -> None:
        """For UCA shorter than LCA, the FVIC is on the opposite side
        of the centreline from the wheel (negative Y for left wheel)."""
        fl = _rcvd_fl()
        r = DWSolver(fl).solve()
        assert r.converged
        fvic = front_view_instant_centre(fl, ubj=r.ubj, lbj=r.lbj, contact_patch=r.contact_patch)
        assert fvic.fvic_y_mm < 0, "FVIC should be on opposite side for SLA"

    def test_fvsa_long_and_negative(self) -> None:
        """Long FVSA = IC far from wheel. Negative = IC on opposite side."""
        fl = _rcvd_fl()
        r = DWSolver(fl).solve()
        assert r.converged
        fvic = front_view_instant_centre(fl, ubj=r.ubj, lbj=r.lbj, contact_patch=r.contact_patch)
        assert fvic.fvsa_mm < -1000

    def test_rc_height_value(self) -> None:
        """Roll centre height -- regression anchor (not externally validated)."""
        fl, fr = _rcvd_fl(), _rcvd_fr()
        axle = Axle(left=fl, right=fr)
        rl = DWSolver(fl).solve()
        rr = DWSolver(fr).solve()
        assert rl.converged and rr.converged
        rc = roll_centre_height(axle, rl, rr)
        # Re-anchored 2026-08-25: was 12.69, recorded while the contact-patch
        # lateral shift carried an inverted sign. This fixture has -1.5 deg of
        # static camber, so the corrected patch moves 2*r*sin(1.5 deg) outboard
        # and the n-line construction lands 0.26 mm higher.
        assert rc.rc_height_mm == pytest.approx(12.95, abs=0.1)

    def test_rc_height_positive(self) -> None:
        """RC should be above ground for this geometry."""
        fl, fr = _rcvd_fl(), _rcvd_fr()
        axle = Axle(left=fl, right=fr)
        rl = DWSolver(fl).solve()
        rr = DWSolver(fr).solve()
        assert rl.converged and rr.converged
        rc = roll_centre_height(axle, rl, rr)
        assert rc.rc_height_mm > 0

    def test_rc_left_right_symmetric(self) -> None:
        """Symmetric axle must produce mirrored FVICs."""
        fl, fr = _rcvd_fl(), _rcvd_fr()
        axle = Axle(left=fl, right=fr)
        rl = DWSolver(fl).solve()
        rr = DWSolver(fr).solve()
        assert rl.converged and rr.converged
        rc = roll_centre_height(axle, rl, rr)
        assert rc.left_fvic.fvic_y_mm == pytest.approx(
            -rc.right_fvic.fvic_y_mm, abs=0.01
        )
        assert rc.left_fvic.fvic_z_mm == pytest.approx(
            rc.right_fvic.fvic_z_mm, abs=0.01
        )
