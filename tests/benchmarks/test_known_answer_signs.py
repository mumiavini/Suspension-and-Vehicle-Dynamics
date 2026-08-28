"""Known-answer cases that pin vdcore's SIGN CONVENTIONS to inspectable geometry.

WHY THIS FILE EXISTS
    The Altair MotionSolve cross-check (``altair_model/validate_kinematics.py``,
    ``altair_model/kpi_runner.py``) proves the two solvers put the upright in the
    same place -- to about 1e-7 mm. It proves nothing about whether the ANGLES
    are labelled correctly, because both sides deliberately run through
    ``DWSolver._extract_angles``. If vdcore's camber convention were inverted,
    both columns would be inverted together and still agree perfectly.

    Comparing against another solver cannot close that gap either: if Altair
    reported +1.5 deg where vdcore reports -1.5, that is ambiguous between "we
    have a sign error" and "Altair is using SAE J670 where we use ISO 8855".
    Two conventions, no referee.

    The referee has to be a geometry whose answer is obvious BEFORE running
    anything. That is what these cases are. Each one is built so extreme and so
    simple that the correct sign can be read off the coordinates by hand, and
    each asserts the sign from the geometry rather than from a stored value.

HOW THESE DIFFER FROM THE EXISTING BENCHMARKS
    ``test_camber_benchmark.py`` and ``test_fsae_representative.py`` are
    regression anchors ("if this number changes, something changed") and
    symmetry checks ("left matches right"). Both would happily pass with every
    sign inverted, as long as it were inverted consistently. These do not.

Frame: ISO 8855 -- X+ forward, Y+ LEFT, Z+ up. Left corners have positive Y.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vdcore.analysis.roll_centre import front_view_instant_centre, roll_centre_height
from vdcore.geometry.derived import mechanical_trail_mm, scrub_radius_mm
from vdcore.geometry.solver import DWSolver, SolverResult
from vdcore.models.hardpoint import Axle, Corner, Hardpoint, TirePackage

# Every case is planar in the front view unless it is testing caster: the
# inboard pivot axes run parallel to X at +/-PIVOT_DX, and both ball joints sit
# at x = 0. That makes the Y-Z coordinates the whole story, which is what makes
# the answers readable by hand.
PIVOT_DX = 100.0


def _hp(name: str, x: float, y: float, z: float) -> Hardpoint:
    return Hardpoint(name=name, x_mm=x, y_mm=y, z_mm=z, source="design_intent", tol_mm=0.0)


def _build(
    corner_id: str,
    *,
    lca_in: tuple[float, float],
    lca_out: tuple[float, float],
    uca_in: tuple[float, float],
    uca_out: tuple[float, float],
    wc: tuple[float, float],
    radius: float,
    tie_in: tuple[float, float, float] = (-60.0, 200.0, 150.0),
    tie_out: tuple[float, float, float] = (-60.0, 560.0, 140.0),
    uca_out_x: float = 0.0,
    lca_out_x: float = 0.0,
    camber_deg: float = 0.0,
    toe_deg: float = 0.0,
) -> Corner:
    """Build a corner from front-view (Y, Z) pairs, mirroring Y for the right side.

    Passing the same numbers for FL and FR is the point: a convention that is
    right on one side and inverted on the other is the classic double-wishbone
    bug, and it only shows up if both sides are built from one description.
    """
    s = 1.0 if corner_id in ("FL", "RL") else -1.0

    def p(name: str, x: float, yz: tuple[float, float]) -> Hardpoint:
        return _hp(name, x, s * yz[0], yz[1])

    return Corner(
        corner_id=corner_id,
        uca_inboard_front=p("UCA_IF", +PIVOT_DX, uca_in),
        uca_inboard_rear=p("UCA_IR", -PIVOT_DX, uca_in),
        uca_outboard=p("UCA_O", uca_out_x, uca_out),
        lca_inboard_front=p("LCA_IF", +PIVOT_DX, lca_in),
        lca_inboard_rear=p("LCA_IR", -PIVOT_DX, lca_in),
        lca_outboard=p("LCA_O", lca_out_x, lca_out),
        tie_rod_inboard=_hp("TR_I", tie_in[0], s * tie_in[1], tie_in[2]),
        tie_rod_outboard=_hp("TR_O", tie_out[0], s * tie_out[1], tie_out[2]),
        wheel_center=p("WC", 0.0, wc),
        tire=TirePackage(loaded_radius_mm=radius, source="design_intent", tol_mm=0.0),
        static_camber_deg=camber_deg,
        static_toe_deg_per_side=toe_deg,
    )


def _wheel_top(corner: Corner, result: SolverResult, radius: float) -> np.ndarray:
    """Where the TOP of the wheel actually is, from the solved spin axis.

    This is the literal definition camber is supposed to encode: the wheel plane
    is perpendicular to the spin axis, and the top of the wheel is the rim point
    directly above the wheel centre within that plane. Computed here WITHOUT
    touching ``_extract_angles``, so it is an independent statement about where
    the wheel physically leans.

    Reaching for ``_reconstruct_spin_axis`` is deliberate: that is the rigid-body
    CONSTRUCTION (which the Altair check confirms), not the angle extraction
    (which it cannot).
    """
    solver = DWSolver(corner)
    ubj = np.array([result.ubj.x_mm, result.ubj.y_mm, result.ubj.z_mm])
    lbj = np.array([result.lbj.x_mm, result.lbj.y_mm, result.lbj.z_mm])
    tro = np.array([result.tro.x_mm, result.tro.y_mm, result.tro.z_mm])
    spin = solver._reconstruct_spin_axis(ubj, lbj, tro)

    wc = np.array([
        result.wheel_center.x_mm, result.wheel_center.y_mm, result.wheel_center.z_mm,
    ])
    # Project global "up" onto the wheel plane to get the in-plane vertical.
    up = np.array([0.0, 0.0, 1.0])
    in_plane = up - np.dot(up, spin) * spin
    in_plane /= np.linalg.norm(in_plane)
    return wc + radius * in_plane


def _wheel_front(corner: Corner, result: SolverResult, radius: float) -> np.ndarray:
    """Where the FRONT edge of the wheel is, from the solved spin axis.

    Same idea as :func:`_wheel_top` but projecting global "forward" (+X) onto
    the wheel plane instead of "up". Toe is the yaw of that plane, so this is
    the point whose sideways position says which way the wheel is pointing.
    """
    solver = DWSolver(corner)
    ubj = np.array([result.ubj.x_mm, result.ubj.y_mm, result.ubj.z_mm])
    lbj = np.array([result.lbj.x_mm, result.lbj.y_mm, result.lbj.z_mm])
    tro = np.array([result.tro.x_mm, result.tro.y_mm, result.tro.z_mm])
    spin = solver._reconstruct_spin_axis(ubj, lbj, tro)

    wc = np.array([
        result.wheel_center.x_mm, result.wheel_center.y_mm, result.wheel_center.z_mm,
    ])
    fwd = np.array([1.0, 0.0, 0.0])
    in_plane = fwd - np.dot(fwd, spin) * spin
    in_plane /= np.linalg.norm(in_plane)
    return wc + radius * in_plane


# =========================================================================== #
# 1. CAMBER -- negative means the TOP of the wheel is INBOARD, on both sides
# =========================================================================== #

class TestCamberSignIsWhereTheWheelActuallyLeans:
    """The definition, checked against the physical top of the wheel.

    Built with a large static camber so the lean is unmistakable: at -8 deg on a
    250 mm wheel the top sits ~35 mm inboard of the wheel centre. No rounding,
    no small-angle ambiguity -- either the top is inboard or the sign is wrong.
    """

    RADIUS = 250.0

    def _corner(self, corner_id: str, camber_deg: float) -> Corner:
        return _build(
            corner_id,
            lca_in=(200.0, 120.0), lca_out=(580.0, 120.0),
            uca_in=(220.0, 330.0), uca_out=(560.0, 340.0),
            wc=(600.0, 250.0), radius=self.RADIUS,
            camber_deg=camber_deg,
        )

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_negative_camber_puts_the_wheel_top_inboard(self, corner_id: str) -> None:
        corner = self._corner(corner_id, -8.0)
        result = DWSolver(corner).solve()
        assert result.converged

        top = _wheel_top(corner, result, self.RADIUS)
        wc_y = result.wheel_center.y_mm
        # Outboard is +Y on the left, -Y on the right.
        outboard = 1.0 if corner_id in ("FL", "RL") else -1.0
        lean_outboard_mm = (top[1] - wc_y) * outboard

        assert result.camber_deg < 0, f"{corner_id}: expected negative camber"
        assert lean_outboard_mm < -30.0, (
            f"{corner_id}: camber reads {result.camber_deg:+.2f} deg but the wheel "
            f"top is {lean_outboard_mm:+.1f} mm outboard of the wheel centre. "
            f"Negative camber must put the top INBOARD."
        )

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_positive_camber_puts_the_wheel_top_outboard(self, corner_id: str) -> None:
        """The mirror case. A hard-coded sign passes one of these, not both."""
        corner = self._corner(corner_id, +8.0)
        result = DWSolver(corner).solve()
        assert result.converged

        top = _wheel_top(corner, result, self.RADIUS)
        outboard = 1.0 if corner_id in ("FL", "RL") else -1.0
        lean_outboard_mm = (top[1] - result.wheel_center.y_mm) * outboard

        assert result.camber_deg > 0
        assert lean_outboard_mm > 30.0

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_camber_magnitude_equals_the_wheel_plane_tilt(self, corner_id: str) -> None:
        """Not just the sign: the number must be the actual tilt in degrees."""
        corner = self._corner(corner_id, -8.0)
        result = DWSolver(corner).solve()

        top = _wheel_top(corner, result, self.RADIUS)
        wc = np.array([
            result.wheel_center.x_mm, result.wheel_center.y_mm, result.wheel_center.z_mm,
        ])
        lean = top - wc
        tilt_deg = math.degrees(math.atan2(abs(lean[1]), lean[2]))

        assert tilt_deg == pytest.approx(abs(result.camber_deg), abs=0.05)


# =========================================================================== #
# 2. CAMBER GAIN -- the arm-length ratio decides the sign, and can flip it
# =========================================================================== #

class TestCamberGainSignFollowsInstantCentreSide:
    """Which SIDE of the car the instant centre falls on decides the sign.

    Worth stating precisely, because the usual shorthand -- "a shorter upper arm
    gains negative camber" -- is not what actually drives it, and building this
    test the naive way is instructive: with both wishbones HORIZONTAL the arm
    lines are parallel, the instant centre is at infinity, and changing the
    upper arm's LENGTH alone does not flip the gain at all (measured -0.0098 vs
    -0.0055 deg/mm, same sign). Length matters only through where it puts the
    instant centre.

    So both cases here keep the LCA horizontal from y=200 to y=580 and move only
    the UCA's OUTBOARD HEIGHT by +/-20 mm, which swings the instant centre from
    one side of the car to the other:

        UCA outboard 20 mm HIGHER -> arm lines meet at y = -3980 (far side)
        UCA outboard 20 mm LOWER  -> arm lines meet at y = +4380 (near side)

    A far-side instant centre is the normal SLA arrangement and must gain
    negative camber in bump; a near-side one must reverse it.
    """

    def _corner(self, corner_id: str, uca_out_z: float) -> Corner:
        return _build(
            corner_id,
            lca_in=(200.0, 120.0), lca_out=(580.0, 120.0),
            uca_in=(200.0, 340.0), uca_out=(580.0, uca_out_z),
            wc=(620.0, 250.0), radius=250.0,
        )

    @staticmethod
    def _gain(corner: Corner) -> float:
        solver = DWSolver(corner)
        up = solver.solve(wheel_travel_mm=+10.0)
        dn = solver.solve(wheel_travel_mm=-10.0)
        assert up.converged and dn.converged
        return (up.camber_deg - dn.camber_deg) / 20.0

    @staticmethod
    def _fvic_y(corner: Corner) -> float:
        result = DWSolver(corner).solve()
        return front_view_instant_centre(
            corner, ubj=result.ubj, lbj=result.lbj, contact_patch=result.contact_patch
        ).fvic_y_mm

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_far_side_instant_centre_gains_negative_camber(self, corner_id: str) -> None:
        corner = self._corner(corner_id, 360.0)
        outboard = 1.0 if corner_id in ("FL", "RL") else -1.0
        assert self._fvic_y(corner) * outboard < -1000.0, (
            "geometry check: instant centre must be on the FAR side"
        )
        gain = self._gain(corner)
        assert gain < -1e-3, (
            f"{corner_id}: a far-side instant centre must gain NEGATIVE camber "
            f"in bump, got {gain:+.5f} deg/mm"
        )

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_near_side_instant_centre_gains_positive_camber(self, corner_id: str) -> None:
        """The reversal. A hard-coded sign passes one of these two, not both."""
        corner = self._corner(corner_id, 320.0)
        outboard = 1.0 if corner_id in ("FL", "RL") else -1.0
        assert self._fvic_y(corner) * outboard > 1000.0, (
            "geometry check: instant centre must be on the NEAR side"
        )
        gain = self._gain(corner)
        assert gain > 1e-3, (
            f"{corner_id}: a near-side instant centre must gain POSITIVE camber "
            f"in bump, got {gain:+.5f} deg/mm"
        )

    def test_instant_centre_side_mirrors_between_left_and_right(self) -> None:
        """Far side means -Y on the left and +Y on the right, same geometry."""
        left = self._fvic_y(self._corner("FL", 360.0))
        right = self._fvic_y(self._corner("FR", 360.0))
        assert left < 0 < right
        assert left == pytest.approx(-right, rel=1e-9)


# =========================================================================== #
# 3. EQUAL PARALLEL ARMS -- the degenerate case with a known answer
# =========================================================================== #

class TestEqualParallelHorizontalArms:
    """Three equal-length parallel horizontal links: the upright cannot rotate.

    It translates vertically, so camber is EXACTLY constant through travel and
    the front-view instant centre is at infinity. Both are known before running
    anything, and the camber one is a strong check: any spurious coupling into
    the camber path shows up as a non-zero change here.

    The tie rod has to be parallel and equal-length too, which is worth spelling
    out because getting it wrong is subtle. With the wishbones parallel but the
    tie rod left at a general position, the tie rod steers the upright through
    travel and camber drifts by ~5e-5 deg over +/-20 mm -- small enough to look
    like solver noise, but it is real linkage behaviour, not noise. Making all
    three links parallel removes it and camber holds to 1e-10.
    """

    def _corner(self, corner_id: str) -> Corner:
        return _build(
            corner_id,
            lca_in=(200.0, 120.0), lca_out=(580.0, 120.0),   # 380 mm, horizontal
            uca_in=(200.0, 340.0), uca_out=(580.0, 340.0),   # 380 mm, horizontal
            tie_in=(-60.0, 200.0, 230.0),                     # 380 mm, horizontal
            tie_out=(-60.0, 580.0, 230.0),
            wc=(600.0, 250.0), radius=250.0,
            camber_deg=-1.0,
        )

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_camber_is_constant_through_travel(self, corner_id: str) -> None:
        solver = DWSolver(self._corner(corner_id))
        static = solver.solve()
        for travel in (-20.0, -10.0, 10.0, 20.0):
            moved = solver.solve(wheel_travel_mm=travel)
            assert moved.converged
            assert moved.camber_deg == pytest.approx(static.camber_deg, abs=1e-8), (
                f"{corner_id}: three parallel equal links cannot rotate the "
                f"upright, so camber must not change at {travel:+.0f} mm"
            )

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_toe_is_constant_through_travel(self, corner_id: str) -> None:
        """Pure translation means zero bump steer, too."""
        solver = DWSolver(self._corner(corner_id))
        static = solver.solve()
        for travel in (-20.0, 20.0):
            moved = solver.solve(wheel_travel_mm=travel)
            assert moved.toe_deg_per_side == pytest.approx(
                static.toe_deg_per_side, abs=1e-8
            )

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_front_view_instant_centre_is_at_infinity(self, corner_id: str) -> None:
        corner = self._corner(corner_id)
        result = DWSolver(corner).solve()
        fvic = front_view_instant_centre(
            corner, ubj=result.ubj, lbj=result.lbj, contact_patch=result.contact_patch
        )
        assert not fvic.is_finite, "parallel arms have no finite instant centre"

    def test_roll_centre_refuses_rather_than_inventing_a_number(self) -> None:
        """Physically the RC is at ground here, but the construction is degenerate.

        The project rule is that a degenerate construction fails loudly instead
        of returning a plausible number, so this pins the RAISE, not a value.
        """
        axle = Axle(left=self._corner("FL"), right=self._corner("FR"))
        left = DWSolver(axle.left).solve()
        right = DWSolver(axle.right).solve()
        with pytest.raises(RuntimeError, match="infinity"):
            roll_centre_height(axle, left, right)


# =========================================================================== #
# 4. ROLL CENTRE -- hand-computed from a geometry with a known instant centre
# =========================================================================== #

class TestRollCentreAgainstHandComputation:
    """A geometry whose RC height can be worked out on paper.

    Front view, left corner (all mm):
        LCA horizontal at z = 120, from y = 200 to y = 580
        UCA from (200, 340) to (580, 360)   ->  slope +20/380

    The LCA line is z = 120. The UCA line is z = 340 + (20/380)(y - 200).
    They meet where 120 = 340 + (20/380)(y - 200), i.e. y - 200 = -4180,
    so the instant centre is at y = -3980, z = 120 -- on the FAR side of the
    car, as it should be for a normal SLA.

    With zero static camber the contact patch is directly under the wheel
    centre at (620, 0). The roll centre is where the patch-to-IC line crosses
    the centreline:

        slope = (120 - 0) / (-3980 - 620) = -0.0260870
        z(y=0) = 0 + (-0.0260870)(0 - 620) = 16.174 mm
    """

    EXPECTED_IC_Y = -3980.0
    EXPECTED_IC_Z = 120.0
    EXPECTED_RC_MM = 16.174

    def _corner(self, corner_id: str) -> Corner:
        return _build(
            corner_id,
            lca_in=(200.0, 120.0), lca_out=(580.0, 120.0),
            uca_in=(200.0, 340.0), uca_out=(580.0, 360.0),
            wc=(620.0, 250.0), radius=250.0,   # patch lands exactly on z = 0
            camber_deg=0.0,
        )

    def test_instant_centre_is_where_the_arm_lines_cross(self) -> None:
        corner = self._corner("FL")
        result = DWSolver(corner).solve()
        fvic = front_view_instant_centre(
            corner, ubj=result.ubj, lbj=result.lbj, contact_patch=result.contact_patch
        )
        assert fvic.is_finite
        assert fvic.fvic_y_mm == pytest.approx(self.EXPECTED_IC_Y, rel=1e-3)
        assert fvic.fvic_z_mm == pytest.approx(self.EXPECTED_IC_Z, abs=0.5)

    def test_contact_patch_sits_on_the_ground_under_the_wheel_centre(self) -> None:
        """Zero camber, loaded radius = wheel-centre height: patch at (620, 0)."""
        result = DWSolver(self._corner("FL")).solve()
        assert result.contact_patch.z_mm == pytest.approx(0.0, abs=1e-9)
        assert result.contact_patch.y_mm == pytest.approx(620.0, abs=1e-6)

    def test_roll_centre_height_matches_the_hand_computation(self) -> None:
        axle = Axle(left=self._corner("FL"), right=self._corner("FR"))
        left = DWSolver(axle.left).solve()
        right = DWSolver(axle.right).solve()
        rc = roll_centre_height(axle, left, right)

        assert rc.rc_height_mm == pytest.approx(self.EXPECTED_RC_MM, abs=0.05)
        assert rc.rc_y_mm == pytest.approx(0.0, abs=1e-6), "symmetric axle: RC on centreline"

    def test_roll_centre_is_above_ground_for_this_geometry(self) -> None:
        """Sanity in the direction a designer would notice: IC above the patch
        plane and outside the car puts the RC above ground, not below."""
        axle = Axle(left=self._corner("FL"), right=self._corner("FR"))
        rc = roll_centre_height(
            axle, DWSolver(axle.left).solve(), DWSolver(axle.right).solve()
        )
        assert 0.0 < rc.rc_height_mm < 60.0


# =========================================================================== #
# 5. CASTER -- positive means the kingpin TOP leans REARWARD (same both sides)
# =========================================================================== #

class TestCasterSign:
    """Caster is a side-view angle, so it is NOT mirrored between sides.

    Built with the upper ball joint 60 mm behind the lower on a ~220 mm
    kingpin: about 15 deg, far too big to be a rounding artefact.
    """

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_upper_ball_joint_behind_lower_is_positive_caster(self, corner_id: str) -> None:
        corner = _build(
            corner_id,
            lca_in=(200.0, 120.0), lca_out=(580.0, 120.0),
            uca_in=(220.0, 340.0), uca_out=(560.0, 340.0),
            wc=(600.0, 250.0), radius=250.0,
            uca_out_x=-60.0, lca_out_x=0.0,   # X+ is forward, so -60 is REARWARD
        )
        result = DWSolver(corner).solve()
        assert result.converged
        assert result.ubj.x_mm < result.lbj.x_mm, "geometry check: top must be rearward"
        assert result.caster_deg > 10.0, (
            f"{corner_id}: kingpin top {result.lbj.x_mm - result.ubj.x_mm:.0f} mm "
            f"rearward must give POSITIVE caster, got {result.caster_deg:+.2f} deg"
        )

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_upper_ball_joint_ahead_of_lower_is_negative_caster(self, corner_id: str) -> None:
        corner = _build(
            corner_id,
            lca_in=(200.0, 120.0), lca_out=(580.0, 120.0),
            uca_in=(220.0, 340.0), uca_out=(560.0, 340.0),
            wc=(600.0, 250.0), radius=250.0,
            uca_out_x=+60.0, lca_out_x=0.0,
        )
        result = DWSolver(corner).solve()
        assert result.converged
        assert result.caster_deg < -10.0

    def test_caster_is_not_mirrored_between_left_and_right(self) -> None:
        """The classic trap: caster must NOT flip sign across the car."""
        kwargs = dict(
            lca_in=(200.0, 120.0), lca_out=(580.0, 120.0),
            uca_in=(220.0, 340.0), uca_out=(560.0, 340.0),
            wc=(600.0, 250.0), radius=250.0, uca_out_x=-60.0,
        )
        left = DWSolver(_build("FL", **kwargs)).solve()      # type: ignore[arg-type]
        right = DWSolver(_build("FR", **kwargs)).solve()     # type: ignore[arg-type]
        assert left.caster_deg == pytest.approx(right.caster_deg, abs=1e-6)
        assert left.caster_deg > 0


# =========================================================================== #
# 6. KPI -- positive means the kingpin TOP leans INBOARD (mirrored per side)
# =========================================================================== #

class TestKingpinInclinationSign:
    """KPI is a front-view angle, so it IS mirrored: inboard is -Y left, +Y right.

    Upper ball joint 80 mm inboard of the lower on a ~220 mm kingpin: ~20 deg.
    """

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_upper_ball_joint_inboard_of_lower_is_positive_kpi(self, corner_id: str) -> None:
        corner = _build(
            corner_id,
            lca_in=(200.0, 120.0), lca_out=(580.0, 120.0),
            uca_in=(220.0, 340.0), uca_out=(500.0, 340.0),   # 80 mm inboard
            wc=(600.0, 250.0), radius=250.0,
        )
        result = DWSolver(corner).solve()
        assert result.converged

        inboard = -1.0 if corner_id in ("FL", "RL") else 1.0
        lean_inboard_mm = (result.ubj.y_mm - result.lbj.y_mm) * inboard
        assert lean_inboard_mm > 50.0, "geometry check: top must be well inboard"
        assert result.kpi_deg > 10.0, (
            f"{corner_id}: kingpin top {lean_inboard_mm:.0f} mm inboard must give "
            f"POSITIVE KPI, got {result.kpi_deg:+.2f} deg"
        )

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_upper_ball_joint_outboard_of_lower_is_negative_kpi(self, corner_id: str) -> None:
        corner = _build(
            corner_id,
            lca_in=(200.0, 120.0), lca_out=(520.0, 120.0),
            uca_in=(220.0, 340.0), uca_out=(600.0, 340.0),   # 80 mm OUTBOARD
            wc=(620.0, 250.0), radius=250.0,
        )
        result = DWSolver(corner).solve()
        assert result.converged
        assert result.kpi_deg < -10.0


# =========================================================================== #
# 7. SCRUB RADIUS -- positive means the contact patch is OUTBOARD of the
#    kingpin's ground intercept, on both sides
# =========================================================================== #

class TestScrubRadiusSign:
    """Extend the kingpin to the ground and see which side of it the patch is on.

    Both cases are built so the intercept lands tens of mm from the patch, and
    the expected value is computed here from similar triangles rather than
    stored, so the assertion states the geometry rather than a past result.
    """

    @staticmethod
    def _kingpin_ground_y(result: SolverResult) -> float:
        """Y where the LBJ->UBJ line crosses z = 0, by similar triangles."""
        dy = result.ubj.y_mm - result.lbj.y_mm
        dz = result.ubj.z_mm - result.lbj.z_mm
        return result.lbj.y_mm - dy * (result.lbj.z_mm / dz)

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_patch_outboard_of_kingpin_intercept_is_positive_scrub(
        self, corner_id: str
    ) -> None:
        """NEGATIVE KPI tips the kingpin top outboard, which swings its GROUND
        intercept inboard -- the intercept moves opposite to the top, because
        the ground is below the lower ball joint. Here: LBJ y=560, UBJ y=620,
        so the intercept lands at 560 - 60*(120/220) = 527.3, and the patch at
        620 sits ~93 mm outboard of it.
        """
        corner = _build(
            corner_id,
            lca_in=(200.0, 120.0), lca_out=(560.0, 120.0),
            uca_in=(220.0, 340.0), uca_out=(620.0, 340.0),
            wc=(620.0, 250.0), radius=250.0,
        )
        result = DWSolver(corner).solve()
        assert result.converged
        assert result.kpi_deg < -10.0, "geometry check: kingpin top must lean outboard"

        outboard = 1.0 if corner_id in ("FL", "RL") else -1.0
        gap_mm = (result.contact_patch.y_mm - self._kingpin_ground_y(result)) * outboard
        assert gap_mm > 50.0, "geometry check: patch must be clearly outboard"
        assert scrub_radius_mm(result) == pytest.approx(gap_mm, abs=0.01), (
            f"{corner_id}: scrub must equal the patch-to-intercept gap, "
            f"positive when the patch is outboard"
        )

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_patch_inboard_of_kingpin_intercept_is_negative_scrub(
        self, corner_id: str
    ) -> None:
        """The reversal: a large POSITIVE KPI throws the ground intercept far
        enough outboard to pass the patch. LBJ y=520, UBJ y=440 gives an
        intercept at 520 + 80*(120/220) = 563.6, outboard of the patch at 540.
        """
        corner = _build(
            corner_id,
            lca_in=(200.0, 120.0), lca_out=(520.0, 120.0),
            uca_in=(220.0, 340.0), uca_out=(440.0, 340.0),
            wc=(540.0, 250.0), radius=250.0,
        )
        result = DWSolver(corner).solve()
        assert result.converged
        assert result.kpi_deg > 10.0, "geometry check: kingpin top must lean inboard"

        outboard = 1.0 if corner_id in ("FL", "RL") else -1.0
        gap_mm = (result.contact_patch.y_mm - self._kingpin_ground_y(result)) * outboard
        assert gap_mm < -10.0, "geometry check: patch must be clearly inboard"
        assert scrub_radius_mm(result) < 0.0
        assert scrub_radius_mm(result) == pytest.approx(gap_mm, abs=0.01)


# =========================================================================== #
# 8. BUMP STEER -- moving the tie rod across the axle line reverses it
# =========================================================================== #

class TestContactPatchMovesOutboardWithNegativeCamber:
    """The patch is at the BOTTOM of the wheel, so it moves OPPOSITE to the top.

    Negative camber tips the top inboard, which swings the bottom -- and the
    contact patch -- OUTBOARD, widening the ground track and increasing scrub.
    This is the counter-intuitive one, and it is load-bearing: scrub radius,
    mechanical trail and the whole roll-centre construction are built on the
    patch position, so an inverted shift here quietly corrupts all three.

    Added after a mutation test: flipping the shift direction in
    ``DWSolver.solve`` was NOT caught by the rest of this file, because every
    other case here uses zero static camber, where the shift vanishes.

    ``loaded_radius_mm`` is the VERTICAL wheel-centre-to-road distance, so the
    patch stays exactly on z = 0 whatever the camber -- also checked here.
    """

    RADIUS = 250.0

    def _corner(self, corner_id: str, camber_deg: float) -> Corner:
        return _build(
            corner_id,
            lca_in=(200.0, 120.0), lca_out=(580.0, 120.0),
            uca_in=(220.0, 340.0), uca_out=(560.0, 340.0),
            wc=(600.0, self.RADIUS), radius=self.RADIUS,
            camber_deg=camber_deg,
        )

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_negative_camber_pushes_the_patch_outboard(self, corner_id: str) -> None:
        result = DWSolver(self._corner(corner_id, -8.0)).solve()
        assert result.converged

        outboard = 1.0 if corner_id in ("FL", "RL") else -1.0
        shift_mm = (result.contact_patch.y_mm - result.wheel_center.y_mm) * outboard
        expected = self.RADIUS * math.tan(math.radians(8.0))

        assert shift_mm == pytest.approx(expected, abs=0.05), (
            f"{corner_id}: -8 deg camber must move the patch {expected:.1f} mm "
            f"OUTBOARD, got {shift_mm:+.1f} mm"
        )

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_positive_camber_pulls_the_patch_inboard(self, corner_id: str) -> None:
        result = DWSolver(self._corner(corner_id, +8.0)).solve()
        outboard = 1.0 if corner_id in ("FL", "RL") else -1.0
        shift_mm = (result.contact_patch.y_mm - result.wheel_center.y_mm) * outboard
        assert shift_mm == pytest.approx(
            -self.RADIUS * math.tan(math.radians(8.0)), abs=0.05
        )

    @pytest.mark.parametrize("camber_deg", [-8.0, 0.0, +8.0])
    def test_patch_stays_on_the_ground_plane_whatever_the_camber(
        self, camber_deg: float
    ) -> None:
        """Loaded radius is the VERTICAL drop, so z = 0 always."""
        result = DWSolver(self._corner("FL", camber_deg)).solve()
        assert result.contact_patch.z_mm == pytest.approx(0.0, abs=1e-9)


class TestMechanicalTrailSign:
    """Positive trail: the kingpin's ground intercept lies AHEAD of the patch.

    That is the self-aligning case -- the tyre trails behind the steer axis, as
    on a caster wheel. X+ is FORWARD in ISO 8855, so "ahead" means larger X.

    Worth an explicit case because the legacy app has this sign INVERTED for
    this frame (see the note in ``vdcore.geometry.derived.mechanical_trail_mm``),
    so it is a known place where two parts of this project disagreed. Added
    after a mutation test showed nothing in this file covered it.

    Geometry: LBJ at x=0 z=120, UBJ 60 mm rearward at z=340. Dropping that axis
    to the ground moves it forward by 60*(120/220) = 32.7 mm, so the intercept
    lands at x=+32.7 against a patch at x=0.
    """

    EXPECTED_TRAIL_MM = 60.0 * (120.0 / 220.0)

    def _corner(self, corner_id: str, uca_out_x: float) -> Corner:
        return _build(
            corner_id,
            lca_in=(200.0, 120.0), lca_out=(580.0, 120.0),
            uca_in=(220.0, 340.0), uca_out=(560.0, 340.0),
            wc=(600.0, 250.0), radius=250.0,
            uca_out_x=uca_out_x, lca_out_x=0.0,
        )

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_positive_caster_gives_positive_trail(self, corner_id: str) -> None:
        result = DWSolver(self._corner(corner_id, -60.0)).solve()
        assert result.converged
        assert result.caster_deg > 10.0, "geometry check: kingpin top rearward"

        trail = mechanical_trail_mm(result)
        assert trail == pytest.approx(self.EXPECTED_TRAIL_MM, abs=0.5), (
            f"{corner_id}: kingpin ground intercept should sit "
            f"{self.EXPECTED_TRAIL_MM:.1f} mm AHEAD of the patch, "
            f"trail reads {trail:+.2f} mm"
        )

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_negative_caster_gives_negative_trail(self, corner_id: str) -> None:
        """The reversal: kingpin top forward puts the intercept behind the patch."""
        result = DWSolver(self._corner(corner_id, +60.0)).solve()
        assert result.caster_deg < -10.0
        assert mechanical_trail_mm(result) == pytest.approx(
            -self.EXPECTED_TRAIL_MM, abs=0.5
        )

    def test_trail_is_not_mirrored_between_sides(self) -> None:
        """Like caster, trail is a side-view quantity: same sign both sides."""
        left = DWSolver(self._corner("FL", -60.0)).solve()
        right = DWSolver(self._corner("FR", -60.0)).solve()
        assert mechanical_trail_mm(left) == pytest.approx(
            mechanical_trail_mm(right), abs=1e-9
        )
        assert mechanical_trail_mm(left) > 0


class TestToeSignIsWhereTheWheelActuallyPoints:
    """Positive toe means TOE-IN: the front of the wheel turns toward the centreline.

    This was added after a mutation test showed the rest of this file did not
    catch a globally inverted toe sign. The bump-steer case below only checks
    that the sign REVERSES between two layouts, which is invariant under a
    global flip -- so an absolute anchor was missing. It is the same trap the
    convention itself sets: per-side versus total toe, and in versus out, are
    both easy to invert consistently and never notice.

    Built at +/-5 deg on a 250 mm wheel, which swings the front of the wheel
    about 22 mm sideways -- unmistakable.
    """

    RADIUS = 250.0

    def _corner(self, corner_id: str, toe_deg: float) -> Corner:
        return _build(
            corner_id,
            lca_in=(200.0, 120.0), lca_out=(580.0, 120.0),
            uca_in=(220.0, 340.0), uca_out=(560.0, 340.0),
            wc=(600.0, 250.0), radius=self.RADIUS,
            toe_deg=toe_deg,
        )

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_positive_toe_points_the_wheel_front_inboard(self, corner_id: str) -> None:
        corner = self._corner(corner_id, +5.0)
        result = DWSolver(corner).solve()
        assert result.converged

        front = _wheel_front(corner, result, self.RADIUS)
        inboard = -1.0 if corner_id in ("FL", "RL") else 1.0
        swing_inboard_mm = (front[1] - result.wheel_center.y_mm) * inboard

        assert result.toe_deg_per_side > 0
        assert swing_inboard_mm > 15.0, (
            f"{corner_id}: toe reads {result.toe_deg_per_side:+.2f} deg but the "
            f"front of the wheel is {swing_inboard_mm:+.1f} mm inboard. Positive "
            f"toe must be toe-IN."
        )

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_negative_toe_points_the_wheel_front_outboard(self, corner_id: str) -> None:
        corner = self._corner(corner_id, -5.0)
        result = DWSolver(corner).solve()
        assert result.converged

        front = _wheel_front(corner, result, self.RADIUS)
        inboard = -1.0 if corner_id in ("FL", "RL") else 1.0
        swing_inboard_mm = (front[1] - result.wheel_center.y_mm) * inboard

        assert result.toe_deg_per_side < 0
        assert swing_inboard_mm < -15.0

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_toe_magnitude_equals_the_wheel_plane_yaw(self, corner_id: str) -> None:
        corner = self._corner(corner_id, +5.0)
        result = DWSolver(corner).solve()

        front = _wheel_front(corner, result, self.RADIUS)
        wc = np.array([
            result.wheel_center.x_mm, result.wheel_center.y_mm, result.wheel_center.z_mm,
        ])
        swing = front - wc
        yaw_deg = math.degrees(math.atan2(abs(swing[1]), swing[0]))
        assert yaw_deg == pytest.approx(abs(result.toe_deg_per_side), abs=0.05)

    def test_both_sides_report_the_same_sign_for_the_same_physical_toe(self) -> None:
        """Toe-in on both wheels must read positive on both, not +/-.

        This is the per-side convention: a symmetric toed-in axle has
        toe_deg_per_side > 0 on BOTH corners, summing to a positive total toe.
        """
        left = DWSolver(self._corner("FL", +5.0)).solve()
        right = DWSolver(self._corner("FR", +5.0)).solve()
        assert left.toe_deg_per_side > 0
        assert right.toe_deg_per_side > 0
        assert left.toe_deg_per_side == pytest.approx(right.toe_deg_per_side, abs=1e-9)


class TestBumpSteerReversesWithTieRodPosition:
    """A rear-steer and a front-steer layout must toe opposite ways in bump.

    The two corners differ ONLY in the sign of the tie rod's X, so nothing but
    the fore/aft placement can explain a sign change. This catches a bump-steer
    or toe path that has picked up a fixed sign.
    """

    @staticmethod
    def _bump_steer(corner: Corner) -> float:
        solver = DWSolver(corner)
        up = solver.solve(wheel_travel_mm=+15.0)
        dn = solver.solve(wheel_travel_mm=-15.0)
        assert up.converged and dn.converged
        return (up.toe_deg_per_side - dn.toe_deg_per_side) / 30.0

    def _corner(self, corner_id: str, tie_x: float) -> Corner:
        # Tie rod deliberately mismatched in height with the wishbones so the
        # geometry actually produces bump steer to measure.
        return _build(
            corner_id,
            lca_in=(200.0, 120.0), lca_out=(580.0, 120.0),
            uca_in=(220.0, 340.0), uca_out=(540.0, 340.0),
            wc=(600.0, 250.0), radius=250.0,
            tie_in=(tie_x, 210.0, 200.0), tie_out=(tie_x, 555.0, 185.0),
        )

    @pytest.mark.parametrize("corner_id", ["FL", "FR"])
    def test_tie_rod_fore_and_aft_give_opposite_bump_steer(self, corner_id: str) -> None:
        behind = self._bump_steer(self._corner(corner_id, -70.0))
        ahead = self._bump_steer(self._corner(corner_id, +70.0))

        assert abs(behind) > 1e-4, "geometry produces no bump steer to test"
        assert abs(ahead) > 1e-4
        assert behind * ahead < 0.0, (
            f"{corner_id}: moving the tie rod across the axle line must REVERSE "
            f"bump steer. behind={behind:+.5f}, ahead={ahead:+.5f} deg/mm"
        )

    def test_both_sides_bump_steer_the_same_way(self) -> None:
        """Toe is per-side and signed the same both sides, so a symmetric axle
        in parallel bump must toe both wheels in, or both out -- never split."""
        left = self._bump_steer(self._corner("FL", -70.0))
        right = self._bump_steer(self._corner("FR", -70.0))
        assert left == pytest.approx(right, rel=1e-6)
