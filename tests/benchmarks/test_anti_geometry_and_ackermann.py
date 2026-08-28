"""Known-answer and cross-solver tests for anti-geometry and Ackermann.

WHY THIS FILE EXISTS
    Anti-dive/anti-squat (``sla_geometry.py``) and geometric Ackermann
    (``steering_geometry.py``) live outside ``vdcore`` and were, until
    2026-08-27, covered only by golden-value regression anchors -- which pin
    whatever the code currently does, right or wrong. An audit over 20
    anti-geometry and 25 Ackermann geometries found a real bug in each:

    * The side-view instant centre was built from the two INBOARD PIVOT lines
      instead of lines through the BALL JOINTS. Worth 0.9 % to 50 % of the
      reported anti. Invisible on the shipped design, which runs zero pivot
      rake and so reports 0 % either way.
    * Geometric Ackermann returned ``100 * x_cross / L`` instead of
      ``100 * L / x_cross`` -- the reciprocal. Correct only at exactly 100 %,
      trend inverted everywhere else. The shipped design sat at 101.1 % (true
      98.9 %), close enough to the fixed point to hide it.

    These tests pin the corrected behaviour to physics rather than to stored
    output, so the same class of error cannot come back.

Frames: ``sla_geometry`` / ``steering_geometry`` work in the DESIGN frame
(X+ REARWARD, front axle at x=0, Y+ outboard). ``vdcore`` is ISO 8855
(X+ forward, Y+ left). The 3D cross-check converts between them explicitly.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import sla_geometry as sla  # noqa: E402
import steering_geometry as stg  # noqa: E402
from vdcore.geometry.solver import DWSolver  # noqa: E402
from vdcore.models.hardpoint import Corner, Hardpoint, TirePackage  # noqa: E402

VEH = sla.VEHICLE_2027

# A fixed tie rod for the 3D cross-check. sla_geometry does not synthesise one,
# and it must not vary between cases -- only the pivot rake is under test.
TIE_IN_DESIGN = (-30.0, 270.0, 158.3)
TIE_OUT_DESIGN = (-84.591, 590.294, 178.75)


# --------------------------------------------------------------------------- #
# helpers: design frame -> ISO 8855, and the instant centre of the real linkage
# --------------------------------------------------------------------------- #

def _iso_corner(geo: sla.AxleGeometry, tie_in, tie_out) -> Corner:
    """Lift an sla front-view solution into a complete ISO 8855 vdcore Corner.

    Mirrors ``sla.build_corner``: ``x_iso = -x_rearward``, and the left side
    keeps ``y`` positive.
    """
    inp = geo.inputs

    def hp(name: str, x_rear: float, y: float, z: float) -> Hardpoint:
        return Hardpoint(name=name, x_mm=-x_rear, y_mm=y, z_mm=z,
                         source="design_intent", tol_mm=0.0)

    xo = inp.axle_x_mm
    return Corner(
        corner_id="FL",
        uca_inboard_front=hp("UCA_IF", geo.uca_in_front_x_mm,
                             float(geo.uca_in[0]), geo.uca_in_front_z_mm),
        uca_inboard_rear=hp("UCA_IR", geo.uca_in_rear_x_mm,
                            float(geo.uca_in[0]), geo.uca_in_rear_z_mm),
        uca_outboard=hp("UCA_O", xo, float(geo.ubj[0]), float(geo.ubj[1])),
        lca_inboard_front=hp("LCA_IF", geo.lca_in_front_x_mm,
                             float(geo.lca_in[0]), geo.lca_in_front_z_mm),
        lca_inboard_rear=hp("LCA_IR", geo.lca_in_rear_x_mm,
                            float(geo.lca_in[0]), geo.lca_in_rear_z_mm),
        lca_outboard=hp("LCA_O", xo, float(geo.lbj[0]), float(geo.lbj[1])),
        tie_rod_inboard=hp("TR_I", *tie_in),
        tie_rod_outboard=hp("TR_O", *tie_out),
        wheel_center=hp("WC", xo, float(inp.wheel_centre[0]),
                        float(inp.wheel_centre[1])),
        tire=TirePackage(loaded_radius_mm=inp.loaded_radius_mm,
                         source="design_intent", tol_mm=0.0),
        static_camber_deg=inp.static_camber_deg,
        static_toe_deg_per_side=0.0,
    )


def _svic_from_3d_linkage(corner: Corner, h: float = 2.0) -> np.ndarray | None:
    """Side-view instant centre of the upright, measured from ``DWSolver``.

    Finite-differences two upright points in the CHASSIS frame and intersects
    the perpendiculars to their X-Z velocities -- the definition of the instant
    centre of the projected motion. Returns ``[x_iso, z]``.

    This is the independent reference: ``DWSolver`` solves the full 3D linkage
    and is itself validated against Altair MotionSolve, so it knows nothing
    about sla_geometry's side-view construction.
    """
    solver = DWSolver(corner)
    up, dn = solver.solve(wheel_travel_mm=+h), solver.solve(wheel_travel_mm=-h)
    if not (up.converged and dn.converged):
        return None

    pts, vels = [], []
    for attr in ("ubj", "lbj"):
        a = np.array([getattr(up, attr).x_mm, getattr(up, attr).z_mm + h])
        b = np.array([getattr(dn, attr).x_mm, getattr(dn, attr).z_mm - h])
        pts.append(0.5 * (a + b))
        vels.append((a - b) / (2.0 * h))

    p1, d1 = pts[0], np.array([-vels[0][1], vels[0][0]])
    p2, d2 = pts[1], np.array([-vels[1][1], vels[1][0]])
    den = d1[0] * (-d2[1]) - d1[1] * (-d2[0])
    if abs(den) < 1e-12:
        return None
    rhs = p2 - p1
    t = (rhs[0] * (-d2[1]) - rhs[1] * (-d2[0])) / den
    return p1 + t * d1


# --------------------------------------------------------------------------- #
# 1. anti-geometry vs the real 3D linkage
# --------------------------------------------------------------------------- #

RAKES = [(0, 20), (0, -20), (20, 0), (0, 40), (10, 40),
         (-15, 25), (0, 80), (25, -25), (30, -30), (5, 45)]


class TestSideViewInstantCentreMatchesThe3DLinkage:
    """The construction must reproduce the instant centre of the real linkage.

    This is the test that catches the original bug: building the SVIC from the
    inboard pivot lines put it ~34 % too close to the axle on every geometry,
    consistently enough to look deliberate.
    """

    @pytest.mark.parametrize("dz_lca,dz_uca", RAKES)
    def test_svic_matches_dwsolver(self, dz_lca: float, dz_uca: float) -> None:
        geo = sla.solve_axle(
            replace(sla.FRONT_2027, dz_lca_mm=float(dz_lca), dz_uca_mm=float(dz_uca)),
            VEH,
        )
        assert geo.svic is not None, "these rakes all give a finite SVIC"

        corner = _iso_corner(geo, TIE_IN_DESIGN, TIE_OUT_DESIGN)
        reference = _svic_from_3d_linkage(corner)
        assert reference is not None

        # sla works in the design frame (X+ rearward); flip to compare.
        assert -geo.svic[0] == pytest.approx(reference[0], abs=0.05), (
            f"dz=({dz_lca},{dz_uca}): side-view IC disagrees with the 3D linkage"
        )
        assert geo.svic[1] == pytest.approx(reference[1], abs=0.05)

    def test_horizontal_lower_axis_puts_the_ic_at_the_ball_joint_height(self) -> None:
        """The specific error the fix corrects, stated as physics.

        With ``dz_lca = 0`` the lower arm's axis is parallel to X, so the LOWER
        BALL JOINT moves purely vertically and the instant centre must lie at
        its height. The old construction used the LCA pickup height instead.
        """
        geo = sla.solve_axle(
            replace(sla.FRONT_2027, dz_lca_mm=0.0, dz_uca_mm=20.0), VEH
        )
        lbj_z = float(geo.lbj[1])
        pickup_z = geo.lca_in_front_z_mm

        assert abs(lbj_z - pickup_z) > 5.0, "fixture must distinguish the two"
        assert geo.svic[1] == pytest.approx(lbj_z, abs=1e-6)
        assert geo.svic[1] != pytest.approx(pickup_z, abs=1.0)


# --------------------------------------------------------------------------- #
# 2. anti-geometry invariants
# --------------------------------------------------------------------------- #

class TestAntiGeometryInvariants:

    def test_shipped_design_has_zero_anti(self) -> None:
        """Both 2027 axles run zero pivot rake, so anti is exactly 0 %."""
        assert sla.solve_axle(sla.FRONT_2027, VEH).anti_percent == pytest.approx(0.0)
        assert sla.solve_axle(sla.REAR_2027, VEH).anti_percent == pytest.approx(0.0)

    def test_horizontal_axes_give_exactly_zero(self) -> None:
        geo = sla.solve_axle(
            replace(sla.FRONT_2027, dz_lca_mm=0.0, dz_uca_mm=0.0), VEH
        )
        assert geo.anti_percent == 0.0

    def test_parallel_inclined_axes_are_not_zero_anti(self) -> None:
        """Equal SLOPES put the swing arm at infinity -- still inclined.

        ``tan(theta) -> m`` in the limit, so anti tends to
        ``100 * m * (L/h) * bias``, not zero. Returning 0 % here (the old
        behaviour when ``line_intersection`` found no crossing) is a
        discontinuity: a hair off parallel already gives the full value.

        Note equal ``dz`` is NOT equal slope when the bases differ (260 vs
        240 mm on the 2027 front), which is what makes this easy to get wrong.
        """
        dz_lca = 30.0
        dz_uca = dz_lca * sla.FRONT_2027.uca_base_mm / sla.FRONT_2027.lca_base_mm
        geo = sla.solve_axle(
            replace(sla.FRONT_2027, dz_lca_mm=dz_lca, dz_uca_mm=dz_uca), VEH
        )
        m = dz_lca / sla.FRONT_2027.lca_base_mm
        expected = 100.0 * m * (VEH.wheelbase_mm / VEH.cg_height_mm) * VEH.brake_bias_front

        assert geo.svic is None, "parallel axes have no finite instant centre"
        assert geo.anti_percent == pytest.approx(expected, rel=1e-9)
        assert abs(geo.anti_percent) > 10.0, "and it is a large number, not zero"

    def test_front_anti_scales_with_brake_bias(self) -> None:
        base = replace(sla.FRONT_2027, dz_uca_mm=20.0)
        full = sla.solve_axle(base, VEH).anti_percent
        half = sla.solve_axle(
            base, replace(VEH, brake_bias_front=VEH.brake_bias_front / 2.0)
        ).anti_percent
        assert half == pytest.approx(full / 2.0, rel=1e-12)

    def test_rear_anti_ignores_brake_bias(self) -> None:
        """Anti-squat is a traction case: all drive torque reacts at the rear."""
        base = replace(sla.REAR_2027, dz_uca_mm=20.0)
        a = sla.solve_axle(base, VEH).anti_percent
        b = sla.solve_axle(
            base, replace(VEH, brake_bias_front=VEH.brake_bias_front / 2.0)
        ).anti_percent
        assert a == pytest.approx(b, rel=1e-12)
        assert abs(a) > 1.0

    def test_anti_scales_inversely_with_cg_height(self) -> None:
        base = replace(sla.FRONT_2027, dz_uca_mm=20.0)
        low = sla.solve_axle(base, VEH).anti_percent
        high = sla.solve_axle(
            base, replace(VEH, cg_height_mm=VEH.cg_height_mm * 2.0)
        ).anti_percent
        assert high == pytest.approx(low / 2.0, rel=1e-12)

    @pytest.mark.parametrize("dz_uca", [10.0, 20.0, 40.0, 80.0])
    def test_anti_grows_with_pivot_rake(self, dz_uca: float) -> None:
        smaller = abs(sla.solve_axle(
            replace(sla.FRONT_2027, dz_uca_mm=dz_uca / 2.0), VEH).anti_percent)
        larger = abs(sla.solve_axle(
            replace(sla.FRONT_2027, dz_uca_mm=dz_uca), VEH).anti_percent)
        assert larger > smaller


# --------------------------------------------------------------------------- #
# 3. geometric Ackermann
# --------------------------------------------------------------------------- #

class TestGeometricAckermann:
    """Pinned to the steering-arm angle, which is what the percentage means.

    Every case builds the arm explicitly at a known angle from the kingpin, so
    the expected value follows from geometry rather than from stored output.
    """

    KP_Y = 582.0

    def _ack(self, arm_angle_deg: float, arm_len: float = 120.0) -> float:
        t = math.radians(arm_angle_deg)
        lbj = np.array([0.0, self.KP_Y, 130.0])
        ubj = np.array([0.0, self.KP_Y, 385.4])
        tro = np.array([arm_len * math.cos(t),
                        self.KP_Y - arm_len * math.sin(t), 178.75])
        return stg._geometric_ackermann_pct(tro, lbj, ubj, VEH, sla.FRONT_2027)

    @property
    def ideal_angle_deg(self) -> float:
        """Arm angle whose extension passes through the rear axle centre."""
        return math.degrees(math.atan2(self.KP_Y, VEH.wheelbase_mm))

    def test_arm_pointing_at_the_rear_axle_centre_is_100_percent(self) -> None:
        assert self._ack(self.ideal_angle_deg) == pytest.approx(100.0, abs=1e-9)

    def test_parallel_steer_is_zero_percent(self) -> None:
        """Arms parallel to the centreline never converge: 0 % by convention."""
        assert self._ack(0.0) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("arm_deg", [1.0, 5.0, 10.0, 15.0, 25.0, 30.0, 45.0, 60.0])
    def test_matches_the_ratio_of_arm_angle_tangents(self, arm_deg: float) -> None:
        """%Ack = 100 * tan(theta_actual) / tan(theta_ideal).

        Equivalent to ``100 * L / x_cross``; the previous code used
        ``100 * x_cross / L``, which agrees only at 100 %.
        """
        expected = 100.0 * math.tan(math.radians(arm_deg)) / math.tan(
            math.radians(self.ideal_angle_deg)
        )
        assert self._ack(arm_deg) == pytest.approx(expected, rel=1e-9)

    def test_increases_monotonically_with_arm_angle(self) -> None:
        """The trend the reciprocal bug inverted."""
        angles = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
        values = [self._ack(a) for a in angles]
        # strict=False: pairing consecutive elements, so the operands differ
        # in length by one on purpose.
        assert all(b > a for a, b in zip(values, values[1:], strict=False)), values

    def test_more_arm_angle_than_ideal_is_over_100_percent(self) -> None:
        """Pro-Ackermann. The old formula reported LESS than 100 % here."""
        assert self._ack(self.ideal_angle_deg * 1.5) > 100.0

    def test_less_arm_angle_than_ideal_is_under_100_percent(self) -> None:
        assert self._ack(self.ideal_angle_deg * 0.5) < 100.0

    def test_is_not_the_reciprocal_formula(self) -> None:
        """Directly pins the bug: the two differ except at exactly 100 %."""
        arm_deg = 10.0
        x_cross = self.KP_Y / math.tan(math.radians(arm_deg))
        old = 100.0 * x_cross / VEH.wheelbase_mm
        assert self._ack(arm_deg) == pytest.approx(
            100.0 * VEH.wheelbase_mm / x_cross, rel=1e-9
        )
        assert abs(self._ack(arm_deg) - old) > 100.0


class TestAckermannOnTheShippedDesign:

    @pytest.fixture(scope="class")
    @classmethod
    def steering(cls) -> stg.SteeringGeometry:
        front = sla.solve_axle(sla.FRONT_2027, VEH)
        return stg.synthesize_steering(front, stg.STEERING_2027, VEH)

    def test_geometric_ackermann_value(self, steering: stg.SteeringGeometry) -> None:
        """98.9 %, not the 101.1 % the reciprocal formula reported."""
        assert steering.geometric_ackermann_pct == pytest.approx(98.89, abs=0.05)

    def test_the_app_reports_the_SOLVED_ackermann_not_a_construction(self) -> None:
        """The app must agree with the linkage, not with a plan-view drawing.

        The plan-view construction assumes a vertical kingpin. With 10 deg KPI
        and 5 deg caster its answer swings ~150 points across defensible
        reference heights and none lands near the truth::

            kingpin @ ground        -58.5 %     kingpin @ wheel-centre  92.6 %
            perpendicular foot       52.4 %     kingpin midpoint        98.9 %
            kingpin @ TRO height     57.7 %
            real linkage (zero-steer limit)     71.2 %

        So the app now sweeps the rack on ``DWSolver`` instead, which
        reproduces ``steering_geometry``'s solved value to 3 decimals.
        """
        import sys as _sys

        legacy = REPO / "legacy_app"
        if str(legacy) not in _sys.path:
            _sys.path.insert(0, str(legacy))
        import polars as pl
        from analysis.vdcore_bridge import (
            CornerInputs,
            df_to_vdcore_axles,
            solved_ackermann_pct,
        )

        df = pl.read_csv(REPO / "Geometry Summary" / "hardpoints_2027_merged.csv")
        front, _ = df_to_vdcore_axles(
            df, CornerInputs(static_camber_deg=-1.5, loaded_radius_mm=245.0)
        )
        value = solved_ackermann_pct(front, 1540.0, outer_steer_deg=10.0)

        assert value == pytest.approx(70.05, abs=0.1)
        assert value < 150.0, "the +173 % reciprocal bug must not return"

    def test_solved_ackermann_matches_steering_geometry(
        self, steering: stg.SteeringGeometry
    ) -> None:
        """Two independent implementations of the same solved measure."""
        import sys as _sys

        legacy = REPO / "legacy_app"
        if str(legacy) not in _sys.path:
            _sys.path.insert(0, str(legacy))
        import polars as pl
        from analysis.vdcore_bridge import (
            CornerInputs,
            df_to_vdcore_axles,
            solved_ackermann_pct,
        )

        front_geo = sla.solve_axle(sla.FRONT_2027, VEH)
        kin = stg.SteeringKinematics(front_geo, steering, stg.STEERING_2027, VEH)
        reference = kin._ackermann_at_steer(10.0)

        df = pl.read_csv(REPO / "Geometry Summary" / "hardpoints_2027_merged.csv")
        front, _ = df_to_vdcore_axles(
            df, CornerInputs(static_camber_deg=-1.5, loaded_radius_mm=245.0)
        )
        assert solved_ackermann_pct(front, 1540.0) == pytest.approx(
            reference, abs=0.01
        )

    def test_solved_ackermann_is_zero_for_parallel_steer(self) -> None:
        """Sanity anchor: tie rods that steer both wheels equally give 0 %."""
        import sys as _sys

        legacy = REPO / "legacy_app"
        if str(legacy) not in _sys.path:
            _sys.path.insert(0, str(legacy))
        import polars as pl
        from analysis.vdcore_bridge import (
            CornerInputs,
            df_to_vdcore_axles,
            solved_ackermann_pct,
        )

        df = pl.read_csv(REPO / "Geometry Summary" / "hardpoints_2027_merged.csv")
        # Put both tie-rod outboard points at the same X as the kingpin, so the
        # steering arms point straight inboard: no fore/aft arm, no Ackermann.
        rows = df.to_dicts()
        by = {(r["corner"], r["point"]): r for r in rows}
        for corner in ("FL", "FR"):
            by[(corner, "TIE_ROD_OUT")]["x_mm"] = by[(corner, "LCA_OUT")]["x_mm"]
            by[(corner, "TIE_ROD_IN")]["x_mm"] = by[(corner, "LCA_OUT")]["x_mm"]
        flat = pl.DataFrame(rows)

        front, _ = df_to_vdcore_axles(
            flat, CornerInputs(static_camber_deg=0.0, loaded_radius_mm=245.0)
        )
        value = solved_ackermann_pct(front, 1540.0, outer_steer_deg=5.0)
        assert math.isnan(value) or abs(value) < 12.0, (
            f"parallel steering arms should give near-zero Ackermann, got {value}"
        )

    def test_legacy_app_anti_features_are_zero_on_the_shipped_geometry(self) -> None:
        """The app's anti-dive/anti-squat used to pin at the +200 % clamp.

        Both 2027 axles run zero pivot rake, so the correct answer is 0 %.
        ``SuspensionCorner.anti_dive_percent`` built its side-view IC from
        ``effective_inboard -> outboard``, which is degenerate in side view for
        a fore-aft pivot axis; the resulting noise saturated the old clamp.
        """
        import sys as _sys

        legacy = REPO / "legacy_app"
        if str(legacy) not in _sys.path:
            _sys.path.insert(0, str(legacy))
        import polars as pl
        from analysis.io_hardpoints import build_vehicle_from_dataframe

        df = pl.read_csv(REPO / "Geometry Summary" / "hardpoints_2027_merged.csv")
        veh, _ = build_vehicle_from_dataframe(df)

        front = veh.front_left.anti_dive_percent(
            brake_bias=0.6, wheelbase_mm=1540.0, cg_height_mm=320.0)
        rear = veh.rear_left.anti_dive_percent(
            brake_bias=1.0, wheelbase_mm=1540.0, cg_height_mm=320.0)

        assert front == pytest.approx(0.0, abs=1e-9)
        assert rear == pytest.approx(0.0, abs=1e-9)
        # The clamp is gone: nothing should be able to return exactly +/-200.
        assert abs(front) != 200.0 and abs(rear) != 200.0

    def test_geometric_reads_higher_than_the_solved_linkage(
        self, steering: stg.SteeringGeometry
    ) -> None:
        """The construction is a small-angle idealisation.

        ``_ackermann_at_steer`` solves the real linkage at 10 deg of outer
        steer and compares actual toe-out on turns with the ideal, so it is the
        rigorous number and reads lower. Both must be positive and ordered --
        a geometric value BELOW the solved one would mean the construction had
        inverted again.
        """
        front = sla.solve_axle(sla.FRONT_2027, VEH)
        kin = stg.SteeringKinematics(front, steering, stg.STEERING_2027, VEH)
        at_steer = kin._ackermann_at_steer(10.0)

        assert 0.0 < at_steer < steering.geometric_ackermann_pct


class TestAppKPIsUseTheValidatedSolver:
    """Guards against KPIs silently drifting back to ``KinematicSolver3D``.

    The legacy strut-to-midpoint solver was still feeding several app surfaces
    after the tab merge. Each test here pins one of them to the real linkage.
    """

    @staticmethod
    def _front_axle():
        import sys as _sys

        legacy = REPO / "legacy_app"
        if str(legacy) not in _sys.path:
            _sys.path.insert(0, str(legacy))
        import polars as pl
        from analysis.vdcore_bridge import CornerInputs, df_to_vdcore_axles

        df = pl.read_csv(REPO / "Geometry Summary" / "hardpoints_2027_merged.csv")
        front, _ = df_to_vdcore_axles(
            df, CornerInputs(static_camber_deg=-1.5, loaded_radius_mm=245.0)
        )
        return front

    def test_steer_ratio_uses_dwsolver_not_the_legacy_solver(self) -> None:
        """Legacy read 1.379 mm/deg against a true 1.251 — 10.2 % high.

        It mis-places the outboard ball joints, so it swings the wrong steering
        arm length. A ratio computed from it came out 4.97:1 instead of 4.51:1.
        """
        from analysis.vdcore_bridge import rack_mm_per_wheel_deg

        value = rack_mm_per_wheel_deg(self._front_axle().left)
        assert value == pytest.approx(1.2513, abs=0.01)
        assert value < 1.32, "this is the legacy value creeping back in"

    def test_legacy_corner_adapter_matches_the_dataframe_path(self) -> None:
        """The Comparison tab converts legacy objects; it must not drift."""
        import sys as _sys

        legacy = REPO / "legacy_app"
        if str(legacy) not in _sys.path:
            _sys.path.insert(0, str(legacy))
        import polars as pl
        from analysis.io_hardpoints import build_vehicle_from_dataframe
        from analysis.vdcore_bridge import (
            CornerInputs,
            df_to_vdcore_corner,
            legacy_corner_to_vdcore,
        )

        df = pl.read_csv(REPO / "Geometry Summary" / "hardpoints_2027_merged.csv")
        vehicle, tie_rods = build_vehicle_from_dataframe(df)
        inputs = CornerInputs(static_camber_deg=-1.5, loaded_radius_mm=245.0)

        via_objects = legacy_corner_to_vdcore(
            vehicle.front_left, tie_rods["FL"], "FL", inputs)
        via_frame = df_to_vdcore_corner(df, "FL", inputs)

        for field in ("uca_inboard_front", "uca_inboard_rear", "uca_outboard",
                      "lca_inboard_front", "lca_inboard_rear", "lca_outboard",
                      "tie_rod_inboard", "tie_rod_outboard", "wheel_center"):
            a, b = getattr(via_objects, field), getattr(via_frame, field)
            assert (a.x_mm, a.y_mm, a.z_mm) == (b.x_mm, b.y_mm, b.z_mm), field

    def test_vdcore_sweep_camber_gain_has_the_right_sign(self) -> None:
        """The legacy heave sweep returned +0.0388; the truth is -0.0384.

        Every chart and every A/B comparison built on it was therefore ranking
        geometries backwards on camber gain.
        """
        import numpy as np
        from analysis.vdcore_bridge import vdcore_sweep

        sweep = vdcore_sweep(self._front_axle().left, "Heave", (-25.0, 25.0, 1.0))
        assert bool(sweep["converged"].all())
        gain = float(np.polyfit(sweep["heave_mm"], sweep["camber_deg"], 1)[0])
        assert gain < 0.0, "bump must gain NEGATIVE camber on this geometry"
        assert gain == pytest.approx(-0.0384, abs=0.002)
