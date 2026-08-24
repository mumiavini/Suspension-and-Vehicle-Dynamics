"""Tests for vdcore.analysis.anti -- SVIC and anti-dive/squat construction.

Coordinate system: ISO 8855 -- X+ forward, Y+ LEFT, Z+ up.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vdcore.analysis.anti import (
    AntiResult,
    AntiSweepResult,
    SVICResult,
    _line_intersect_xz,
    _pivot_axis_xz,
    anti_percent,
    anti_sweep,
    side_view_instant_centre,
)
from vdcore.geometry.solver import DWSolver
from vdcore.models.hardpoint import Corner, Hardpoint, TirePackage


def _hp(name: str, x: float, y: float, z: float) -> Hardpoint:
    return Hardpoint(name=name, x_mm=x, y_mm=y, z_mm=z, source="cad", tol_mm=0.5)


def _tire(r: float = 228.0) -> TirePackage:
    return TirePackage(loaded_radius_mm=r, source="cad", tol_mm=1.0)


def _flat_fl(
    uca_if_z: float = 308.58,
    uca_ir_z: float = 308.58,
    lca_if_z: float = 117.383,
    lca_ir_z: float = 117.383,
) -> Corner:
    """FL corner based on 2027 PUCPR geometry, with parameterised pivot-axis Z.

    Default: horizontal pivot axes (zero anti-dive) — matches the 2027 design.
    Setting uca_ir_z != uca_if_z tilts the UCA pivot axis in side view,
    introducing anti-dive.

    The inboard pivots share the same Y=175 on both front and rear, so
    _effective_pivot_at_y returns the midpoint. That midpoint's X is 0,
    matching the outboard X, which makes the XZ arm lines vertical — and
    parallel — producing an infinite SVIC and 0% anti.
    """
    return Corner(
        corner_id="FL",
        uca_inboard_front=_hp("UCA_IF", 120, 175, uca_if_z),
        uca_inboard_rear=_hp("UCA_IR", -120, 175, uca_ir_z),
        uca_outboard=_hp("UCA_O", 0, 537, 385.4),
        lca_inboard_front=_hp("LCA_IF", 130, 175, lca_if_z),
        lca_inboard_rear=_hp("LCA_IR", -130, 175, lca_ir_z),
        lca_outboard=_hp("LCA_O", 0, 582, 130),
        tie_rod_inboard=_hp("TR_I", -120, 175, 117),
        tie_rod_outboard=_hp("TR_O", -60, 545, 135),
        wheel_center=_hp("WC", 0, 620, 245),
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


WHEELBASE = 1550.0
CG_HEIGHT = 280.0


class TestPivotAxisXZ:
    def test_projects_to_xz(self) -> None:
        front = np.array([120.0, 175.0, 300.0])
        rear = np.array([-120.0, 175.0, 300.0])
        x1, z1, x2, z2 = _pivot_axis_xz(front, rear)
        assert x1 == pytest.approx(120.0)
        assert z1 == pytest.approx(300.0)
        assert x2 == pytest.approx(-120.0)
        assert z2 == pytest.approx(300.0)

    def test_tilted_axis(self) -> None:
        front = np.array([120.0, 175.0, 300.0])
        rear = np.array([-120.0, 175.0, 310.0])
        x1, z1, x2, z2 = _pivot_axis_xz(front, rear)
        assert z1 == pytest.approx(300.0)
        assert z2 == pytest.approx(310.0)

    def test_y_is_ignored(self) -> None:
        front = np.array([50.0, 100.0, 200.0])
        rear = np.array([-50.0, 300.0, 200.0])
        x1, z1, x2, z2 = _pivot_axis_xz(front, rear)
        assert x1 == pytest.approx(50.0)
        assert x2 == pytest.approx(-50.0)
        assert z1 == pytest.approx(200.0)
        assert z2 == pytest.approx(200.0)


class TestLineIntersectXZ:
    def test_known_intersection(self) -> None:
        x, z, ok = _line_intersect_xz(0, 0, 10, 10, 0, 10, 10, 0)
        assert ok
        assert x == pytest.approx(5.0)
        assert z == pytest.approx(5.0)

    def test_parallel_lines(self) -> None:
        x, z, ok = _line_intersect_xz(0, 0, 10, 0, 0, 5, 10, 5)
        assert not ok

    def test_coincident_lines(self) -> None:
        _, _, ok = _line_intersect_xz(0, 0, 10, 10, 5, 5, 15, 15)
        assert not ok


class TestSideViewInstantCentre:
    def test_horizontal_axes_give_infinite_svic(self) -> None:
        fl = _flat_fl()
        r = DWSolver(fl).solve()
        assert r.converged
        svic = side_view_instant_centre(fl, ubj=r.ubj, lbj=r.lbj,
                                        contact_patch=r.contact_patch)
        assert not svic.is_finite
        assert svic.svsa_mm == float("inf")

    def test_tilted_uca_gives_finite_svic(self) -> None:
        fl = _flat_fl(uca_ir_z=310.0)
        r = DWSolver(fl).solve()
        assert r.converged
        svic = side_view_instant_centre(fl, ubj=r.ubj, lbj=r.lbj,
                                        contact_patch=r.contact_patch)
        assert svic.is_finite
        assert math.isfinite(svic.svic_x_mm)
        assert math.isfinite(svic.svic_z_mm)

    def test_fr_mirror_gives_same_svic_z(self) -> None:
        fl = _flat_fl(uca_ir_z=310.0)
        fr = _mirror_to_fr(fl)
        rl = DWSolver(fl).solve()
        rr = DWSolver(fr).solve()
        assert rl.converged and rr.converged
        svic_l = side_view_instant_centre(fl, ubj=rl.ubj, lbj=rl.lbj,
                                          contact_patch=rl.contact_patch)
        svic_r = side_view_instant_centre(fr, ubj=rr.ubj, lbj=rr.lbj,
                                          contact_patch=rr.contact_patch)
        assert svic_l.svic_z_mm == pytest.approx(svic_r.svic_z_mm, abs=0.1)

    def test_no_silent_fallback(self) -> None:
        fl = _flat_fl()
        with pytest.raises(TypeError):
            side_view_instant_centre(fl)  # type: ignore[call-arg]


class TestAntiPercent:
    def test_horizontal_axes_give_zero_anti(self) -> None:
        fl = _flat_fl()
        r = DWSolver(fl).solve()
        assert r.converged
        ar = anti_percent(fl, ubj=r.ubj, lbj=r.lbj,
                          contact_patch=r.contact_patch,
                          wheelbase_mm=WHEELBASE, cg_height_mm=CG_HEIGHT,
                          is_front_axle=True)
        assert ar.anti_dive_pct == pytest.approx(0.0)
        assert ar.anti_lift_pct == pytest.approx(0.0)
        assert ar.anti_squat_pct == pytest.approx(0.0)

    def test_tilted_uca_gives_positive_anti_dive(self) -> None:
        fl = _flat_fl(uca_ir_z=310.0)
        r = DWSolver(fl).solve()
        assert r.converged
        ar = anti_percent(fl, ubj=r.ubj, lbj=r.lbj,
                          contact_patch=r.contact_patch,
                          wheelbase_mm=WHEELBASE, cg_height_mm=CG_HEIGHT,
                          is_front_axle=True)
        assert ar.anti_dive_pct > 0.0
        assert ar.svic.is_finite

    def test_front_axle_reports_dive_not_squat(self) -> None:
        fl = _flat_fl(uca_ir_z=310.0)
        r = DWSolver(fl).solve()
        assert r.converged
        ar = anti_percent(fl, ubj=r.ubj, lbj=r.lbj,
                          contact_patch=r.contact_patch,
                          wheelbase_mm=WHEELBASE, cg_height_mm=CG_HEIGHT,
                          is_front_axle=True)
        assert ar.anti_dive_pct > 0.0
        assert ar.anti_squat_pct == 0.0
        assert ar.anti_lift_pct == 0.0

    def test_rear_axle_reports_lift_and_squat_not_dive(self) -> None:
        fl = _flat_fl(uca_ir_z=310.0)
        r = DWSolver(fl).solve()
        assert r.converged
        ar = anti_percent(fl, ubj=r.ubj, lbj=r.lbj,
                          contact_patch=r.contact_patch,
                          wheelbase_mm=WHEELBASE, cg_height_mm=CG_HEIGHT,
                          is_front_axle=False)
        assert ar.anti_dive_pct == 0.0
        assert ar.anti_lift_pct != 0.0
        assert ar.anti_squat_pct != 0.0

    def test_more_tilt_gives_more_anti(self) -> None:
        fl_small = _flat_fl(uca_ir_z=305.0)
        fl_large = _flat_fl(uca_ir_z=315.0)
        r_small = DWSolver(fl_small).solve()
        r_large = DWSolver(fl_large).solve()
        assert r_small.converged and r_large.converged
        ar_small = anti_percent(fl_small, ubj=r_small.ubj, lbj=r_small.lbj,
                                contact_patch=r_small.contact_patch,
                                wheelbase_mm=WHEELBASE, cg_height_mm=CG_HEIGHT,
                                is_front_axle=True)
        ar_large = anti_percent(fl_large, ubj=r_large.ubj, lbj=r_large.lbj,
                                contact_patch=r_large.contact_patch,
                                wheelbase_mm=WHEELBASE, cg_height_mm=CG_HEIGHT,
                                is_front_axle=True)
        assert ar_large.anti_dive_pct > ar_small.anti_dive_pct

    def test_left_right_symmetry(self) -> None:
        fl = _flat_fl(uca_ir_z=310.0)
        fr = _mirror_to_fr(fl)
        rl = DWSolver(fl).solve()
        rr = DWSolver(fr).solve()
        assert rl.converged and rr.converged
        ar_l = anti_percent(fl, ubj=rl.ubj, lbj=rl.lbj,
                            contact_patch=rl.contact_patch,
                            wheelbase_mm=WHEELBASE, cg_height_mm=CG_HEIGHT,
                            is_front_axle=True)
        ar_r = anti_percent(fr, ubj=rr.ubj, lbj=rr.lbj,
                            contact_patch=rr.contact_patch,
                            wheelbase_mm=WHEELBASE, cg_height_mm=CG_HEIGHT,
                            is_front_axle=True)
        assert ar_l.anti_dive_pct == pytest.approx(ar_r.anti_dive_pct, abs=0.1)

    def test_higher_brake_bias_increases_front_anti_dive(self) -> None:
        fl = _flat_fl(uca_ir_z=310.0)
        r = DWSolver(fl).solve()
        assert r.converged
        ar_low = anti_percent(fl, ubj=r.ubj, lbj=r.lbj,
                              contact_patch=r.contact_patch,
                              wheelbase_mm=WHEELBASE, cg_height_mm=CG_HEIGHT,
                              brake_bias_front=0.50, is_front_axle=True)
        ar_high = anti_percent(fl, ubj=r.ubj, lbj=r.lbj,
                               contact_patch=r.contact_patch,
                               wheelbase_mm=WHEELBASE, cg_height_mm=CG_HEIGHT,
                               brake_bias_front=0.70, is_front_axle=True)
        assert ar_high.anti_dive_pct > ar_low.anti_dive_pct

    def test_returns_frozen_model(self) -> None:
        fl = _flat_fl()
        r = DWSolver(fl).solve()
        assert r.converged
        ar = anti_percent(fl, ubj=r.ubj, lbj=r.lbj,
                          contact_patch=r.contact_patch,
                          wheelbase_mm=WHEELBASE, cg_height_mm=CG_HEIGHT,
                          is_front_axle=True)
        with pytest.raises(Exception):
            ar.anti_dive_pct = 999.0  # type: ignore[misc]


class TestAntiSweep:
    def test_sweep_returns_correct_length(self) -> None:
        fl = _flat_fl(uca_ir_z=310.0)
        sweep = anti_sweep(fl, wheelbase_mm=WHEELBASE, cg_height_mm=CG_HEIGHT,
                           is_front_axle=True, steps=21,
                           wheel_travel_min_mm=-10, wheel_travel_max_mm=10)
        assert len(sweep.wheel_travel_mm) == 21
        assert len(sweep.anti_dive_pct) == 21
        assert len(sweep.converged) == 21

    def test_sweep_all_converged(self) -> None:
        fl = _flat_fl(uca_ir_z=310.0)
        sweep = anti_sweep(fl, wheelbase_mm=WHEELBASE, cg_height_mm=CG_HEIGHT,
                           is_front_axle=True, steps=21,
                           wheel_travel_min_mm=-10, wheel_travel_max_mm=10)
        assert all(sweep.converged)

    def test_anti_varies_with_travel(self) -> None:
        fl = _flat_fl(uca_ir_z=310.0)
        sweep = anti_sweep(fl, wheelbase_mm=WHEELBASE, cg_height_mm=CG_HEIGHT,
                           is_front_axle=True, steps=41,
                           wheel_travel_min_mm=-20, wheel_travel_max_mm=20)
        dive_vals = [v for v, c in zip(sweep.anti_dive_pct, sweep.converged) if c]
        assert max(dive_vals) > min(dive_vals)
