"""Swept KPIs, kinematic curves, 3D pose and steering pages: audit fixes pinned.

Each test removes one audited symptom and checks a known answer, mostly the
audit's own repro numbers on the default 2027 car (ISO 8855: X forward, Y LEFT,
Z up; positive roll lifts the LEFT side, i.e. a left turn; toe per side,
toe-in positive; rate and roll Z values chassis-referenced).
"""

from __future__ import annotations

import math
import os
from dataclasses import replace

import numpy as np
import pytest

import sla_geometry as sla
import steering_geometry as stg
from apps.vdstudio.engine import curves as C
from apps.vdstudio.engine import geometry as G
from apps.vdstudio.engine import kpis as K
from apps.vdstudio.engine.pose import PoseSolver, inclination_deg
from apps.vdstudio.engine.project import Project, with_path
from vdcore.analysis.axle import axle_roll
from vdcore.geometry.solver import DWSolver


@pytest.fixture(scope="module")
def geo() -> G.CarGeometry:
    return G.resolve(Project())


@pytest.fixture(scope="module")
def ks(geo) -> K.KpiSet:
    return K.compute_all(geo)


def _hardpoint_project(edit) -> Project:
    p = Project()
    hs = G.freeze_to_hardpoints(p)
    edit(hs.points)
    p.hardpoints = hs
    p.geometry_mode = "hardpoints"
    return p


@pytest.fixture(scope="module")
def asym() -> G.CarGeometry:
    """FR upper ball joint 8 mm higher than the mirrored FL one (hardpoint mode)."""
    def edit(pts) -> None:
        x, y, z = pts["FR"]["UCA_OUT"]
        pts["FR"]["UCA_OUT"] = (x, y, z + 8.0)
    return G.resolve(_hardpoint_project(edit))


@pytest.fixture(scope="module")
def rear_arms() -> G.CarGeometry:
    """Steering arms behind the axle: TRO mirrored about the wheel centre, rack at x = -140."""
    def edit(pts) -> None:
        for cid in ("FL", "FR"):
            x, y, z = pts[cid]["TIE_ROD_OUT"]
            pts[cid]["TIE_ROD_OUT"] = (2 * pts[cid]["WHEEL_CENTER"][0] - x, y, z)
            _, yi, zi = pts[cid]["TIE_ROD_IN"]
            pts[cid]["TIE_ROD_IN"] = (-140.0, yi, zi)
    return G.resolve(_hardpoint_project(edit))


def _vec(p) -> np.ndarray:
    return np.array([p.x_mm, p.y_mm, p.z_mm])


# --------------------------------------------------------------------------- #
# swept-rod-end-silent-skip
# --------------------------------------------------------------------------- #

def test_rod_end_default_value_unchanged(ks) -> None:
    k = ks.get("steering.rod_end")
    assert k.value == pytest.approx(7.825, abs=1e-3) and k.status == "in"


@pytest.mark.parametrize("rack", [50.0, 60.0])
def test_rod_end_fails_when_part_of_the_envelope_does_not_assemble(rack: float) -> None:
    """Was 7.70 / 10.43 deg 'in' from the points that happened to solve (9/81 failed)."""
    g = G.resolve(with_path(Project(), "steering.max_rack_travel_mm", rack))
    k = K.compute_dynamic(g, ["rod_end"]).get("steering.rod_end")
    assert k.status == "fail" and not math.isfinite(k.value)
    assert "9/81 rod-end envelope points did not converge" in k.error
    with pytest.raises(RuntimeError, match="9/81"):
        K.worst_rod_end_swing(g)


def test_steering_geometry_rod_end_raises_too() -> None:
    front = sla.solve_axle(sla.FRONT_2027, sla.VEHICLE_2027)
    steer = replace(stg.STEERING_2027, max_rack_travel_mm=60.0)
    sg = stg.synthesize_steering(front, steer, sla.VEHICLE_2027)
    kin = stg.SteeringKinematics(front, sg, steer, sla.VEHICLE_2027)
    # its solve_pair needs FL AND FR: the inner FR fails too on the -rack side
    with pytest.raises(sla.KinematicError, match="18/81 rod-end envelope points"):
        kin._worst_rod_end_misalignment()


# --------------------------------------------------------------------------- #
# swept-fvic-projection-with-caster
# --------------------------------------------------------------------------- #

def test_rc_comes_from_the_3d_instant_axis(ks) -> None:
    """Verifier's screw-axis RC, now vdcore's own: front 35.373 mm, -0.1999 mm/mm
    (the old projected construction gave 35.500 / -0.3106)."""
    assert ks.value("front.rc_static") == pytest.approx(35.373, abs=0.005)
    assert ks.value("front.rc_migration") == pytest.approx(-0.1999, abs=5e-4)
    assert ks.get("front.rc_static_3d") is None          # no duplicate rows any more


def test_rc_without_caster_is_the_planar_answer(ks) -> None:
    """Rear (ball joints at the same X): the 3D and planar constructions agree."""
    assert ks.value("rear.rc_static") == pytest.approx(55.000, abs=1e-3)
    assert ks.value("rear.rc_migration") == pytest.approx(-0.3380, abs=5e-4)


def test_rc_tracks_caster() -> None:
    """Verifier: at 8 deg caster screw RC 38.08 mm / -0.1340 mm/mm (projection: -0.3121)."""
    g = G.resolve(with_path(Project(), "steering.caster_deg", 8.0))
    d = K.compute_all(g, ["rates"])
    assert d.value("front.rc_static") == pytest.approx(38.08, abs=0.01)
    assert d.value("front.rc_migration") == pytest.approx(-0.1340, abs=5e-4)


# --------------------------------------------------------------------------- #
# swept-roll-direction
# --------------------------------------------------------------------------- #

def test_roll_kpis_describe_the_left_turn_like_the_pose(geo, ks) -> None:
    """+1.5 deg lifts the LEFT side: outer = RIGHT wheel; RC lateral sign matches the 3D pose."""
    pose = PoseSolver(geo).solve(roll_deg=1.5)
    # front 55.76: RC from the 3D instant axis (the projected FVIC gave 86.90); rear unchanged
    for axle, rc, lat in (("front", "FR", 55.76), ("rear", "RR", 56.34)):
        assert ks.value(f"{axle}.roll_rc_lateral") == pytest.approx(lat, abs=0.01)  # was -lat
        world_y = pose.to_world(pose.rc_chassis[axle])[1]
        assert world_y > 0 and world_y == pytest.approx(lat, abs=0.1)
        s = axle_roll(geo.axle(axle), -1.5)
        assert ks.value(f"{axle}.roll_outer_camber") == pytest.approx(s.inner_camber_deg)
        assert ks.value(f"{axle}.roll_inner_camber") == pytest.approx(s.outer_camber_deg)
        assert ks.value(f"{axle}.roll_travel") == pytest.approx(-s.wheel_travel_mm)
        assert ks.value(f"{axle}.roll_travel") > 0
        road_outer = pose.corners[rc].camber_deg + 1.5
        assert ks.value(f"{axle}.roll_outer_camber") == pytest.approx(road_outer, abs=0.01)
        assert ks.get(f"{axle}.roll_outer_camber_right_turn") is None   # symmetric car


def test_asymmetric_axle_reports_both_turns(asym) -> None:
    """Hardpoint FR UCA_OUT +8 mm: the old KPI was the right turn only."""
    d = K.compute_dynamic(asym, ["roll", "roll_camber"])
    assert d.value("front.roll_outer_camber") == pytest.approx(-0.7028, abs=1e-3)
    assert d.value("front.roll_inner_camber") == pytest.approx(-2.3924, abs=1e-3)
    assert d.value("front.roll_rc_lateral") == pytest.approx(14.97, abs=0.02)  # 3D (proj. 39.27)
    pose = PoseSolver(asym).solve(roll_deg=1.5)
    assert pose.corners["FR"].camber_deg + 1.5 == pytest.approx(-0.7031, abs=1e-3)
    assert pose.to_world(pose.rc_chassis["front"])[1] == pytest.approx(14.97, abs=0.05)
    assert d.value("front.roll_outer_camber_right_turn") == pytest.approx(-0.6475, abs=1e-3)
    assert d.value("front.roll_inner_camber_right_turn") == pytest.approx(-2.3362, abs=1e-3)
    # 3D instant-axis RC (the projected FVIC gave -122.35)
    assert d.value("front.roll_rc_lateral_right_turn") == pytest.approx(-89.99, abs=0.02)
    assert d.value("front.roll_camber_right_turn") == pytest.approx(-0.4186, abs=1e-3)
    assert d.value("front.roll_camber") != pytest.approx(d.value("front.roll_camber_right_turn"),
                                                         abs=1e-3)
    assert d.get("rear.roll_outer_camber_right_turn") is None      # rear is still symmetric


def test_roll_curves_use_the_same_mapping(geo) -> None:
    r = C.roll_curves(geo, 2.0, 3)
    s = axle_roll(geo.axle("front"), -2.0)
    assert r.axles["front"]["outer_camber"][-1] == pytest.approx(s.inner_camber_deg)
    assert r.axles["front"]["rc_lateral"][-1] == pytest.approx(s.rc_lateral_mm)
    assert r.axles["front"]["rc_lateral"][-1] > 0
    assert r.axles["front"]["travel"][-1] == pytest.approx(-s.wheel_travel_mm)


def test_roll_camber_is_the_outer_right_wheel_slope(geo) -> None:
    """Right wheel, chassis-referenced, d(camber)/d(ISO roll): -0.4186 (verifier)."""
    ax = geo.axle("front")
    hi, lo = axle_roll(ax, -2.0), axle_roll(ax, +2.0)          # ISO +2 and -2 deg
    ref = ((hi.inner_camber_deg + hi.roll_deg) - (lo.inner_camber_deg + lo.roll_deg)) / 4.0
    assert K.roll_camber_deg_per_deg(ax) == pytest.approx(ref)
    assert K.roll_camber_deg_per_deg(ax) == pytest.approx(-0.4186, abs=1e-3)


# --------------------------------------------------------------------------- #
# swept-roll-camber-note
# --------------------------------------------------------------------------- #

def test_roll_camber_note_reads_the_chassis_scale_and_recovery_is_reported(ks) -> None:
    k = ks.get("front.roll_camber")
    assert "0 = wheel rolls with the body" in k.note and "-1 = stays upright" in k.note
    assert "1.0 = wheel follows the body" not in k.note
    assert ks.value("front.camber_recovery") == pytest.approx(41.86, abs=0.01)
    assert ks.value("front.camber_recovery") == pytest.approx(-100 * k.value)
    assert ks.get("front.camber_recovery").unit == "%"


# --------------------------------------------------------------------------- #
# swept-bump-steer-range
# --------------------------------------------------------------------------- #

def test_bump_steer_sweeps_the_real_droop() -> None:
    """Droop 40 / bump 25: peak was 0.1598 (toe at -25) 'in'; the real peak is toe(-40)."""
    g = G.resolve(with_path(Project(), "front.travel_droop_mm", 40.0))
    d = K.compute_dynamic(g, ["bump_steer"])
    assert d.value("front.bump_steer_peak") == pytest.approx(0.4116, abs=1e-3)
    assert d.get("front.bump_steer_peak").status == "out"
    assert d.value("front.toe_full_droop") == pytest.approx(0.4116, abs=1e-3)
    assert "−40" in d.get("front.toe_full_droop").label
    # the linear rate stays symmetric: no quadratic leak (-0.00387 over [-40, 25])
    assert d.value("front.bump_steer") == pytest.approx(0.0, abs=1e-4)
    assert "±25 mm" in d.get("front.bump_steer").note


def test_bump_steer_full_droop_is_taken_at_the_droop_limit() -> None:
    """Droop 15 / bump 30: 'full droop' was toe(-30) = 0.2306, outside the travel."""
    p = with_path(with_path(Project(), "front.travel_droop_mm", 15.0), "front.travel_bump_mm",
                  30.0)
    d = K.compute_dynamic(G.resolve(p), ["bump_steer"])
    assert d.value("front.toe_full_droop") == pytest.approx(0.0574, abs=1e-3)


def test_bump_steer_unsolved_point_fails() -> None:
    g = G.resolve(with_path(Project(), "front.travel_bump_mm", 400.0))
    k = K.compute_dynamic(g, ["bump_steer"]).get("front.bump_steer_peak")
    assert k.status == "fail" and "toe-sweep points did not converge" in k.error


def test_bump_steer_default_unchanged(geo, ks) -> None:
    from vdcore.analysis.toe import bump_steer
    bs = bump_steer(geo.corner("FL"), wheel_travel_range_mm=25.0)
    assert ks.value("front.bump_steer") == pytest.approx(bs.linear_deg_per_mm_per_side)
    assert ks.value("front.bump_steer_peak") == pytest.approx(0.1598, abs=1e-4)


# --------------------------------------------------------------------------- #
# swept-camber-in-steer
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cid, expected", [("FL", 3.917), ("FR", -2.625)])
def test_inclination_is_the_solver_spin_axis_angle(geo, cid: str, expected: float) -> None:
    """At 38 mm rack DW camber reads +5.082 (FL) / -2.947 (FR): a Y-Z projection."""
    solver = DWSolver(geo.corner(cid))
    r = solver.solve(rack_mm=38.0)
    spin = solver._reconstruct_spin_axis(_vec(r.ubj), _vec(r.lbj), _vec(r.tro))
    assert inclination_deg(r) == pytest.approx(-math.degrees(math.asin(spin[2])), abs=1e-9)
    assert inclination_deg(r) == pytest.approx(expected, abs=1e-3)
    s0 = solver.solve()
    assert inclination_deg(s0) == pytest.approx(s0.camber_deg, abs=1e-9)   # zero toe


def test_steer_curves_and_pose_report_inclination(geo) -> None:
    s = C.steer_curves(geo, 21)
    assert s.camber_fl[-1] == pytest.approx(3.917, abs=1e-3)       # was 5.082
    pose = PoseSolver(geo).solve(rack_mm=38.0)
    assert pose.corners["FL"].inclination_deg == pytest.approx(3.917, abs=1e-3)


# --------------------------------------------------------------------------- #
# swept-ackermann-reference / swept-ackermann-unreachable-msg
# --------------------------------------------------------------------------- #

def test_ackermann_states_its_reference(geo, ks) -> None:
    a = K.solved_ackermann_pct(geo.axle("front"), geo.wheelbase_mm, 10.0, 38.0)
    assert a.track_mm == pytest.approx(1240.0, abs=0.01)
    assert a.rack_mm == pytest.approx(13.3638, abs=1e-3)
    assert a.inner_deg == pytest.approx(11.1295, abs=1e-3)
    assert a.inner_ideal_deg == pytest.approx(11.613, abs=1e-3)
    assert ks.value("steering.ackermann") == pytest.approx(70.03, abs=0.01)
    note = ks.get("steering.ackermann").note
    assert "cot δo − cot δi = T/L" in note
    assert "T = static contact-patch track 1240 mm" in note
    assert "L = static wheelbase 1540 mm" in note
    assert ks.value("steering.ackermann_inner_error") == pytest.approx(11.1295 - 11.613, abs=1e-3)


def test_unreachable_ackermann_angle_says_why() -> None:
    g = G.resolve(with_path(Project(), "steering.ackermann_at_steer_deg", 35.0))
    k = K.compute_dynamic(g, ["ackermann"]).get("steering.ackermann")
    assert k.status == "fail"
    assert k.error == "outer 35° not reachable within 38 mm rack (max outer 27.06°)"
    assert "nan" not in k.note


# --------------------------------------------------------------------------- #
# swept-effort-on-centre
# --------------------------------------------------------------------------- #

def test_parking_torque_over_the_stroke(geo, ks) -> None:
    """On-centre 7.96 N·m 'in'; the local rates climb toward lock (verifier: 2.139 at 35 mm)."""
    fl, fr = DWSolver(geo.corner("FL")), DWSolver(geo.corner("FR"))
    assert K.steer_rate_sum(fl, fr, 0.0) == pytest.approx(
        2.0 / ks.value("steering.rack_per_deg"), rel=1e-3)
    assert K.steer_rate_sum(fl, fr, 35.0) == pytest.approx(2.1394, abs=2e-3)
    on = ks.get("steering.effort_sw_torque")
    assert on.value == pytest.approx(7.961, abs=1e-3) and on.status == "in"
    assert "on-centre" in on.label
    mx = ks.get("steering.effort_sw_torque_max")
    assert mx.value == pytest.approx(11.25, abs=0.01) and mx.status == "out"
    assert mx.band == on.band
    m_kp = ks.value("steering.effort_kp_moment") * 1000.0
    r = 0.95 * geo.steering_inputs.max_rack_travel_mm
    ref = m_kp * K.steer_rate_sum(fl, fr, r) * math.pi / 180 * 16.0 / 1000.0
    assert mx.value == pytest.approx(ref, rel=1e-9)
    for key in ("steering.effort_sw_torque_max", "steering.effort_kp_moment"):
        assert "no patch-twist (bore) torque" in ks.get(key).note


# --------------------------------------------------------------------------- #
# swept-ratio-on-centre
# --------------------------------------------------------------------------- #

def test_ratio_on_centre_overall_and_along_the_stroke(geo, ks) -> None:
    assert "on-centre" in ks.get("steering.ratio").label
    assert ks.value("steering.ratio_overall") == pytest.approx(4.079, abs=1e-3)
    assert ks.value("steering.ratio_overall") == pytest.approx(
        ks.value("steering.sw_lock") / (0.5 * (ks.value("steering.max_steer_inner")
                                               + ks.value("steering.max_steer_outer"))))
    s = C.steer_curves(geo, 41)
    mid = 20
    assert s.ratio_fl[mid] == pytest.approx(ks.value("steering.ratio"), rel=2e-3)
    assert s.ratio_fl[-1] < 2.5 < 5.0 < s.ratio_fr[-1]    # inner quicker, outer slower


# --------------------------------------------------------------------------- #
# swept-rack-sign-label
# --------------------------------------------------------------------------- #

def test_turn_direction_comes_from_a_rack_probe(geo, rear_arms) -> None:
    assert C.positive_rack_turns_left(geo) is True
    assert C.positive_rack_turns_left(rear_arms) is False
    assert C.steer_curves(rear_arms, 5).left_turn_for_positive_rack is False
    # FL toes IN at +10 mm with the arms behind the axle: both wheels yaw right
    fl = DWSolver(rear_arms.corner("FL"))
    assert fl.solve(rack_mm=10.0).toe_deg_per_side - fl.solve().toe_deg_per_side == \
        pytest.approx(6.110, abs=1e-3)


# --------------------------------------------------------------------------- #
# swept-pose-travel-limits
# --------------------------------------------------------------------------- #

def test_pose_flags_corners_beyond_design_travel(geo) -> None:
    ps = PoseSolver(geo)
    p = ps.solve(heave_mm=-35.0, roll_deg=4.0, pitch_deg=3.0)
    assert p.corners["RL"].travel_mm == pytest.approx(-113.3, abs=0.1)
    assert all(c.beyond_travel for c in p.corners.values())
    assert p.all_converged      # DWSolver still assembles it: hence the flag
    edge = ps.solve(heave_mm=-25.0)
    assert not any(c.beyond_travel for c in edge.corners.values())
    assert all(c.beyond_travel for c in ps.solve(heave_mm=-25.5).corners.values())


# --------------------------------------------------------------------------- #
# UI: labels, slalom pairing, readout marking
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("VDSTUDIO_NO_GL", "1")
    qtw = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("pyqtgraph")
    from apps.vdstudio.ui import theme
    a = qtw.QApplication.instance() or qtw.QApplication([])
    theme.apply(a, theme.DARK)
    return a


@pytest.fixture
def ctrl(qt_app):
    from apps.vdstudio.ui.controller import ProjectController
    c = ProjectController()
    c.recompute()
    yield c
    c.shutdown()


def test_steering_page_labels(ctrl, geo, rear_arms) -> None:
    import pyqtgraph as pg

    from apps.vdstudio.ui.pages.steering import SteeringPage, rack_axis_label
    page = SteeringPage(ctrl)
    page._draw((C.steer_curves(geo, 21), C.heave_curves(geo, -25.0, 25.0, 5)))
    bottom = page.p_rack.getPlotItem().getAxis("bottom").labelText
    assert bottom == "rack [mm] (+ = left turn)"
    lines = [i for i in page.p_rad.getPlotItem().items if isinstance(i, pg.InfiniteLine)]
    assert "the car's OUTERMOST point must stay inside" in lines[0].label.format
    names = [it.name() for it in page.p_rad.getPlotItem().listDataItems()]
    assert names == ["patch-centre path (L / sin δo)"]
    assert page.p_ratio.getPlotItem().listDataItems()            # Ratio vs rack plot
    page._draw((C.steer_curves(rear_arms, 5), C.heave_curves(geo, -25.0, 25.0, 5)))
    assert "right turn" in page.p_rack.getPlotItem().getAxis("bottom").labelText
    assert "right turn" in rack_axis_label(False)


def test_motion_page_turn_direction_and_travel_marking(ctrl, rear_arms) -> None:
    from apps.vdstudio.ui.pages.motion import MotionPage
    page = MotionPage(ctrl)
    page.on_geometry(ctrl.geometry)
    assert page.steer.label.text() == "Steering-wheel angle (+ = left turn)"
    page.heave.set(-35.0)
    page.roll.set(4.0)
    page.pitch.set(3.0)
    page._update()
    col = page.readout.columnCount() - 1
    assert page.readout.horizontalHeaderItem(col).text() == "design travel"
    assert all(page.readout.item(i, col).text().startswith("beyond")
               for i in range(page.readout.rowCount()))
    page.heave.set(0.0)
    page.roll.set(0.0)
    page.pitch.set(0.0)
    page.steer.set(100.0)
    page._update()
    assert page.readout.item(0, col).text() == ""
    assert page.readout.horizontalHeaderItem(2).text().startswith("incl.°")
    fl = page.pose.corners["FL"]
    assert float(page.readout.item(0, 2).text()) == pytest.approx(fl.inclination_deg, abs=1e-3)
    # slalom: + steering pairs with + roll only when + rack is a left turn
    page.isVisible = lambda: True  # type: ignore[method-assign]
    page._anim, page._phase = "slalom", 1.0
    page._tick()
    assert page.steer.value() > 0 and page.roll.value() > 0
    ctrl.geometry = rear_arms
    page.on_geometry(rear_arms)
    assert "right turn" in page.steer.label.text()
    page._anim, page._phase = "slalom", 1.0
    page._tick()
    assert page.steer.value() > 0 and page.roll.value() < 0
    page.shutdown()
