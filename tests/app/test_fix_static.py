"""VD Studio: static geometry and static KPI fixes (audit group "static").

Each test pins one fixed behaviour with the audit's repro numbers: the symptom
is gone and a known answer holds. Frame: ISO 8855 (X+ forward, Y+ LEFT, Z+ up).
"""

from __future__ import annotations

import copy
import math
import os

import numpy as np
import pytest

import sla_geometry as sla
from apps.vdstudio.engine import geometry as G
from apps.vdstudio.engine import kpis as K
from apps.vdstudio.engine import mass
from apps.vdstudio.engine.project import (
    Layout,
    Project,
    project_from_dict,
    project_to_dict,
    with_path,
)
from vdcore.analysis.toe import bump_steer
from vdcore.geometry.solver import DWSolver

L_OVER_H = 1540.0 / 320.0 * 100.0


def _static(p: Project) -> tuple[G.CarGeometry, K.KpiSet]:
    geo = G.resolve(p)
    return geo, K.compute_static(geo)


def _hp_project() -> Project:
    p = Project()
    p.hardpoints = G.freeze_to_hardpoints(p)
    p.geometry_mode = "hardpoints"
    return p


def _shift(p: Project, dz: float = 0.0, wc_dz: float = 0.0) -> Project:
    q = copy.deepcopy(p)
    for pts in q.hardpoints.points.values():
        for n, (x, y, z) in pts.items():
            pts[n] = (x, y, z + dz + (wc_dz if n == "WHEEL_CENTER" else 0.0))
    return q


@pytest.fixture(scope="module")
def default() -> tuple[G.CarGeometry, K.KpiSet]:
    return _static(Project())


# --------------------------------------------------------------------------- #
# static-01 / static-14: brake / drive layout sets the anti reference point
# --------------------------------------------------------------------------- #

def test_layout_defaults_are_the_2027_car_and_json_is_backward_compatible() -> None:
    p = Project()
    assert p.layout == Layout(front_brakes_inboard=False, rear_brakes_inboard=False,
                              rear_drive_inboard=True)
    d = project_to_dict(p)
    assert d["layout"] == {"front_brakes_inboard": False, "rear_brakes_inboard": False,
                           "rear_drive_inboard": True}
    del d["layout"]                       # a project saved before the flags existed
    assert project_from_dict(d).layout == Layout()
    d["layout"] = {"rear_drive_inboard": False}
    assert project_from_dict(d).layout == Layout(rear_drive_inboard=False)
    q = with_path(p, "layout.front_brakes_inboard", True)
    assert project_from_dict(project_to_dict(q)).layout.front_brakes_inboard
    assert not p.layout.front_brakes_inboard


def test_rear_anti_squat_runs_from_the_wheel_centre_with_inboard_drive() -> None:
    """Rear dz_uca +15: the CP line gave +22.53 %; the inboard-drive car has pro-squat."""
    p = with_path(Project(), "rear.dz_uca_mm", 15.0)
    _, ks = _static(p)
    k = ks.get("rear.anti")
    sx, sz = ks.value("rear.svic_x"), ks.value("rear.svic_z")
    assert (sx, sz) == pytest.approx((1236.95, 130.0), abs=0.01)
    # known answer: tan = (z_svic - z_wc) / (x_svic - x_wc), WC at (-1540, 245)
    assert k.value == pytest.approx((sz - 245.0) / (sx + 1540.0) * L_OVER_H, abs=1e-6)
    assert k.value == pytest.approx(-19.93, abs=0.01)
    assert "wheel centre" in k.note and "driveshafts" in k.note
    assert k.status == "out"
    # outboard drive keeps the contact-patch line (the old number)
    _, ko = _static(with_path(p, "layout.rear_drive_inboard", False))
    assert ko.value("rear.anti") == pytest.approx(sz / (sx + 1540.0) * L_OVER_H, abs=1e-6)
    assert ko.value("rear.anti") == pytest.approx(22.53, abs=0.01)
    assert "contact patch" in ko.get("rear.anti").note


def test_front_anti_dive_reference_follows_the_front_brakes() -> None:
    _, ks = _static(with_path(Project(), "layout.front_brakes_inboard", True))
    sx, sz = ks.value("front.svic_x"), ks.value("front.svic_z")
    tan = (sz - 245.0) / (sx - 0.0)
    assert ks.value("front.anti") == pytest.approx(-0.65 * tan * L_OVER_H, abs=1e-6)
    assert ks.value("front.anti") == pytest.approx(-6.62, abs=0.01)
    assert "inboard front brakes" in ks.get("front.anti").note


def test_rear_anti_lift_is_reported_with_its_own_reference(default) -> None:
    _, ks0 = default
    assert ks0.get("rear.anti_lift").ok and ks0.value("rear.anti_lift") == pytest.approx(0.0)
    p = with_path(Project(), "rear.dz_uca_mm", 15.0)
    _, ks = _static(p)
    lift = ks.get("rear.anti_lift")
    assert lift.value == pytest.approx(0.35 * 22.529, abs=0.01)          # 7.89 %, CP
    assert "contact patch" in lift.note and "3D virtual work: 7.66 %" in lift.note
    _, ksi = _static(with_path(p, "layout.rear_brakes_inboard", True))
    assert ksi.value("rear.anti_lift") == pytest.approx(0.35 * -19.930, abs=0.01)
    assert "wheel centre" in ksi.get("rear.anti_lift").note


def test_svic_x_carries_the_iso_frame_note(default) -> None:
    _, ks = default
    k = ks.get("front.svic_x")
    assert k.value == pytest.approx(-5434.31, abs=0.01)
    assert "ISO 8855" in k.note and "behind the front axle" in k.note


# --------------------------------------------------------------------------- #
# static-06: anti by virtual work on DWSolver
# --------------------------------------------------------------------------- #

def test_anti_3d_matches_the_verifier_and_is_step_converged(default) -> None:
    geo, ks = default
    assert ks.value("front.anti_3d") == pytest.approx(7.087, abs=0.002)
    assert ks.get("front.anti_3d").status == "in"
    assert ks.get("front.anti").source == \
        "planar side-view construction (SVIC verified against DWSolver)"
    r01 = K.upright_point_rates(geo, "FL", probe_mm=0.1)
    r001 = K.upright_point_rates(geo, "FL", probe_mm=0.01)
    a01 = K.anti_from_rates(r01, "front", "cp", 0.65, 320.0, 1540.0)
    a001 = K.anti_from_rates(r001, "front", "cp", 0.65, 320.0, 1540.0)
    assert a01 == pytest.approx(a001, abs=1e-4)


def test_upright_rates_reproduce_dwsolvers_own_wheel_centre(default) -> None:
    """Known answer: the Kabsch-fitted WC velocity is the solver's WC velocity."""
    geo, _ = default
    rates = K.upright_point_rates(geo, "FL", probe_mm=0.1)
    s = DWSolver(geo.corner("FL"))
    up, dn = s.solve(wheel_travel_mm=0.1), s.solve(wheel_travel_mm=-0.1)
    vx = (up.wheel_center.x_mm - dn.wheel_center.x_mm) / 0.2
    assert rates["wc"][0] == pytest.approx(vx, abs=1e-6)
    assert rates["wc"][0] == pytest.approx(-0.019833, abs=1e-5)
    assert rates["wc"][2] == pytest.approx(1.0, abs=1e-6)     # chassis frame: WC rises 1:1


def test_anti_3d_rear_cases_from_the_verifier() -> None:
    p = with_path(Project(), "rear.dz_uca_mm", 15.0)
    _, ks = _static(p)
    assert ks.value("rear.anti_3d") == pytest.approx(-18.213, abs=0.01)       # WC
    _, ko = _static(with_path(p, "layout.rear_drive_inboard", False))
    assert ko.value("rear.anti_3d") == pytest.approx(21.895, abs=0.01)        # CP
    assert ko.get("rear.anti_3d").band == sla.CheckLimits().anti_percent


# --------------------------------------------------------------------------- #
# static-02: the ground must be at z = 0
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dz, wc_dz", [(20.0, 0.0), (0.0, 5.0)])
def test_datum_off_the_ground_is_refused(dz: float, wc_dz: float) -> None:
    """Whole car +20 mm gave scrub 11.55 / RC 43.84 with no warning; now it is refused."""
    with pytest.raises(G.GeometryError, match=r"Corner FL: wheel-centre z .*translate"):
        G.resolve(_shift(_hp_project(), dz, wc_dz))


def test_datum_tolerance_and_rim_window_on_the_actual_wheel_centre() -> None:
    p = _shift(_hp_project(), 0.8)                       # inside the 1 mm tyre tolerance
    geo, ks = _static(p)
    r_rim = 13 * 25.4 / 2.0
    assert ks.get("front.ubj_z").band == pytest.approx(
        (245.8 - r_rim + 15.0, 245.8 + r_rim - 15.0))
    q = copy.deepcopy(p)
    q.hardpoints.front.tire_tol_mm = 0.2                 # floor is 0.5 mm
    with pytest.raises(G.GeometryError, match="FL"):
        G.resolve(q)


# --------------------------------------------------------------------------- #
# static-03: one wheelbase in synthesis mode
# --------------------------------------------------------------------------- #

def test_vehicle_wheelbase_moves_the_rear_axle_in_synthesis() -> None:
    p = with_path(Project(), "vehicle.wheelbase_mm", 1600.0)
    geo, ks = _static(p)
    assert geo.wheelbase_mm == pytest.approx(1600.0)
    assert geo.rear_inputs.axle_x_mm == pytest.approx(1600.0)
    assert ks.value("vehicle.wheelbase") == pytest.approx(1600.0)
    assert ks.value("vehicle.front_mass_fraction") == pytest.approx(100 * (1 - 847 / 1600))
    # the app and sla now use the same L (7.775 vs 7.792 is the caster merge only)
    assert ks.value("front.anti") == pytest.approx(geo.design.front.anti_percent, abs=0.05)
    assert G.synthesis_rear_axle_x_mm(p) == 1600.0


def test_stored_rear_axle_station_is_not_read_in_synthesis() -> None:
    geo = G.resolve(with_path(Project(), "rear.axle_x_mm", 1600.0))
    assert geo.wheelbase_mm == pytest.approx(1540.0)


# --------------------------------------------------------------------------- #
# static-04: the rear toe link follows the rear axle station
# --------------------------------------------------------------------------- #

def test_rear_toe_link_moves_with_the_rear_axle() -> None:
    """axle 1540 -> 1600 took bump steer from -0.00015 to -0.00132 deg/mm; now unchanged."""
    g0 = G.resolve(Project())
    g1 = G.resolve(with_path(Project(), "vehicle.wheelbase_mm", 1600.0))
    for g in (g0, g1):
        rl = g.points["RL"]
        assert rl["TIE_ROD_OUT"][0] - rl["LCA_OUT"][0] == pytest.approx(-54.6, abs=1e-6)
    b0 = bump_steer(g0.corner("RL"), wheel_travel_range_mm=25.0).linear_deg_per_mm_per_side
    b1 = bump_steer(g1.corner("RL"), wheel_travel_range_mm=25.0).linear_deg_per_mm_per_side
    assert b0 == pytest.approx(-0.00015, abs=1e-5)
    assert b1 == pytest.approx(b0, abs=1e-6)
    assert g1.points["RR"]["TIE_ROD_IN"][0] == pytest.approx(
        g0.points["RR"]["TIE_ROD_IN"][0] - 60.0)


def test_toe_link_station_round_trips_and_defaults() -> None:
    p = Project()
    assert p.rear_toe_link_axle_x_mm == 1540.0
    d = project_to_dict(p)
    del d["rear_toe_link_axle_x_mm"]
    assert project_from_dict(d).rear_toe_link_axle_x_mm == 1540.0
    # declared at the new station: no shift
    q = with_path(with_path(p, "vehicle.wheelbase_mm", 1600.0), "rear_toe_link_axle_x_mm",
                  1600.0)
    assert G.resolve(q).points["RL"]["TIE_ROD_OUT"][0] == pytest.approx(
        p.rear_toe_link_outboard[0])


# --------------------------------------------------------------------------- #
# static-05: right corner evaluated, asymmetry disclosed
# --------------------------------------------------------------------------- #

def test_asymmetric_car_notes_the_right_corner_and_warns_once() -> None:
    p = _hp_project()
    x, y, z = p.hardpoints.points["FR"]["UCA_IN_REAR"]
    p.hardpoints.points["FR"]["UCA_IN_REAR"] = (x, y, z + 10.0)
    geo, ks = _static(p)
    anti = ks.get("front.anti")
    assert anti.value == pytest.approx(7.483, abs=0.001)      # left value kept on the key
    assert "L 7.48 / R 0.87" in anti.note                     # FR's own construction: 0.866
    assert "L 352.34 / R 350.34" in ks.get("front.uca_rear_leg").note
    assert "/ R " in ks.get("front.svic_x").note
    K.compute_static(geo)                                     # recompute: still one warning
    asym = [w for w in geo.warnings if "asymmetric" in w]
    assert len(asym) == 1 and "LEFT corner" in asym[0]


def test_symmetric_cars_look_unchanged(default) -> None:
    for geo, ks in (default, _static(_hp_project())):
        assert geo.warnings == []
        assert not any("/ R " in k.note for k in ks.items.values())


# --------------------------------------------------------------------------- #
# static-07: horizontal FVSA
# --------------------------------------------------------------------------- #

def test_fvsa_is_horizontal_and_the_slant_value_is_an_info_row(default) -> None:
    geo, ks = default
    assert ks.value("rear.fvsa") == pytest.approx(1400.0, abs=1e-6)  # = the synthesis input
    assert ks.value("rear.fvsa_slant") == pytest.approx(1405.870, abs=1e-3)
    cp_y = geo.p("FL", "CONTACT_PATCH")[1]
    assert ks.value("front.fvsa") == pytest.approx(cp_y - ks.value("front.fvic_y"), abs=1e-9)
    assert ks.value("front.fvsa") == pytest.approx(1489.614, abs=1e-3)  # 3D IC (proj. 1486.298)
    assert ks.get("front.fvsa_slant").band is None
    assert ks.value("rear.camber_gain_fvsa") == pytest.approx(math.degrees(1.0) / 1400.0)


# --------------------------------------------------------------------------- #
# static-08: SVIC above the reference point is undefined, not 0 %
# --------------------------------------------------------------------------- #

def test_svic_above_the_contact_patch_fails_instead_of_reading_zero() -> None:
    p = _hp_project()
    geo0 = G.resolve(p)
    ubj, lbj = geo0.p("FL", "UCA_OUT"), geo0.p("FL", "LCA_OUT")   # LCA axis is horizontal
    m_u = (ubj[2] - lbj[2]) / (ubj[0] - geo0.p("FL", "CONTACT_PATCH")[0])
    for c in ("FL", "FR"):
        f, r = p.hardpoints.points[c]["UCA_IN_FRONT"], p.hardpoints.points[c]["UCA_IN_REAR"]
        mid, dz = 0.5 * (f[2] + r[2]), m_u * (r[0] - f[0])
        p.hardpoints.points[c]["UCA_IN_FRONT"] = (f[0], f[1], mid - dz / 2)
        p.hardpoints.points[c]["UCA_IN_REAR"] = (r[0], r[1], mid + dz / 2)
    _, ks = _static(p)
    k = ks.get("front.anti")
    assert k.status == "fail" and math.isnan(k.value)
    assert k.error == "SVIC directly above the reference point: anti undefined"
    assert ks.value("front.svic_x") == pytest.approx(0.0, abs=1e-6)   # the SVIC itself is fine


def test_sla_side_view_anti_singularity_is_nan() -> None:
    # LCA horizontal through the LBJ at z=130; UCA slope 23 through the UBJ (10, 360)
    # meets it at x = 0 = the axle station: the CP-SVIC line is vertical.
    svic, anti = sla._side_view_anti(sla.FRONT_2027, sla.VEHICLE_2027, True,
                                     (-100.0, 117.0), (100.0, 117.0),
                                     (-1.0, 300.0), (1.0, 346.0),
                                     (10.0, 130.0), (10.0, 360.0))
    assert svic is not None and svic[0] == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(anti)


# --------------------------------------------------------------------------- #
# static-09 / static-12: refused imports name their cause
# --------------------------------------------------------------------------- #

def test_x_reversed_import_is_refused() -> None:
    p = _hp_project()
    for pts in p.hardpoints.points.values():
        for n, (x, y, z) in pts.items():
            pts[n] = (-x + 0.0, y, z)
    with pytest.raises(G.GeometryError, match=r"X looks reversed: check the import frame"):
        G.resolve(p)


def test_wheelbase_warning_quotes_the_cg_rescale_it_did() -> None:
    p = _hp_project()
    for c in ("RL", "RR"):
        p.hardpoints.points[c] = {n: (x - 20.0, y, z)
                                  for n, (x, y, z) in p.hardpoints.points[c].items()}
    geo = G.resolve(p)
    assert geo.vehicle_data.cg_from_front_axle_mm == pytest.approx(847.0 * 1560.0 / 1540.0)
    assert any("scaled from 847.0 to 858.0 mm" in w for w in geo.warnings)


@pytest.mark.parametrize("corner, point, onto, match", [
    ("FL", "UCA_IN_FRONT", "UCA_IN_REAR", "Corner FL: the UCA pivot axis has zero length"),
    ("RR", "LCA_IN_REAR", "LCA_IN_FRONT", "Corner RR: the LCA pivot axis has zero length"),
    ("FL", "UCA_OUT", "LCA_OUT", "Corner FL: UCA_OUT coincides with LCA_OUT"),
])
def test_degenerate_points_are_named(corner: str, point: str, onto: str, match: str) -> None:
    p = _hp_project()
    p.hardpoints.points[corner][point] = p.hardpoints.points[corner][onto]
    with pytest.raises(G.GeometryError, match=match):
        G.resolve(p)


def test_tie_rod_on_the_kingpin_axis_is_named() -> None:
    p = _hp_project()
    u = np.array(p.hardpoints.points["RL"]["UCA_OUT"])
    lo = np.array(p.hardpoints.points["RL"]["LCA_OUT"])
    p.hardpoints.points["RL"]["TIE_ROD_OUT"] = tuple(float(v) for v in lo + 0.4 * (u - lo))
    with pytest.raises(G.GeometryError, match="Corner RL: TIE_ROD_OUT lies on the kingpin"):
        G.resolve(p)


# --------------------------------------------------------------------------- #
# static-10 / static-11: roll budget on the sprung CG
# --------------------------------------------------------------------------- #

def test_roll_budget_uses_the_sprung_cg(default) -> None:
    geo, ks = default
    h_s = (315.0 * 320.0 - 45.0 * 245.0) / 270.0                     # 332.5 mm
    a_s = (315.0 * 847.0 - 22.5 * 1540.0) / 270.0                    # 859.83 mm
    rc_f, rc_r = ks.value("front.rc_static"), ks.value("rear.rc_static")
    h_axis = rc_f + (rc_r - rc_f) * a_s / 1540.0
    assert ks.value("vehicle.roll_axis_at_cg") == pytest.approx(h_axis)
    assert ks.value("vehicle.roll_arm") == pytest.approx(h_s - h_axis)
    assert ks.value("vehicle.roll_arm") == pytest.approx(286.17, abs=0.01)  # front RC 35.373 (3D)
    assert ks.value("vehicle.k_roll_required") == pytest.approx(
        270.0 * K.GRAVITY * (h_s - h_axis) / 1000.0)                 # 757.8 (was 725.1)
    arm = ks.get("vehicle.roll_arm")
    assert arm.band == (0.0, math.inf) and arm.status == "in"
    assert "332.5" in arm.note and mass.UNSPRUNG_ASSUMPTION in arm.note
    whole = sla.solve_vehicle(geo.vehicle_data, rc_f, rc_r, 1200.0)
    assert f"{whole.required_roll_stiffness_Nm_per_deg:.1f}" in \
        ks.get("vehicle.k_roll_required").note                       # 725.1 parity quote


def test_roll_axis_above_the_sprung_cg_fails_the_requirement() -> None:
    p = with_path(with_path(Project(), "front.rc_height_mm", 400.0), "rear.rc_height_mm", 400.0)
    _, ks = _static(p)
    assert ks.get("vehicle.roll_arm").status == "out" and ks.value("vehicle.roll_arm") < 0
    for key in ("vehicle.k_roll_required", "vehicle.k_roll_required_pend",
                "vehicle.chassis_torsion_min", "vehicle.chassis_torsion_target"):
        k = ks.get(key)
        assert k.status == "fail"
        assert k.error == "roll axis at or above the sprung CG: requirement undefined"


# --------------------------------------------------------------------------- #
# static-13: synthesis input next to the achieved value
# --------------------------------------------------------------------------- #

def test_synthesis_inputs_are_shown_next_to_the_achieved_values(default) -> None:
    _, ks = default
    assert "synthesis input 35 mm, Δ +0.37 mm" in ks.get("front.rc_static").note
    assert "synthesis input 1500 mm, Δ -10.39 mm" in ks.get("front.fvsa").note
    assert "synthesis input 1400 mm, Δ +0.00 mm" in ks.get("rear.fvsa").note
    assert "7.50 %, Δ -0.02 %" in ks.get("front.anti").note
    _, kh = _static(_hp_project())
    assert "synthesis input" not in kh.get("front.rc_static").note


# --------------------------------------------------------------------------- #
# static-15: ball joints against the rim, radially
# --------------------------------------------------------------------------- #

def test_ball_joint_radial_distance_from_the_spin_axis(default) -> None:
    geo, ks = default
    assert ks.value("front.ubj_rim_radial") == pytest.approx(142.89, abs=0.01)
    assert ks.value("rear.ubj_rim_radial") == pytest.approx(147.22, abs=0.01)
    assert ks.value("front.lbj_rim_radial") == pytest.approx(114.58, abs=0.01)
    assert ks.get("front.ubj_rim_radial").band == pytest.approx((0.0, 13 * 12.7 - 15.0))
    # known answer: FL spin axis (0, cos g, -sin g), g = -1.5 deg; the patch lies in the
    # wheel plane, and the radial distance follows by projection
    g = math.radians(-1.5)
    a = np.array([0.0, math.cos(g), -math.sin(g)])
    wc, cp = geo.p("FL", "WHEEL_CENTER"), geo.p("FL", "CONTACT_PATCH")
    assert float(np.dot(cp - wc, a)) == pytest.approx(0.0, abs=1e-9)
    v = geo.p("FL", "UCA_OUT") - wc
    assert ks.value("front.ubj_rim_radial") == pytest.approx(
        float(np.linalg.norm(v - np.dot(v, a) * a)), abs=1e-6)
    assert ks.value("front.ubj_z") == pytest.approx(385.4, abs=0.05)   # z rows kept


# --------------------------------------------------------------------------- #
# static-16: static camber / toe are the design inputs echoed
# --------------------------------------------------------------------------- #

def test_static_alignment_rows_name_the_design_input(default) -> None:
    _, ks = default
    src = "design input (static camber / toe per side), carried through DWSolver"
    for key in ("front.camber_L", "front.toe_R", "rear.camber_R", "rear.total_toe"):
        assert ks.get(key).source == src
    assert "design_intent" in ks.get("front.camber_L").note
    assert ks.get("front.caster_L").source == "vdcore DWSolver"
    _, kh = _static(_hp_project())
    assert "AxleAlignment" in kh.get("front.toe_L").note


# --------------------------------------------------------------------------- #
# UI: layout checkboxes and the derived rear axle station
# --------------------------------------------------------------------------- #

def test_geometry_page_layout_flags_and_read_only_rear_station() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("VDSTUDIO_NO_GL", "1")
    qtw = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("pyqtgraph")
    from apps.vdstudio.ui.controller import ProjectController
    from apps.vdstudio.ui.pages.geometry import GeometryPage

    _ = qtw.QApplication.instance() or qtw.QApplication([])
    ctrl = ProjectController()
    try:
        page = GeometryPage(ctrl)
        ctrl.set_value("vehicle.wheelbase_mm", 1600.0)
        page.on_project()
        station = next(ed for f in page.forms for path, (_, ed) in f._editors.items()
                       if path == "rear.axle_x_mm")
        assert station.isReadOnly() and station.value() == pytest.approx(1600.0)
        assert "Derived" in station.toolTip()
        flags = {path: ed for f in page.layout_forms for path, (_, ed) in f._editors.items()}
        assert set(flags) == {"layout.front_brakes_inboard", "layout.rear_drive_inboard",
                              "layout.rear_brakes_inboard"}
        assert flags["layout.rear_drive_inboard"].isChecked()
        assert "design_intent" in flags["layout.rear_drive_inboard"].toolTip()
        flags["layout.rear_drive_inboard"].setChecked(False)
        assert ctrl.project.layout.rear_drive_inboard is False
        assert "contact patch" in ctrl.static.get("rear.anti").note

        def freeze(p: Project) -> None:
            p.hardpoints = G.freeze_to_hardpoints(p)
            p.geometry_mode = "hardpoints"

        ctrl.apply(freeze)
        page.on_project()
        assert all(f.isEnabled() for f in page.layout_forms)        # flags apply in both modes
        assert not any(f.isEnabled() for f in page.forms)
    finally:
        ctrl.shutdown()
