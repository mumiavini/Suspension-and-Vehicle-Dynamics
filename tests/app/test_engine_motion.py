"""VD Studio engine: pose solver, sweep curves, quasi-static maneuver, concept checks.

Physical invariants (ISO 8855): the zero pose is the static car; a symmetric
car rolled one way mirrors itself rolled the other; patches stay on the road;
load transfer conserves total load; the steady-state outer camber equals the
exact both-wheels-on-road solve.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from apps.vdstudio.engine import concept, curves, maneuver
from apps.vdstudio.engine import geometry as G
from apps.vdstudio.engine import kpis as K
from apps.vdstudio.engine.pose import PoseSolver, spin_axis_from_angles
from apps.vdstudio.engine.project import MassItem, Project

from vdcore.analysis.axle import axle_roll  # noqa: E402


@pytest.fixture(scope="module")
def geo() -> G.CarGeometry:
    return G.resolve(Project())


def test_zero_pose_is_the_static_car(geo) -> None:
    pose = PoseSolver(geo).solve()
    for cid, c in pose.corners.items():
        assert c.converged
        for n in ("UCA_OUT", "LCA_OUT", "TIE_ROD_OUT", "WHEEL_CENTER", "CONTACT_PATCH"):
            assert c.points[n] == pytest.approx(np.array(geo.points[cid][n]), abs=1e-6)
        assert c.camber_deg == pytest.approx(-1.5, abs=1e-9)


def test_roll_is_mirror_symmetric(geo) -> None:
    ps = PoseSolver(geo)
    a, b = ps.solve(roll_deg=+2.0), ps.solve(roll_deg=-2.0)
    for left, right in (("FL", "FR"), ("RL", "RR")):
        assert a.corners[left].camber_deg == pytest.approx(b.corners[right].camber_deg, abs=1e-6)
        assert a.corners[left].travel_mm == pytest.approx(b.corners[right].travel_mm, abs=1e-9)
    # positive roll lifts the LEFT side: left wheels droop
    assert a.corners["FL"].travel_mm < 0 < a.corners["FR"].travel_mm


def test_patches_stay_near_the_road(geo) -> None:
    pose = PoseSolver(geo).solve(heave_mm=10.0, roll_deg=1.5, pitch_deg=0.8, rack_mm=10.0)
    for c in pose.corners.values():
        z = pose.to_world(c.points["CONTACT_PATCH"])[2]
        assert abs(z) < 0.5


def test_positive_rack_steers_left(geo) -> None:
    pose = PoseSolver(geo).solve(rack_mm=10.0)
    # toe-in positive: a left turn toes the LEFT wheel out and the RIGHT wheel in
    assert pose.corners["FL"].toe_deg_per_side < 0 < pose.corners["FR"].toe_deg_per_side
    assert abs(pose.corners["FL"].toe_deg_per_side) > abs(pose.corners["FR"].toe_deg_per_side)


@pytest.mark.parametrize("left", [True, False])
def test_spin_axis_reproduces_solver_angles(left: bool) -> None:
    v = spin_axis_from_angles(-1.5, 0.3, left)
    sy = v[1] if left else -v[1]
    assert -math.degrees(math.atan2(v[2], sy)) == pytest.approx(-1.5, abs=1e-12)
    toe = math.degrees(math.atan2(v[0], v[1])) if left else \
        -math.degrees(math.atan2(-v[0], -v[1]))
    assert toe == pytest.approx(0.3, abs=1e-12)


def test_heave_curves_recover_static_and_gaps(geo) -> None:
    h = curves.heave_curves(geo, -25.0, 25.0, 11)
    mid = 5
    for cid in ("FL", "FR", "RL", "RR"):
        assert h.converged[cid].all()
        assert h.corners[cid]["camber"][mid] == pytest.approx(-1.5, abs=1e-9)
        assert h.corners[cid]["toe"][mid] == pytest.approx(0.0, abs=1e-9)
    # camber gets more negative in bump on this car
    assert h.corners["FL"]["camber"][-1] < h.corners["FL"]["camber"][0]


def test_roll_curves_match_axle_roll(geo) -> None:
    r = curves.roll_curves(geo, 2.0, 3)
    s = axle_roll(geo.axle("front"), 2.0)
    assert r.axles["front"]["outer_camber"][-1] == pytest.approx(s.outer_camber_deg)
    assert r.axles["front"]["outer_camber"][0] == pytest.approx(-1.5, abs=1e-9)


def test_steer_curves_ackermann_is_between_parallel_and_ideal(geo) -> None:
    s = curves.steer_curves(geo, 21)
    m = s.ack_outer > 5
    assert np.all(s.ack_inner[m] > s.ack_outer[m])
    assert np.all(s.ack_inner[m] <= s.ack_ideal[m] + 1e-9)


def test_maneuver_conserves_load_and_matches_axle_roll(geo) -> None:
    st = K.compute_static(geo)
    rc = {"front": st.value("front.rc_static"), "rear": st.value("rear.rc_static")}
    inp = maneuver.ManeuverInputs(ay_g=1.2, ax_g=-0.4)
    r = maneuver.run_maneuver(geo, inp, rc)
    W = geo.vehicle_data.total_mass_kg * K.GRAVITY
    assert sum(w.fz_N for w in r.wheels.values()) == pytest.approx(W, rel=1e-12)
    # left turn: right wheels are outer and gain load
    assert r.wheels["FR"].fz_N > r.wheels["FL"].fz_N
    # braking: front gains
    assert r.longitudinal_transfer_N < 0
    pure = maneuver.run_maneuver(geo, maneuver.ManeuverInputs(ay_g=1.5, ax_g=0.0), rc)
    s = axle_roll(geo.axle("front"), 1.5)
    assert pure.wheels["FR"].camber_road_deg == pytest.approx(s.outer_camber_deg, abs=1e-6)
    assert pure.wheels["FL"].camber_road_deg == pytest.approx(s.inner_camber_deg, abs=1e-6)


def test_maneuver_zero_g_is_static(geo) -> None:
    r = maneuver.run_maneuver(geo, maneuver.ManeuverInputs(ay_g=0.0, ax_g=0.0))
    v = geo.vehicle_data
    W = v.total_mass_kg * K.GRAVITY
    assert r.wheels["FL"].fz_N == pytest.approx(W * v.front_mass_fraction / 2)
    assert r.roll_deg == 0.0 and r.pitch_deg == 0.0
    assert all(w.camber_road_deg == pytest.approx(-1.5, abs=1e-9) for w in r.wheels.values())


def test_lltd_follows_the_elastic_split(geo) -> None:
    lo = maneuver.run_maneuver(geo, maneuver.ManeuverInputs(roll_stiffness_front_share=0.3))
    hi = maneuver.run_maneuver(geo, maneuver.ManeuverInputs(roll_stiffness_front_share=0.7))
    assert lo.lltd_front < hi.lltd_front


def test_cg_estimator_weighted_average() -> None:
    items = [MassItem("a", 10.0, 100.0, 0.0), MassItem("b", 30.0, 300.0, 1000.0)]
    est = concept.cg_estimate(items, 1600.0)
    assert est.total_mass_kg == 40.0
    assert est.cg_height_mm == pytest.approx(250.0)
    assert est.cg_x_mm == pytest.approx(750.0)
    assert est.front_mass_fraction == pytest.approx(1 - 750.0 / 1600.0)
    est2 = concept.cg_estimate([MassItem("a", 10.0, 100.0)], 1600.0)
    assert est2.front_mass_fraction is None


def test_concept_numbers_match_sla_vehicle_results() -> None:
    import sla_geometry as sla

    p = Project()
    ck = concept.concept_kpis(p)
    res = sla.solve_vehicle(p.vehicle, p.front.rc_height_mm, p.rear.rc_height_mm, 1200.0)
    # the roll budget uses the sprung CG; sla_geometry's whole-car value is quoted in the note
    assert f"{res.required_roll_stiffness_Nm_per_deg:.1f} N·m/deg" in ck.get("concept.k_roll").note
    assert f"d = {res.roll_moment_arm_mm:.1f} mm" in ck.get("concept.roll_arm").note
    assert ck.value("concept.tilt_min_track") == pytest.approx(res.tilt_min_track_mm)
    lc = sla.load_cases(sla.run().front, p.vehicle)
    assert ck.value("concept.front_fz_inner") == pytest.approx(lc["Fz_inner"])
