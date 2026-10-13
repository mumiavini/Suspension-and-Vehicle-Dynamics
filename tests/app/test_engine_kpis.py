"""VD Studio engine: the KPI layer must quote the same numbers as the reference code.

Every value the app shows is cross-checked against the module that owns it
(vdcore.analysis.*, steering_geometry.py, sla_geometry.py), and a failed solve
must surface as a failure, never as a plausible number.
"""

from __future__ import annotations

import math

import pytest

from apps.vdstudio.engine import geometry as G
from apps.vdstudio.engine import kpis as K
from apps.vdstudio.engine.project import Project, with_path

import geometry_summary as gs  # noqa: E402
import sla_geometry as sla  # noqa: E402
import steering_geometry as stg  # noqa: E402
from vdcore.analysis.axle import axle_rates, axle_roll  # noqa: E402
from vdcore.analysis.toe import bump_steer  # noqa: E402


@pytest.fixture(scope="module")
def geo() -> G.CarGeometry:
    return G.resolve(Project())


@pytest.fixture(scope="module")
def ks(geo) -> K.KpiSet:
    return K.compute_all(geo)


@pytest.fixture(scope="module")
def steer_ref() -> stg.SteeringReport:
    return stg.run()


def test_static_values_pinned_on_2027(ks) -> None:
    assert ks.value("front.camber_L") == pytest.approx(-1.5, abs=1e-6)
    assert ks.value("front.camber_R") == pytest.approx(-1.5, abs=1e-6)
    assert ks.value("front.caster_L") == pytest.approx(5.0, abs=1e-6)
    assert ks.value("front.kpi_L") == pytest.approx(10.0, abs=1e-6)
    # OptimumK-correlated values pinned in vdcore's own tests
    assert ks.value("front.scrub_L") == pytest.approx(15.08, abs=0.01)
    assert ks.value("front.trail_L") == pytest.approx(21.43, abs=0.01)
    assert ks.value("front.scrub_R") == pytest.approx(ks.value("front.scrub_L"), abs=1e-9)


def test_rates_and_roll_match_vdcore_axle(geo, ks) -> None:
    for axle in ("front", "rear"):
        inp = geo.axle_inputs(axle)
        r = axle_rates(geo.axle(axle), travel_bump_mm=inp.travel_bump_mm,
                       travel_droop_mm=inp.travel_droop_mm)
        s = axle_roll(geo.axle(axle), inp.roll_reference_deg)
        assert ks.value(f"{axle}.camber_gain") == pytest.approx(r.camber_gain_deg_per_mm)
        assert ks.value(f"{axle}.rc_migration") == pytest.approx(r.rc_migration_mm_per_mm)
        assert ks.value(f"{axle}.roll_outer_camber") == pytest.approx(s.outer_camber_deg)
        assert ks.value(f"{axle}.roll_rc_height") == pytest.approx(s.rc_height_mm)


def test_camber_gain_is_negative_in_bump_and_inside_the_signed_band(ks) -> None:
    k = ks.get("front.camber_gain")
    assert k is not None and k.value < 0
    assert k.status == "in"


def test_bump_steer_matches_vdcore_toe(geo, ks) -> None:
    for axle, cid in (("front", "FL"), ("rear", "RL")):
        bs = bump_steer(geo.corner(cid), wheel_travel_range_mm=geo.axle_inputs(axle)
                        .travel_bump_mm)
        assert ks.value(f"{axle}.bump_steer") == pytest.approx(bs.linear_deg_per_mm_per_side)
        assert ks.value(f"{axle}.bump_steer_peak") == pytest.approx(bs.peak_abs_deg_per_side)


def test_steering_matches_steering_geometry(ks, steer_ref) -> None:
    r = steer_ref.rates
    assert ks.value("steering.ackermann") == pytest.approx(r.ackermann_pct_at_target, abs=0.01)
    assert ks.value("steering.ratio") == pytest.approx(r.steering_ratio, rel=1e-6)
    assert ks.value("steering.rack_per_deg") == pytest.approx(abs(r.c_factor_mm_per_deg),
                                                              rel=1e-6)
    assert ks.value("steering.max_steer_outer") == pytest.approx(r.max_steer_at_stroke_deg)
    assert ks.value("steering.max_steer_inner") == pytest.approx(
        r.max_steer_inner_at_stroke_deg)
    e = steer_ref.effort
    assert ks.value("steering.effort_rack_force") == pytest.approx(e.rack_force_N, rel=1e-6)
    assert ks.value("steering.effort_sw_torque") == pytest.approx(e.steering_wheel_torque_Nm,
                                                                  rel=1e-6)


def test_rod_end_swing_agrees_with_steering_geometry(ks, steer_ref) -> None:
    """Both now hold the two tie-rod ends in one frame (see test_steering_rod_end)."""
    assert ks.value("steering.rod_end") == pytest.approx(
        steer_ref.rates.worst_rod_end_misalignment_deg, abs=1e-9)


def test_geometric_ackermann_is_the_scripts_value(ks, steer_ref) -> None:
    assert ks.value("steering.geometric_ackermann") == pytest.approx(
        steer_ref.geometry.geometric_ackermann_pct)


def test_anti_dive_matches_sla_construction_to_caster_effect(ks) -> None:
    """sla uses zero-caster ball joints; the merged corner carries caster. Close, not equal."""
    design = sla.run()
    assert ks.value("front.anti") == pytest.approx(design.front.anti_percent, abs=0.1)
    assert ks.value("rear.anti") == pytest.approx(0.0, abs=1e-9)


def test_parallel_pivot_axes_report_svic_at_infinity_not_a_failure(ks) -> None:
    rear = ks.get("rear.anti")
    assert rear is not None and rear.ok
    assert "infinity" in rear.note
    assert ks.get("rear.svic_x") is None


def test_member_lengths_match_geometry_summary(geo) -> None:
    hp = gs.build_merged(sla.run(), stg.run(with_sweep=False))
    for cid in ("FL", "RL"):
        ref = gs.member_lengths(hp, cid)
        mine = K.member_lengths(geo, cid)
        assert mine["lca_front_leg"] == pytest.approx(ref.lca_front)
        assert mine["uca_rear_leg"] == pytest.approx(ref.uca_rear)
        assert mine["kingpin_3d"] == pytest.approx(ref.kingpin_3d)
        assert mine["lca_ea"] == pytest.approx(ref.lca_sweep / (ref.lca_base / 2))


def test_rules_and_tilt(ks) -> None:
    assert ks.get("vehicle.wheelbase").status == "in"
    tilt = 2 * 320.0 * math.tan(math.radians(60.0))
    assert ks.value("vehicle.tilt_margin") == pytest.approx(1200.0 - tilt)


def test_bands_are_information_with_a_source(ks) -> None:
    for k in ks.items.values():
        if k.band is not None:
            assert k.band_source, k.key


def test_out_of_band_items_on_2027_are_the_known_kingpin_lengths(ks) -> None:
    out = sorted(k.key for k in ks.items.values() if k.status == "out")
    assert out == ["front.kingpin_3d", "rear.kingpin_2d", "rear.kingpin_3d"]


def test_impossible_travel_fails_loudly_instead_of_quoting_a_number() -> None:
    """Asking for more bump than the linkage can reach must mark the KPI failed."""
    p = with_path(Project(), "front.travel_bump_mm", 400.0)
    geo = G.resolve(p)
    dyn = K.compute_dynamic(geo, ["rates"])
    k = dyn.get("front.camber_gain")
    assert k is not None
    assert k.status == "fail" and not math.isfinite(k.value) and k.error
    assert k.text() == "—"


def test_kpi_text_never_prints_negative_zero() -> None:
    k = K.KPI("x", "x", "g", "front", -1e-12, fmt=".3f")
    assert k.text() == "0.000"


def test_dynamic_groups_can_be_requested_individually(geo) -> None:
    only = K.compute_dynamic(geo, ["bump_steer"])
    assert set(k.split(".")[1] for k in only.items) <= {
        "bump_steer", "bump_steer_peak", "toe_full_bump", "toe_full_droop"}


def test_edit_moves_the_static_roll_centre() -> None:
    p = with_path(Project(), "front.rc_height_mm", 45.0)
    st = K.compute_static(G.resolve(p))
    # 48.27 (45.5 before 2026-10-12): the instant-axis RC sees the 0.0126 deg/mm bump
    # steer the fixed rack height leaves on the edited front; re-nulled it reads 45.42.
    assert st.value("front.rc_static") == pytest.approx(48.27, abs=0.1)
