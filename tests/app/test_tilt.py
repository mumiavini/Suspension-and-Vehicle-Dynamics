"""Tilt table (engine/tilt.py): tip angle, wheel loads, sensitivities, KPI."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from apps.vdstudio.engine import geometry as G
from apps.vdstudio.engine import kpis as K
from apps.vdstudio.engine import mass
from apps.vdstudio.engine.project import Project
from apps.vdstudio.engine.tilt import TiltInputs, tilt_table


@pytest.fixture(scope="module")
def setup():
    geo = G.resolve(Project())
    st = K.compute_static(geo)
    return geo, {"front": st.value("front.rc_static"), "rear": st.value("rear.rc_static")}


def test_rigid_car_tips_at_atan_of_half_track_over_cg_height(setup) -> None:
    geo, rc = setup
    veh = geo.vehicle_data
    r = tilt_table(geo, rc, TiltInputs(roll_gradient_deg_per_g=0.0))
    a = veh.cg_from_front_axle_mm
    y = (geo.track_mm("front") / 2) + (geo.track_mm("rear") / 2 - geo.track_mm("front") / 2) * \
        a / geo.wheelbase_mm                         # half track at the CG station
    assert r.tip_line_mm == pytest.approx(y)
    assert r.tip_angle_deg == pytest.approx(math.degrees(math.atan(y / veh.cg_height_mm)))
    assert r.tip_angle_deg == pytest.approx(r.tip_angle_rigid_deg)
    assert r.cg_shift_mm == 0.0 and r.roll_deg == 0.0
    # the CG height at which a rigid car tips exactly at the test angle
    assert r.max_cg_height_mm == pytest.approx(y / math.tan(math.radians(60.0)))


def test_wheel_loads_balance_force_and_moment(setup) -> None:
    """Σ Fz = W·cos θ, and Σ ΔFz·T = W·sin θ·h + W_s·cos θ·d·φ (Milliken RCVD Ch.18)."""
    geo, rc = setup
    veh = geo.vehicle_data
    th = math.radians(60.0)
    r = tilt_table(geo, rc, TiltInputs())
    W = veh.total_mass_kg * K.GRAVITY
    assert sum(r.wheel_fz_N.values()) == pytest.approx(W * math.cos(th), rel=1e-12)
    moment = sum((r.wheel_fz_N[o] - r.wheel_fz_N[i]) / 2 * geo.track_mm(ax)
                 for ax, o, i in (("front", "FR", "FL"), ("rear", "RR", "RL")))
    sm = mass.sprung_mass(veh.total_mass_kg, veh.unsprung_mass_kg, veh.cg_height_mm,
                          veh.cg_from_front_axle_mm, geo.wheelbase_mm,
                          geo.axles["front"].loaded_radius_mm, geo.axles["rear"].loaded_radius_mm)
    phi = math.radians(r.roll_deg)
    want = W * math.sin(th) * veh.cg_height_mm + sm.mass_kg * K.GRAVITY * math.cos(th) * \
        r.roll_arm_mm * phi
    assert moment == pytest.approx(want, rel=1e-9)
    # downhill (right) wheels carry more; level table = static split
    assert r.wheel_fz_N["FR"] > r.wheel_fz_N["FL"] and r.wheel_fz_N["RR"] > r.wheel_fz_N["RL"]
    flat = tilt_table(geo, rc, TiltInputs(angle_deg=1e-9))
    assert flat.wheel_fz_N["FL"] == pytest.approx(flat.wheel_fz_N["FR"], rel=1e-9)


def test_compliance_and_sensitivities_lower_the_tip_angle(setup) -> None:
    geo, rc = setup
    base = tilt_table(geo, rc, TiltInputs())
    assert base.tip_angle_deg < base.tip_angle_rigid_deg            # body roll moves the CG
    assert base.roll_deg == pytest.approx(1.0 * math.sin(math.radians(60.0)))
    risen = tilt_table(geo, rc, TiltInputs(cg_rise_mm=10.0))
    shifted = tilt_table(geo, rc, TiltInputs(lateral_shift_mm=10.0))
    assert risen.tip_angle_deg < base.tip_angle_deg and shifted.tip_angle_deg < base.tip_angle_deg
    # at the tip, the CG is over the tipping line moved in by the roll shift and the 10 mm
    th = math.radians(shifted.tip_angle_deg)
    roll_shift = base.sprung_fraction * base.roll_arm_mm * math.radians(math.sin(th))
    assert math.tan(th) * geo.vehicle_data.cg_height_mm == pytest.approx(
        base.tip_line_mm - roll_shift - 10.0, abs=1e-6)


def test_first_lift_depends_on_the_roll_stiffness_split(setup) -> None:
    geo, rc = setup
    lifts = {s: tilt_table(geo, rc, TiltInputs(roll_stiffness_front_share=s)).first_lift
             for s in (0.3, 0.5, 0.7)}
    assert lifts[0.3][0] == "RL" and lifts[0.7][0] == "FL"          # the stiffer end lifts
    for share, (corner, ang) in lifts.items():
        r = tilt_table(geo, rc, TiltInputs(angle_deg=ang, roll_stiffness_front_share=share))
        assert r.wheel_fz_N[corner] == pytest.approx(0.0, abs=1e-6)


def test_kpi(setup) -> None:
    geo, _rc = setup
    k = K.compute_static(geo).get("vehicle.tilt_angle")
    assert k.ok and k.unit == "deg" and k.band == (60.0, 90.0)
    assert k.value == pytest.approx(62.13, abs=0.01) and "rigid 62.28°" in k.note
    taller = replace(Project(), vehicle=replace(Project().vehicle, cg_height_mm=360.0))
    assert K.compute_static(G.resolve(taller)).get("vehicle.tilt_angle").value < 60.0
    assert np.isfinite(k.value)
