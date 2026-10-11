"""VD Studio engine: telemetry import, validation checks and model-vs-data analysis.

A synthetic log with KNOWN roll gradient, understeer gradient and motion
ratios is written to disk, read back through the real loaders, and the
analyses must recover what was put in. Sign conventions (ISO 8855): ay, yaw
rate and steering positive in a LEFT turn; damper travel positive in
compression; positive roll lifts the left side.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from apps.vdstudio.engine import geometry as G
from apps.vdstudio.engine.project import Project, TelemetrySettings
from apps.vdstudio.engine.telemetry import analysis as A
from apps.vdstudio.engine.telemetry import mapping as M
from apps.vdstudio.engine.telemetry.ld import LdChannel, read_ld, write_ld
from apps.vdstudio.engine.telemetry.session import load_session

from vdcore.geometry.solver import DWSolver  # noqa: E402

G0 = 9.80665
ROLL_GRAD = 1.1          # deg/g
US_GRAD = 1.5            # deg/g
RATIO = 4.5
MR_F, MR_R = 0.9, 0.8
OFFSET = 5.0             # damper zero offset, mm
MONZA = Path("/home/user/random-things/racetrack_dynamics/data/MONZA.ld")


@pytest.fixture(scope="module")
def geo() -> G.CarGeometry:
    return G.resolve(Project())


def _synthetic(geo: G.CarGeometry) -> dict[str, np.ndarray]:
    t = np.arange(0.0, 60.0, 0.02)
    v = np.full_like(t, 15.0)
    ay = 1.2 * np.sin(2 * np.pi * t / 12.0)
    ax = 0.05 * np.cos(2 * np.pi * t / 7.0)
    yaw = np.degrees(ay * G0 / v)
    L = geo.wheelbase_mm / 1000.0
    delta_road = np.degrees(L * np.radians(yaw) / v) + US_GRAD * ay
    swa = RATIO * delta_road
    roll = np.radians(ROLL_GRAD * ay)
    tf, tr = geo.track_mm("front"), geo.track_mm("rear")
    tfr = 0.5 * tf * np.tan(roll)
    trr = 0.5 * tr * np.tan(roll)
    return {
        "Time": t, "Speed": v * 3.6, "LatAcc": ay, "LongAcc": ax, "YawRate": yaw,
        "SteeringAngle": swa,
        "Damper_FL": -tfr * MR_F + OFFSET, "Damper_FR": tfr * MR_F + OFFSET,
        "Damper_RL": -trr * MR_R + OFFSET, "Damper_RR": trr * MR_R + OFFSET,
    }


def _write_csv(path: Path, data: dict[str, np.ndarray], units: dict[str, str]) -> None:
    names = list(data)
    lines = ['"Format","MoTeC CSV File"', '"Venue","synthetic"', "",
             ",".join(names), ",".join(units.get(n, "") for n in names)]
    for i in range(len(data["Time"])):
        lines.append(",".join(f"{data[n][i]:.6f}" for n in names))
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture(scope="module")
def session(tmp_path_factory, geo):
    path = tmp_path_factory.mktemp("tel") / "synthetic.csv"
    _write_csv(path, _synthetic(geo), {"Speed": "km/h", "LatAcc": "g", "LongAcc": "g",
                                       "YawRate": "deg/s", "SteeringAngle": "deg",
                                       "Damper_FL": "mm"})
    return load_session(path)


@pytest.fixture(scope="module")
def settings(session) -> TelemetrySettings:
    cm = M.auto_map(session.names())
    scale = {r: M.guess_scale(r, session.units.get(ch, ""), session.get(ch))[0]
             for r, ch in cm.items()}
    return TelemetrySettings(channel_map=cm, scale=scale, damper_motion_ratio_front=MR_F,
                             damper_motion_ratio_rear=MR_R, min_speed_mps=5.0,
                             min_abs_ay_g=0.3)


def test_csv_loader_skips_metadata_and_reads_units(session) -> None:
    assert session.rate_hz == pytest.approx(50.0)
    assert "Time" not in session.channels
    assert session.units["Speed"] == "km/h"
    assert len(session.channels) == 9


def test_auto_map_and_unit_guess(settings) -> None:
    cm = settings.channel_map
    assert cm["speed"] == "Speed" and cm["ay"] == "LatAcc" and cm["yaw_rate"] == "YawRate"
    assert cm["steer_sw"] == "SteeringAngle"
    assert {cm[f"damper_{c}"] for c in ("FL", "FR", "RL", "RR")} == {
        "Damper_FL", "Damper_FR", "Damper_RL", "Damper_RR"}
    assert settings.scale["speed"] == pytest.approx(1 / 3.6)


def _signals(session, settings):
    return M.signals(session, settings.channel_map, settings.scale, settings.flip_sign)


def test_consistency_checks_pass_on_consistent_data(session, settings, geo) -> None:
    sig = _signals(session, settings)
    zero = A.straight_line_zero(sig, settings)
    travel = A.wheel_travel(sig, settings, zero)
    checks = {c.name: c for c in A.consistency_checks(sig, settings, session.rate_hz, geo,
                                                       travel)}
    assert checks["Yaw rate vs ay·g/V"].status == "ok"
    assert checks["Steering rises with ay"].status == "ok"
    assert checks["Roll (dampers) rises with ay"].status == "ok"


def test_a_flipped_yaw_sign_is_caught(session, settings, geo) -> None:
    bad = replace(settings, flip_sign={"yaw_rate": True})
    sig = _signals(session, bad)
    c = next(c for c in A.consistency_checks(sig, bad, session.rate_hz) if c.name.startswith(
        "Yaw"))
    assert c.status == "fail" and "opposite sign" in c.detail


def test_a_km_h_vs_m_s_mix_up_is_caught(session, settings) -> None:
    bad = replace(settings, scale={**settings.scale, "speed": 1.0})
    sig = _signals(session, bad)
    c = next(c for c in A.consistency_checks(sig, bad, session.rate_hz) if c.name.startswith(
        "Yaw"))
    assert c.status == "fail" and "km/h" in c.detail


def test_roll_gradient_is_recovered(session, settings, geo) -> None:
    sig = _signals(session, settings)
    travel = A.wheel_travel(sig, settings, A.straight_line_zero(sig, settings))
    grads = A.roll_gradients(sig, travel, geo, settings, session.rate_hz, 1.0)
    assert len(grads) == 2
    for g in grads:
        assert g.fit.slope == pytest.approx(ROLL_GRAD, rel=0.01)
        assert g.fit.r2 > 0.999


def test_damper_zero_offset_is_removed(session, settings) -> None:
    sig = _signals(session, settings)
    zero = A.straight_line_zero(sig, settings)
    assert zero["damper_FL"] == pytest.approx(OFFSET, abs=0.5)


def test_understeer_gradient_is_recovered(session, settings, geo) -> None:
    sig = _signals(session, settings)
    us = A.understeer_gradient(sig, settings, session.rate_hz, RATIO, geo.wheelbase_mm)
    assert us is not None
    assert us.fit.slope == pytest.approx(US_GRAD, abs=0.02)


def test_kinematic_replay_matches_a_direct_solve(session, settings, geo) -> None:
    sig = _signals(session, settings)
    travel = A.wheel_travel(sig, settings, A.straight_line_zero(sig, settings))
    maps = A.build_maps(geo, n_travel=31, n_rack=15)
    r = A.replay(session, sig, travel, maps, geo, settings)
    i = int(np.argmax(np.abs(sig["ay"])))
    t_fl = travel["FL"][i]
    rack = sig["steer_sw"][i] / 360.0 * 2 * math.pi * geo.steering_inputs.pinion_radius_mm
    direct = DWSolver(geo.corner("FL")).solve(wheel_travel_mm=float(t_fl), rack_mm=float(rack))
    assert r.camber_chassis["FL"][i] == pytest.approx(direct.camber_deg, abs=0.002)
    assert r.toe["FL"][i] == pytest.approx(direct.toe_deg_per_side, abs=0.005)
    assert r.outside_pct["FL"] == 0.0


def test_ld_round_trip(tmp_path) -> None:
    t = np.linspace(0, 1, 101)
    chans = [LdChannel("Speed", "km/h", 100.0, 50 + 10 * t),
             LdChannel("LatAcc", "g", 100.0, np.sin(t)),
             LdChannel("Damper_FL", "mm", 50.0, np.cos(t[::2]))]
    path = tmp_path / "x.ld"
    write_ld(path, chans, {"venue": "Curitiba", "driver": "test"})
    ld = read_ld(path)
    assert ld.header["venue"] == "Curitiba"
    assert [c.name for c in ld.channels] == ["Speed", "LatAcc", "Damper_FL"]
    assert ld.channels[0].unit == "km/h"
    assert ld.channels[1].data == pytest.approx(np.sin(t).astype("<f4"))
    s = load_session(path)
    assert s.rate_hz == pytest.approx(100.0)
    assert len(s.channels["Damper_FL"]) == len(s.time_s)       # resampled to 100 Hz


def test_channel_quality_flags_a_dead_sensor(tmp_path) -> None:
    t = np.arange(0, 20, 0.05)
    path = tmp_path / "q.ld"
    write_ld(path, [LdChannel("Speed", "m/s", 20.0, 10 + np.sin(t)),
                    LdChannel("Damper_FL", "mm", 20.0, np.zeros_like(t))])
    q = {c.name: c for c in A.channel_quality(load_session(path))}
    assert q["Speed"].status == "ok"
    assert any("constant" in i for i in q["Damper_FL"].issues)


@pytest.mark.skipif(not MONZA.is_file(), reason="sample MoTeC log not present")
def test_reads_the_sample_motec_log() -> None:
    s = load_session(MONZA)
    assert len(s.channels) == 53 and s.rate_hz == pytest.approx(20.0)
    cm = M.auto_map(s.names())
    scale = {r: M.guess_scale(r, s.units.get(ch, ""), s.get(ch))[0] for r, ch in cm.items()}
    sig = M.signals(s, cm, scale, {})
    st = TelemetrySettings(channel_map=cm, scale=scale)
    c = A.consistency_checks(sig, st, s.rate_hz)[0]
    assert c.status == "ok"          # yaw = ay·g/V holds on the sample (slope ~1.00)
