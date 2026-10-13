"""VD Studio telemetry: pins the fixes from the telemetry audit.

Every test reproduces the audited symptom on a synthetic log with a KNOWN
answer, then checks the symptom is gone and the known answer comes back.
Sign conventions (ISO 8855): ay, yaw rate and steering positive in a LEFT
turn; positive roll lifts the left side; damper travel positive in bump.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from apps.vdstudio.engine import geometry as Gm
from apps.vdstudio.engine.project import Project, TelemetrySettings
from apps.vdstudio.engine.telemetry import analysis as A
from apps.vdstudio.engine.telemetry import mapping as M
from apps.vdstudio.engine.telemetry.ld import LdChannel, read_ld, write_ld
from apps.vdstudio.engine.telemetry.session import Session, load_session
from vdcore.geometry.solver import DWSolver

G0 = 9.80665
K_TRUE = 1.8                    # deg/g
TEL = TelemetrySettings()


@pytest.fixture(scope="module")
def geo() -> Gm.CarGeometry:
    return Gm.resolve(Project())


def _mean_steer(geo: Gm.CarGeometry, racks: np.ndarray) -> np.ndarray:
    """Mean road-wheel steer (deg, + left) from static toe, solved directly."""
    fl, fr = DWSolver(geo.corner("FL")), DWSolver(geo.corner("FR"))
    l0, r0 = fl.solve(0.0).toe_deg_per_side, fr.solve(0.0).toe_deg_per_side
    return np.array([0.5 * (-(fl.solve(0.0, rack_mm=float(r)).toe_deg_per_side - l0)
                            + (fr.solve(0.0, rack_mm=float(r)).toe_deg_per_side - r0))
                     for r in racks])


@pytest.fixture(scope="module")
def truth_inverse(geo):
    """Road-wheel steer -> SW angle, from a fine independent DWSolver sweep (121 racks)."""
    st = geo.steering_inputs
    racks = np.linspace(-st.max_rack_travel_mm, st.max_rack_travel_mm, 121)
    return CubicSpline(_mean_steer(geo, racks),
                       racks / (2 * math.pi * st.pinion_radius_mm) * 360.0)


@pytest.fixture(scope="module")
def on_centre_ratio(geo) -> float:
    """The KPI's on-centre ratio (FL toe derivative over ±0.5 mm rack)."""
    fl = DWSolver(geo.corner("FL"))
    d = fl.solve(rack_mm=0.5).toe_deg_per_side - fl.solve(rack_mm=-0.5).toe_deg_per_side
    return 360.0 * abs(1.0 / d) / (2 * math.pi * geo.steering_inputs.pinion_radius_mm)


@pytest.fixture(scope="module")
def table(geo):
    return A.steer_table(geo)


def _ramps(geo, inverse, radii, both=True, lo=0.3, hi=1.5, rate=100.0, dur=40.0):
    """Quasi-steady ay ramps at constant radius; truth δ = L/R + K·ay."""
    L = geo.wheelbase_mm / 1000.0
    out = {k: [] for k in ("speed", "ay", "yaw_rate", "steer_sw")}
    for R in radii:
        for sgn in ((1, -1) if both else (1,)):
            ay = np.linspace(lo, hi, int(dur * rate))
            v = np.sqrt(ay * G0 * R)
            out["speed"].append(v)
            out["ay"].append(sgn * ay)
            out["yaw_rate"].append(sgn * np.degrees(v / R))
            out["steer_sw"].append(sgn * inverse(np.degrees(L / R) + K_TRUE * ay))
    return {k: np.concatenate(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# telemetry-us-linear-ratio
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("radii", [[9.125], [5, 7, 10, 15, 25, 40]], ids=["skidpad", "mix"])
def test_understeer_uses_the_solved_steer_table(geo, truth_inverse, on_centre_ratio, table,
                                                radii) -> None:
    sig = _ramps(geo, truth_inverse, radii)
    lin = A.understeer_gradient(sig, TEL, 100.0, on_centre_ratio, geo.wheelbase_mm)
    # the symptom: the on-centre linear ratio misreads K (1.7308 skidpad, 1.5905 mix)
    assert abs(lin.fit.slope - K_TRUE) > 0.05
    assert "constant ratio" in lin.note
    us = A.understeer_gradient(sig, TEL, 100.0, on_centre_ratio, geo.wheelbase_mm,
                               sw_to_delta=table)
    assert us.fit.slope == pytest.approx(K_TRUE, abs=5e-4)
    assert us.fit.intercept == pytest.approx(0.0, abs=5e-3)
    assert us.fit.r2 > 0.9999
    assert "solved mean FL/FR steer (DWSolver)" in us.note


def test_steer_table_matches_direct_solves_and_is_nan_beyond_the_stroke(geo, table) -> None:
    st = geo.steering_inputs
    racks = np.array([-25.0, -9.3, 0.0, 4.1, 17.7, 25.0])
    sw = racks / (2 * math.pi * st.pinion_radius_mm) * 360.0
    assert table(sw) == pytest.approx(_mean_steer(geo, racks), abs=1e-4)
    lock = st.max_rack_travel_mm / (2 * math.pi * st.pinion_radius_mm) * 360.0
    out = table(np.array([lock + 1.0, -lock - 1.0, np.nan]))
    assert np.isnan(out).all()


def test_steer_table_raises_on_a_non_converged_solve(geo, monkeypatch) -> None:
    orig = DWSolver.solve

    def bad(self, wheel_travel_mm=0.0, roll_deg=0.0, rack_mm=0.0):
        r = orig(self, wheel_travel_mm, roll_deg, rack_mm)
        return r.model_copy(update={"converged": False}) if rack_mm > 30.0 else r

    monkeypatch.setattr(A.DWSolver, "solve", bad)
    with pytest.raises(RuntimeError, match="did not converge"):
        A.steer_table(geo)


# --------------------------------------------------------------------------- #
# telemetry-steer-offset-not-removed
# --------------------------------------------------------------------------- #

def _unbalanced_with_straight(geo, inverse, offset_deg: float) -> dict[str, np.ndarray]:
    """R 15 m: left turns 0.4-1.0 g, right turns 0.8-1.4 g, plus 15 s straight."""
    L, R = geo.wheelbase_mm / 1000.0, 15.0
    parts = []
    for sgn, lo, hi in ((1, 0.4, 1.0), (-1, 0.8, 1.4)):
        a = np.linspace(lo, hi, 3000)
        v = np.sqrt(a * G0 * R)
        parts.append((sgn * a, v, sgn * np.degrees(v / R),
                      sgn * inverse(np.degrees(L / R) + K_TRUE * a)))
    parts.append((np.zeros(1500), np.full(1500, 15.0), np.zeros(1500), np.zeros(1500)))
    ay, v, yaw, sw = (np.concatenate([p[i] for p in parts]) for i in range(4))
    return {"ay": ay, "speed": v, "yaw_rate": yaw, "steer_sw": sw + offset_deg}


def test_steering_zero_from_straight_running(geo, truth_inverse, on_centre_ratio, table) -> None:
    sig = _unbalanced_with_straight(geo, truth_inverse, 3.0)
    raw = A.understeer_gradient(sig, TEL, 100.0, on_centre_ratio, geo.wheelbase_mm,
                                sw_to_delta=table)
    assert raw.fit.slope < 0.5               # the 3 deg offset wrecks K (-0.17)
    assert "no zero applied" in raw.note
    zero = A.straight_line_steer_zero(sig, TEL)
    assert zero == {"steer_sw": pytest.approx(3.0, abs=1e-9)}
    sig0 = dict(sig, steer_sw=sig["steer_sw"] - zero["steer_sw"])
    us = A.understeer_gradient(sig0, TEL, 100.0, on_centre_ratio, geo.wheelbase_mm,
                               sw_to_delta=table, steer_zero=zero)
    assert us.fit.slope == pytest.approx(K_TRUE, abs=5e-4)
    assert "straight-running median (+3.00 deg)" in us.note
    # no straight running: no zero, and nothing invented
    assert A.straight_line_steer_zero({k: v[:6000] for k, v in sig.items()}, TEL) == {}


def test_standstill_steering_tolerance_is_half_a_degree_with_road_wheel_equivalent() -> None:
    n = 2000
    sig = {"speed": np.zeros(n), "ay": np.zeros(n), "steer_sw": np.full(n, 1.0)}
    c = next(c for c in A.consistency_checks(sig, TEL, 100.0, steering_ratio=4.5)
             if c.name == "steer_sw at standstill")
    assert c.status == "warn"                 # 1 deg SW used to pass as 'zeroed' (limit 3)
    assert "≈ +0.222 deg road wheel" in c.value
    sig["steer_sw"] = np.full(n, 0.4)
    c = next(c for c in A.consistency_checks(sig, TEL, 100.0) if c.name.startswith("steer_sw"))
    assert c.status == "ok"


# --------------------------------------------------------------------------- #
# front-corner replay maps: wheel inclination (from swept-camber-in-steer)
# --------------------------------------------------------------------------- #

def _pt(p) -> np.ndarray:
    return np.array([p.x_mm, p.y_mm, p.z_mm])


def test_inclination_matches_the_spin_axis_and_camber_at_zero_steer(geo) -> None:
    for cid in ("FL", "FR"):
        sv = DWSolver(geo.corner(cid))
        for rack in (-38.0, -19.0, 9.5, 38.0):
            for tr in (-25.0, 0.0, 25.0):
                r = sv.solve(wheel_travel_mm=tr, rack_mm=rack)
                spin = sv._reconstruct_spin_axis(_pt(r.ubj), _pt(r.lbj), _pt(r.tro))
                direct = -math.degrees(math.asin(float(np.clip(spin[2], -1.0, 1.0))))
                assert A.inclination_deg(r.camber_deg, r.toe_deg_per_side) == pytest.approx(
                    direct, abs=1e-9)
    for cid in ("FL", "FR", "RL", "RR"):
        r = DWSolver(geo.corner(cid)).solve(0.0)
        assert A.inclination_deg(r.camber_deg, r.toe_deg_per_side) == pytest.approx(
            r.camber_deg, abs=1e-9)
    # the symptom at full lock (verifier: FL +38 mm, toe -39.65): 5.082 projected vs 3.917
    r = DWSolver(geo.corner("FL")).solve(0.0, rack_mm=38.0)
    assert r.camber_deg == pytest.approx(5.082, abs=1e-3)
    assert A.inclination_deg(r.camber_deg, r.toe_deg_per_side) == pytest.approx(3.917, abs=1e-3)


def test_front_maps_carry_inclination_rear_maps_camber(geo) -> None:
    maps = A.build_maps(geo, n_travel=7, n_rack=5)
    node = np.array([[maps.travel[3], maps.rack[-1]]])
    r = DWSolver(geo.corner("FL")).solve(float(node[0, 0]), rack_mm=float(node[0, 1]))
    # scipy's cubic grid spline is only ~1e-5 exact at its own nodes
    assert maps.camber["FL"](node)[0] == pytest.approx(
        A.inclination_deg(r.camber_deg, r.toe_deg_per_side), abs=1e-4)
    assert abs(maps.camber["FL"](node)[0] - r.camber_deg) > 0.5
    rr = DWSolver(geo.corner("RL")).solve(float(node[0, 0]))
    assert maps.camber["RL"](np.array([[node[0, 0], 0.0]]))[0] == pytest.approx(
        rr.camber_deg, abs=1e-4)


# --------------------------------------------------------------------------- #
# telemetry-buildmaps-nan-crash
# --------------------------------------------------------------------------- #

def test_one_failed_grid_solve_keeps_the_maps(geo, monkeypatch) -> None:
    orig = DWSolver.solve
    count = {"n": 0}

    def bad(self, wheel_travel_mm=0.0, roll_deg=0.0, rack_mm=0.0):
        r = orig(self, wheel_travel_mm, roll_deg, rack_mm)
        count["n"] += 1
        return r.model_copy(update={"converged": False}) if count["n"] == 20 else r

    monkeypatch.setattr(A.DWSolver, "solve", bad)
    maps = A.build_maps(geo, n_travel=7, n_rack=5)          # used to raise (cubic + NaN)
    assert maps.failed == 1
    # the 20th solve is FL node (3, 4); far cells stay finite, cells touching it are NaN
    assert np.isfinite(maps.camber["FL"](np.array([[maps.travel[0], maps.rack[0]]]))[0])
    near = np.array([[0.5 * (maps.travel[3] + maps.travel[2]), maps.rack[4]]])
    assert np.isnan(maps.camber["FL"](near)[0])
    assert np.isfinite(maps.camber["FR"](near)[0])


# --------------------------------------------------------------------------- #
# telemetry-lltd-static-offset, telemetry-lltd-moment-vs-force
# --------------------------------------------------------------------------- #

def _loads(ay: np.ndarray, cross_n: float, share: float = 0.48) -> dict[str, np.ndarray]:
    W = 300 * G0
    tot = W * ay * 0.30 / 1.22
    df, dr, s = share * tot, (1 - share) * tot, W / 4
    return {"speed": np.full(len(ay), 15.0), "ay": ay,
            "fz_FL": s - df + cross_n, "fz_FR": s + df - cross_n,
            "fz_RL": s - dr - cross_n, "fz_RR": s + dr + cross_n}


@pytest.mark.parametrize("cross", [0.0, 25.0, 50.0])
def test_lltd_is_offset_free_with_force_and_moment_shares(cross) -> None:
    ay = np.linspace(0.35, 1.5, 4000)                       # left turns only
    lt = A.measured_lltd(_loads(ay, cross), TEL, 100.0, (1240.0, 1200.0))
    assert lt.force_share == pytest.approx(0.48, abs=1e-6)  # was 0.3868 / 0.2936 at ±25/50 N
    tf, tr = 1240.0, 1200.0
    assert lt.moment_share == pytest.approx(0.48 * tf / (0.48 * tf + 0.52 * tr), abs=1e-6)
    assert np.nanmean(lt.y) == pytest.approx(0.48, abs=1e-6)   # scatter is offset-free too
    assert "static L-R offsets removed" in lt.note


# --------------------------------------------------------------------------- #
# CSV loader: NaN time, time units, beacon header, monotonic time, TSV comma
# --------------------------------------------------------------------------- #

N, RATE = 3000, 100.0
T = np.arange(N) / RATE
AY = np.sin(2 * np.pi * T / 20.0)
V = np.full(N, 15.0)
YAW = np.degrees(AY * G0 / V)


def _rows(tcol, header=("Time", "G_Lat", "Ground Speed", "Yaw Rate"),
          units=("s", "g", "m/s", "deg/s")) -> list[list[str]]:
    rows = [list(header)] + ([list(units)] if units else [])
    rows += [[f"{tcol[i]:.4f}", f"{AY[i]:.5f}", f"{V[i]:.3f}", f"{YAW[i]:.4f}"]
             for i in range(N)]
    return rows


def _write(path: Path, rows: list[list[str]], delim: str = ",") -> Path:
    path.write_text("\n".join(delim.join(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_an_empty_time_cell_is_dropped_not_propagated(tmp_path) -> None:
    rows = _rows(T)
    rows[2 + 1500][0] = ""
    s = load_session(_write(tmp_path / "a.csv", rows))
    assert np.isfinite(s.time_s).all() and len(s.time_s) == N - 1
    assert s.rate_hz == pytest.approx(100.0)
    assert s.meta["dropped rows (no time)"] == "1"
    sig = {"speed": s.channels["Ground Speed"], "ay": s.channels["G_Lat"]}
    assert A.steady_mask(sig, TEL, s.rate_hz).sum() > 2000      # used to raise
    with pytest.raises(ValueError, match="invalid sample rate"):
        A.steady_mask(sig, TEL, math.nan)


@pytest.mark.parametrize("header,units", [("Time", "ms"), ("Timestamp", "ms"), ("time_ms", None),
                                          ("Time [ms]", None)])
def test_time_column_unit_is_applied(tmp_path, header, units) -> None:
    rows = _rows(T * 1000.0, header=(header, "G_Lat", "Ground Speed", "Yaw Rate"),
                 units=(units or "", "g", "m/s", "deg/s"))
    s = load_session(_write(tmp_path / "ms.csv", rows))
    assert s.rate_hz == pytest.approx(100.0)                    # was 0.1 Hz
    assert s.duration_s == pytest.approx(29.99)
    assert s.meta["time column"] == f"{header} [ms]"
    heading = float(np.nansum(s.channels["Yaw Rate"]) / s.rate_hz)
    assert heading == pytest.approx(np.sum(YAW) / RATE, rel=1e-3)


def test_an_implausible_time_base_is_refused(tmp_path) -> None:
    rows = _rows(T * 1000.0, header=("time_ms", "G_Lat", "Ground Speed", "Yaw Rate"))
    with pytest.raises(ValueError, match="implausible time base.*name says ms"):
        load_session(_write(tmp_path / "clash.csv", rows))


def _motec(n_beacons: int, blank_after_units: int) -> list[str]:
    meta = ['"Format","MoTeC CSV File","","","Workbook","x"',
            '"Venue","Test","","","Worksheet","y"',
            '"Sample Rate","100.000","Hz","","End Time","29.990","s"',
            '"Duration","29.990","s","","Start Distance","0","m"',
            '"Range","entire outing","","","End Distance","450","m"',
            '"Beacon Markers",' + ",".join(f'"{6.0 * (k + 1):.3f}"' for k in range(n_beacons)),
            "", ""]
    body = ['"Time","G_Lat","Ground Speed","Yaw Rate"', '"s","g","m/s","deg/s"']
    body += [""] * blank_after_units
    body += [f'"{T[i]:.3f}","{AY[i]:.5f}","{V[i]:.3f}","{YAW[i]:.4f}"' for i in range(N)]
    return meta + body


@pytest.mark.parametrize("beacons", [3, 4, 6])
@pytest.mark.parametrize("blank", [0, 2])
def test_motec_metadata_and_blank_rows_do_not_hide_the_header(tmp_path, beacons, blank) -> None:
    p = tmp_path / f"m{beacons}_{blank}.csv"
    p.write_text("\n".join(_motec(beacons, blank)) + "\n", encoding="utf-8")
    s = load_session(p)
    assert s.names() == ["G_Lat", "Ground Speed", "Yaw Rate"]
    assert s.units["Ground Speed"] == "m/s"
    assert s.rate_hz == pytest.approx(100.0)


def test_time_resets_duplicates_and_distance_columns(tmp_path) -> None:
    tt = T.copy()
    tt[1500:] = T[:1500]
    s = load_session(_write(tmp_path / "reset.csv", _rows(tt)))
    assert np.all(np.diff(s.time_s) > 0)
    assert s.duration_s == pytest.approx(29.99)                 # was 14.99
    assert "1 at data row(s) 1500" in s.meta["time resets"]
    rows = _rows(T)
    rows.insert(100, list(rows[99]))                            # an exact duplicate row
    s = load_session(_write(tmp_path / "dup.csv", rows))
    assert s.meta["dropped rows (duplicate time)"] == "1" and len(s.time_s) == N
    rows = [["Distance", "G_Lat", "Ground Speed"], ["m", "g", "m/s"]]
    rows += [[f"{15 * T[i]:.3f}", f"{AY[i]:.5f}", f"{V[i]:.3f}"] for i in range(N)]
    with pytest.raises(ValueError, match="no time column"):    # distance is never time
        load_session(_write(tmp_path / "dist.csv", rows))


def test_tab_separated_decimal_comma(tmp_path) -> None:
    rows = _rows(T)
    rows = rows[:2] + [[v.replace(".", ",") for v in r] for r in rows[2:]]
    s = load_session(_write(tmp_path / "c.tsv", rows, "\t"))
    assert s.rate_hz == pytest.approx(100.0) and len(s.channels) == 3
    assert s.channels["G_Lat"][:N - 1] == pytest.approx(AY[:N - 1], abs=1e-5)


# --------------------------------------------------------------------------- #
# .ld reader: skipped channels, non-integer rates
# --------------------------------------------------------------------------- #

def test_ld_skipped_channels_are_reported_and_fractional_rates_refused(tmp_path) -> None:
    p = tmp_path / "skip.ld"
    write_ld(p, [LdChannel("A", "g", 100.0, np.arange(400, dtype=np.float32)),
                 LdChannel("B", "mm", 100.0, np.arange(400, dtype=np.float32))])
    buf = bytearray(p.read_bytes())
    meta0 = struct.unpack_from("<I", buf, 0x08)[0]
    struct.pack_into("<H", buf, meta0 + 124 + 18, 0x09)        # channel B: unknown dtype
    p.write_bytes(bytes(buf))
    ld = read_ld(p)
    assert [c.name for c in ld.channels] == ["A"]
    assert ld.skipped == [("B", "dtype 0x09 size 4 not supported")]
    s = load_session(p)
    assert s.skipped == [("B", "dtype 0x09 size 4 not supported")]
    assert "skipped channels" in s.meta
    with pytest.raises(ValueError, match="whole Hz"):
        write_ld(tmp_path / "c.ld", [LdChannel("C", "deg", 12.5, np.zeros(50))])


# --------------------------------------------------------------------------- #
# channel quality: spikes on native samples, rest values are not faults
# --------------------------------------------------------------------------- #

def _quality_log(path: Path, n_ay_spikes=0, n_damper_spikes=0) -> Session:
    rng = np.random.default_rng(1)
    dur = 120.0
    t500, t100, t10 = (np.arange(int(dur * f)) / f for f in (500, 100, 10))

    def moving(t):
        return t > 15.0                                        # 15 s in the pits

    v10 = np.where(moving(t10), 15 + 5 * np.sin(2 * np.pi * t10 / 23), 0.0)
    ay = np.where(moving(t100), 1.2 * np.sin(2 * np.pi * t100 / 11), 0.0)
    ay += 0.01 * rng.standard_normal(len(t100))
    damp = 10 * np.sin(2 * np.pi * t500 / 11) + 0.05 * rng.standard_normal(len(t500))
    thr = np.where(moving(t100), np.clip(60 + 80 * np.sin(2 * np.pi * t100 / 9), 0, 100), 0.0)
    brk = np.where(moving(t100), np.clip(-30 + 60 * np.sin(2 * np.pi * t100 / 9 + 2), 0, 40), 0)
    if n_ay_spikes:
        ay[rng.choice(np.arange(2000, len(ay) - 10), n_ay_spikes, replace=False)] += 3.0
    if n_damper_spikes:
        idx = rng.choice(np.arange(10000, len(damp) - 10), n_damper_spikes, replace=False)
        damp[idx] += 40.0
    write_ld(path, [LdChannel("Ground Speed", "m/s", 10.0, v10.astype(np.float32)),
                    LdChannel("G_Lat", "g", 100.0, ay.astype(np.float32)),
                    LdChannel("Damper FL", "mm", 500.0, damp.astype(np.float32)),
                    LdChannel("Throttle", "%", 100.0, thr.astype(np.float32)),
                    LdChannel("Brake Press", "bar", 100.0, brk.astype(np.float32))])
    return load_session(path)


def test_a_clean_log_raises_no_quality_issue(tmp_path) -> None:
    s = _quality_log(tmp_path / "clean.ld")
    for speed in (None, s.channels["Ground Speed"]):
        q = {c.name: c for c in A.channel_quality(s, speed)}
        assert all(not c.issues for c in q.values()), {c.name: c.issues for c in q.values()}
    assert q["G_Lat"].n == 12000 and q["G_Lat"].rate_hz == 100.0    # native samples


def test_spikes_are_found_on_native_samples_and_counted_as_events(tmp_path) -> None:
    q = {c.name: c for c in A.channel_quality(_quality_log(tmp_path / "a.ld", n_ay_spikes=40))}
    assert q["G_Lat"].spikes == 40 and "40 spikes" in q["G_Lat"].issues   # was 0 (resampled)
    q = {c.name: c for c in A.channel_quality(_quality_log(tmp_path / "d.ld",
                                                           n_damper_spikes=6))}
    assert q["Damper FL"].spikes == 6 and "6 spikes" in q["Damper FL"].issues   # was 12, unflagged


def test_real_flat_lines_and_saturation_are_still_flagged() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(60000) / 500.0
    damp = 10 * np.sin(2 * np.pi * t / 11) + 0.05 * rng.standard_normal(len(t))
    frozen = damp.copy()
    frozen[20000:22500] = frozen[20000]                       # 5 s stuck mid-stroke
    s = Session("x", t, {"clip": np.minimum(damp, 9.0), "frozen": frozen},
                {"clip": "mm", "frozen": "mm"}, {"clip": 500.0, "frozen": 500.0})
    q = {c.name: c for c in A.channel_quality(s, np.full(len(t), 15.0))}
    assert any("saturation" in i for i in q["clip"].issues)
    assert any(i.startswith("flat for 5.0 s") for i in q["frozen"].issues)


# --------------------------------------------------------------------------- #
# channel mapping and unit guessing
# --------------------------------------------------------------------------- #

def test_auto_map_skips_look_alikes_and_asks_for_review() -> None:
    assert M.auto_map(["Steering Angle", "Steer Torque", "G_Lat"])["steer_sw"] == "Steering Angle"
    assert M.auto_map(["Steer Rate", "Steering Angle"])["steer_sw"] == "Steering Angle"
    for names in (["Damper Vel FL", "Damper Pos FL"], ["Damper Pos FL", "Damper Vel FL"]):
        assert M.auto_map(names)["damper_FL"] == "Damper Pos FL"
    assert M.auto_map(["DamperVelFL", "Damper Travel FL"])["damper_FL"] == "Damper Travel FL"
    review: dict[str, list[str]] = {}
    cm = M.auto_map(["GPS_LatAcc", "LateralAcc", "Fz FL", "Load FL"], review)
    assert cm["ay"] == "LateralAcc" and review["ay"] == ["GPS_LatAcc"]    # IMU over GPS
    assert review["fz_FL"] == ["Load FL"]


def test_auto_map_knows_common_logger_names() -> None:
    motec = M.auto_map(["G Force Lat", "G Force Long", "Ground Speed", "Brake Pres Front",
                        "Steered Angle", "Yaw Rate"])
    assert motec["ay"] == "G Force Lat" and motec["ax"] == "G Force Long"
    assert motec["brake"] == "Brake Pres Front"
    assert M.auto_map(["GPS_LonAcc", "GPS_Speed"])["ax"] == "GPS_LonAcc"
    loads = M.auto_map(["Load FL", "Load FR", "Load RL", "Load RR"])
    assert loads == {f"fz_{c}": f"Load {c}" for c in ("FL", "FR", "RL", "RR")}


def test_guess_scale_units() -> None:
    data = np.linspace(-20, 20, 100)
    for role, unit in (("steer_sw", "Nm"), ("damper_FL", "mm/s"), ("damper_FL", "V")):
        k, why = M.guess_scale(role, unit, data)
        assert k == 1.0 and "no conversion applied" in why and "consistent" not in why
    assert M.guess_scale("speed", "km/hr", data)[0] == pytest.approx(1 / 3.6)
    assert M.guess_scale("yaw_rate", "deg/sec", np.array([0.0, 2.5]))[0] == 1.0   # was ×57.3
    assert M.guess_scale("steer_sw", "degrees", data)[0] == 1.0
    assert M.guess_scale("ay", "m/s/s", data)[0] == pytest.approx(1 / G0)
    k, why = M.guess_scale("speed", "furlong/fortnight", np.array([0.0, 20.0]))
    assert why.startswith("unknown unit 'furlong/fortnight'")
    k, why = M.guess_scale("speed", "", np.array([0.0, 50.4]))    # unlabelled skidpad, km/h
    assert k == pytest.approx(1 / 3.6) and "> 40" in why
    assert M.guess_scale("fz_FL", "N", data) == (1.0, "unit 'N'")


# --------------------------------------------------------------------------- #
# gradients: pitch split, ay disclosure, steady mask
# --------------------------------------------------------------------------- #

def _travel_from(geo, roll_f, roll_r, pitch):
    tf, tr, L = geo.track_mm("front"), geo.track_mm("rear"), geo.wheelbase_mm
    df, dr = tf * np.tan(np.radians(roll_f)), tr * np.tan(np.radians(roll_r))
    dp = L * np.tan(np.radians(pitch))
    return {"FL": 0.5 * dp - 0.5 * df, "FR": 0.5 * dp + 0.5 * df,
            "RL": -0.5 * dp - 0.5 * dr, "RR": -0.5 * dp + 0.5 * dr}


def test_pitch_gradient_is_split_into_braking_and_traction(geo) -> None:
    ax = np.concatenate([np.linspace(-1.5, -0.16, 3000), np.linspace(0.16, 1.0, 3000)])
    pitch = np.where(ax < 0, -0.8 * ax, -0.4 * ax)
    z = np.zeros_like(ax)
    sig = {"speed": np.full(len(ax), 15.0), "ax": ax, "ay": z}
    brake, trac = A.pitch_gradient(sig, _travel_from(geo, z, z, pitch), geo, TEL, 100.0)
    assert "braking" in brake.name and "traction" in trac.name
    assert brake.fit.slope == pytest.approx(-0.8, abs=1e-6)    # one line gave -0.644
    assert trac.fit.slope == pytest.approx(-0.4, abs=1e-6)


def test_ay_roll_contamination_is_disclosed(geo, truth_inverse, table) -> None:
    ay = np.concatenate([np.linspace(0.35, 1.5, 4000), -np.linspace(0.35, 1.5, 4000)])
    sig = {"speed": np.full(len(ay), 15.0), "ay": ay}
    for g in A.roll_gradients(sig, _travel_from(geo, ay, 1.25 * ay, 0 * ay), geo, TEL, 100.0,
                              1.0):
        assert "ay·cosφ + sinφ" in g.note
    us = A.understeer_gradient(_ramps(geo, truth_inverse, [9.125]), TEL, 100.0, 4.5,
                               geo.wheelbase_mm, sw_to_delta=table)
    assert "ay·cosφ + sinφ" in us.note


def test_a_sparse_unrelated_channel_does_not_shrink_the_steady_mask() -> None:
    ay = np.sin(2 * np.pi * np.arange(6000) / 100.0 / 20.0)
    base = {"speed": np.full(6000, 15.0), "ay": ay}
    n0 = A.steady_mask(base, TEL, 100.0).sum()
    thr = np.full(6000, np.nan)
    thr[::10] = 50.0
    dist = np.where(np.arange(6000) < 3000, np.nan, np.arange(6000) * 0.15)
    assert A.steady_mask(dict(base, throttle=thr), TEL, 100.0).sum() == n0      # was -90 %
    assert A.steady_mask(dict(base, distance=dist), TEL, 100.0).sum() == n0     # was -50 %
    nan_ay = ay.copy()
    nan_ay[:100] = np.nan
    assert A.steady_mask(dict(base, ay=nan_ay), TEL, 100.0)[:100].sum() == 0


# --------------------------------------------------------------------------- #
# telemetry-consistency-diagnosis
# --------------------------------------------------------------------------- #

def _checks(geo, sig, travel):
    return {c.name: c for c in A.consistency_checks(sig, TEL, 100.0, geo, travel)}


def test_consistency_diagnoses(geo) -> None:
    t = np.arange(12000) / 100.0
    ay = 1.2 * np.sin(2 * np.pi * t / 20.0)
    v = np.full_like(t, 15.0)
    yaw = np.degrees(ay * G0 / v)
    ok = {"speed": v, "ay": ay, "yaw_rate": yaw, "steer_sw": 40 * ay}
    trv = _travel_from(geo, ay, 1.25 * ay, 0 * ay)
    c = _checks(geo, ok, dict(trv, RL=trv["RR"], RR=trv["RL"]))     # rear swapped only
    assert c["Roll (front dampers) rises with ay"].status == "ok"
    assert c["Roll (rear dampers) rises with ay"].status == "fail"  # used to pass
    c = _checks(geo, {"speed": v, "ay": -ay, "yaw_rate": -yaw, "steer_sw": -40 * ay}, trv)
    assert c["Yaw rate vs ay·g/V"].status == "ok"
    assert all("Y-right (SAE)" in c[f"Roll ({a} dampers) rises with ay"].detail
               for a in ("front", "rear"))
    yaw_check = "Yaw rate vs ay·g/V"
    assert "mph" in _checks(geo, dict(ok, speed=v / 0.44704), trv)[yaw_check].detail
    d = _checks(geo, dict(ok, ay=ay * G0), trv)[yaw_check].detail
    assert "0.102" in d and "m/s²" in d and "9.8:" not in d
    d = _checks(geo, dict(ok, ay=-ay, yaw_rate=np.radians(yaw)), trv)[yaw_check].detail
    assert "rad/s" in d and "opposite sign" in d                      # was 'inconsistent'


# --------------------------------------------------------------------------- #
# UI: the telemetry page shows what the engine now computes
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("VDSTUDIO_NO_GL", "1")
    qtw = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("pyqtgraph")
    from apps.vdstudio.ui import theme
    a = qtw.QApplication.instance() or qtw.QApplication([])
    theme.apply(a, theme.DARK)
    return a


def _page_log(geo, inverse, path: Path) -> Path:
    """75 s at 50 Hz: 5 s parked, 5 s straight, cornering; 3 deg SW offset, ±40 N cross."""
    t = np.arange(0.0, 75.0, 0.02)
    v = np.where(t < 5, 0.0, 12.0)
    ay = np.where(t >= 10, 1.3 * np.sin(2 * np.pi * (t - 10) / 15.0), 0.0)
    yaw = np.where(v > 0, np.degrees(ay * G0 / np.maximum(v, 1e-9)), 0.0)
    L = geo.wheelbase_mm / 1000.0
    delta = np.where(v > 0, np.degrees(L * np.radians(yaw) / np.maximum(v, 1e-9)), 0.0)
    sw = inverse(delta + K_TRUE * ay) + 3.0
    df = 0.5 * geo.track_mm("front") * np.tan(np.radians(ay))
    dr = 0.5 * geo.track_mm("rear") * np.tan(np.radians(1.25 * ay))
    W = 300 * G0
    tot = W * ay * 0.3 / 1.22
    data = {"Time": t, "Ground Speed": v * 3.6, "G_Lat": ay, "Yaw Rate": yaw,
            "Steering Angle": sw, "Damper FL": -0.9 * df + 4, "Damper FR": 0.9 * df + 4,
            "Damper RL": -0.8 * dr + 4, "Damper RR": 0.8 * dr + 4,
            "Fz FL": W / 4 - 0.47 * tot + 40, "Fz FR": W / 4 + 0.47 * tot - 40,
            "Fz RL": W / 4 - 0.53 * tot - 40, "Fz RR": W / 4 + 0.53 * tot + 40}
    units = ["s", "km/h", "g", "deg/s", "deg"] + ["mm"] * 4 + ["N"] * 4
    lines = ['"Format","MoTeC CSV File"', '"Beacon Markers","10.0","20.0","30.0","40.0","50.0"',
             "", ",".join(f'"{n}"' for n in data), ",".join(f'"{u}"' for u in units), "", ""]
    lines += [",".join(f"{data[n][i]:.6f}" for n in data) for i in range(len(t))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _rows_of(tbl) -> list[list[str]]:
    return [[tbl.item(i, j).text() if tbl.item(i, j) else "" for j in range(tbl.columnCount())]
            for i in range(tbl.rowCount())]


def test_telemetry_page_end_to_end(qapp, geo, truth_inverse, tmp_path, monkeypatch) -> None:
    import time

    from PySide6 import QtWidgets

    from apps.vdstudio.ui.controller import ProjectController
    from apps.vdstudio.ui.pages.telemetry import TelemetryPage

    ctrl = ProjectController()
    ctrl.recompute()
    page = TelemetryPage(ctrl)

    def edit(p):
        p.telemetry = replace(p.telemetry, damper_motion_ratio_front=0.9,
                              damper_motion_ratio_rear=0.8, min_speed_mps=5.0)

    ctrl.apply(edit, geometry=False)
    page.on_project()
    page.load_path(str(_page_log(geo, truth_inverse, tmp_path / "log.csv")))
    assert "time column: Time [s]" in page.info.text()
    model = {r[0]: r for r in _rows_of(page.model_table)}
    us = model["Understeer gradient"]
    assert us[1].startswith("+1.800 deg/g (intercept +0.00 deg)")
    assert "DWSolver" in us[4] and "(+3.00 deg)" in us[4]
    lt = model["Measured LLTD (front share)"]
    assert lt[1] == "0.470 force / 0.478 moment"                # ±40 N cross-weight removed
    assert lt[3].startswith("0.442 force / 0.45 moment")        # lltd_front as a force share
    checks = {r[0]: r for r in _rows_of(page.checks)}
    assert checks["steer_sw zero (straight running)"][2].startswith("+3.000 deg")
    assert "Roll (rear dampers) rises with ay" in checks

    page._replay()
    t0 = time.time()
    while page.maps is None and time.time() - t0 < 120:
        qapp.processEvents()
        time.sleep(0.02)
    assert page.maps is not None
    hdr = [page.replay_table.horizontalHeaderItem(j).text()
           for j in range(page.replay_table.columnCount())]
    assert hdr[1:3] == ["mean road camber, outer", "mean road camber, inner"]
    rows = {r[0]: r for r in _rows_of(page.replay_table)}
    # rear: outer and inner differ by twice the roll-induced change (1.25 deg/g roll)
    rl_out, rl_in = (float(rows["RL"][k].split()[0]) for k in (1, 2))
    assert rl_out > rl_in + 1.0
    status = page.replay_status.text()
    assert "front" in status and "rear" in status and "inclination" in status
    assert "did not converge" not in status
    page.maps = replace(page.maps, failed=2)
    page._show_replay()
    assert "2 grid solves did not converge" in page.replay_status.text()

    # an unreadable time base reaches the user as a message, not a traceback
    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        lambda *a, **k: warned.append(a[2]))
    bad = _write(tmp_path / "bad.csv",
                 _rows(T * 1000.0, header=("time_ms", "G_Lat", "Ground Speed", "Yaw Rate")))
    page.load_path(str(bad))
    assert warned and "implausible time base" in warned[0]
    ctrl.shutdown()
