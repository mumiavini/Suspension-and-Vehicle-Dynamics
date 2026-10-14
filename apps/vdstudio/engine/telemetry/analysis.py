"""Telemetry validation: is the data right, and does the car do what the model says?

Two questions, kept apart:

1. **Data validation** -- per-channel quality (rate, gaps, flat-lines, spikes,
   saturation) and cross-channel physics checks that need no model: yaw rate
   must equal ``ay·g/V`` in steady cornering (catches unit and sign errors in
   three channels at once), steering and roll must rise with ``ay``, sensors
   must read zero at standstill.
2. **Model vs measurement** -- roll and pitch gradients from the dampers
   against the design-intent targets, the understeer gradient, and a
   kinematic replay that pushes the measured wheel travel and rack through
   ``DWSolver`` maps to give the camber and toe each tyre actually ran.

Assumptions are stated where they are made: dampers measure suspension travel
only (no tyre compliance), the damper-to-wheel motion ratio is an input
(``estimate`` until measured -- this tool does not model the pushrod), the
steering column is rigid. No tyre data is used anywhere (none exists yet).

Frames: ISO 8855 after the mapping's sign flips -- ay, yaw rate and steering
positive in a LEFT turn; roll positive about +X (left side up); pitch positive
nose-down; damper travel positive in compression.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline, RegularGridInterpolator

from vdcore.geometry.solver import DWSolver

from ..geometry import CarGeometry
from ..project import TelemetrySettings
from .mapping import DAMPERS
from .session import Session

G = 9.80665
# a chassis-mounted accelerometer also senses gravity through the roll angle
AY_NOTE = ("ay as logged: a chassis-mounted accelerometer reads ay·cosφ + sinφ, which biases "
           "gradients vs ay by ≈ −φ/ay (≈ −1.7 % at 1 deg/g); not corrected")


# --------------------------------------------------------------------------- #
# 1. Channel quality
# --------------------------------------------------------------------------- #

@dataclass
class ChannelQuality:
    name: str
    unit: str
    rate_hz: float
    n: int
    nan_pct: float
    vmin: float
    vmax: float
    longest_flat_s: float
    spikes: int
    saturated_pct: float
    issues: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "warn" if self.issues else "ok"


STANDSTILL_MPS = 0.5      # below this the car is parked: flat channels are expected


def channel_quality(session: Session, speed: np.ndarray | None = None) -> list[ChannelQuality]:
    """Per-channel quality on each channel's NATIVE samples (before resampling).

    ``speed`` (m/s on the session time base, e.g. the mapped speed role) lets
    the flat-line test skip standstill, where speed, throttle and brake sit
    still by design. A rest value -- 0, or 100 for a '%' channel, when it is
    also the channel's min or max (pedal released, WOT, parked) -- is neither
    a flat-line nor saturation. Spikes are counted as events (a step out and
    straight back), and flagged from 3 events. Saturation also needs the
    plateau to be entered with a step larger than the channel's
    99th-percentile step.
    """
    out = []
    rate = session.rate_hz
    moving_s = None
    if speed is not None and len(speed) == len(session.time_s):
        moving_s = ~(np.asarray(speed, dtype=float) < STANDSTILL_MPS)   # NaN speed: unknown
    for name, y_s in session.channels.items():
        y = session.native.get(name, y_s)
        ch_rate = session.rates_hz.get(name, rate)
        unit = session.units.get(name, "")
        n = len(y)
        finite = np.isfinite(y)
        nan_pct = 100.0 * (1 - finite.mean()) if n else 100.0
        issues = []
        if not finite.any():
            out.append(ChannelQuality(name, unit, ch_rate, n, 100.0,
                                      math.nan, math.nan, math.nan, 0, 0.0, ["no data"]))
            continue
        if moving_s is None:
            moving = np.ones(n, dtype=bool)
        elif y is y_s:
            moving = moving_s
        else:
            t_nat = np.arange(n) / ch_rate if ch_rate > 0 else session.time_s[:n]
            moving = np.interp(t_nat, session.time_s, moving_s.astype(float)) > 0.5
        yf = y[finite]
        vmin, vmax = float(yf.min()), float(yf.max())
        rest_vals = (0.0, 100.0) if unit.strip() == "%" else (0.0,)
        rest = [e for e in (vmin, vmax) if any(np.isclose(e, r) for r in rest_vals)]
        # flat-lines: longest run of identical consecutive values while moving,
        # not counting a rest value
        ym = y[finite & moving]
        flat = np.diff(ym) == 0
        for e in rest:
            flat &= ~np.isclose(ym[1:], e)
        runs, best, cur = 0, 0, 0
        for same in flat:
            cur = cur + 1 if same else 0
            best = max(best, cur)
        runs = best
        flat_s = runs / ch_rate if ch_rate > 0 else math.nan
        # spikes: steps far outside the channel's own step distribution
        d = np.diff(yf)
        spikes = 0
        if len(d) > 10:
            mad = float(np.median(np.abs(d - np.median(d))))
            # a spike is a step both far outside the step distribution AND a
            # sizeable fraction of the channel's range (smooth logs have a tiny
            # MAD, which alone would flag every corner entry)
            thr = max(12.0 * 1.4826 * mad, 0.25 * (vmax - vmin))
            if thr > 0:
                # an event is a big step immediately undone by a big step back
                big = np.abs(d) > thr
                last = -2
                for k in np.flatnonzero(big[:-1] & big[1:] & (d[:-1] * d[1:] < 0)):
                    if k > last + 1:            # pairs do not share a step
                        spikes += 1
                        last = k
        # saturation: a large share of samples pinned to an extreme that is not
        # a rest value, reached abruptly (a sensor rail, not a physical stop)
        sat, abrupt = 0.0, False
        if vmax > vmin:
            ad = np.abs(d)
            p99 = float(np.percentile(ad, 99)) if ad.size else 0.0
            at_lim = np.zeros(len(yf), dtype=bool)
            for ext in (vmin, vmax):
                if ext in rest:
                    continue
                at = np.isclose(yf, ext)
                at_lim |= at
                entry = np.flatnonzero(at[1:] & ~at[:-1])      # step d[k] enters the plateau
                abrupt = abrupt or bool(entry.size and np.max(ad[entry]) > p99)
            sat = 100.0 * float(at_lim.mean())
        if nan_pct > 1.0:
            issues.append(f"{nan_pct:.1f} % missing")
        if vmax == vmin:
            issues.append("constant -- sensor dead or not connected")
        elif flat_s > 2.0 and session.duration_s > 10.0:
            issues.append(f"flat for {flat_s:.1f} s" + (" while moving" if moving_s is not None
                                                        else ""))
        if spikes >= 3:
            issues.append(f"{spikes} spikes")
        if sat > 2.0 and abrupt:
            issues.append(f"{sat:.1f} % at the range limit (saturation?)")
        out.append(ChannelQuality(name, unit, ch_rate, n, nan_pct, vmin, vmax, flat_s, spikes,
                                  sat, issues))
    return out


# --------------------------------------------------------------------------- #
# Derived signals
# --------------------------------------------------------------------------- #

def _smooth(y: np.ndarray, rate: float, window_s: float = 0.2) -> np.ndarray:
    if not (math.isfinite(rate) and rate > 0):
        raise ValueError(f"invalid sample rate {rate} Hz -- check the log's time column")
    n = max(1, int(round(window_s * rate)))
    if n <= 1:
        return y
    k = np.ones(n) / n
    yy = np.where(np.isfinite(y), y, 0.0)
    w = np.convolve(np.isfinite(y).astype(float), k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.convolve(yy, k, mode="same") / w


def wheel_travel(sig: dict[str, np.ndarray], settings: TelemetrySettings,
                 zero: dict[str, float] | None = None) -> dict[str, np.ndarray]:
    """Per-corner wheel travel (mm, bump +) from damper travel and motion ratio."""
    out = {}
    for role in DAMPERS:
        if role not in sig:
            continue
        cid = role[-2:]
        mr = settings.damper_motion_ratio_front if cid[0] == "F" \
            else settings.damper_motion_ratio_rear
        if mr == 0:
            continue
        z = (zero or {}).get(role, 0.0)
        sgn = 1.0 if settings.damper_compression_positive else -1.0
        out[cid] = sgn * (sig[role] - z) / mr
    return out


def straight_line_zero(sig: dict[str, np.ndarray], settings: TelemetrySettings
                       ) -> dict[str, float]:
    """Damper reference = median over straight, steady running.

    Loggers are rarely zeroed at the design ride height; this takes the median
    where |ay| and |ax| are small and the car is moving. Note it absorbs any
    aero load at that speed.
    """
    mask = np.ones_like(next(iter(sig.values())), dtype=bool)
    if "speed" in sig:
        mask &= sig["speed"] > settings.min_speed_mps
    if "ay" in sig:
        mask &= np.abs(sig["ay"]) < 0.1
    if "ax" in sig:
        mask &= np.abs(sig["ax"]) < 0.1
    out = {}
    for role in DAMPERS:
        if role in sig:
            vals = sig[role][mask & np.isfinite(sig[role])]
            out[role] = float(np.median(vals)) if len(vals) else 0.0
    return out


def straight_line_steer_zero(sig: dict[str, np.ndarray], settings: TelemetrySettings
                             ) -> dict[str, float]:
    """Steering reference = median over straight running.

    Straight running: moving above the speed threshold with |ay| < 0.05 g and
    (if logged) |yaw rate| < 1 deg/s. Applies to ``steer_sw`` and
    ``steer_road``. A role is left out when no straight running is found, so
    an empty result means no zero could be taken (the caller must say so).
    """
    roles = [r for r in ("steer_sw", "steer_road") if r in sig]
    if not roles or "speed" not in sig or "ay" not in sig:
        return {}
    mask = (sig["speed"] > settings.min_speed_mps) & (np.abs(sig["ay"]) < 0.05)
    if "yaw_rate" in sig:
        mask &= np.abs(sig["yaw_rate"]) < 1.0
    out = {}
    for role in roles:
        vals = sig[role][mask & np.isfinite(sig[role])]
        if len(vals) >= 10:
            out[role] = float(np.median(vals))
    return out


def roll_pitch(travel: dict[str, np.ndarray], geo: CarGeometry) -> tuple[np.ndarray | None,
                                                                          np.ndarray | None,
                                                                          np.ndarray | None]:
    """(roll_front, roll_rear, pitch) in deg from wheel travel. ISO signs."""
    rf = rr = pitch = None
    if "FL" in travel and "FR" in travel:
        rf = np.degrees(np.arctan((travel["FR"] - travel["FL"]) / geo.track_mm("front")))
    if "RL" in travel and "RR" in travel:
        rr = np.degrees(np.arctan((travel["RR"] - travel["RL"]) / geo.track_mm("rear")))
    if all(c in travel for c in ("FL", "FR", "RL", "RR")):
        f = 0.5 * (travel["FL"] + travel["FR"])
        r = 0.5 * (travel["RL"] + travel["RR"])
        pitch = np.degrees(np.arctan((f - r) / geo.wheelbase_mm))
    return rf, rr, pitch


def steady_mask(sig: dict[str, np.ndarray], settings: TelemetrySettings, rate: float,
                lateral: bool = True) -> np.ndarray:
    n = len(next(iter(sig.values())))
    m = np.ones(n, dtype=bool)
    if "speed" in sig:
        m &= sig["speed"] > settings.min_speed_mps
    if lateral and "ay" in sig:
        ay = sig["ay"]
        m &= np.abs(ay) > settings.min_abs_ay_g
        jerk = np.abs(np.gradient(_smooth(ay, rate), 1.0 / rate))
        m &= jerk < 1.0                         # g/s: quasi-steady
        if "ax" in sig:
            m &= np.abs(sig["ax"]) < 0.3
    # only the signals the mask itself uses; each fit drops its own NaN, so a
    # sparse channel no analysis here reads (throttle, distance) removes nothing
    for k in ("speed", "ay", "ax"):
        if k in sig:
            m &= np.isfinite(sig[k])
    return m


@dataclass
class Fit:
    slope: float
    intercept: float
    r2: float
    n: int

    @classmethod
    def of(cls, x: np.ndarray, y: np.ndarray) -> Fit:
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 10 or np.ptp(x) < 1e-9:
            return cls(math.nan, math.nan, math.nan, int(len(x)))
        a, b = np.polyfit(x, y, 1)
        pred = a * x + b
        ss = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss if ss > 0 else math.nan
        return cls(float(a), float(b), r2, int(len(x)))


# --------------------------------------------------------------------------- #
# 2. Physics consistency checks
# --------------------------------------------------------------------------- #

@dataclass
class Check:
    name: str
    status: str            # "ok", "warn", "fail", "n/a"
    value: str
    expected: str
    detail: str


# (factor, what it means) for the yaw check: slope of yaw vs ay·g/V when one
# channel carries a common unit mix-up. Matched on |slope|; the sign is a
# separate diagnosis, so a sign error and a unit error are both named.
_YAW_SLOPE_UNITS = (
    (57.29578, 0.15, "yaw rate is in deg/s but scaled as rad/s"),
    (1 / 57.29578, 0.3, "yaw rate is in rad/s but read as deg/s"),
    (3.6, 0.11, "speed is in km/h but read as m/s"),
    (1 / 3.6, 0.15, "speed is in m/s but scaled as km/h"),
    (2.23694, 0.10, "speed is in mph but read as m/s"),
    (0.44704, 0.10, "speed is in m/s but scaled as mph"),
    (9.80665, 0.10, "ay is in g but scaled as m/s²"),
    (1 / 9.80665, 0.15, "ay is in m/s² but read as g"),
)


def _yaw_diagnosis(s: float) -> str:
    a = abs(s)
    sign = "yaw rate or ay has the opposite sign convention"
    if abs(a - 1.0) < 0.2:
        return f"slope ≈ −1: {sign}" if s < 0 else (
            "slope near 1 but outside ±0.15 or a poor fit: check the filtering, the time "
            "alignment and the speed source")
    for k, tol, what in _YAW_SLOPE_UNITS:
        if abs(a / k - 1.0) < tol:
            label = f"{k:.3g}" if k >= 1 else f"{k:.3g} (1/{1 / k:.3g})"
            msg = f"slope ≈ {'−' if s < 0 else ''}{label}: {what}"
            return msg + (f", and {sign}" if s < 0 else "")
    return "inconsistent: check the three channels' units, signs and filtering"


def consistency_checks(sig: dict[str, np.ndarray], settings: TelemetrySettings, rate: float,
                       geo: CarGeometry | None = None,
                       travel: dict[str, np.ndarray] | None = None,
                       steering_ratio: float | None = None) -> list[Check]:
    """Model-free cross-channel checks. ``steering_ratio`` (SW : road wheel)
    only converts the standstill steering offset to a road-wheel angle."""
    out: list[Check] = []
    has = sig.keys()
    yaw_status = steer_status = None
    # (a) yaw rate = ay·g/V
    if {"yaw_rate", "ay", "speed"} <= has:
        v = sig["speed"]
        m = (v > settings.min_speed_mps) & (np.abs(sig["ay"]) > 0.2)
        with np.errstate(divide="ignore", invalid="ignore"):
            ref = np.degrees(sig["ay"] * G / v)
        f = Fit.of(ref[m], sig["yaw_rate"][m])
        if not math.isfinite(f.slope):
            out.append(Check("Yaw rate vs ay·g/V", "n/a", "—", "slope 1.00",
                             "not enough cornering above the speed threshold"))
        else:
            ok = abs(f.slope - 1.0) < 0.15 and f.r2 > 0.6
            diag = "units and signs of yaw, ay and speed are mutually consistent"
            if not ok:
                diag = _yaw_diagnosis(f.slope)
            yaw_status = "ok" if ok else "fail"
            out.append(Check("Yaw rate vs ay·g/V", yaw_status,
                             f"slope {f.slope:.3f}, R² {f.r2:.3f}, n {f.n}", "slope 1.00", diag))
    # (b) steering sign
    if {"steer_sw", "ay"} <= has:
        m = steady_mask(sig, settings, rate)
        f = Fit.of(sig["ay"][m], sig["steer_sw"][m])
        st = "n/a" if not math.isfinite(f.slope) else ("ok" if f.slope > 0 else "fail")
        steer_status = st
        out.append(Check("Steering rises with ay", st,
                         f"{f.slope:.1f} deg/g" if math.isfinite(f.slope) else "—",
                         "positive", "steering and ay share a sign convention" if st == "ok"
                         else "steering sign is opposite to ay -- flip one of them"
                         if st == "fail" else "no steady cornering found"))
    # (c) roll sign from dampers, per axle (a swap on one axle only must show)
    if travel and geo is not None and "ay" in has:
        rf, rr, _ = roll_pitch(travel, geo)
        roll_checks: list[Check] = []
        m = steady_mask(sig, settings, rate)
        for axle, roll in (("front", rf), ("rear", rr)):
            if roll is None:
                continue
            f = Fit.of(sig["ay"][m], roll[m])
            st = "n/a" if not math.isfinite(f.slope) else ("ok" if f.slope > 0 else "fail")
            roll_checks.append(Check(
                f"Roll ({axle} dampers) rises with ay", st,
                f"{f.slope:.2f} deg/g" if math.isfinite(f.slope) else "—", "positive (ISO)",
                f"{axle} damper sides and compression sign are consistent" if st == "ok"
                else f"{axle} dampers: left/right swapped or extension-positive"
                if st == "fail" else "no steady cornering found"))
        if roll_checks and all(c.status == "fail" for c in roll_checks) \
                and steer_status == "ok" and yaw_status != "fail":
            # yaw and steering agree with ay, every axle disagrees: ay itself
            # (and with it yaw and steering) may be mirrored
            for c in roll_checks:
                c.detail += ("; or ay, yaw rate and steering are all in a Y-right (SAE) "
                             "frame -- flip all three")
        out.extend(roll_checks)
    # (d) static offsets
    if "speed" in has:
        still = sig["speed"] < 0.5
        if still.sum() > 0.5 * rate:
            ratio_ok = steering_ratio is not None and math.isfinite(steering_ratio) \
                and steering_ratio > 0
            for role, lim, unit in (("ay", 0.05, "g"), ("ax", 0.05, "g"),
                                    ("steer_sw", 0.5, "deg"), ("yaw_rate", 1.0, "deg/s")):
                if role in has:
                    mu = float(np.nanmean(sig[role][still]))
                    value = f"{mu:+.3f} {unit}"
                    if role == "steer_sw" and ratio_ok:
                        value += f" SW (≈ {mu / steering_ratio:+.3f} deg road wheel)"
                    out.append(Check(f"{role} at standstill", "ok" if abs(mu) <= lim else "warn",
                                     value, f"|x| ≤ {lim:g} {unit}",
                                     "zero offset; subtract it or re-zero the sensor"
                                     if abs(mu) > lim else "zeroed"))
        else:
            out.append(Check("Standstill offsets", "n/a", "—", "—",
                             "no standstill samples in this log"))
    # (e) heading / turn direction
    if "yaw_rate" in has:
        heading = float(np.nansum(sig["yaw_rate"]) / rate)
        out.append(Check("Net heading change", "n/a", f"{heading:+.0f} deg", "±360 per closed lap",
                         "positive = counter-clockwise (ISO, left turns positive). If the "
                         "track is known to run the other way, the yaw sign is flipped."))
    return out


# --------------------------------------------------------------------------- #
# 3. Gradients
# --------------------------------------------------------------------------- #

@dataclass
class GradientResult:
    name: str
    fit: Fit
    unit: str
    target: float | None
    note: str
    x: np.ndarray
    y: np.ndarray


def roll_gradients(sig: dict[str, np.ndarray], travel: dict[str, np.ndarray], geo: CarGeometry,
                   settings: TelemetrySettings, rate: float, target: float
                   ) -> list[GradientResult]:
    out = []
    if "ay" not in sig:
        return out
    m = steady_mask(sig, settings, rate)
    rf, rr, _ = roll_pitch(travel, geo)
    for name, roll, mr in (
            ("Roll gradient, front dampers", rf, settings.damper_motion_ratio_front),
            ("Roll gradient, rear dampers", rr, settings.damper_motion_ratio_rear)):
        if roll is None:
            continue
        out.append(GradientResult(
            name, Fit.of(sig["ay"][m], roll[m]), "deg/g", target,
            f"suspension roll only (no tyre deflection); motion ratio {mr:g} "
            f"({settings.motion_ratio_source}); target is design_intent; {AY_NOTE}",
            sig["ay"][m], roll[m]))
    return out


def pitch_gradient(sig: dict[str, np.ndarray], travel: dict[str, np.ndarray], geo: CarGeometry,
                   settings: TelemetrySettings, rate: float) -> list[GradientResult]:
    """Braking and traction pitch gradients, fitted separately.

    Anti-dive, anti-squat and the drive layout differ between the two, so one
    line through both would match neither (the intercept absorbs the kink).
    """
    _, _, pitch = roll_pitch(travel, geo)
    if pitch is None or "ax" not in sig:
        return []
    base = steady_mask(sig, settings, rate, lateral=False)
    if "ay" in sig:
        base &= np.abs(sig["ay"]) < 0.3
    out = []
    for name, m, note in (
            ("Pitch gradient, braking (nose-down +)", base & (sig["ax"] < -0.15),
             "ax < −0.15 g; negative slope expected: braking pitches nose-down (anti-dive, "
             "rear anti-lift)"),
            ("Pitch gradient, traction (nose-down +)", base & (sig["ax"] > 0.15),
             "ax > +0.15 g; negative slope expected: traction pitches nose-up (anti-squat)")):
        out.append(GradientResult(name, Fit.of(sig["ax"][m], pitch[m]), "deg/g", None, note,
                                  sig["ax"][m], pitch[m]))
    return out


def steer_table(geo: CarGeometry, n: int = 41) -> Callable[[np.ndarray], np.ndarray]:
    """Steering-wheel angle (deg) -> mean road-wheel steer (deg, + left), solved.

    FL and FR are solved on ``DWSolver`` at zero travel over the rack stroke;
    the road-wheel angle is the mean steer of the two from static toe
    (δ = ½·(−Δtoe_FL + Δtoe_FR), toe per side), the steering-wheel angle is
    rack / (2π·r_pinion) · 360 (rigid column, as in the replay). The returned
    callable is a cubic spline through those nodes and is NaN beyond the
    rack stroke. A non-converged solve raises -- no partial table.
    """
    st = geo.steering_inputs
    if not st.pinion_radius_mm > 0:
        raise ValueError(f"steer table: pinion radius {st.pinion_radius_mm:g} mm")
    solvers = {cid: DWSolver(geo.corner(cid)) for cid in ("FL", "FR")}
    racks = np.linspace(-st.max_rack_travel_mm, st.max_rack_travel_mm, n)

    def toe(cid: str, rack: float) -> float:
        res = solvers[cid].solve(wheel_travel_mm=0.0, rack_mm=float(rack))
        if not res.converged:
            raise RuntimeError(f"steer table: {cid} did not converge at rack {rack:+.2f} mm")
        return res.toe_deg_per_side

    l0, r0 = toe("FL", 0.0), toe("FR", 0.0)
    delta = np.array([0.5 * (-(toe("FL", r) - l0) + (toe("FR", r) - r0)) for r in racks])
    sw = racks / (2.0 * math.pi * st.pinion_radius_mm) * 360.0
    spline = CubicSpline(sw, delta)

    def sw_to_delta(sw_deg: np.ndarray) -> np.ndarray:
        x = np.asarray(sw_deg, dtype=float)
        out = np.asarray(spline(np.nan_to_num(x)), dtype=float)
        out[~np.isfinite(x) | (x < sw[0]) | (x > sw[-1])] = np.nan
        return out

    return sw_to_delta


def understeer_gradient(sig: dict[str, np.ndarray], settings: TelemetrySettings, rate: float,
                        steering_ratio: float, wheelbase_mm: float,
                        sw_to_delta: Callable[[np.ndarray], np.ndarray] | None = None,
                        steer_zero: dict[str, float] | None = None) -> GradientResult | None:
    """K = d(δ_road − L/R)/d(ay), turns folded by the sign of ay. Positive = understeer.

    δ_road from the road-wheel channel if logged; else the steering wheel
    through ``sw_to_delta`` (see ``steer_table``: SW -> rack -> solved mean
    FL/FR steer, which captures the linkage's non-linearity); else
    steering-wheel ÷ a constant ratio (on-centre, so only right at small
    angles). ``steer_zero`` is the zero already subtracted from the steering
    channels (``straight_line_steer_zero``); it only feeds the note, which
    states which conversion and which zero were used.
    Path curvature from yaw rate / speed (falls back to ay·g/V²).
    """
    if "ay" not in sig or "speed" not in sig:
        return None
    if "steer_road" in sig:
        role = "steer_road"
        delta = sig["steer_road"]
        src = "logged road-wheel angle"
    elif "steer_sw" in sig and sw_to_delta is not None:
        role = "steer_sw"
        delta = sw_to_delta(sig["steer_sw"])
        src = "SW → rack → solved mean FL/FR steer (DWSolver)"
        beyond = np.isfinite(sig["steer_sw"]) & ~np.isfinite(delta)
        if beyond.any():
            src += f"; {int(beyond.sum())} samples beyond the rack stroke excluded"
    elif "steer_sw" in sig and steering_ratio and math.isfinite(steering_ratio):
        role = "steer_sw"
        delta = sig["steer_sw"] / steering_ratio
        src = f"SW ÷ {steering_ratio:.2f} (constant ratio)"
    else:
        return None
    z = (steer_zero or {}).get(role)
    zero = (f"steering zeroed to the straight-running median ({z:+.2f} deg)" if z is not None
            else "steering as logged (no zero applied)")
    v = sig["speed"]
    L = wheelbase_mm / 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        if "yaw_rate" in sig:
            kin = np.degrees(L * np.radians(sig["yaw_rate"]) / v)
        else:
            kin = np.degrees(L * sig["ay"] * G / v ** 2)
    m = steady_mask(sig, settings, rate)
    sgn = np.sign(sig["ay"])
    x = np.abs(sig["ay"])[m]
    y = ((delta - kin) * sgn)[m]
    return GradientResult("Understeer gradient", Fit.of(x, y), "deg/g", None,
                          f"δ from {src}; {zero}; rigid column, steady state. "
                          f"Positive = understeer. {AY_NOTE}", x, y)


# --------------------------------------------------------------------------- #
# 4. Kinematic replay
# --------------------------------------------------------------------------- #

def inclination_deg(camber_deg: float, toe_deg: float) -> float:
    """Wheel inclination to the vertical (deg, sign as camber) from DWSolver's angles.

    ``DWSolver.camber_deg`` is the spin axis projected on the vehicle YZ
    plane, which for a steered wheel overstates the tilt of the wheel plane
    by ≈ 1/cos(toe) (+1.2 deg on the default FL at full lock). The spin axis
    is proportional to (tan toe, ±1, −tan camber), so the inclination
    −asin(s_z) = asin(tan γ / √(1 + tan²δ + tan²γ)). Equal to the camber at
    zero toe. (App-side: vdcore's SolverResult has no inclination field.)
    """
    tg, td = math.tan(math.radians(camber_deg)), math.tan(math.radians(toe_deg))
    return math.degrees(math.asin(tg / math.sqrt(1.0 + td * td + tg * tg)))


@dataclass
class KinematicMaps:
    """Camber/toe vs (travel[, rack]) per corner, solved once on DWSolver.

    Front "camber" is the wheel inclination (``inclination_deg``) because the
    front wheels are steered; rear camber is DWSolver's (the two differ by
    ≈ γ·(1/cos δ − 1), < 2e-4 deg for |toe| ≤ 0.5 deg and |γ| ≤ 5 deg).
    """

    travel: np.ndarray
    rack: np.ndarray
    camber: dict[str, RegularGridInterpolator]
    toe: dict[str, RegularGridInterpolator]
    failed: int


def build_maps(geo: CarGeometry, n_travel: int = 21, n_rack: int = 13, margin_mm: float = 10.0,
               cancelled=None) -> KinematicMaps:
    fi = geo.front_inputs
    ri = geo.rear_inputs
    lo = -max(fi.travel_droop_mm, ri.travel_droop_mm) - margin_mm
    hi = max(fi.travel_bump_mm, ri.travel_bump_mm) + margin_mm
    travel = np.linspace(lo, hi, n_travel)
    rmax = geo.steering_inputs.max_rack_travel_mm
    rack = np.linspace(-rmax, rmax, n_rack)
    camber, toe = {}, {}
    failed = 0
    for cid in ("FL", "FR", "RL", "RR"):
        solver = DWSolver(geo.corner(cid))
        front = cid[0] == "F"
        # rear corners have no rack: solve one column and repeat it, so both
        # axes support cubic interpolation (needs >= 4 points per axis)
        racks = rack if front else np.linspace(-1.0, 1.0, 4)
        cam = np.full((len(travel), len(racks)), np.nan)
        to = np.full_like(cam, np.nan)
        for i, t in enumerate(travel):
            cols = range(len(racks)) if front else range(1)
            for j in cols:
                if cancelled is not None and cancelled():
                    raise InterruptedError
                res = solver.solve(wheel_travel_mm=float(t),
                                   rack_mm=float(racks[j]) if front else 0.0)
                if res.converged:
                    cam[i, j] = inclination_deg(res.camber_deg, res.toe_deg_per_side) \
                        if front else res.camber_deg
                    to[i, j] = res.toe_deg_per_side
                else:
                    failed += 1
            if not front:
                cam[i, :], to[i, :] = cam[i, 0], to[i, 0]
        # a cubic spline cannot hold a failed (NaN) node; linear interpolation
        # keeps the map and turns only the cells touching it into NaN, which
        # the replay then counts as outside the solved envelope
        ok = bool(np.isfinite(cam).all() and np.isfinite(to).all())
        kw = dict(bounds_error=False, fill_value=np.nan, method="cubic" if ok else "linear")
        camber[cid] = RegularGridInterpolator((travel, racks), cam, **kw)
        toe[cid] = RegularGridInterpolator((travel, racks), to, **kw)
    return KinematicMaps(travel, rack, camber, toe, failed)


@dataclass
class ReplayResult:
    time_s: np.ndarray
    travel: dict[str, np.ndarray]
    camber_chassis: dict[str, np.ndarray]
    camber_road: dict[str, np.ndarray]
    toe: dict[str, np.ndarray]
    rack_mm: np.ndarray
    roll_deg: np.ndarray
    outside_pct: dict[str, float]
    cornering: np.ndarray


def replay(session: Session, sig: dict[str, np.ndarray], travel: dict[str, np.ndarray],
           maps: KinematicMaps, geo: CarGeometry, settings: TelemetrySettings) -> ReplayResult:
    """Measured travel + rack -> camber/toe per corner, by interpolation in the maps.

    Samples outside the solved envelope are NaN and counted, never clamped.
    Road camber = chassis camber − side·roll, roll from the dampers. Front
    corners carry the wheel inclination (see ``KinematicMaps``).
    """
    n = len(session.time_s)
    steer = geo.steering_inputs
    if "steer_sw" in sig and steer.pinion_radius_mm > 0:
        rack = sig["steer_sw"] / 360.0 * 2.0 * math.pi * steer.pinion_radius_mm
    else:
        rack = np.zeros(n)
    rf, rr, _ = roll_pitch(travel, geo)
    roll = rf if rf is not None else (rr if rr is not None else np.zeros(n))
    cam_c, cam_r, toe, outside = {}, {}, {}, {}
    for cid, t in travel.items():
        rk = rack if cid[0] == "F" else np.zeros(n)
        pts = np.column_stack([t, rk])
        cc = maps.camber[cid](pts)
        to = maps.toe[cid](pts)
        side = 1.0 if cid in ("FL", "RL") else -1.0
        r_ax = (rf if cid[0] == "F" else rr)
        r_ax = r_ax if r_ax is not None else roll
        cam_c[cid] = cc
        cam_r[cid] = cc - side * r_ax
        toe[cid] = to
        valid = np.isfinite(t)
        outside[cid] = 100.0 * float(np.mean(~np.isfinite(cc[valid]))) if valid.any() else 100.0
    corner = np.abs(sig["ay"]) > settings.min_abs_ay_g if "ay" in sig else np.zeros(n, bool)
    return ReplayResult(session.time_s, travel, cam_c, cam_r, toe, rack, roll, outside, corner)


# --------------------------------------------------------------------------- #
# 5. GG and track map
# --------------------------------------------------------------------------- #

def track_map(sig: dict[str, np.ndarray], rate: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Dead-reckoned XY (m) from speed and yaw rate. Drifts; a shape check only."""
    if "speed" not in sig or "yaw_rate" not in sig:
        return None
    dt = 1.0 / rate
    psi = np.cumsum(np.nan_to_num(np.radians(sig["yaw_rate"]))) * dt
    v = np.nan_to_num(sig["speed"])
    return np.cumsum(v * np.cos(psi)) * dt, np.cumsum(v * np.sin(psi)) * dt


@dataclass
class LltdResult(GradientResult):
    """Measured LLTD: the offset-free front share as a force and as a moment share."""

    force_share: float = math.nan       # ΔFz_f / (ΔFz_f + ΔFz_r)
    moment_share: float = math.nan      # ΔFz_f·t_f / (ΔFz_f·t_f + ΔFz_r·t_r)


def measured_lltd(sig: dict[str, np.ndarray], settings: TelemetrySettings, rate: float,
                  tracks_mm: tuple[float, float] | None = None) -> LltdResult | None:
    """Front share of lateral load transfer from logged wheel loads (if any).

    The per-axle transfer is the SLOPE of ½(F_right − F_left) against signed
    ay, so static left-right offsets (cross-weight, driver, load-cell zero) go
    into the intercepts and drop out. Force share kf/(kf+kr); moment share
    weights each axle by its track (``tracks_mm`` = (front, rear)), which is
    how ``sla_geometry`` uses ``lltd_front``. The scatter is the per-sample
    force share with the fitted static offsets removed, against |ay|.
    """
    if not all(f"fz_{c}" in sig for c in ("FL", "FR", "RL", "RR")) or "ay" not in sig:
        return None
    m = steady_mask(sig, settings, rate)
    ay = sig["ay"]
    df = 0.5 * (sig["fz_FR"] - sig["fz_FL"])
    dr = 0.5 * (sig["fz_RR"] - sig["fz_RL"])
    ff, fr = Fit.of(ay[m], df[m]), Fit.of(ay[m], dr[m])
    kf, kr = ff.slope, fr.slope
    force = moment = math.nan
    if math.isfinite(kf) and math.isfinite(kr) and abs(kf + kr) > 1e-9:
        force = kf / (kf + kr)
        if tracks_mm is not None:
            tf, tr = tracks_mm
            if abs(kf * tf + kr * tr) > 1e-9:
                moment = kf * tf / (kf * tf + kr * tr)
    df0 = df - (ff.intercept if math.isfinite(ff.intercept) else 0.0)
    dr0 = dr - (fr.intercept if math.isfinite(fr.intercept) else 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = df0 / (df0 + dr0)
    big = m & (np.abs(df0 + dr0) > 50.0)
    x = np.abs(ay[big])
    y = share[big]
    return LltdResult("Measured LLTD (front share)", Fit.of(x, y), "-", None,
                      "from logged wheel loads: slope ratio of per-axle transfer vs ay, static "
                      "L-R offsets removed; scatter slope = dependence on |ay|", x, y,
                      force, moment)
