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
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from vdcore.geometry.solver import DWSolver

from ..geometry import CarGeometry
from ..project import TelemetrySettings
from .mapping import DAMPERS
from .session import Session

G = 9.80665


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


def channel_quality(session: Session) -> list[ChannelQuality]:
    out = []
    rate = session.rate_hz
    for name, y in session.channels.items():
        n = len(y)
        finite = np.isfinite(y)
        nan_pct = 100.0 * (1 - finite.mean()) if n else 100.0
        issues = []
        if not finite.any():
            out.append(ChannelQuality(name, session.units.get(name, ""), rate, n, 100.0,
                                      math.nan, math.nan, math.nan, 0, 0.0, ["no data"]))
            continue
        yf = y[finite]
        vmin, vmax = float(yf.min()), float(yf.max())
        # flat-lines: longest run of identical consecutive values
        d = np.diff(yf)
        runs, best, cur = 0, 0, 0
        for same in (d == 0):
            cur = cur + 1 if same else 0
            best = max(best, cur)
        runs = best
        flat_s = runs / rate if rate > 0 else math.nan
        # spikes: steps far outside the channel's own step distribution
        spikes = 0
        if len(d) > 10:
            mad = float(np.median(np.abs(d - np.median(d))))
            # a spike is a step both far outside the step distribution AND a
            # sizeable fraction of the channel's range (smooth logs have a tiny
            # MAD, which alone would flag every corner entry)
            thr = max(12.0 * 1.4826 * mad, 0.25 * (vmax - vmin))
            if thr > 0:
                spikes = int(np.sum(np.abs(d) > thr))
        # saturation: a large share of samples pinned to the extreme value
        sat = 0.0
        if vmax > vmin:
            at_ext = np.isclose(yf, vmax) | np.isclose(yf, vmin)
            sat = 100.0 * float(at_ext.mean())
        if nan_pct > 1.0:
            issues.append(f"{nan_pct:.1f} % missing")
        if vmax == vmin:
            issues.append("constant -- sensor dead or not connected")
        elif flat_s > 2.0 and session.duration_s > 10.0:
            issues.append(f"flat for {flat_s:.1f} s")
        if spikes > max(3, 0.002 * n):
            issues.append(f"{spikes} spikes")
        if sat > 2.0 and vmax > vmin:
            issues.append(f"{sat:.1f} % at the range limit (saturation?)")
        out.append(ChannelQuality(name, session.units.get(name, ""), session.rates_hz.get(
            name, rate), n, nan_pct, vmin, vmax, flat_s, spikes, sat, issues))
    return out


# --------------------------------------------------------------------------- #
# Derived signals
# --------------------------------------------------------------------------- #

def _smooth(y: np.ndarray, rate: float, window_s: float = 0.2) -> np.ndarray:
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
    return m & np.all([np.isfinite(v) for v in sig.values()], axis=0)


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


def consistency_checks(sig: dict[str, np.ndarray], settings: TelemetrySettings, rate: float,
                       geo: CarGeometry | None = None,
                       travel: dict[str, np.ndarray] | None = None) -> list[Check]:
    out: list[Check] = []
    has = sig.keys()
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
                s = f.slope
                if abs(s + 1.0) < 0.2:
                    diag = "slope ≈ −1: yaw rate or ay has the opposite sign convention"
                elif abs(s - 57.3) < 8 or abs(s - 1 / 57.3) < 0.005:
                    diag = "slope ≈ 57 or 1/57: a rad vs deg mix-up on the yaw rate"
                elif abs(s - 3.6) < 0.4 or abs(s - 1 / 3.6) < 0.05:
                    diag = "slope ≈ 3.6 or 1/3.6: speed is in km/h vs m/s"
                elif abs(abs(s) - 9.81) < 1.0 or abs(abs(s) - 0.102) < 0.02:
                    diag = "slope ≈ 9.8: ay is in m/s² vs g"
                else:
                    diag = "inconsistent: check the three channels' units, signs and filtering"
            out.append(Check("Yaw rate vs ay·g/V", "ok" if ok else "fail",
                             f"slope {f.slope:.3f}, R² {f.r2:.3f}, n {f.n}", "slope 1.00", diag))
    # (b) steering sign
    if {"steer_sw", "ay"} <= has:
        m = steady_mask(sig, settings, rate)
        f = Fit.of(sig["ay"][m], sig["steer_sw"][m])
        st = "n/a" if not math.isfinite(f.slope) else ("ok" if f.slope > 0 else "fail")
        out.append(Check("Steering rises with ay", st,
                         f"{f.slope:.1f} deg/g" if math.isfinite(f.slope) else "—",
                         "positive", "steering and ay share a sign convention" if st == "ok"
                         else "steering sign is opposite to ay -- flip one of them"
                         if st == "fail" else "no steady cornering found"))
    # (c) roll sign from dampers
    if travel and geo is not None and "ay" in has:
        rf, rr, _ = roll_pitch(travel, geo)
        roll = rf if rf is not None else rr
        if roll is not None:
            m = steady_mask(sig, settings, rate)
            f = Fit.of(sig["ay"][m], roll[m])
            st = "n/a" if not math.isfinite(f.slope) else ("ok" if f.slope > 0 else "fail")
            out.append(Check("Roll (dampers) rises with ay", st,
                             f"{f.slope:.2f} deg/g" if math.isfinite(f.slope) else "—",
                             "positive (ISO)",
                             "damper sides and compression sign are consistent" if st == "ok"
                             else "left/right swapped or extension-positive dampers"
                             if st == "fail" else "no steady cornering found"))
    # (d) static offsets
    if "speed" in has:
        still = sig["speed"] < 0.5
        if still.sum() > 0.5 * rate:
            for role, lim, unit in (("ay", 0.05, "g"), ("ax", 0.05, "g"),
                                    ("steer_sw", 3.0, "deg"), ("yaw_rate", 1.0, "deg/s")):
                if role in has:
                    mu = float(np.nanmean(sig[role][still]))
                    out.append(Check(f"{role} at standstill", "ok" if abs(mu) <= lim else "warn",
                                     f"{mu:+.3f} {unit}", f"|x| ≤ {lim:g} {unit}",
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
            f"({settings.motion_ratio_source}); target is design_intent",
            sig["ay"][m], roll[m]))
    return out


def pitch_gradient(sig: dict[str, np.ndarray], travel: dict[str, np.ndarray], geo: CarGeometry,
                   settings: TelemetrySettings, rate: float) -> GradientResult | None:
    _, _, pitch = roll_pitch(travel, geo)
    if pitch is None or "ax" not in sig:
        return None
    m = steady_mask(sig, settings, rate, lateral=False) & (np.abs(sig["ax"]) > 0.15)
    if "ay" in sig:
        m &= np.abs(sig["ay"]) < 0.3
    return GradientResult("Pitch gradient (nose-down +)", Fit.of(sig["ax"][m], pitch[m]),
                          "deg/g", None,
                          "negative slope expected: braking (ax < 0) pitches nose-down",
                          sig["ax"][m], pitch[m])


def understeer_gradient(sig: dict[str, np.ndarray], settings: TelemetrySettings, rate: float,
                        steering_ratio: float, wheelbase_mm: float) -> GradientResult | None:
    """K = d(δ_road − L/R)/d(ay), turns folded by the sign of ay. Positive = understeer.

    δ_road from the road-wheel channel if logged, else steering-wheel ÷ ratio.
    Path curvature from yaw rate / speed (falls back to ay·g/V²).
    """
    if "ay" not in sig or "speed" not in sig:
        return None
    if "steer_road" in sig:
        delta = sig["steer_road"]
        src = "logged road-wheel angle"
    elif "steer_sw" in sig and steering_ratio and math.isfinite(steering_ratio):
        delta = sig["steer_sw"] / steering_ratio
        src = f"steering wheel ÷ {steering_ratio:.2f}"
    else:
        return None
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
                          f"δ from {src}; rigid column, steady state, small angles. "
                          "Positive = understeer.", x, y)


# --------------------------------------------------------------------------- #
# 4. Kinematic replay
# --------------------------------------------------------------------------- #

@dataclass
class KinematicMaps:
    """Camber/toe vs (travel[, rack]) per corner, solved once on DWSolver."""

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
                    cam[i, j], to[i, j] = res.camber_deg, res.toe_deg_per_side
                else:
                    failed += 1
            if not front:
                cam[i, :], to[i, :] = cam[i, 0], to[i, 0]
        kw = dict(bounds_error=False, fill_value=np.nan, method="cubic")
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
    Road camber = chassis camber − side·roll, roll from the dampers.
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


def measured_lltd(sig: dict[str, np.ndarray], settings: TelemetrySettings, rate: float
                  ) -> GradientResult | None:
    """Front share of lateral load transfer from logged wheel loads (if any)."""
    if not all(f"fz_{c}" in sig for c in ("FL", "FR", "RL", "RR")) or "ay" not in sig:
        return None
    m = steady_mask(sig, settings, rate)
    df = 0.5 * (sig["fz_FR"] - sig["fz_FL"])
    dr = 0.5 * (sig["fz_RR"] - sig["fz_RL"])
    with np.errstate(divide="ignore", invalid="ignore"):
        share = df / (df + dr)
    big = m & (np.abs(df + dr) > 50.0)
    x = np.abs(sig["ay"][big])
    y = share[big]
    return GradientResult("Measured LLTD (front share)", Fit.of(x, y), "-", None,
                          "from logged wheel loads; intercept is the LLTD, slope its "
                          "dependence on ay", x, y)
