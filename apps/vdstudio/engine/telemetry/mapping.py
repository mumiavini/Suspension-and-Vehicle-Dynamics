"""Semantic channel roles, automatic mapping and unit conversion.

Every analysis works on ROLES (``speed``, ``ay``, ``damper_FL`` ...) in the
app's units, never on logger channel names. ``auto_map`` proposes a mapping
from channel names; ``guess_scale`` proposes a unit conversion from the unit
string or, failing that, from magnitude. Both are proposals shown to the user,
who owns the final mapping (stored in ``TelemetrySettings``).

App units after conversion: m/s, g, deg/s, deg, mm, m, N.
Sign convention after the per-role flip: ISO 8855 -- ``ay`` and yaw rate
positive in a LEFT turn, steering positive to the LEFT, damper travel positive
in COMPRESSION (bump).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .session import Session


@dataclass(frozen=True)
class Role:
    key: str
    label: str
    unit: str
    group: str
    pattern: str


ROLES: tuple[Role, ...] = (
    Role("speed", "Vehicle speed", "m/s", "Vehicle",
         r"(car_?v_?speed|vehicle.?speed|gps.?speed|ground.?speed|^speed$|^v$|velocidade|^vel)"),
    Role("ay", "Lateral acceleration", "g", "Vehicle",
         r"(g_?ay\b|g_?lat|lat.*acc|acc.*lat|^ay$|lateral.?g|acel.*lat)"),
    Role("ax", "Longitudinal acceleration", "g", "Vehicle",
         r"(g_?ax\b|g_?long|long.*acc|acc.*long|^ax$|inline|acel.*long)"),
    Role("yaw_rate", "Yaw rate", "deg/s", "Vehicle", r"(yaw|guinada)"),
    Role("steer_sw", "Steering-wheel angle", "deg", "Steering",
         r"(str_?a_?h\b|steer.*(angle|wheel|sw)|^swa$|volante|^steer)"),
    Role("steer_road", "Road-wheel steer angle (if logged)", "deg", "Steering",
         r"(str_?a_?whl|road.?wheel.?angle)"),
    Role("damper_FL", "Damper travel FL", "mm", "Suspension",
         r"((spr|damp|shock|susp?|pot|amort).*(_|\b)(fl|lf|de)\b)"),
    Role("damper_FR", "Damper travel FR", "mm", "Suspension",
         r"((spr|damp|shock|susp?|pot|amort).*(_|\b)(fr|rf|dd)\b)"),
    Role("damper_RL", "Damper travel RL", "mm", "Suspension",
         r"((spr|damp|shock|susp?|pot|amort).*(_|\b)(rl|lr|te)\b)"),
    Role("damper_RR", "Damper travel RR", "mm", "Suspension",
         r"((spr|damp|shock|susp?|pot|amort).*(_|\b)(rr|td)\b)"),
    Role("ride_front", "Ride height front", "mm", "Suspension",
         r"(rhl?_?x?_?fa|ride.*front|front.*ride|rh_?f\b)"),
    Role("ride_rear", "Ride height rear", "mm", "Suspension",
         r"(rhl?_?x?_?ra|ride.*rear|rear.*ride|rh_?r\b)"),
    Role("fz_FL", "Wheel load FL (if logged)", "N", "Loads", r"(fz_?fl|load_?fl)"),
    Role("fz_FR", "Wheel load FR (if logged)", "N", "Loads", r"(fz_?fr|load_?fr)"),
    Role("fz_RL", "Wheel load RL (if logged)", "N", "Loads", r"(fz_?rl|load_?rl)"),
    Role("fz_RR", "Wheel load RR (if logged)", "N", "Loads", r"(fz_?rr|load_?rr)"),
    Role("distance", "Distance", "m", "Other", r"(dist)"),
    Role("throttle", "Throttle", "%", "Other", r"(throttle|tps|^thr|acelerador)"),
    Role("brake", "Brake pressure", "bar", "Other", r"(brk_?p|brake.*press|freio)"),
)
ROLE_BY_KEY = {r.key: r for r in ROLES}
DAMPERS = ("damper_FL", "damper_FR", "damper_RL", "damper_RR")


def auto_map(names: list[str]) -> dict[str, str]:
    """Propose ``role -> channel``. Each channel is used at most once."""
    used: set[str] = set()
    out: dict[str, str] = {}
    for role in ROLES:
        rx = re.compile(role.pattern, re.I)
        hits = [n for n in names if n not in used and rx.search(n)]
        if not hits:
            continue
        # prefer the shortest match (e.g. 'CAR_G_AY' over 'CAR_G_AYX_FILTERED')
        best = sorted(hits, key=len)[0]
        out[role.key] = best
        used.add(best)
    return out


_UNIT_SCALE = {
    "speed": {"km/h": 1 / 3.6, "kph": 1 / 3.6, "kmh": 1 / 3.6, "mph": 0.44704, "m/s": 1.0},
    "ay": {"m/s^2": 1 / 9.80665, "m/s2": 1 / 9.80665, "m/s²": 1 / 9.80665, "g": 1.0},
    "ax": {"m/s^2": 1 / 9.80665, "m/s2": 1 / 9.80665, "m/s²": 1 / 9.80665, "g": 1.0},
    "yaw_rate": {"rad/s": 57.29578, "deg/s": 1.0, "°/s": 1.0},
    "steer_sw": {"rad": 57.29578, "deg": 1.0, "°": 1.0},
    "steer_road": {"rad": 57.29578, "deg": 1.0, "°": 1.0},
    "distance": {"km": 1000.0, "m": 1.0},
}
_LENGTH = {"m": 1000.0, "cm": 10.0, "mm": 1.0, "in": 25.4}


def guess_scale(role: str, unit: str, data: np.ndarray | None) -> tuple[float, str]:
    """(factor to app units, reason). Unit string first, magnitude second."""
    u = (unit or "").strip().lower()
    table = _UNIT_SCALE.get(role, {})
    if role.startswith(("damper", "ride")) and u in _LENGTH:
        return _LENGTH[u], f"unit '{unit}'"
    if u in table:
        return table[u], f"unit '{unit}'"
    if data is None or not np.isfinite(data).any():
        return 1.0, "no data"
    peak = float(np.nanmax(np.abs(data)))
    if role == "speed" and peak > 60.0:
        return 1 / 3.6, f"no unit; peak {peak:.0f} > 60 -> assumed km/h"
    if role in ("ay", "ax") and peak > 6.0:
        return 1 / 9.80665, f"no unit; peak {peak:.1f} > 6 -> assumed m/s²"
    if role == "yaw_rate" and peak < 3.0:
        return 57.29578, f"no unit; peak {peak:.2f} < 3 -> assumed rad/s"
    if role in ("steer_sw", "steer_road") and peak < 3.5:
        return 57.29578, f"no unit; peak {peak:.2f} < 3.5 -> assumed rad"
    if role.startswith(("damper", "ride")) and peak < 0.3:
        return 1000.0, f"no unit; peak {peak:.3f} < 0.3 -> assumed m"
    return 1.0, "no unit; magnitude consistent with app units"


def signals(session: Session, channel_map: dict[str, str], scale: dict[str, float],
            flip: dict[str, bool]) -> dict[str, np.ndarray]:
    """Mapped roles in app units with the per-role sign flips applied."""
    out: dict[str, np.ndarray] = {}
    for role, ch in channel_map.items():
        arr = session.get(ch)
        if arr is None:
            continue
        k = float(scale.get(role, 1.0))
        out[role] = arr * k * (-1.0 if flip.get(role, False) else 1.0)
    return out
