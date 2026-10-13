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
         r"(g_?ay\b|g_?lat|g.?force.?lat|lat.*acc|acc.*lat|^ay$|lateral.?g|acel.*lat)"),
    Role("ax", "Longitudinal acceleration", "g", "Vehicle",
         r"(g_?ax\b|g_?long|g.?force.?long|long.*acc|lon.?acc|acc.*long|^ax$|inline"
         r"|acel.*long)"),
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
    Role("fz_FL", "Wheel load FL (if logged)", "N", "Loads", r"((fz|load)[ _]?fl(?![a-z]))"),
    Role("fz_FR", "Wheel load FR (if logged)", "N", "Loads", r"((fz|load)[ _]?fr(?![a-z]))"),
    Role("fz_RL", "Wheel load RL (if logged)", "N", "Loads", r"((fz|load)[ _]?rl(?![a-z]))"),
    Role("fz_RR", "Wheel load RR (if logged)", "N", "Loads", r"((fz|load)[ _]?rr(?![a-z]))"),
    Role("distance", "Distance", "m", "Other", r"(dist)"),
    Role("throttle", "Throttle", "%", "Other", r"(throttle|tps|^thr|acelerador)"),
    Role("brake", "Brake pressure", "bar", "Other", r"(brk_?p|brake.?pres|freio)"),
)
ROLE_BY_KEY = {r.key: r for r in ROLES}
DAMPERS = ("damper_FL", "damper_FR", "damper_RL", "damper_RR")

# look-alike channels a role's pattern also matches (steering torque or rate,
# damper velocity or load); 'travel' is not a velocity
_STEER_NOT = r"(torque|rate|(?<!tra)vel|speed|force)"
_DAMPER_NOT = r"((?<!tra)vel|speed|rate|force|load)"
_EXCLUDE = {"steer_sw": _STEER_NOT, "steer_road": _STEER_NOT,
            **{d: _DAMPER_NOT for d in DAMPERS}}


def auto_map(names: list[str], review: dict[str, list[str]] | None = None) -> dict[str, str]:
    """Propose ``role -> channel``. Each channel is used at most once.

    Candidates matching the role's exclusion pattern (steering torque or rate,
    damper velocity ...) are skipped. When several candidates remain and
    ``review`` is given, the ones not chosen are put in ``review[role]`` so the
    UI can ask the user to check the pick.
    """
    used: set[str] = set()
    out: dict[str, str] = {}
    for role in ROLES:
        rx = re.compile(role.pattern, re.I)
        nx = re.compile(_EXCLUDE[role.key], re.I) if role.key in _EXCLUDE else None
        hits = [n for n in names if n not in used and rx.search(n)
                and not (nx is not None and nx.search(n))]
        if not hits:
            continue
        # prefer the shortest match (e.g. 'CAR_G_AY' over 'CAR_G_AYX_FILTERED'),
        # and an IMU channel over a GPS-derived one
        ranked = sorted(hits, key=lambda n: ("gps" in n.lower(), len(n)))
        best = ranked[0]
        out[role.key] = best
        used.add(best)
        if review is not None and len(ranked) > 1:
            review[role.key] = ranked[1:]
    return out


_UNIT_SCALE = {
    "speed": {"km/h": 1 / 3.6, "kph": 1 / 3.6, "kmh": 1 / 3.6, "mph": 0.44704, "m/s": 1.0},
    "ay": {"m/s^2": 1 / 9.80665, "m/s2": 1 / 9.80665, "m/s²": 1 / 9.80665, "g": 1.0},
    "ax": {"m/s^2": 1 / 9.80665, "m/s2": 1 / 9.80665, "m/s²": 1 / 9.80665, "g": 1.0},
    "yaw_rate": {"rad/s": 57.29578, "deg/s": 1.0, "°/s": 1.0},
    "steer_sw": {"rad": 57.29578, "deg": 1.0, "°": 1.0},
    "steer_road": {"rad": 57.29578, "deg": 1.0, "°": 1.0},
    "distance": {"km": 1000.0, "m": 1.0, "cm": 0.01, "mm": 0.001},
    **{f"fz_{c}": {"n": 1.0, "kn": 1000.0} for c in ("FL", "FR", "RL", "RR")},
    "throttle": {"%": 1.0},
    "brake": {"bar": 1.0, "psi": 0.0689476, "kpa": 0.01, "mpa": 10.0},
}
_LENGTH = {"m": 1000.0, "cm": 10.0, "mm": 1.0, "in": 25.4}
# spellings loggers use for the same unit -> the key used in the tables
_UNIT_ALIAS = {
    "km/hr": "km/h", "kmh": "km/h", "kph": "km/h", "kmph": "km/h", "m/sec": "m/s",
    "deg/sec": "deg/s", "°/sec": "deg/s", "dps": "deg/s", "degrees/s": "deg/s",
    "rad/sec": "rad/s",
    "degrees": "deg", "degree": "deg", "degs": "deg", "radians": "rad",
    "m/s/s": "m/s²", "m/sec²": "m/s²", "m/sec^2": "m/s²", "m/sec2": "m/s²",
}
# physical dimension of known units, to refuse a channel of the wrong kind
_DIM = {
    **dict.fromkeys(("m/s", "km/h", "mph", "mm/s", "ft/s"), "velocity"),
    **dict.fromkeys(("g", "m/s²", "m/s^2", "m/s2"), "acceleration"),
    **dict.fromkeys(("deg/s", "rad/s", "°/s"), "angular rate"),
    **dict.fromkeys(("deg", "rad", "°"), "angle"),
    **dict.fromkeys(("m", "cm", "mm", "in", "km"), "length"),
    **dict.fromkeys(("n", "kn", "lbf"), "force"),
    **dict.fromkeys(("nm", "n.m", "n·m", "n m", "kgf.m"), "torque"),
    **dict.fromkeys(("v", "mv"), "voltage"),
    **dict.fromkeys(("bar", "psi", "kpa", "mpa"), "pressure"),
    "%": "percentage",
}


def _role_dim(role: str) -> str | None:
    if role.startswith(("damper", "ride")) or role == "distance":
        return "length"
    if role.startswith("fz_"):
        return "force"
    return {"speed": "velocity", "ay": "acceleration", "ax": "acceleration",
            "yaw_rate": "angular rate", "steer_sw": "angle", "steer_road": "angle",
            "throttle": "percentage", "brake": "pressure"}.get(role)


def guess_scale(role: str, unit: str, data: np.ndarray | None) -> tuple[float, str]:
    """(factor to app units, reason). Unit string first, magnitude second.

    Common spellings are normalised (km/hr, deg/sec, degrees, m/s/s ...). A
    unit of the wrong dimension for the role (Nm on a steering angle, mm/s on
    a damper) gets no conversion and a reason that says so; an unknown unit
    falls back to the magnitude guess and the reason says the unit is unknown.
    """
    u = (unit or "").strip().lower()
    u = _UNIT_ALIAS.get(u, u)
    table = _UNIT_SCALE.get(role, {})
    if role.startswith(("damper", "ride")) and u in _LENGTH:
        return _LENGTH[u], f"unit '{unit}'"
    if u in table:
        return table[u], f"unit '{unit}'"
    dim, udim = _role_dim(role), _DIM.get(u)
    if u and udim is not None and dim is not None and udim != dim:
        return 1.0, (f"unit '{unit}' measures {udim}, this role needs {dim} -- wrong channel "
                     "or uncalibrated? no conversion applied")
    why = f"unknown unit '{unit}' -- verify" if u else "no unit"
    if data is None or not np.isfinite(data).any():
        return 1.0, f"{why}; no data"
    peak = float(np.nanmax(np.abs(data)))
    # an FSAE car never reaches 40 m/s (144 km/h); km/h logs rarely peak below 40
    if role == "speed" and peak > 40.0:
        return 1 / 3.6, f"{why}; peak {peak:.0f} > 40 -> assumed km/h"
    if role in ("ay", "ax") and peak > 6.0:
        return 1 / 9.80665, f"{why}; peak {peak:.1f} > 6 -> assumed m/s²"
    if role == "yaw_rate" and peak < 3.0:
        return 57.29578, f"{why}; peak {peak:.2f} < 3 -> assumed rad/s"
    if role in ("steer_sw", "steer_road") and peak < 3.5:
        return 57.29578, f"{why}; peak {peak:.2f} < 3.5 -> assumed rad"
    if role.startswith(("damper", "ride")) and peak < 0.3:
        return 1000.0, f"{why}; peak {peak:.3f} < 0.3 -> assumed m"
    return 1.0, f"{why}; magnitude consistent with app units"


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
