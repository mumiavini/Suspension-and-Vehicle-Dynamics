"""The VD Studio project: every input the designer owns, in one serialisable object.

A project bundles the design inputs the repository scripts already define --
``sla_geometry.VehicleData`` / ``AxleInputs`` and
``steering_geometry.SteeringInputs`` -- with the things the scripts keep as
module constants (the rear toe link) and the things only the app needs
(hardpoint-mode geometry, concept estimator, design targets, design log, FSAE
stage tracker, telemetry settings, setup-sheet pass-through fields).

The defaults ARE the shipped 2027 configuration, read from the scripts rather
than restated, so a fresh project reproduces ``scripts/geometry_summary.py``
exactly (pinned by ``tests/app/test_engine_geometry.py``).

Frames: geometry is stored in ISO 8855 (X+ forward, Y+ LEFT, Z+ up, origin at
the front axle centreline on the ground plane). The rear toe link is stored as
the LEFT-side point; the right side is its mirror; its X values move with the
rear axle station they were declared against. Synthesis inputs keep the
scripts' DESIGN frame (x rearward, y outboard) because that is the frame the
designer edits them in -- the conversion lives in the scripts.
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

import geometry_summary as gs

import sla_geometry as sla
import steering_geometry as stg

SCHEMA_VERSION = 1
PROJECT_SUFFIX = ".vdproj.json"

Source = Literal["cad", "measured", "estimate", "design_intent"]
SOURCES: tuple[str, ...] = ("cad", "measured", "estimate", "design_intent")
GeometryMode = Literal["synthesis", "hardpoints"]

CORNERS: tuple[str, ...] = ("FL", "FR", "RL", "RR")
INPUT_POINTS: tuple[str, ...] = (
    "UCA_IN_FRONT", "UCA_IN_REAR", "UCA_OUT",
    "LCA_IN_FRONT", "LCA_IN_REAR", "LCA_OUT",
    "TIE_ROD_IN", "TIE_ROD_OUT",
    "WHEEL_CENTER",
)
Pt = tuple[float, float, float]


# --------------------------------------------------------------------------- #
# Brake / drive layout
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Layout:
    """Where the brake and drive torques react: it sets the anti-geometry reference point.

    An OUTBOARD brake (caliper on the upright) or outboard drive (hub motor,
    live axle) reacts its torque on the upright, so the longitudinal force acts
    at the CONTACT PATCH. An INBOARD brake or a chassis-mounted diff with
    driveshafts reacts it on the chassis; the patch force plus the shaft couple
    is then equivalent to the force at the WHEEL CENTRE (Milliken RCVD Ch.17).

    Defaults are the 2027 car: outboard brakes on both axles and a
    chassis-mounted diff with driveshafts (docs/GEOMETRY_AUDIT_2026-09-02_rev6.md,
    "driveshaft/diff packaging"). All three are design_intent.
    """

    front_brakes_inboard: bool = False
    rear_brakes_inboard: bool = False
    rear_drive_inboard: bool = True


# --------------------------------------------------------------------------- #
# Hardpoint-mode geometry
# --------------------------------------------------------------------------- #

@dataclass
class AxleAlignment:
    """Inputs a hardpoint file cannot carry: static camber/toe and tyre radius.

    Static camber is built into the upright and static toe is set by the tie
    rod length, so neither is recoverable from the nine input points. Both are
    design variables; the tyre loaded radius is usually the least certain
    input of all, hence its own provenance tag.
    """

    static_camber_deg: float = -1.5
    static_toe_deg_per_side: float = 0.0
    loaded_radius_mm: float = 245.0
    tire_source: Source = "estimate"
    tire_tol_mm: float = 1.0


@dataclass
class HardpointSet:
    """Directly specified hardpoints (CAD import, measurement, or a frozen design).

    ``points[corner][name] = (x, y, z)`` in ISO 8855 mm. ``sources`` and
    ``tols`` carry provenance per point, as ``vdcore.models.Hardpoint`` requires.
    """

    points: dict[str, dict[str, Pt]]
    sources: dict[str, dict[str, str]]
    tols: dict[str, dict[str, float]]
    front: AxleAlignment = field(default_factory=AxleAlignment)
    rear: AxleAlignment = field(default_factory=AxleAlignment)
    origin: str = ""

    @classmethod
    def from_points(
        cls,
        points: dict[str, dict[str, Pt]],
        *,
        source: Source = "estimate",
        tol_mm: float = 1.0,
        origin: str = "",
        front: AxleAlignment | None = None,
        rear: AxleAlignment | None = None,
    ) -> HardpointSet:
        pts = {c: {n: tuple(float(v) for v in xyz) for n, xyz in d.items()}  # type: ignore[misc]
               for c, d in points.items()}
        return cls(
            points=pts,  # type: ignore[arg-type]
            sources={c: {n: source for n in d} for c, d in pts.items()},
            tols={c: {n: tol_mm for n in d} for c, d in pts.items()},
            front=front or AxleAlignment(),
            rear=rear or AxleAlignment(),
            origin=origin,
        )

    def missing(self) -> list[str]:
        return [f"{c}/{n}" for c in CORNERS for n in INPUT_POINTS
                if n not in self.points.get(c, {})]


# --------------------------------------------------------------------------- #
# Concept stage
# --------------------------------------------------------------------------- #

@dataclass
class MassItem:
    """One row of the CG estimator: a subsystem's mass and CG location.

    ``x_mm`` is the distance BEHIND the front axle (positive rearward) and is
    optional -- the early concept table only has heights. Without every x the
    estimator reports the CG height but not the weight distribution.
    """

    name: str
    mass_kg: float
    z_mm: float
    x_mm: float | None = None
    source: Source = "estimate"


def _default_mass_items() -> list[MassItem]:
    # The subsystem table from random_calcs.ipynb (concept stage, PT-BR source).
    return [
        MassItem("Driver", 70.0, 325.0),
        MassItem("Engine / powertrain", 62.0, 275.0),
        MassItem("Chassis + bodywork", 32.0, 315.0),
        MassItem("Unsprung (4 corners)", 48.0, 240.0),
        MassItem("Fuel, systems", 18.0, 280.0),
    ]


@dataclass
class TargetEntry:
    """A literature/design-intent target with the reason it exists.

    Mirrors ``vdcore.models.target.DesignTarget`` and adds the provenance tag
    CLAUDE.md requires on every literature-derived number. Targets are
    documentation and query arguments -- never chased automatically.
    """

    name: str
    value: float
    unit: str
    tolerance: float
    event: str = "all"
    rationale: str = ""
    reference: str = ""
    source: str = "design_intent"


def _default_targets() -> list[TargetEntry]:
    return [
        TargetEntry("Front roll centre height", 35.0, "mm", 10.0, "all",
                    "Low RC keeps most lateral transfer elastic (tunable) and "
                    "jacking small; 25-45 mm band.", "Milliken RCVD Ch.17; Optimum G"),
        TargetEntry("Rear roll centre height", 55.0, "mm", 10.0, "all",
                    "Rear above front tilts the roll axis nose-down: directional "
                    "stability, more geometric transfer at the rear.", "Milliken RCVD Ch.17"),
        TargetEntry("Front FVSA length", 1500.0, "mm", 200.0, "all",
                    "Camber gain ~57.3/FVSA deg/mm recovers 60-80 % of roll over "
                    "the first 25 mm of travel without a nervous RC.", "Milliken RCVD Ch.17"),
        TargetEntry("Roll gradient", 1.0, "deg/g", 0.3, "all",
                    "0.8-1.5 deg/g: enough roll to use the camber gain, little "
                    "enough to keep camber negative on the outer tyre.",
                    "Milliken RCVD Ch.16; FSAE design reports"),
        TargetEntry("Static camber", -1.5, "deg", 0.5, "all",
                    "Offsets the +roll camber the outer tyre sees at 1.5 g.",
                    "Concept closure, random_calcs.ipynb"),
        TargetEntry("Front anti-dive", 7.5, "%", 2.5, "autocross",
                    "Mid-band of the 5-10 % the team asked for; limits pitch "
                    "under braking without killing brake feel.", "sla_geometry.FRONT_2027"),
        TargetEntry("Ackermann at 10 deg outer", 100.0, "%", 40.0, "autocross",
                    "Tight FSAE hairpins favour near-true Ackermann at low speed.",
                    "steering_geometry.STEERING_2027"),
        TargetEntry("Chassis torsion / roll stiffness", 4.0, "x", 1.0, "all",
                    "3-5x keeps the front/rear roll-stiffness split meaningful.",
                    "SAE 2002-01-3300; Optimum G seminars"),
    ]


# --------------------------------------------------------------------------- #
# FSAE workflow tracker and design log
# --------------------------------------------------------------------------- #

STAGES: tuple[tuple[str, str, str], ...] = (
    ("concept", "1. Concept & targets",
     "Mass, CG, wheelbase/track, rules, roll-stiffness budget, design targets with rationale."),
    ("geometry", "2. Suspension geometry",
     "SLA synthesis from RC/FVSA/KPI/kingpin/pickup planes; anti-geometry; member lengths."),
    ("steering", "3. Steering system",
     "Caster/trail, steering arm, rack, bump steer, Ackermann, ratio, effort."),
    ("kinematics", "4. Kinematic validation",
     "Heave/roll/steer sweeps on DWSolver, trade-off sweeps, quasi-static maneuvers."),
    ("crosscheck", "5. Independent cross-check",
     "OptimumK correlation, Altair MotionSolve replay."),
    ("cad", "6. CAD integration",
     "SolidWorks skeleton export, CAD read-back diff, frozen hardpoint set."),
    ("manufacturing", "7. Manufacturing & build",
     "Tolerances, shim steps, alignment sheet, measured hardpoints."),
    ("testing", "8. Testing & telemetry",
     "Data validation, roll/pitch/understeer gradients, kinematic replay vs design."),
    ("design_event", "9. Design Event",
     "Decision log, trade-off evidence, provenance of every number quoted."),
)
STAGE_STATUSES: tuple[str, ...] = ("not started", "in progress", "done", "blocked")


@dataclass
class StageState:
    status: str = "not started"
    notes: str = ""


@dataclass
class LogEntry:
    date: str
    stage: str
    title: str
    decision: str = ""
    rationale: str = ""
    alternatives: str = ""
    author: str = ""


# --------------------------------------------------------------------------- #
# Telemetry settings
# --------------------------------------------------------------------------- #

@dataclass
class TelemetrySettings:
    """How the team's logger maps onto physical quantities.

    ``channel_map`` maps a semantic role (``speed``, ``ay``, ``damper_FL`` ...)
    to a channel name in the loaded file. The scale factors convert the
    logger's units to the app's (m/s, g, deg, mm). Sign flips state the
    logger's convention explicitly -- ISO 8855 (Y+ LEFT, positive ay in a LEFT
    turn) is what every analysis assumes after the flip.

    ``damper_motion_ratio`` is damper travel per unit wheel travel. This tool
    does not model the pushrod/rocker, so the value is an input and is tagged
    ``estimate`` until measured.
    """

    channel_map: dict[str, str] = field(default_factory=dict)
    scale: dict[str, float] = field(default_factory=dict)
    flip_sign: dict[str, bool] = field(default_factory=dict)
    damper_motion_ratio_front: float = 1.0
    damper_motion_ratio_rear: float = 1.0
    motion_ratio_source: Source = "estimate"
    damper_compression_positive: bool = True
    steering_ratio_override: float | None = None
    min_speed_mps: float = 8.0
    min_abs_ay_g: float = 0.3


# --------------------------------------------------------------------------- #
# The project
# --------------------------------------------------------------------------- #

def _default_rear_toe_in() -> Pt:
    return tuple(gs.REAR_TOE_LINK_INBOARD)  # type: ignore[return-value]


def _default_rear_toe_out() -> Pt:
    return tuple(gs.REAR_TOE_LINK_OUTBOARD)  # type: ignore[return-value]


@dataclass
class Project:
    """Everything the designer owns. Derived values are never stored."""

    name: str = "FSAE 2027 -- PUCPR Racing"
    team: str = "PUCPR Racing #27"
    author: str = ""
    notes: str = ""
    created: str = field(default_factory=lambda: _dt.date.today().isoformat())

    vehicle: sla.VehicleData = field(default_factory=lambda: sla.VEHICLE_2027)
    front: sla.AxleInputs = field(default_factory=lambda: sla.FRONT_2027)
    rear: sla.AxleInputs = field(default_factory=lambda: sla.REAR_2027)
    steering: stg.SteeringInputs = field(default_factory=lambda: stg.STEERING_2027)

    # The rear has no rack: its toe link is a declared design input (LEFT side,
    # ISO 8855). Defaults are geometry_summary.REAR_TOE_LINK_*. Its X values
    # were declared against the rear axle station ``rear_toe_link_axle_x_mm``
    # (design frame, rearward +); synthesis shifts both ends with the axle so
    # the upright's steering-arm point follows the upright. Y and Z are absolute.
    rear_toe_link_inboard: Pt = field(default_factory=_default_rear_toe_in)
    rear_toe_link_outboard: Pt = field(default_factory=_default_rear_toe_out)
    rear_toe_link_axle_x_mm: float = sla.REAR_2027.axle_x_mm
    rear_static_toe_deg_per_side: float = 0.0
    synthesis_tol_mm: float = 0.5
    layout: Layout = field(default_factory=Layout)

    geometry_mode: GeometryMode = "synthesis"
    hardpoints: HardpointSet | None = None

    mass_items: list[MassItem] = field(default_factory=_default_mass_items)
    targets: list[TargetEntry] = field(default_factory=_default_targets)
    stages: dict[str, StageState] = field(
        default_factory=lambda: {k: StageState() for k, _, _ in STAGES})
    design_log: list[LogEntry] = field(default_factory=list)
    telemetry: TelemetrySettings = field(default_factory=TelemetrySettings)
    setup_sheet: dict[str, Any] = field(default_factory=dict)

    # -- convenience --------------------------------------------------------
    def copy(self) -> Project:
        return copy.deepcopy(self)

    def axle_inputs(self, axle: str) -> sla.AxleInputs:
        return self.front if axle == "front" else self.rear


def default_project() -> Project:
    return Project()


# --------------------------------------------------------------------------- #
# Dotted-path access (used by the editors and the what-if engine)
# --------------------------------------------------------------------------- #

def get_path(project: Project, path: str) -> Any:
    """Read ``"front.rc_height_mm"`` / ``"steering.limits.ackermann_pct"`` etc."""
    obj: Any = project
    for part in path.split("."):
        if isinstance(obj, dict):
            obj = obj[part]
        elif isinstance(obj, (list, tuple)) and part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def with_path(project: Project, path: str, value: Any) -> Project:
    """Return a copy of ``project`` with one dotted field replaced.

    The script dataclasses are frozen, so every level is rebuilt with
    ``dataclasses.replace`` rather than mutated. Tuples (bands, points) accept
    an index as the last component: ``"front.limits.rc_height_mm.0"``.
    """
    parts = path.split(".")

    def rebuild(obj: Any, i: int) -> Any:
        key = parts[i]
        last = i == len(parts) - 1
        if isinstance(obj, tuple) and key.isdigit():
            lst = list(obj)
            lst[int(key)] = value if last else rebuild(lst[int(key)], i + 1)
            return tuple(lst)
        if isinstance(obj, dict):
            new = dict(obj)
            new[key] = value if last else rebuild(obj[key], i + 1)
            return new
        child = value if last else rebuild(getattr(obj, key), i + 1)
        if isinstance(obj, Project):
            new_p = copy.copy(obj)
            setattr(new_p, key, child)
            return new_p
        return replace(obj, **{key: child})

    return rebuild(project, 0)


# --------------------------------------------------------------------------- #
# JSON (de)serialisation
# --------------------------------------------------------------------------- #

def _tuplify(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuplify(v) for v in value)
    return value


def _merge_frozen(default: Any, data: dict[str, Any]) -> Any:
    """Rebuild a (possibly frozen, nested) dataclass from ``data`` over a default.

    Unknown keys are ignored (forward compatibility); missing keys keep the
    default, so a project saved by an older version still opens.
    """
    kwargs: dict[str, Any] = {}
    names = {f.name for f in fields(default)}
    for key, val in data.items():
        if key not in names:
            continue
        cur = getattr(default, key)
        if hasattr(cur, "__dataclass_fields__") and isinstance(val, dict):
            kwargs[key] = _merge_frozen(cur, val)
        elif isinstance(cur, tuple):
            kwargs[key] = _tuplify(val)
        else:
            kwargs[key] = val
    return replace(default, **kwargs)


def project_to_dict(project: Project) -> dict[str, Any]:
    d = asdict(project)
    d["schema_version"] = SCHEMA_VERSION
    d["frame"] = "ISO 8855: X+ forward, Y+ left, Z+ up; units mm / deg"
    return d


def project_from_dict(data: dict[str, Any]) -> Project:
    version = int(data.get("schema_version", SCHEMA_VERSION))
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"Project schema {version} is newer than this VD Studio supports "
            f"({SCHEMA_VERSION}). Update the app to open it.")
    base = Project()
    p = Project(
        name=data.get("name", base.name),
        team=data.get("team", base.team),
        author=data.get("author", base.author),
        notes=data.get("notes", base.notes),
        created=data.get("created", base.created),
    )
    if "vehicle" in data:
        p.vehicle = _merge_frozen(base.vehicle, data["vehicle"])
    if "front" in data:
        p.front = _merge_frozen(base.front, data["front"])
    if "rear" in data:
        p.rear = _merge_frozen(base.rear, data["rear"])
    if "steering" in data:
        p.steering = _merge_frozen(base.steering, data["steering"])
    if "rear_toe_link_inboard" in data:
        p.rear_toe_link_inboard = tuple(float(v) for v in data["rear_toe_link_inboard"])  # type: ignore[assignment]
    if "rear_toe_link_outboard" in data:
        p.rear_toe_link_outboard = tuple(float(v) for v in data["rear_toe_link_outboard"])  # type: ignore[assignment]
    p.rear_toe_link_axle_x_mm = float(
        data.get("rear_toe_link_axle_x_mm", base.rear_toe_link_axle_x_mm))
    p.rear_static_toe_deg_per_side = float(
        data.get("rear_static_toe_deg_per_side", base.rear_static_toe_deg_per_side))
    p.synthesis_tol_mm = float(data.get("synthesis_tol_mm", base.synthesis_tol_mm))
    if isinstance(data.get("layout"), dict):
        p.layout = _merge_frozen(base.layout, data["layout"])
    p.geometry_mode = data.get("geometry_mode", "synthesis")
    hp = data.get("hardpoints")
    if hp:
        p.hardpoints = HardpointSet(
            points={c: {n: tuple(float(v) for v in xyz) for n, xyz in d.items()}  # type: ignore[misc]
                    for c, d in hp["points"].items()},
            sources=hp.get("sources", {}),
            tols=hp.get("tols", {}),
            front=AxleAlignment(**hp.get("front", {})),
            rear=AxleAlignment(**hp.get("rear", {})),
            origin=hp.get("origin", ""),
        )
    if "mass_items" in data:
        p.mass_items = [MassItem(**m) for m in data["mass_items"]]
    if "targets" in data:
        p.targets = [TargetEntry(**t) for t in data["targets"]]
    if "stages" in data:
        for k, v in data["stages"].items():
            p.stages[k] = StageState(**v)
    if "design_log" in data:
        p.design_log = [LogEntry(**e) for e in data["design_log"]]
    if "telemetry" in data:
        tel = TelemetrySettings()
        for k, v in data["telemetry"].items():
            if hasattr(tel, k):
                setattr(tel, k, v)
        p.telemetry = tel
    p.setup_sheet = dict(data.get("setup_sheet", {}))
    if p.geometry_mode == "hardpoints" and p.hardpoints is None:
        p.geometry_mode = "synthesis"
    return p


def save_project(project: Project, path: Path) -> None:
    text = json.dumps(project_to_dict(project), indent=2, ensure_ascii=False)
    Path(path).write_text(text, encoding="utf-8")


def load_project(path: Path) -> Project:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return project_from_dict(data)
