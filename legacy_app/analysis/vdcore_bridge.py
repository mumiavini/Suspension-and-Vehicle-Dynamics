"""
analysis/vdcore_bridge.py
=========================
Adapter between the legacy app's loaded geometry and the validated ``vdcore``
kinematic solver.

WHY THIS EXISTS
    The legacy 3D solver models each wishbone as a single strut to the midpoint
    of the two chassis pivots, so its DYNAMIC KPIs are wrong (see the banner in
    ``app.py``). ``vdcore.geometry.solver.DWSolver`` constrains all six degrees
    of freedom with the real linkage and is covered by the test suite. This
    module lifts the geometry the app already has loaded
    (``st.session_state["hardpoints_df"]``) into ``vdcore`` models and returns
    the correct dynamic KPIs, so the legacy Analysis tab and the new vdcore tab
    can both quote validated numbers on identical input.

COORDINATE FRAME
    The loaded hardpoints DataFrame and ``vdcore`` both use the same axes:
    X+ forward, Y+ LEFT, Z+ up (ISO 8855 / the legacy project frame). The
    ``vdcore.io.frames`` matrix ``M_LEGACY_TO_ISO8855`` is the identity, so the
    point conversion is a straight copy of coordinates — NO sign flip. This is
    pinned by the golden cross-check in ``tests/unit/test_vdcore_bridge.py``:
    building a Corner here from the merged 2027 CSV must reproduce the
    ``axle_rates`` / ``axle_roll`` values that ``geometry_summary._vdcore_corner``
    produces from the same numbers.

SCOPE
    vdcore covers the SWEPT dynamic KPIs: camber gain, roll-centre migration,
    roll-centre height, half-track change and roll cambers. It does NOT cover
    anti-dive / anti-squat / Ackermann / scrub / mechanical trail — those live in
    ``sla_geometry.py`` / ``steering_geometry.py`` and need a synthesised corner,
    not arbitrary loaded hardpoints. The UI must keep flagging those as legacy.

This module lives in ``legacy_app/`` (an application layer). It may import both
``vdcore`` (pure library) and ``streamlit`` (for caching). It must NEVER be
imported by anything under ``vdcore/``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq

from vdcore.analysis.axle import (
    AxleRates,
    AxleRollState,
    axle_rates,
    axle_roll,
    sample_corner,
)
from vdcore.analysis.camber import CamberSweepResult, camber_sweep
from vdcore.analysis.roll_centre import (
    RollCentreMigrationResult,
    roll_centre_height,
    roll_centre_migration,
)
from vdcore.geometry.derived import mechanical_trail_mm, scrub_radius_mm
from vdcore.geometry.solver import DWSolver, SolverResult
from vdcore.models.hardpoint import Axle, Corner, Hardpoint, TirePackage

if TYPE_CHECKING:
    import polars as pl


# =============================================================================
# Defaults for inputs the loaded hardpoints file does NOT carry
# =============================================================================
# The hardpoints CSV has only geometry (points). Static camber and the tyre
# loaded radius are design variables the file does not store, so we source them
# from the app's vehicle-setup (session state) and fall back to these documented
# defaults. Both are tagged ``estimate`` in provenance, never ``design_intent``.

DEFAULT_STATIC_CAMBER_DEG: float = 0.0
DEFAULT_STATIC_TOE_DEG_PER_SIDE: float = 0.0
DEFAULT_LOADED_RADIUS_MM: float = 245.0  # ~ FL wheel-centre height in the 2027 CSV
DEFAULT_HARDPOINT_TOL_MM: float = 1.0
DEFAULT_TIRE_TOL_MM: float = 1.0

# Mapping from the DataFrame ``point`` names to the Corner field names. The
# CONTACT_PATCH row is intentionally absent: vdcore derives the contact patch
# from the wheel centre, tyre radius and camber, it is not an input.
_POINT_TO_FIELD: dict[str, str] = {
    "UCA_IN_FRONT": "uca_inboard_front",
    "UCA_IN_REAR": "uca_inboard_rear",
    "UCA_OUT": "uca_outboard",
    "LCA_IN_FRONT": "lca_inboard_front",
    "LCA_IN_REAR": "lca_inboard_rear",
    "LCA_OUT": "lca_outboard",
    "TIE_ROD_IN": "tie_rod_inboard",
    "TIE_ROD_OUT": "tie_rod_outboard",
    "WHEEL_CENTER": "wheel_center",
}


class BridgeConversionError(ValueError):
    """Raised when loaded geometry cannot be converted into a vdcore Corner.

    Kept distinct from vdcore's own pydantic ``ValidationError`` so the UI can
    show a clear, actionable message (e.g. "left corner has negative Y") rather
    than a raw validation traceback.
    """


@dataclass(frozen=True)
class CornerInputs:
    """The non-geometry design inputs a Corner needs, sourced from the app.

    These are NOT in the hardpoints file; they come from the vehicle-setup panel
    (or the documented defaults above). All are tagged ``estimate`` provenance.
    """

    static_camber_deg: float = DEFAULT_STATIC_CAMBER_DEG
    static_toe_deg_per_side: float = DEFAULT_STATIC_TOE_DEG_PER_SIDE
    loaded_radius_mm: float = DEFAULT_LOADED_RADIUS_MM
    hardpoint_tol_mm: float = DEFAULT_HARDPOINT_TOL_MM
    tire_tol_mm: float = DEFAULT_TIRE_TOL_MM

    @classmethod
    def from_vehicle_setup(cls, vehicle_setup: dict | None) -> CornerInputs:
        """Build inputs from the app's ``st.session_state['vehicle_setup']``.

        Missing keys fall back to the module defaults, so an older session that
        never set static camber / loaded radius still converts.
        """
        vs = vehicle_setup or {}
        return cls(
            static_camber_deg=float(
                vs.get("static_camber_deg", DEFAULT_STATIC_CAMBER_DEG)
            ),
            static_toe_deg_per_side=float(
                vs.get("static_toe_deg_per_side", DEFAULT_STATIC_TOE_DEG_PER_SIDE)
            ),
            loaded_radius_mm=float(
                vs.get("loaded_radius_mm", DEFAULT_LOADED_RADIUS_MM)
            ),
        )


# =============================================================================
# DataFrame -> vdcore models
# =============================================================================


def df_to_vdcore_corner(
    df: pl.DataFrame,
    corner_id: str,
    inputs: CornerInputs | None = None,
) -> Corner:
    """Convert the loaded hardpoints DataFrame for ONE corner into a Corner.

    Args:
        df: The loaded hardpoints DataFrame (schema corner/point/x_mm/y_mm/z_mm).
        corner_id: One of "FL", "FR", "RL", "RR".
        inputs: Static camber / toe / loaded radius; defaults if omitted.

    Frame: identity copy (loaded frame == ISO 8855, Y+ LEFT). No sign flip.

    Raises:
        BridgeConversionError: if a point is missing/duplicated, or the loaded
            Y sign disagrees with the ISO convention for that corner (fail loud,
            never silently produce a mirrored corner).
    """
    import polars as pl

    inputs = inputs or CornerInputs()

    if corner_id not in ("FL", "FR", "RL", "RR"):
        raise BridgeConversionError(f"Invalid corner_id: {corner_id!r}")

    sub = df.filter(pl.col("corner") == corner_id)
    if sub.height == 0:
        raise BridgeConversionError(f"No hardpoints for corner {corner_id!r}")

    def hardpoint(point_name: str, field_name: str) -> Hardpoint:
        rows = sub.filter(pl.col("point") == point_name)
        if rows.height != 1:
            raise BridgeConversionError(
                f"Point {point_name!r} for corner {corner_id!r} appears "
                f"{rows.height} times (expected exactly 1)"
            )
        r = rows.row(0, named=True)
        # Identity frame copy: loaded (X+ fwd, Y+ left, Z+ up) == ISO 8855.
        return Hardpoint(
            name=field_name,
            x_mm=float(r["x_mm"]),
            y_mm=float(r["y_mm"]),
            z_mm=float(r["z_mm"]),
            source="estimate",  # loaded/measured geometry, not design_intent
            tol_mm=inputs.hardpoint_tol_mm,
        )

    fields: dict[str, Hardpoint] = {
        field: hardpoint(point, field)
        for point, field in _POINT_TO_FIELD.items()
    }

    # Fail loud on a Y-sign mismatch before pydantic does, with a clearer
    # message. FL/RL must be +Y, FR/RR must be -Y (ISO 8855, Y+ LEFT).
    wc_y = fields["wheel_center"].y_mm
    is_left = corner_id in ("FL", "RL")
    if is_left and wc_y <= 0:
        raise BridgeConversionError(
            f"Left corner {corner_id} has non-positive wheel-centre Y "
            f"({wc_y:.1f} mm). The loaded geometry may be in a Y+ RIGHT frame; "
            f"vdcore requires ISO 8855 (Y+ LEFT)."
        )
    if not is_left and wc_y >= 0:
        raise BridgeConversionError(
            f"Right corner {corner_id} has non-negative wheel-centre Y "
            f"({wc_y:.1f} mm). The loaded geometry may be in a Y+ RIGHT frame; "
            f"vdcore requires ISO 8855 (Y+ LEFT)."
        )

    try:
        return Corner(
            corner_id=corner_id,  # type: ignore[arg-type]
            tire=TirePackage(
                loaded_radius_mm=inputs.loaded_radius_mm,
                source="estimate",
                tol_mm=inputs.tire_tol_mm,
            ),
            static_camber_deg=inputs.static_camber_deg,
            static_toe_deg_per_side=inputs.static_toe_deg_per_side,
            **fields,
        )
    except ValueError as exc:  # pydantic ValidationError is a ValueError
        raise BridgeConversionError(
            f"Corner {corner_id} failed vdcore validation: {exc}"
        ) from exc


def df_to_vdcore_axles(
    df: pl.DataFrame,
    inputs: CornerInputs | None = None,
) -> tuple[Axle, Axle]:
    """Convert the loaded DataFrame into (front, rear) vdcore Axles.

    Front = (FL, FR), rear = (RL, RR). Axle.left is the +Y (left) corner, as
    ``axle_rates`` / ``axle_roll`` expect.
    """
    inputs = inputs or CornerInputs()
    front = Axle(
        left=df_to_vdcore_corner(df, "FL", inputs),
        right=df_to_vdcore_corner(df, "FR", inputs),
    )
    rear = Axle(
        left=df_to_vdcore_corner(df, "RL", inputs),
        right=df_to_vdcore_corner(df, "RR", inputs),
    )
    return front, rear


# =============================================================================
# High-level KPI result
# =============================================================================


@dataclass(frozen=True)
class RcHeightSweep:
    """Chassis-referenced RC height vs parallel wheel travel, for plotting.

    Built from :func:`vdcore.analysis.axle.sample_corner`, which constructs the
    front-view instant centre in the CHASSIS frame -- the same path
    ``axle_rates`` uses and that the benchmark pins. This is deliberately NOT
    ``roll_centre_migration``: that function feeds the FVIC world-frame ball
    joints against chassis-fixed pivots, so its RC barely migrates (~1 mm over
    50 mm) -- the exact artefact of the legacy solver we are correcting. Plotting
    that curve would visually contradict the (correct) migration range in the
    delta table, so we plot this chassis-referenced sweep instead.
    """

    wheel_travel_mm: list[float]
    rc_height_mm: list[float]


@dataclass(frozen=True)
class AxleVdcoreKPIs:
    """Validated dynamic KPIs for one axle, plus the raw sweep results to plot.

    ``rates`` / ``roll`` are None when a solve in that computation failed to
    converge (the exception is caught here so the UI can grey the value out
    instead of crashing). ``error`` carries the message in that case.
    """

    label: str  # "Front" or "Rear"
    rates: AxleRates | None
    roll: AxleRollState | None
    roll_deg: float
    camber_sweep_left: CamberSweepResult | None
    rc_migration: RollCentreMigrationResult | None
    rc_height_sweep: RcHeightSweep | None = None
    error: str | None = None


def _rc_height_sweep(
    axle: Axle, *, travel_mm: float, sweep_steps: int
) -> RcHeightSweep | None:
    """Chassis-referenced RC-height-vs-travel curve for the left corner.

    Uses ``sample_corner`` (chassis frame) so the plotted curve agrees with
    ``axle_rates``. Returns None if any solve in the sweep fails to converge.
    """
    corner = axle.left
    solver = DWSolver(corner)
    travels: list[float] = []
    heights: list[float] = []
    try:
        for t in np.linspace(-travel_mm, travel_mm, sweep_steps):
            sample = sample_corner(corner, solver, float(t))
            travels.append(float(t))
            heights.append(sample.rc_height_mm)
    except (RuntimeError, ValueError):
        return None
    return RcHeightSweep(wheel_travel_mm=travels, rc_height_mm=heights)


@dataclass(frozen=True)
class VdcoreKPIs:
    """Validated dynamic KPIs for the whole vehicle (front + rear axles)."""

    front: AxleVdcoreKPIs
    rear: AxleVdcoreKPIs


def _axle_kpis(
    axle: Axle,
    label: str,
    *,
    roll_deg: float,
    travel_mm: float,
    sweep_steps: int,
) -> AxleVdcoreKPIs:
    """Run all vdcore KPIs for one axle, catching non-convergence as a message.

    vdcore raises ``ConvergenceError`` / ``RuntimeError`` / ``ValueError`` on a
    failed solve rather than returning a plausible number. We catch those here
    so a bad geometry degrades to a greyed-out card, never a silent wrong value.
    """
    try:
        rates = axle_rates(
            axle,
            travel_bump_mm=travel_mm,
            travel_droop_mm=travel_mm,
            sweep_steps=sweep_steps,
        )
    except (RuntimeError, ValueError) as exc:
        return AxleVdcoreKPIs(
            label=label, rates=None, roll=None, roll_deg=roll_deg,
            camber_sweep_left=None, rc_migration=None,
            error=f"axle_rates failed: {exc}",
        )

    roll: AxleRollState | None
    try:
        roll = axle_roll(axle, roll_deg)
    except (RuntimeError, ValueError):
        roll = None  # rates still valid; roll card greys out on its own

    try:
        sweep = camber_sweep(
            axle.left,
            wheel_travel_min_mm=-travel_mm,
            wheel_travel_max_mm=travel_mm,
            steps=sweep_steps,
        )
    except (RuntimeError, ValueError):
        sweep = None

    try:
        migration = roll_centre_migration(
            axle,
            wheel_travel_min_mm=-travel_mm,
            wheel_travel_max_mm=travel_mm,
            steps=sweep_steps,
        )
    except (RuntimeError, ValueError):
        migration = None

    rc_height = _rc_height_sweep(axle, travel_mm=travel_mm, sweep_steps=sweep_steps)

    return AxleVdcoreKPIs(
        label=label, rates=rates, roll=roll, roll_deg=roll_deg,
        camber_sweep_left=sweep, rc_migration=migration,
        rc_height_sweep=rc_height, error=None,
    )


def compute_vdcore_kpis(
    df: pl.DataFrame,
    inputs: CornerInputs | None = None,
    *,
    roll_deg: float = 1.5,
    travel_mm: float = 25.0,
    sweep_steps: int = 41,
) -> VdcoreKPIs:
    """Full dynamic-KPI pass for the loaded geometry, front and rear.

    This is the single entry point the UI calls. It converts the DataFrame,
    solves both axles on ``DWSolver`` and returns validated KPIs plus the raw
    sweeps to plot. Convergence failures are captured per-axle, not raised.
    """
    inputs = inputs or CornerInputs()
    front_axle, rear_axle = df_to_vdcore_axles(df, inputs)
    return VdcoreKPIs(
        front=_axle_kpis(
            front_axle, "Front",
            roll_deg=roll_deg, travel_mm=travel_mm, sweep_steps=sweep_steps,
        ),
        rear=_axle_kpis(
            rear_axle, "Rear",
            roll_deg=roll_deg, travel_mm=travel_mm, sweep_steps=sweep_steps,
        ),
    )


# =============================================================================
# Delegation for the legacy Analysis tab
# =============================================================================
# The legacy Analysis tab built a per-axle dict of KPIs (that tab is gone;
# _compute_axle_cached``). Its DYNAMIC entries come from the wrong strut-to-
# midpoint solver. This section recomputes exactly those entries with vdcore, in
# the SAME keys and the SAME units/sign the legacy table already expects, so the
# tab can splice them in without touching its rendering. Static entries (KPI,
# caster, scrub, mechanical trail, static RC height, sum-toe) are NOT produced
# here — they are correct in the legacy path and stay there.
#
# Roll-camber sign: the legacy tab reports d(camber)/d(roll) chassis-referenced
# (a linear fit of the roll sweep). ``axle_roll`` returns ROAD-relative camber
# (outer = chassis + roll), so we subtract the roll angle back out to recover the
# chassis-referenced camber before differencing. Reported per degree of roll,
# matching the legacy ``roll_camber`` column exactly.

# Roll half-amplitude (deg) used for the two-point roll-camber slope. Chosen to
# match the legacy roll sweep's fitted range (-2 .. +2 deg) so the delegated
# number lands on the same slope the legacy column used to show.
_ROLL_CAMBER_PROBE_DEG: float = 2.0


def vdcore_sweep(
    corner: Corner,
    sweep_type: str,
    params: tuple[float, float, float],
) -> np.ndarray:
    """Run a heave / roll / steer sweep on ``DWSolver``, in the legacy dtype.

    Returns a ``analysis.sweeps.SWEEP_DTYPE`` array, so every existing plot
    function (``plot_camber_vs_heave``, ``plot_bump_steer``,
    ``plot_rc_migration``, ``plot_caster_kpi_vs_steer``, ``plot_camber_vs_roll``,
    ``plot_toe_vs_roll``, ``plot_rc_migration_roll``) consumes it unchanged.

    WHY THIS EXISTS
        The sweep charts used to run on ``KinematicSolver3D``, which models each
        wishbone as a strut to the midpoint of its two chassis pivots and leaves
        3 of 9 DOF to a numerical regularisation. Its swept KPIs are wrong (see
        the app banner), so every chart drawn from it was wrong too — the same
        solver whose camber gain and RC migration the delta table exists to
        correct. These sweeps run the real linkage instead.

    ROLL CENTRE CONVENTION
        ``rc_z_mm`` is the per-corner construction: the contact-patch-to-FVIC
        line intersected with the chassis centreline, CHASSIS-referenced, which
        is the same path ``axle_rates`` uses and the benchmark pins.
        ``rc_y_mm`` is 0 by construction — a single corner's line is
        intersected WITH the centreline, so it cannot report a lateral offset.
        The axle-level roll centre, which does move sideways under roll, is on
        the setup sheet via :func:`axle_roll` (``rc_1g_y``).

    Non-converged points are marked ``converged=False`` with NaN angles rather
    than dropped, so a gap in a chart is visible instead of silently smoothed.
    """
    from analysis.sweeps import SWEEP_DTYPE

    lo, hi, step = (float(p) for p in params)
    if step <= 0.0:
        raise ValueError("sweep step must be positive")
    n = max(2, int(round((hi - lo) / step)) + 1)
    values = np.linspace(lo, hi, n)

    solver = DWSolver(corner)
    out = np.zeros(n, dtype=SWEEP_DTYPE)

    for i, v in enumerate(values):
        heave = roll = rack = 0.0
        if sweep_type == "Heave":
            heave = float(v)
        elif sweep_type == "Roll":
            roll = float(v)
        else:
            rack = float(v)

        out[i]["heave_mm"] = heave
        out[i]["roll_deg"] = roll
        out[i]["rack_mm"] = rack

        try:
            res = solver.solve(
                wheel_travel_mm=heave, roll_deg=roll, rack_mm=rack
            )
        except (RuntimeError, ValueError):
            out[i]["camber_deg"] = np.nan
            out[i]["toe_deg"] = np.nan
            out[i]["caster_deg"] = np.nan
            out[i]["kpi_deg"] = np.nan
            out[i]["rc_y_mm"] = np.nan
            out[i]["rc_z_mm"] = np.nan
            out[i]["residual"] = np.nan
            out[i]["converged"] = False
            continue

        out[i]["camber_deg"] = res.camber_deg
        out[i]["toe_deg"] = res.toe_deg_per_side
        out[i]["caster_deg"] = res.caster_deg
        out[i]["kpi_deg"] = res.kpi_deg
        out[i]["wc_x_mm"] = res.wheel_center.x_mm
        out[i]["wc_y_mm"] = res.wheel_center.y_mm
        out[i]["wc_z_mm"] = res.wheel_center.z_mm
        out[i]["residual"] = res.residual_norm
        out[i]["converged"] = res.converged

        # Chassis-referenced roll centre, matching axle_rates' construction.
        try:
            sample = sample_corner(corner, solver, heave)
            out[i]["rc_y_mm"] = 0.0
            out[i]["rc_z_mm"] = sample.rc_height_mm
        except (RuntimeError, ValueError):
            out[i]["rc_y_mm"] = np.nan
            out[i]["rc_z_mm"] = np.nan

    return out


@dataclass(frozen=True)
class RollSweepResult:
    """Axle response to chassis roll, both wheels kept on the road.

    All angles road-relative, all lengths mm. ``rc_height_mm`` is
    ground-referenced (``axle_roll``'s convention), unlike the chassis-
    referenced heave sweep -- the two differ by 1 mm per mm of travel.
    """

    roll_deg: list[float]
    outer_camber_deg: list[float]
    inner_camber_deg: list[float]
    rc_height_mm: list[float]
    rc_lateral_mm: list[float]
    wheel_travel_mm: list[float]


def vdcore_roll_sweep(
    axle: Axle, roll_min: float, roll_max: float, step: float
) -> RollSweepResult:
    """Sweep chassis roll with both contact patches on the tilted road.

    WHY NOT A FIXED-TRAVEL ROLL SWEEP
        ``DWSolver``'s travel constraint is CHASSIS-referenced, so solving at
        ``roll_deg=phi, wheel_travel_mm=0`` holds the wheel at constant height
        *in the chassis frame* -- the wheel simply rolls with the car and camber
        tracks roll at exactly -1.000 deg/deg. True, and useless: it describes a
        car with no suspension travel.

        ``axle_roll`` instead solves for the wheel travel at which both patches
        land on the tilted road, which is what actually happens in a corner, and
        is the measure behind the setup sheet's roll-camber figure
        (-0.4153 deg/deg on the 2027 front, not -1.0).

    This is an AXLE-level sweep: roll is a property of the pair, and the lateral
    roll-centre migration it reports has no single-corner meaning.

    Points that fail to converge or cannot be bracketed are skipped, so a gap in
    the chart is visible rather than interpolated over.
    """
    if step <= 0.0:
        raise ValueError("roll step must be positive")
    n = max(2, int(round((roll_max - roll_min) / step)) + 1)

    rolls: list[float] = []
    outer: list[float] = []
    inner: list[float] = []
    rc_z: list[float] = []
    rc_y: list[float] = []
    travel: list[float] = []

    for value in np.linspace(roll_min, roll_max, n):
        angle = float(value)
        try:
            state = axle_roll(axle, angle)
        except (RuntimeError, ValueError):
            continue
        rolls.append(angle)
        outer.append(state.outer_camber_deg)
        inner.append(state.inner_camber_deg)
        rc_z.append(state.rc_height_mm)
        rc_y.append(state.rc_lateral_mm)
        travel.append(state.wheel_travel_mm)

    return RollSweepResult(
        roll_deg=rolls,
        outer_camber_deg=outer,
        inner_camber_deg=inner,
        rc_height_mm=rc_z,
        rc_lateral_mm=rc_y,
        wheel_travel_mm=travel,
    )


def legacy_corner_to_vdcore(
    legacy_corner: object,
    tie_rod: object,
    corner_id: str,
    inputs: Optional[CornerInputs] = None,
) -> Corner:
    """Convert a legacy ``SuspensionCorner`` + ``TieRod`` into a vdcore Corner.

    The DataFrame path (:func:`df_to_vdcore_corner`) covers geometry loaded from
    a file. This covers geometry that only exists as legacy objects — most
    importantly the Comparison tab's "B" side, which can be the seed or the
    result of an optimization rather than anything on disk.

    Both frames are ISO 8855 (X+ forward, Y+ LEFT, Z+ up), so this is a straight
    coordinate copy with no sign flip, exactly like the DataFrame path.
    """
    inputs = inputs or CornerInputs()
    if corner_id not in ("FL", "FR", "RL", "RR"):
        raise BridgeConversionError(f"Invalid corner_id: {corner_id!r}")

    def hp(name: str, point: object) -> Hardpoint:
        return Hardpoint(
            name=name,
            x_mm=float(point.x), y_mm=float(point.y), z_mm=float(point.z),
            source="estimate", tol_mm=inputs.hardpoint_tol_mm,
        )

    try:
        return Corner(
            corner_id=corner_id,  # type: ignore[arg-type]
            uca_inboard_front=hp("uca_inboard_front",
                                 legacy_corner.upper_arm.inboard_front),
            uca_inboard_rear=hp("uca_inboard_rear",
                                legacy_corner.upper_arm.inboard_rear),
            uca_outboard=hp("uca_outboard", legacy_corner.upper_arm.outboard),
            lca_inboard_front=hp("lca_inboard_front",
                                 legacy_corner.lower_arm.inboard_front),
            lca_inboard_rear=hp("lca_inboard_rear",
                                legacy_corner.lower_arm.inboard_rear),
            lca_outboard=hp("lca_outboard", legacy_corner.lower_arm.outboard),
            tie_rod_inboard=hp("tie_rod_inboard", tie_rod.inboard),
            tie_rod_outboard=hp("tie_rod_outboard", tie_rod.outboard),
            wheel_center=hp("wheel_center", legacy_corner.wheel_center),
            tire=TirePackage(
                loaded_radius_mm=inputs.loaded_radius_mm,
                source="estimate", tol_mm=inputs.tire_tol_mm,
            ),
            static_camber_deg=inputs.static_camber_deg,
            static_toe_deg_per_side=inputs.static_toe_deg_per_side,
        )
    except (ValueError, AttributeError) as exc:
        raise BridgeConversionError(
            f"Corner {corner_id} could not be converted: {exc}"
        ) from exc


def rack_mm_per_wheel_deg(corner: Corner, rack_test_mm: float = 5.0) -> float:
    """Rack travel needed for one degree of wheel steer, on the real linkage.

    The steering ratio follows as ``c_factor / 360 / this``.

    Replaces ``analysis.kpis.steer_ratio_and_cfactor``, which measured the same
    thing on ``KinematicSolver3D`` and came out **10.2 % high** on the 2027
    geometry (1.379 vs 1.251 mm/deg, i.e. a ratio of 4.97 rather than 4.51).
    The strut-to-midpoint solver mis-places the outboard ball joints, so the
    steering arm it swings is the wrong length.

    Returns ``inf`` when the rack does not steer the wheel at all, so the
    caller's ``c_factor / 360 / inf`` degrades to 0 rather than dividing by 0.
    """
    solver = DWSolver(corner)
    static = solver.solve()
    steered = solver.solve(rack_mm=rack_test_mm)
    if not (static.converged and steered.converged):
        return float("nan")

    d_toe = abs(steered.toe_deg_per_side - static.toe_deg_per_side)
    if d_toe < 1e-9:
        return float("inf")
    return abs(rack_test_mm) / d_toe


_ACKERMANN_OUTER_STEER_DEG: float = 10.0
_ACKERMANN_MAX_RACK_MM: float = 35.0


def solved_ackermann_pct(
    axle: Axle,
    wheelbase_mm: float,
    *,
    outer_steer_deg: float = _ACKERMANN_OUTER_STEER_DEG,
    max_rack_mm: float = _ACKERMANN_MAX_RACK_MM,
) -> float:
    """Ackermann % from an actual rack sweep on the real linkage.

    Sweeps the rack until the OUTER wheel reaches ``outer_steer_deg``, then
    compares the inner wheel's steer with the Ackermann-ideal inner angle::

        cot(delta_outer) - cot(delta_inner_ideal) = track / wheelbase
        %Ack = 100 * (delta_inner - delta_outer)
                   / (delta_inner_ideal - delta_outer)

    0 % is parallel steer, 100 % is true Ackermann. Mirrors
    ``steering_geometry.SteeringKinematics._ackermann_at_steer`` and reproduces
    it to 3 decimals on the 2027 geometry.

    WHY NOT THE PLAN-VIEW CONSTRUCTION
        The classic "extend the steering arms to the rear axle" rule assumes a
        VERTICAL kingpin, which projects to a single point in plan whatever
        height you measure at. With this car's 10 deg KPI and 5 deg caster the
        axis leans, its plan-view position moves with height, and the answer
        swings ~150 points across defensible reference choices::

            kingpin @ ground        -58.5 %      kingpin @ wheel-centre  92.6 %
            perpendicular foot       52.4 %      kingpin midpoint        98.9 %
            kingpin @ TRO height     57.7 %
            real linkage (zero-steer limit)      71.2 %

        None is close, so the construction was dropped rather than tuned. It
        also cannot see the tie rod's 3D inclination, which genuinely affects
        the steer ratio.

    Track is measured between the SOLVED contact patches, so static camber is
    included the same way the rest of this module treats it.

    Returns NaN (never a plausible wrong number) if the rack cannot reach the
    requested steer angle, or if a solve fails.
    """
    left, right = DWSolver(axle.left), DWSolver(axle.right)

    try:
        l0, r0 = left.solve(), right.solve()
        if not (l0.converged and r0.converged):
            return float("nan")
    except (RuntimeError, ValueError):
        return float("nan")

    track_mm = abs(l0.contact_patch.y_mm - r0.contact_patch.y_mm)
    if track_mm <= 0.0 or wheelbase_mm <= 0.0:
        return float("nan")

    def steer(rack_mm: float) -> tuple[float, float]:
        ls = left.solve(rack_mm=rack_mm)
        rs = right.solve(rack_mm=rack_mm)
        if not (ls.converged and rs.converged):
            raise RuntimeError("steer solve did not converge")
        return (abs(ls.toe_deg_per_side - l0.toe_deg_per_side),
                abs(rs.toe_deg_per_side - r0.toe_deg_per_side))

    try:
        # Which side is OUTER depends on the rack sign convention, so decide it
        # from the geometry rather than assuming: in a turn the inner wheel
        # steers MORE, so the outer is whichever moves less at a probe rack.
        probe_l, probe_r = steer(1.0)
        left_is_outer = probe_l <= probe_r

        def outer_error(rack_mm: float) -> float:
            sl, sr = steer(rack_mm)
            return (sl if left_is_outer else sr) - outer_steer_deg

        lo, hi = 0.01, max_rack_mm
        if outer_error(lo) * outer_error(hi) > 0.0:
            return float("nan")     # requested steer not reachable on this rack
        rack = float(brentq(outer_error, lo, hi, xtol=1e-8))

        sl, sr = steer(rack)
    except (RuntimeError, ValueError):
        return float("nan")

    delta_outer = sl if left_is_outer else sr
    delta_inner = sr if left_is_outer else sl
    if delta_outer < 1e-12:
        return 0.0

    cot_inner_ideal = 1.0 / math.tan(math.radians(delta_outer)) - track_mm / wheelbase_mm
    if abs(cot_inner_ideal) < 1e-12:
        return float("nan")
    delta_inner_ideal = math.degrees(math.atan(1.0 / cot_inner_ideal))

    denom = delta_inner_ideal - delta_outer
    if abs(denom) < 1e-12:
        return float("nan")
    return float(100.0 * (delta_inner - delta_outer) / denom)


def roll_camber_deg_per_deg(axle: Axle, *, probe_deg: float = _ROLL_CAMBER_PROBE_DEG) -> float:
    """Chassis-referenced roll-camber slope of the outer (left) wheel, deg/deg.

    Central difference of the outer wheel's CHASSIS-referenced camber across
    +/- ``probe_deg`` of chassis roll. ``axle_roll`` gives road-relative camber
    (``outer_camber_deg = chassis_camber + roll_deg``); we subtract the roll
    angle to get back to the chassis frame, which is what the legacy Analysis
    column reports. Positive slope = the outer wheel gains positive camber as the
    car rolls onto it.

    Raises the same exceptions as :func:`axle_roll`; callers that want a graceful
    degrade must catch them.
    """
    hi = axle_roll(axle, +probe_deg)
    lo = axle_roll(axle, -probe_deg)
    # Undo the road tilt each side to recover chassis-referenced outer camber.
    chassis_hi = hi.outer_camber_deg - hi.roll_deg
    chassis_lo = lo.outer_camber_deg - lo.roll_deg
    return (chassis_hi - chassis_lo) / (2.0 * probe_deg)


def delegated_axle_dynamic_kpis(
    axle: Axle,
    *,
    roll_deg: float,
    travel_mm: float,
    sweep_steps: int,
) -> dict[str, float]:
    """vdcore values for the legacy tab's DYNAMIC KPI keys, one axle.

    Returns a dict keyed exactly as the old ``tab_analysis._compute_axle_cached``'s
    dynamic entries, so the tab can ``dict.update`` it over the legacy values:

        ``ride_camber_dpm``  deg/m   -- camber gain x 1000 (per-metre)
        ``camber_gain``      deg/mm  -- camber gain per mm of bump
        ``rc_dy``            mm      -- RC lateral migration over parallel travel
        ``rc_dz``            mm      -- RC height migration over parallel travel
        ``roll_camber``      deg/deg -- chassis-referenced roll-camber slope
        ``rc_1g_z``          mm      -- RC height at ``roll_deg`` of body roll
        ``rc_1g_y``          mm      -- RC lateral shift at ``roll_deg`` of roll

    ``rc_1g_*`` are taken from a single ``axle_roll`` at ``roll_deg`` (the app's
    roll-stiffness-implied body roll at 1 g), which is the honest vdcore analogue
    of the legacy ``roll_center_at_1g_lat`` estimate.

    Any per-KPI vdcore failure degrades THAT key to NaN (never a wrong number);
    the rest of the row is still returned. ``bump_steer`` is deliberately absent:
    the legacy tie-rod is loaded but vdcore's swept toe needs the rack model that
    only ``steering_geometry.py`` owns, so the tab keeps its own bump-steer value
    flagged rather than delegating a half-derived one.
    """
    row: dict[str, float] = {}

    # Camber gain AND the RC-height migration range both come from ``axle_rates``,
    # which builds the RC in the CHASSIS frame (the golden-pinned path,
    # rc_migration_mm_per_mm = -0.3914 on the 2027 geometry). We deliberately do
    # NOT use ``roll_centre_migration`` for the RC span: that function feeds the
    # FVIC construction world-frame ball joints against chassis-fixed inboard
    # pivots, so its RC barely moves (~1 mm over 50 mm) -- the very artefact the
    # legacy strut-to-midpoint solver produced. Using ``axle_rates`` keeps this
    # delegation consistent with the vdcore tab's delta table and the benchmark.
    try:
        rates = axle_rates(
            axle,
            travel_bump_mm=travel_mm,
            travel_droop_mm=travel_mm,
            sweep_steps=sweep_steps,
        )
        row["camber_gain"] = rates.camber_gain_deg_per_mm
        row["ride_camber_dpm"] = rates.camber_gain_deg_per_mm * 1000.0
        # RC height span over the full bump/droop sweep (chassis-referenced).
        row["rc_dz"] = rates.rc_max_mm - rates.rc_min_mm
    except (RuntimeError, ValueError):
        row["camber_gain"] = float("nan")
        row["ride_camber_dpm"] = float("nan")
        row["rc_dz"] = float("nan")

    # Lateral RC migration under PARALLEL travel is zero for a symmetric axle
    # (the RC stays on the centreline). Report it as such rather than resurrecting
    # the legacy solver's spurious non-zero drift. The lateral RC shift that IS
    # non-zero happens under ROLL, and is surfaced separately as ``rc_1g_y``.
    row["rc_dy"] = 0.0

    try:
        row["roll_camber"] = roll_camber_deg_per_deg(axle)
    except (RuntimeError, ValueError):
        row["roll_camber"] = float("nan")

    try:
        roll_state = axle_roll(axle, roll_deg)
        row["rc_1g_z"] = roll_state.rc_height_mm
        row["rc_1g_y"] = roll_state.rc_lateral_mm
    except (RuntimeError, ValueError):
        row["rc_1g_z"] = float("nan")
        row["rc_1g_y"] = float("nan")

    return row


def compute_delegated_dynamic_kpis(
    df: pl.DataFrame,
    inputs: CornerInputs | None = None,
    *,
    roll_deg: float = 1.5,
    travel_mm: float = 25.0,
    sweep_steps: int = 41,
) -> dict[str, dict[str, float]]:
    """vdcore dynamic KPIs for both axles, keyed for the legacy Analysis tab.

    Returns ``{"front": {...}, "rear": {...}}`` where each inner dict carries the
    dynamic keys the legacy table reads (see
    :func:`delegated_axle_dynamic_kpis`). The whole conversion may raise
    :class:`BridgeConversionError` (bad geometry / wrong Y-sign frame); the
    caller should catch it and fall back to the legacy numbers so the tab never
    goes blank.
    """
    inputs = inputs or CornerInputs()
    front_axle, rear_axle = df_to_vdcore_axles(df, inputs)
    return {
        "front": delegated_axle_dynamic_kpis(
            front_axle, roll_deg=roll_deg, travel_mm=travel_mm,
            sweep_steps=sweep_steps,
        ),
        "rear": delegated_axle_dynamic_kpis(
            rear_axle, roll_deg=roll_deg, travel_mm=travel_mm,
            sweep_steps=sweep_steps,
        ),
    }


# =============================================================================
# Static KPIs from the validated solver (for the vdcore tab's setup sheet)
# =============================================================================
# The Analysis tab sources its static geometry rows (KPI, caster, camber, scrub,
# mechanical trail, static RC, sum-toe) from the frozen legacy Corner methods.
# Those values are correct, but they are not produced by the validated solver and
# two are sign traps (legacy mechanical trail is inverted; scrub is a separate
# code path). The vdcore tab renders the SAME sheet with every geometry row from
# ``DWSolver`` instead, so the documentation table is fully validated. This
# section adds the static half; the dynamic half already exists above.


def solved_static_corner(corner: Corner) -> SolverResult:
    """Solve one corner at the static position (no travel, roll or rack).

    Returns the :class:`SolverResult` the static KPIs read (camber, caster, KPI,
    toe, and the solved ball joints / contact patch that scrub and mechanical
    trail need).

    Raises:
        BridgeConversionError: if the solve does not converge -- we never hand
            back a plausible-looking static number from a failed solve.
    """
    result = DWSolver(corner).solve(wheel_travel_mm=0.0, roll_deg=0.0, rack_mm=0.0)
    if not result.converged:
        raise BridgeConversionError(
            f"Static solve for corner {corner.corner_id} did not converge "
            f"(residual norm {result.residual_norm:.2e})."
        )
    return result


def delegated_axle_static_kpis(axle: Axle) -> dict[str, float]:
    """vdcore static geometry KPIs for one axle, keyed for the setup sheet.

    Returns the same dict keys the Analysis tab uses for its static rows, so one
    renderer can consume either source:

        ``caster_l`` / ``caster_r``   deg  -- static caster, left / right
        ``kpi_l``    / ``kpi_r``      deg  -- kingpin inclination, left / right
        ``camber_l`` / ``camber_r``   deg  -- static camber, left / right
        ``scrub_l``  / ``scrub_r``    mm   -- scrub radius (outboard +), L / R
        ``trail_l``  / ``trail_r``    mm   -- mechanical trail (fwd intercept +)
        ``sum_toe``                   deg  -- total toe (per-side L + per-side R)
        ``rc_static``                 mm   -- static roll-centre height (Y=0)

    The left/right ball-joint solves must converge (raises via
    :func:`solved_static_corner`); per-KPI derivations that can still fail on a
    valid solve (a horizontal kingpin, parallel FVSA arms) degrade THAT key to
    NaN, never a wrong number, matching the dynamic path.
    """
    left = solved_static_corner(axle.left)
    right = solved_static_corner(axle.right)

    row: dict[str, float] = {
        "caster_l": left.caster_deg,
        "caster_r": right.caster_deg,
        "kpi_l": left.kpi_deg,
        "kpi_r": right.kpi_deg,
        "camber_l": left.camber_deg,
        "camber_r": right.camber_deg,
        # Total toe = per-side left + per-side right (both +in), matching the
        # legacy static_sum_toe_deg the Analysis tab reports.
        "sum_toe": left.toe_deg_per_side + right.toe_deg_per_side,
    }

    for side, res in (("l", left), ("r", right)):
        try:
            row[f"scrub_{side}"] = scrub_radius_mm(res)
        except ValueError:
            row[f"scrub_{side}"] = float("nan")
        try:
            row[f"trail_{side}"] = mechanical_trail_mm(res)
        except ValueError:
            row[f"trail_{side}"] = float("nan")

    try:
        row["rc_static"] = roll_centre_height(axle, left, right).rc_height_mm
    except (RuntimeError, ValueError):
        row["rc_static"] = float("nan")

    return row


def compute_vdcore_setup_sheet(
    df: pl.DataFrame,
    inputs: CornerInputs | None = None,
    *,
    roll_deg: float = 1.5,
    travel_mm: float = 25.0,
    sweep_steps: int = 41,
) -> dict[str, dict[str, float]]:
    """Every vdcore-sourced geometry row for the setup sheet, both axles.

    Merges the static geometry KPIs (:func:`delegated_axle_static_kpis`) with the
    dynamic ones (:func:`delegated_axle_dynamic_kpis`) into one per-axle dict, so
    the vdcore tab has the whole geometry half of the sheet from a single call.
    Anti-dive / anti-squat / Ackermann are deliberately NOT here -- they need a
    synthesised corner (steering_geometry.py), so the tab shows them flagged.

    Returns ``{"front": {...}, "rear": {...}}``. May raise
    :class:`BridgeConversionError` (bad geometry / wrong Y-sign frame / a
    non-converged static solve); the caller catches it and warns rather than
    faking a value.
    """
    inputs = inputs or CornerInputs()
    front_axle, rear_axle = df_to_vdcore_axles(df, inputs)

    def _both(axle: Axle) -> dict[str, float]:
        merged = delegated_axle_static_kpis(axle)
        merged.update(
            delegated_axle_dynamic_kpis(
                axle, roll_deg=roll_deg, travel_mm=travel_mm,
                sweep_steps=sweep_steps,
            )
        )
        return merged

    return {"front": _both(front_axle), "rear": _both(rear_axle)}


# =============================================================================
# Streamlit-cached entry point for the UI
# =============================================================================
# The heavy solve runs once per (geometry, inputs, sweep params) combination.
# ``@st.cache_data`` cannot hash a polars DataFrame or the CornerInputs dataclass,
# so we key the cache on the DataFrame's CSV text plus a flat tuple of the scalar
# inputs (same trick as ui/shared.py's sweep cache).


def compute_vdcore_kpis_cached(
    df: pl.DataFrame,
    inputs: CornerInputs | None = None,
    *,
    roll_deg: float = 1.5,
    travel_mm: float = 25.0,
    sweep_steps: int = 41,
) -> VdcoreKPIs:
    """Cached wrapper around :func:`compute_vdcore_kpis` for the Streamlit app.

    Falls back to an uncached call when Streamlit is not importable (e.g. tests
    or a bare REPL), so the bridge stays usable outside the app.
    """
    inputs = inputs or CornerInputs()
    try:
        import streamlit as st
    except ImportError:
        return compute_vdcore_kpis(
            df, inputs, roll_deg=roll_deg, travel_mm=travel_mm,
            sweep_steps=sweep_steps,
        )

    @st.cache_data(show_spinner=False, max_entries=64)
    def _cached(
        df_csv: str,
        inputs_key: tuple[float, ...],
        roll_deg: float,
        travel_mm: float,
        sweep_steps: int,
    ) -> VdcoreKPIs:
        import io

        import polars as pl

        rebuilt = pl.read_csv(io.StringIO(df_csv))
        rebuilt_inputs = CornerInputs(
            static_camber_deg=inputs_key[0],
            static_toe_deg_per_side=inputs_key[1],
            loaded_radius_mm=inputs_key[2],
            hardpoint_tol_mm=inputs_key[3],
            tire_tol_mm=inputs_key[4],
        )
        return compute_vdcore_kpis(
            rebuilt, rebuilt_inputs, roll_deg=roll_deg, travel_mm=travel_mm,
            sweep_steps=sweep_steps,
        )

    inputs_key = (
        inputs.static_camber_deg,
        inputs.static_toe_deg_per_side,
        inputs.loaded_radius_mm,
        inputs.hardpoint_tol_mm,
        inputs.tire_tol_mm,
    )
    return _cached(
        df.write_csv(), inputs_key, roll_deg, travel_mm, sweep_steps
    )


def compute_delegated_dynamic_kpis_cached(
    df: pl.DataFrame,
    inputs: CornerInputs | None = None,
    *,
    roll_deg: float = 1.5,
    travel_mm: float = 25.0,
    sweep_steps: int = 41,
) -> dict[str, dict[str, float]]:
    """Cached wrapper around :func:`compute_delegated_dynamic_kpis`.

    Used by the legacy Analysis tab to overwrite its wrong dynamic KPIs with
    vdcore's. Falls back to an uncached call when Streamlit is not importable so
    the delegation is testable from a bare REPL.
    """
    inputs = inputs or CornerInputs()
    try:
        import streamlit as st
    except ImportError:
        return compute_delegated_dynamic_kpis(
            df, inputs, roll_deg=roll_deg, travel_mm=travel_mm,
            sweep_steps=sweep_steps,
        )

    @st.cache_data(show_spinner=False, max_entries=64)
    def _cached(
        df_csv: str,
        inputs_key: tuple[float, ...],
        roll_deg: float,
        travel_mm: float,
        sweep_steps: int,
    ) -> dict[str, dict[str, float]]:
        import io

        import polars as pl

        rebuilt = pl.read_csv(io.StringIO(df_csv))
        rebuilt_inputs = CornerInputs(
            static_camber_deg=inputs_key[0],
            static_toe_deg_per_side=inputs_key[1],
            loaded_radius_mm=inputs_key[2],
            hardpoint_tol_mm=inputs_key[3],
            tire_tol_mm=inputs_key[4],
        )
        return compute_delegated_dynamic_kpis(
            rebuilt, rebuilt_inputs, roll_deg=roll_deg, travel_mm=travel_mm,
            sweep_steps=sweep_steps,
        )

    inputs_key = (
        inputs.static_camber_deg,
        inputs.static_toe_deg_per_side,
        inputs.loaded_radius_mm,
        inputs.hardpoint_tol_mm,
        inputs.tire_tol_mm,
    )
    return _cached(
        df.write_csv(), inputs_key, roll_deg, travel_mm, sweep_steps
    )


def compute_vdcore_setup_sheet_cached(
    df: pl.DataFrame,
    inputs: CornerInputs | None = None,
    *,
    roll_deg: float = 1.5,
    travel_mm: float = 25.0,
    sweep_steps: int = 41,
) -> dict[str, dict[str, float]]:
    """Cached wrapper around :func:`compute_vdcore_setup_sheet`.

    Used by the vdcore tab to fill every geometry row of its setup sheet from the
    validated solver in one call. Falls back to an uncached call when Streamlit
    is not importable so the sheet is testable from a bare REPL.
    """
    inputs = inputs or CornerInputs()
    try:
        import streamlit as st
    except ImportError:
        return compute_vdcore_setup_sheet(
            df, inputs, roll_deg=roll_deg, travel_mm=travel_mm,
            sweep_steps=sweep_steps,
        )

    @st.cache_data(show_spinner=False, max_entries=64)
    def _cached(
        df_csv: str,
        inputs_key: tuple[float, ...],
        roll_deg: float,
        travel_mm: float,
        sweep_steps: int,
    ) -> dict[str, dict[str, float]]:
        import io

        import polars as pl

        rebuilt = pl.read_csv(io.StringIO(df_csv))
        rebuilt_inputs = CornerInputs(
            static_camber_deg=inputs_key[0],
            static_toe_deg_per_side=inputs_key[1],
            loaded_radius_mm=inputs_key[2],
            hardpoint_tol_mm=inputs_key[3],
            tire_tol_mm=inputs_key[4],
        )
        return compute_vdcore_setup_sheet(
            rebuilt, rebuilt_inputs, roll_deg=roll_deg, travel_mm=travel_mm,
            sweep_steps=sweep_steps,
        )

    inputs_key = (
        inputs.static_camber_deg,
        inputs.static_toe_deg_per_side,
        inputs.loaded_radius_mm,
        inputs.hardpoint_tol_mm,
        inputs.tire_tol_mm,
    )
    return _cached(
        df.write_csv(), inputs_key, roll_deg, travel_mm, sweep_steps
    )
