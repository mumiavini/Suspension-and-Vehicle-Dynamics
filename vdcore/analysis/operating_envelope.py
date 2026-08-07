"""Operating envelope: per-corner loads and achievable lateral acceleration.

Bridges tire data (binned metrics from Phase A) to the vehicle load-transfer
model (Phase B) to answer: given these tires on this car, how fast can it
corner?

The achievable Ay is a fixed-point problem:
  Ay determines load transfer → per-tire Fz → μ through load sensitivity
  → μ determines the available lateral force → Ay.

Iterate until convergence.  This number is the single most important output
of the design tool so far — it is the first quantitative statement about how
fast the car can corner, and every target downstream is derived from it.

Coordinate system: ISO 8855 — X+ forward, Y+ left, Z+ up.
Positive Ay = leftward acceleration (left turn, right side loaded).
"""

from __future__ import annotations

from dataclasses import dataclass

from vdcore.analysis.load_transfer import (
    VehicleLoadTransferResult,
    lateral_load_transfer,
)
from vdcore.explain import Explained
from vdcore.models.mass import MassProperties, UnsprungMassSet
from vdcore.tire.metrics import BinMetrics

_G = 9.81


# ---------------------------------------------------------------------------
# Tire-metric lookup
# ---------------------------------------------------------------------------


def mu_at_fz(
    fz_n: float,
    bin_metrics: list[BinMetrics],
    *,
    ia_nominal_deg: float | None = None,
    p_nominal_kpa: float | None = None,
) -> tuple[float, bool]:
    """Friction coefficient at a given FZ by linear interpolation between bins.

    Parameters
    ----------
    fz_n:
        Normal load in N (positive, ISO 8855).
    bin_metrics:
        List of BinMetrics from :func:`compute_bin_metrics`.
    ia_nominal_deg:
        Filter bins to this inclination angle (optional).
    p_nominal_kpa:
        Filter bins to this pressure (optional).

    Returns
    -------
    tuple[float, bool]
        ``(mu, was_clamped)``.  If *fz_n* is outside the bin range,
        clamps to the nearest bin and sets *was_clamped* to True.
        Returns ``(0.0, True)`` for *fz_n* <= 0.
    """
    if fz_n <= 0.0:
        return 0.0, True

    bins = bin_metrics
    if ia_nominal_deg is not None:
        bins = [b for b in bins if b.ia_nominal_deg == ia_nominal_deg]
    if p_nominal_kpa is not None:
        bins = [b for b in bins if b.p_nominal_kpa == p_nominal_kpa]

    if not bins:
        raise ValueError(
            "No bin metrics available after filtering "
            f"(ia={ia_nominal_deg}, p={p_nominal_kpa}). "
            "Check that bin_metrics is not empty and that the filter "
            "values match existing bins."
        )

    sorted_bins = sorted(bins, key=lambda b: b.fz_nominal_n)

    if len(sorted_bins) == 1:
        return sorted_bins[0].peak_mu_lateral, fz_n != sorted_bins[0].fz_nominal_n

    fz_lo = sorted_bins[0].fz_nominal_n
    fz_hi = sorted_bins[-1].fz_nominal_n

    if fz_n <= fz_lo:
        return sorted_bins[0].peak_mu_lateral, True
    if fz_n >= fz_hi:
        return sorted_bins[-1].peak_mu_lateral, True

    for i in range(len(sorted_bins) - 1):
        b_lo = sorted_bins[i]
        b_hi = sorted_bins[i + 1]
        if b_lo.fz_nominal_n <= fz_n <= b_hi.fz_nominal_n:
            span = b_hi.fz_nominal_n - b_lo.fz_nominal_n
            if span < 1e-10:
                return b_lo.peak_mu_lateral, False
            t = (fz_n - b_lo.fz_nominal_n) / span
            mu = b_lo.peak_mu_lateral + t * (b_hi.peak_mu_lateral - b_lo.peak_mu_lateral)
            return mu, False

    return sorted_bins[-1].peak_mu_lateral, True


# ---------------------------------------------------------------------------
# Per-corner loads at a given Ay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CornerLoadResult:
    """Per-corner normal load and available friction at a given Ay."""

    corner: str
    static_fz_n: float
    delta_fz_n: float
    total_fz_n: float
    mu_available: float
    fz_was_clamped: bool


@dataclass(frozen=True)
class VehicleCornerLoads:
    """All four corners at a given Ay."""

    ay_g: float
    fl: CornerLoadResult
    fr: CornerLoadResult
    rl: CornerLoadResult
    rr: CornerLoadResult
    load_transfer: VehicleLoadTransferResult
    estimate_inputs: list[str]

    def corners(self) -> list[CornerLoadResult]:
        """Return all four corners as a list."""
        return [self.fl, self.fr, self.rl, self.rr]


def corner_loads_at_ay(
    mass: MassProperties,
    unsprung: UnsprungMassSet,
    bin_metrics: list[BinMetrics],
    *,
    ay_g: float,
    front_rc_height_mm: float,
    rear_rc_height_mm: float,
    front_track_mm: float,
    rear_track_mm: float,
    front_roll_stiffness_nm_per_deg: float,
    rear_roll_stiffness_nm_per_deg: float,
    wheelbase_mm: float,
    ia_nominal_deg: float | None = None,
    p_nominal_kpa: float | None = None,
) -> VehicleCornerLoads:
    """Compute per-corner normal load and available μ at a given Ay.

    Parameters
    ----------
    mass:
        Vehicle mass properties.
    unsprung:
        Per-corner unsprung masses.
    bin_metrics:
        Tire bin metrics (same tire assumed all four corners).
    ay_g:
        Lateral acceleration in g.  Positive = leftward (ISO 8855).
    ia_nominal_deg, p_nominal_kpa:
        Optional filters for bin lookup.

    Returns
    -------
    VehicleCornerLoads
        Per-corner Fz and μ.  The four total_fz_n values sum to
        m_total * g (conservation).
    """
    lt = lateral_load_transfer(
        mass,
        unsprung,
        ay_g=ay_g,
        front_rc_height_mm=front_rc_height_mm,
        rear_rc_height_mm=rear_rc_height_mm,
        front_track_mm=front_track_mm,
        rear_track_mm=rear_track_mm,
        front_roll_stiffness_nm_per_deg=front_roll_stiffness_nm_per_deg,
        rear_roll_stiffness_nm_per_deg=rear_roll_stiffness_nm_per_deg,
        wheelbase_mm=wheelbase_mm,
    )

    m_total = mass.total_mass_kg.value
    fmf = mass.front_mass_fraction.value
    w_total = m_total * _G

    static_front_per_corner = w_total * fmf / 2.0
    static_rear_per_corner = w_total * (1.0 - fmf) / 2.0

    # Positive delta_fz = load transferred to the outside wheel.
    # Positive ay = leftward → right side is outside.
    # FR gets +delta, FL gets -delta.  RR gets +delta, RL gets -delta.
    front_delta = lt.front.total_delta_fz_n
    rear_delta = lt.rear.total_delta_fz_n

    def _corner(
        name: str,
        static: float,
        delta: float,
    ) -> CornerLoadResult:
        total = max(static + delta, 0.0)
        mu, clamped = mu_at_fz(
            total,
            bin_metrics,
            ia_nominal_deg=ia_nominal_deg,
            p_nominal_kpa=p_nominal_kpa,
        )
        return CornerLoadResult(
            corner=name,
            static_fz_n=static,
            delta_fz_n=delta,
            total_fz_n=total,
            mu_available=mu,
            fz_was_clamped=clamped,
        )

    fl = _corner("FL", static_front_per_corner, -front_delta)
    fr = _corner("FR", static_front_per_corner, front_delta)
    rl = _corner("RL", static_rear_per_corner, -rear_delta)
    rr = _corner("RR", static_rear_per_corner, rear_delta)

    estimates: list[str] = []
    if mass.has_estimates():
        estimates.extend(mass.estimate_fields())

    return VehicleCornerLoads(
        ay_g=ay_g,
        fl=fl,
        fr=fr,
        rl=rl,
        rr=rr,
        load_transfer=lt,
        estimate_inputs=estimates,
    )


# ---------------------------------------------------------------------------
# Achievable lateral acceleration (fixed-point iteration)
# ---------------------------------------------------------------------------


def achievable_ay(
    mass: MassProperties,
    unsprung: UnsprungMassSet,
    bin_metrics: list[BinMetrics],
    *,
    front_rc_height_mm: float,
    rear_rc_height_mm: float,
    front_track_mm: float,
    rear_track_mm: float,
    front_roll_stiffness_nm_per_deg: float,
    rear_roll_stiffness_nm_per_deg: float,
    wheelbase_mm: float,
    ia_nominal_deg: float | None = None,
    p_nominal_kpa: float | None = None,
    ay_initial_guess_g: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> Explained[float]:
    """Achievable steady-state lateral acceleration.

    This is the total-force fixed point: the Ay at which the four tires
    can generate exactly enough lateral force to sustain that Ay.  It is
    **not** the balance-limited Ay (which axle saturates first) — that
    comes from :mod:`vdcore.analysis.balance_targets`.

    Uses successive substitution because the μ-vs-Fz relationship comes
    from discrete binned data (no analytic Jacobian).  Convergence is
    reliable because load sensitivity is negative: μ decreases with
    increasing Fz, creating a self-stabilizing feedback loop.

    Raises
    ------
    RuntimeError
        If the iteration does not converge within *max_iter*.
        Never returns a plausible-looking number on non-convergence.
    """
    m_total = mass.total_mass_kg.value
    w_total = m_total * _G

    ay_k = ay_initial_guess_g
    history: list[float] = [ay_k]
    any_clamped = False

    for i in range(max_iter):
        loads = corner_loads_at_ay(
            mass,
            unsprung,
            bin_metrics,
            ay_g=ay_k,
            front_rc_height_mm=front_rc_height_mm,
            rear_rc_height_mm=rear_rc_height_mm,
            front_track_mm=front_track_mm,
            rear_track_mm=rear_track_mm,
            front_roll_stiffness_nm_per_deg=front_roll_stiffness_nm_per_deg,
            rear_roll_stiffness_nm_per_deg=rear_roll_stiffness_nm_per_deg,
            wheelbase_mm=wheelbase_mm,
            ia_nominal_deg=ia_nominal_deg,
            p_nominal_kpa=p_nominal_kpa,
        )

        fy_total = sum(c.mu_available * c.total_fz_n for c in loads.corners())
        any_clamped = any_clamped or any(c.fz_was_clamped for c in loads.corners())

        ay_next = fy_total / w_total
        history.append(ay_next)

        if abs(ay_next - ay_k) < tol:
            return _build_explained(
                ay_next,
                history,
                loads,
                mass,
                any_clamped,
                converged=True,
                iterations=i + 1,
                residual=abs(ay_next - ay_k),
                front_rc_height_mm=front_rc_height_mm,
                rear_rc_height_mm=rear_rc_height_mm,
                front_track_mm=front_track_mm,
                rear_track_mm=rear_track_mm,
                front_roll_stiffness_nm_per_deg=front_roll_stiffness_nm_per_deg,
                rear_roll_stiffness_nm_per_deg=rear_roll_stiffness_nm_per_deg,
                wheelbase_mm=wheelbase_mm,
                ia_nominal_deg=ia_nominal_deg,
                p_nominal_kpa=p_nominal_kpa,
            )

        if i >= 2 and abs(ay_next) > 2.0 * abs(ay_k) and abs(history[-2]) > 2.0 * abs(history[-3]):
            break

        ay_k = ay_next

    raise RuntimeError(
        f"achievable_ay did not converge after {max_iter} iterations. "
        f"Last Ay = {ay_k:.4f} g, residual = {abs(history[-1] - history[-2]):.2e}. "
        f"History: {[round(h, 4) for h in history[-5:]]}"
    )


def _build_explained(
    value: float,
    history: list[float],
    loads: VehicleCornerLoads,
    mass: MassProperties,
    any_clamped: bool,
    *,
    converged: bool,
    iterations: int,
    residual: float,
    front_rc_height_mm: float,
    rear_rc_height_mm: float,
    front_track_mm: float,
    rear_track_mm: float,
    front_roll_stiffness_nm_per_deg: float,
    rear_roll_stiffness_nm_per_deg: float,
    wheelbase_mm: float,
    ia_nominal_deg: float | None,
    p_nominal_kpa: float | None,
) -> Explained[float]:
    """Build the Explained wrapper for achievable_ay."""

    def _src(field_name: str) -> str:
        pf = getattr(mass, field_name, None)
        if pf is not None:
            return str(pf.source)
        return "computed"

    assumptions = [
        "Quasi-static steady-state cornering",
        "Linear interpolation between tire FZ bins",
        "Both axles contribute proportionally (not balance-limited)",
        "LLTD and roll gradient fixed at static values",
        "Small roll angle",
    ]
    if any_clamped:
        assumptions.append("FZ outside measured bin range — clamped to nearest bin")

    corners = loads.corners()

    return Explained(
        value=value,
        formula="ay = Fy_total / (m_total * g)",
        inputs={
            "m_total": (mass.total_mass_kg.value, "kg", _src("total_mass_kg")),
            "g": (_G, "m/s^2", "computed"),
            "cg_height_mm": (
                mass.cg_height_mm.value,
                "mm",
                _src("cg_height_mm"),
            ),
            "front_mass_fraction": (
                mass.front_mass_fraction.value,
                "",
                _src("front_mass_fraction"),
            ),
            "front_rc_height_mm": (front_rc_height_mm, "mm", "computed"),
            "rear_rc_height_mm": (rear_rc_height_mm, "mm", "computed"),
            "front_track_mm": (front_track_mm, "mm", "computed"),
            "rear_track_mm": (rear_track_mm, "mm", "computed"),
            "front_roll_stiffness_nm_per_deg": (
                front_roll_stiffness_nm_per_deg,
                "Nm/deg",
                "computed",
            ),
            "rear_roll_stiffness_nm_per_deg": (
                rear_roll_stiffness_nm_per_deg,
                "Nm/deg",
                "computed",
            ),
            "wheelbase_mm": (wheelbase_mm, "mm", "computed"),
            "Fy_total": (value * mass.total_mass_kg.value * _G, "N", "computed"),
        },
        intermediates={
            "iterations": float(iterations),
            "residual": residual,
            "fl_fz_n": corners[0].total_fz_n,
            "fr_fz_n": corners[1].total_fz_n,
            "rl_fz_n": corners[2].total_fz_n,
            "rr_fz_n": corners[3].total_fz_n,
            "fl_mu": corners[0].mu_available,
            "fr_mu": corners[1].mu_available,
            "rl_mu": corners[2].mu_available,
            "rr_mu": corners[3].mu_available,
            "lltd": loads.load_transfer.lltd,
        },
        reference=(
            "Fixed-point iteration on the total lateral force equation. "
            "Converges because tire load sensitivity is negative."
        ),
        assumptions=assumptions,
    )
