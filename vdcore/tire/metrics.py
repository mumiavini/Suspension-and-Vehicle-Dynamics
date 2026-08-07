"""Design-relevant tire metrics computed from binned raw TTC data.

No curve fitting — all metrics are derived from the raw force/moment
data by binning and simple regression. This captures 80 % of the
design-relevant information without the fragility of MF fitting.

All inputs must be in ISO 8855 convention (output of load_ttc_mat).
FZ values are positive for a loaded tire (Z+ up).

Coordinate system: ISO 8855 — X+ forward, Y+ left, Z+ up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl


@dataclass(frozen=True)
class BinMetrics:
    """Metrics for a single (FZ, IA, P) bin."""

    fz_nominal_n: float
    ia_nominal_deg: float
    p_nominal_kpa: float
    n_points: int

    peak_mu_lateral: float
    peak_mu_sa_deg: float

    cornering_stiffness_n_per_deg: float
    cs_regression_window_deg: float
    cs_r_squared: float

    peak_sharpness: float

    pneumatic_trail_at_peak_mm: float | None


@dataclass(frozen=True)
class LoadSensitivity:
    """Load sensitivity: how friction coefficient changes with normal load."""

    slope_per_n: float
    r_squared: float
    fz_range_n: tuple[float, float]
    ia_nominal_deg: float
    p_nominal_kpa: float


@dataclass(frozen=True)
class CamberSensitivity:
    """Camber sensitivity: lateral force change per degree of inclination."""

    dfy_dia_n_per_deg: float
    r_squared: float
    at_sa_deg: float
    fz_nominal_n: float
    p_nominal_kpa: float


@dataclass(frozen=True)
class LoadedRadiusFit:
    """Loaded radius vs FZ linear regression."""

    slope_mm_per_n: float
    intercept_mm: float
    r_squared: float
    fz_range_n: tuple[float, float]
    tolerance_mm: float


@dataclass(frozen=True)
class TireMetricsReport:
    """Complete metrics report for a single tire run."""

    tire_designation: str
    bin_metrics: list[BinMetrics]
    load_sensitivity: list[LoadSensitivity]
    camber_sensitivity: list[CamberSensitivity]
    loaded_radius_fit: LoadedRadiusFit | None


def _bin_centers(values: npt.NDArray[np.float64], n_bins: int) -> npt.NDArray[np.float64]:
    """Compute bin edges and return centre values for each bin."""
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    if vmin == vmax:
        return np.array([vmin])
    edges = np.linspace(vmin, vmax, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers


def _assign_bins(
    values: npt.NDArray[np.float64],
    n_bins: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
    """Assign each value to a bin index and return (indices, bin_centers)."""
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    if vmin == vmax:
        return np.zeros(len(values), dtype=np.int64), np.array([vmin])
    edges = np.linspace(vmin, vmax, n_bins + 1)
    indices = np.clip(np.digitize(values, edges) - 1, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return indices, centers


def compute_bin_metrics(
    df: pl.DataFrame,
    *,
    fz_min_n: float,
    fz_max_n: float,
    fz_bins: int = 4,
    ia_bins: int = 3,
    p_bins: int = 1,
    cs_window_deg: float = 2.0,
    sharpness_window_deg: float = 2.0,
) -> list[BinMetrics]:
    """Compute per-bin metrics from conditioned TTC data.

    Parameters
    ----------
    df:
        Conditioned DataFrame in ISO 8855 (from ttc.condition()).
    fz_min_n, fz_max_n:
        Normal load window (positive, ISO 8855). No default — the caller
        must specify the FZ range relevant to their car.
    fz_bins:
        Number of FZ bins within the window.
    ia_bins:
        Number of inclination angle bins.
    p_bins:
        Number of pressure bins.
    cs_window_deg:
        Half-width of the slip angle window around α=0 for cornering
        stiffness regression.
    sharpness_window_deg:
        How far from peak SA to measure retained force fraction.

    Returns
    -------
    list[BinMetrics]
        One entry per populated (FZ, IA, P) bin.
    """
    filtered = df.filter(
        (pl.col("fz_n") >= fz_min_n) & (pl.col("fz_n") <= fz_max_n)
    )
    if filtered.height == 0:
        return []

    fz = filtered["fz_n"].to_numpy()
    sa = filtered["sa_deg"].to_numpy()
    fy = filtered["fy_n"].to_numpy()
    ia = filtered["ia_deg"].to_numpy()
    p = filtered["p_kpa"].to_numpy()

    has_mz = "mz_nm" in filtered.columns
    mz = filtered["mz_nm"].to_numpy() if has_mz else None

    fz_idx, fz_centers = _assign_bins(fz, fz_bins)
    ia_idx, ia_centers = _assign_bins(ia, ia_bins)
    p_idx, p_centers = _assign_bins(p, p_bins)

    results: list[BinMetrics] = []

    for fi in range(len(fz_centers)):
        for ii in range(len(ia_centers)):
            for pi in range(len(p_centers)):
                mask = (fz_idx == fi) & (ia_idx == ii) & (p_idx == pi)
                n_pts = int(np.sum(mask))
                if n_pts < 10:
                    continue

                sa_bin = sa[mask]
                fy_bin = fy[mask]
                fz_bin = fz[mask]
                mz_bin = mz[mask] if mz is not None else None

                fz_mean = float(np.mean(fz_bin))
                if fz_mean <= 0:
                    continue

                mu = np.abs(fy_bin) / fz_bin
                peak_idx = int(np.argmax(mu))
                peak_mu = float(mu[peak_idx])
                peak_sa = float(sa_bin[peak_idx])

                cs_mask = np.abs(sa_bin) <= cs_window_deg
                cs_n = int(np.sum(cs_mask))
                if cs_n >= 3:
                    sa_cs = sa_bin[cs_mask]
                    fy_cs = fy_bin[cs_mask]
                    coeffs = np.polyfit(sa_cs, fy_cs, 1)
                    cs_slope = float(coeffs[0])
                    fy_pred = np.polyval(coeffs, sa_cs)
                    ss_res = float(np.sum((fy_cs - fy_pred) ** 2))
                    ss_tot = float(np.sum((fy_cs - np.mean(fy_cs)) ** 2))
                    cs_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                else:
                    cs_slope = 0.0
                    cs_r2 = 0.0

                peak_fy_abs = float(np.max(np.abs(fy_bin)))
                if peak_fy_abs > 0:
                    near_peak_mask = np.abs(sa_bin - peak_sa) <= sharpness_window_deg
                    if np.any(near_peak_mask):
                        retained = float(np.mean(np.abs(fy_bin[near_peak_mask])))
                        sharpness = retained / peak_fy_abs
                    else:
                        sharpness = 0.0
                else:
                    sharpness = 0.0

                ptrail: float | None = None
                if mz_bin is not None:
                    near_peak = np.abs(sa_bin - peak_sa) <= 1.0
                    fy_near = fy_bin[near_peak]
                    mz_near = mz_bin[near_peak]
                    nonzero_fy = np.abs(fy_near) > 1.0
                    if np.sum(nonzero_fy) >= 3:
                        trail_vals = -mz_near[nonzero_fy] / fy_near[nonzero_fy]
                        ptrail = float(np.median(trail_vals)) * 1000.0

                results.append(BinMetrics(
                    fz_nominal_n=float(fz_centers[fi]),
                    ia_nominal_deg=float(ia_centers[ii]),
                    p_nominal_kpa=float(p_centers[pi]),
                    n_points=n_pts,
                    peak_mu_lateral=peak_mu,
                    peak_mu_sa_deg=peak_sa,
                    cornering_stiffness_n_per_deg=cs_slope,
                    cs_regression_window_deg=cs_window_deg,
                    cs_r_squared=cs_r2,
                    peak_sharpness=sharpness,
                    pneumatic_trail_at_peak_mm=ptrail,
                ))

    return results


def compute_load_sensitivity(
    bin_metrics: list[BinMetrics],
) -> list[LoadSensitivity]:
    """Compute load sensitivity (dμ/dFZ) from bin metrics.

    Groups by (IA, P) and regresses peak_mu_lateral against FZ across
    the FZ bins within each group.
    """
    from itertools import groupby

    def group_key(b: BinMetrics) -> tuple[float, float]:
        return (b.ia_nominal_deg, b.p_nominal_kpa)

    sorted_bins = sorted(bin_metrics, key=group_key)
    results: list[LoadSensitivity] = []

    for (ia, p), group in groupby(sorted_bins, key=group_key):
        bins = list(group)
        if len(bins) < 2:
            continue

        fz_arr = np.array([b.fz_nominal_n for b in bins])
        mu_arr = np.array([b.peak_mu_lateral for b in bins])

        coeffs = np.polyfit(fz_arr, mu_arr, 1)
        slope = float(coeffs[0])
        mu_pred = np.polyval(coeffs, fz_arr)
        ss_res = float(np.sum((mu_arr - mu_pred) ** 2))
        ss_tot = float(np.sum((mu_arr - np.mean(mu_arr)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        results.append(LoadSensitivity(
            slope_per_n=slope,
            r_squared=r2,
            fz_range_n=(float(np.min(fz_arr)), float(np.max(fz_arr))),
            ia_nominal_deg=ia,
            p_nominal_kpa=p,
        ))

    return results


def compute_camber_sensitivity(
    df: pl.DataFrame,
    *,
    fz_min_n: float,
    fz_max_n: float,
    fz_bins: int = 4,
    p_bins: int = 1,
    peak_sa_tolerance_deg: float = 1.0,
) -> list[CamberSensitivity]:
    """Compute camber sensitivity: dFY/dIA near the peak slip angle.

    For each (FZ, P) bin, finds the peak-μ slip angle, selects data
    near that SA, and regresses FY against IA.
    """
    filtered = df.filter(
        (pl.col("fz_n") >= fz_min_n) & (pl.col("fz_n") <= fz_max_n)
    )
    if filtered.height == 0:
        return []

    fz = filtered["fz_n"].to_numpy()
    sa = filtered["sa_deg"].to_numpy()
    fy = filtered["fy_n"].to_numpy()
    ia = filtered["ia_deg"].to_numpy()
    p = filtered["p_kpa"].to_numpy()

    fz_idx, fz_centers = _assign_bins(fz, fz_bins)
    p_idx, p_centers = _assign_bins(p, p_bins)

    results: list[CamberSensitivity] = []

    for fi in range(len(fz_centers)):
        for pi in range(len(p_centers)):
            mask = (fz_idx == fi) & (p_idx == pi)
            n_pts = int(np.sum(mask))
            if n_pts < 10:
                continue

            sa_bin = sa[mask]
            fy_bin = fy[mask]
            fz_bin = fz[mask]
            ia_bin = ia[mask]

            fz_mean = float(np.mean(fz_bin))
            if fz_mean <= 0:
                continue

            mu = np.abs(fy_bin) / fz_bin
            peak_idx = int(np.argmax(mu))
            peak_sa = float(sa_bin[peak_idx])

            near_peak = np.abs(sa_bin - peak_sa) <= peak_sa_tolerance_deg
            ia_near = ia_bin[near_peak]
            fy_near = fy_bin[near_peak]

            if len(ia_near) < 3 or np.ptp(ia_near) < 0.5:
                continue

            coeffs = np.polyfit(ia_near, fy_near, 1)
            slope = float(coeffs[0])
            fy_pred = np.polyval(coeffs, ia_near)
            ss_res = float(np.sum((fy_near - fy_pred) ** 2))
            ss_tot = float(np.sum((fy_near - np.mean(fy_near)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            results.append(CamberSensitivity(
                dfy_dia_n_per_deg=slope,
                r_squared=r2,
                at_sa_deg=peak_sa,
                fz_nominal_n=float(fz_centers[fi]),
                p_nominal_kpa=float(p_centers[pi]),
            ))

    return results


def compute_loaded_radius_fit(
    df: pl.DataFrame,
    *,
    fz_min_n: float,
    fz_max_n: float,
) -> LoadedRadiusFit | None:
    """Fit loaded radius vs FZ by linear regression.

    Returns the slope, intercept, R², and the tolerance (max absolute
    residual) — this replaces the estimated tire_loaded_radius_mm with
    a measured value.
    """
    if "rl_mm" not in df.columns:
        return None

    filtered = df.filter(
        (pl.col("fz_n") >= fz_min_n) & (pl.col("fz_n") <= fz_max_n)
    )
    if filtered.height < 5:
        return None

    fz = filtered["fz_n"].to_numpy()
    rl = filtered["rl_mm"].to_numpy()

    coeffs = np.polyfit(fz, rl, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])
    rl_pred = np.polyval(coeffs, fz)
    residuals = rl - rl_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((rl - np.mean(rl)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    tol = float(np.max(np.abs(residuals)))

    return LoadedRadiusFit(
        slope_mm_per_n=slope,
        intercept_mm=intercept,
        r_squared=r2,
        fz_range_n=(float(np.min(fz)), float(np.max(fz))),
        tolerance_mm=tol,
    )


def compute_tire_metrics(
    df: pl.DataFrame,
    tire_designation: str,
    *,
    fz_min_n: float,
    fz_max_n: float,
    fz_bins: int = 4,
    ia_bins: int = 3,
    p_bins: int = 1,
    cs_window_deg: float = 2.0,
    sharpness_window_deg: float = 2.0,
) -> TireMetricsReport:
    """Compute all design-relevant metrics for a single tire run.

    This is the main entry point for tire analysis. It wraps the
    individual metric functions into a single report.

    Parameters
    ----------
    df:
        Conditioned DataFrame in ISO 8855.
    tire_designation:
        Human-readable tire name for the report.
    fz_min_n, fz_max_n:
        Normal load window in N (positive, ISO 8855). No default.
    """
    bins = compute_bin_metrics(
        df,
        fz_min_n=fz_min_n,
        fz_max_n=fz_max_n,
        fz_bins=fz_bins,
        ia_bins=ia_bins,
        p_bins=p_bins,
        cs_window_deg=cs_window_deg,
        sharpness_window_deg=sharpness_window_deg,
    )

    load_sens = compute_load_sensitivity(bins)
    camber_sens = compute_camber_sensitivity(
        df,
        fz_min_n=fz_min_n,
        fz_max_n=fz_max_n,
        fz_bins=fz_bins,
        p_bins=p_bins,
    )
    rl_fit = compute_loaded_radius_fit(df, fz_min_n=fz_min_n, fz_max_n=fz_max_n)

    return TireMetricsReport(
        tire_designation=tire_designation,
        bin_metrics=bins,
        load_sensitivity=load_sens,
        camber_sensitivity=camber_sens,
        loaded_radius_fit=rl_fit,
    )
