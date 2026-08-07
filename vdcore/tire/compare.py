"""Tire comparison: one row per tire with all design-relevant metrics.

This module produces the artifact that answers "which tire" — a table
the designer reads and chooses from. It does not rank or recommend.

Coordinate system: ISO 8855 — X+ forward, Y+ left, Z+ up.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from vdcore.tire.metrics import TireMetricsReport


def compare_tires(
    reports: list[TireMetricsReport],
) -> pl.DataFrame:
    """Build a comparison DataFrame with one row per tire.

    Each row summarises the metrics across all (FZ, IA, P) bins by
    reporting the mean, min, and max where a single number would be
    misleading. Load sensitivity and camber sensitivity are averaged
    across their respective groups.

    Parameters
    ----------
    reports:
        One TireMetricsReport per tire, from compute_tire_metrics().

    Returns
    -------
    pl.DataFrame
        One row per tire. Columns include the tire designation and
        aggregated metric values.
    """
    rows: list[dict[str, object]] = []

    for report in reports:
        row: dict[str, object] = {"tire": report.tire_designation}

        if report.bin_metrics:
            peak_mus = [b.peak_mu_lateral for b in report.bin_metrics]
            row["peak_mu_mean"] = float(np.mean(peak_mus))
            row["peak_mu_min"] = float(np.min(peak_mus))
            row["peak_mu_max"] = float(np.max(peak_mus))

            peak_sas = [b.peak_mu_sa_deg for b in report.bin_metrics]
            row["peak_sa_mean_deg"] = float(np.mean(peak_sas))

            cs_vals = [b.cornering_stiffness_n_per_deg for b in report.bin_metrics]
            row["cs_mean_n_per_deg"] = float(np.mean(cs_vals))
            row["cs_min_n_per_deg"] = float(np.min(cs_vals))
            row["cs_max_n_per_deg"] = float(np.max(cs_vals))

            sharp_vals = [b.peak_sharpness for b in report.bin_metrics]
            row["sharpness_mean"] = float(np.mean(sharp_vals))

            trail_vals = [
                b.pneumatic_trail_at_peak_mm
                for b in report.bin_metrics
                if b.pneumatic_trail_at_peak_mm is not None
            ]
            if trail_vals:
                row["pneumatic_trail_mean_mm"] = float(np.mean(trail_vals))
            else:
                row["pneumatic_trail_mean_mm"] = None
        else:
            row["peak_mu_mean"] = None
            row["peak_mu_min"] = None
            row["peak_mu_max"] = None
            row["peak_sa_mean_deg"] = None
            row["cs_mean_n_per_deg"] = None
            row["cs_min_n_per_deg"] = None
            row["cs_max_n_per_deg"] = None
            row["sharpness_mean"] = None
            row["pneumatic_trail_mean_mm"] = None

        if report.load_sensitivity:
            slopes = [ls.slope_per_n for ls in report.load_sensitivity]
            row["load_sens_mean_per_n"] = float(np.mean(slopes))
            r2s = [ls.r_squared for ls in report.load_sensitivity]
            row["load_sens_r2_mean"] = float(np.mean(r2s))
        else:
            row["load_sens_mean_per_n"] = None
            row["load_sens_r2_mean"] = None

        if report.camber_sensitivity:
            cam_slopes = [cs.dfy_dia_n_per_deg for cs in report.camber_sensitivity]
            row["camber_sens_mean_n_per_deg"] = float(np.mean(cam_slopes))
        else:
            row["camber_sens_mean_n_per_deg"] = None

        if report.loaded_radius_fit is not None:
            rl = report.loaded_radius_fit
            row["rl_intercept_mm"] = rl.intercept_mm
            row["rl_slope_mm_per_n"] = rl.slope_mm_per_n
            row["rl_tolerance_mm"] = rl.tolerance_mm
            row["rl_r2"] = rl.r_squared
        else:
            row["rl_intercept_mm"] = None
            row["rl_slope_mm_per_n"] = None
            row["rl_tolerance_mm"] = None
            row["rl_r2"] = None

        rows.append(row)

    return pl.DataFrame(rows)
