"""TTC .mat file loader with adapted-SAE → ISO 8855 sign conversion.

Sign convention conversion — derived from frame definitions
===========================================================

TTC data uses adapted-SAE: X+ forward, Y+ right, Z+ down (right-handed).
This project uses ISO 8855: X+ forward, Y+ left, Z+ up (right-handed).

The two frames share X but invert Y and Z. The transform matrix is::

    M = diag(1, -1, -1)

For forces and moments, the conversion rules follow from this transform:

- **FX**: X-axis force → no sign change.
- **FY**: Y-axis force → negate (Y reverses).
- **FZ**: Z-axis force → negate (Z reverses). A loaded tire in SAE has
  FZ < 0 (force points down = negative Z-down). After negation, ISO 8855
  FZ > 0 (force points up = positive Z-up) for a loaded tire.
- **MX**: Overturning moment (about X) → negate. MX involves torque
  in the YZ plane; inverting both Y and Z flips the cross-product sign once.
- **MZ**: Self-aligning torque (about Z) → negate. SAE positive = clockwise
  top-view; ISO positive = counterclockwise (right-hand rule about Z-up).
- **SA**: Slip angle → negate. SAE positive = wheel pointed right of travel;
  ISO positive = wheel pointed left of travel.
- **IA**: Inclination angle → negate. SAE positive = top of wheel tilted
  toward Y+ (rightward); ISO positive = top tilted toward Y+ (leftward).
- **SR**: Slip ratio → no sign change (longitudinal, X-axis quantity).
- **Scalars** (P, V, RL, RE, ET, temperatures): no sign change.

Unit conversions applied at load time:

- RL, RE: TTC ships metres → multiply by 1000 → mm.

After conversion, the fundamental sanity check is: a positive slip angle
must produce a positive lateral force (leftward in ISO 8855).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.io import loadmat

from vdcore.tire.models import FilterReport, TTCRun

# Channels we extract from TTC .mat files.  Keys are the MATLAB variable
# names; values are the ISO 8855 column names we use in the DataFrame.
_TTC_CHANNELS: dict[str, str] = {
    "SA": "sa_deg",
    "SR": "sr",
    "FZ": "fz_n",
    "FY": "fy_n",
    "FX": "fx_n",
    "MZ": "mz_nm",
    "MX": "mx_nm",
    "IA": "ia_deg",
    "P": "p_kpa",
    "RL": "rl_mm",
    "RE": "re_mm",
    "V": "v_kmh",
    "ET": "et_s",
    "TSTC": "tstc_degc",
    "TSTI": "tsti_degc",
    "TSTO": "tsto_degc",
}

# Channels that require sign negation (adapted-SAE → ISO 8855).
_NEGATE_CHANNELS: frozenset[str] = frozenset({
    "SA", "FY", "FZ", "MX", "MZ", "IA",
})

# Channels that require unit conversion from metres to millimetres.
_M_TO_MM_CHANNELS: frozenset[str] = frozenset({"RL", "RE"})


def _extract_channel(mat: dict[str, Any], key: str) -> npt.NDArray[np.float64]:
    """Extract a single channel from a loaded .mat dict, flattening to 1-D."""
    if key not in mat:
        raise KeyError(
            f"TTC .mat file missing expected channel '{key}'. "
            f"Available keys: {sorted(k for k in mat if not k.startswith('__'))}"
        )
    arr = np.asarray(mat[key], dtype=np.float64).ravel()
    return arr


def load_ttc_mat(
    path: str | Path,
    *,
    tire_designation: str,
    rim_width_in: float,
    test_round: str,
) -> tuple[pl.DataFrame, TTCRun]:
    """Load a TTC .mat file, convert to ISO 8855, return (DataFrame, metadata).

    The sign convention conversion is applied here and only here. All
    downstream code receives ISO 8855 data. See module docstring for the
    full channel-by-channel derivation.

    Parameters
    ----------
    path:
        Path to the TTC .mat file.
    tire_designation:
        Tire name/code as labelled in TTC data (e.g. "Hoosier 18x6-10 R25B").
    rim_width_in:
        Rim width in inches (e.g. 7.0).
    test_round:
        TTC round identifier (e.g. "Round9").

    Returns
    -------
    tuple[pl.DataFrame, TTCRun]
        DataFrame with ISO 8855 columns and a metadata model.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"TTC .mat file not found: {filepath}")

    mat = loadmat(str(filepath), squeeze_me=True)

    columns: dict[str, list[float]] = {}
    n_rows: int | None = None

    for ttc_key, col_name in _TTC_CHANNELS.items():
        try:
            arr = _extract_channel(mat, ttc_key)
        except KeyError:
            if ttc_key in {"TSTC", "TSTI", "TSTO", "RE"}:
                continue
            raise

        if n_rows is None:
            n_rows = len(arr)
        elif len(arr) != n_rows:
            raise ValueError(
                f"Channel '{ttc_key}' has {len(arr)} rows, "
                f"expected {n_rows} (from first channel)."
            )

        if ttc_key in _NEGATE_CHANNELS:
            arr = -arr

        if ttc_key in _M_TO_MM_CHANNELS:
            arr = arr * 1000.0

        columns[col_name] = arr.tolist()

    if not columns:
        raise ValueError(f"No recognised channels in {filepath}")

    df = pl.DataFrame(columns)

    meta = TTCRun(
        tire_designation=tire_designation,
        rim_width_in=rim_width_in,
        test_round=test_round,
        file_path=str(filepath),
        source="measured",
    )

    return df, meta


def condition(
    df: pl.DataFrame,
    *,
    warmup_seconds: float = 0.0,
    pressure_target_kpa: float | None = None,
    pressure_tolerance_kpa: float = 7.0,
    velocity_min_kmh: float | None = None,
    velocity_max_kmh: float | None = None,
    temp_min_degc: float | None = None,
    temp_max_degc: float | None = None,
) -> tuple[pl.DataFrame, list[FilterReport]]:
    """Apply conditioning filters to a loaded TTC DataFrame.

    Every filter reports how many rows it removed — silent data loss is
    not acceptable. Filters are applied in the order listed.

    Parameters
    ----------
    df:
        DataFrame from load_ttc_mat() with ISO 8855 columns.
    warmup_seconds:
        Drop rows where et_s < warmup_seconds.
    pressure_target_kpa:
        If set, keep only rows within ±pressure_tolerance_kpa of this value.
    pressure_tolerance_kpa:
        Half-width of the pressure band (default 7 kPa ≈ 1 psi).
    velocity_min_kmh, velocity_max_kmh:
        If set, keep only rows within this speed range.
    temp_min_degc, temp_max_degc:
        If set, keep only rows where tread centre temperature (tstc_degc)
        falls within this range. Ignored if tstc_degc column is absent.

    Returns
    -------
    tuple[pl.DataFrame, list[FilterReport]]
        Filtered DataFrame and a list of per-filter audit reports.
    """
    reports: list[FilterReport] = []

    if warmup_seconds > 0.0:
        before = df.height
        df = df.filter(pl.col("et_s") >= warmup_seconds)
        reports.append(FilterReport(
            filter_name="warmup_drop",
            rows_before=before,
            rows_after=df.height,
            rows_removed=before - df.height,
            parameters={"warmup_seconds": warmup_seconds},
        ))

    if pressure_target_kpa is not None:
        before = df.height
        lo = pressure_target_kpa - pressure_tolerance_kpa
        hi = pressure_target_kpa + pressure_tolerance_kpa
        df = df.filter(
            (pl.col("p_kpa") >= lo) & (pl.col("p_kpa") <= hi)
        )
        reports.append(FilterReport(
            filter_name="pressure_band",
            rows_before=before,
            rows_after=df.height,
            rows_removed=before - df.height,
            parameters={
                "target_kpa": pressure_target_kpa,
                "tolerance_kpa": pressure_tolerance_kpa,
            },
        ))

    if velocity_min_kmh is not None or velocity_max_kmh is not None:
        before = df.height
        expr = pl.lit(True)
        params: dict[str, float | str] = {}
        if velocity_min_kmh is not None:
            expr = expr & (pl.col("v_kmh") >= velocity_min_kmh)
            params["min_kmh"] = velocity_min_kmh
        if velocity_max_kmh is not None:
            expr = expr & (pl.col("v_kmh") <= velocity_max_kmh)
            params["max_kmh"] = velocity_max_kmh
        df = df.filter(expr)
        reports.append(FilterReport(
            filter_name="velocity_band",
            rows_before=before,
            rows_after=df.height,
            rows_removed=before - df.height,
            parameters=params,
        ))

    if (temp_min_degc is not None or temp_max_degc is not None) and "tstc_degc" in df.columns:
        before = df.height
        expr = pl.lit(True)
        params = {}
        if temp_min_degc is not None:
            expr = expr & (pl.col("tstc_degc") >= temp_min_degc)
            params["min_degc"] = temp_min_degc
        if temp_max_degc is not None:
            expr = expr & (pl.col("tstc_degc") <= temp_max_degc)
            params["max_degc"] = temp_max_degc
        df = df.filter(expr)
        reports.append(FilterReport(
            filter_name="temperature_window",
            rows_before=before,
            rows_after=df.height,
            rows_removed=before - df.height,
            parameters=params,
        ))

    return df, reports
