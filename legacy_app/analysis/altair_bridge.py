"""
analysis/altair_bridge.py
=========================
Adapter between the app's loaded geometry and the Altair MotionSolve
cross-check, so the vdcore tab can show an "Altair" column beside its own.

WHY THIS EXISTS
    ``vdcore_bridge`` gives the app numbers from the validated ``DWSolver``.
    This module gives the same numbers from a completely different solver:
    Altair MotionSolve assembles revolute/spherical/universal joints into an
    index-3 DAE and integrates it with DASPK, where vdcore writes nine
    distance residuals and drives them to zero with ``least_squares``. Nothing
    is shared but the hardpoints, so agreement is real evidence and a
    disagreement localises to whichever KPI diverges.

    Both columns run through vdcore's OWN KPI formulas
    (``sample_corner`` / ``axle_rates`` / the roll-centre construction) --
    ``altair_model.kpi_runner`` replays MotionSolve's solved positions into
    them. So the columns differ only in where the upright ended up, never in
    what a KPI means.

WHY IT IS CACHED ON DISK
    A full four-corner pass is ~150 s of subprocess work, far too slow to run
    on a Streamlit rerun. Results are cached by a signature over the geometry
    AND the design inputs, so the column reappears instantly on a rerun and
    goes stale the moment either changes. A stale entry is never shown as
    current: the tab offers a re-run instead.

REQUIREMENTS
    Altair 2026.1 installed locally. Without it the column is greyed out with
    a reason -- the app stays fully usable, it just cannot offer a second
    opinion.

This module lives in ``legacy_app/`` (an application layer). It may import
``vdcore``, ``altair_model`` and ``streamlit``. It must NEVER be imported by
anything under ``vdcore/``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from altair_model.kpi_runner import (  # noqa: E402
    AltairUnavailableError,
    altair_kpis_from_csv,
)
from altair_model.msolve_driver import altair_available, altair_python  # noqa: E402

if TYPE_CHECKING:
    import polars as pl

# Cached runs live next to the model that produced them, not in the user's
# project tree. Kept out of git: they are machine-specific and regenerable.
CACHE_DIR = REPO / "altair_model" / ".kpi_cache"

# Bumped when the KPI definitions or the replay change, so old cache entries
# computed by different code are not silently shown as current.
_CACHE_SCHEMA = 1


@dataclass(frozen=True)
class AltairKPIs:
    """MotionSolve-sourced KPIs for both axles, keyed as the setup sheet expects.

    ``values`` mirrors ``vdcore_bridge.compute_vdcore_setup_sheet``:
    ``{"front": {...}, "rear": {...}}``.
    """

    values: dict[str, dict[str, float]]
    roll_patch_residual_mm: float
    signature: str
    elapsed_s: float

    def get(self, axle: str, key: str) -> float | None:
        """One KPI, or None when Altair does not produce it."""
        return self.values.get(axle, {}).get(key)


class AltairRunError(RuntimeError):
    """A MotionSolve run started but failed. Distinct from "not installed"."""


def availability() -> tuple[bool, str]:
    """(usable, reason). ``reason`` is empty when usable."""
    if altair_available():
        return True, ""
    return False, (
        f"Altair 2026.1 was not found at `{altair_python()}`. The Altair "
        "column needs a local MotionSolve install; every other number on this "
        "tab is unaffected."
    )


def geometry_signature(
    df: pl.DataFrame,
    *,
    static_camber_deg: float,
    loaded_radius_mm: float,
    static_toe_deg_per_side: float,
    roll_deg: float,
    travel_mm: float,
    sweep_steps: int,
) -> str:
    """Stable hash over everything that changes the answer.

    Covers the hardpoints AND the design inputs, because static camber and the
    loaded radius are built into the upright before MotionSolve ever runs. A
    change to any of them invalidates the cached column rather than letting a
    number computed from different inputs sit next to the vdcore one.
    """
    payload = json.dumps(
        {
            "schema": _CACHE_SCHEMA,
            "hardpoints": df.sort(["corner", "point"]).write_csv(),
            "static_camber_deg": round(float(static_camber_deg), 9),
            "loaded_radius_mm": round(float(loaded_radius_mm), 9),
            "static_toe_deg_per_side": round(float(static_toe_deg_per_side), 9),
            "roll_deg": round(float(roll_deg), 9),
            "travel_mm": round(float(travel_mm), 9),
            "sweep_steps": int(sweep_steps),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _cache_path(signature: str) -> Path:
    return CACHE_DIR / f"{signature}.json"


def load_cached(signature: str) -> AltairKPIs | None:
    """Return the cached run for this signature, or None.

    A cache file that cannot be parsed is treated as absent, never as partial
    data: half a column is worse than no column.
    """
    path = _cache_path(signature)
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
        return AltairKPIs(
            values={k: {kk: float(vv) for kk, vv in v.items()}
                    for k, v in blob["values"].items()},
            roll_patch_residual_mm=float(blob["roll_patch_residual_mm"]),
            signature=str(blob["signature"]),
            elapsed_s=float(blob.get("elapsed_s", 0.0)),
        )
    except (ValueError, KeyError, OSError):
        return None


def _store(result: AltairKPIs) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(result.signature).write_text(
        json.dumps(
            {
                "schema": _CACHE_SCHEMA,
                "signature": result.signature,
                "values": result.values,
                "roll_patch_residual_mm": result.roll_patch_residual_mm,
                "elapsed_s": result.elapsed_s,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_altair(
    df: pl.DataFrame,
    *,
    static_camber_deg: float,
    loaded_radius_mm: float,
    static_toe_deg_per_side: float = 0.0,
    roll_deg: float = 1.5,
    travel_mm: float = 25.0,
    sweep_steps: int = 41,
) -> AltairKPIs:
    """Run MotionSolve over the loaded geometry and cache the result.

    Blocking, and slow (~150 s) -- the caller is expected to have asked for it
    explicitly and to show a spinner.

    Raises:
        AltairUnavailableError: Altair is not installed here.
        AltairRunError: MotionSolve ran and failed.
    """
    import time

    usable, reason = availability()
    if not usable:
        raise AltairUnavailableError(reason)

    signature = geometry_signature(
        df,
        static_camber_deg=static_camber_deg,
        loaded_radius_mm=loaded_radius_mm,
        static_toe_deg_per_side=static_toe_deg_per_side,
        roll_deg=roll_deg,
        travel_mm=travel_mm,
        sweep_steps=sweep_steps,
    )

    started = time.time()
    with tempfile.TemporaryDirectory(prefix="altair_bridge_") as tmp:
        # The Altair interpreter has no polars, so the geometry crosses the
        # process boundary as the same CSV schema msolve_corner.py already reads.
        csv_path = Path(tmp) / "hardpoints.csv"
        df.write_csv(csv_path)
        try:
            per_axle = altair_kpis_from_csv(
                csv_path,
                roll_deg=roll_deg,
                travel_mm=travel_mm,
                sweep_steps=sweep_steps,
                static_toe_deg=static_toe_deg_per_side,
                static_camber_deg=static_camber_deg,
                loaded_radius_mm=loaded_radius_mm,
                workdir=Path(tmp),
            )
        except AltairUnavailableError:
            raise
        except Exception as exc:  # MotionSolve failure, bad geometry, ...
            raise AltairRunError(str(exc)) from exc

    result = AltairKPIs(
        values={k: dict(v.values) for k, v in per_axle.items()},
        roll_patch_residual_mm=max(
            v.roll_patch_residual_mm for v in per_axle.values()
        ),
        signature=signature,
        elapsed_s=time.time() - started,
    )
    _store(result)
    return result
