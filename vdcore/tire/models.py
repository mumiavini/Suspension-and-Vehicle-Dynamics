"""Pydantic models for TTC tire test data.

All data leaving these models is in ISO 8855 convention:
X+ forward, Y+ left, Z+ up.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel


class TTCRun(BaseModel, frozen=True):
    """Metadata for a single TTC test run.

    Carries provenance so every downstream metric can trace back to
    the exact file and test conditions that produced it.
    """

    tire_designation: str
    rim_width_in: float
    test_round: str
    file_path: str
    test_date: date | None = None
    source: Literal["cad", "measured", "estimate", "design_intent"] = "measured"
    tol_mm: float = 0.0
    notes: str = ""


class FilterReport(BaseModel, frozen=True):
    """Audit trail for a single conditioning filter step."""

    filter_name: str
    rows_before: int
    rows_after: int
    rows_removed: int
    parameters: dict[str, float | str]
