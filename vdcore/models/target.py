"""Design target models for suspension optimization.

Targets describe what the suspension should achieve. Each target
must have a rationale — a target you cannot justify should not be
constructible. Weight belongs in the optimizer's objective config,
not here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class DesignTarget(BaseModel, frozen=True):
    """A single design target with justification.

    The event field links the target to the FSAE event that drives it.
    The rationale field is required — it forces the designer to articulate
    why this target exists before encoding it.
    """

    name: str
    value: float
    unit: str
    tolerance: float
    event: Literal["skidpad", "autocross", "accel", "endurance", "all"]
    rationale: str


class DesignTargets(BaseModel, frozen=True):
    """Collection of static and dynamic design targets."""

    static: list[DesignTarget]
    dynamic: list[DesignTarget]
