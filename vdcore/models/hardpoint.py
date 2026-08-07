"""Data models for suspension hardpoints and vehicle geometry.

Coordinate system: ISO 8855 — X+ forward, Y+ LEFT, Z+ up.
Left wheels (FL, RL) have positive Y. Right wheels (FR, RR) have negative Y.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator


class DerivedPoint(BaseModel, frozen=True):
    """A computed 3D point with no provenance — derived, not measured.

    ISO 8855: X+ forward, Y+ LEFT, Z+ up.
    """

    x_mm: float
    y_mm: float
    z_mm: float


class Hardpoint(BaseModel, frozen=True):
    """A 3D suspension hardpoint with provenance tracking.

    ISO 8855: X+ forward, Y+ LEFT, Z+ up.
    Left wheels have positive Y, right wheels have negative Y.
    """

    name: str
    x_mm: float
    y_mm: float
    z_mm: float
    source: Literal["cad", "measured", "estimate", "design_intent"]
    tol_mm: float


class TirePackage(BaseModel, frozen=True):
    """Tire geometry with provenance. The loaded radius is the single most
    uncertain input when no tire data is available.
    """

    loaded_radius_mm: float
    source: Literal["cad", "measured", "estimate", "design_intent"]
    tol_mm: float


_LEFT_CORNERS = frozenset({"FL", "RL"})
_RIGHT_CORNERS = frozenset({"FR", "RR"})


class Corner(BaseModel, frozen=True):
    """One suspension corner — inputs only, no derived geometry.

    ISO 8855: FL/RL corners have positive Y coordinates,
    FR/RR corners have negative Y coordinates.
    Derived quantities (contact patch, track, wheelbase) live in
    vdcore.geometry.derived as free functions.
    """

    corner_id: Literal["FL", "FR", "RL", "RR"]
    uca_inboard_front: Hardpoint
    uca_inboard_rear: Hardpoint
    uca_outboard: Hardpoint
    lca_inboard_front: Hardpoint
    lca_inboard_rear: Hardpoint
    lca_outboard: Hardpoint
    tie_rod_inboard: Hardpoint
    tie_rod_outboard: Hardpoint
    wheel_center: Hardpoint
    tire: TirePackage
    static_camber_deg: float
    static_toe_deg_per_side: float

    @model_validator(mode="after")
    def _validate_y_sign(self) -> Corner:
        """FL/RL wheel centres must have positive Y; FR/RR negative Y."""
        wc_y = self.wheel_center.y_mm
        if self.corner_id in _LEFT_CORNERS and wc_y <= 0:
            raise ValueError(
                f"Left corner {self.corner_id} must have positive Y "
                f"(ISO 8855: Y+ is LEFT), got wheel_center.y_mm={wc_y}"
            )
        if self.corner_id in _RIGHT_CORNERS and wc_y >= 0:
            raise ValueError(
                f"Right corner {self.corner_id} must have negative Y "
                f"(ISO 8855: Y+ is LEFT), got wheel_center.y_mm={wc_y}"
            )
        return self

    def has_estimates(self) -> bool:
        """True if any hardpoint or the tire package is tagged source='estimate'."""
        points = [
            self.uca_inboard_front,
            self.uca_inboard_rear,
            self.uca_outboard,
            self.lca_inboard_front,
            self.lca_inboard_rear,
            self.lca_outboard,
            self.tie_rod_inboard,
            self.tie_rod_outboard,
            self.wheel_center,
        ]
        if any(p.source == "estimate" for p in points):
            return True
        return self.tire.source == "estimate"

    def hardpoints(self) -> list[Hardpoint]:
        """All 9 measured hardpoints in definition order."""
        return [
            self.uca_inboard_front,
            self.uca_inboard_rear,
            self.uca_outboard,
            self.lca_inboard_front,
            self.lca_inboard_rear,
            self.lca_outboard,
            self.tie_rod_inboard,
            self.tie_rod_outboard,
            self.wheel_center,
        ]


class Axle(BaseModel, frozen=True):
    """A pair of left and right corners forming one axle."""

    left: Corner
    right: Corner


class Vehicle(BaseModel, frozen=True):
    """Full vehicle: front and rear axles.

    Derived dimensions (track, wheelbase) are in vdcore.geometry.derived.
    """

    front: Axle
    rear: Axle
    schema_version: int = 1
