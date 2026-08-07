"""Vehicle mass and inertia models with per-field provenance.

Coordinate system: ISO 8855 — X+ forward, Y+ left, Z+ up.
Origin: front axle centreline, ground plane, vehicle centreline.

CG position is given relative to this origin:
  - cg_x_mm: positive = behind front axle (toward rear)
  - cg_height_mm: positive = above ground

Every field carries source and tolerance. CG height in particular
will start as an estimate and is the single input with the largest
effect on load transfer — it must be visibly uncertain everywhere
it is used downstream.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator

SourceType = Literal["cad", "measured", "estimate", "design_intent"]


class ProvenanceFloat(BaseModel, frozen=True):
    """A scalar value with provenance tracking."""

    value: float
    source: SourceType
    tol: float


class MassProperties(BaseModel, frozen=True):
    """Vehicle mass properties including driver.

    Coordinate system: ISO 8855. CG position relative to front axle
    centreline at ground level.

    front_mass_fraction is derived from cg_x_mm and wheelbase and
    is validated at construction time. It is stored rather than
    recomputed so that the model is self-contained (does not need
    access to geometry).
    """

    total_mass_kg: ProvenanceFloat
    driver_mass_kg: ProvenanceFloat
    cg_height_mm: ProvenanceFloat
    cg_x_mm: ProvenanceFloat
    front_mass_fraction: ProvenanceFloat
    yaw_inertia_kgm2: ProvenanceFloat
    roll_inertia_kgm2: ProvenanceFloat

    @model_validator(mode="after")
    def _validate_mass_fractions(self) -> MassProperties:
        fmf = self.front_mass_fraction.value
        if not 0.0 < fmf < 1.0:
            raise ValueError(
                f"front_mass_fraction must be between 0 and 1, got {fmf}"
            )
        if self.driver_mass_kg.value > self.total_mass_kg.value:
            raise ValueError(
                f"driver_mass_kg ({self.driver_mass_kg.value}) exceeds "
                f"total_mass_kg ({self.total_mass_kg.value})"
            )
        return self

    def has_estimates(self) -> bool:
        """True if any field is tagged source='estimate'."""
        fields = [
            self.total_mass_kg, self.driver_mass_kg, self.cg_height_mm,
            self.cg_x_mm, self.front_mass_fraction,
            self.yaw_inertia_kgm2, self.roll_inertia_kgm2,
        ]
        return any(f.source == "estimate" for f in fields)

    def estimate_fields(self) -> list[str]:
        """Names of fields currently tagged as estimates."""
        result: list[str] = []
        for name in [
            "total_mass_kg", "driver_mass_kg", "cg_height_mm",
            "cg_x_mm", "front_mass_fraction",
            "yaw_inertia_kgm2", "roll_inertia_kgm2",
        ]:
            pf: ProvenanceFloat = getattr(self, name)
            if pf.source == "estimate":
                result.append(name)
        return result


class UnsprungMass(BaseModel, frozen=True):
    """Unsprung mass for a single suspension corner.

    CG height is measured from the ground. For a typical FSAE car
    with 10" wheels and no heavy uprights, this is roughly at
    wheel centre height.
    """

    mass_kg: ProvenanceFloat
    cg_height_mm: ProvenanceFloat

    def has_estimates(self) -> bool:
        """True if any field is tagged source='estimate'."""
        return (
            self.mass_kg.source == "estimate"
            or self.cg_height_mm.source == "estimate"
        )


class UnsprungMassSet(BaseModel, frozen=True):
    """Unsprung masses for all four corners."""

    fl: UnsprungMass
    fr: UnsprungMass
    rl: UnsprungMass
    rr: UnsprungMass

    def total_kg(self) -> float:
        """Total unsprung mass across all corners."""
        return (
            self.fl.mass_kg.value
            + self.fr.mass_kg.value
            + self.rl.mass_kg.value
            + self.rr.mass_kg.value
        )

    def front_pair(self) -> tuple[UnsprungMass, UnsprungMass]:
        """(left, right) for the front axle."""
        return (self.fl, self.fr)

    def rear_pair(self) -> tuple[UnsprungMass, UnsprungMass]:
        """(left, right) for the rear axle."""
        return (self.rl, self.rr)
