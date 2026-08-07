"""Explained[T] — a computed value with full derivation audit trail.

Pure data, zero rendering. Downstream UI or report code is responsible
for presenting this in whatever format is appropriate (HTML, LaTeX, etc.).

The ``inputs`` dict maps each variable name to a ``(value, unit, source)``
triple.  The ``source`` string uses the same vocabulary as
``ProvenanceFloat.source`` — ``"cad"``, ``"measured"``, ``"estimate"``,
``"design_intent"`` — plus ``"computed"`` for values derived from other
computations (e.g. sprung mass = total - unsprung).

The ``formula`` string contains variable names that must appear as keys in
``inputs``.  This contract is enforced by tests, not at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Explained(Generic[T]):
    """A computed value with full derivation audit trail."""

    value: T
    formula: str
    inputs: dict[str, tuple[float, str, str]]
    intermediates: dict[str, float]
    reference: str
    assumptions: list[str]

    def has_estimate_inputs(self) -> bool:
        """True if any input has source ``"estimate"``."""
        return any(src == "estimate" for _, _, src in self.inputs.values())

    def estimate_input_names(self) -> list[str]:
        """Return names of inputs with source ``"estimate"``."""
        return [name for name, (_, _, src) in self.inputs.items() if src == "estimate"]
