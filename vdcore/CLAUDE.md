# vdcore — Pure Library

## Purity rule

This package may import: `numpy`, `scipy`, `pydantic`, `polars`.
It may **NEVER** import: `streamlit`, `plotly`, `PySide6`, `pyqtgraph`, `pyvista`, `matplotlib`.
Every function must be callable from a bare Python REPL.

## Coordinate system — ISO 8855

X+ forward, Y+ LEFT, Z+ up. Right-handed (X × Y = Z).

**Y+ is LEFT.** Left wheels have positive Y. Right wheels have negative Y.

State the coordinate frame in every docstring that returns a vector or a Y coordinate. The camber extraction function must handle left/right corners explicitly — the sign of the wheel-plane normal differs between sides.

## Numerical conventions

- Solvers return convergence status. Non-convergence raises or returns `converged=False` — never a plausible-looking number.
- Use scipy's default trust-region method (not LM). Numerical Jacobian by default.
- Seed random state for reproducibility in optimizers.

## Typing

Full type hints on all public and private functions. mypy strict-equivalent flags must pass.
Use `Literal` types for enums where the set is small and fixed.
Frozen pydantic models for data classes.
