"""Tests for the Explained[T] type and its retrofits.

Verifies:
  - Construction and immutability
  - Every Explained result lists ≥1 assumption
  - inputs dict keys ⊇ variable names in formula string
  - estimate inputs are flagged
  - _explained variants match their plain counterparts
"""

from __future__ import annotations

import re

import pytest

from vdcore.analysis.load_transfer import (
    lateral_load_transfer,
    lateral_load_transfer_explained,
)
from vdcore.analysis.roll_gradient import (
    roll_gradient_deg_per_g,
    roll_gradient_deg_per_g_explained,
)
from vdcore.explain import Explained
from vdcore.models.mass import (
    MassProperties,
    ProvenanceFloat,
    UnsprungMass,
    UnsprungMassSet,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _pf(value: float, tol: float, source: str = "estimate") -> ProvenanceFloat:
    return ProvenanceFloat(value=value, source=source, tol=tol)  # type: ignore[arg-type]


def _mass_props() -> MassProperties:
    return MassProperties(
        total_mass_kg=_pf(300.0, 2.0),
        driver_mass_kg=_pf(75.0, 1.0),
        cg_height_mm=_pf(300.0, 20.0),
        cg_x_mm=_pf(800.0, 10.0),
        front_mass_fraction=_pf(0.484, 0.01),
        yaw_inertia_kgm2=_pf(108.0, 30.0),
        roll_inertia_kgm2=_pf(17.0, 5.0),
    )


def _unsprung() -> UnsprungMassSet:
    corner = UnsprungMass(
        mass_kg=_pf(15.0, 1.0),
        cg_height_mm=_pf(254.0, 5.0),
    )
    return UnsprungMassSet(fl=corner, fr=corner, rl=corner, rr=corner)


_RG_KW: dict[str, float] = dict(
    front_rc_height_mm=50.0,
    rear_rc_height_mm=60.0,
    front_roll_stiffness_nm_per_deg=500.0,
    rear_roll_stiffness_nm_per_deg=350.0,
)

_LT_KW: dict[str, float] = dict(
    ay_g=1.0,
    front_rc_height_mm=50.0,
    rear_rc_height_mm=60.0,
    front_track_mm=1220.0,
    rear_track_mm=1220.0,
    front_roll_stiffness_nm_per_deg=500.0,
    rear_roll_stiffness_nm_per_deg=350.0,
    wheelbase_mm=1550.0,
)


def _formula_variables(formula: str) -> set[str]:
    """Extract plausible variable names from a formula string.

    Matches identifiers that are NOT known constants or operators.
    """
    # Match sequences of letters, digits, and underscores that start with a letter
    all_tokens = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", formula))
    # Remove known non-variable tokens
    non_vars = {"sum", "min", "max", "abs", "sqrt", "sin", "cos", "tan", "log"}
    return all_tokens - non_vars


# ---------------------------------------------------------------------------
# Explained type tests
# ---------------------------------------------------------------------------


class TestExplainedType:
    def test_construction(self) -> None:
        ex = Explained(
            value=1.5,
            formula="ay = F / (m * g)",
            inputs={
                "F": (1000.0, "N", "computed"),
                "m": (300.0, "kg", "estimate"),
                "g": (9.81, "m/s^2", "computed"),
            },
            intermediates={"total_force": 1000.0},
            reference="Test reference",
            assumptions=["Test assumption"],
        )
        assert ex.value == 1.5
        assert len(ex.assumptions) >= 1

    def test_frozen(self) -> None:
        ex = Explained(
            value=1.0,
            formula="x = a",
            inputs={"a": (1.0, "", "computed")},
            intermediates={},
            reference="",
            assumptions=["none"],
        )
        with pytest.raises(Exception):
            ex.value = 2.0  # type: ignore[misc]

    def test_has_estimate_inputs(self) -> None:
        ex = Explained(
            value=1.0,
            formula="x = a",
            inputs={
                "a": (1.0, "", "estimate"),
                "b": (2.0, "", "measured"),
            },
            intermediates={},
            reference="",
            assumptions=["test"],
        )
        assert ex.has_estimate_inputs()
        assert ex.estimate_input_names() == ["a"]

    def test_no_estimate_inputs(self) -> None:
        ex = Explained(
            value=1.0,
            formula="x = a",
            inputs={"a": (1.0, "", "measured")},
            intermediates={},
            reference="",
            assumptions=["test"],
        )
        assert not ex.has_estimate_inputs()
        assert ex.estimate_input_names() == []


# ---------------------------------------------------------------------------
# Roll gradient explained
# ---------------------------------------------------------------------------


class TestRollGradientExplained:
    def test_matches_plain(self) -> None:
        """_explained variant must return the same value as the plain function."""
        plain = roll_gradient_deg_per_g(_mass_props(), _unsprung(), **_RG_KW)
        explained = roll_gradient_deg_per_g_explained(
            _mass_props(), _unsprung(), **_RG_KW,
        )
        assert explained.value == pytest.approx(plain)

    def test_has_assumptions(self) -> None:
        ex = roll_gradient_deg_per_g_explained(
            _mass_props(), _unsprung(), **_RG_KW,
        )
        assert len(ex.assumptions) >= 1

    def test_formula_variables_in_inputs(self) -> None:
        """Every variable in the formula must appear in the inputs dict."""
        ex = roll_gradient_deg_per_g_explained(
            _mass_props(), _unsprung(), **_RG_KW,
        )
        formula_vars = _formula_variables(ex.formula)
        input_keys = set(ex.inputs.keys())
        # Formula uses short names (m_s, g, d, K_roll); these must be in inputs.
        missing = formula_vars - input_keys
        # Allow the result variable (left side of '=') to be absent from inputs.
        lhs = ex.formula.split("=")[0].strip()
        missing.discard(lhs)
        assert not missing, f"Formula variables not in inputs: {missing}"

    def test_flags_estimate_inputs(self) -> None:
        ex = roll_gradient_deg_per_g_explained(
            _mass_props(), _unsprung(), **_RG_KW,
        )
        assert ex.has_estimate_inputs()
        assert "cg_height_mm" in ex.estimate_input_names()


# ---------------------------------------------------------------------------
# Lateral load transfer explained
# ---------------------------------------------------------------------------


class TestLateralLTExplained:
    def test_matches_plain(self) -> None:
        plain = lateral_load_transfer(_mass_props(), _unsprung(), **_LT_KW)
        explained = lateral_load_transfer_explained(
            _mass_props(), _unsprung(), **_LT_KW,
        )
        assert explained.value.front.total_delta_fz_n == pytest.approx(
            plain.front.total_delta_fz_n
        )
        assert explained.value.rear.total_delta_fz_n == pytest.approx(
            plain.rear.total_delta_fz_n
        )
        assert explained.value.lltd == pytest.approx(plain.lltd)

    def test_has_assumptions(self) -> None:
        ex = lateral_load_transfer_explained(
            _mass_props(), _unsprung(), **_LT_KW,
        )
        assert len(ex.assumptions) >= 1

    def test_formula_variables_in_inputs(self) -> None:
        ex = lateral_load_transfer_explained(
            _mass_props(), _unsprung(), **_LT_KW,
        )
        formula_vars = _formula_variables(ex.formula)
        input_keys = set(ex.inputs.keys())
        lhs = ex.formula.split("=")[0].strip()
        missing = formula_vars - input_keys
        missing.discard(lhs)
        assert not missing, f"Formula variables not in inputs: {missing}"
