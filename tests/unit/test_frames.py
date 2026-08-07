"""Tests for vdcore.io.frames — coordinate frame transforms."""

from __future__ import annotations

import numpy as np
import pytest

from vdcore.io.frames import (
    M_ISO8855_TO_J670E,
    M_ISO8855_TO_OPTIMUM_K,
    M_ISO8855_TO_SOLIDWORKS,
    M_ISO8855_TO_LEGACY,
    iso8855_to_j670e,
    iso8855_to_legacy,
    iso8855_to_optimum_k,
    iso8855_to_solidworks,
    j670e_to_iso8855,
    legacy_to_iso8855,
    optimum_k_to_iso8855,
    solidworks_to_iso8855,
)

PAIRS = [
    ("ISO8855↔J670e", iso8855_to_j670e, j670e_to_iso8855, M_ISO8855_TO_J670E),
    ("ISO8855↔Legacy", iso8855_to_legacy, legacy_to_iso8855, M_ISO8855_TO_LEGACY),
    ("ISO8855↔OptimumK", iso8855_to_optimum_k, optimum_k_to_iso8855, M_ISO8855_TO_OPTIMUM_K),
    ("ISO8855↔SolidWorks", iso8855_to_solidworks, solidworks_to_iso8855, M_ISO8855_TO_SOLIDWORKS),
]


@pytest.mark.parametrize("label,fwd,rev,matrix", PAIRS, ids=[p[0] for p in PAIRS])
class TestRoundTrip:
    def test_single_vector_round_trip(
        self, label: str, fwd: object, rev: object, matrix: object
    ) -> None:
        """Transforming forward then backward must recover the original vector."""
        v = np.array([100.0, 200.0, 300.0])
        result = rev(fwd(v))  # type: ignore[operator]
        np.testing.assert_allclose(result, v, atol=1e-12)

    def test_batch_round_trip(
        self, label: str, fwd: object, rev: object, matrix: object
    ) -> None:
        """Round-trip must work for (N,3) arrays."""
        vs = np.array([[1, 2, 3], [4, 5, 6], [-1, -2, -3]], dtype=float)
        result = rev(fwd(vs))  # type: ignore[operator]
        np.testing.assert_allclose(result, vs, atol=1e-12)

    def test_matrix_self_inverse(
        self, label: str, fwd: object, rev: object, matrix: np.ndarray  # type: ignore[type-arg]
    ) -> None:
        """All current transforms are self-inverse (M @ M = I)."""
        np.testing.assert_allclose(matrix @ matrix, np.eye(3), atol=1e-12)


class TestJ670eSpecific:
    def test_y_negation(self) -> None:
        """ISO 8855 Y+ left → J670e Y+ right: Y is negated."""
        v = np.array([0.0, 100.0, 0.0])
        result = iso8855_to_j670e(v)
        assert result[1] == pytest.approx(-100.0)

    def test_z_negation(self) -> None:
        """ISO 8855 Z+ up → J670e Z+ down: Z is negated."""
        v = np.array([0.0, 0.0, 200.0])
        result = iso8855_to_j670e(v)
        assert result[2] == pytest.approx(-200.0)

    def test_x_preserved(self) -> None:
        """Both frames have X+ forward."""
        v = np.array([300.0, 0.0, 0.0])
        result = iso8855_to_j670e(v)
        assert result[0] == pytest.approx(300.0)


class TestOptimumKSpecific:
    def test_y_negation_only(self) -> None:
        """ISO 8855 → Optimum K: only Y is negated (left→right)."""
        v = np.array([10.0, 20.0, 30.0])
        result = iso8855_to_optimum_k(v)
        np.testing.assert_allclose(result, [10.0, -20.0, 30.0])


class TestSolidWorksSpecific:
    def test_x_and_y_negated(self) -> None:
        """ISO 8855 → SolidWorks: X negated (fwd→rear), Y negated (left→right)."""
        v = np.array([10.0, 20.0, 30.0])
        result = iso8855_to_solidworks(v)
        np.testing.assert_allclose(result, [-10.0, -20.0, 30.0])


class TestLegacy:
    def test_identity(self) -> None:
        """Legacy frame is identical to ISO 8855."""
        v = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(iso8855_to_legacy(v), v)
        np.testing.assert_allclose(legacy_to_iso8855(v), v)
