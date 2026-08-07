"""Tests for vdcore.geometry.primitives — Point3D, Vector3D, frame constants."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vdcore.geometry.primitives import Point3D, Vector3D, X_HAT, Y_HAT, Z_HAT


class TestFrameConstants:
    def test_handedness(self) -> None:
        """X × Y = Z (right-handed coordinate system)."""
        np.testing.assert_allclose(np.cross(X_HAT, Y_HAT), Z_HAT)

    def test_orthonormal(self) -> None:
        for v in (X_HAT, Y_HAT, Z_HAT):
            assert np.linalg.norm(v) == pytest.approx(1.0)
        assert np.dot(X_HAT, Y_HAT) == pytest.approx(0.0)
        assert np.dot(X_HAT, Z_HAT) == pytest.approx(0.0)
        assert np.dot(Y_HAT, Z_HAT) == pytest.approx(0.0)


class TestPoint3D:
    def test_construction(self) -> None:
        p = Point3D(x_mm=1.0, y_mm=2.0, z_mm=3.0)
        assert p.x_mm == 1.0
        assert p.y_mm == 2.0
        assert p.z_mm == 3.0

    def test_frozen(self) -> None:
        p = Point3D(x_mm=1.0, y_mm=2.0, z_mm=3.0)
        with pytest.raises(AttributeError):
            p.x_mm = 99.0  # type: ignore[misc]

    def test_to_array(self) -> None:
        p = Point3D(x_mm=1.0, y_mm=2.0, z_mm=3.0)
        np.testing.assert_array_equal(p.to_array(), [1.0, 2.0, 3.0])

    def test_from_array(self) -> None:
        arr = np.array([4.0, 5.0, 6.0])
        p = Point3D.from_array(arr)
        assert p.x_mm == 4.0
        assert p.y_mm == 5.0
        assert p.z_mm == 6.0

    def test_distance_to(self) -> None:
        a = Point3D(x_mm=0.0, y_mm=0.0, z_mm=0.0)
        b = Point3D(x_mm=3.0, y_mm=4.0, z_mm=0.0)
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_distance_to_self(self) -> None:
        p = Point3D(x_mm=1.0, y_mm=2.0, z_mm=3.0)
        assert p.distance_to(p) == pytest.approx(0.0)

    def test_subtraction_gives_vector(self) -> None:
        a = Point3D(x_mm=5.0, y_mm=3.0, z_mm=1.0)
        b = Point3D(x_mm=1.0, y_mm=1.0, z_mm=1.0)
        v = a - b
        assert isinstance(v, Vector3D)
        assert v.x == pytest.approx(4.0)
        assert v.y == pytest.approx(2.0)
        assert v.z == pytest.approx(0.0)


class TestVector3D:
    def test_magnitude(self) -> None:
        v = Vector3D(x=3.0, y=4.0, z=0.0)
        assert v.magnitude() == pytest.approx(5.0)

    def test_normalize(self) -> None:
        v = Vector3D(x=0.0, y=0.0, z=5.0)
        n = v.normalize()
        assert n.magnitude() == pytest.approx(1.0)
        assert n.z == pytest.approx(1.0)

    def test_normalize_zero_raises(self) -> None:
        v = Vector3D(x=0.0, y=0.0, z=0.0)
        with pytest.raises(ValueError, match="zero-length"):
            v.normalize()

    def test_dot(self) -> None:
        a = Vector3D(x=1.0, y=0.0, z=0.0)
        b = Vector3D(x=0.0, y=1.0, z=0.0)
        assert a.dot(b) == pytest.approx(0.0)

    def test_cross(self) -> None:
        a = Vector3D(x=1.0, y=0.0, z=0.0)
        b = Vector3D(x=0.0, y=1.0, z=0.0)
        c = a.cross(b)
        assert c.x == pytest.approx(0.0)
        assert c.y == pytest.approx(0.0)
        assert c.z == pytest.approx(1.0)

    def test_angle_to_deg_orthogonal(self) -> None:
        a = Vector3D(x=1.0, y=0.0, z=0.0)
        b = Vector3D(x=0.0, y=1.0, z=0.0)
        assert a.angle_to_deg(b) == pytest.approx(90.0)

    def test_angle_to_deg_parallel(self) -> None:
        a = Vector3D(x=1.0, y=0.0, z=0.0)
        b = Vector3D(x=2.0, y=0.0, z=0.0)
        assert a.angle_to_deg(b) == pytest.approx(0.0)

    def test_angle_to_deg_45(self) -> None:
        a = Vector3D(x=1.0, y=0.0, z=0.0)
        b = Vector3D(x=1.0, y=1.0, z=0.0)
        assert a.angle_to_deg(b) == pytest.approx(45.0)

    def test_from_points(self) -> None:
        a = Point3D(x_mm=1.0, y_mm=2.0, z_mm=3.0)
        b = Point3D(x_mm=4.0, y_mm=6.0, z_mm=3.0)
        v = Vector3D.from_points(a, b)
        assert v.x == pytest.approx(3.0)
        assert v.y == pytest.approx(4.0)
        assert v.z == pytest.approx(0.0)

    def test_from_array(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        v = Vector3D.from_array(arr)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0
