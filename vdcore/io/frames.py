"""Coordinate frame transforms between ISO 8855 and other systems.

This project uses ISO 8855: X+ forward, Y+ LEFT, Z+ up.
All transforms are 3x3 sign/permutation matrices.
To convert: v_target = M @ v_source.

Frames:
  ISO 8855 (this project): X+ fwd, Y+ left, Z+ up   (right-handed)
  J670e (SAE z-down):      X+ fwd, Y+ right, Z+ down (right-handed)
  Legacy project frame:    X+ fwd, Y+ left, Z+ up    (identity — same as ISO 8855)
  Optimum Kinematics:      X+ fwd, Y+ right, Z+ up   (left-handed)
  SolidWorks (FSAE typ.):  X+ rear, Y+ right, Z+ up  (left-handed)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

Vec3 = npt.NDArray[np.floating[Any]]

# --- Transform matrices ---

M_ISO8855_TO_J670E: npt.NDArray[np.float64] = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
)

M_J670E_TO_ISO8855: npt.NDArray[np.float64] = M_ISO8855_TO_J670E  # self-inverse

M_ISO8855_TO_LEGACY: npt.NDArray[np.float64] = np.eye(3)

M_LEGACY_TO_ISO8855: npt.NDArray[np.float64] = np.eye(3)

M_ISO8855_TO_OPTIMUM_K: npt.NDArray[np.float64] = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
)

M_OPTIMUM_K_TO_ISO8855: npt.NDArray[np.float64] = M_ISO8855_TO_OPTIMUM_K  # self-inverse

M_ISO8855_TO_SOLIDWORKS: npt.NDArray[np.float64] = np.array(
    [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
)

M_SOLIDWORKS_TO_ISO8855: npt.NDArray[np.float64] = M_ISO8855_TO_SOLIDWORKS  # self-inverse


def _transform(matrix: npt.NDArray[np.float64], v: Vec3) -> Vec3:
    """Apply a 3x3 transform to a 3-vector or (N,3) array of vectors."""
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        return matrix @ v
    return (matrix @ v.T).T


def iso8855_to_j670e(v: Vec3) -> Vec3:
    """ISO 8855 (X fwd, Y left, Z up) → J670e (X fwd, Y right, Z down)."""
    return _transform(M_ISO8855_TO_J670E, v)


def j670e_to_iso8855(v: Vec3) -> Vec3:
    """J670e (X fwd, Y right, Z down) → ISO 8855 (X fwd, Y left, Z up)."""
    return _transform(M_J670E_TO_ISO8855, v)


def iso8855_to_legacy(v: Vec3) -> Vec3:
    """ISO 8855 → legacy project frame. Identity (same axes)."""
    return _transform(M_ISO8855_TO_LEGACY, v)


def legacy_to_iso8855(v: Vec3) -> Vec3:
    """Legacy project frame → ISO 8855. Identity (same axes)."""
    return _transform(M_LEGACY_TO_ISO8855, v)


def iso8855_to_optimum_k(v: Vec3) -> Vec3:
    """ISO 8855 (Y left) → Optimum Kinematics (Y right, Z up)."""
    return _transform(M_ISO8855_TO_OPTIMUM_K, v)


def optimum_k_to_iso8855(v: Vec3) -> Vec3:
    """Optimum Kinematics (Y right, Z up) → ISO 8855 (Y left)."""
    return _transform(M_OPTIMUM_K_TO_ISO8855, v)


def iso8855_to_solidworks(v: Vec3) -> Vec3:
    """ISO 8855 → SolidWorks typical FSAE (X rear, Y right, Z up)."""
    return _transform(M_ISO8855_TO_SOLIDWORKS, v)


def solidworks_to_iso8855(v: Vec3) -> Vec3:
    """SolidWorks typical FSAE (X rear, Y right, Z up) → ISO 8855."""
    return _transform(M_SOLIDWORKS_TO_ISO8855, v)
