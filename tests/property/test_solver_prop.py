"""Hypothesis property tests for vdcore.geometry.solver."""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from vdcore.geometry.solver import DWSolver
from vdcore.models.hardpoint import Corner, Hardpoint, TirePackage


@st.composite
def realistic_corner(draw: st.DrawFn, corner_id: str = "FL") -> Corner:
    """Generate a geometrically plausible suspension corner.

    Keeps hardpoints within physically sensible ranges for an FSAE car
    to ensure the solver can find a solution.
    """
    y_sign = 1.0 if corner_id in ("FL", "RL") else -1.0

    # Inboard points near the chassis centreline
    ib_y = draw(st.floats(min_value=100, max_value=200, allow_nan=False, allow_infinity=False))
    # Outboard points further out
    ob_y = draw(st.floats(min_value=450, max_value=600, allow_nan=False, allow_infinity=False))
    # UCA higher than LCA
    uca_z = draw(st.floats(min_value=250, max_value=320, allow_nan=False, allow_infinity=False))
    lca_z = draw(st.floats(min_value=60, max_value=100, allow_nan=False, allow_infinity=False))

    wc_y = draw(st.floats(min_value=550, max_value=650, allow_nan=False, allow_infinity=False))
    wc_z = draw(st.floats(min_value=180, max_value=230, allow_nan=False, allow_infinity=False))

    def hp(name: str, x: float, y: float, z: float) -> Hardpoint:
        return Hardpoint(name=name, x_mm=x, y_mm=y_sign * y, z_mm=z, source="cad", tol_mm=0.5)

    camber = draw(st.floats(min_value=-4.0, max_value=0.0, allow_nan=False, allow_infinity=False))
    toe = draw(st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False))

    return Corner(
        corner_id=corner_id,
        uca_inboard_front=hp("UCA_IF", 80, ib_y, uca_z),
        uca_inboard_rear=hp("UCA_IR", -80, ib_y, uca_z),
        uca_outboard=hp("UCA_O", 0, ob_y, uca_z + 10),
        lca_inboard_front=hp("LCA_IF", 100, ib_y - 20, lca_z),
        lca_inboard_rear=hp("LCA_IR", -100, ib_y - 20, lca_z),
        lca_outboard=hp("LCA_O", 0, ob_y + 50, lca_z - 5),
        tie_rod_inboard=hp("TR_I", -60, ib_y + 10, 120),
        tie_rod_outboard=hp("TR_O", -50, ob_y + 10, 110),
        wheel_center=hp("WC", 0, wc_y, wc_z),
        tire=TirePackage(loaded_radius_mm=228.0, source="cad", tol_mm=1.0),
        static_camber_deg=camber,
        static_toe_deg_per_side=toe,
    )


@given(corner=realistic_corner("FL"))
@settings(max_examples=20, deadline=5000)
def test_static_always_converges(corner: Corner) -> None:
    """Static solve must converge for realistic geometry."""
    solver = DWSolver(corner)
    result = solver.solve()
    assert result.converged, f"Failed to converge, residual: {result.residual_norm:.2e}"


@given(corner=realistic_corner("FL"))
@settings(max_examples=10, deadline=5000)
def test_zero_travel_recovers_static(corner: Corner) -> None:
    """Zero wheel travel must recover the static wheel centre position."""
    solver = DWSolver(corner)
    result = solver.solve(wheel_travel_mm=0.0)
    assume(result.converged)
    assert result.wheel_center.y_mm == pytest.approx(corner.wheel_center.y_mm, abs=0.1)
    assert result.wheel_center.z_mm == pytest.approx(corner.wheel_center.z_mm, abs=0.1)


@given(data=st.data())
@settings(max_examples=10, deadline=10000)
def test_symmetric_geometry_gives_equal_camber(data: st.DataObject) -> None:
    """Mirrored left/right geometry must give the same camber value."""
    fl = data.draw(realistic_corner("FL"))
    # Mirror to FR
    fr = Corner(
        corner_id="FR",
        uca_inboard_front=Hardpoint(name="UCA_IF", x_mm=fl.uca_inboard_front.x_mm, y_mm=-fl.uca_inboard_front.y_mm, z_mm=fl.uca_inboard_front.z_mm, source="cad", tol_mm=0.5),
        uca_inboard_rear=Hardpoint(name="UCA_IR", x_mm=fl.uca_inboard_rear.x_mm, y_mm=-fl.uca_inboard_rear.y_mm, z_mm=fl.uca_inboard_rear.z_mm, source="cad", tol_mm=0.5),
        uca_outboard=Hardpoint(name="UCA_O", x_mm=fl.uca_outboard.x_mm, y_mm=-fl.uca_outboard.y_mm, z_mm=fl.uca_outboard.z_mm, source="cad", tol_mm=0.5),
        lca_inboard_front=Hardpoint(name="LCA_IF", x_mm=fl.lca_inboard_front.x_mm, y_mm=-fl.lca_inboard_front.y_mm, z_mm=fl.lca_inboard_front.z_mm, source="cad", tol_mm=0.5),
        lca_inboard_rear=Hardpoint(name="LCA_IR", x_mm=fl.lca_inboard_rear.x_mm, y_mm=-fl.lca_inboard_rear.y_mm, z_mm=fl.lca_inboard_rear.z_mm, source="cad", tol_mm=0.5),
        lca_outboard=Hardpoint(name="LCA_O", x_mm=fl.lca_outboard.x_mm, y_mm=-fl.lca_outboard.y_mm, z_mm=fl.lca_outboard.z_mm, source="cad", tol_mm=0.5),
        tie_rod_inboard=Hardpoint(name="TR_I", x_mm=fl.tie_rod_inboard.x_mm, y_mm=-fl.tie_rod_inboard.y_mm, z_mm=fl.tie_rod_inboard.z_mm, source="cad", tol_mm=0.5),
        tie_rod_outboard=Hardpoint(name="TR_O", x_mm=fl.tie_rod_outboard.x_mm, y_mm=-fl.tie_rod_outboard.y_mm, z_mm=fl.tie_rod_outboard.z_mm, source="cad", tol_mm=0.5),
        wheel_center=Hardpoint(name="WC", x_mm=fl.wheel_center.x_mm, y_mm=-fl.wheel_center.y_mm, z_mm=fl.wheel_center.z_mm, source="cad", tol_mm=0.5),
        tire=fl.tire,
        static_camber_deg=fl.static_camber_deg,
        static_toe_deg_per_side=fl.static_toe_deg_per_side,
    )

    solver_l = DWSolver(fl)
    solver_r = DWSolver(fr)
    rl = solver_l.solve()
    rr = solver_r.solve()
    assume(rl.converged and rr.converged)

    assert rl.camber_deg == pytest.approx(rr.camber_deg, abs=0.05)


@given(corner=realistic_corner("FL"))
@settings(max_examples=10, deadline=5000)
def test_convergence_always_reported(corner: Corner) -> None:
    """Every solve must report convergence status, regardless of outcome."""
    solver = DWSolver(corner)
    result = solver.solve(wheel_travel_mm=5.0)
    assert isinstance(result.converged, bool)
    assert isinstance(result.residual_norm, float)
    assert isinstance(result.nfev, int)
    assert isinstance(result.njev, int)
