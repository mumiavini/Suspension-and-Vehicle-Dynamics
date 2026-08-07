"""Benchmark / known-answer tests for the camber solver.

These test against hand-computed or reference values for specific
suspension geometries, serving as regression anchors.
"""

from __future__ import annotations

import pytest

from vdcore.analysis.camber import camber_gain_deg_per_mm, static_camber_deg
from vdcore.geometry.solver import DWSolver
from vdcore.models.hardpoint import Corner, Hardpoint, TirePackage


def _hp(name: str, x: float, y: float, z: float) -> Hardpoint:
    return Hardpoint(name=name, x_mm=x, y_mm=y, z_mm=z, source="cad", tol_mm=0.5)


def _tire(r: float = 228.0) -> TirePackage:
    return TirePackage(loaded_radius_mm=r, source="cad", tol_mm=1.0)


def _symmetric_planar_fl() -> Corner:
    """A planar (zero-caster) symmetric front-left corner.

    UCA and LCA axes are parallel to X (front-rear line), both
    at the same X offset. This gives a pure 2D front-view linkage.
    UCA_O is inboard of LCA_O → negative static camber.
    """
    return Corner(
        corner_id="FL",
        uca_inboard_front=_hp("UCA_IF", 100, 150, 280),
        uca_inboard_rear=_hp("UCA_IR", -100, 150, 280),
        uca_outboard=_hp("UCA_O", 0, 530, 290),
        lca_inboard_front=_hp("LCA_IF", 100, 130, 80),
        lca_inboard_rear=_hp("LCA_IR", -100, 130, 80),
        lca_outboard=_hp("LCA_O", 0, 580, 75),
        tie_rod_inboard=_hp("TR_I", -60, 160, 120),
        tie_rod_outboard=_hp("TR_O", -50, 540, 110),
        wheel_center=_hp("WC", 0, 600, 200),
        tire=_tire(),
        static_camber_deg=-2.0,
        static_toe_deg_per_side=0.0,
    )


class TestBenchmarkSymmetricPlanar:
    """Benchmark: symmetric planar front-left corner.

    Reference values computed from this solver on initial validation.
    If these change, the solver behaviour has regressed.
    """

    def test_static_camber_value(self) -> None:
        """Static camber equals the design-intent input at zero wheel travel."""
        c = static_camber_deg(_symmetric_planar_fl())
        assert c < 0
        assert c == pytest.approx(-2.0, abs=0.01)

    def test_camber_gain_value(self) -> None:
        """Camber gain for the reference geometry.

        With UCA shorter than LCA, bump produces more negative camber,
        so d(camber_deg)/d(wheel_travel_mm) is negative.
        """
        gain = camber_gain_deg_per_mm(_symmetric_planar_fl(), wheel_travel_range_mm=25.0, steps=50)
        assert abs(gain) < 0.5
        assert gain == pytest.approx(-0.0099, abs=0.01)

    def test_wheel_travel_sweep_endpoints(self) -> None:
        """Camber at ±25mm wheel travel must differ from static."""
        solver = DWSolver(_symmetric_planar_fl())
        r_static = solver.solve()
        r_bump = solver.solve(wheel_travel_mm=25.0)
        r_droop = solver.solve(wheel_travel_mm=-25.0)

        assert all(r.converged for r in [r_static, r_bump, r_droop])
        assert r_bump.camber_deg != pytest.approx(r_static.camber_deg, abs=0.01)
        assert r_droop.camber_deg != pytest.approx(r_static.camber_deg, abs=0.01)


class TestBenchmarkLeftRightSymmetry:
    """Benchmark: left-right mirror must produce identical angles."""

    def test_fl_fr_static_camber_match(self) -> None:
        fl = _symmetric_planar_fl()
        fr = Corner(
            corner_id="FR",
            uca_inboard_front=_hp("UCA_IF", 100, -150, 280),
            uca_inboard_rear=_hp("UCA_IR", -100, -150, 280),
            uca_outboard=_hp("UCA_O", 0, -530, 290),
            lca_inboard_front=_hp("LCA_IF", 100, -130, 80),
            lca_inboard_rear=_hp("LCA_IR", -100, -130, 80),
            lca_outboard=_hp("LCA_O", 0, -580, 75),
            tie_rod_inboard=_hp("TR_I", -60, -160, 120),
            tie_rod_outboard=_hp("TR_O", -50, -540, 110),
            wheel_center=_hp("WC", 0, -600, 200),
            tire=_tire(),
            static_camber_deg=-2.0,
            static_toe_deg_per_side=0.0,
        )
        cl = static_camber_deg(fl)
        cr = static_camber_deg(fr)
        assert cl == pytest.approx(cr, abs=0.01)

    def test_fl_fr_camber_gain_match(self) -> None:
        fl = _symmetric_planar_fl()
        fr = Corner(
            corner_id="FR",
            uca_inboard_front=_hp("UCA_IF", 100, -150, 280),
            uca_inboard_rear=_hp("UCA_IR", -100, -150, 280),
            uca_outboard=_hp("UCA_O", 0, -530, 290),
            lca_inboard_front=_hp("LCA_IF", 100, -130, 80),
            lca_inboard_rear=_hp("LCA_IR", -100, -130, 80),
            lca_outboard=_hp("LCA_O", 0, -580, 75),
            tie_rod_inboard=_hp("TR_I", -60, -160, 120),
            tie_rod_outboard=_hp("TR_O", -50, -540, 110),
            wheel_center=_hp("WC", 0, -600, 200),
            tire=_tire(),
            static_camber_deg=-2.0,
            static_toe_deg_per_side=0.0,
        )
        gl = camber_gain_deg_per_mm(fl)
        gr = camber_gain_deg_per_mm(fr)
        assert gl == pytest.approx(gr, abs=0.001)
