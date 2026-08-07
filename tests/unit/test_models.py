"""Tests for vdcore.models.hardpoint and vdcore.models.target."""

from __future__ import annotations

import pytest

from vdcore.models.hardpoint import (
    Axle,
    Corner,
    DerivedPoint,
    Hardpoint,
    TirePackage,
    Vehicle,
)
from vdcore.models.target import DesignTarget, DesignTargets


def _hp(name: str, x: float, y: float, z: float) -> Hardpoint:
    return Hardpoint(name=name, x_mm=x, y_mm=y, z_mm=z, source="cad", tol_mm=0.5)


def _tire() -> TirePackage:
    return TirePackage(loaded_radius_mm=228.0, source="cad", tol_mm=1.0)


def _fl_corner() -> Corner:
    """A left-front corner with positive Y (ISO 8855)."""
    return Corner(
        corner_id="FL",
        uca_inboard_front=_hp("UCA_IF", 80, 150, 280),
        uca_inboard_rear=_hp("UCA_IR", -80, 150, 280),
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


def _fr_corner() -> Corner:
    """A right-front corner with negative Y (ISO 8855)."""
    return Corner(
        corner_id="FR",
        uca_inboard_front=_hp("UCA_IF", 80, -150, 280),
        uca_inboard_rear=_hp("UCA_IR", -80, -150, 280),
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


class TestHardpoint:
    def test_construction(self) -> None:
        hp = _hp("test", 1, 2, 3)
        assert hp.name == "test"
        assert hp.x_mm == 1.0
        assert hp.source == "cad"

    def test_tol_mm_required(self) -> None:
        """tol_mm has no default — omitting it must raise."""
        with pytest.raises(Exception):
            Hardpoint(name="x", x_mm=0, y_mm=0, z_mm=0, source="cad")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        hp = _hp("test", 1, 2, 3)
        with pytest.raises(Exception):
            hp.x_mm = 99  # type: ignore[misc]


class TestTirePackage:
    def test_construction(self) -> None:
        t = TirePackage(loaded_radius_mm=228, source="estimate", tol_mm=10)
        assert t.loaded_radius_mm == 228.0

    def test_tol_mm_required(self) -> None:
        with pytest.raises(Exception):
            TirePackage(loaded_radius_mm=228, source="cad")  # type: ignore[call-arg]


class TestCorner:
    def test_fl_positive_y(self) -> None:
        """FL corner must accept positive Y."""
        c = _fl_corner()
        assert c.corner_id == "FL"
        assert c.wheel_center.y_mm > 0

    def test_fr_negative_y(self) -> None:
        """FR corner must accept negative Y."""
        c = _fr_corner()
        assert c.corner_id == "FR"
        assert c.wheel_center.y_mm < 0

    def test_fl_rejects_negative_y(self) -> None:
        """FL with negative-Y wheel centre must be rejected."""
        with pytest.raises(ValueError, match="positive Y"):
            Corner(
                corner_id="FL",
                uca_inboard_front=_hp("UCA_IF", 80, -150, 280),
                uca_inboard_rear=_hp("UCA_IR", -80, -150, 280),
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

    def test_fr_rejects_positive_y(self) -> None:
        """FR with positive-Y wheel centre must be rejected."""
        with pytest.raises(ValueError, match="negative Y"):
            Corner(
                corner_id="FR",
                uca_inboard_front=_hp("UCA_IF", 80, 150, 280),
                uca_inboard_rear=_hp("UCA_IR", -80, 150, 280),
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

    def test_has_estimates_false(self) -> None:
        assert not _fl_corner().has_estimates()

    def test_has_estimates_true(self) -> None:
        c = Corner(
            corner_id="FL",
            uca_inboard_front=Hardpoint(
                name="UCA_IF", x_mm=80, y_mm=150, z_mm=280,
                source="estimate", tol_mm=10,
            ),
            uca_inboard_rear=_hp("UCA_IR", -80, 150, 280),
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
        assert c.has_estimates()

    def test_hardpoints_returns_nine(self) -> None:
        assert len(_fl_corner().hardpoints()) == 9

    def test_serialization_excludes_no_derived(self) -> None:
        """Corner.model_dump() should NOT contain contact_patch or track_mm."""
        d = _fl_corner().model_dump()
        assert "contact_patch" not in d
        assert "track_mm" not in d


class TestAxle:
    def test_construction(self) -> None:
        axle = Axle(left=_fl_corner(), right=_fr_corner())
        assert axle.left.corner_id == "FL"
        assert axle.right.corner_id == "FR"

    def test_serialization_excludes_track(self) -> None:
        d = Axle(left=_fl_corner(), right=_fr_corner()).model_dump()
        assert "track_mm" not in d


class TestVehicle:
    def test_serialization_excludes_wheelbase(self) -> None:
        fl = _fl_corner()
        fr = _fr_corner()
        rl = Corner(**{**fl.model_dump(), "corner_id": "RL"})
        rr = Corner(**{**fr.model_dump(), "corner_id": "RR"})
        v = Vehicle(front=Axle(left=fl, right=fr), rear=Axle(left=rl, right=rr))
        d = v.model_dump()
        assert "wheelbase_mm" not in d
        assert v.schema_version == 1

    def test_round_trip_json(self) -> None:
        fl = _fl_corner()
        fr = _fr_corner()
        rl = Corner(**{**fl.model_dump(), "corner_id": "RL"})
        rr = Corner(**{**fr.model_dump(), "corner_id": "RR"})
        v = Vehicle(front=Axle(left=fl, right=fr), rear=Axle(left=rl, right=rr))
        data = v.model_dump()
        v2 = Vehicle.model_validate(data)
        assert v == v2


class TestDesignTarget:
    def test_construction(self) -> None:
        t = DesignTarget(
            name="static_camber",
            value=-2.0,
            unit="deg",
            tolerance=0.5,
            event="skidpad",
            rationale="Maximise lateral grip at 1g steady state",
        )
        assert t.event == "skidpad"
        assert t.rationale != ""

    def test_rationale_required(self) -> None:
        with pytest.raises(Exception):
            DesignTarget(  # type: ignore[call-arg]
                name="test", value=0, unit="mm", tolerance=1, event="all",
            )

    def test_targets_collection(self) -> None:
        t = DesignTarget(
            name="test", value=0, unit="mm", tolerance=1,
            event="all", rationale="test",
        )
        ts = DesignTargets(static=[t], dynamic=[])
        assert len(ts.static) == 1
        assert len(ts.dynamic) == 0
