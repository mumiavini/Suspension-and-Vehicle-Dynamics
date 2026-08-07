"""Tests for vdcore.io.config — vehicle JSON save/load."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vdcore.io.config import load_vehicle, save_vehicle
from vdcore.models.hardpoint import (
    Axle,
    Corner,
    Hardpoint,
    TirePackage,
    Vehicle,
)


def _hp(name: str, x: float, y: float, z: float) -> Hardpoint:
    return Hardpoint(name=name, x_mm=x, y_mm=y, z_mm=z, source="cad", tol_mm=0.5)


def _tire() -> TirePackage:
    return TirePackage(loaded_radius_mm=228.0, source="cad", tol_mm=1.0)


def _make_vehicle() -> Vehicle:
    fl = Corner(
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
    fr = Corner(
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
    rl = Corner(**{**fl.model_dump(), "corner_id": "RL"})
    rr = Corner(**{**fr.model_dump(), "corner_id": "RR"})
    return Vehicle(front=Axle(left=fl, right=fr), rear=Axle(left=rl, right=rr))


def test_round_trip(tmp_path: Path) -> None:
    """save then load must return an identical Vehicle."""
    v = _make_vehicle()
    path = tmp_path / "vehicle.json"
    save_vehicle(v, path)
    v2 = load_vehicle(path)
    assert v == v2


def test_saved_json_excludes_derived(tmp_path: Path) -> None:
    """The JSON file must not contain wheelbase_mm or track_mm."""
    v = _make_vehicle()
    path = tmp_path / "vehicle.json"
    save_vehicle(v, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "wheelbase_mm" not in data
    assert "track_mm" not in data
    assert "contact_patch" not in data


def test_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    """A config with hand-edited derived values must be rejected."""
    v = _make_vehicle()
    path = tmp_path / "vehicle.json"
    save_vehicle(v, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["wheelbase_mm"] = 1550.0
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown top-level keys"):
        load_vehicle(path)


def test_rejects_future_schema_version(tmp_path: Path) -> None:
    """A config with a higher schema version must be rejected."""
    v = _make_vehicle()
    path = tmp_path / "vehicle.json"
    save_vehicle(v, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = 999
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported schema version"):
        load_vehicle(path)


def test_schema_version_in_output(tmp_path: Path) -> None:
    """Saved JSON must contain schema_version."""
    v = _make_vehicle()
    path = tmp_path / "vehicle.json"
    save_vehicle(v, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
