"""VD Studio engine: project model and geometry resolution.

The app must run the SAME car as the reference pipeline: a default project
synthesises exactly the merged hardpoints ``scripts/geometry_summary.py``
builds, and freezing that synthesis into a hardpoint set changes nothing.

Frame: ISO 8855 (X+ forward, Y+ LEFT, Z+ up).
"""

from __future__ import annotations

import json
import math

import pytest

from apps.vdstudio.engine import geometry as G
from apps.vdstudio.engine.project import (
    CORNERS,
    INPUT_POINTS,
    Project,
    get_path,
    load_project,
    project_from_dict,
    project_to_dict,
    save_project,
    with_path,
)

import geometry_summary as gs  # noqa: E402  (on sys.path via apps.vdstudio)
import sla_geometry as sla  # noqa: E402
import steering_geometry as stg  # noqa: E402


@pytest.fixture(scope="module")
def reference() -> gs.MergedHardpoints:
    return gs.build_merged(sla.run(), stg.run(with_sweep=False))


def test_default_project_reproduces_geometry_summary(reference) -> None:
    geo = G.resolve(Project())
    assert set(geo.points) == set(reference.points)
    for c, pts in reference.points.items():
        assert set(pts) == set(geo.points[c])
        for n, xyz in pts.items():
            assert geo.points[c][n] == pytest.approx(xyz, abs=1e-9), (c, n)


def test_synthesis_provenance_is_design_intent() -> None:
    geo = G.resolve(Project())
    for c in CORNERS:
        for n in INPUT_POINTS:
            assert geo.sources[c][n] == "design_intent"
    assert geo.corner("FL").hardpoints()[0].source == "design_intent"
    assert geo.estimate_count() == 0


def test_left_right_signs_follow_iso8855() -> None:
    geo = G.resolve(Project())
    for left, right in (("FL", "FR"), ("RL", "RR")):
        for n in INPUT_POINTS:
            yl, yr = geo.points[left][n][1], geo.points[right][n][1]
            assert yl > 0 > yr
            assert yl == pytest.approx(-yr, abs=1e-9)


def test_track_and_wheelbase_at_contact_patches() -> None:
    geo = G.resolve(Project())
    assert geo.track_mm("front") == pytest.approx(sla.FRONT_2027.track_mm, abs=1e-9)
    assert geo.track_mm("rear") == pytest.approx(sla.REAR_2027.track_mm, abs=1e-9)
    assert geo.wheelbase_mm == pytest.approx(sla.VEHICLE_2027.wheelbase_mm, abs=1e-9)


def test_freeze_to_hardpoints_is_lossless() -> None:
    p = Project()
    syn = G.resolve(p)
    p.hardpoints = G.freeze_to_hardpoints(p)
    p.geometry_mode = "hardpoints"
    hp = G.resolve(p)
    assert hp.mode == "hardpoints"
    for c in CORNERS:
        for n in (*INPUT_POINTS, "CONTACT_PATCH"):
            assert hp.points[c][n] == pytest.approx(syn.points[c][n], abs=1e-9), (c, n)
    assert hp.warnings == []


def test_hardpoint_mode_rejects_incomplete_sets() -> None:
    p = Project()
    hs = G.freeze_to_hardpoints(p)
    del hs.points["RR"]["TIE_ROD_IN"]
    p.hardpoints, p.geometry_mode = hs, "hardpoints"
    with pytest.raises(G.GeometryError, match="RR/TIE_ROD_IN"):
        G.resolve(p)


def test_hardpoint_mode_rejects_a_mirrored_car() -> None:
    """A left corner with negative Y must fail loudly, not solve as a mirror image."""
    p = Project()
    hs = G.freeze_to_hardpoints(p)
    hs.points["FL"] = {n: (x, -y, z) for n, (x, y, z) in hs.points["FL"].items()}
    p.hardpoints, p.geometry_mode = hs, "hardpoints"
    with pytest.raises(G.GeometryError, match="FL"):
        G.resolve(p)


def test_wheelbase_mismatch_is_reported() -> None:
    p = Project()
    hs = G.freeze_to_hardpoints(p)
    for c in ("RL", "RR"):
        hs.points[c] = {n: (x - 20.0, y, z) for n, (x, y, z) in hs.points[c].items()}
    p.hardpoints, p.geometry_mode = hs, "hardpoints"
    geo = G.resolve(p)
    assert geo.wheelbase_mm == pytest.approx(1560.0)
    assert any("Wheelbase" in w for w in geo.warnings)
    # front mass fraction preserved when the CG station is rescaled
    assert geo.vehicle_data.front_mass_fraction == pytest.approx(
        p.vehicle.front_mass_fraction)


def test_contact_patch_moves_outboard_with_negative_camber() -> None:
    left = G.contact_patch_point((0.0, 600.0, 245.0), 245.0, -1.5, True)
    right = G.contact_patch_point((0.0, -600.0, 245.0), 245.0, -1.5, False)
    assert left[1] > 600.0 and right[1] < -600.0
    assert left[2] == right[2] == 0.0
    assert left[1] - 600.0 == pytest.approx(245.0 * math.tan(math.radians(1.5)))


def test_with_path_rebuilds_frozen_dataclasses_without_mutation() -> None:
    p = Project()
    q = with_path(p, "front.limits.rc_height_mm.0", 15.0)
    assert get_path(q, "front.limits.rc_height_mm") == (15.0, 70.0)
    assert get_path(p, "front.limits.rc_height_mm") == (20.0, 70.0)
    r = with_path(q, "rear_toe_link_inboard.2", 170.0)
    assert r.rear_toe_link_inboard[2] == 170.0
    assert q.rear_toe_link_inboard[2] == 169.0


def test_project_json_round_trip(tmp_path) -> None:
    p = Project()
    p.hardpoints = G.freeze_to_hardpoints(p)
    p.front = with_path(p, "front.kpi_deg", 9.0).front
    p.setup_sheet["spring_f"] = "35"
    path = tmp_path / "car.vdproj.json"
    save_project(p, path)
    q = load_project(path)
    assert q.front == p.front and q.rear == p.rear and q.steering == p.steering
    assert q.vehicle == p.vehicle
    assert q.hardpoints is not None and q.hardpoints.points == p.hardpoints.points
    assert q.setup_sheet == {"spring_f": "35"}
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1 and "ISO 8855" in data["frame"]


def test_newer_schema_is_refused() -> None:
    d = project_to_dict(Project())
    d["schema_version"] = 99
    with pytest.raises(ValueError, match="newer"):
        project_from_dict(d)


def test_unknown_keys_are_ignored_for_forward_compatibility() -> None:
    d = project_to_dict(Project())
    d["front"]["some_future_field"] = 1.0
    assert project_from_dict(d).front == Project().front


def test_mirror_left_to_right() -> None:
    geo = G.resolve(Project())
    pts = {c: {n: geo.points[c][n] for n in INPUT_POINTS} for c in CORNERS}
    pts["FL"]["UCA_OUT"] = (1.0, 2.0, 3.0)
    m = G.mirror_left_to_right(pts)
    assert m["FR"]["UCA_OUT"] == (1.0, -2.0, 3.0)
