"""Per-joint rod-end misalignment (kpis.rod_end_misalignment) with bolt axes as inputs."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from apps.vdstudio.engine import geometry as G
from apps.vdstudio.engine import kpis as K
from apps.vdstudio.engine.project import Project, project_from_dict, project_to_dict
from vdcore.geometry.solver import DWSolver


@pytest.fixture(scope="module")
def geo():
    return G.resolve(Project())


def test_default_car(geo) -> None:
    f = K.rod_end_misalignment(geo, "front")
    r = K.rod_end_misalignment(geo, "rear")
    # vertical bolts: the static rod is already tilted 3.59 deg (front) / 2.20 deg (rear)
    assert f["inner"].static_deg == pytest.approx(3.5928, abs=1e-3)
    assert f["inner"].max_deg == pytest.approx(9.785, abs=1e-3)
    assert (f["inner"].at_bump_mm, f["inner"].at_rack_mm) == (25.0, 38.0)
    assert f["outer"].max_deg == pytest.approx(7.675, abs=1e-3)
    assert r["inner"].max_deg == pytest.approx(5.782, abs=1e-3)
    assert r["outer"].max_deg == pytest.approx(4.811, abs=1e-3)
    # the screening value comes out of the same solves
    assert f["swing"].max_deg == pytest.approx(K.worst_rod_end_swing(geo), abs=1e-12)


def _triangle_frame(r, lift):
    """Upright orientation from its own UBJ-LBJ-TRO triangle (no Kabsch fit)."""
    u, lo, t = (np.array([p.x_mm, p.y_mm, p.z_mm]) + lift for p in (r.ubj, r.lbj, r.tro))
    e1 = (u - lo) / np.linalg.norm(u - lo)
    e3 = np.cross(e1, t - lo)
    e3 /= np.linalg.norm(e3)
    return np.column_stack([e1, np.cross(e3, e1), e3]), t


def test_matches_an_independent_upright_frame(geo) -> None:
    """3×3 envelope, outer bolt carried by the triangle frame instead of a rigid fit."""
    b = np.array([0.3, -0.2, 0.93])
    b /= np.linalg.norm(b)
    q = replace(Project(), rod_end_bolts=replace(Project().rod_end_bolts,
                                                 front_outer=tuple(b), front_inner=tuple(b)))
    g = G.resolve(q)
    got = K.rod_end_misalignment(g, "front", n=3)
    s = DWSolver(g.corner("FL"))
    f0, _ = _triangle_frame(s.solve(), np.zeros(3))
    b_body = f0.T @ b
    tri0 = g.p("FL", "TIE_ROD_IN")
    best = {"inner": 0.0, "outer": 0.0}
    for bump in (-25.0, 0.0, 25.0):
        for rack in (-38.0, 0.0, 38.0):
            r = s.solve(wheel_travel_mm=bump, rack_mm=rack)
            fr, t = _triangle_frame(r, np.array([0.0, 0.0, bump]))
            d = t - (tri0 + [0.0, rack, 0.0])
            d /= np.linalg.norm(d)
            best["inner"] = max(best["inner"], math.degrees(math.asin(abs(d @ b))))
            best["outer"] = max(best["outer"], math.degrees(math.asin(abs(d @ (fr @ b_body)))))
    assert got["inner"].max_deg == pytest.approx(best["inner"], abs=1e-6)
    assert got["outer"].max_deg == pytest.approx(best["outer"], abs=1e-6)


def test_a_bolt_normal_to_the_rod_is_centred_at_ride_height(geo) -> None:
    d = geo.p("FL", "TIE_ROD_OUT") - geo.p("FL", "TIE_ROD_IN")
    b = np.cross(d, [1.0, 0.0, 0.0])                     # normal to the static rod
    q = replace(Project(), rod_end_bolts=replace(Project().rod_end_bolts, front_inner=tuple(b)))
    f = K.rod_end_misalignment(G.resolve(q), "front")
    assert f["inner"].static_deg == pytest.approx(0.0, abs=1e-9)
    assert f["inner"].max_deg > 0.0


def test_kpis_and_explicit_failures() -> None:
    ks = K.compute_dynamic(G.resolve(Project()), ["rod_end"])
    for key in ("steering.rod_end_inner", "steering.rod_end_outer",
                "rear.toe_link_rod_end_inner", "rear.toe_link_rod_end_outer"):
        k = ks.get(key)
        assert k.ok and k.unit == "deg" and k.band == (0.0, 12.0) and "'estimate'" in k.note
    # a zero bolt axis fails the rows, it is never treated as some default direction
    q = replace(Project(), rod_end_bolts=replace(Project().rod_end_bolts,
                                                 rear_outer=(0.0, 0.0, 0.0)))
    kz = K.compute_dynamic(G.resolve(q), ["rod_end"])
    assert not kz.get("rear.toe_link_rod_end_outer").ok
    assert "zero length" in kz.get("rear.toe_link_rod_end_outer").error
    assert kz.get("steering.rod_end_inner").ok
    # an envelope point that does not solve fails the front rows
    p = Project()
    q = replace(p, steering=replace(p.steering, max_rack_travel_mm=60.0))
    kf = K.compute_dynamic(G.resolve(q), ["rod_end"])
    assert not kf.get("steering.rod_end_inner").ok and "did not converge" in \
        kf.get("steering.rod_end_inner").error


def test_bolts_round_trip_and_old_files_get_the_defaults() -> None:
    p = replace(Project(), rod_end_bolts=replace(Project().rod_end_bolts,
                                                 front_outer=(0.1, 0.0, 1.0)))
    back = project_from_dict(project_to_dict(p))
    assert back.rod_end_bolts.front_outer == (0.1, 0.0, 1.0)
    d = project_to_dict(Project())
    del d["rod_end_bolts"]
    assert project_from_dict(d).rod_end_bolts == Project().rod_end_bolts
