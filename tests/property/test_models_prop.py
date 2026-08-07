"""Hypothesis property tests for vdcore.models."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from vdcore.models.hardpoint import (
    Axle,
    Corner,
    Hardpoint,
    TirePackage,
    Vehicle,
)


@st.composite
def hardpoints(draw: st.DrawFn, name: str = "HP", y_sign: float = 1.0) -> Hardpoint:
    x = draw(st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False))
    y = abs(draw(st.floats(min_value=50, max_value=700, allow_nan=False, allow_infinity=False)))
    z = draw(st.floats(min_value=0, max_value=400, allow_nan=False, allow_infinity=False))
    source = draw(st.sampled_from(["cad", "measured", "estimate", "design_intent"]))
    tol = draw(st.floats(min_value=0.01, max_value=50, allow_nan=False, allow_infinity=False))
    return Hardpoint(name=name, x_mm=x, y_mm=y_sign * y, z_mm=z, source=source, tol_mm=tol)


@st.composite
def tire_packages(draw: st.DrawFn) -> TirePackage:
    r = draw(st.floats(min_value=180, max_value=280, allow_nan=False, allow_infinity=False))
    source = draw(st.sampled_from(["cad", "measured", "estimate", "design_intent"]))
    tol = draw(st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False))
    return TirePackage(loaded_radius_mm=r, source=source, tol_mm=tol)


@st.composite
def corners(draw: st.DrawFn, corner_id: str = "FL") -> Corner:
    y_sign = 1.0 if corner_id in ("FL", "RL") else -1.0
    camber = draw(st.floats(min_value=-5.0, max_value=0.0, allow_nan=False, allow_infinity=False))
    toe = draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    return Corner(
        corner_id=corner_id,
        uca_inboard_front=draw(hardpoints(name="UCA_IF", y_sign=y_sign)),
        uca_inboard_rear=draw(hardpoints(name="UCA_IR", y_sign=y_sign)),
        uca_outboard=draw(hardpoints(name="UCA_O", y_sign=y_sign)),
        lca_inboard_front=draw(hardpoints(name="LCA_IF", y_sign=y_sign)),
        lca_inboard_rear=draw(hardpoints(name="LCA_IR", y_sign=y_sign)),
        lca_outboard=draw(hardpoints(name="LCA_O", y_sign=y_sign)),
        tie_rod_inboard=draw(hardpoints(name="TR_I", y_sign=y_sign)),
        tie_rod_outboard=draw(hardpoints(name="TR_O", y_sign=y_sign)),
        wheel_center=draw(hardpoints(name="WC", y_sign=y_sign)),
        tire=draw(tire_packages()),
        static_camber_deg=camber,
        static_toe_deg_per_side=toe,
    )


@given(corner=corners("FL"))
@settings(max_examples=50)
def test_corner_round_trip(corner: Corner) -> None:
    """A Corner must survive a dump/validate round-trip."""
    data = corner.model_dump()
    restored = Corner.model_validate(data)
    assert corner == restored


@given(corner=corners("FL"))
@settings(max_examples=50)
def test_serialization_has_no_derived_fields(corner: Corner) -> None:
    """model_dump() must not contain derived geometry."""
    data = corner.model_dump()
    assert "contact_patch" not in data
    assert "track_mm" not in data


@given(corner=corners("FL"))
@settings(max_examples=20)
def test_hardpoints_count(corner: Corner) -> None:
    """hardpoints() always returns exactly 9 points."""
    assert len(corner.hardpoints()) == 9


@given(data=st.data())
@settings(max_examples=20)
def test_vehicle_round_trip(data: st.DataObject) -> None:
    """A full Vehicle must survive dump/validate round-trip."""
    fl = data.draw(corners("FL"))
    fr = data.draw(corners("FR"))
    rl = data.draw(corners("RL"))
    rr = data.draw(corners("RR"))
    v = Vehicle(front=Axle(left=fl, right=fr), rear=Axle(left=rl, right=rr))
    restored = Vehicle.model_validate(v.model_dump())
    assert v == restored
