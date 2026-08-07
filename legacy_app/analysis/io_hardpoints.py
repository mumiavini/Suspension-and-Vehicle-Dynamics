"""
analysis/io_hardpoints.py
=========================
Reading, validation and WRITING of suspension hardpoints.

SUPPORTED FORMATS:
    .xlsx, .csv, .json

FILE STRUCTURE (one row per hardpoint):
    corner | point        | x_mm | y_mm | z_mm
    -------+--------------+------+------+------
    FL     | UCA_IN_FRONT |  60  |  150 |  295
    FL     | UCA_IN_REAR  | -70  |  150 |  295
    ...

EXPECTED POINTS PER CORNER (10 points):
    UCA_IN_FRONT, UCA_IN_REAR, UCA_OUT
    LCA_IN_FRONT, LCA_IN_REAR, LCA_OUT
    TIE_ROD_IN,   TIE_ROD_OUT
    WHEEL_CENTER, CONTACT_PATCH

VALID CORNERS:
    FL (front-left), FR (front-right), RL (rear-left), RR (rear-right)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from geometry.primitives import Point3D
from geometry.model_3d import ControlArm, SuspensionCorner, Vehicle
from geometry.solver_3d import TieRod

if TYPE_CHECKING:
    import polars as pl


# =============================================================================
# Schema constants
# =============================================================================

REQUIRED_COLUMNS: list[str] = ["corner", "point", "x_mm", "y_mm", "z_mm"]

REQUIRED_POINTS_PER_CORNER: list[str] = [
    "UCA_IN_FRONT", "UCA_IN_REAR", "UCA_OUT",
    "LCA_IN_FRONT", "LCA_IN_REAR", "LCA_OUT",
    "TIE_ROD_IN",   "TIE_ROD_OUT",
    "WHEEL_CENTER", "CONTACT_PATCH",
]

VALID_CORNERS: list[str] = ["FL", "FR", "RL", "RR"]


class HardpointValidationError(ValueError):
    """Validation error for the hardpoints file."""
    pass


# =============================================================================
# Reading
# =============================================================================

def read_hardpoints(filepath: str | Path) -> "pl.DataFrame":
    """
    Read a hardpoints file (.xlsx, .csv, .json) and return a polars DataFrame.

    Applies validations:
        - required columns present
        - valid corners
        - all 10 points per corner
        - finite numeric coordinates
    """
    import polars as pl

    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pl.read_excel(path)
    elif ext == ".csv":
        df = pl.read_csv(path)
    elif ext == ".json":
        df = pl.read_json(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")

    # Normalize column names and categorical values
    df = df.rename({col: col.lower().strip() for col in df.columns})
    df = df.with_columns([
        pl.col("corner").str.strip_chars().str.to_uppercase(),
        pl.col("point").str.strip_chars().str.to_uppercase(),
    ])

    _validate_dataframe(df)
    return df


def _validate_dataframe(df: "pl.DataFrame") -> None:
    """Full validation of the DataFrame schema."""
    import polars as pl

    # 1. Required columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise HardpointValidationError(
            f"Missing columns: {sorted(missing)}. Expected: {REQUIRED_COLUMNS}"
        )

    # 2. Valid corners
    found_corners = df["corner"].unique().to_list()
    invalid = [c for c in found_corners if c not in VALID_CORNERS]
    if invalid:
        raise HardpointValidationError(
            f"Invalid corners: {invalid}. Valid: {VALID_CORNERS}"
        )

    # 3. Valid point names
    found_points = set(df["point"].unique().to_list())
    invalid_points = found_points - set(REQUIRED_POINTS_PER_CORNER)
    if invalid_points:
        raise HardpointValidationError(
            f"Unknown points: {sorted(invalid_points)}"
        )

    # 4. Each corner has all the points
    for corner in found_corners:
        corner_points = set(
            df.filter(pl.col("corner") == corner)["point"].unique().to_list()
        )
        missing = set(REQUIRED_POINTS_PER_CORNER) - corner_points
        if missing:
            raise HardpointValidationError(
                f"Corner '{corner}' missing points: {sorted(missing)}"
            )

    # 5. Numeric and finite coordinates
    for col in ("x_mm", "y_mm", "z_mm"):
        if not df[col].dtype.is_numeric():
            raise HardpointValidationError(
                f"Column '{col}' must be numeric (dtype: {df[col].dtype})"
            )
        if df[col].is_null().any():
            raise HardpointValidationError(f"Column '{col}' contains nulls")
        if df[col].is_infinite().any():
            raise HardpointValidationError(f"Column '{col}' contains infinities")


# =============================================================================
# Building objects from the DataFrame
# =============================================================================

def build_corner_from_dataframe(
    df: "pl.DataFrame",
    corner_id: str,
) -> tuple[SuspensionCorner, TieRod]:
    """Build a SuspensionCorner + TieRod from the DataFrame for ONE corner."""
    import polars as pl

    if corner_id not in VALID_CORNERS:
        raise ValueError(f"Invalid corner_id: {corner_id}")

    sub = df.filter(pl.col("corner") == corner_id)
    if sub.height == 0:
        raise HardpointValidationError(f"No hardpoint for corner '{corner_id}'")

    def get_point(name: str) -> Point3D:
        row = sub.filter(pl.col("point") == name)
        if row.height != 1:
            raise HardpointValidationError(
                f"Point '{name}' for corner '{corner_id}' found "
                f"{row.height} times (expected 1)"
            )
        r = row.row(0, named=True)
        return Point3D(float(r["x_mm"]), float(r["y_mm"]), float(r["z_mm"]))

    uca = ControlArm(
        inboard_front=get_point("UCA_IN_FRONT"),
        inboard_rear =get_point("UCA_IN_REAR"),
        outboard     =get_point("UCA_OUT"),
        name=f"UCA_{corner_id}",
    )
    lca = ControlArm(
        inboard_front=get_point("LCA_IN_FRONT"),
        inboard_rear =get_point("LCA_IN_REAR"),
        outboard     =get_point("LCA_OUT"),
        name=f"LCA_{corner_id}",
    )
    tie_rod = TieRod(
        inboard =get_point("TIE_ROD_IN"),
        outboard=get_point("TIE_ROD_OUT"),
        name=f"TR_{corner_id}",
    )
    corner = SuspensionCorner(
        upper_arm=uca,
        lower_arm=lca,
        wheel_center =get_point("WHEEL_CENTER"),
        contact_patch=get_point("CONTACT_PATCH"),
        corner_id=corner_id,
    )
    return corner, tie_rod


def build_vehicle_from_dataframe(
    df: "pl.DataFrame",
) -> tuple[Vehicle, dict[str, TieRod]]:
    """Build the complete Vehicle + a dict of tie-rods per corner."""
    fl_corner, fl_tr = build_corner_from_dataframe(df, "FL")
    fr_corner, fr_tr = build_corner_from_dataframe(df, "FR")
    rl_corner, rl_tr = build_corner_from_dataframe(df, "RL")
    rr_corner, rr_tr = build_corner_from_dataframe(df, "RR")

    wheelbase   = abs(fl_corner.wheel_center.x - rl_corner.wheel_center.x)
    track_front = abs(fl_corner.wheel_center.y - fr_corner.wheel_center.y)
    track_rear  = abs(rl_corner.wheel_center.y - rr_corner.wheel_center.y)

    vehicle = Vehicle(
        front_left =fl_corner,
        front_right=fr_corner,
        rear_left  =rl_corner,
        rear_right =rr_corner,
        wheelbase_mm=wheelbase,
        track_front_mm=track_front,
        track_rear_mm=track_rear,
    )
    return vehicle, {"FL": fl_tr, "FR": fr_tr, "RL": rl_tr, "RR": rr_tr}


# =============================================================================
# Writing / Export
# =============================================================================

def dataframe_from_corner(
    corner: SuspensionCorner,
    tie_rod: TieRod,
) -> "pl.DataFrame":
    """
    Convert ONE corner + tie_rod into a DataFrame in the standard format.
    Useful for exporting optimized geometries.
    """
    import polars as pl

    rows = [
        ("UCA_IN_FRONT",  corner.upper_arm.inboard_front),
        ("UCA_IN_REAR",   corner.upper_arm.inboard_rear),
        ("UCA_OUT",       corner.upper_arm.outboard),
        ("LCA_IN_FRONT",  corner.lower_arm.inboard_front),
        ("LCA_IN_REAR",   corner.lower_arm.inboard_rear),
        ("LCA_OUT",       corner.lower_arm.outboard),
        ("TIE_ROD_IN",    tie_rod.inboard),
        ("TIE_ROD_OUT",   tie_rod.outboard),
        ("WHEEL_CENTER",  corner.wheel_center),
        ("CONTACT_PATCH", corner.contact_patch),
    ]
    return pl.DataFrame([
        {
            "corner": corner.corner_id,
            "point":  point_name,
            "x_mm":   round(p.x, 4),
            "y_mm":   round(p.y, 4),
            "z_mm":   round(p.z, 4),
        }
        for point_name, p in rows
    ])


def save_dataframe(df: "pl.DataFrame", filepath: str | Path) -> None:
    """Save a DataFrame to .xlsx, .csv or .json."""
    path = Path(filepath)
    ext = path.suffix.lower()
    if ext == ".xlsx":
        df.write_excel(path)
    elif ext == ".csv":
        df.write_csv(path)
    elif ext == ".json":
        df.write_json(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")


# =============================================================================
# Demo template (realistic FSAE geometry)
# =============================================================================

def generate_template_dataframe() -> "pl.DataFrame":
    """
    Generate a template DataFrame with realistic FSAE geometry, mirrored for
    the 4 corners. Use as a starting point.
    """
    import polars as pl

    # Base geometry of the FL corner (left side, front)
    fl_data: dict[str, tuple[float, float, float]] = {
        "UCA_IN_FRONT":  ( 60.0, 150.0, 295.0),
        "UCA_IN_REAR":   (-70.0, 150.0, 295.0),
        "UCA_OUT":       ( -5.0, 590.0, 280.0),
        "LCA_IN_FRONT":  ( 90.0, 130.0, 162.0),
        "LCA_IN_REAR":   (-70.0, 130.0, 162.0),
        "LCA_OUT":       ( 15.0, 600.0, 152.0),
        "TIE_ROD_IN":    (-50.0, 180.0, 200.0),
        "TIE_ROD_OUT":   (-60.0, 580.0, 195.0),
        "WHEEL_CENTER":  (  5.0, 610.0, 220.0),
        "CONTACT_PATCH": (  5.0, 610.0,   0.0),
    }
    rear_offset_x = -1550.0   # wheelbase ≈ 1550 mm

    rows: list[dict[str, object]] = []
    for corner_id in VALID_CORNERS:
        y_sign  = 1.0 if corner_id.endswith("L") else -1.0
        x_shift = rear_offset_x if corner_id.startswith("R") else 0.0
        for pt_name, (x, y, z) in fl_data.items():
            rows.append({
                "corner": corner_id,
                "point":  pt_name,
                "x_mm":   float(x + x_shift),
                "y_mm":   float(y * y_sign),
                "z_mm":   float(z),
            })

    return pl.DataFrame(rows)


def save_template(filepath: str | Path) -> None:
    """Save a template file to .xlsx, .csv or .json."""
    save_dataframe(generate_template_dataframe(), filepath)
