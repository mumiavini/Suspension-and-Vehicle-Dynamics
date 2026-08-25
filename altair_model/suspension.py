"""Create suspension geometry in the active MotionView model (Altair 2026.1).

Parse hardpoints from carro_formula_2027.csv and create:
- All suspension hardpoints (UCA/LCA inner/outer, tie rod, wheel center)
- Linkages for each corner (UCA arm, LCA arm, tie rod, wheel spindle)

Run it from MotionView's Python console (the Jupyter QtConsole with the `hwx`
profile), with a model open:

    exec(open(r"c:\\soft\\FSAE\\Suspension and Vehicle Dynamics\\altair_model\\suspension.py").read())

Units: MMKS (mm, kg, s, N), matching the rest of the project.
Frame: ISO 8855 — X+ forward, Y+ LEFT, Z+ up, origin at the front axle
centreline on the ground plane.
"""

import csv
from pathlib import Path
from hw.mview.mbd import Model, Point, Link


def load_hardpoints_from_csv(csv_path):
    """Load hardpoints from CSV file. Return dict: (corner, point_name) -> (x, y, z)."""
    hardpoints = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            corner = row["corner"]
            point = row["point"]
            x = float(row["x_mm"])
            y = float(row["y_mm"])
            z = float(row["z_mm"])
            hardpoints[(corner, point)] = (x, y, z)
    return hardpoints


def create_suspension_geometry():
    """Create all suspension hardpoints and linkages from CSV data."""
    model = Model.getCurrentModel(create=True)

    # Load hardpoints from CSV (assumes CSV is in the same directory as this script)
    csv_path = Path(__file__).parent / ".." / "legacy_app" / "carro_formula_2027.csv"
    hardpoints = load_hardpoints_from_csv(csv_path)

    # Map to store created Point objects
    points = {}
    corners = ["FL", "FR", "RL", "RR"]
    point_names = [
        "UCA_IN_FRONT",
        "UCA_IN_REAR",
        "UCA_OUT",
        "LCA_IN_FRONT",
        "LCA_IN_REAR",
        "LCA_OUT",
        "TIE_ROD_IN",
        "TIE_ROD_OUT",
        "WHEEL_CENTER",
        "CONTACT_PATCH",
    ]

    # Create all hardpoints
    for corner in corners:
        for pnt_name in point_names:
            if (corner, pnt_name) in hardpoints:
                x, y, z = hardpoints[(corner, pnt_name)]
                mview_name = f"p_{corner.lower()}_{pnt_name.lower()}"
                point = Point(
                    name=mview_name,
                    label=f"{corner} {pnt_name}",
                    x=x,
                    y=y,
                    z=z,
                )
                points[(corner, pnt_name)] = point

    # Create linkages for each corner
    for corner in corners:
        # Upper Control Arm (connects two inner joints to outer joint)
        if (corner, "UCA_IN_FRONT") in points and (corner, "UCA_OUT") in points:
            uca_link = Link(
                name=f"l_{corner.lower()}_uca",
                label=f"{corner} UCA",
            )
            uca_link.add_point(points[(corner, "UCA_IN_FRONT")])
            uca_link.add_point(points[(corner, "UCA_IN_REAR")])
            uca_link.add_point(points[(corner, "UCA_OUT")])

        # Lower Control Arm
        if (corner, "LCA_IN_FRONT") in points and (corner, "LCA_OUT") in points:
            lca_link = Link(
                name=f"l_{corner.lower()}_lca",
                label=f"{corner} LCA",
            )
            lca_link.add_point(points[(corner, "LCA_IN_FRONT")])
            lca_link.add_point(points[(corner, "LCA_IN_REAR")])
            lca_link.add_point(points[(corner, "LCA_OUT")])

        # Tie Rod
        if (corner, "TIE_ROD_IN") in points and (corner, "TIE_ROD_OUT") in points:
            tie_rod_link = Link(
                name=f"l_{corner.lower()}_tierod",
                label=f"{corner} Tie Rod",
            )
            tie_rod_link.add_point(points[(corner, "TIE_ROD_IN")])
            tie_rod_link.add_point(points[(corner, "TIE_ROD_OUT")])

    print(f"Created {len(points)} hardpoints and linkages for 4 suspension corners.")
    print(f"Model now has {len(model.points())} point(s).")


if __name__ == "__main__":
    create_suspension_geometry()
