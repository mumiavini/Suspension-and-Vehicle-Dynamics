"""
analysis/viz3d.py
=================
3D visualization of the hardpoints and the suspension using Plotly.

Main functions:
    plot_corner_3d       : 1 corner in isolation (UCA, LCA, TR, upright, wheel)
    plot_vehicle_3d      : all 4 corners + chassis axes/silhouette
    plot_corner_animated : animated version with frames for a sweep
                           (heave/roll/steer shown with a slider)

CONVENTIONS:
    - Points colored by group (UCA blue, LCA red, TR green, Wheel orange)
    - Thin lines connect points of the same arm
    - Tire rim drawn as a circle in the wheel plane
    - World axes (X+ front, Y+ left, Z+ up) always visible
"""

from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from geometry.primitives import Point3D
from geometry.model_3d import SuspensionCorner, Vehicle
from geometry.solver_3d import TieRod, KinematicSolver3D, KinematicState3D

if TYPE_CHECKING:
    import plotly.graph_objects as go


# =============================================================================
# Style constants
# =============================================================================

# Color per point group — consistent across all visualizations
GROUP_COLORS: dict[str, str] = {
    "UCA":      "#1f77b4",   # blue
    "LCA":      "#d62728",   # red
    "Tie-rod":  "#2ca02c",   # green
    "Wheel":    "#ff7f0e",   # orange
    "Upright":  "#9467bd",   # purple (upright lines)
    "Chassis":  "#7f7f7f",   # gray (chassis silhouette)
}

# Mapping of hardpoint → group
POINT_TO_GROUP: dict[str, str] = {
    "UCA_IN_FRONT":  "UCA",
    "UCA_IN_REAR":   "UCA",
    "UCA_OUT":       "UCA",
    "LCA_IN_FRONT":  "LCA",
    "LCA_IN_REAR":   "LCA",
    "LCA_OUT":       "LCA",
    "TIE_ROD_IN":    "Tie-rod",
    "TIE_ROD_OUT":   "Tie-rod",
    "WHEEL_CENTER":  "Wheel",
    "CONTACT_PATCH": "Wheel",
}


# =============================================================================
# Collecting a corner's points in an arbitrary state
# =============================================================================

def _collect_corner_points(
    corner:  SuspensionCorner,
    tie_rod: TieRod,
    state:   Optional[KinematicState3D] = None,
) -> dict[str, Point3D]:
    """
    Gather all 10 hardpoints of a corner into a dict {name → Point3D}.

    If `state` is provided, use the dynamic positions (outboards + WC + CP of
    the rotated upright). Otherwise, use the original static positions.

    The chassis inboards ALWAYS come from the original corner — they move with
    the CHASSIS heave/roll, not with the upright, and that transformation is
    applied only in the animated visualization.
    """
    pts: dict[str, Point3D] = {
        "UCA_IN_FRONT": corner.upper_arm.inboard_front,
        "UCA_IN_REAR":  corner.upper_arm.inboard_rear,
        "LCA_IN_FRONT": corner.lower_arm.inboard_front,
        "LCA_IN_REAR":  corner.lower_arm.inboard_rear,
        "TIE_ROD_IN":   tie_rod.inboard,
    }
    if state is None:
        pts["UCA_OUT"]       = corner.upper_arm.outboard
        pts["LCA_OUT"]       = corner.lower_arm.outboard
        pts["TIE_ROD_OUT"]   = tie_rod.outboard
        pts["WHEEL_CENTER"]  = corner.wheel_center
        pts["CONTACT_PATCH"] = corner.contact_patch
    else:
        pts["UCA_OUT"]       = state.uca_outboard
        pts["LCA_OUT"]       = state.lca_outboard
        pts["TIE_ROD_OUT"]   = state.tie_rod_outboard
        pts["WHEEL_CENTER"]  = state.wheel_center
        pts["CONTACT_PATCH"] = state.contact_patch
    return pts


def _move_chassis_points_for_state(
    points:   dict[str, Point3D],
    heave_mm: float,
    roll_deg: float,
) -> dict[str, Point3D]:
    """
    Apply heave+roll to the chassis points (the inboards only).
    Replicates the logic of `KinematicSolver3D._move_chassis_point` to keep
    visual consistency with the solver.
    """
    chassis_keys = {"UCA_IN_FRONT", "UCA_IN_REAR",
                    "LCA_IN_FRONT", "LCA_IN_REAR",
                    "TIE_ROD_IN"}
    if abs(heave_mm) < 1e-12 and abs(roll_deg) < 1e-12:
        return dict(points)

    cos_t = math.cos(math.radians(roll_deg))
    sin_t = math.sin(math.radians(roll_deg))

    result: dict[str, Point3D] = {}
    for name, p in points.items():
        if name in chassis_keys:
            # Roll rotation about X, then Z translation (heave)
            y_new = p.y * cos_t - p.z * sin_t
            z_new = p.y * sin_t + p.z * cos_t + heave_mm
            result[name] = Point3D(p.x, y_new, z_new)
        else:
            result[name] = p
    return result


# =============================================================================
# Tire rim — circle in the wheel plane
# =============================================================================

def _generate_wheel_outline(
    wheel_center:  Point3D,
    contact_patch: Point3D,
    n_segments: int = 32,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Generate a circle representing the outer outline of the tire, in the plane
    perpendicular to the hub axis.

    The hub axis is assumed to be PERPENDICULAR to the WC→CP vector
    (and horizontal, parallel to Y). For a cambered wheel, that axis is
    slightly tilted.

    Returns arrays (x, y, z) with `n_segments + 1` points forming a closed
    circle (last point = first).
    """
    wc = wheel_center.to_array()
    cp = contact_patch.to_array()

    # Tire radius = distance WC → CP
    radius = float(np.linalg.norm(wc - cp))
    if radius < 1e-6:
        return (np.array([wc[0]]), np.array([wc[1]]), np.array([wc[2]]))

    # Radial vector "downward" in the wheel plane (WC→CP)
    radial_down = (cp - wc) / radius   # magnitude = 1

    # Hub axis: perpendicular to the radial and roughly parallel to Y.
    # Simple approximation: Y axis projected and orthonormalized against radial.
    y_axis = np.array([0.0, 1.0, 0.0])
    hub_axis = y_axis - np.dot(y_axis, radial_down) * radial_down
    hub_norm = float(np.linalg.norm(hub_axis))
    if hub_norm < 1e-6:
        # Degenerate case: use the X axis
        hub_axis = np.array([1.0, 0.0, 0.0])
        hub_axis = hub_axis - np.dot(hub_axis, radial_down) * radial_down
        hub_norm = float(np.linalg.norm(hub_axis))
    hub_axis = hub_axis / hub_norm

    # Tangential vector in the wheel plane: radial × hub_axis
    tangent = np.cross(radial_down, hub_axis)
    tangent = tangent / float(np.linalg.norm(tangent))

    # Generate the circle
    angles = np.linspace(0, 2 * math.pi, n_segments + 1)
    xs = wc[0] + radius * (np.cos(angles) * radial_down[0] + np.sin(angles) * tangent[0])
    ys = wc[1] + radius * (np.cos(angles) * radial_down[1] + np.sin(angles) * tangent[1])
    zs = wc[2] + radius * (np.cos(angles) * radial_down[2] + np.sin(angles) * tangent[2])
    return (xs, ys, zs)


# =============================================================================
# Plot of ONE corner
# =============================================================================

def plot_corner_3d(
    corner:    SuspensionCorner,
    tie_rod:   TieRod,
    *,
    state:     Optional[KinematicState3D] = None,
    show_tire: bool = True,
    title:     Optional[str] = None,
) -> "go.Figure":
    """
    Generate a 3D Plotly figure of ONE corner.

    Shows:
        - 10 hardpoints as spheres colored by group
        - UCA arms (2 lines: front→out, rear→out)
        - LCA arms (likewise)
        - Tie-rod (1 line)
        - Upright (triangle UBJ-LBJ-TRO)
        - Wheel (vertical line WC→CP)
        - Tire (circular rim in the wheel plane)
        - Reference axes in the corner

    If `state` is provided (from the solver), shows the displaced geometry.
    """
    import plotly.graph_objects as go

    pts = _collect_corner_points(corner, tie_rod, state)

    fig = go.Figure()

    # ─── Hardpoints (points) grouped by category for the legend ──────────────
    for group_name in ["UCA", "LCA", "Tie-rod", "Wheel"]:
        group_pts = [(name, p) for name, p in pts.items()
                      if POINT_TO_GROUP[name] == group_name]
        if not group_pts:
            continue
        xs = [p.x for _, p in group_pts]
        ys = [p.y for _, p in group_pts]
        zs = [p.z for _, p in group_pts]
        names = [name for name, _ in group_pts]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers+text",
            marker=dict(size=6, color=GROUP_COLORS[group_name]),
            text=names,
            textfont=dict(size=9, color=GROUP_COLORS[group_name]),
            textposition="top center",
            name=group_name,
            hovertemplate="<b>%{text}</b><br>X=%{x:.1f}<br>Y=%{y:.1f}<br>Z=%{z:.1f}<extra></extra>",
        ))

    # ─── Arm lines ───────────────────────────────────────────────────────────
    def add_line(p1: Point3D, p2: Point3D, color: str, name: str,
                  width: float = 4, dash: str = "solid",
                  showlegend: bool = False) -> None:
        fig.add_trace(go.Scatter3d(
            x=[p1.x, p2.x], y=[p1.y, p2.y], z=[p1.z, p2.z],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            name=name, showlegend=showlegend, hoverinfo="skip",
        ))

    # UCA: 2 lines (front-out and rear-out)
    add_line(pts["UCA_IN_FRONT"], pts["UCA_OUT"], GROUP_COLORS["UCA"], "UCA")
    add_line(pts["UCA_IN_REAR"],  pts["UCA_OUT"], GROUP_COLORS["UCA"], "UCA")
    # LCA: likewise
    add_line(pts["LCA_IN_FRONT"], pts["LCA_OUT"], GROUP_COLORS["LCA"], "LCA")
    add_line(pts["LCA_IN_REAR"],  pts["LCA_OUT"], GROUP_COLORS["LCA"], "LCA")
    # Tie-rod
    add_line(pts["TIE_ROD_IN"], pts["TIE_ROD_OUT"], GROUP_COLORS["Tie-rod"], "Tie-rod")
    # Upright: triangle UBJ-LBJ-TRO
    add_line(pts["UCA_OUT"], pts["LCA_OUT"], GROUP_COLORS["Upright"], "Upright", width=3)
    add_line(pts["UCA_OUT"], pts["TIE_ROD_OUT"], GROUP_COLORS["Upright"], "Upright", width=2, dash="dot")
    add_line(pts["LCA_OUT"], pts["TIE_ROD_OUT"], GROUP_COLORS["Upright"], "Upright", width=2, dash="dot")
    # WC ↔ CP (vertical wheel axis)
    add_line(pts["WHEEL_CENTER"], pts["CONTACT_PATCH"],
              GROUP_COLORS["Wheel"], "WC-CP", width=3, dash="dash")

    # ─── Tire rim ────────────────────────────────────────────────────────────
    if show_tire:
        xs, ys, zs = _generate_wheel_outline(pts["WHEEL_CENTER"], pts["CONTACT_PATCH"])
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=GROUP_COLORS["Wheel"], width=2),
            name="Tire", showlegend=False, hoverinfo="skip",
        ))

    # ─── Ground (transparent Z=0 plane around the contact patch) ─────────────
    cp = pts["CONTACT_PATCH"]
    floor_size = 200.0
    fig.add_trace(go.Mesh3d(
        x=[cp.x - floor_size, cp.x + floor_size, cp.x + floor_size, cp.x - floor_size],
        y=[cp.y - floor_size, cp.y - floor_size, cp.y + floor_size, cp.y + floor_size],
        z=[0, 0, 0, 0],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="lightgray", opacity=0.2,
        name="Ground", showlegend=False, hoverinfo="skip",
    ))

    # ─── Layout: axes with physical aspect ratio (1:1:1) ─────────────────────
    fig.update_layout(
        title=title or f"Corner {corner.corner_id}",
        scene=dict(
            xaxis_title="X (mm) — front",
            yaxis_title="Y (mm) — left",
            zaxis_title="Z (mm) — up",
            aspectmode="data",   # keep real physical scale across the 3 axes
            camera=dict(eye=dict(x=1.3, y=-1.5, z=1.0)),
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.05),
    )

    return fig


# =============================================================================
# Plot of the COMPLETE VEHICLE
# =============================================================================

def plot_vehicle_3d(
    vehicle:  Vehicle,
    tie_rods: dict[str, TieRod],
    *,
    show_tires:  bool = True,
    show_chassis_box: bool = True,
    title: Optional[str] = None,
) -> "go.Figure":
    """
    Generate a 3D Plotly figure of the complete vehicle (4 corners).

    Args:
        vehicle    : Vehicle object with the 4 corners
        tie_rods   : dict {"FL", "FR", "RL", "RR"} → TieRod
        show_chassis_box : if True, draw a simplified chassis wireframe
                            connecting the inboards of the 4 corners
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    corners_map = {
        "FL": vehicle.front_left,
        "FR": vehicle.front_right,
        "RL": vehicle.rear_left,
        "RR": vehicle.rear_right,
    }

    # ─── Collect points by group for the legend (one entry per group) ────────
    all_pts: dict[str, list[tuple[str, Point3D]]] = {
        "UCA": [], "LCA": [], "Tie-rod": [], "Wheel": [],
    }
    for cid, corner in corners_map.items():
        tr = tie_rods[cid]
        for name, p in _collect_corner_points(corner, tr).items():
            grp = POINT_TO_GROUP[name]
            all_pts[grp].append((f"{cid}_{name}", p))

    # Points (single legend entry per group, but hover shows each point)
    for grp, ptlist in all_pts.items():
        xs = [p.x for _, p in ptlist]
        ys = [p.y for _, p in ptlist]
        zs = [p.z for _, p in ptlist]
        names = [name for name, _ in ptlist]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(size=4, color=GROUP_COLORS[grp]),
            text=names,
            name=grp,
            hovertemplate="<b>%{text}</b><br>X=%{x:.1f}<br>Y=%{y:.1f}<br>Z=%{z:.1f}<extra></extra>",
        ))

    # ─── Lines (arms, upright, tie-rods) for each corner ─────────────────────
    def add_segment(p1: Point3D, p2: Point3D, color: str, width: int = 3,
                     dash: str = "solid") -> None:
        fig.add_trace(go.Scatter3d(
            x=[p1.x, p2.x], y=[p1.y, p2.y], z=[p1.z, p2.z],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            showlegend=False, hoverinfo="skip",
        ))

    for cid, corner in corners_map.items():
        tr = tie_rods[cid]
        pts = _collect_corner_points(corner, tr)
        # UCA
        add_segment(pts["UCA_IN_FRONT"], pts["UCA_OUT"], GROUP_COLORS["UCA"])
        add_segment(pts["UCA_IN_REAR"],  pts["UCA_OUT"], GROUP_COLORS["UCA"])
        # LCA
        add_segment(pts["LCA_IN_FRONT"], pts["LCA_OUT"], GROUP_COLORS["LCA"])
        add_segment(pts["LCA_IN_REAR"],  pts["LCA_OUT"], GROUP_COLORS["LCA"])
        # Tie-rod
        add_segment(pts["TIE_ROD_IN"], pts["TIE_ROD_OUT"], GROUP_COLORS["Tie-rod"])
        # Upright
        add_segment(pts["UCA_OUT"], pts["LCA_OUT"], GROUP_COLORS["Upright"], width=3)
        add_segment(pts["UCA_OUT"], pts["TIE_ROD_OUT"], GROUP_COLORS["Upright"], width=2, dash="dot")
        add_segment(pts["LCA_OUT"], pts["TIE_ROD_OUT"], GROUP_COLORS["Upright"], width=2, dash="dot")
        # WC-CP
        add_segment(pts["WHEEL_CENTER"], pts["CONTACT_PATCH"],
                     GROUP_COLORS["Wheel"], width=2, dash="dash")

        # Tire
        if show_tires:
            xs, ys, zs = _generate_wheel_outline(pts["WHEEL_CENTER"], pts["CONTACT_PATCH"])
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(color=GROUP_COLORS["Wheel"], width=2),
                showlegend=False, hoverinfo="skip",
            ))

    # ─── Chassis wireframe (connecting the inboards of the 4 corners) ────────
    if show_chassis_box:
        chassis_pts = {}
        for cid, corner in corners_map.items():
            # Use the centroid of the 2 UCA inboards as the "upper chassis point"
            uca_in = corner.upper_arm.effective_inboard
            lca_in = corner.lower_arm.effective_inboard
            chassis_pts[f"{cid}_top"]    = uca_in
            chassis_pts[f"{cid}_bottom"] = lca_in

        # Connections: edges of the chassis "box"
        box_edges = [
            # Top (UCA inboards)
            ("FL_top", "FR_top"), ("RL_top", "RR_top"),
            ("FL_top", "RL_top"), ("FR_top", "RR_top"),
            # Bottom (LCA inboards)
            ("FL_bottom", "FR_bottom"), ("RL_bottom", "RR_bottom"),
            ("FL_bottom", "RL_bottom"), ("FR_bottom", "RR_bottom"),
            # Verticals
            ("FL_top", "FL_bottom"), ("FR_top", "FR_bottom"),
            ("RL_top", "RL_bottom"), ("RR_top", "RR_bottom"),
        ]
        for a, b in box_edges:
            add_segment(chassis_pts[a], chassis_pts[b],
                         GROUP_COLORS["Chassis"], width=1, dash="dot")

    # ─── Ground ──────────────────────────────────────────────────────────────
    xs_floor = [vehicle.front_left.contact_patch.x + 200,
                vehicle.rear_left.contact_patch.x  - 200]
    ys_floor = [vehicle.front_left.contact_patch.y + 200,
                vehicle.front_right.contact_patch.y - 200]
    fig.add_trace(go.Mesh3d(
        x=[xs_floor[0], xs_floor[0], xs_floor[1], xs_floor[1]],
        y=[ys_floor[0], ys_floor[1], ys_floor[1], ys_floor[0]],
        z=[0, 0, 0, 0],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="lightgray", opacity=0.15,
        name="Ground", showlegend=False, hoverinfo="skip",
    ))

    # ─── Layout ──────────────────────────────────────────────────────────────
    fig.update_layout(
        title=title or "Complete vehicle",
        scene=dict(
            xaxis_title="X (mm) — front",
            yaxis_title="Y (mm) — left",
            zaxis_title="Z (mm) — up",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)),
        ),
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.05),
    )

    return fig


# =============================================================================
# ANIMATED plot of a corner (sweep)
# =============================================================================

def plot_corner_animated(
    corner:    SuspensionCorner,
    tie_rod:   TieRod,
    *,
    sweep_axis: str = "heave",      # "heave", "roll" or "steer"
    sweep_min:  float = -25.0,
    sweep_max:  float =  25.0,
    n_frames:   int   = 15,
    show_tire:  bool  = True,
) -> "go.Figure":
    """
    Animated version of the 3D corner visualization.

    Adds a slider that runs through `n_frames` positions along the chosen axis
    (heave in mm, roll in °, or rack in mm) and shows the geometry moving.

    Useful for visually seeing how camber, scrub, etc. change during the
    suspension motion.
    """
    import plotly.graph_objects as go

    if sweep_axis not in ("heave", "roll", "steer"):
        raise ValueError(f"Invalid sweep_axis: {sweep_axis}")

    # Generate frames by solving the solver at each position
    solver = KinematicSolver3D(corner, tie_rod)
    values = np.linspace(sweep_min, sweep_max, n_frames)

    # Solve the state at each point (keep continuity)
    solver.reset_seed()
    frame_states: list[tuple[float, KinematicState3D, float, float]] = []
    # tuple: (slider_value, state, heave, roll)
    for v in values:
        if sweep_axis == "heave":
            state = solver.solve(float(v), 0.0, 0.0)
            frame_states.append((float(v), state, float(v), 0.0))
        elif sweep_axis == "roll":
            state = solver.solve(0.0, float(v), 0.0)
            frame_states.append((float(v), state, 0.0, float(v)))
        else:  # steer
            state = solver.solve(0.0, 0.0, float(v))
            frame_states.append((float(v), state, 0.0, 0.0))

    # Take the neutral state (closest to zero) for the initial frame
    idx_zero = int(np.argmin(np.abs(values)))
    initial_state = frame_states[idx_zero][1]
    initial_heave = frame_states[idx_zero][2]
    initial_roll  = frame_states[idx_zero][3]

    # ─── Initial frame: build the base figure ────────────────────────────────
    fig = plot_corner_3d(corner, tie_rod, state=initial_state, show_tire=show_tire,
                          title=f"Corner {corner.corner_id} — {sweep_axis} sweep")

    # ─── Generate the animation frames ───────────────────────────────────────
    # For each frame, recompute the positions including the chassis inboards
    # (which move under heave/roll, but not under pure steer).
    frames: list[go.Frame] = []
    slider_steps: list[dict] = []

    for slider_val, state, heave, roll in frame_states:
        pts_static = _collect_corner_points(corner, tie_rod, state)
        pts = _move_chassis_points_for_state(pts_static, heave, roll)

        # Re-generate all traces (in the same order as the base figure)
        frame_data: list = []

        # 1. Markers per group
        for grp in ["UCA", "LCA", "Tie-rod", "Wheel"]:
            grp_pts = [(n, pts[n]) for n in pts if POINT_TO_GROUP[n] == grp]
            xs = [p.x for _, p in grp_pts]
            ys = [p.y for _, p in grp_pts]
            zs = [p.z for _, p in grp_pts]
            frame_data.append(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="markers+text",
                marker=dict(size=6, color=GROUP_COLORS[grp]),
                text=[n for n, _ in grp_pts],
                textfont=dict(size=9, color=GROUP_COLORS[grp]),
                textposition="top center",
            ))

        # 2. Lines (same order as plot_corner_3d)
        def line_data(p1, p2):
            return go.Scatter3d(x=[p1.x, p2.x], y=[p1.y, p2.y], z=[p1.z, p2.z],
                                 mode="lines")

        line_specs = [
            (pts["UCA_IN_FRONT"], pts["UCA_OUT"]),
            (pts["UCA_IN_REAR"],  pts["UCA_OUT"]),
            (pts["LCA_IN_FRONT"], pts["LCA_OUT"]),
            (pts["LCA_IN_REAR"],  pts["LCA_OUT"]),
            (pts["TIE_ROD_IN"],   pts["TIE_ROD_OUT"]),
            (pts["UCA_OUT"],      pts["LCA_OUT"]),
            (pts["UCA_OUT"],      pts["TIE_ROD_OUT"]),
            (pts["LCA_OUT"],      pts["TIE_ROD_OUT"]),
            (pts["WHEEL_CENTER"], pts["CONTACT_PATCH"]),
        ]
        for p1, p2 in line_specs:
            frame_data.append(line_data(p1, p2))

        # 3. Tire rim
        if show_tire:
            xs_t, ys_t, zs_t = _generate_wheel_outline(
                pts["WHEEL_CENTER"], pts["CONTACT_PATCH"]
            )
            frame_data.append(go.Scatter3d(x=xs_t, y=ys_t, z=zs_t, mode="lines"))

        # 4. Ground (does not change)
        cp_static = pts["CONTACT_PATCH"]
        floor_size = 200.0
        frame_data.append(go.Mesh3d(
            x=[cp_static.x - floor_size, cp_static.x + floor_size,
               cp_static.x + floor_size, cp_static.x - floor_size],
            y=[cp_static.y - floor_size, cp_static.y - floor_size,
               cp_static.y + floor_size, cp_static.y + floor_size],
            z=[0, 0, 0, 0],
            i=[0, 0], j=[1, 2], k=[2, 3],
        ))

        # Add the frame with label "{sweep_axis}={value:.1f}"
        unit = {"heave": "mm", "roll": "°", "steer": "mm"}[sweep_axis]
        frame_name = f"{slider_val:+.2f}"
        frames.append(go.Frame(data=frame_data, name=frame_name))

        slider_steps.append({
            "args": [[frame_name], {"frame": {"duration": 50, "redraw": True},
                                     "mode": "immediate"}],
            "label": f"{slider_val:+.1f} {unit}",
            "method": "animate",
        })

    fig.frames = frames

    # ─── Slider and Play button ──────────────────────────────────────────────
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.05, "y": -0.05,
            "buttons": [
                {"label": "▶ Play",
                 "method": "animate",
                 "args": [None, {"frame": {"duration": 100, "redraw": True},
                                  "fromcurrent": True}]},
                {"label": "⏸ Pause",
                 "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}]},
            ],
        }],
        sliders=[{
            "active": idx_zero,
            "y": 0,  "x": 0.15,
            "len": 0.8,
            "currentvalue": {"prefix": f"{sweep_axis}: ", "visible": True},
            "steps": slider_steps,
        }],
    )

    return fig
