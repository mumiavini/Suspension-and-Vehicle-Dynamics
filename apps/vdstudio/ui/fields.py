"""Editor field specifications for the design inputs (one place, reused by pages).

Synthesis inputs live in the scripts' DESIGN frame: y OUTBOARD, z UP, x
REARWARD of the front axle. Hardpoint tables are ISO 8855. Tooltips say which.
"""

from __future__ import annotations

from ..engine.project import SOURCES
from .widgets import Field

DI = " (design_intent)"


def vehicle_fields() -> list[tuple[str, list[Field]]]:
    v = "vehicle."
    return [
        ("Mass & CG" + DI, [
            Field(v + "total_mass_kg", "Total mass (with driver)", "kg", 50, 1000, 1, 1),
            Field(v + "unsprung_mass_kg", "Unsprung mass (4 corners)", "kg", 0, 200, 0.5, 1),
            Field(v + "cg_height_mm", "CG height", "mm", 50, 1000, 5, 1),
            Field(v + "wheelbase_mm", "Wheelbase", "mm", 1000, 3500, 5, 1,
                  tip="Authoritative in synthesis: the rear axle sits at the front axle station "
                      "+ this wheelbase (the CG station is kept, so the weight distribution "
                      "follows). In hardpoint mode the wheelbase is read from the points."),
            Field(v + "cg_from_front_axle_mm", "CG behind front axle", "mm", 0, 3000, 5, 1,
                  tip="Sets the weight distribution: front fraction = 1 - x/L."),
        ]),
        ("Roll budget targets" + DI, [
            Field(v + "target_roll_gradient_deg_per_g", "Target roll gradient", "deg/g", 0.1,
                  5, 0.05, 2),
            Field(v + "chassis_factor_min", "Chassis torsion factor (min)", "×", 1, 20, 0.5, 1),
            Field(v + "chassis_factor_target", "Chassis torsion factor (target)", "×", 1, 20,
                  0.5, 1),
            Field(v + "tilt_test_angle_deg", "Tilt-test angle", "deg", 30, 80, 1, 0),
        ]),
        ("Provenance of non-point inputs", [
            Field("mass_source", "Mass / CG provenance", kind="choice", choices=SOURCES,
                  tip="Set to 'estimate' automatically when the subsystem CG estimate is "
                      "applied. Counted in the status bar's estimate-tagged inputs."),
            Field("tire_source", "Tyre loaded radius provenance (synthesis)", kind="choice",
                  choices=SOURCES,
                  tip="An estimate until TTC data exists. Hardpoint mode has its own field "
                      "per axle on the Hardpoints page."),
        ]),
        ("Load-case assumptions (tyre: no TTC data)", [
            Field(v + "brake_bias_front", "Brake bias front", "-", 0, 1, 0.01, 2),
            Field(v + "lltd_front", "LLTD front share", "-", 0, 1, 0.01, 2,
                  tip="Front share of total lateral load transfer used by the load cases."),
            Field(v + "mu_lateral", "Lateral μ", "-", 0.1, 3, 0.05, 2),
            Field(v + "mu_longitudinal", "Longitudinal μ", "-", 0.1, 3, 0.05, 2),
            Field(v + "design_ay_g", "Design lateral acceleration", "g", 0.1, 4, 0.05, 2),
        ]),
    ]


def axle_fields(axle: str) -> list[tuple[str, list[Field]]]:
    a = axle + "."
    dfr = "DESIGN frame: y outboard from the car centreline, z up from the ground."
    return [
        ("Wheel & tyre package", [
            Field(a + "track_mm", "Track (contact patches)", "mm", 800, 2000, 5, 1,
                  tip="Measured between contact patches: the rules / tilt-test datum."),
            Field(a + "loaded_radius_mm", "Tyre loaded radius", "mm", 150, 400, 1, 1),
            Field(a + "rim_diameter_in", "Rim diameter", "in", 8, 18, 1, 0),
            Field(a + "static_camber_deg", "Static camber" + DI, "deg", -6, 3, 0.1, 2,
                  tip="Built into the upright; negative = top inboard."),
        ]),
        ("Outboard: ball joints & kingpin", [
            Field(a + "lbj_y_mm", "LBJ y (outboard)", "mm", 100, 1000, 1, 2, tip=dfr),
            Field(a + "lbj_z_mm", "LBJ z", "mm", 20, 400, 1, 2, tip=dfr),
            Field(a + "kpi_deg", "KPI" + DI, "deg", -5, 25, 0.25, 2),
            Field(a + "kingpin_length_mm", "Kingpin length (front view)", "mm", 100, 450, 1, 2),
        ]),
        ("Front-view targets" + DI, [
            Field(a + "rc_height_mm", "Roll centre height", "mm", -100, 300, 1, 1,
                  tip="Places the FVIC: z_ic = h_rc · FVSA / (T/2)."),
            Field(a + "fvsa_length_mm", "FVSA length", "mm", 200, 10000, 25, 0,
                  tip="Places the FVIC: y_ic = T/2 - FVSA."),
        ]),
        ("Inboard: chassis pickup planes", [
            Field(a + "inner_pickup_y_mm", "Shared pickup plane y", "mm", 0, 600, 1, 1, tip=dfr),
            Field(a + "lca_inner_y_mm", "LCA plane override", "mm", 0, 600, 1, 1, "optfloat",
                  tip="Unchecked: use the shared plane. Only the arm length changes."),
            Field(a + "uca_inner_y_mm", "UCA plane override", "mm", 0, 600, 1, 1, "optfloat"),
        ]),
        ("Side view: layout & anti-geometry", [
            Field(a + "axle_x_mm", "Axle station (rearward of front)", "mm", -100, 4000, 5, 1,
                  tip="DESIGN frame: x positive REARWARD from the front axle." if axle == "front"
                  else "Derived, read-only: front axle station + vehicle wheelbase (edit the "
                       "wheelbase in the vehicle data). DESIGN frame: x positive REARWARD."),
            Field(a + "lca_base_mm", "LCA base", "mm", 20, 800, 5, 1),
            Field(a + "lca_sweep_mm", "LCA sweep", "mm", -500, 500, 5, 1),
            Field(a + "uca_base_mm", "UCA base", "mm", 20, 800, 5, 1),
            Field(a + "uca_sweep_mm", "UCA sweep", "mm", -500, 500, 5, 1),
            Field(a + "dz_lca_mm", "LCA pivot-axis rise", "mm", -200, 200, 0.5, 3,
                  tip="z(rear pickup) - z(front pickup). Inclines the axis: anti-geometry."),
            Field(a + "dz_uca_mm", "UCA pivot-axis rise", "mm", -200, 200, 0.5, 3,
                  tip="On the 2027 front a NEGATIVE value gives anti-dive."),
        ]),
        ("Sweep settings", [
            Field(a + "travel_bump_mm", "Bump travel", "mm", 1, 100, 1, 1),
            Field(a + "travel_droop_mm", "Droop travel", "mm", 1, 100, 1, 1),
            Field(a + "roll_reference_deg", "Reference roll angle", "deg", 0.1, 6, 0.1, 2),
        ]),
    ]


def axle_band_fields(axle: str) -> list[tuple[str, list[Field]]]:
    a = axle + ".limits."
    bands = [
        ("rc_height_mm", "Roll centre height", "mm", 1, 1),
        ("fvsa_length_mm", "FVSA length", "mm", 25, 0),
        ("scrub_radius_mm", "Scrub radius", "mm", 1, 1),
        ("kpi_deg", "KPI", "deg", 0.5, 1),
        ("kingpin_length_mm", "Kingpin length", "mm", 5, 0),
        ("lca_length_mm", "Wishbone leg length", "mm", 5, 0),
        ("uca_lca_ratio", "UCA / LCA ratio", "-", 0.01, 2),
        ("camber_gain_deg_per_mm", "|Camber gain|", "deg/mm", 0.001, 4),
        ("anti_percent", "Anti-dive / anti-squat", "%", 1, 1),
        ("outer_camber_in_roll_deg", "Outer camber in roll", "deg", 0.1, 2),
    ]
    rows: list[Field] = []
    for key, lab, unit, step, dec in bands:
        rows.append(Field(a + key + ".0", lab + " — min", unit, -1e5, 1e5, step, dec))
        rows.append(Field(a + key + ".1", lab + " — max", unit, -1e5, 1e5, step, dec))
    rows.append(Field(a + "ea_ratio_max", "Sweep e/a — max", "-", 0, 10, 0.1, 2))
    rows.append(Field(a + "ball_joint_clearance_mm", "Ball-joint clearance to rim", "mm", 0, 100,
                      1, 1))
    return [(f"{axle.title()} design-intent bands (information, not targets)", rows)]


def steering_fields() -> list[tuple[str, list[Field]]]:
    s = "steering."
    dfr = "DESIGN frame: x rearward, y outboard, z up."
    return [
        ("Kingpin in side view" + DI, [
            Field(s + "caster_deg", "Caster", "deg", -5, 15, 0.25, 2),
            Field(s + "caster_offset_mm", "Caster offset (axis shift)", "mm", -80, 80, 1, 1,
                  tip="Translates the kingpin fore-aft: trail without changing caster."),
        ]),
        ("Steering arm" + DI, [
            Field(s + "tro_height_along_kingpin_mm", "Height along kingpin from LBJ", "mm", -100,
                  300, 1, 2),
            Field(s + "steer_arm_length_mm", "Arm length", "mm", 20, 200, 1, 2),
            Field(s + "steer_arm_angle_deg", "Arm angle", "deg", -60, 60, 0.5, 2,
                  tip="Rotation of the arm about the kingpin from straight ahead; sets "
                      "Ackermann."),
        ]),
        ("Rack (lateral, horizontal)" + DI, [
            Field(s + "rack_x_mm", "Rack x (rearward)", "mm", -300, 300, 1, 2, tip=dfr),
            Field(s + "rack_z_mm", "Rack z", "mm", 0, 400, 0.5, 4,
                  tip="Main bump-steer knob."),
            Field(s + "rack_half_length_mm", "Rack half length (inner joint y)", "mm", 50, 500, 1,
                  2, tip=dfr),
        ]),
        ("Rack hardware & driver", [
            Field(s + "pinion_radius_mm", "Pinion radius", "mm", 5, 50, 0.5, 2),
            Field(s + "max_rack_travel_mm", "Max rack travel (half stroke)", "mm", 5, 100, 1, 1),
            Field(s + "steering_wheel_diameter_mm", "Steering wheel diameter", "mm", 150, 400, 5,
                  0),
            Field(s + "mu_parking", "Parking μ (tyre assumption)", "-", 0.1, 3, 0.05, 2),
        ]),
        ("Alignment & references", [
            Field(s + "static_toe_deg_per_side", "Static toe per side (+ in)" + DI, "deg", -3, 3,
                  0.05, 3),
            Field(s + "ackermann_at_steer_deg", "Ackermann reference outer angle", "deg", 1, 40,
                  1, 1),
        ]),
    ]


def steering_band_fields() -> list[tuple[str, list[Field]]]:
    s = "steering.limits."
    rows: list[Field] = []
    for key, lab, unit, step, dec in (
            ("bump_steer_deg_per_mm", "|Bump steer|", "deg/mm", 0.0005, 4),
            ("ackermann_pct", "Ackermann", "%", 5, 0),
            ("steering_ratio", "Steering ratio", ":1", 0.1, 1),
            ("tie_rod_length_mm", "Tie rod length", "mm", 5, 0),
            ("mechanical_trail_mm", "Mechanical trail", "mm", 1, 0),
            ("rack_x_window_mm", "Rack x window", "mm", 5, 0),
            ("rack_z_window_mm", "Rack z window", "mm", 5, 0)):
        rows.append(Field(s + key + ".0", lab + " — min", unit, -1e5, 1e5, step, dec))
        rows.append(Field(s + key + ".1", lab + " — max", unit, -1e5, 1e5, step, dec))
    rows.append(Field(s + "rod_end_misalignment_deg", "Tie-rod swing — max", "deg", 0, 60, 1, 1))
    rows.append(Field(s + "steering_wheel_torque_Nm", "Steering-wheel torque — max", "N·m", 0, 50,
                      0.5, 1))
    return [("Steering design-intent bands (information, not targets)", rows)]


def layout_fields(axle: str) -> list[tuple[str, list[Field]]]:
    """Brake / drive layout: where the torque reacts sets the anti-geometry reference point."""
    ref = (" Inboard: the torque reacts on the chassis, so the anti line runs from the WHEEL "
           "CENTRE to the side-view IC. Outboard: it reacts on the upright, so the line runs "
           "from the CONTACT PATCH. Source: design_intent.")
    lay = "layout."
    if axle == "front":
        rows = [Field(lay + "front_brakes_inboard", "Front brakes inboard", kind="bool",
                      tip="Sets the anti-dive reference point." + ref)]
    else:
        rows = [Field(lay + "rear_drive_inboard", "Drive inboard (chassis diff + driveshafts)",
                      kind="bool", tip="Sets the anti-squat reference point. The 2027 car has "
                                       "a chassis-mounted diff with driveshafts." + ref),
                Field(lay + "rear_brakes_inboard", "Rear brakes inboard", kind="bool",
                      tip="Sets the anti-lift (braking) reference point." + ref)]
    return [("Anti-geometry reference: brake / drive layout" + DI, rows)]


def toe_link_fields() -> list[tuple[str, list[Field]]]:
    rows: list[Field] = []
    for end, lab in (("inboard", "Inboard (chassis)"), ("outboard", "Outboard (upright)")):
        for i, ax in enumerate("XYZ"):
            rows.append(Field(f"rear_toe_link_{end}.{i}", f"{lab} {ax}", "mm", -5000, 5000, 1, 3,
                              tip="ISO 8855, LEFT side (mirrored to the right). X is entered "
                                  "for the declared rear axle station below and moves with the "
                                  "axle; absolute Y/Z: retune after changing rear track, LBJ or "
                                  "pickup planes. Z inboard is the rear bump-steer knob."))
    rows.append(Field("rear_toe_link_axle_x_mm", "Declared at rear axle station", "mm", -100,
                      4000, 5, 1,
                      tip="DESIGN frame (x rearward). The rear axle station the X values above "
                          "were entered against; synthesis shifts both X values by the "
                          "difference to the current rear axle station."))
    rows.append(Field("rear_static_toe_deg_per_side", "Rear static toe per side (+ in)" + DI,
                      "deg", -3, 3, 0.05, 3))
    return [("Rear toe link (declared design input, ISO 8855 left side)", rows)]
