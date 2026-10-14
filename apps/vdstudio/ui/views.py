"""Drawings of the suspension itself: 2D construction views and the 3D model.

All views draw a :class:`~apps.vdstudio.engine.pose.Pose` -- the static car is
just the zero pose -- so the same code serves the editors (static) and the
motion page (heave/roll/pitch/steer). Coordinates are ISO 8855 mm; the views
are drawn in the CHASSIS frame (the chassis is the reference, the road moves),
except the 3D view, which places the chassis in the world.

Front view (Y-Z): wishbones projected through the effective pivot at each ball
joint's X (the FVIC construction), FVIC, contact-patch-to-FVIC lines and the
roll centre where they meet. Side view (X-Z): pivot axes, the side-view
instant centre from lines of pivot-axis slope through the ball joints, and the
contact-patch-to-SVIC line whose slope sets anti-dive / anti-squat. Top view:
links, wheels at their toe/steer angle, and the steering-arm lines extended
toward the rear axle (the plan-view Ackermann construction -- shown for
intuition only; the quoted Ackermann is solved on the real linkage).
"""

from __future__ import annotations

import math
import os

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..engine.geometry import CarGeometry
from ..engine.pose import Pose, wheel_ring
from . import theme

CORNERS = ("FL", "FR", "RL", "RR")


def _pen(color: str, width: float = 2.0, dash: bool = False) -> QtGui.QPen:
    p = pg.mkPen(color, width=width)
    if dash:
        p.setStyle(QtCore.Qt.DashLine)
    p.setCosmetic(True)
    return p


def _line_intersect(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray
                    ) -> np.ndarray | None:
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
    return p1 + t * d1


def _eff_pivot(f: np.ndarray, r: np.ndarray, x: float) -> np.ndarray:
    a = r - f
    if abs(a[0]) < 1e-10:
        return 0.5 * (f + r)
    return f + (x - f[0]) / a[0] * a


class _View(pg.PlotWidget):
    def __init__(self, title: str, xl: str, yl: str) -> None:
        super().__init__()
        pi = self.getPlotItem()
        pi.setTitle(title, size="10pt")
        pi.setLabel("bottom", xl)
        pi.setLabel("left", yl)
        pi.showGrid(x=True, y=True, alpha=0.18)
        for ax in ("left", "bottom"):
            pi.getAxis(ax).enableAutoSIPrefix(False)
        self.setAspectLocked(True)
        self.setMinimumHeight(260)
        self._auto = True

    def _clear(self) -> None:
        self.clear()

    def line(self, pts: np.ndarray, color: str, width: float = 2.0, dash: bool = False) -> None:
        pts = np.asarray(pts, float)
        self.plot(pts[:, 0], pts[:, 1], pen=_pen(color, width, dash))

    def dots(self, pts: np.ndarray, color: str, size: float = 7, symbol: str = "o") -> None:
        pts = np.atleast_2d(np.asarray(pts, float))
        self.plot(pts[:, 0], pts[:, 1], pen=None, symbol=symbol, symbolSize=size,
                  symbolBrush=pg.mkBrush(color), symbolPen=pg.mkPen(None))

    def text(self, xy: np.ndarray, s: str, color: str) -> None:
        t = pg.TextItem(s, color=color, anchor=(0, 1))
        t.setPos(float(xy[0]), float(xy[1]))
        self.addItem(t)


class FrontView(_View):
    def __init__(self) -> None:
        # Standing ahead of the car looking rearward, its LEFT (+Y) is on the right.
        super().__init__("Front view (looking rearward)", "Y [mm]  (+ = left)", "Z [mm]")

    def draw(self, geo: CarGeometry, pose: Pose, axle: str, tyre_width: float = 190.0,
             show_construction: bool = True) -> None:
        t = theme.current()
        self._clear()
        lc, rc = ("FL", "FR") if axle == "front" else ("RL", "RR")
        half = max(abs(geo.points[lc]["CONTACT_PATCH"][1]), 1.0)
        ground_z = 0.0
        self.line(np.array([[-half - 300, ground_z], [half + 300, ground_z]]), t.text_muted, 1)
        self.line(np.array([[0, -50], [0, 500]]), t.grid, 1, dash=True)
        lines = []
        for cid in (lc, rc):
            cp = pose.corners[cid]
            P = cp.points
            yz = {k: np.array([v[1], v[2]]) for k, v in P.items()}
            r = geo.axles[axle].loaded_radius_mm
            # tyre outline: rectangle in the wheel plane, rotated by camber
            cam = math.radians(cp.camber_deg)
            side = 1.0 if cid in ("FL", "RL") else -1.0
            # wheel-plane "up": negative camber tips the top INBOARD on both sides
            up = np.array([side * math.sin(cam), math.cos(cam)])
            out = np.array([side * math.cos(cam), -math.sin(cam)])
            wc = yz["WHEEL_CENTER"]
            corners = [wc + a * r * up + b * tyre_width / 2 * out
                       for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1))]
            self.line(np.array(corners), t.text_muted, 1.5)
            for arm, col in (("UCA", t.series[0]), ("LCA", t.series[1])):
                bj = P[f"{arm}_OUT"]
                piv = _eff_pivot(P[f"{arm}_IN_FRONT"], P[f"{arm}_IN_REAR"], bj[0])
                self.line(np.array([[piv[1], piv[2]], [bj[1], bj[2]]]), col, 3)
                self.dots(np.array([[piv[1], piv[2]], [bj[1], bj[2]]]), col, 7)
                lines.append((cid, arm, np.array([piv[1], piv[2]]), np.array([bj[1], bj[2]])))
            self.line(np.array([yz["TIE_ROD_IN"], yz["TIE_ROD_OUT"]]), t.series[2], 2)
            self.line(np.array([yz["LCA_OUT"], yz["UCA_OUT"]]), t.series[3], 3)
            # kingpin to the ground
            kp = yz["UCA_OUT"] - yz["LCA_OUT"]
            if abs(kp[1]) > 1e-9:
                g = yz["LCA_OUT"] + (ground_z - yz["LCA_OUT"][1]) / kp[1] * kp
                self.line(np.array([yz["LCA_OUT"], g]), t.series[3], 1, dash=True)
            self.dots(yz["CONTACT_PATCH"], t.text, 8, "s")
            self.dots(wc, t.text_muted, 6, "+")
        if not show_construction:
            return
        ics = {}
        for cid in (lc, rc):
            (_, _, p_u, b_u), (_, _, p_l, b_l) = [x for x in lines if x[0] == cid]
            ic = _line_intersect(b_u, p_u - b_u, b_l, p_l - b_l)
            if ic is None or np.linalg.norm(ic) > 2e5:
                continue
            ics[cid] = ic
            for b in (b_u, b_l):
                self.line(np.array([b, ic]), t.text_muted, 1, dash=True)
            self.dots(ic, t.accent, 9, "d")
            cp = np.array(pose.corners[cid].points["CONTACT_PATCH"][1:])
            self.line(np.array([cp, ic]), t.accent, 1, dash=True)
        if len(ics) == 2:
            cpl = np.array(pose.corners[lc].points["CONTACT_PATCH"][1:])
            cpr = np.array(pose.corners[rc].points["CONTACT_PATCH"][1:])
            rcpt = _line_intersect(cpl, ics[lc] - cpl, cpr, ics[rc] - cpr)
            if rcpt is not None and abs(rcpt[0]) < 5e3:
                self.dots(rcpt, t.bad, 12, "o")
                self.text(rcpt + np.array([0, 15]), f" RC {rcpt[1]:.1f} mm", t.bad)
        self.setRange(xRange=(-half - 250, half + 250), yRange=(-60, 520), padding=0.02)


class SideView(_View):
    def __init__(self) -> None:
        super().__init__("Side view, left side (front to the left)", "X [mm]  (+ = forward)",
                         "Z [mm]")
        self.invertX(True)

    def draw(self, geo: CarGeometry, pose: Pose, show_construction: bool = True) -> None:
        t = theme.current()
        self._clear()
        xs = [pose.corners[c].points["WHEEL_CENTER"][0] for c in ("FL", "RL")]
        self.line(np.array([[max(xs) + 600, 0], [min(xs) - 600, 0]]), t.text_muted, 1)
        veh = geo.vehicle_data
        cg = np.array([-veh.cg_from_front_axle_mm * geo.wheelbase_mm / veh.wheelbase_mm,
                       veh.cg_height_mm])
        self.dots(cg, t.warn, 12, "o")
        self.text(cg + np.array([0, 20]), " CG", t.warn)
        for cid in ("FL", "RL"):
            P = pose.corners[cid].points
            xz = {k: np.array([v[0], v[2]]) for k, v in P.items()}
            r = geo.axles["front" if cid == "FL" else "rear"].loaded_radius_mm
            th = np.linspace(0, 2 * math.pi, 64)
            ring = xz["WHEEL_CENTER"] + r * np.column_stack([np.cos(th), np.sin(th)])
            self.line(ring, t.text_muted, 1.5)
            slopes = {}
            for arm, col in (("UCA", t.series[0]), ("LCA", t.series[1])):
                f, rr, bj = xz[f"{arm}_IN_FRONT"], xz[f"{arm}_IN_REAR"], xz[f"{arm}_OUT"]
                self.line(np.array([f, rr]), col, 3)
                self.line(np.array([f, bj]), col, 1.5)
                self.line(np.array([rr, bj]), col, 1.5)
                self.dots(np.array([f, rr, bj]), col, 7)
                run = rr[0] - f[0]
                slopes[arm] = (rr[1] - f[1]) / run if abs(run) > 1e-9 else None
            self.line(np.array([xz["LCA_OUT"], xz["UCA_OUT"]]), t.series[3], 3)
            self.line(np.array([xz["TIE_ROD_IN"], xz["TIE_ROD_OUT"]]), t.series[2], 2)
            cp = xz["CONTACT_PATCH"]
            self.dots(cp, t.text, 8, "s")
            if not show_construction or None in slopes.values():
                continue
            mu, ml = slopes["UCA"], slopes["LCA"]
            ubj, lbj = xz["UCA_OUT"], xz["LCA_OUT"]
            if abs(mu - ml) < 1e-12:
                d = np.array([1.0, ml])
                d /= np.linalg.norm(d)
                self.line(np.array([cp, cp + 1500 * d * (1 if cid == "RL" else -1)]),
                          t.accent, 1, dash=True)
                self.text(cp + np.array([0, -30]), " SVIC at ∞", t.accent)
                continue
            x = ((ubj[1] - lbj[1]) + ml * lbj[0] - mu * ubj[0]) / (ml - mu)
            sv = np.array([x, lbj[1] + ml * (x - lbj[0])])
            if np.linalg.norm(sv - cp) > 2e4:
                d = (sv - cp) / np.linalg.norm(sv - cp)
                self.line(np.array([cp, cp + 1500 * d]), t.accent, 1, dash=True)
                continue
            for b in (ubj, lbj):
                self.line(np.array([b, sv]), t.text_muted, 1, dash=True)
            self.line(np.array([cp, sv]), t.accent, 1, dash=True)
            self.dots(sv, t.accent, 9, "d")


class TopView(_View):
    def __init__(self) -> None:
        super().__init__("Top view (front up)", "Y [mm]  (+ = left)", "X [mm]  (+ = forward)")
        self.invertX(True)

    def draw(self, geo: CarGeometry, pose: Pose, tyre_width: float = 190.0,
             show_construction: bool = True) -> None:
        t = theme.current()
        self._clear()
        for cid in CORNERS:
            cp = pose.corners[cid]
            P = cp.points
            yx = {k: np.array([v[1], v[0]]) for k, v in P.items()}
            r = geo.axles["front" if cid[0] == "F" else "rear"].loaded_radius_mm
            # wheel footprint rotated by toe (toe-in positive for both sides)
            side = 1.0 if cid in ("FL", "RL") else -1.0
            toe = math.radians(cp.toe_deg_per_side)
            fwd = np.array([-side * math.sin(toe), math.cos(toe)])   # (y, x)
            lat = np.array([fwd[1] * side, -fwd[0] * side])
            wc = yx["WHEEL_CENTER"]
            rect = [wc + a * r * fwd + b * tyre_width / 2 * lat
                    for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1))]
            self.line(np.array(rect), t.text_muted, 1.5)
            for arm, col in (("UCA", t.series[0]), ("LCA", t.series[1])):
                self.line(np.array([yx[f"{arm}_IN_FRONT"], yx[f"{arm}_OUT"],
                                    yx[f"{arm}_IN_REAR"]]), col, 2)
            self.line(np.array([yx["TIE_ROD_IN"], yx["TIE_ROD_OUT"]]), t.series[2], 2.5)
            self.dots(np.array([yx["TIE_ROD_IN"], yx["TIE_ROD_OUT"]]), t.series[2], 6)
            if show_construction and cid in ("FL", "FR"):
                kp = 0.5 * (yx["UCA_OUT"] + yx["LCA_OUT"])
                d = yx["TIE_ROD_OUT"] - kp
                if np.linalg.norm(d) > 1e-6:
                    reach = d / np.linalg.norm(d) * max(geo.wheelbase_mm * 1.2, 500)
                    self.line(np.array([kp, kp - reach]), t.text_muted, 1, dash=True)
        fl, fr = pose.corners["FL"].points, pose.corners["FR"].points
        self.line(np.array([[fl["TIE_ROD_IN"][1], fl["TIE_ROD_IN"][0]],
                            [fr["TIE_ROD_IN"][1], fr["TIE_ROD_IN"][0]]]), t.series[4], 4)
        rx = pose.corners["RL"].points["WHEEL_CENTER"][0]
        self.line(np.array([[-800, rx], [800, rx]]), t.grid, 1, dash=True)


# --------------------------------------------------------------------------- #
# 3D
# --------------------------------------------------------------------------- #

def gl_available() -> tuple[bool, str]:
    if os.environ.get("VDSTUDIO_NO_GL"):
        return False, "3D view disabled (VDSTUDIO_NO_GL is set)."
    try:
        import OpenGL  # noqa: F401
        import pyqtgraph.opengl  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"3D view unavailable: {exc}. Install PyOpenGL."
    return True, ""


class CarGL(QtWidgets.QWidget):
    """OpenGL model of the whole car at a pose (world frame, road at z = 0)."""

    def __init__(self) -> None:
        super().__init__()
        import pyqtgraph.opengl as gl

        self.gl = gl
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(theme.current().plot_bg)
        lay.addWidget(self.view)
        self.view.opts["distance"] = 4200
        self.view.opts["elevation"] = 22
        self.view.opts["azimuth"] = -128
        self.view.opts["center"] = pg.Vector(-770, 0, 150)
        self.view.opts["fov"] = 45
        grid = gl.GLGridItem()
        grid.setSize(4000, 2400)
        grid.setSpacing(100, 100)
        grid.translate(-770, 0, 0)
        grid.setColor((120, 130, 145, 70))
        self.view.addItem(grid)
        self.items: dict[str, object] = {}
        for key, width in (("uca", 4), ("lca", 4), ("tie", 3), ("upright", 5), ("wheel", 2),
                           ("chassis", 2), ("roll", 2), ("pivot", 1)):
            it = gl.GLLinePlotItem(mode="lines", width=width, antialias=True)
            self.view.addItem(it)
            self.items[key] = it
        self.points = gl.GLScatterPlotItem(size=9, pxMode=True)
        self.view.addItem(self.points)
        self.markers = gl.GLScatterPlotItem(size=14, pxMode=True)
        self.view.addItem(self.markers)
        self.labels: list[object] = []
        self.show_labels = False
        self.show_chassis = True
        self.show_rc = True

    def preset(self, name: str) -> None:
        o = self.view.opts
        o["center"] = pg.Vector(-770, 0, 150)
        if name == "front":
            o["elevation"], o["azimuth"], o["distance"] = 2, 0, 3200
        elif name == "side":
            o["elevation"], o["azimuth"], o["distance"] = 2, 90, 3800
        elif name == "top":
            o["elevation"], o["azimuth"], o["distance"] = 89, 0, 3800
        elif name == "corner":
            o["center"] = pg.Vector(0, 450, 220)
            o["elevation"], o["azimuth"], o["distance"] = 18, -60, 1500
        else:
            o["elevation"], o["azimuth"], o["distance"] = 22, -128, 4200
        self.view.update()

    @staticmethod
    def _color(hex_: str, a: float = 1.0) -> tuple[float, float, float, float]:
        c = QtGui.QColor(hex_)
        return (c.redF(), c.greenF(), c.blueF(), a)

    def draw(self, geo: CarGeometry, pose: Pose, tyre_width: float = 190.0) -> None:
        t = theme.current()
        W = pose.to_world
        seg: dict[str, list[np.ndarray]] = {k: [] for k in self.items}
        pts, pcol = [], []
        for cid in CORNERS:
            P = pose.corners[cid].points
            for arm, key in (("UCA", "uca"), ("LCA", "lca")):
                seg[key] += [P[f"{arm}_IN_FRONT"], P[f"{arm}_OUT"],
                             P[f"{arm}_IN_REAR"], P[f"{arm}_OUT"]]
                seg["pivot"] += [P[f"{arm}_IN_FRONT"], P[f"{arm}_IN_REAR"]]
            seg["tie"] += [P["TIE_ROD_IN"], P["TIE_ROD_OUT"]]
            seg["upright"] += [P["LCA_OUT"], P["UCA_OUT"], P["LCA_OUT"], P["TIE_ROD_OUT"],
                               P["UCA_OUT"], P["TIE_ROD_OUT"], P["LCA_OUT"], P["WHEEL_CENTER"],
                               P["UCA_OUT"], P["WHEEL_CENTER"]]
            r = geo.axles["front" if cid[0] == "F" else "rear"].loaded_radius_mm
            ax = pose.corners[cid].spin_axis
            for off in (-tyre_width / 2, tyre_width / 2):
                ring = wheel_ring(P["WHEEL_CENTER"] + off * ax, ax, r)
                for a, b in zip(ring[:-1], ring[1:], strict=True):
                    seg["wheel"] += [a, b]
            ring_o = wheel_ring(P["WHEEL_CENTER"] + tyre_width / 2 * ax, ax, r, 12)
            ring_i = wheel_ring(P["WHEEL_CENTER"] - tyre_width / 2 * ax, ax, r, 12)
            for a, b in zip(ring_o, ring_i, strict=True):
                seg["wheel"] += [a, b]
            for name in ("UCA_IN_FRONT", "UCA_IN_REAR", "LCA_IN_FRONT", "LCA_IN_REAR",
                         "TIE_ROD_IN"):
                pts.append(P[name])
                pcol.append(self._color(t.text_muted))
            for name in ("UCA_OUT", "LCA_OUT", "TIE_ROD_OUT"):
                pts.append(P[name])
                pcol.append(self._color(t.text))
            pts.append(P["CONTACT_PATCH"])
            pcol.append(self._color(t.warn))
        if self.show_chassis:
            c = pose.corners
            for arm in ("UCA", "LCA"):
                for side in ("L", "R"):
                    seg["chassis"] += [c[f"F{side}"].points[f"{arm}_IN_REAR"],
                                       c[f"R{side}"].points[f"{arm}_IN_FRONT"]]
                seg["chassis"] += [c["FL"].points[f"{arm}_IN_FRONT"],
                                   c["FR"].points[f"{arm}_IN_FRONT"],
                                   c["RL"].points[f"{arm}_IN_REAR"],
                                   c["RR"].points[f"{arm}_IN_REAR"]]
        mk, mcol = [], []
        rcs = [pose.rc_chassis.get(a) for a in ("front", "rear")]
        if self.show_rc and all(r is not None for r in rcs):
            seg["roll"] += [rcs[0], rcs[1]]
            for r in rcs:
                mk.append(r)
                mcol.append(self._color(t.bad))
        veh = geo.vehicle_data
        cg = np.array([-veh.cg_from_front_axle_mm * geo.wheelbase_mm / veh.wheelbase_mm, 0.0,
                       veh.cg_height_mm])
        mk.append(cg)
        mcol.append(self._color(t.warn))
        colors = {"uca": t.series[0], "lca": t.series[1], "tie": t.series[2],
                  "upright": t.series[3], "wheel": t.text_muted, "chassis": t.info,
                  "roll": t.bad, "pivot": t.text_muted}
        alphas = {"wheel": 0.75, "chassis": 0.6, "pivot": 0.5}
        for key, it in self.items.items():
            s = seg[key]
            if s:
                it.setData(pos=W(np.array(s)), color=self._color(colors[key], alphas.get(key, 1)))
                it.setVisible(True)
            else:
                it.setVisible(False)
        self.points.setData(pos=W(np.array(pts)), color=np.array(pcol), size=8)
        self.markers.setData(pos=W(np.array(mk)), color=np.array(mcol), size=13)
        for lab in self.labels:
            self.view.removeItem(lab)
        self.labels = []
        if self.show_labels and hasattr(self.gl, "GLTextItem"):
            for cid in CORNERS:
                p = W(pose.corners[cid].points["WHEEL_CENTER"][None, :])[0]
                lab = self.gl.GLTextItem(pos=p + np.array([0, 0, 320]), text=cid,
                                         color=QtGui.QColor(t.text))
                self.view.addItem(lab)
                self.labels.append(lab)

    def grab_image(self) -> QtGui.QImage:
        return self.view.grabFramebuffer()
