"""Render the VD Studio icon: ``apps/vdstudio/resources/vdstudio.{png,ico}``.

The icon is drawn with Qt so it can be regenerated without an image editor:

    uv run --extra desktop python scripts/make_icon.py

It is a front view of a double-wishbone corner: tyre, rim, the two wishbones
and the coil-over. The ``.ico`` holds PNG-compressed entries from 16 to 256 px
(the format Windows has read since Vista) for the desktop and Start-menu
shortcuts; the ``.png`` is the window and taskbar icon.
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "apps" / "vdstudio" / "resources"
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)

BG = "#1b2333"
TYRE = "#0d1117"
TYRE_EDGE = "#9aa4b2"
ACCENT = "#3b82f6"
LINK = "#e6e9ee"


def render(size: int):  # -> QtGui.QImage
    """Draw the icon at ``size`` px. Geometry is laid out on a 256 grid and scaled."""
    from PySide6 import QtCore, QtGui

    img = QtGui.QImage(size, size, QtGui.QImage.Format_ARGB32)
    img.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.scale(size / 256.0, size / 256.0)

    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(QtGui.QColor(BG))
    p.drawRoundedRect(QtCore.QRectF(8, 8, 240, 240), 44, 44)

    def pen(color: str, width: float) -> QtGui.QPen:
        q = QtGui.QPen(QtGui.QColor(color), width)
        q.setCapStyle(QtCore.Qt.RoundCap)
        q.setJoinStyle(QtCore.Qt.RoundJoin)
        return q

    # wishbones and coil-over, chassis side on the left (drawn first, the tyre covers the ends)
    upright_top, upright_bot = QtCore.QPointF(150, 86), QtCore.QPointF(150, 170)
    p.setPen(pen(LINK, 13))
    p.drawLine(QtCore.QPointF(40, 98), upright_top)
    p.drawLine(QtCore.QPointF(40, 176), upright_bot)
    p.setPen(pen(ACCENT, 12))
    p.drawLine(QtCore.QPointF(56, 62), QtCore.QPointF(118, 168))
    p.setPen(pen(ACCENT, 6))
    for i in range(5):                      # spring coils across the damper body
        t = 0.25 + 0.12 * i
        cx, cy = 56 + t * 62, 62 + t * 106
        p.drawLine(QtCore.QPointF(cx - 13, cy + 7), QtCore.QPointF(cx + 13, cy - 7))

    # tyre (front view: a tall rounded block) with the rim inside it
    p.setPen(pen(TYRE_EDGE, 5))
    p.setBrush(QtGui.QColor(TYRE))
    p.drawRoundedRect(QtCore.QRectF(146, 40, 78, 176), 26, 26)
    p.setPen(pen(ACCENT, 7))
    p.setBrush(QtGui.QColor(BG))
    p.drawRoundedRect(QtCore.QRectF(160, 80, 50, 96), 10, 10)
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(QtGui.QColor(LINK))
    p.drawEllipse(QtCore.QPointF(185, 128), 11, 11)       # hub
    p.end()
    return img


def png_bytes(img) -> bytes:
    from PySide6 import QtCore

    buf = QtCore.QBuffer()
    buf.open(QtCore.QIODevice.WriteOnly)
    if not img.save(buf, "PNG"):
        raise RuntimeError("Qt could not encode the icon as PNG")
    return bytes(buf.data())


def ico_bytes(pngs: dict[int, bytes]) -> bytes:
    """An ICO file whose entries are the given PNGs (one per size, 256 written as 0)."""
    sizes = sorted(pngs)
    header = struct.pack("<HHH", 0, 1, len(sizes))
    offset = len(header) + 16 * len(sizes)
    entries, blobs = b"", b""
    for s in sizes:
        data = pngs[s]
        d = 0 if s >= 256 else s
        entries += struct.pack("<BBBBHHII", d, d, 0, 0, 1, 32, len(data), offset)
        blobs += data
        offset += len(data)
    return header + entries + blobs


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtGui

    _app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication(sys.argv[:1])
    OUT.mkdir(parents=True, exist_ok=True)
    pngs = {s: png_bytes(render(s)) for s in ICO_SIZES}
    (OUT / "vdstudio.png").write_bytes(pngs[256])
    (OUT / "vdstudio.ico").write_bytes(ico_bytes(pngs))
    print(f"wrote {OUT / 'vdstudio.png'} and {OUT / 'vdstudio.ico'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
