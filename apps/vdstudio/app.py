"""Entry point: ``uv run --extra desktop python -m apps.vdstudio [project.vdproj.json]``."""

from __future__ import annotations

import multiprocessing
import os
import sys


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()
    argv = list(sys.argv if argv is None else argv)
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

    from PySide6 import QtCore, QtWidgets

    from . import APP_NAME
    from .ui import theme
    from .ui.controller import ProjectController
    from .ui.main_window import MainWindow

    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("PUCPR Racing")
    saved = QtCore.QSettings("PUCPR Racing", APP_NAME).value("theme", "dark")
    theme.apply(app, theme.LIGHT if saved == "light" else theme.DARK)

    ctrl = ProjectController()
    win = MainWindow(ctrl)
    if len(argv) > 1 and os.path.isfile(argv[1]):
        win.open_project(argv[1])
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
