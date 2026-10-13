"""Entry point: ``uv run --extra desktop python -m apps.vdstudio [project.vdproj.json]``."""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import sys
import traceback
from pathlib import Path
from types import TracebackType
from typing import TextIO

ICON_PNG = Path(__file__).resolve().parent / "resources" / "vdstudio.png"
APP_USER_MODEL_ID = "PUCPRRacing.VDStudio"
LOG_MAX_BYTES = 1_000_000


def log_path() -> Path:
    """Where a windowless start (the desktop shortcut, ``pythonw``) writes its output."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_STATE_HOME") \
        or str(Path.home() / ".local" / "state")
    return Path(base) / "VD Studio" / "vdstudio.log"


def redirect_missing_streams(path: Path | None = None) -> Path | None:
    """Give ``sys.stdout`` / ``sys.stderr`` a log file when there is no console.

    Under ``pythonw`` both are ``None``: a stray ``print`` raises and every
    traceback is lost. Returns the log path, or ``None`` when a console exists.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return None
    path = path or log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if path.is_file() and path.stat().st_size > LOG_MAX_BYTES else "a"
    stream: TextIO = open(path, mode, encoding="utf-8", buffering=1)  # noqa: SIM115 (process lifetime)
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream
    return path


def _native_error_box(text: str) -> None:
    """A message box that needs no Qt (for failures before the QApplication exists)."""
    if os.name == "nt":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, "VD Studio", 0x10)  # MB_ICONERROR


def install_error_dialog(log: Path) -> None:
    """Show unhandled exceptions in a dialog: a windowless start has no console to show them."""
    busy = False

    def hook(etype: type[BaseException], value: BaseException,
             tb: TracebackType | None) -> None:
        nonlocal busy
        print("".join(traceback.format_exception(etype, value, tb)), file=sys.stderr, flush=True)
        if busy or issubclass(etype, KeyboardInterrupt):
            return
        from PySide6 import QtWidgets

        if QtWidgets.QApplication.instance() is None:
            return
        busy = True
        try:
            QtWidgets.QMessageBox.critical(
                None, "VD Studio",
                f"Unexpected error: {etype.__name__}: {value}\n\nThe full traceback is in\n{log}")
        finally:
            busy = False

    sys.excepthook = hook


def _set_windows_taskbar_identity() -> None:
    """Group the window under its own taskbar icon instead of python's."""
    if os.name == "nt":
        import ctypes

        with contextlib.suppress(AttributeError, OSError):
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()
    argv = list(sys.argv if argv is None else argv)
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
    log = redirect_missing_streams()
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as e:
        msg = (f"VD Studio could not start: {e}.\n\nInstall the desktop dependencies "
               "(uv sync --extra desktop --inexact), or run "
               "'Install VD Studio shortcut.bat' again.")
        print(msg, file=sys.stderr, flush=True)
        if log is not None:
            _native_error_box(msg)
        return 1

    from . import APP_NAME
    from .ui import theme
    from .ui.controller import ProjectController
    from .ui.main_window import MainWindow

    if log is not None:
        install_error_dialog(log)
    _set_windows_taskbar_identity()
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("PUCPR Racing")
    app.setDesktopFileName("vdstudio")
    if ICON_PNG.is_file():
        app.setWindowIcon(QtGui.QIcon(str(ICON_PNG)))
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
