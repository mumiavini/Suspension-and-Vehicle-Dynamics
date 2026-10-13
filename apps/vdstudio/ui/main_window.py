"""Main window: navigation by FSAE stage, file handling, undo/redo, computation status."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .. import APP_NAME, __version__
from ..engine import recent as R
from ..engine.project import PROJECT_SUFFIX
from . import theme
from .controller import ProjectController

# (section, key, label, module, class)
NAV: tuple[tuple[str, str, str, str, str], ...] = (
    ("Overview", "dashboard", "Dashboard", "dashboard", "DashboardPage"),
    ("Design", "concept", "Concept & targets", "concept", "ConceptPage"),
    ("Design", "geometry", "Suspension geometry", "geometry", "GeometryPage"),
    ("Design", "hardpoints", "Hardpoints", "hardpoints", "HardpointsPage"),
    ("Design", "steering", "Steering", "steering", "SteeringPage"),
    ("Simulation & analysis", "kinematics", "Kinematic sweeps", "kinematics", "KinematicsPage"),
    ("Simulation & analysis", "motion", "3D motion", "motion", "MotionPage"),
    ("Simulation & analysis", "tradeoff", "Trade-offs (what-if)", "tradeoff", "TradeoffPage"),
    ("Simulation & analysis", "maneuver", "Quasi-static maneuver", "maneuver", "ManeuverPage"),
    ("Validation", "telemetry", "Telemetry", "telemetry", "TelemetryPage"),
    ("Validation", "integrations", "CAD & cross-checks", "integrations", "IntegrationsPage"),
    ("Documentation", "reports", "Results & reports", "reports", "ReportsPage"),
    ("Documentation", "log", "Design log", "log", "LogPage"),
)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, ctrl: ProjectController,
                 settings: QtCore.QSettings | None = None) -> None:
        super().__init__()
        self.ctrl = ctrl
        self.settings = settings or QtCore.QSettings("PUCPR Racing", APP_NAME)
        # 1500x920 where the screen allows; pages scroll, so the window also fits 1366x768
        screen = QtGui.QGuiApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else QtCore.QRect(0, 0, 1500, 920)
        self.resize(min(1500, avail.width()), min(920, avail.height()))
        self.pages: dict[str, QtWidgets.QWidget] = {}
        self._build()
        ctrl.dirtyChanged.connect(lambda _d: self._title())
        ctrl.projectChanged.connect(self._title)
        ctrl.projectChanged.connect(self._update_actions)
        ctrl.message.connect(lambda m: self.statusBar().showMessage(m, 8000))
        ctrl.dynamicBusy.connect(self._busy)
        ctrl.dynamicProgress.connect(self._progress)
        ctrl.dynamicStale.connect(self._calc_status)
        ctrl.dynamicFailed.connect(lambda _m: self._calc_status())
        ctrl.geometryFailed.connect(self._geo_failed)
        ctrl.geometryChanged.connect(self._geo_ok)
        self._title()
        self._update_actions()
        ctrl.recompute()

    # -- construction -------------------------------------------------------------------
    def _build(self) -> None:
        import importlib

        # A plain row, not a splitter: the nav has a fixed width, and a splitter laid out at
        # a small window size keeps a dead band beside it once the window grows.
        central = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(central)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.nav = QtWidgets.QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(230)
        self.stack = QtWidgets.QStackedWidget()
        row.addWidget(self.nav)
        row.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        brand = QtWidgets.QListWidgetItem(f"{APP_NAME}")
        f = brand.font()
        f.setPointSize(14)
        f.setBold(True)
        brand.setFont(f)
        brand.setFlags(QtCore.Qt.NoItemFlags)
        self.nav.addItem(brand)
        section = None
        self._row_key: dict[int, str] = {}
        for sec, key, label, mod, cls in NAV:
            if sec != section:
                section = sec
                it = QtWidgets.QListWidgetItem(sec.upper())
                it.setFlags(QtCore.Qt.NoItemFlags)
                fnt = it.font()
                fnt.setPointSize(8)
                fnt.setBold(True)
                it.setFont(fnt)
                it.setForeground(QtGui.QColor(theme.current().text_muted))
                self.nav.addItem(it)
            module = importlib.import_module(f"apps.vdstudio.ui.pages.{mod}")
            page = getattr(module, cls)(self.ctrl)
            page.main_window = self  # type: ignore[attr-defined]
            self.pages[key] = page
            self.stack.addWidget(page)
            item = QtWidgets.QListWidgetItem("   " + label)
            self.nav.addItem(item)
            self._row_key[self.nav.row(item)] = key
        self.nav.currentRowChanged.connect(self._nav_changed)

        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.act_new = tb.addAction("New", self.new_project)
        self.act_open = tb.addAction("Open…", self.open_project)
        self.recent_menu = QtWidgets.QMenu(self)
        self.recent_button = QtWidgets.QToolButton()
        self.recent_button.setText("Recent")
        self.recent_button.setToolTip("Recently opened or saved projects")
        self.recent_button.setMenu(self.recent_menu)
        self.recent_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        tb.addWidget(self.recent_button)
        self._rebuild_recent()
        self.act_save = tb.addAction("Save", self.save_project)
        self.act_saveas = tb.addAction("Save as…", lambda: self.save_project(True))
        tb.addSeparator()
        self.act_undo = tb.addAction("Undo", self.ctrl.undo)
        self.act_redo = tb.addAction("Redo", self.ctrl.redo)
        tb.addSeparator()
        tb.addAction("Recompute", self.ctrl.recompute)
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.mode_label = QtWidgets.QLabel()
        tb.addWidget(self.mode_label)
        tb.addSeparator()
        tb.addAction("Guide", self.guide)
        tb.addAction("Theme", self.toggle_theme)
        tb.addAction("About", self.about)
        self.act_new.setShortcut(QtGui.QKeySequence.New)
        self.act_open.setShortcut(QtGui.QKeySequence.Open)
        self.act_save.setShortcut(QtGui.QKeySequence.Save)
        self.act_undo.setShortcut(QtGui.QKeySequence.Undo)
        self.act_redo.setShortcut(QtGui.QKeySequence.Redo)

        sb = self.statusBar()
        self.geo_status = QtWidgets.QLabel()
        self.calc_label = QtWidgets.QLabel("")
        self.calc_bar = QtWidgets.QProgressBar()
        self.calc_bar.setFixedWidth(160)
        self.calc_bar.setRange(0, 100)
        self.calc_bar.setVisible(False)
        sb.addPermanentWidget(self.geo_status)
        sb.addPermanentWidget(self.calc_label)
        sb.addPermanentWidget(self.calc_bar)
        self.nav.setCurrentRow(2)

    def _nav_changed(self, row: int) -> None:
        key = self._row_key.get(row)
        if key is not None:
            self.stack.setCurrentWidget(self.pages[key])

    def goto(self, key: str) -> None:
        for row, k in self._row_key.items():
            if k == key:
                self.nav.setCurrentRow(row)
                return

    # -- status --------------------------------------------------------------------------
    def _title(self) -> None:
        name = self.ctrl.path.name if self.ctrl.path else "untitled"
        star = " *" if self.ctrl.dirty else ""
        self.setWindowTitle(f"{name}{star} — {self.ctrl.project.name} — {APP_NAME}")
        mode = self.ctrl.project.geometry_mode
        label = "synthesis (design intent)" if mode == "synthesis" else "hardpoints"
        self.mode_label.setText(f"  Geometry source: <b>{label}</b>  ")

    def _update_actions(self) -> None:
        self.act_undo.setEnabled(self.ctrl.can_undo())
        self.act_redo.setEnabled(self.ctrl.can_redo())

    def _busy(self, busy: bool) -> None:
        if busy:
            self.calc_bar.setValue(0)
        self.calc_bar.setVisible(busy)
        self._calc_status()

    def _calc_status(self) -> None:
        """Swept-KPI state, from the controller: never 'up to date' unless it is."""
        c = self.ctrl
        tip = ""
        if c.geometry is None:
            text = "swept KPIs: none (geometry failed)"
        elif c.dynamic_busy:
            text = "computing swept KPIs… (stale)"
        elif c.dynamic_error:
            text = f"<span style='color:{theme.current().bad}'><b>swept KPIs FAILED</b></span>"
            tip = c.dynamic_error
        elif c.dynamic_stale:
            text = "swept KPIs: recomputing… (stale)"
        else:
            text = "swept KPIs up to date"
        self.calc_label.setText(text)
        self.calc_label.setToolTip(tip)

    def _progress(self, f: float, msg: str) -> None:
        self.calc_bar.setValue(int(100 * f))
        self.calc_label.setText(f"computing swept KPIs: {msg} (stale)")

    def _geo_failed(self, msg: str) -> None:
        self.geo_status.setText(f"<span style='color:{theme.current().bad}'>geometry: "
                                f"{msg[:90]}</span>")
        self.geo_status.setToolTip(msg)
        self._calc_status()

    def _geo_ok(self, geo) -> None:
        n = geo.estimate_count()
        warn = f" · {len(geo.warnings)} warning(s)" if geo.warnings else ""
        self.geo_status.setText(f"geometry OK · {n} estimate-tagged input(s){warn}")
        self.geo_status.setToolTip("\n".join(geo.warnings))

    # -- files -------------------------------------------------------------------------
    def _confirm_discard(self) -> bool:
        """True when it is fine to replace or close the project (saved, discarded or clean)."""
        if not self.ctrl.dirty:
            return True
        name = self.ctrl.path.name if self.ctrl.path else "the untitled project"
        r = QtWidgets.QMessageBox.question(
            self, APP_NAME, f"Save the changes to {name}?",
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard
            | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Save)
        if r == QtWidgets.QMessageBox.Save:
            return self.save_project()
        return r == QtWidgets.QMessageBox.Discard

    def new_project(self) -> None:
        if self._confirm_discard():
            self.ctrl.new()

    def open_project(self, path: str | Path | None = None) -> bool:
        if not self._confirm_discard():
            return False
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open project", str(R.dialog_dir(self.ctrl.path)),
                f"VD Studio project (*{PROJECT_SUFFIX} *.json)")
        if not path:
            return False
        path = Path(path)
        if not path.is_file():
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"{path} no longer exists.")
            self._set_recent(R.drop_recent(self._recent(), path))
            return False
        try:
            self.ctrl.open(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QtWidgets.QMessageBox.critical(self, APP_NAME, f"Could not open {path}:\n{exc}")
            return False
        self._remember(path)
        return True

    def open_startup_project(self, argv_path: str | None = None) -> Path | None:
        """Open the file the app starts with (see engine/recent.py); None = built-in car."""
        last = self.settings.value("last_project", "")
        path, why = R.startup_path(argv_path, str(last) if last else None)
        if path is not None:
            if self.open_project(path):
                self.statusBar().showMessage(f"Opened {path} ({why})", 10000)
                return path
            if why == "last project":       # unreadable: do not fail on every start
                self.settings.remove("last_project")
        self.statusBar().showMessage(
            "Built-in 2027 car from sla_geometry.py / steering_geometry.py", 10000)
        return None

    def save_project(self, save_as: bool = False) -> bool:
        path = self.ctrl.path
        if save_as or path is None:
            default = (self.ctrl.project.name.split("--")[0].strip().replace(" ", "_")
                       or "project") + PROJECT_SUFFIX
            p, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save project", str(R.dialog_dir(self.ctrl.path) / default),
                f"VD Studio project (*{PROJECT_SUFFIX})")
            if not p:
                return False
            if not p.endswith(PROJECT_SUFFIX):
                p += PROJECT_SUFFIX
            path = Path(p)
        try:
            self.ctrl.save(path)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, APP_NAME, f"Could not save:\n{exc}")
            return False
        self._remember(path)
        self.statusBar().showMessage(f"Saved {path}", 5000)
        return True

    # -- recent projects -----------------------------------------------------------------
    def _recent(self) -> list[str]:
        return R.as_list(self.settings.value("recent_projects", []))

    def _set_recent(self, paths: list[str]) -> None:
        self.settings.setValue("recent_projects", paths)
        self._rebuild_recent()

    def _remember(self, path: Path) -> None:
        """The file opened or saved last is the one the next start opens."""
        self.settings.setValue("last_project", str(path.resolve()))
        self._set_recent(R.push_recent(self._recent(), path.resolve()))

    def _rebuild_recent(self) -> None:
        m = self.recent_menu
        m.clear()
        paths = self._recent()
        for i, p in enumerate(paths, 1):
            act = m.addAction(f"&{i}  {Path(p).name}", lambda p=p: self.open_project(p))
            act.setToolTip(p)
            act.setStatusTip(p)
        if not paths:
            m.addAction("No recent projects").setEnabled(False)
        m.addSeparator()
        team = m.addAction(f"Team project ({R.TEAM_PROJECT.name})",
                           lambda: self.open_project(R.TEAM_PROJECT))
        team.setEnabled(R.TEAM_PROJECT.is_file())
        team.setStatusTip(str(R.TEAM_PROJECT))
        clear = m.addAction("Clear list", lambda: self._set_recent([]))
        clear.setEnabled(bool(paths))
        m.setToolTipsVisible(True)

    def toggle_theme(self) -> None:
        app = QtWidgets.QApplication.instance()
        new = theme.LIGHT if theme.current().name == "dark" else theme.DARK
        theme.apply(app, new)
        self.settings.setValue("theme", new.name)
        from .widgets import restyle_plots

        restyle_plots(self)
        self.ctrl.recompute()

    def guide(self) -> None:
        from .._paths import REPO

        path = REPO / "docs" / "VD_STUDIO.md"
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"{APP_NAME} — guide")
        dlg.resize(980, 820)
        lay = QtWidgets.QVBoxLayout(dlg)
        tb = QtWidgets.QTextBrowser()
        try:
            tb.setMarkdown(path.read_text(encoding="utf-8"))
        except OSError as exc:
            tb.setPlainText(f"Guide not found ({exc}).")
        lay.addWidget(tb)
        dlg.exec()

    def about(self) -> None:
        QtWidgets.QMessageBox.about(
            self, APP_NAME,
            f"<h3>{APP_NAME} {__version__}</h3>"
            "<p>PUCPR Racing #27 — suspension &amp; steering workbench for the FSAE car.</p>"
            "<p>Every geometry number comes from <code>vdcore.geometry.solver.DWSolver</code>, "
            "the solver validated against OptimumK and Altair MotionSolve. Design synthesis "
            "runs the repository's <code>sla_geometry.py</code> and "
            "<code>steering_geometry.py</code>.</p>"
            "<p><b>The tool computes trade-offs; the designer chooses the geometry.</b> "
            "Bands are design-intent windows, not verdicts, and no value is ever "
            "recommended.</p>"
            "<p>Frame: ISO 8855 — X forward, Y LEFT, Z up. Units: mm, N, deg.</p>")

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:  # noqa: N802
        if self._confirm_discard():
            for p in self.pages.values():
                closer = getattr(p, "shutdown", None)
                if callable(closer):
                    closer()
            self.ctrl.shutdown()
            e.accept()
        else:
            e.ignore()
