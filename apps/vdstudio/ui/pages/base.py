"""Common page behaviour: lazy refresh when visible, header, controller wiring."""

from __future__ import annotations

from PySide6 import QtWidgets

from ...engine.geometry import CarGeometry
from ...engine.kpis import KpiSet
from ..controller import ProjectController
from ..widgets import page_header, scroll


class _PageLayout(QtWidgets.QVBoxLayout):
    """The page body's layout, as seen by its scroll area.

    A word-wrapped label makes a layout height-for-width, and a resizable
    scroll area then sizes the body to its *preferred* height, so a page would
    scroll although it fits. Hiding that from the scroll area sizes the body to
    the viewport, down to the body's minimum size; inside, the layout still
    gives every wrapped label the height its width needs.
    """

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return False


class Page(QtWidgets.QWidget):
    title = ""
    subtitle = ""

    def __init__(self, ctrl: ProjectController) -> None:
        super().__init__()
        self.ctrl = ctrl
        self._need_geo = True
        self._need_kpi = True
        self._need_project = True
        # Pages lay out into ``self.root`` inside a scroll area: a page whose content
        # needs more room than the window has scrolls instead of forcing the window
        # larger than the screen (every page must be usable at 1366x768).
        body = QtWidgets.QWidget()
        body.setObjectName("pageBody")
        self.root = _PageLayout(body)
        self.root.setContentsMargins(18, 14, 18, 14)
        self.root.setSpacing(8)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll(body))
        if self.title:
            self.root.addWidget(page_header(self.title, self.subtitle))
        ctrl.geometryChanged.connect(self._sig_geometry)
        ctrl.geometryFailed.connect(self._sig_geometry_failed)
        ctrl.staticReady.connect(self._sig_kpis)
        ctrl.dynamicReady.connect(self._sig_kpis)
        ctrl.projectReplaced.connect(self._sig_replaced)
        ctrl.projectChanged.connect(self._sig_project)

    # -- dispatch (only work while visible; catch up on show) ----------------------
    def _sig_geometry(self, _geo: CarGeometry) -> None:
        self._need_geo = True
        if self.isVisible():
            self._page_flush()

    def _sig_geometry_failed(self, msg: str) -> None:
        if self.isVisible():
            self.on_geometry_failed(msg)

    def _sig_kpis(self, _ks: KpiSet) -> None:
        self._need_kpi = True
        if self.isVisible():
            self._page_flush()

    def _sig_replaced(self) -> None:
        self.on_project_replaced()

    def _sig_project(self) -> None:
        self._need_project = True
        if self.isVisible():
            self._page_flush()

    def showEvent(self, e) -> None:  # noqa: N802
        super().showEvent(e)
        self._page_flush()

    def _page_flush(self) -> None:
        if self._need_project:
            self._need_project = False
            self.on_project()
        if self._need_geo and self.ctrl.geometry is not None:
            self._need_geo = False
            self.on_geometry(self.ctrl.geometry)
        elif self.ctrl.geometry is None and self.ctrl.geometry_error:
            self.on_geometry_failed(self.ctrl.geometry_error)
        if self._need_kpi:
            self._need_kpi = False
            self.on_kpis(self.ctrl.kpis)

    # -- overridables ------------------------------------------------------------------
    def on_project(self) -> None:
        pass

    def on_project_replaced(self) -> None:
        pass

    def on_geometry(self, geo: CarGeometry) -> None:
        pass

    def on_geometry_failed(self, msg: str) -> None:
        pass

    def on_kpis(self, ks: KpiSet) -> None:
        pass
