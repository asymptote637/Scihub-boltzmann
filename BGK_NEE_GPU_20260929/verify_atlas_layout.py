"""Render the Atlas desktop skin against real saved results, without new jobs."""
import datetime as dt
import hashlib
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from PySide6 import QtCore, QtSvg, QtTest, QtWidgets as W

from dashboard.app import Dashboard, ICONS, configure_fonts
from dashboard.store import ACTIVE, ROOT, Store, atomic_json, read_json, receipt_status
from verify_dashboard_ui import capture, click, wait_for
from verify_dashboard_delivery import original_audit


def in_window(window, widget):
    assert widget.isVisible(), widget.objectName()
    rectangle = QtCore.QRect(widget.mapTo(window, QtCore.QPoint(0, 0)), widget.size())
    assert window.rect().contains(rectangle), (widget.objectName(), rectangle, window.rect())


def check_plot(plot):
    plot.canvas.draw()
    pixels = np.asarray(plot.canvas.buffer_rgba())[:, :, :3]
    assert pixels.std() > 5
    renderer = plot.canvas.get_renderer()
    bounds = plot.figure.bbox
    for axis in plot.figure.axes:
        if axis.get_title():
            title = axis.title.get_window_extent(renderer)
            assert title.x0 >= bounds.x0 - 2 and title.x1 <= bounds.x1 + 2, (title, bounds)
            assert title.y0 >= bounds.y0 - 2 and title.y1 <= bounds.y1 + 2, (title, bounds)


def main():
    folder = ROOT / "validation" / ("atlas_layout_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True)
    store = Store(folder / "workspace")
    reference = read_json(ROOT / "validation/dashboard_delivery.json", {})
    identifiers = []
    for case in reference["cases"]:
        identifier = store.register_result(case["data_dir"], group="工作台验收")
        store.update_case(identifier, name=case["name"])
        identifiers.append(identifier)
    app = W.QApplication([])
    app.setStyle("Fusion")
    configure_fonts()
    for name in set(ICONS.values()):
        assert QtSvg.QSvgRenderer(str(ROOT / "dashboard/icons" / (name + ".svg"))).isValid(), name
    window = Dashboard(store)
    errors = []
    window.error = errors.append
    window.show()
    report = dict(passed=False, source_sha256={}, screenshots=[], checks=[], errors=errors,
                  original_audit=original_audit())
    try:
        wait_for(lambda: len(store.cases()) >= 11 and not window.tasks)
        window.open_case(identifiers[-1])
        wait_for(lambda: window.loaded_monitor and window.loaded_monitor["id"] == identifiers[-1] and not window.tasks)
        window.checked = set(identifiers)
        window.show_page(2)
        wait_for(lambda: len(window.analysis_data) == 2 and not window.tasks)
        window.analysis_mode.setCurrentIndex(window.analysis_mode.findData("centerlines"))
        for width, height in ((1440, 940), (1200, 800), (960, 680)):
            window.resize(width, height)
            QtTest.QTest.qWait(300)
            assert (window.width(), window.height()) == (width, height)
            assert window.overview.isVisible() == (height >= 820)
            for page in range(4):
                click(window.navigation.button(page))
                wait_for(lambda: not window.tasks)
                assert window.pages.currentIndex() == page
                assert window.navigation.button(page).isChecked()
                in_window(window, window.title)
                in_window(window, window.validation_label)
                if page == 0:
                    in_window(window, window.start_button)
                    in_window(window, window.derived)
                    window.monitor_tabs.setCurrentIndex(1)
                    QtTest.QTest.qWait(300)
                    check_plot(window.field_plot)
                if page == 2:
                    check_plot(window.comparison_plot)
                report["screenshots"].append(capture(window, folder, f"{width}_{height}_page_{page}"))
            report["checks"].append(f"{width}x{height}: four pages, navigation, visible actions, plots")
        window.resize(1440, 940)
        window.show_page(0)
        window.monitor_tabs.setCurrentIndex(0)
        report["screenshots"].append(capture(window, folder, "control_diagnostics"))
        assert all(receipt_status().values())
        assert not any(case["status"] in (*ACTIVE, "queued") for case in store.cases())
        assert not errors, errors
        report["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in (ROOT / "dashboard").rglob("*")
                                   if path.is_file() and path.suffix in (".py", ".svg")}
        report["passed"] = True
        print(f"PASS Atlas layout: {folder}", flush=True)
    finally:
        window.close()
        atomic_json(ROOT / "validation/atlas_layout.json", report)


if __name__ == "__main__":
    main()
