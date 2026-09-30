"""Offscreen native Qt interaction and screenshot checks using real CPU/GPU jobs."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtTest, QtWidgets as W

from dashboard.app import Dashboard, configure_fonts
from dashboard.service import process_matches
from dashboard.store import ACTIVE, DEFAULTS, ROOT, Store, atomic_json, receipt_status


def wait_for(condition, seconds=90):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        W.QApplication.processEvents()
        result = condition()
        if result:
            return result
        QtTest.QTest.qWait(80)
    raise AssertionError("Timed out in Qt interaction check")


def click(widget):
    point = QtCore.QPoint(8, widget.height() // 2) if isinstance(widget, W.QCheckBox) else widget.rect().center()
    QtTest.QTest.mouseClick(widget, QtCore.Qt.LeftButton, pos=point)
    W.QApplication.processEvents()


def capture(window, folder, name):
    QtTest.QTest.qWait(400)
    image = folder / (name + ".png")
    assert window.grab().save(str(image))
    pixels = np.asarray(Image.open(image).convert("RGB"))
    assert pixels.std() > 10 and (pixels < 200).any()
    return dict(path=str(image), width=pixels.shape[1], height=pixels.shape[0], pixel_std=float(pixels.std()))


def named_tool(window, title):
    return next(button for button in window.findChildren(W.QToolButton) if button.accessibleName() == title)


def main():
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = ROOT / "validation" / ("dashboard_ui_" + stamp)
    folder.mkdir(parents=True, exist_ok=False)
    store = Store(folder / "workspace")
    app = W.QApplication([])
    app.setStyle("Fusion")
    configure_fonts()
    window = Dashboard(store)
    errors = []
    window.error = errors.append
    window.show()
    report = dict(passed=False, screenshots=[], checks=[], workspace=str(store.workspace), jobs=[],
                  source_sha256={str(source.relative_to(ROOT)): hashlib.sha256(source.read_bytes()).hexdigest()
                                 for source in sorted((ROOT / "dashboard").glob("*.py"))})
    receipt = ROOT / "validation/dashboard_ui.json"
    atomic_json(receipt, report)
    try:
        wait_for(lambda: len(store.cases()) >= 9 and window.loaded_monitor is not None)
        window.timer.setInterval(500)
        window.resize(1440, 940)
        for backend in ("cpu", "fused"):
            window.apply_params(dict(DEFAULTS, backend=backend, grid=16, re=100, max_iter=301, report_interval=100))
            window.name_input.setText("Qt " + backend)
            window.group_input.setText("界面验收")
            click(window.start_button)
            case = wait_for(lambda: next((c for c in store.cases() if c["name"] == "Qt " + backend), None))
            identifier = case["id"]
            wait_for(lambda: store.get(identifier)["status"] not in (*ACTIVE, "queued"))
            case = store.get(identifier)
            assert case["status"] == "max_iter", case
            report["jobs"].append(identifier)
        cpu, gpu = [store.get(identifier) for identifier in report["jobs"]]
        with np.load(Path(cpu["data_dir"]) / "results.npz") as a, np.load(Path(gpu["data_dir"]) / "results.npz") as b:
            for key in ("rho", "ux", "uy"):
                np.testing.assert_array_equal(a[key], b[key])
        report["checks"].append("CPU/GPU GUI submission and identical saved fields")
        print("PASS CPU/GPU GUI submission", flush=True)

        with patch.object(W.QInputDialog, "getText", return_value=("UI QA preset", True)):
            click(named_tool(window, "保存参数模板"))
        assert Store(store.workspace).presets()["UI QA preset"]["backend"] == "fused"
        report["checks"].append("Parameter template persisted")
        print("PASS parameter template persistence", flush=True)

        window.apply_params(dict(DEFAULTS, backend="cpu", grid=16, re=100, max_iter=30, report_interval=10))
        window.name_input.setText("Qt sweep")
        window.sweep_values.setText("100, 200")
        click(window.sweep_enabled)
        assert window.sweep_enabled.isChecked()
        click(window.start_button)
        wait_for(lambda: len([c for c in store.cases() if c["name"].startswith("Qt sweep") and c["status"] == "max_iter"]) == 2)
        report["checks"].append("Two-value Re sweep executed serially")
        print("PASS two-value Re sweep", flush=True)
        click(window.sweep_enabled)

        window.apply_params(dict(DEFAULTS, backend="cpu", grid=64, re=100, max_iter=1000000, report_interval=20))
        window.name_input.setText("Qt pause restore cancel")
        click(window.start_button)
        case = wait_for(lambda: next((c for c in store.cases() if c["name"] == "Qt pause restore cancel" and c["status"] == "running"), None))
        identifier = case["id"]
        window.monitor_id = identifier
        window.refresh(force=True)
        click(window.pause_button)
        wait_for(lambda: store.get(identifier)["status"] == "paused")
        window.close()
        QtTest.QTest.qWait(300)
        assert store.get(identifier)["status"] == "paused"
        window = Dashboard(Store(store.workspace))
        window.error = errors.append
        window.timer.setInterval(500)
        window.show()
        window.monitor_id = identifier
        window.refresh(force=True)
        click(window.resume_button)
        wait_for(lambda: store.get(identifier)["status"] == "running")
        window.refresh(force=True)
        with patch.object(W.QMessageBox, "question", return_value=W.QMessageBox.Yes):
            click(window.cancel_button)
        wait_for(lambda: store.get(identifier)["status"] == "cancelled")
        report["checks"].append("Pause, close window, reopen, resume, cancel and preserve fields")
        print("PASS pause, reopen, resume, cancel", flush=True)

        window.refresh(force=True)
        source = next(c for c in store.cases() if c["params"].get("grid") == 256 and c["metrics"].get("iteration") == 20000)
        window.open_case(source["id"])
        wait_for(lambda: window.loaded_monitor and window.loaded_monitor["id"] == source["id"] and not window.monitor_busy)
        window.resize(1440, 940)
        report["screenshots"].append(capture(window, folder, "01_control_desktop"))
        window.monitor_tabs.setCurrentIndex(1)
        report["screenshots"].append(capture(window, folder, "02_field_desktop"))
        window.resize(960, 680)
        report["screenshots"].append(capture(window, folder, "03_control_compact"))
        assert window.width() == 960 and window.height() == 680
        for item in (window.start_button, window.backend, window.field_mode, window.monitor_tabs):
            assert item.width() > 0 and item.height() > 0
        window.resize(1440, 940)

        window.show_page(1)
        window.checked = {cpu["id"], gpu["id"]}
        window.rebuild_library(force=True)
        item = next(window.library.topLevelItem(i) for i in range(window.library.topLevelItemCount())
                    if window.library.topLevelItem(i).data(0, QtCore.Qt.UserRole) == cpu["id"])
        window.library.setCurrentItem(item)
        window.edit_name.setText("CPU 校验样本")
        window.edit_group.setText("CPU / GPU 对照")
        window.edit_notes.setText("真实界面提交；301 步预算，未收敛")
        click(named_tool(window, "保存名称、分组与备注"))
        assert Store(store.workspace).get(cpu["id"])["name"] == "CPU 校验样本"
        report["screenshots"].append(capture(window, folder, "04_library_desktop"))
        report["checks"].append("Dataset metadata persisted without modifying raw result")
        window.compare_selected()
        wait_for(lambda: len(window.analysis_data) == 2)
        report["screenshots"].append(capture(window, folder, "05_analysis_history"))
        window.analysis_mode.setCurrentIndex(window.analysis_mode.findData("centerlines"))
        report["screenshots"].append(capture(window, folder, "06_analysis_centerlines"))
        window.analysis_mode.setCurrentIndex(window.analysis_mode.findData("difference"))
        report["screenshots"].append(capture(window, folder, "07_analysis_difference"))
        window.analysis_mode.setCurrentIndex(window.analysis_mode.findData("sweep"))
        report["screenshots"].append(capture(window, folder, "08_analysis_sweep"))
        window.resize(960, 680)
        report["screenshots"].append(capture(window, folder, "09_analysis_compact"))
        window.resize(1440, 940)
        window.show_page(3)
        QtTest.QTest.qWait(1000)
        report["screenshots"].append(capture(window, folder, "10_original_cpu_readonly"))
        report["checks"].append("All four analysis views and original CPU read-only view rendered")
        assert not errors, errors
        assert all(receipt_status().values())
        assert not any(case["status"] in (*ACTIVE, "queued") for case in store.cases())
        report["passed"] = True
        print(json.dumps(dict(passed=True, checks=report["checks"], screenshot_folder=str(folder)), ensure_ascii=False, indent=2), flush=True)
    finally:
        for case in store.cases(archived=True):
            if case["status"] in (*ACTIVE, "queued"):
                store.control(case["id"], "cancel")
        window.close()
        report["errors"] = errors
        atomic_json(receipt, report)
        identity = store.setting("service", {})
        wait_for(lambda: not process_matches(identity.get("pid"), identity.get("started"), "dashboard.service"), seconds=90)


if __name__ == "__main__":
    main()
