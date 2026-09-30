"""Verify saved-field views in an isolated catalog; never submit solver jobs."""
import datetime as dt
import hashlib
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from PySide6 import QtTest, QtWidgets as W

from dashboard.app import Dashboard, Plot, configure_fonts
from dashboard.store import ACTIVE, ROOT, Store, atomic_json, receipt_status
from verify_atlas_layout import check_plot, in_window
from verify_dashboard_delivery import original_audit
from verify_dashboard_ui import capture, click, wait_for


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder = ROOT / "validation" / ("streamline_ui_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True)
    sources = [ROOT / "workspace/runs" / key for key in
               ("11d23ff24df14bcdaad3a9aa21f99687", "46381ebe347a45ed91803babfd63f882")]
    before = {str(path): sha(path) for source in sources for path in source.iterdir() if path.is_file()}
    store = Store(folder / "workspace")
    identifiers = [store.register_result(source, group="流线只读验收") for source in sources]
    app = W.QApplication([])
    app.setStyle("Fusion")
    configure_fonts()
    window = Dashboard(store)
    window.show()
    errors = []
    window.error = errors.append
    report = dict(passed=False, folder=str(folder), screenshots=[], errors=errors, checks=[])
    try:
        wait_for(lambda: not window.tasks)
        window.open_case(identifiers[-1])
        wait_for(lambda: window.loaded_monitor and window.loaded_monitor["id"] == identifiers[-1] and not window.tasks)
        window.timer.stop()
        window.monitor_tabs.setCurrentIndex(1)
        for width, height in ((1440, 940), (1200, 800), (960, 680)):
            window.resize(width, height)
            QtTest.QTest.qWait(200)
            for mode in ("speed", "ux", "uy", "rho", "streamlines", "overlay", "vortices"):
                window.field_mode.setCurrentIndex(window.field_mode.findData(mode))
                window.render_field()
                QtTest.QTest.qWait(100)
                check_plot(window.field_plot)
                for widget in (window.field_mode, window.field_region, window.field_density,
                               window.field_extent, window.expand_field_button, window.start_button):
                    in_window(window, widget)
                figure = window.field_plot.figure
                title = figure._suptitle.get_window_extent(figure.canvas.get_renderer())
                assert figure.bbox.contains(title.x0, title.y0) and figure.bbox.contains(title.x1, title.y1)
                if mode in ("streamlines", "overlay", "vortices"):
                    report["screenshots"].append(capture(window, folder, f"{width}_{height}_{mode}"))
            report["checks"].append(f"{width}x{height}: seven modes, controls visible, nonblank canvases, titles contained")
        window.resize(1440, 940)
        window.field_mode.setCurrentIndex(window.field_mode.findData("streamlines"))
        for region in ("full", "bottom_left", "bottom_right", "top_left", "top_right"):
            window.field_region.setCurrentIndex(window.field_region.findData(region))
            window.field_density.setValue(1.7)
            window.field_extent.setValue(30)
            window.render_field()
            check_plot(window.field_plot)
        assert window.field_extent.isEnabled()
        assert len(window.field_plot.figure.axes[0].collections[0].get_segments()) > 0
        report["checks"].append("All corners, density and extent controls")
        window.field_mode.setCurrentIndex(window.field_mode.findData("vortices"))
        window.field_extent.setValue(35)
        window.render_field()
        assert not window.field_region.isEnabled()
        click(window.expand_field_button)
        assert len(window.field_windows) == 1
        dialog = window.field_windows[0]
        plot = dialog.findChild(Plot)
        check_plot(plot)
        report["screenshots"].append(capture(dialog, folder, "vortices_detached"))
        plot.figure.savefig(folder / "vortices_re9569.png", dpi=160)
        assert (folder / "vortices_re9569.png").stat().st_size > 10000
        dialog.close()
        wait_for(lambda: not window.field_windows)
        report["checks"].append("Detached immutable snapshot, full figure export, dialog cleanup")
        case = dict(window.loaded_monitor)
        fields = dict(case["fields"])
        fields.update(preview=True, preview_iteration=321)
        case.update(fields=fields, status="running")
        window.loaded_monitor = case
        window.field_mode.setCurrentIndex(window.field_mode.findData("streamlines"))
        window.render_field()
        assert "321" in window.field_plot.figure._suptitle.get_text()
        assert "采样预览" in window.field_plot.figure._suptitle.get_text()
        window.monitor_id = "missing"
        window.render_field()
        assert len(window.field_plot.figure.axes) == 1
        assert "尚无当前" in window.field_plot.figure.axes[0].texts[0].get_text()
        report["checks"].append("Preview iteration label and no stale field on selection change")
        window.open_case(identifiers[0])
        wait_for(lambda: window.loaded_monitor["id"] == identifiers[0] and not window.tasks)
        window.render_field()
        assert "已收敛" in window.field_plot.figure._suptitle.get_text()
        assert not errors
        assert not any(c["status"] in (*ACTIVE, "queued") for c in store.cases())
        assert all(sha(Path(path)) == digest for path, digest in before.items())
        report.update(passed=True, original_audit=original_audit(), backend_gates=receipt_status(),
                      saved_result_files_unchanged=len(before),
                      source_sha256={str(path.relative_to(ROOT)): sha(path) for path in (ROOT / "dashboard").glob("*.py")})
        assert all(report["backend_gates"].values())
        print(f"PASS streamline UI: {folder}", flush=True)
    finally:
        window.close()
        atomic_json(ROOT / "validation/streamline_ui.json", report)


if __name__ == "__main__":
    main()
