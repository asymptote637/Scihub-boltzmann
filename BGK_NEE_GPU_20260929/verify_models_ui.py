"""Exercise model selection through the native dashboard and real workers."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from datetime import datetime
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
from PySide6 import QtCore, QtTest, QtWidgets as W
from dashboard.app import Dashboard, configure_fonts
from dashboard.store import ACTIVE, DEFAULTS, ROOT, Store, atomic_json, receipt_status
from dashboard import analysis


def wait_for(predicate, seconds=90):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        W.QApplication.processEvents()
        result = predicate()
        if result:
            return result
        QtTest.QTest.qWait(80)
    raise AssertionError("Timed out waiting for model UI/worker")


def main():
    assert all(receipt_status().values())
    folder = ROOT / "validation" / ("models_ui_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    store = Store(folder / "workspace")
    app = W.QApplication([])
    app.setStyle("Fusion")
    configure_fonts()
    window = Dashboard(store)
    errors = []
    window.error = errors.append
    window.show()
    window.timer.setInterval(400)
    report = dict(passed=False, workspace=str(store.workspace), cases=[], screenshots=[])
    receipt = ROOT / "validation/models_ui.json"
    try:
        for backend, collision, boundary in product(("cpu", "array", "fused"), ("BGK", "MRT"), ("nee", "halfway")):
            window.apply_params(dict(DEFAULTS, backend=backend, grid=16, re=100, max_iter=101,
                                     report_interval=50, mrt_s_e=1.31, mrt_s_eps=1.47, mrt_s_q=1.81))
            # Change the two independent selectors; preview must reflect geometry.
            window.collision.setCurrentIndex(window.collision.findData(collision))
            window.boundary.setCurrentIndex(window.boundary.findData(boundary))
            assert window.inputs["mrt_s_e"].isEnabled() == (collision == "MRT")
            assert f"L = {16 if boundary == 'halfway' else 15}" in window.derived.text()
            name = f"UI {backend} {collision} {boundary}"
            window.name_input.setText(name)
            assert window.start_button.isEnabled(), window.derived.text()
            QtTest.QTest.mouseClick(window.start_button, QtCore.Qt.LeftButton)
            case = wait_for(lambda: next((c for c in store.cases() if c["name"] == name), None))
            case = wait_for(lambda: (c if (c := store.get(case["id"]))["status"] not in (*ACTIVE, "queued") else None))
            assert case["status"] == "max_iter", case
            assert case["metrics"]["iteration"] == 101
            cfg = json.loads((Path(case["data_dir"]) / "config.json").read_text(encoding="utf-8"))
            assert cfg["collision_model"] == collision
            assert cfg["mrt_s_e"] == 1.31
            assert case["params"]["boundary"] == boundary
            fields = analysis.load_fields(case["data_dir"])
            assert fields["x"][0] == (0.5/16 if boundary == "halfway" else 0)
            summary = json.loads((Path(case["data_dir"]) / "run_summary.json").read_text(encoding="utf-8"))
            assert summary["collision"] == collision and summary["boundary"] == boundary
            report["cases"].append(dict(id=case["id"], backend=backend, collision=collision, boundary=boundary,
                                        status=case["status"], iteration=101))
            print(name + " PASS", flush=True)
        assert not errors, errors
        expected = window.form_params()
        store.save_preset("MRT-HBB 验收", expected)
        window.refresh_presets()
        window.apply_params(DEFAULTS)
        window.preset.setCurrentIndex(window.preset.findText("MRT-HBB 验收"))
        assert window.form_params() == expected
        # An invalid inactive MRT field must not block selecting BGK.
        window.inputs["mrt_s_e"].setText("invalid")
        assert not window.start_button.isEnabled()
        window.collision.setCurrentIndex(window.collision.findData("BGK"))
        assert window.start_button.isEnabled()
        window.apply_params(expected)
        for width, height in ((1440, 940), (960, 680)):
            window.resize(width, height)
            QtTest.QTest.qWait(400)
            path = folder / f"models_{width}x{height}.png"
            assert window.grab().save(str(path))
            report["screenshots"].append(str(path))
        window.resize(1440, 940)
        window.open_case(report["cases"][-1]["id"])
        wait_for(lambda: window.loaded_monitor and window.loaded_monitor["id"] == report["cases"][-1]["id"])
        window.monitor_tabs.setCurrentIndex(1)
        window.render_field()
        QtTest.QTest.qWait(500)
        assert "MRT / HBB" in window.field_plot.figure._suptitle.get_text()
        path = folder / "mrt_halfway_field.png"
        assert window.grab().save(str(path))
        report["screenshots"].append(str(path))
        window.close()
        W.QApplication.processEvents()
        reopened = Dashboard(Store(store.workspace))
        assert reopened.form_params() == expected
        reopened.close()
        W.QApplication.processEvents()
        # Compare final fields written by all three independent workers.
        for collision, boundary in product(("BGK", "MRT"), ("nee", "halfway")):
            matches = [r for r in report["cases"] if r["collision"] == collision and r["boundary"] == boundary]
            fields = [analysis.load_fields(store.get(r["id"])["data_dir"]) for r in matches]
            for other in fields[1:]:
                for key in ("rho", "ux", "uy"):
                    np.testing.assert_allclose(fields[0][key], other[key], rtol=0, atol=3e-12)
        report["passed"] = True
    finally:
        window.close()
        atomic_json(receipt, report)
    print(f"PASS: {receipt}", flush=True)


if __name__ == "__main__":
    main()
