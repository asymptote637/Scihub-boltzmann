"""Open the native CPU/GPU simulation workbench."""
import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

from PySide6 import QtCore, QtWidgets
from dashboard.app import Dashboard, configure_fonts
from dashboard.store import DEFAULT_WORKSPACE, Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    application = QtWidgets.QApplication(sys.argv[:1])
    application.setApplicationName("LBM CPU GPU Workbench")
    application.setStyle("Fusion")
    configure_fonts()
    window = Dashboard(Store(args.workspace))
    available = application.primaryScreen().availableGeometry()
    window.resize(min(1440, max(960, available.width() - 48)), min(940, max(680, available.height() - 64)))
    window.show()
    if args.screenshot:
        def capture():
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(args.screenshot))
            args.screenshot.with_suffix(".json").write_text(json.dumps(dict(
                pid=os.getpid(), title=window.windowTitle(), width=window.width(), height=window.height(),
                visible=window.isVisible(), workspace=str(args.workspace.resolve())), ensure_ascii=False), encoding="utf-8")
        QtCore.QTimer.singleShot(6000, capture)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
