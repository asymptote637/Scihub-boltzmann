"""PySide6 desktop UI for the customized 2D LBM simulator.

This UI does not open a browser and does not bind a web port.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

Path("logs/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("logs") / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import (
    CORE_BOUNDARY_TYPES,
    BoundaryConfig,
    ObstacleConfig,
    OutputConfig,
    SolverConfig,
    case_preset,
    derived_parameters,
    recommend_u_ref,
    validate_config,
)
from lbm_solver import LBMSolver, Report
from postprocess import save_results, speed, vorticity


class SimulationWorker(QObject):
    report = Signal(dict)
    preview = Signal(str, str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, cfg: SolverConfig):
        super().__init__()
        self.cfg = cfg
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            solver = LBMSolver(self.cfg)
            for report in solver.run():
                self.report.emit(report.as_dict())
                if report.iteration % max(1, self.cfg.output.plot_interval) == 0:
                    speed_path, vort_path = self._write_preview(solver, report)
                    self.preview.emit(str(speed_path), str(vort_path))
                if self._stop:
                    break
            run_dir = save_results(solver, self.cfg)
            self.finished.emit(str(run_dir))
        except Exception as exc:
            self.failed.emit(str(exc))

    def _write_preview(self, solver: LBMSolver, report: Report) -> tuple[Path, Path]:
        preview_dir = Path("logs/desktop_preview")
        preview_dir.mkdir(parents=True, exist_ok=True)
        fields = solver.fields()
        speed_path = preview_dir / "speed.png"
        vort_path = preview_dir / "vorticity.png"
        self._imshow(speed(fields["ux"], fields["uy"]), fields["solid_mask"], speed_path, "|u|")
        self._imshow(vorticity(fields["ux"], fields["uy"]), fields["solid_mask"], vort_path, "vorticity")
        return speed_path, vort_path

    @staticmethod
    def _imshow(values: np.ndarray, solid_mask: np.ndarray, path: Path, title: str) -> None:
        plt.figure(figsize=(5.2, 4.0))
        plt.imshow(np.ma.masked_where(solid_mask, values), origin="lower", cmap="viridis")
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()


class BoundaryPanel(QGroupBox):
    def __init__(self, side: str, default: BoundaryConfig):
        super().__init__(f"{side} boundary")
        self.side = side
        layout = QFormLayout(self)
        self.type_box = QComboBox()
        self.type_box.addItems(sorted(CORE_BOUNDARY_TYPES))
        self.type_box.setCurrentText(default.type if default.type in CORE_BOUNDARY_TYPES else "no_slip_bounce_back")
        self.ux = self._double(default.ux, -1.0, 1.0, 0.005, 6)
        self.uy = self._double(default.uy, -1.0, 1.0, 0.005, 6)
        self.rho = self._double(default.rho, 0.0001, 10.0, 0.01, 6)
        self.rb = self._double(default.rb, 0.0, 1.0, 0.05, 3)
        layout.addRow("type", self.type_box)
        layout.addRow("ux", self.ux)
        layout.addRow("uy", self.uy)
        layout.addRow("rho", self.rho)
        layout.addRow("rb", self.rb)

    @staticmethod
    def _double(value: float, minimum: float, maximum: float, step: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin

    def value(self) -> BoundaryConfig:
        return BoundaryConfig(
            type=self.type_box.currentText(),
            ux=self.ux.value(),
            uy=self.uy.value(),
            rho=self.rho.value(),
            rb=self.rb.value(),
        )

    def set_boundary(self, bc: BoundaryConfig) -> None:
        self.type_box.setCurrentText(bc.type if bc.type in CORE_BOUNDARY_TYPES else "no_slip_bounce_back")
        self.ux.setValue(bc.ux)
        self.uy.setValue(bc.uy)
        self.rho.setValue(bc.rho)
        self.rb.setValue(bc.rb)


class DesktopWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D LBM Desktop Simulator")
        self.resize(1280, 820)
        self.thread: QThread | None = None
        self.worker: SimulationWorker | None = None
        self.last_run_dir: str | None = None

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_output())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.apply_preset()
        self.refresh_diagnostics()

    def _build_controls(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)

        model_box = QGroupBox("Model")
        model_form = QFormLayout(model_box)
        self.case_type = QComboBox()
        self.case_type.addItems(
            ["lid_driven_cavity", "poiseuille_channel", "couette_flow", "periodic_channel", "cylinder_flow", "custom"]
        )
        self.case_type.currentTextChanged.connect(self.apply_preset)
        self.collision_model = QComboBox()
        self.collision_model.addItems(["BGK"])
        model_form.addRow("case_type", self.case_type)
        model_form.addRow("collision_model", self.collision_model)
        layout.addWidget(model_box)

        grid_box = QGroupBox("Grid and Flow")
        grid_form = QFormLayout(grid_box)
        self.nx = self._int(128, 16, 2048, 16)
        self.ny = self._int(128, 16, 2048, 16)
        self.rho0 = self._double(1.0, 0.01, 10.0, 0.01, 5)
        self.reynolds = self._double(1000.0, 1.0, 1_000_000.0, 50.0, 3)
        self.u_ref = self._double(0.05, 0.0, 0.5, 0.005, 6)
        self.l_ref = self._double(0.0, 0.0, 100000.0, 1.0, 3)
        for widget in (self.nx, self.ny, self.rho0, self.reynolds, self.u_ref, self.l_ref):
            widget.valueChanged.connect(self.refresh_diagnostics)
        grid_form.addRow("NX", self.nx)
        grid_form.addRow("NY", self.ny)
        grid_form.addRow("rho0", self.rho0)
        grid_form.addRow("Re", self.reynolds)
        grid_form.addRow("U_ref", self.u_ref)
        grid_form.addRow("L_ref, 0 = auto", self.l_ref)
        layout.addWidget(grid_box)

        conv_box = QGroupBox("Convergence")
        conv_form = QFormLayout(conv_box)
        self.tol = self._double(1e-6, 1e-12, 1e-1, 1e-6, 12)
        self.max_iter = self._int(5000, 1, 10_000_000, 1000)
        self.min_iter = self._int(500, 0, 10_000_000, 100)
        self.report_interval = self._int(100, 1, 1_000_000, 50)
        self.ramp_steps = self._int(1000, 1, 1_000_000, 100)
        conv_form.addRow("tol", self.tol)
        conv_form.addRow("max_iter", self.max_iter)
        conv_form.addRow("min_iter", self.min_iter)
        conv_form.addRow("report_interval", self.report_interval)
        conv_form.addRow("ramp_steps", self.ramp_steps)
        layout.addWidget(conv_box)

        self.tabs = QTabWidget()
        self.boundary_widget = QWidget()
        boundary_layout = QGridLayout(self.boundary_widget)
        preset = case_preset("lid_driven_cavity", 128, 128)
        self.left_panel = BoundaryPanel("left", preset.left)
        self.right_panel = BoundaryPanel("right", preset.right)
        self.bottom_panel = BoundaryPanel("bottom", preset.bottom)
        self.top_panel = BoundaryPanel("top", preset.top)
        boundary_layout.addWidget(self.left_panel, 0, 0)
        boundary_layout.addWidget(self.right_panel, 0, 1)
        boundary_layout.addWidget(self.bottom_panel, 1, 0)
        boundary_layout.addWidget(self.top_panel, 1, 1)
        self.tabs.addTab(self.boundary_widget, "Boundaries")

        obstacle_widget = QWidget()
        obstacle_form = QFormLayout(obstacle_widget)
        self.obstacle_type = QComboBox()
        self.obstacle_type.addItems(["none", "cylinder", "rectangle"])
        self.obs_cx = self._double(0.25, 0.0, 1.0, 0.01, 3)
        self.obs_cy = self._double(0.5, 0.0, 1.0, 0.01, 3)
        self.obs_radius = self._double(0.08, 0.01, 0.5, 0.01, 3)
        self.obs_width = self._double(0.12, 0.01, 1.0, 0.01, 3)
        self.obs_height = self._double(0.20, 0.01, 1.0, 0.01, 3)
        obstacle_form.addRow("obstacle type", self.obstacle_type)
        obstacle_form.addRow("cx", self.obs_cx)
        obstacle_form.addRow("cy", self.obs_cy)
        obstacle_form.addRow("radius", self.obs_radius)
        obstacle_form.addRow("width", self.obs_width)
        obstacle_form.addRow("height", self.obs_height)
        self.tabs.addTab(obstacle_widget, "Obstacle")
        layout.addWidget(self.tabs)

        output_box = QGroupBox("Output")
        output_form = QFormLayout(output_box)
        self.output_dir = QLabel("results/desktop_runs")
        pick_dir = QPushButton("Choose")
        pick_dir.clicked.connect(self.choose_output_dir)
        self.save_npz = QCheckBox("save_npz")
        self.save_npz.setChecked(True)
        self.save_csv = QCheckBox("save_csv")
        self.save_csv.setChecked(True)
        self.save_png = QCheckBox("save_png")
        self.save_png.setChecked(True)
        output_form.addRow("output_dir", self.output_dir)
        output_form.addRow("", pick_dir)
        output_form.addRow(self.save_npz)
        output_form.addRow(self.save_csv)
        output_form.addRow(self.save_png)
        layout.addWidget(output_box)

        button_row = QWidget()
        button_layout = QGridLayout(button_row)
        self.recommend_button = QPushButton("Recommend U_ref")
        self.recommend_button.clicked.connect(self.recommend_parameters)
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_simulation)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_simulation)
        button_layout.addWidget(self.recommend_button, 0, 0)
        button_layout.addWidget(self.start_button, 0, 1)
        button_layout.addWidget(self.stop_button, 0, 2)
        layout.addWidget(button_row)
        layout.addStretch(1)
        return root

    def _build_output(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        self.diagnostics = QTextEdit()
        self.diagnostics.setReadOnly(True)
        self.diagnostics.setMaximumHeight(190)
        layout.addWidget(self.diagnostics)

        image_row = QWidget()
        image_layout = QGridLayout(image_row)
        self.speed_label = QLabel("velocity magnitude preview")
        self.vort_label = QLabel("vorticity preview")
        for label in (self.speed_label, self.vort_label):
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(420, 300)
            label.setStyleSheet("border: 1px solid #999; background: #202020; color: #ddd;")
        image_layout.addWidget(self.speed_label, 0, 0)
        image_layout.addWidget(self.vort_label, 0, 1)
        layout.addWidget(image_row, stretch=1)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(240)
        layout.addWidget(self.log)
        return root

    @staticmethod
    def _int(value: int, minimum: int, maximum: int, step: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    @staticmethod
    def _double(value: float, minimum: float, maximum: float, step: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin

    def apply_preset(self, *_args: object) -> None:
        cfg = case_preset(self.case_type.currentText(), self.nx.value(), self.ny.value())
        cfg.u_ref = self.u_ref.value()
        cfg.reynolds = self.reynolds.value()
        if cfg.case_type == "lid_driven_cavity":
            cfg.top.ux = cfg.u_ref
        self.left_panel.set_boundary(cfg.left)
        self.right_panel.set_boundary(cfg.right)
        self.bottom_panel.set_boundary(cfg.bottom)
        self.top_panel.set_boundary(cfg.top)
        self.obstacle_type.setCurrentText(cfg.obstacle.type)
        self.obs_cx.setValue(cfg.obstacle.cx)
        self.obs_cy.setValue(cfg.obstacle.cy)
        self.obs_radius.setValue(cfg.obstacle.radius)
        self.obs_width.setValue(cfg.obstacle.width)
        self.obs_height.setValue(cfg.obstacle.height)
        self.refresh_diagnostics()

    def choose_output_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Choose output directory", self.output_dir.text())
        if selected:
            self.output_dir.setText(selected)

    def make_config(self) -> SolverConfig:
        return SolverConfig(
            case_type=self.case_type.currentText(),
            collision_model=self.collision_model.currentText(),
            boundary_scheme_default="desktop",
            nx=self.nx.value(),
            ny=self.ny.value(),
            rho0=self.rho0.value(),
            u_ref=self.u_ref.value(),
            reynolds=self.reynolds.value(),
            l_ref=None if self.l_ref.value() <= 0 else self.l_ref.value(),
            tol=self.tol.value(),
            max_iter=self.max_iter.value(),
            min_iter=self.min_iter.value(),
            report_interval=self.report_interval.value(),
            ramp_steps=self.ramp_steps.value(),
            left=self.left_panel.value(),
            right=self.right_panel.value(),
            bottom=self.bottom_panel.value(),
            top=self.top_panel.value(),
            obstacle=ObstacleConfig(
                type=self.obstacle_type.currentText(),
                cx=self.obs_cx.value(),
                cy=self.obs_cy.value(),
                radius=self.obs_radius.value(),
                width=self.obs_width.value(),
                height=self.obs_height.value(),
            ),
            output=OutputConfig(
                output_dir=self.output_dir.text(),
                save_npz=self.save_npz.isChecked(),
                save_csv=self.save_csv.isChecked(),
                save_png=self.save_png.isChecked(),
                save_animation=False,
                plot_interval=self.report_interval.value(),
            ),
        )

    def refresh_diagnostics(self, *_args: object) -> None:
        try:
            cfg = self.make_config()
            errors, warnings = validate_config(cfg)
            derived = derived_parameters(cfg)
            lines = [
                f"Re: {derived['Re']:.6g}",
                f"Ma: {derived['Ma']:.6g}",
                f"nu_lattice: {derived['nu_lattice']:.6g}",
                f"tau: {derived['tau']:.6g}",
                f"omega: {derived['omega']:.6g}",
                f"stability: {derived['stability_level']}",
            ]
            if warnings:
                lines.append("\nWarnings:")
                lines.extend(f"- {warning}" for warning in warnings)
            if errors:
                lines.append("\nErrors:")
                lines.extend(f"- {error}" for error in errors)
            self.diagnostics.setPlainText("\n".join(lines))
            self.start_button.setEnabled(not errors and self.worker is None)
        except Exception as exc:
            self.diagnostics.setPlainText(str(exc))
            self.start_button.setEnabled(False)

    def recommend_parameters(self) -> None:
        rec = recommend_u_ref(
            reynolds=self.reynolds.value(),
            nx=self.nx.value(),
            ny=self.ny.value(),
            case_type=self.case_type.currentText(),
        )
        self.u_ref.setValue(float(rec["U_ref"]))
        self.log.append(f"Recommended U_ref={rec['U_ref']:.6g}. {rec['message']}")
        self.refresh_diagnostics()

    def start_simulation(self) -> None:
        cfg = self.make_config()
        errors, warnings = validate_config(cfg)
        if errors:
            QMessageBox.critical(self, "Invalid configuration", "\n".join(errors))
            return
        for warning in warnings:
            self.log.append(f"Warning: {warning}")

        self.thread = QThread()
        self.worker = SimulationWorker(cfg)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.report.connect(self.on_report)
        self.worker.preview.connect(self.on_preview)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.cleanup_worker)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log.append("Simulation started.")
        self.thread.start()

    def stop_simulation(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.log.append("Stop requested; current partial fields will be saved.")

    def on_report(self, data: dict) -> None:
        self.log.append(
            "iter={iteration} residual={residual:.3e} q={q} q_avg={q_avg} "
            "mass={mass_drift:.3e} max_u={max_velocity:.3e} status={status}".format(**data)
        )

    def on_preview(self, speed_path: str, vort_path: str) -> None:
        self._set_pixmap(self.speed_label, speed_path)
        self._set_pixmap(self.vort_label, vort_path)

    @staticmethod
    def _set_pixmap(label: QLabel, path: str) -> None:
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_finished(self, run_dir: str) -> None:
        self.last_run_dir = run_dir
        self.log.append(f"Finished. Results saved to: {run_dir}")
        QMessageBox.information(self, "Simulation finished", f"Results saved to:\n{run_dir}")

    def on_failed(self, message: str) -> None:
        self.log.append(f"Failed: {message}")
        QMessageBox.critical(self, "Simulation failed", message)

    def cleanup_worker(self) -> None:
        self.worker = None
        self.thread = None
        self.stop_button.setEnabled(False)
        self.refresh_diagnostics()


def main() -> int:
    app = QApplication(sys.argv)
    window = DesktopWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
