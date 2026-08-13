"""PySide6 desktop UI for the customized 2D LBM simulator.

This UI does not open a browser and does not bind a web port.
"""

from __future__ import annotations

import math
import os
import sys
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
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import (
    CORE_BOUNDARY_TYPES,
    SUPPORTED_CASE_TYPES,
    SUPPORTED_COLLISION_MODELS,
    SUPPORTED_RAMP_PROFILES,
    SUPPORTED_THERMAL_BOUNDARIES,
    BoundaryConfig,
    ObstacleConfig,
    OutputConfig,
    SolverConfig,
    ThermalBoundaryConfig,
    case_preset,
    derived_parameters,
    recommend_u_ref,
    validate_config,
)
from lbm_solver import LBMSolver, Report
from postprocess import save_results, speed, vorticity


class SimulationWorker(QObject):
    report = Signal(dict)
    preview = Signal(str, str, str)
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
                    speed_path, vort_path, temperature_path = self._write_preview(solver, report)
                    self.preview.emit(
                        str(speed_path), str(vort_path), str(temperature_path or "")
                    )
                if self._stop:
                    break
            run_dir = save_results(solver, self.cfg)
            self.finished.emit(str(run_dir))
        except Exception as exc:  # noqa: BLE001 - worker errors must reach the UI
            self.failed.emit(str(exc))

    def _write_preview(
        self, solver: LBMSolver, report: Report
    ) -> tuple[Path, Path, Path | None]:
        preview_dir = Path("logs/desktop_preview")
        preview_dir.mkdir(parents=True, exist_ok=True)
        fields = solver.fields()
        speed_path = preview_dir / "speed.png"
        vort_path = preview_dir / "vorticity.png"
        self._imshow(speed(fields["ux"], fields["uy"]), fields["solid_mask"], speed_path, "|u|")
        self._imshow(vorticity(fields["ux"], fields["uy"]), fields["solid_mask"], vort_path, "vorticity")
        temperature_path = None
        if "temperature" in fields:
            temperature_path = preview_dir / "temperature.png"
            self._imshow(
                fields["temperature"],
                fields["solid_mask"],
                temperature_path,
                "temperature",
            )
        return speed_path, vort_path, temperature_path

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
        layout.setContentsMargins(12, 18, 12, 12)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        self.setMinimumHeight(185)
        self.setMinimumWidth(275)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.type_box = QComboBox()
        self.type_box.addItems(sorted(CORE_BOUNDARY_TYPES))
        self.type_box.setCurrentText(default.type if default.type in CORE_BOUNDARY_TYPES else "no_slip_bounce_back")
        self.type_box.setMinimumHeight(28)
        self.type_box.setMinimumWidth(185)
        self.ux = self._double(default.ux, -1.0, 1.0, 0.005, 6)
        self.uy = self._double(default.uy, -1.0, 1.0, 0.005, 6)
        self.rho = self._double(default.rho, 0.0001, 10.0, 0.01, 6)
        self.rb = self._double(default.rb, 0.0, 1.0, 0.05, 3)
        self.labels: dict[str, QLabel] = {}
        for name, field in (
            ("type", self.type_box),
            ("ux", self.ux),
            ("uy", self.uy),
            ("rho", self.rho),
            ("rb", self.rb),
        ):
            label = QLabel(name)
            label.setMinimumWidth(42)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.labels[name] = label
            layout.addRow(label, field)
        self.type_box.currentTextChanged.connect(self.update_type_controls)
        self.update_type_controls()

    def update_type_controls(self, *_args: object) -> None:
        uses_mixing = self.type_box.currentText() == "mixed_bounce_specular"
        self.rb.setEnabled(uses_mixing)
        self.labels["rb"].setEnabled(uses_mixing)
        self.rb.setToolTip(
            "Bounce-back fraction: 1 is no-slip, 0 is specular slip."
            if uses_mixing
            else "Used only by mixed_bounce_specular."
        )

    @staticmethod
    def _double(value: float, minimum: float, maximum: float, step: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setMinimumHeight(28)
        spin.setMinimumWidth(185)
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


class ThermalBoundaryPanel(QGroupBox):
    def __init__(self, side: str, default: ThermalBoundaryConfig):
        super().__init__(f"{side} thermal boundary")
        layout = QFormLayout(self)
        layout.setContentsMargins(12, 18, 12, 12)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)
        self.setMinimumWidth(275)
        self.type_box = QComboBox()
        self.type_box.addItems(SUPPORTED_THERMAL_BOUNDARIES)
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(-1e6, 1e6)
        self.temperature.setDecimals(6)
        self.temperature.setSingleStep(0.1)
        self.temperature.setMinimumWidth(185)
        self.temperature.setMinimumHeight(28)
        self.type_box.setMinimumWidth(185)
        self.type_box.setMinimumHeight(28)
        layout.addRow("type", self.type_box)
        layout.addRow("temperature", self.temperature)
        self.type_box.currentTextChanged.connect(self.update_controls)
        self.set_boundary(default)

    def update_controls(self, *_args: object) -> None:
        self.temperature.setEnabled(self.type_box.currentText() == "isothermal")

    def value(self) -> ThermalBoundaryConfig:
        return ThermalBoundaryConfig(
            type=self.type_box.currentText(), temperature=self.temperature.value()
        )

    def set_boundary(self, bc: ThermalBoundaryConfig) -> None:
        self.type_box.setCurrentText(bc.type)
        self.temperature.setValue(bc.temperature)
        self.update_controls()


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
        root.setMinimumWidth(620)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        model_box = QGroupBox("Model")
        model_form = QFormLayout(model_box)
        self.case_type = QComboBox()
        self.case_type.addItems(SUPPORTED_CASE_TYPES)
        self.case_type.currentTextChanged.connect(self.apply_preset)
        self.collision_model = QComboBox()
        self.collision_model.addItems(SUPPORTED_COLLISION_MODELS)
        self.trt_magic_parameter = self._double(3.0 / 16.0, 0.001, 1.0, 0.01, 6)
        self.trt_magic_parameter.setToolTip(
            "TRT magic parameter Lambda; 3/16 is a common bounce-back default."
        )
        self.mrt_preset = QComboBox()
        self.mrt_preset.addItem("Lallemand-Luo", "lallemand_luo")
        self.mrt_preset.addItem("BGK-equivalent", "bgk_equivalent")
        self.mrt_preset.addItem("Custom", "custom")
        self.mrt_s_e = self._double(1.64, 0.001, 1.999, 0.01, 3)
        self.mrt_s_epsilon = self._double(1.54, 0.001, 1.999, 0.01, 3)
        self.mrt_s_q = self._double(1.90, 0.001, 1.999, 0.01, 3)
        for widget in (self.mrt_s_e, self.mrt_s_epsilon, self.mrt_s_q):
            widget.setToolTip("MRT non-conserved moment relaxation rate; valid range is 0 to 2.")
        model_form.addRow("case_type", self.case_type)
        model_form.addRow("collision_model", self.collision_model)
        model_form.addRow("TRT Lambda", self.trt_magic_parameter)
        model_form.addRow("MRT preset", self.mrt_preset)
        model_form.addRow("MRT s_e", self.mrt_s_e)
        model_form.addRow("MRT s_epsilon", self.mrt_s_epsilon)
        model_form.addRow("MRT s_q", self.mrt_s_q)
        self.collision_parameter_rows = {
            "TRT": [self.trt_magic_parameter],
            "MRT": [self.mrt_preset, self.mrt_s_e, self.mrt_s_epsilon, self.mrt_s_q],
        }
        self.collision_model.currentTextChanged.connect(self.update_collision_controls)
        self.mrt_preset.currentIndexChanged.connect(self.update_collision_controls)
        layout.addWidget(model_box)

        grid_box = QGroupBox("Grid and Flow")
        grid_form = QFormLayout(grid_box)
        self.parameter_mode = QComboBox()
        self.parameter_mode.addItem("Re -> tau", "reynolds")
        self.parameter_mode.addItem("Direct tau", "tau")
        self.parameter_mode.addItem("Physical units", "physical")
        self.parameter_mode.addItem("Rayleigh / Pr", "rayleigh")
        self.nx = self._int(128, 16, 2048, 16)
        self.ny = self._int(128, 16, 2048, 16)
        self.rho0 = self._double(1.0, 0.01, 10.0, 0.01, 5)
        self.reynolds = self._double(1000.0, 1.0, 1_000_000.0, 50.0, 3)
        self.tau_target = self._double(0.6, 0.500001, 10.0, 0.01, 6)
        self.u_ref = self._double(0.05, 0.0, 0.5, 0.005, 6)
        self.l_ref = self._double(0.0, 0.0, 100000.0, 1.0, 3)
        self.length_phys = self._double(1.0, 1e-12, 1e12, 0.1, 6)
        self.velocity_phys = self._double(1.0, 1e-12, 1e12, 0.1, 6)
        self.nu_phys = self._double(1e-6, 1e-15, 1e6, 1e-6, 12)
        for widget in (
            self.nx,
            self.ny,
            self.rho0,
            self.reynolds,
            self.tau_target,
            self.u_ref,
            self.l_ref,
            self.length_phys,
            self.velocity_phys,
            self.nu_phys,
        ):
            widget.valueChanged.connect(self.refresh_diagnostics)
        self.parameter_mode.currentIndexChanged.connect(self.update_flow_parameter_controls)
        grid_form.addRow("parameter_mode", self.parameter_mode)
        grid_form.addRow("NX", self.nx)
        grid_form.addRow("NY", self.ny)
        grid_form.addRow("rho0", self.rho0)
        grid_form.addRow("Re", self.reynolds)
        grid_form.addRow("tau_target", self.tau_target)
        grid_form.addRow("U_ref", self.u_ref)
        grid_form.addRow("L_ref, 0 = auto", self.l_ref)
        grid_form.addRow("L_phys", self.length_phys)
        grid_form.addRow("U_phys", self.velocity_phys)
        grid_form.addRow("nu_phys", self.nu_phys)
        layout.addWidget(grid_box)

        conv_box = QGroupBox("Convergence")
        conv_form = QFormLayout(conv_box)
        self.tol = self._double(1e-6, 1e-12, 1e-1, 1e-6, 12)
        self.max_iter = self._int(5000, 1, 10_000_000, 1000)
        self.min_iter = self._int(500, 0, 10_000_000, 100)
        self.report_interval = self._int(100, 1, 1_000_000, 50)
        self.ramp_steps = self._int(1000, 1, 1_000_000, 100)
        self.ramp_profile = QComboBox()
        self.ramp_profile.addItems(SUPPORTED_RAMP_PROFILES)
        self.ramp_profile.setCurrentText("smoothstep")
        conv_form.addRow("tol", self.tol)
        conv_form.addRow("max_iter", self.max_iter)
        conv_form.addRow("min_iter", self.min_iter)
        conv_form.addRow("report_interval", self.report_interval)
        layout.addWidget(conv_box)

        self.tabs = QTabWidget()
        self.tabs.setMinimumHeight(455)
        self.boundary_widget = QWidget()
        self.boundary_widget.setMinimumHeight(400)
        boundary_layout = QGridLayout(self.boundary_widget)
        boundary_layout.setContentsMargins(12, 12, 12, 12)
        boundary_layout.setHorizontalSpacing(12)
        boundary_layout.setVerticalSpacing(14)
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

        numerics_widget = QWidget()
        numerics_form = QFormLayout(numerics_widget)
        self.body_force_x = self._double(0.0, -0.01, 0.01, 1e-7, 10)
        self.body_force_y = self._double(0.0, -0.01, 0.01, 1e-7, 10)
        self.mass_drift_warning = self._double(1e-4, 1e-12, 1.0, 1e-5, 10)
        self.mass_drift_limit = self._double(1e-3, 1e-12, 1.0, 1e-4, 10)
        self.residual_limit = self._double(1e3, 1.0, 1e12, 100.0, 3)
        self.max_velocity_limit = self._double(0.3, 0.001, 10.0, 0.01, 4)
        numerics_form.addRow("body_force_x", self.body_force_x)
        numerics_form.addRow("body_force_y", self.body_force_y)
        numerics_form.addRow("ramp_profile", self.ramp_profile)
        numerics_form.addRow("ramp_steps", self.ramp_steps)
        numerics_form.addRow("mass_drift_warning", self.mass_drift_warning)
        numerics_form.addRow("mass_drift_limit", self.mass_drift_limit)
        numerics_form.addRow("residual_limit", self.residual_limit)
        numerics_form.addRow("max_velocity_limit", self.max_velocity_limit)
        for widget in (
            self.body_force_x,
            self.body_force_y,
            self.mass_drift_warning,
            self.mass_drift_limit,
            self.residual_limit,
            self.max_velocity_limit,
        ):
            widget.valueChanged.connect(self.refresh_diagnostics)
        self.ramp_profile.currentTextChanged.connect(self.refresh_diagnostics)
        self.tabs.addTab(numerics_widget, "Numerics")

        thermal_widget = QWidget()
        thermal_form = QFormLayout(thermal_widget)
        self.thermal_enabled = QCheckBox()
        self.thermal_model = QComboBox()
        self.thermal_model.addItems(["D2Q5_BGK"])
        self.thermal_buoyancy = QCheckBox()
        self.prandtl = self._double(0.71, 1e-6, 1e6, 0.01, 6)
        self.rayleigh = self._double(1e4, 0.0, 1e12, 1e3, 3)
        self.temperature_hot = self._double(1.0, -1e6, 1e6, 0.1, 6)
        self.temperature_cold = self._double(0.0, -1e6, 1e6, 0.1, 6)
        self.temperature_initial = self._double(0.5, -1e6, 1e6, 0.1, 6)
        self.temperature_reference = self._double(0.5, -1e6, 1e6, 0.1, 6)
        self.gravity_x = self._double(0.0, -1.0, 1.0, 0.1, 4)
        self.gravity_y = self._double(-1.0, -1.0, 1.0, 0.1, 4)
        self.thermal_tol = self._double(1e-6, 1e-12, 1.0, 1e-6, 12)
        thermal_form.addRow("thermal_enabled", self.thermal_enabled)
        thermal_form.addRow("thermal_model", self.thermal_model)
        thermal_form.addRow("Boussinesq buoyancy", self.thermal_buoyancy)
        thermal_form.addRow("Pr", self.prandtl)
        thermal_form.addRow("Ra", self.rayleigh)
        thermal_form.addRow("T_hot", self.temperature_hot)
        thermal_form.addRow("T_cold", self.temperature_cold)
        thermal_form.addRow("T_initial", self.temperature_initial)
        thermal_form.addRow("T_reference", self.temperature_reference)
        thermal_form.addRow("gravity_x", self.gravity_x)
        thermal_form.addRow("gravity_y", self.gravity_y)
        thermal_form.addRow("thermal_tol", self.thermal_tol)
        self.thermal_tab_index = self.tabs.addTab(thermal_widget, "Thermal")

        thermal_boundary_widget = QWidget()
        thermal_boundary_layout = QGridLayout(thermal_boundary_widget)
        thermal_preset = case_preset("custom", 128, 128)
        self.thermal_left_panel = ThermalBoundaryPanel("left", thermal_preset.thermal_left)
        self.thermal_right_panel = ThermalBoundaryPanel("right", thermal_preset.thermal_right)
        self.thermal_bottom_panel = ThermalBoundaryPanel("bottom", thermal_preset.thermal_bottom)
        self.thermal_top_panel = ThermalBoundaryPanel("top", thermal_preset.thermal_top)
        thermal_boundary_layout.addWidget(self.thermal_left_panel, 0, 0)
        thermal_boundary_layout.addWidget(self.thermal_right_panel, 0, 1)
        thermal_boundary_layout.addWidget(self.thermal_bottom_panel, 1, 0)
        thermal_boundary_layout.addWidget(self.thermal_top_panel, 1, 1)
        self.thermal_boundary_tab_index = self.tabs.addTab(
            thermal_boundary_widget, "Thermal BCs"
        )
        self.temperature_hot.valueChanged.connect(self.sync_thermal_boundary_temperatures)
        self.temperature_cold.valueChanged.connect(self.sync_thermal_boundary_temperatures)
        self.thermal_enabled.stateChanged.connect(self.update_thermal_controls)
        self.thermal_buoyancy.stateChanged.connect(self.update_thermal_controls)
        for widget in (
            self.prandtl,
            self.rayleigh,
            self.temperature_hot,
            self.temperature_cold,
            self.temperature_initial,
            self.temperature_reference,
            self.gravity_x,
            self.gravity_y,
            self.thermal_tol,
        ):
            widget.valueChanged.connect(self.refresh_diagnostics)
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

        self.update_collision_controls()
        self.update_flow_parameter_controls()
        self.update_thermal_controls()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(root)
        scroll.setMinimumWidth(650)
        return scroll

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
        self.temperature_label = QLabel("temperature preview")
        self.temperature_label.setAlignment(Qt.AlignCenter)
        self.temperature_label.setMinimumSize(420, 300)
        self.temperature_label.setStyleSheet(
            "border: 1px solid #999; background: #202020; color: #ddd;"
        )
        image_layout.addWidget(self.speed_label, 0, 0)
        image_layout.addWidget(self.vort_label, 0, 1)
        image_layout.addWidget(self.temperature_label, 1, 0, 1, 2)
        self.temperature_label.hide()
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
        cfg = case_preset(
            self.case_type.currentText(),
            self.nx.value(),
            self.ny.value(),
            u_ref=self.u_ref.value(),
        )
        if cfg.case_type == "heated_channel_flow":
            self.reynolds.setValue(cfg.reynolds)
        cfg.reynolds = self.reynolds.value()
        self.parameter_mode.setCurrentIndex(
            max(0, self.parameter_mode.findData(cfg.parameter_mode))
        )
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
        self.body_force_x.setValue(cfg.body_force_x)
        self.body_force_y.setValue(cfg.body_force_y)
        self.thermal_enabled.setChecked(cfg.thermal_enabled)
        self.thermal_buoyancy.setChecked(cfg.thermal_buoyancy)
        self.prandtl.setValue(cfg.prandtl)
        self.rayleigh.setValue(cfg.rayleigh)
        self.temperature_hot.setValue(cfg.temperature_hot)
        self.temperature_cold.setValue(cfg.temperature_cold)
        self.temperature_initial.setValue(cfg.temperature_initial)
        self.temperature_reference.setValue(cfg.temperature_reference)
        self.gravity_x.setValue(cfg.gravity_x)
        self.gravity_y.setValue(cfg.gravity_y)
        self.thermal_left_panel.set_boundary(cfg.thermal_left)
        self.thermal_right_panel.set_boundary(cfg.thermal_right)
        self.thermal_bottom_panel.set_boundary(cfg.thermal_bottom)
        self.thermal_top_panel.set_boundary(cfg.thermal_top)
        self.update_thermal_controls()
        self.refresh_diagnostics()

    def update_collision_controls(self, *_args: object) -> None:
        selected = self.collision_model.currentText()
        for model, fields in self.collision_parameter_rows.items():
            for field in fields:
                visible = selected == model
                if model == "MRT" and field is not self.mrt_preset:
                    visible = visible and self.mrt_preset.currentData() == "custom"
                label = field.parentWidget().layout().labelForField(field)
                if label is not None:
                    label.setVisible(visible)
                field.setVisible(visible)
        if hasattr(self, "diagnostics"):
            self.refresh_diagnostics()

    def update_flow_parameter_controls(self, *_args: object) -> None:
        mode = self.parameter_mode.currentData()
        field_visibility = (
            (self.reynolds, mode == "reynolds"),
            (self.tau_target, mode in {"tau", "rayleigh"}),
            (self.length_phys, mode == "physical"),
            (self.velocity_phys, mode == "physical"),
            (self.nu_phys, mode == "physical"),
        )
        for field, visible in field_visibility:
            label = field.parentWidget().layout().labelForField(field)
            if label is not None:
                label.setVisible(visible)
            field.setVisible(visible)
        if hasattr(self, "recommend_button"):
            self.recommend_button.setEnabled(mode == "reynolds")
        if hasattr(self, "diagnostics"):
            self.refresh_diagnostics()

    def update_thermal_controls(self, *_args: object) -> None:
        enabled = self.thermal_enabled.isChecked()
        buoyancy = enabled and self.thermal_buoyancy.isChecked()
        for field in (
            self.thermal_model,
            self.thermal_buoyancy,
            self.prandtl,
            self.temperature_hot,
            self.temperature_cold,
            self.temperature_initial,
            self.temperature_reference,
            self.thermal_tol,
        ):
            field.setEnabled(enabled)
        for field in (self.rayleigh, self.gravity_x, self.gravity_y):
            field.setEnabled(buoyancy)
        self.tabs.setTabEnabled(self.thermal_boundary_tab_index, enabled)
        if hasattr(self, "diagnostics"):
            self.refresh_diagnostics()

    def sync_thermal_boundary_temperatures(self, *_args: object) -> None:
        case_type = self.case_type.currentText()
        if case_type == "natural_convection_cavity":
            self.thermal_left_panel.temperature.setValue(self.temperature_hot.value())
            self.thermal_right_panel.temperature.setValue(self.temperature_cold.value())
        elif case_type in {"rayleigh_benard_convection", "heated_channel_flow"}:
            self.thermal_bottom_panel.temperature.setValue(self.temperature_hot.value())
            self.thermal_top_panel.temperature.setValue(self.temperature_cold.value())

    def choose_output_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Choose output directory", self.output_dir.text())
        if selected:
            self.output_dir.setText(selected)

    def make_config(self) -> SolverConfig:
        return SolverConfig(
            case_type=self.case_type.currentText(),
            collision_model=self.collision_model.currentText(),
            trt_magic_parameter=self.trt_magic_parameter.value(),
            mrt_preset=str(self.mrt_preset.currentData()),
            mrt_s_e=self.mrt_s_e.value(),
            mrt_s_epsilon=self.mrt_s_epsilon.value(),
            mrt_s_q=self.mrt_s_q.value(),
            boundary_scheme_default="desktop",
            parameter_mode=str(self.parameter_mode.currentData()),
            nx=self.nx.value(),
            ny=self.ny.value(),
            rho0=self.rho0.value(),
            u_ref=self.u_ref.value(),
            reynolds=self.reynolds.value(),
            tau_target=self.tau_target.value(),
            length_phys=self.length_phys.value(),
            velocity_phys=self.velocity_phys.value(),
            nu_phys=self.nu_phys.value(),
            l_ref=None if self.l_ref.value() <= 0 else self.l_ref.value(),
            thermal_enabled=self.thermal_enabled.isChecked(),
            thermal_model=self.thermal_model.currentText(),
            thermal_buoyancy=self.thermal_buoyancy.isChecked(),
            prandtl=self.prandtl.value(),
            rayleigh=self.rayleigh.value(),
            temperature_hot=self.temperature_hot.value(),
            temperature_cold=self.temperature_cold.value(),
            temperature_initial=self.temperature_initial.value(),
            temperature_reference=self.temperature_reference.value(),
            gravity_x=self.gravity_x.value(),
            gravity_y=self.gravity_y.value(),
            thermal_tol=self.thermal_tol.value(),
            tol=self.tol.value(),
            max_iter=self.max_iter.value(),
            min_iter=self.min_iter.value(),
            report_interval=self.report_interval.value(),
            ramp_steps=self.ramp_steps.value(),
            ramp_profile=self.ramp_profile.currentText(),
            body_force_x=self.body_force_x.value(),
            body_force_y=self.body_force_y.value(),
            mass_drift_warning=self.mass_drift_warning.value(),
            mass_drift_limit=self.mass_drift_limit.value(),
            residual_limit=self.residual_limit.value(),
            max_velocity_limit=self.max_velocity_limit.value(),
            left=self.left_panel.value(),
            right=self.right_panel.value(),
            bottom=self.bottom_panel.value(),
            top=self.top_panel.value(),
            thermal_left=self.thermal_left_panel.value(),
            thermal_right=self.thermal_right_panel.value(),
            thermal_bottom=self.thermal_bottom_panel.value(),
            thermal_top=self.thermal_top_panel.value(),
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
                f"collision_model: {derived['collision_model']}",
                f"parameter_mode: {derived['parameter_mode']}",
                f"Re: {derived['Re']:.6g}",
                f"Ma: {derived['Ma']:.6g}",
                f"nu_lattice: {derived['nu_lattice']:.6g}",
                f"tau: {derived['tau']:.6g}",
                f"omega: {derived['omega']:.6g}",
                f"stability: {derived['stability_level']}",
            ]
            if cfg.collision_model == "TRT":
                lines.extend(
                    [
                        f"TRT Lambda: {derived['trt_magic_parameter']:.6g}",
                        f"tau_minus: {derived['tau_minus']:.6g}",
                        f"omega_minus: {derived['omega_minus']:.6g}",
                    ]
                )
            elif cfg.collision_model == "MRT":
                lines.append(
                    f"MRT preset: {derived['mrt_preset']}\nMRT rates: "
                    f"s_e={derived['mrt_s_e']:.4g}, "
                    f"s_epsilon={derived['mrt_s_epsilon']:.4g}, "
                    f"s_q={derived['mrt_s_q']:.4g}"
                )
            if math.hypot(cfg.body_force_x, cfg.body_force_y) > 0.0:
                lines.append(
                    f"body_force: ({cfg.body_force_x:.4g}, {cfg.body_force_y:.4g}), "
                    f"ramp={cfg.ramp_profile}"
                )
            if cfg.parameter_mode == "physical":
                lines.append(
                    f"physical scaling: dx={derived['dx_phys']:.6g}, "
                    f"dt={derived['dt_phys']:.6g}, nu_phys={derived['nu_phys']:.6g}"
                )
            if cfg.thermal_enabled:
                lines.extend(
                    [
                        (
                            f"thermal: {derived['thermal_model']}, "
                            f"Pr={derived['Pr']:.6g}, Ra={derived['Ra']:.6g}"
                        ),
                        f"alpha_lattice: {derived['alpha_lattice']:.6g}",
                        f"tau_thermal: {derived['tau_thermal']:.6g}",
                    ]
                )
            if warnings:
                lines.append("\nWarnings:")
                lines.extend(f"- {warning}" for warning in warnings)
            if errors:
                lines.append("\nErrors:")
                lines.extend(f"- {error}" for error in errors)
            self.diagnostics.setPlainText("\n".join(lines))
            self.start_button.setEnabled(not errors and self.worker is None)
        except Exception as exc:  # noqa: BLE001 - diagnostics must not crash the window
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
        message = (
            "iter={iteration} residual={residual:.3e} q={q} q_avg={q_avg} "
            "mass={mass_drift:.3e} max_u={max_velocity:.3e} status={status}".format(**data)
        )
        if data.get("temperature_residual") is not None:
            message += (
                f" T_res={data['temperature_residual']:.3e} "
                f"Nu_avg={data['nusselt_average']}"
            )
        self.log.append(message)

    def on_preview(self, speed_path: str, vort_path: str, temperature_path: str) -> None:
        self._set_pixmap(self.speed_label, speed_path)
        self._set_pixmap(self.vort_label, vort_path)
        self.temperature_label.setVisible(bool(temperature_path))
        if temperature_path:
            self._set_pixmap(self.temperature_label, temperature_path)

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
