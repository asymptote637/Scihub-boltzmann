"""PySide6 desktop UI for the 2D LBM simulator.

This is the default non-web frontend. It does not bind to any local port.
"""

from __future__ import annotations

import io
import queue
import sys
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QTimer
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
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import (
    BOUNDARY_TYPES,
    CASE_TYPES,
    BoundaryConfig,
    ConvergenceConfig,
    FlowConfig,
    GridConfig,
    OutputConfig,
    SimulationConfig,
    default_boundaries_for_case,
    recommend_parameters,
    validate_config,
    characteristic_length,
)
from lbm_solver import LBMSolver



class LBMQtWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("2D LBM Simulator")
        self.resize(1240, 820)
        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.boundary_widgets: dict[str, dict[str, object]] = {}

        self._build_ui()
        self._case_changed()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll_queue)
        self.timer.start(100)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(420)
        left_panel = QWidget()
        left_scroll.setWidget(left_panel)
        left = QVBoxLayout(left_panel)
        root.addWidget(left_scroll, 0)

        right = QVBoxLayout()
        root.addLayout(right, 1)

        tabs = QTabWidget()
        left.addWidget(tabs)
        params = QWidget()
        bounds = QWidget()
        tabs.addTab(params, "参数")
        tabs.addTab(bounds, "边界")

        self._build_params(params)
        self._build_boundaries(bounds)

        controls = QHBoxLayout()
        left.addLayout(controls)
        self.recommend_btn = QPushButton("推荐 U_ref")
        self.validate_btn = QPushButton("校验")
        self.run_btn = QPushButton("开始计算")
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        controls.addWidget(self.recommend_btn)
        controls.addWidget(self.validate_btn)
        controls.addWidget(self.run_btn)
        controls.addWidget(self.stop_btn)

        self.recommend_btn.clicked.connect(self._recommend_u)
        self.validate_btn.clicked.connect(self._validate_dialog)
        self.run_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)

        self.status = QLabel("就绪")
        left.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        left.addWidget(self.progress)

        metrics_group = QGroupBox("实时状态")
        metrics_layout = QVBoxLayout(metrics_group)
        self.metrics = QLabel("iteration: 0")
        self.metrics.setMinimumHeight(95)
        self.metrics.setTextInteractionFlags(Qt.TextSelectableByMouse)
        metrics_layout.addWidget(self.metrics)
        right.addWidget(metrics_group, 0)

        self.plot_label = QLabel("计算开始后显示速度、涡量和残差图")
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setMinimumSize(640, 360)
        self.plot_label.setStyleSheet("background: #fafafa; border: 1px solid #d0d0d0;")
        right.addWidget(self.plot_label, 1)

        output_group = QGroupBox("输出文件")
        output_layout = QVBoxLayout(output_group)
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMaximumHeight(110)
        output_layout.addWidget(self.output_text)
        right.addWidget(output_group, 0)

    def _build_params(self, parent: QWidget) -> None:
        layout = QFormLayout(parent)

        self.case = QComboBox()
        self.case.addItems(CASE_TYPES)
        self.case.currentTextChanged.connect(self._case_changed)
        layout.addRow("case_type", self.case)
        self.training_preset = QComboBox()
        self.training_preset.addItems(("T07 Couette", "T08 体力 Poiseuille", "T10 Re=100 方腔"))
        load_preset = QPushButton("载入训练参数")
        load_preset.clicked.connect(self._load_training_preset)
        layout.addRow("训练算例", self.training_preset)
        layout.addRow("", load_preset)

        self.nx = self._spin(8, 2048, 128)
        self.ny = self._spin(8, 2048, 128)
        self.l_ref = self._dspin(0.0, 1.0e9, 0.0, decimals=3)
        self.rho0 = self._dspin(1.0e-12, 1.0e6, 1.0, decimals=6)
        self.re = self._dspin(1.0e-9, 1.0e9, 100.0, decimals=6)
        self.u_ref = self._dspin(0.0, 1.0, 0.05, decimals=6, step=0.005)
        self.u_ref.valueChanged.connect(self._u_ref_changed)

        self.nu = self._dspin(0.0, 10.0, 0.0, decimals=8)
        self.force_x = self._dspin(-1.0, 1.0, 0.0, decimals=10, step=1e-6)
        self.force_y = self._dspin(-1.0, 1.0, 0.0, decimals=10, step=1e-6)
        self.initial_velocity = QComboBox()
        self.initial_velocity.addItems(("rest", "uniform", "couette"))
        self.steady_windows = self._spin(1, 100, 3)
        self.fixed_steps = QCheckBox("固定步数（瞬态，不按稳态停止）")
        self.obstacle = QComboBox()
        self.obstacle.addItems(("none", "circle", "rectangle"))
        self.obstacle_x = self._dspin(0.0, 1.0, 0.33, decimals=3, step=0.01)
        self.obstacle_y = self._dspin(0.0, 1.0, 0.50, decimals=3, step=0.01)
        self.obstacle_radius = self._dspin(0.001, 0.5, 0.08, decimals=3, step=0.01)
        self.obstacle_width = self._dspin(0.001, 1.0, 0.12, decimals=3, step=0.01)
        self.obstacle_height = self._dspin(0.001, 1.0, 0.20, decimals=3, step=0.01)

        self.tol = self._dspin(1.0e-14, 1.0, 1.0e-6, decimals=12)
        self.max_iter = self._spin(1, 10_000_000, 5000)
        self.min_iter = self._spin(0, 10_000_000, 500)
        self.report_interval = self._spin(1, 1_000_000, 100)
        self.ramp_steps = self._spin(0, 1_000_000, 1000)

        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.output_dir = QLineEdit("outputs")
        choose = QPushButton("选择")
        choose.clicked.connect(self._choose_output_dir)
        output_layout.addWidget(self.output_dir, 1)
        output_layout.addWidget(choose)

        self.save_npz = QCheckBox("save_npz")
        self.save_npz.setChecked(True)
        self.save_csv = QCheckBox("save_csv")
        self.save_csv.setChecked(True)
        self.save_png = QCheckBox("save_png")
        self.save_png.setChecked(True)

        for label, widget in (
            ("NX", self.nx),
            ("NY", self.ny),
            ("L_ref (0=auto)", self.l_ref),
            ("rho0", self.rho0),
            ("Re", self.re),
            ("U_ref", self.u_ref),
            ("ν (0=由Re计算)", self.nu),
            ("ax 格子加速度", self.force_x),
            ("ay 格子加速度", self.force_y),
            ("初始速度", self.initial_velocity),
            ("连续稳态窗口", self.steady_windows),
            ("运行模式", self.fixed_steps),
            ("obstacle", self.obstacle),
            ("obstacle_x", self.obstacle_x),
            ("obstacle_y", self.obstacle_y),
            ("obstacle_radius", self.obstacle_radius),
            ("obstacle_width", self.obstacle_width),
            ("obstacle_height", self.obstacle_height),
            ("tol", self.tol),
            ("max_iter", self.max_iter),
            ("min_iter", self.min_iter),
            ("report_interval", self.report_interval),
            ("ramp_steps", self.ramp_steps),
            ("output_dir", output_row),
            ("", self.save_npz),
            ("", self.save_csv),
            ("", self.save_png),
        ):
            layout.addRow(label, widget)

    def _build_boundaries(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        for side in ("left", "right", "bottom", "top"):
            group = QGroupBox(side)
            grid = QGridLayout(group)
            btype = QComboBox()
            btype.addItems(BOUNDARY_TYPES)
            ux = self._dspin(-1.0, 1.0, 0.0, decimals=6, step=0.005)
            uy = self._dspin(-1.0, 1.0, 0.0, decimals=6, step=0.005)
            rho = self._dspin(1.0e-12, 1.0e6, 1.0, decimals=6)
            rb = self._dspin(0.0, 1.0, 1.0, decimals=3, step=0.05)
            grid.addWidget(QLabel("type"), 0, 0)
            grid.addWidget(btype, 0, 1, 1, 4)
            for col, (name, widget) in enumerate((("ux", ux), ("uy", uy), ("rho", rho), ("rb", rb))):
                grid.addWidget(QLabel(name), 1, col)
                grid.addWidget(widget, 2, col)
            layout.addWidget(group)
            self.boundary_widgets[side] = {"type": btype, "ux": ux, "uy": uy, "rho": rho, "rb": rb}
        layout.addStretch(1)

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        return widget

    @staticmethod
    def _dspin(
        minimum: float,
        maximum: float,
        value: float,
        decimals: int = 6,
        step: float = 0.1,
    ) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setSingleStep(step)
        widget.setValue(value)
        return widget

    def _choose_output_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择输出目录", str(Path.cwd()))
        if selected:
            self.output_dir.setText(selected)

    def _case_changed(self) -> None:
        case_type = self.case.currentText()
        defaults = default_boundaries_for_case(case_type, self.u_ref.value())
        self._preset_u_ref = self.u_ref.value()
        if case_type == "cylinder_flow":
            self.obstacle.setCurrentText("circle")
        for side, bc in defaults.items():
            widgets = self.boundary_widgets.get(side)
            if not widgets:
                continue
            widgets["type"].setCurrentText(bc.type)
            widgets["ux"].setValue(bc.ux)
            widgets["uy"].setValue(bc.uy)
            widgets["rho"].setValue(bc.rho)
            widgets["rb"].setValue(bc.rb)

    def _u_ref_changed(self) -> None:
        old = getattr(self, "_preset_u_ref", self.u_ref.value())
        before = default_boundaries_for_case(self.case.currentText(), old)
        after = default_boundaries_for_case(self.case.currentText(), self.u_ref.value())
        for side, widgets in self.boundary_widgets.items():
            bc = before[side]
            if widgets["type"].currentText() == bc.type and abs(widgets["ux"].value()-bc.ux)<1e-12:
                widgets["ux"].setValue(after[side].ux)
        self._preset_u_ref = self.u_ref.value()

    def _load_training_preset(self) -> None:
        index = self.training_preset.currentIndex()
        case = ("couette_flow", "force_poiseuille", "lid_driven_cavity")[index]
        self.case.setCurrentText(case)
        self.nx.setValue(64); self.ny.setValue(64 if index == 2 else 32)
        self.l_ref.setValue(0); self.rho0.setValue(1); self.re.setValue(100)
        self.u_ref.setValue(.04 if index == 2 else .02)
        self.nu.setValue(0 if index == 2 else .1)
        self.force_x.setValue(8*.1*.02/32**2 if index == 1 else 0)
        self.force_y.setValue(0); self.initial_velocity.setCurrentText("rest")
        self.obstacle.setCurrentText("none")
        self.tol.setValue(1e-7);self.max_iter.setValue(50000 if index == 2 else 30000)
        self.min_iter.setValue(1000); self.report_interval.setValue(200)
        self.ramp_steps.setValue(500 if index == 2 else 0)
        self.steady_windows.setValue(3);self.fixed_steps.setChecked(False)
        self.output_dir.setText(str(Path(__file__).parent / "training_runs" / case))
        self._case_changed()

    def _recommend_u(self) -> None:
        rec = recommend_parameters(self.case.currentText(), self.nx.value(), self.ny.value(), self.re.value(),
            l_ref=characteristic_length(self._config()))
        self.u_ref.setValue(float(rec["U_ref"]))
        QMessageBox.information(
            self,
            "推荐参数",
            "\n".join(
                [
                    f"U_ref = {rec['U_ref']:.6g}",
                    f"tau = {rec['tau']:.6g}",
                    f"Ma = {rec['Ma']:.6g}",
                    f"N_required = {rec['N_required']:.2f}",
                    f"status = {rec['status']}",
                ]
            ),
        )

    def _config(self) -> SimulationConfig:
        grid = GridConfig(
            NX=self.nx.value(),
            NY=self.ny.value(),
            L_ref=None if self.l_ref.value() <= 0 else self.l_ref.value(),
            obstacle_type=self.obstacle.currentText(),
            obstacle_x=self.obstacle_x.value(),
            obstacle_y=self.obstacle_y.value(),
            obstacle_radius=self.obstacle_radius.value(),
            obstacle_width=self.obstacle_width.value(),
            obstacle_height=self.obstacle_height.value(),
        )
        flow = FlowConfig(rho0=self.rho0.value(), U_ref=self.u_ref.value(), Re=self.re.value(),
            nu_lattice=self.nu.value() if self.nu.value() > 0 else None,
            body_force_x=self.force_x.value(), body_force_y=self.force_y.value(),
            initial_velocity=self.initial_velocity.currentText())
        convergence = ConvergenceConfig(
            tol=self.tol.value(),
            max_iter=self.max_iter.value(),
            min_iter=self.min_iter.value(),
            report_interval=self.report_interval.value(),
            ramp_steps=self.ramp_steps.value(),
            consecutive_reports=self.steady_windows.value(),
            steady=not self.fixed_steps.isChecked(),
        )
        output = OutputConfig(
            output_dir=self.output_dir.text(),
            save_npz=self.save_npz.isChecked(),
            save_csv=self.save_csv.isChecked(),
            save_png=self.save_png.isChecked(),
        )
        boundaries = {}
        for side, widgets in self.boundary_widgets.items():
            boundaries[side] = BoundaryConfig(
                type=widgets["type"].currentText(),
                ux=widgets["ux"].value(),
                uy=widgets["uy"].value(),
                rho=widgets["rho"].value(),
                rb=widgets["rb"].value(),
            )
        return SimulationConfig(
            case_type=self.case.currentText(),
            grid=grid,
            flow=flow,
            convergence=convergence,
            output=output,
            boundaries=boundaries,
        )

    def _validate_dialog(self) -> bool:
        try:
            config = self._config()
            errors, warnings, transport = validate_config(config)
        except Exception as exc:
            QMessageBox.critical(self, "参数错误", str(exc))
            return False
        lines = [
            f"Re = {transport['Re']:.6g}",
            f"Ma = {transport['Ma']:.6g}",
            f"nu_lattice = {transport['nu_lattice']:.6g}",
            f"tau = {transport['tau']:.6g}",
            f"omega = {transport['omega']:.6g}",
        ]
        if errors:
            lines.extend(["", "错误:"] + [f"- {item}" for item in errors])
            QMessageBox.critical(self, "校验失败", "\n".join(lines))
            return False
        if warnings:
            lines.extend(["", "警告:"] + [f"- {item}" for item in warnings])
        QMessageBox.information(self, "校验结果", "\n".join(lines))
        return True

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        config = self._config()
        errors, warnings, transport = validate_config(config)
        if errors:
            QMessageBox.critical(self, "校验失败", "\n".join(errors))
            return
        if warnings:
            reply = QMessageBox.question(
                self,
                "存在警告",
                "\n".join(warnings) + "\n\n是否继续运行？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self.stop_event.clear()
        self.progress.setValue(0)
        self.output_text.clear()
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText("正在计算...")
        self.metrics.setText(
            f"Re={transport['Re']:.4g}, Ma={transport['Ma']:.4g}, "
            f"tau={transport['tau']:.4g}, omega={transport['omega']:.4g}"
        )
        self.worker = threading.Thread(target=self._worker, args=(config,), daemon=True)
        self.worker.start()

    def _stop(self) -> None:
        self.stop_event.set()
        self.status.setText("收到停止请求，等待当前报告点结束...")

    def _worker(self, config: SimulationConfig) -> None:
        try:
            solver = LBMSolver(config)
            for report in solver.run():
                self.queue.put(("report", (report, solver.field_snapshot(), list(solver.history), config.convergence.max_iter)))
                if self.stop_event.is_set():
                    solver.cancel()
                    break
            self.queue.put(("done", solver.finalize(save_outputs=True)))
        except Exception as exc:
            self.queue.put(("error", exc))

    def _poll_queue(self) -> None:
        latest_report = None
        terminal = None
        while True:
            try:
                kind, payload = self.queue.get_nowait()
            except queue.Empty:
                break
            if kind == "report":
                latest_report = payload
            else:
                terminal = kind, payload
        # Draw at most one snapshot per timer tick; do not block the event loop
        # while replaying a backlog of every intermediate frame.
        if latest_report is not None:
            self._render_report(*latest_report)
        if terminal is not None:
            kind, payload = terminal
            if kind == "done":
                self._render_done(payload)
            elif kind == "error":
                self.run_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.status.setText("计算失败")
                QMessageBox.critical(self, "计算失败", str(payload))

    def _render_report(self, report, fields: dict[str, np.ndarray], history: list[dict[str, float]], max_iter: int) -> None:
        self.progress.setValue(min(100, int(100 * report.iteration / max(1, max_iter))))
        self.status.setText(report.message)
        self.metrics.setText(
            f"iteration: {report.iteration} | residual: {report.residual:.3e}\n"
            f"q: {report.q:.3g} | q_avg: {report.q_avg:.3g} | max_u: {report.max_velocity:.5g}\n"
            f"mass drift: {report.mass_drift:.3e} | balance error: {report.mass_balance_error:.3e}\n"
            f"rho_min: {report.min_density:.6g} | Ma_max: {report.max_mach:.5g}\n"
            f"status: {report.message}"
        )
        self._draw(fields, history)

    def _draw(self, fields: dict[str, np.ndarray], history: list[dict[str, float]]) -> None:
        fig = Figure(figsize=(8, 6), dpi=120)
        ax_speed = fig.add_subplot(221)
        ax_vort = fig.add_subplot(222)
        ax_res = fig.add_subplot(212)

        solid = fields["solid_mask"]
        im = ax_speed.imshow(fields["speed"], origin="lower", cmap="viridis", interpolation="nearest")
        fig.colorbar(im, ax=ax_speed, label="speed (lattice units)")
        ax_speed.set_title("Velocity magnitude")
        finite = fields["vorticity"][np.isfinite(fields["vorticity"])]
        vmax = max(float(np.max(np.abs(finite))) if finite.size else 0., 1e-12)
        im = ax_vort.imshow(fields["vorticity"], origin="lower", cmap="coolwarm", interpolation="nearest", vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax_vort, label="vorticity (1 / lattice time)")
        ax_vort.set_title("Vorticity")
        for ax in (ax_speed, ax_vort):
            ax.set_xlabel("x (lattice units)");ax.set_ylabel("y (lattice units)")
        if np.any(solid):
            mask = np.ma.masked_where(~solid, solid)
            ax_speed.imshow(mask, origin="lower", cmap="gray_r", vmin=0, vmax=1, alpha=0.8, interpolation="nearest")
            ax_vort.imshow(mask, origin="lower", cmap="gray_r", vmin=0, vmax=1, alpha=0.8, interpolation="nearest")
        if history:
            it = [row["iteration"] for row in history]
            ax_res.semilogy(it, [row["residual"] for row in history], label="residual")
            ax_res.semilogy(it, [row["mass_drift"] for row in history], label="mass drift")
            ax_res.legend(loc="best")
        ax_res.set_title("Residual history")
        ax_res.set_xlabel("iteration")
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        image = Image.open(buffer)
        image.thumbnail((max(400, self.plot_label.width() - 20), max(300, self.plot_label.height() - 20)))
        png = io.BytesIO()
        image.save(png, format="PNG")
        pixmap = QPixmap()
        pixmap.loadFromData(png.getvalue(), "PNG")
        self.plot_label.setPixmap(pixmap)

    def _render_done(self, result) -> None:
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setValue(100)
        self.status.setText(f"停止原因: {result.stop_reason} | {result.message}")
        self.output_text.clear()
        for name, path in result.output_files.items():
            self.output_text.append(f"{name}: {path}")


def main() -> int:
    app = QApplication(sys.argv)
    window = LBMQtWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
