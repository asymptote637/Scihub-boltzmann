"""Dedicated desktop workflow for configuring, running, and validating LBM cases."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
from PySide6.QtCore import QProcess, QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox,
    QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from config import BOUNDARY_TYPES, CASE_TYPES, SimulationConfig, validate_config
from workbench.project import (PRESETS, Workspace, atomic_json, config_from_dict,
                               load_case, now, preset, read_json, unique_id)
from workbench.plots import draw_comparison, draw_fields, draw_validation
from workbench.validation import reference_kind
from workbench.theme import BoundarySketch, STYLESHEET

STATES = {"created": "待启动", "starting": "启动中", "running": "计算中", "finalizing": "归档中",
          "converged": "已收敛", "completed_steps": "完成固定步数", "max_iter": "步数耗尽 · 未收敛",
          "cancelled": "已取消", "diverged": "发散", "failed": "执行失败"}
VALIDATION = {"passed": "通过", "failed": "未通过", "not_converged": "未收敛",
              "not_applicable": "参考解不适用"}

# Keep numerical input as text to preserve small forces and full float precision.
LABELS = {
    "NX": "流向格点 NX", "NY": "横向格点 NY", "L_ref": "参考长度（留空自动）",
    "obstacle_type": "障碍物", "obstacle_x": "中心 x / 域宽", "obstacle_y": "中心 y / 域高",
    "obstacle_radius": "半径 / 较短边", "obstacle_width": "宽度 / 域宽", "obstacle_height": "高度 / 域高",
    "rho0": "初始密度 ρ₀", "U_ref": "参考速度 U_ref", "Re": "输入 Re（显式 ν 时忽略）",
    "nu_lattice": "格子黏度 ν（留空由 Re）", "body_force_x": "格子加速度 ax", "body_force_y": "格子加速度 ay",
    "initial_velocity": "初始速度", "tol": "残差阈值", "max_iter": "最大步数", "min_iter": "最少步数",
    "report_interval": "报告间隔（步）", "ramp_steps": "壁速渐启（步）", "consecutive_reports": "连续稳态窗口数",
    "steady": "按稳态停止（取消勾选为固定步数）", "mass_tolerance": "质量诊断阈值", "max_mach": "实际 Ma 上限",
    "physical_mode": "使用物理单位映射", "L_phys": "物理长度 L", "U_phys": "物理速度 U",
    "nu_phys": "物理运动黏度 ν", "save_png": "自动导出四类流场及残差 PNG",
    "ux": "u（格子速度）", "uy": "v（格子速度）", "rho": "密度 ρ", "rb": "兼容字段 rb",
}
ENUMS = {"obstacle_type": ("none", "circle", "rectangle"),
         "initial_velocity": ("rest", "uniform", "couette"), "type": BOUNDARY_TYPES}
FIELD_DEFAULTS = asdict(SimulationConfig())


class PlotView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(9, 6), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas, 1)


class WorkbenchWindow(QMainWindow):
    def __init__(self, workspace: Path | None = None, remember: bool = True) -> None:
        super().__init__()
        self.settings = QSettings("LocalLBM", "SimulationWorkbench") if remember else None
        default = Path(__file__).resolve().parents[2] / "lbm_workspace"
        chosen = workspace or (Path(self.settings.value("workspace", str(default))) if self.settings else default)
        self.workspace = Workspace(chosen)
        self.setWindowTitle("LBM 仿真工作台 · 等温训练 1.0")
        self.resize(1440, 940)
        available = self.screen().availableGeometry()
        self.compact = available.width() < 1280 or available.height() < 900
        compact_style = """
            QWidget { font-size: 12px; }
            QLineEdit, QComboBox { padding: 4px 7px; min-height: 16px; }
            QPushButton { padding: 6px 9px; }
            QTabBar::tab { padding: 8px 12px; margin-bottom: 4px; }
            QGroupBox { padding: 13px 9px 9px; margin-top: 14px; }
            QWidget#Sidebar QListWidget::item { padding: 8px 7px; }
            QWidget#Sidebar QLabel { font-size: 11px; }
            QPlainTextEdit { padding: 6px; font-size: 11px; }
            QLabel#AppTitle { font-size: 22px; }
        """ if self.compact else ""
        self.setStyleSheet(STYLESHEET + compact_style)
        self.process = None
        self.active_run = None
        self.active_config = None
        self._close_when_done = False
        self._buffer = b""
        self._latest_progress = None
        self._history = []
        self._records = []
        self._selected_record = None
        self._base = asdict(preset("couette"))
        self.widgets = {}
        self.preflight_timer = QTimer(self)
        self.preflight_timer.setSingleShot(True)
        self.preflight_timer.timeout.connect(self.preflight)
        self._build()
        self.apply_preset("couette")
        self.refresh_library()
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._draw_preview)
        self.preview_timer.start(800)
        self.resize(min(1440, available.width()-24), min(940, available.height()-40))

    def _button(self, text: str, callback, layout) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(lambda checked=False: self._guard(callback))
        layout.addWidget(button)
        return button

    def _guard(self, callback) -> None:
        try:
            callback()
        except (ValueError, OSError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, "操作未完成", str(exc))

    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12 if self.compact else 20, 12, 12 if self.compact else 20, 8)
        root.setSpacing(9)
        heading = QHBoxLayout()
        title = QLabel("LBM 仿真工作台")
        title.setObjectName("AppTitle")
        heading.addWidget(title)
        badge = QLabel("等温训练  /  V1.0")
        badge.setObjectName("Badge")
        heading.addSpacing(14)
        heading.addWidget(badge)
        heading.addStretch()
        self.workspace_button = self._button("切换工作区", self.choose_workspace, heading)
        self._button("使用说明", self.open_help, heading)
        root.addLayout(heading)
        self.workspace_label = QLabel(str(self.workspace.root))
        self.workspace_label.setObjectName("WorkspacePath")
        self.workspace_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.workspace_label)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)
        sidebar = QWidget()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(210 if self.compact else 270)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(12, 18, 12, 15)
        side.setSpacing(8 if self.compact else 12)
        sidebar_title = QLabel("训练空间")
        sidebar_title.setObjectName("SidebarTitle")
        side.addWidget(sidebar_title)
        side.addWidget(QLabel("选择一个问题，开始计算"))
        self.presets = QListWidget()
        self.presets.setFixedHeight(174 if self.compact else 216)
        for key, (name, description) in PRESETS.items():
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, key)
            item.setToolTip(description)
            self.presets.addItem(item)
        side.addWidget(self.presets)
        self.preset_button = self._button("载入选中预设", self.load_selected_preset, side)
        self.presets.itemDoubleClicked.connect(lambda item: self._guard(self.load_selected_preset))
        self.case_description = QLabel()
        self.case_description.setWordWrap(True)
        side.addWidget(self.case_description)
        side.addSpacing(6)
        side.addWidget(QLabel("我的算例  /  已保存版本"))
        self.saved_cases = QListWidget()
        self.saved_cases.itemDoubleClicked.connect(lambda item: self._guard(self.load_selected_case))
        side.addWidget(self.saved_cases, 1)
        case_actions = QHBoxLayout()
        self.load_button = self._button("载入配置", self.load_selected_case, case_actions)
        self.import_button = self._button("导入文件", self.import_case, case_actions)
        self.import_button.setToolTip("导入工作台算例或求解器 config.json")
        side.addLayout(case_actions)
        self._button("刷新档案", self.refresh_library, side)
        note = QLabel("结果状态、收敛与基准验证分别记录。\n原始宏观场和 CSV 自动保留。")
        note.setWordWrap(True)
        side.addWidget(note)
        splitter.addWidget(sidebar)

        self.pages = QTabWidget()
        self.pages.setDocumentMode(True)
        splitter.addWidget(self.pages)
        splitter.setStretchFactor(1, 1)
        self.editor_page = QWidget()
        editor = QVBoxLayout(self.editor_page)
        editor.setContentsMargins(12 if self.compact else 18, 12, 12 if self.compact else 18, 12)
        editor.setSpacing(8 if self.compact else 11)
        self.pages.addTab(self.editor_page, "01   配置算例")
        name_row = QFormLayout()
        self.case_name = QLineEdit()
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("研究目的、变化的参数、待检验的问题")
        self.case_type = QComboBox()
        self.case_type.addItems(CASE_TYPES)
        self.case_type.currentTextChanged.connect(self._schedule_preflight)
        name_row.addRow("算例名称", self.case_name)
        name_row.addRow("实验备注", self.notes)
        editor.addLayout(name_row)
        hint = QLabel("参考尺度与驱动分别设置；完整切换问题请载入预设。默认使用格子单位。")
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        editor.addWidget(hint)
        config_tabs = QTabWidget()
        config_tabs.setObjectName("ConfigTabs")
        editor.addWidget(config_tabs, 1)
        groups = [
            ("网格与流动", [("网格与参考尺度", "grid", ["NX", "NY", "L_ref"]),
                         ("流动与驱动", "flow", ["Re", "nu_lattice", "body_force_x", "body_force_y", "initial_velocity"])]),
            ("停止与输出", [("收敛 / 瞬态", "convergence", list(self._base["convergence"])),
                          ("保存", "output", ["save_png"])]),
            ("几何与单位", [("障碍物（基准之外）", "grid", [k for k in self._base["grid"] if k.startswith("obstacle")]),
                           ("物理单位映射（ν 不可同时显式指定）", "flow", ["physical_mode", "L_phys", "U_phys", "nu_phys"])])]
        for tab_name, sections in groups:
            area = QScrollArea()
            area.setWidgetResizable(True)
            content = QWidget()
            content_layout = QVBoxLayout(content)
            content_layout.setContentsMargins(0, 0, 0, 0)
            layout = QHBoxLayout()
            layout.setSpacing(15)
            content_layout.addLayout(layout)
            for title, group, keys in sections:
                box = QGroupBox(title)
                form = QFormLayout(box)
                form.setVerticalSpacing(8)
                form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
                for key in keys:
                    self._add_field(form, group, key, self._base[group][key])
                if tab_name == "网格与流动" and group == "grid":
                    for key in ("rho0", "U_ref"):
                        self._add_field(form, "flow", key, self._base["flow"][key])
                if group == "output":
                    form.addRow("算例类型（不重设边界）", self.case_type)
                if tab_name == "网格与流动" and group == "grid":
                    column = QVBoxLayout()
                    column.setSpacing(4)
                    column.addWidget(box)
                    self.boundary_sketch = BoundarySketch()
                    if self.compact:
                        self.boundary_sketch.setMinimumHeight(140)
                    column.addWidget(self.boundary_sketch, 1)
                    layout.addLayout(column, 1)
                else:
                    layout.addWidget(box, 1, Qt.AlignTop)
            content_layout.addStretch(1)
            area.setWidget(content)
            config_tabs.addTab(area, tab_name)
        area = QScrollArea()
        area.setWidgetResizable(True)
        content = QWidget()
        grid = QGridLayout(content)
        for i, side in enumerate(("left", "right", "bottom", "top")):
            box = QGroupBox({"left": "左边界", "right": "右边界", "bottom": "下边界", "top": "上边界"}[side])
            form = QFormLayout(box)
            for key, value in self._base["boundaries"][side].items():
                self._add_field(form, f"boundaries.{side}", key, value)
            grid.addWidget(box, i//2, i % 2)
        area.setWidget(content)
        config_tabs.addTab(area, "四面边界")
        self.preflight_text = QPlainTextEdit()
        self.preflight_text.setReadOnly(True)
        self.preflight_text.setObjectName("Preflight")
        self.preflight_text.setMaximumHeight(95 if self.compact else 115)
        editor.addWidget(self.preflight_text)
        actions = QHBoxLayout()
        self._button("检查参数", self.preflight, actions)
        self._button("保存为新算例版本", self.save_current_case, actions)
        self.run_button = self._button("开始仿真", self.start_run, actions)
        self.run_button.setObjectName("Primary")
        editor.addLayout(actions)

        monitor = QWidget()
        monitor_layout = QVBoxLayout(monitor)
        monitor_layout.setContentsMargins(18, 16, 18, 16)
        monitor_layout.setSpacing(12)
        self.pages.addTab(monitor, "02   监测运行")
        actions = QHBoxLayout()
        self.status = QLabel("就绪 · 选择预设并开始仿真")
        self.status.setObjectName("RunStatus")
        actions.addWidget(self.status, 1)
        self.stop_button = self._button("取消并保存", self.cancel_run, actions)
        self.stop_button.setObjectName("Cancel")
        self.stop_button.setEnabled(False)
        self._button("导出监测图", lambda: self.export_figure(self.monitor_plot), actions)
        monitor_layout.addLayout(actions)
        self.progress = QProgressBar()
        monitor_layout.addWidget(self.progress)
        self.metrics = QLabel("尚无运行数据")
        self.metrics.setObjectName("Metrics")
        self.metrics.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.metrics.setWordWrap(True)
        monitor_layout.addWidget(self.metrics)
        self.monitor_plot = PlotView()
        monitor_layout.addWidget(self.monitor_plot, 1)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        self.log.setMaximumHeight(90)
        monitor_layout.addWidget(self.log)

        archive = QWidget()
        layout = QVBoxLayout(archive)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        self.pages.addTab(archive, "03   结果档案")
        self.runs_table = QTableWidget(0, 7)
        self.runs_table.setHorizontalHeaderLabels(["算例 / 运行", "网格", "实际 Re", "停止原因", "步数", "基准验证", "误差"])
        self.runs_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.runs_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.runs_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.runs_table.setAlternatingRowColors(True)
        self.runs_table.verticalHeader().setVisible(False)
        self.runs_table.verticalHeader().setDefaultSectionSize(42)
        self.runs_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.runs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.runs_table.setMinimumHeight(115)
        self.runs_table.setMaximumHeight(180)
        self.runs_table.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(self.runs_table)
        actions = QHBoxLayout()
        self._button("查看选中结果", self.view_selected, actions)
        self.reuse_button = self._button("用此配置新建运行", self.reuse_selected, actions)
        self._button("对比选中 2–4 项", self.compare_selected, actions)
        self._button("打开结果目录", self.open_selected_directory, actions)
        self._button("导出当前图", self.export_archive_figure, actions)
        layout.addLayout(actions)
        self.result_info = QPlainTextEdit()
        self.result_info.setReadOnly(True)
        self.result_info.setMaximumHeight(125)
        layout.addWidget(self.result_info)
        self.result_tabs = QTabWidget()
        self.result_plot, self.validation_plot, self.comparison_plot = PlotView(), PlotView(), PlotView()
        self.diagnostics_plot = PlotView()
        self.result_tabs.addTab(self.result_plot, "流场分布")
        self.result_tabs.addTab(self.validation_plot, "基准对照")
        self.result_tabs.addTab(self.comparison_plot, "多次运行比较")
        self.result_tabs.addTab(self.diagnostics_plot, "收敛与中心线")
        layout.addWidget(self.result_tabs, 1)

    def _add_field(self, form: QFormLayout, group: str, key: str, value) -> None:
        declared_default = FIELD_DEFAULTS
        for part in f"{group}.{key}".split("."):
            declared_default = declared_default[part]
        if key in ENUMS:
            widget = QComboBox()
            widget.addItems(ENUMS[key])
            widget.currentTextChanged.connect(self._schedule_preflight)
        elif type(value) is bool:
            widget = QCheckBox()
            widget.toggled.connect(self._schedule_preflight)
        else:
            widget = QLineEdit()
            widget.setMinimumWidth(95)
            if value is None:
                widget.setPlaceholderText("自动")
            widget.textChanged.connect(self._schedule_preflight)
        widget.setToolTip(f"{group}.{key}")
        self.widgets[f"{group}.{key}"] = (widget, type(declared_default))
        form.addRow(LABELS.get(key, key), widget)

    def _schedule_preflight(self, *args) -> None:
        self.preflight_timer.start(250)

    def set_config(self, config, name: str, notes: str = "") -> None:
        self._base = asdict(config)
        self.case_name.setText(name)
        self.notes.setText(notes)
        self.case_type.setCurrentText(config.case_type)
        for dotted, (widget, _) in self.widgets.items():
            value = self._base
            for key in dotted.split("."):
                value = value[key]
            if isinstance(widget, QComboBox):
                widget.setCurrentText(value)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(value)
            else:
                widget.setText("" if value is None else str(value))
        self.preflight()
        self.pages.setCurrentIndex(0)

    def current_config(self):
        data = json.loads(json.dumps(self._base))
        data["case_type"] = self.case_type.currentText()
        for dotted, (widget, kind) in self.widgets.items():
            if isinstance(widget, QComboBox):
                value = widget.currentText()
            elif isinstance(widget, QCheckBox):
                value = widget.isChecked()
            else:
                raw = widget.text().strip()
                try:
                    value = None if not raw and kind is type(None) else (int(raw) if kind is int else float(raw))
                except ValueError:
                    raise ValueError(f"{dotted}：请输入{'整数' if kind is int else '数值'}")
            target = data
            parts = dotted.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
        return config_from_dict(data)

    def preflight(self) -> bool:
        try:
            config = self.current_config()
            _, warnings, t = validate_config(config)
            _, ref_note = reference_kind(config)
            lines = [f"参数有效  |  实际参考 Re={t['Re']:.6g}   ν={t['nu_lattice']:.6g}   τ={t['tau']:.6g}   设计 Ma={t['Ma']:.5g}",
                     f"参考长度={t['L_ref']:g} 格子单位  |  {config.grid.NX} × {config.grid.NY}  |  "
                     f"{'稳态判据' if config.convergence.steady else '固定步数，不判稳态'}",
                     f"实际驱动：上壁 u={config.boundaries['top'].ux:g}，ax={config.flow.body_force_x:g}，"
                     f"入口/出口 ρ={config.boundaries['left'].rho:g}/{config.boundaries['right'].rho:g}",
                     f"可用对照：{ref_note}"]
            lines += [f"提示：{w}" for w in warnings]
            if config.grid.NX*config.grid.NY >= 512**2:
                lines.append("提示：较大网格会增加内存与运行时间；预览抽样至每边不超过 256 点，完整结果保留。")
            self.preflight_text.setPlainText("\n".join(lines))
            self.boundary_sketch.set_config(config)
            self.boundary_sketch.setToolTip(ref_note)
            return True
        except (ValueError, TypeError) as exc:
            self.preflight_text.setPlainText(f"参数未通过：\n{exc}")
            return False

    def apply_preset(self, key: str) -> None:
        if self.process is not None:
            return
        self.set_config(preset(key), PRESETS[key][0])
        self.case_description.setText(PRESETS[key][1])
        for i in range(self.presets.count()):
            if self.presets.item(i).data(Qt.UserRole) == key:
                self.presets.setCurrentRow(i)

    def load_selected_preset(self) -> None:
        item = self.presets.currentItem()
        if item:
            self.apply_preset(item.data(Qt.UserRole))

    def save_current_case(self) -> Path:
        path = self.workspace.save_case(self.current_config(), self.case_name.text(), self.notes.text())
        self.refresh_library()
        self.statusBar().showMessage(f"已保存新版本：{path}", 15000)
        return path

    def load_selected_case(self) -> None:
        if self.process is not None:
            return
        item = self.saved_cases.currentItem()
        if item:
            self.load_case_path(Path(item.data(Qt.UserRole)))

    def load_case_path(self, path: Path) -> None:
        document = load_case(path)
        self.set_config(config_from_dict(document["config"]), document["name"], document["notes"])
        self.case_description.setText(f"载入配置：{path.name}；运行将使用新目录。")

    def import_case(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "导入算例或求解器 config.json", str(self.workspace.root), "JSON (*.json)")
        if path:
            self.load_case_path(Path(path))

    def choose_workspace(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择独立工作区", str(self.workspace.root))
        if path:
            self.workspace = Workspace(Path(path))
            self.workspace_label.setText(str(self.workspace.root))
            if self.settings:
                self.settings.setValue("workspace", str(self.workspace.root))
            self._selected_record = None
            self.result_info.clear()
            for plot in (self.result_plot, self.validation_plot, self.comparison_plot, self.diagnostics_plot):
                plot.figure.clear()
                plot.canvas.draw_idle()
            self.refresh_library()

    def _set_busy(self, busy: bool) -> None:
        for widget in (self.editor_page, self.workspace_button, self.load_button, self.import_button,
                       self.preset_button, self.reuse_button, self.presets, self.saved_cases):
            widget.setEnabled(not busy)
        self.stop_button.setEnabled(busy)

    def start_run(self) -> Path:
        if self.process is not None:
            raise ValueError("当前任务尚未结束")
        config = self.current_config()
        root = self.workspace.create_run(config, self.case_name.text(), self.notes.text())
        self.active_run, self.active_config = root, config
        self._buffer, self._latest_progress, self._history = b"", None, []
        self.log.clear()
        self.log.appendPlainText(f"输出目录：{root}")
        self.log.appendPlainText(self.preflight_text.toPlainText())
        self.monitor_plot.figure.clear()
        self.monitor_plot.canvas.draw_idle()
        self.progress.setValue(0)
        self.metrics.setText("初始化计算进程…")
        self.status.setText("启动中")
        self._set_busy(True)
        self.pages.setCurrentIndex(1)
        process = QProcess(self)
        self.process = process
        process.setProgram(sys.executable)
        process.setArguments(["-u", str(Path(__file__).with_name("worker.py")), "--run-dir", str(root)])
        process.setWorkingDirectory(str(root))
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._process_error)
        process.start()
        self.refresh_library()
        return root

    def cancel_run(self) -> None:
        if self.process is not None and self.active_run:
            (self.active_run / "cancel.request").touch()
            self.status.setText("已请求取消 · 等待当前步与结果归档")
            self.stop_button.setEnabled(False)

    def _read_stdout(self) -> None:
        if self.process is None:
            return
        self._buffer += bytes(self.process.readAllStandardOutput())
        lines = self._buffer.split(b"\n")
        self._buffer = lines.pop()
        for raw in lines:
            if not raw.strip():
                continue
            try:
                message = json.loads(raw.decode("utf-8"))
                event = message.get("event")
                if event == "progress":
                    self._latest_progress = message
                    r = message["report"]
                    self._history.append(r)
                    # Display history is decimated only; full diagnostics are archived.
                    if len(self._history) > 1000:
                        self._history = self._history[::2]
                    self._show_metrics(r, message["elapsed_seconds"])
                elif event == "state":
                    self.status.setText(STATES.get(message["state"], message["state"]))
                elif event == "failed":
                    self.log.appendPlainText(message["error"])
                elif event == "finished":
                    self.log.appendPlainText(f"归档完成；停止原因：{message['state']}；验证：{message['validation']}")
            except (ValueError, KeyError, TypeError) as exc:
                self.log.appendPlainText(f"进度消息无法解析：{exc}")

    def _read_stderr(self) -> None:
        if self.process is not None:
            text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
            if text:
                with (self.active_run / "process_stderr.log").open("a", encoding="utf-8") as handle:
                    handle.write(text)
                self.log.appendPlainText(text[-2000:])

    def _show_metrics(self, report: dict, elapsed: float) -> None:
        def fmt(key):
            value = report.get(key)
            return "—" if value is None else f"{value:.4g}"
        maximum = self.active_config.convergence.max_iter
        self.progress.setValue(min(100, int(100*report["iteration"]/maximum)))
        self.metrics.setText(f"步数 {report['iteration']} / {maximum}   |   已用 {elapsed:.1f} s   |   残差 {fmt('residual')}\n"
            f"质量漂移 {fmt('mass_drift')}   |   离散收支误差 {fmt('mass_balance_error')}   |   最大速度 {fmt('max_velocity')}\n"
            f"密度 {fmt('min_density')} ～ {fmt('max_density')}   |   实际最大 Ma {fmt('max_mach')}")

    def _draw_preview(self) -> None:
        if not self._latest_progress or not self.active_run or self.pages.currentIndex() != 1:
            return
        self._latest_progress = None
        path = self.active_run / "preview.npz"
        try:
            with np.load(path, allow_pickle=False) as data:
                fields = {k: data[k].copy() for k in ("rho", "ux", "uy", "speed", "vorticity", "solid_mask")}
                stride = int(data["stride"])
            draw_fields(self.monitor_plot.figure, self.active_config, fields, self._history, stride)
            self.monitor_plot.canvas.draw_idle()
        except (OSError, ValueError) as exc:
            self.log.appendPlainText(f"预览不可用：{exc}")

    def _process_error(self, error) -> None:
        if error == QProcess.FailedToStart:
            atomic_json(self.active_run / "job.json", dict(state="failed", error=self.process.errorString(), finished_at=now()))
            self._finished(-1, QProcess.CrashExit)

    def _finished(self, exit_code: int, exit_status) -> None:
        if self.process is None:
            return
        self._read_stdout()
        self._read_stderr()
        self._latest_progress = None
        process = self.process
        self.process = None
        process.deleteLater()
        job = read_json(self.active_run / "job.json")
        if job.get("state") in {"created", "starting", "running", "finalizing"}:
            job.update(state="failed", error=f"计算进程未完成归档：exit={exit_code}", finished_at=now())
            atomic_json(self.active_run / "job.json", job)
        self.status.setText(STATES.get(job.get("state"), "未知状态"))
        if job.get("error"):
            self.log.appendPlainText(job["error"])
        self._set_busy(False)
        self.refresh_library()
        for i, record in enumerate(self._records):
            if record["path"] == self.active_run:
                self.runs_table.selectRow(i)
                if record["summary"] and not self._close_when_done:
                    self._guard(self.view_selected)
                    self.pages.setCurrentIndex(2)
                break
        if self._close_when_done:
            self.close()

    def refresh_library(self) -> None:
        self.saved_cases.clear()
        for path, document, error in self.workspace.cases():
            item = QListWidgetItem(f"损坏：{path.name}" if error else document["name"])
            item.setData(Qt.UserRole, str(path))
            item.setToolTip(error or f"{document['created_at']}\n{path.name}")
            self.saved_cases.addItem(item)
        self._records = self.workspace.runs()
        self.runs_table.setRowCount(len(self._records))
        for i, r in enumerate(self._records):
            if r["error"]:
                values = [r["path"].name, "—", "—", "档案损坏", "—", "—", "—"]
            else:
                c = config_from_dict(r["request"]["config"])
                _, _, t = validate_config(c)
                status = r["job"].get("state", "unknown")
                if status in {"running", "starting", "created", "finalizing"} and r["path"] != self.active_run:
                    status_text = "未完成 / 状态待确认"
                else:
                    status_text = STATES.get(status, status)
                metrics = r["validation"].get("metrics", {})
                err = metrics.get("relative_L2", metrics.get("u_normalized_RMSE"))
                values = [r["request"]["name"], f"{c.grid.NX} × {c.grid.NY}", f"{t['Re']:.5g}", status_text,
                          str(r["summary"].get("iteration", "—")), VALIDATION.get(r["validation"].get("status"), "未生成"),
                          "—" if err is None else f"{err:.3e}"]
            for j, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(r["error"] or str(r["path"]))
                self.runs_table.setItem(i, j, item)

    def selected_records(self) -> list:
        rows = sorted({index.row() for index in self.runs_table.selectionModel().selectedRows()})
        return [self._records[i] for i in rows]

    def _selection_changed(self) -> None:
        records = self.selected_records()
        self._selected_record = records[0] if records else None

    def _one_record(self) -> dict:
        records = self.selected_records()
        if len(records) != 1:
            raise ValueError("请在档案中选中一项运行")
        if records[0]["error"]:
            raise ValueError(records[0]["error"])
        return records[0]

    def _load_fields(self, record: dict) -> tuple:
        root = record["path"] / "result"
        config = config_from_dict(read_json(root / "config.json"))
        with np.load(root / "results.npz", allow_pickle=False) as data:
            fields = {k: data[k].copy() for k in ("rho", "ux", "uy", "solid_mask", "speed", "vorticity")}
        expected = (config.grid.NY, config.grid.NX)
        if any(a.shape != expected for a in fields.values()):
            raise ValueError("结果字段尺寸与配置不一致")
        return config, fields

    def view_selected(self) -> None:
        r = self._one_record()
        config, fields = self._load_fields(r)
        history_path = r["path"] / "result/residual_history.csv"
        with history_path.open(encoding="utf-8") as handle:
            history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]
        draw_fields(self.result_plot.figure, config, fields, history, mode="fields")
        draw_fields(self.diagnostics_plot.figure, config, fields, history, mode="diagnostics")
        draw_validation(self.validation_plot.figure, r["validation"])
        self.result_plot.canvas.draw_idle()
        self.diagnostics_plot.canvas.draw_idle()
        self.validation_plot.canvas.draw_idle()
        validation = r["validation"]
        details = [f"{r['request']['name']}  ·  {config.grid.NX} × {config.grid.NY}",
                   f"停止：{STATES.get(r['job'].get('state'), r['job'].get('state'))}；验证：{VALIDATION.get(validation.get('status'), '未生成')}",
                   validation.get("reference_note", ""), validation.get("scope", "")]
        for name, measured in validation.get("metrics", {}).items():
            threshold = validation.get("thresholds", {}).get(name)
            value_text = "缺失" if measured is None else f"{measured:.5g}"
            threshold_text = "—" if threshold is None else f"{threshold:.4g}"
            details.append(f"{name} = {value_text}；阈值 {threshold_text}；{'满足' if validation.get('checks', {}).get(name) else '不满足'}")
        if r["job"].get("error"):
            details.append(f"执行错误：{r['job']['error']}")
        details.append(f"结果位置：{r['path']}")
        self.result_info.setPlainText("\n".join(details))
        self.result_tabs.setCurrentIndex(0)
        self.pages.setCurrentIndex(2)
        if r["path"] == self.active_run and r["summary"]:
            self._show_metrics(dict(r["summary"]["final_diagnostics"], iteration=r["summary"]["iteration"]),
                               r["job"].get("elapsed_seconds", 0))

    def reuse_selected(self) -> None:
        if self.process is not None:
            return
        record = self._one_record()
        request = record["request"]
        self.set_config(config_from_dict(request["config"]), request["name"] + " · 副本", request["notes"])

    def compare_selected(self) -> None:
        records = self.selected_records()
        if not 2 <= len(records) <= 4:
            raise ValueError("请按 Command 或 Shift 选中 2–4 项运行")
        entries, details = [], ["位置/域长、速度/U_ref 归一化中心线；叠图本身不构成网格无关或参数独立性结论。"]
        for i, r in enumerate(records):
            config, fields = self._load_fields(r)
            _, _, t = validate_config(config)
            label = f"#{i+1} {config.grid.NX}x{config.grid.NY} Re={t['Re']:.4g}"
            entries.append((config, fields, label))
            details.append(f"#{i+1} {r['request']['name']} | {config.case_type} | U_ref={config.flow.U_ref:g} "
                           f"ν={t['nu_lattice']:.6g} Ma={t['Ma']:.5g} | {r['job'].get('state')} | {r['path'].name}")
        draw_comparison(self.comparison_plot.figure, entries)
        self.comparison_plot.canvas.draw_idle()
        self.result_info.setPlainText("\n".join(details))
        self.result_tabs.setCurrentIndex(2)

    def open_selected_directory(self) -> None:
        record = self._one_record()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(record["path"])))

    def export_archive_figure(self) -> None:
        self.export_figure((self.result_plot, self.validation_plot, self.comparison_plot, self.diagnostics_plot)[self.result_tabs.currentIndex()])

    def export_figure(self, view: PlotView) -> None:
        if not view.figure.axes:
            raise ValueError("尚无可导出的图形")
        filename = self.workspace.root / f"{unique_id('figure')}.png"
        path, _ = QFileDialog.getSaveFileName(self, "导出当前图", str(filename), "PNG (*.png);;SVG (*.svg)")
        if path:
            if Path(path).suffix.lower() not in {".png", ".svg"}:
                path += ".png"
            view.figure.savefig(path, dpi=180, facecolor="white")
            self.statusBar().showMessage(f"已导出：{path}", 12000)

    def open_help(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(__file__).with_name("README.md"))))

    def closeEvent(self, event) -> None:
        if self.process is not None:
            self._close_when_done = True
            self.cancel_run()
            self.statusBar().showMessage("正在取消并保存结果；归档完成后自动关闭。")
            event.ignore()
        else:
            event.accept()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="LBM 等温桌面仿真工作台")
    parser.add_argument("--workspace", type=Path, help="独立工作区路径")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    window = WorkbenchWindow(args.workspace)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
