"""Native Qt workbench for frozen CPU/GPU simulations and local result analysis."""
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import time

import numpy as np
import psutil
from PySide6 import QtCore, QtGui, QtSvg, QtWidgets as W
from matplotlib import font_manager, rcParams
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from . import analysis
from .fieldplots import FLOW_MODES, REGIONS, draw_field
from .service import ensure_service
from .store import (ACTIVE, BACKEND_NAMES, DEFAULTS, ROOT, STATUS_NAMES, Store, encode,
                    backend_label, read_json, receipt_status, transport, validate_parameters,
                    COLLISION_NAMES, BOUNDARY_NAMES, model_label)
from research_options import EXTRA_DEFAULTS, LATTICES, SCENARIOS, RECIPES, TRANSIENT, BUOYANT, THERMAL, OBSTACLES, generic_engine
from .experiment_tools import ExperimentToolsMixin

COLORS = ["#23594b", "#bd6253", "#4877b6", "#ae7d22", "#a25ca3", "#329aaa", "#65734c", "#755ec4"]
PAGE_NAMES = ("仿真控制", "数据档案", "对比分析", "原 CPU 任务")
PAGE_ENGLISH = ("SIMULATION", "DATA ARCHIVE", "COMPARATIVE ANALYSIS", "CPU OBSERVATORY")
ICONS = {W.QStyle.SP_ComputerIcon: "cpu", W.QStyle.SP_DirIcon: "database",
         W.QStyle.SP_FileDialogDetailedView: "chart-no-axes-combined", W.QStyle.SP_DriveHDIcon: "hard-drive",
         W.QStyle.SP_DialogHelpButton: "globe", W.QStyle.SP_BrowserReload: "refresh-cw",
         W.QStyle.SP_DialogSaveButton: "save", W.QStyle.SP_TrashIcon: "trash",
         W.QStyle.SP_MediaPlay: "play", W.QStyle.SP_MediaPause: "pause", W.QStyle.SP_MediaStop: "square",
         W.QStyle.SP_ArrowUp: "arrow-up", W.QStyle.SP_ArrowDown: "arrow-down",
         W.QStyle.SP_DirOpenIcon: "folder-open", W.QStyle.SP_DialogApplyButton: "check",
         W.QStyle.SP_DialogResetButton: "rotate-ccw", W.QStyle.SP_FileDialogBack: "arrow-left",
         W.QStyle.SP_DirClosedIcon: "archive", W.QStyle.SP_DriveFDIcon: "package"}


def configure_fonts():
    application = W.QApplication.instance()
    if application is not None:
        palette = application.palette()
        for role, color in ((QtGui.QPalette.Window, "#f7f8f6"), (QtGui.QPalette.Base, "#ffffff"),
                            (QtGui.QPalette.WindowText, "#202d2b"), (QtGui.QPalette.Text, "#202d2b"),
                            (QtGui.QPalette.Highlight, "#e9f0eb"), (QtGui.QPalette.HighlightedText, "#23594b")):
            palette.setColor(role, QtGui.QColor(color))
        application.setPalette(palette)
    fonts = ROOT.parent / "BGK_NEE_Windows_20260928_191621/dashboard/fonts"
    for name in ("NotoSansSC-Regular.ttf", "SourceSans3-Regular.ttf",
                 "NotoSerifSC-Regular.ttf", "SourceSerif4-Regular.ttf"):
        path = fonts / name
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            QtGui.QFontDatabase.addApplicationFont(str(path))
    rcParams.update({"font.family": ["Noto Sans SC", "Microsoft YaHei", "DejaVu Sans"], "font.size": 9,
                     "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#dbe2dd", "axes.labelcolor": "#61706b", "text.color": "#202d2b",
                     "xtick.color": "#61706b", "ytick.color": "#61706b", "grid.color": "#e5e9e6",
                     "figure.facecolor": "#f7f8f6", "axes.facecolor": "#f7f8f6",
                     "legend.frameon": False, "axes.titlesize": 10})


def fmt(value, precision=5):
    if value is None:
        return "未记录"
    try:
        number = float(value)
        if not math.isfinite(number):
            return "非有限"
        return f"{number:.{precision}g}"
    except (ValueError, TypeError):
        return str(value)


def short_label(case):
    title = case['name'][:16] + ("…" if len(case['name']) > 16 else "")
    return f"{title} · {model_label(case['params'])} · {case['id'][:5]} [{STATUS_NAMES.get(case['status'], case['status'])}]"


def physical_label(params):
    if params.get("scenario") in BUOYANT:
        return f"Ra {fmt(params.get('rayleigh'),8)} · Pr {fmt(params.get('prandtl'))}"
    value = (transport(params)["effective_re"] if params.get("viscosity") else params.get("re"))
    return f"Re {fmt(value,8)}"


def grid_label(params):
    sizes = [params.get("grid"), params.get("grid_y") or params.get("grid")]
    if params.get("lattice", "").startswith("D3"): sizes.append(params.get("grid_z"))
    return "×".join(fmt(n,8) for n in sizes)


def diagnostic_scale(axis, series):
    """Keep exact zeros visible without crowding decades around a flat curve."""
    arrays = [np.asarray(values) for values in series]
    finite = np.concatenate([values[np.isfinite(values)] for values in arrays]) if arrays else np.array([])
    if finite.size and np.max(np.abs(finite)) <= 1e-12:
        axis.set_yscale("linear")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if not np.any(finite):
            axis.set_ylim(-1e-12, 1e-12)
            axis.set_yticks([-1e-12, 0, 1e-12])
            axis.text(.98, .96, "记录值均为 0", ha="right", va="top", transform=axis.transAxes, fontsize=8)
    else:
        axis.set_yscale("symlog", linthresh=1e-12)
        axis.yaxis.get_major_locator().set_params(numticks=6)


class Signals(QtCore.QObject):
    completed = QtCore.Signal(int, object, object)


class Task(QtCore.QRunnable):
    def __init__(self, token, function):
        super().__init__()
        self.token, self.function = token, function
        self.signals = Signals()

    @QtCore.Slot()
    def run(self):
        try:
            result, message = self.function(), None
        except Exception as error:
            result, message = None, f"{type(error).__name__}: {error}"
        try:
            self.signals.completed.emit(self.token, result, message)
        except RuntimeError as error:
            if 'already deleted' not in str(error): raise


class Plot(W.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = W.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.figure = Figure(figsize=(7, 4), constrained_layout=True, facecolor="#f7f8f6")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(220, 170)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setIconSize(QtCore.QSize(16, 16))
        self.toolbar.setMaximumHeight(34)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

    def empty(self, message):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_axis_off()
        axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes, color="#7b8881", wrap=True)
        self.canvas.draw_idle()


class Dashboard(ExperimentToolsMixin,W.QMainWindow):
    def __init__(self, store):
        super().__init__()
        self.store = store
        self.cases = []
        self.checked = set()
        self.tasks = {}
        self.task_serial = 0
        self.monitor_id = None
        self.monitor_busy = False
        self.monitor_key = None
        self.loaded_monitor = None
        self.last_field_render = 0.0
        self.field_windows = []
        self.field_render_timer = QtCore.QTimer(self)
        self.field_render_timer.setSingleShot(True)
        self.field_render_timer.timeout.connect(self.render_field)
        self.analysis_data = []
        self.analysis_version = 0
        self.library_key = None
        self.queue_key = None
        self.last_service_check = 0.0
        self.closing = False
        self.setWindowTitle("LBM Atlas · 仿真工作台 | CPU / GPU")
        self.resize(1440, 940)
        self.setMinimumSize(960, 680)
        self.build_shell()
        self.setStyleSheet(STYLE)
        self.apply_params(self.store.setting("last_parameters", DEFAULTS))
        self.refresh_presets()
        self.refresh()
        self.refresh_validation()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.async_run(self.store.discover, lambda result: (self.refresh(force=True), self.notice(f"已登记 {len(result[0])} 组历史记录")))

    def button(self, text, icon, action, tooltip=None):
        button = W.QPushButton(text)
        button.setIcon(self.symbol(icon))
        button.setIconSize(QtCore.QSize(16, 16))
        button.setToolTip(tooltip or text)
        button.clicked.connect(action)
        return button

    def tool(self, icon, tooltip, action):
        button = W.QToolButton()
        button.setIcon(self.symbol(icon))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setFixedSize(32, 32)
        button.clicked.connect(action)
        return button

    def symbol(self, standard, color="#61706b"):
        path = Path(__file__).parent / "icons" / (ICONS.get(standard, "square") + ".svg")
        renderer = QtSvg.QSvgRenderer(str(path))
        if not renderer.isValid():
            return self.style().standardIcon(standard)
        source = QtGui.QPixmap(32, 32)
        source.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(source)
        renderer.render(painter)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceIn)
        painter.fillRect(source.rect(), QtGui.QColor(color))
        painter.end()
        return QtGui.QIcon(source)

    def build_shell(self):
        shell = W.QWidget()
        shell.setObjectName("shell")
        self.setCentralWidget(shell)
        layout = W.QHBoxLayout(shell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        sidebar = self.sidebar = W.QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(226)
        side = W.QVBoxLayout(sidebar)
        side.setContentsMargins(20, 29, 20, 22)
        side.setSpacing(7)
        brand = self.brand = W.QLabel("LBM Atlas")
        brand.setObjectName("brand")
        side.addWidget(brand)
        name = self.side_title = W.QLabel("RESEARCH WORKSPACE")
        name.setObjectName("sideTitle")
        side.addWidget(name)
        side.addSpacing(28)
        self.nav_label = W.QLabel("研究空间")
        self.nav_label.setObjectName("navLabel")
        side.addWidget(self.nav_label)
        side.addSpacing(7)
        self.navigation = W.QButtonGroup(self)
        self.pages = W.QStackedWidget()
        for index, (text, icon) in enumerate((("仿真控制", W.QStyle.SP_ComputerIcon),
                                             ("数据档案", W.QStyle.SP_DirIcon),
                                             ("对比分析", W.QStyle.SP_FileDialogDetailedView),
                                             ("原 CPU 任务", W.QStyle.SP_DriveHDIcon))):
            button = self.button(text, icon, lambda checked=False, i=index: self.show_page(i))
            button.setCheckable(True)
            button.setObjectName("nav")
            button.setMinimumHeight(45)
            self.navigation.addButton(button, index)
            side.addWidget(button)
        self.navigation.button(0).setChecked(True)
        side.addStretch()
        self.atlas_link = self.button("学术资料站", W.QStyle.SP_DialogHelpButton,
                                      lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(
                                          "https://lbm-research-atlas.leochou8.chatgpt.site")))
        self.atlas_link.setObjectName("nav")
        self.atlas_link.setToolTip("打开 LBM Atlas 学术资料站")
        side.addWidget(self.atlas_link)
        side.addSpacing(16)
        mode = self.side_model = W.QLabel("D2Q9 · D2Q25 / D3Q19 · D3Q27\nBGK · MRT · TRT / 热流耦合")
        mode.setObjectName("sideInfo")
        side.addWidget(mode)
        self.side_count = W.QLabel()
        self.side_count.setObjectName("sideInfo")
        self.side_count.setWordWrap(True)
        side.addWidget(self.side_count)
        layout.addWidget(sidebar)
        container = W.QWidget()
        body = W.QVBoxLayout(container)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.topbar = W.QWidget()
        self.topbar.setObjectName("topbar")
        self.topbar.setFixedHeight(67)
        top = W.QHBoxLayout(self.topbar)
        top.setContentsMargins(30, 0, 26, 0)
        self.breadcrumb = W.QLabel("研究空间  /  仿真控制")
        self.breadcrumb.setObjectName("muted")
        top.addWidget(self.breadcrumb)
        top.addStretch()
        self.validation_label = W.QLabel("正在核验冻结代码")
        self.validation_label.setObjectName("muted")
        top.addWidget(self.validation_label)
        top.addWidget(self.tool(W.QStyle.SP_BrowserReload, "刷新记录与源码核验", self.manual_refresh))
        body.addWidget(self.topbar)
        content = W.QWidget()
        self.content_layout = content_layout = W.QVBoxLayout(content)
        content_layout.setContentsMargins(30, 19, 30, 10)
        content_layout.setSpacing(12)
        header = W.QHBoxLayout()
        heading = W.QVBoxLayout()
        heading.setSpacing(2)
        self.eyebrow = W.QLabel("SIMULATION / LOCAL COMPUTE")
        self.eyebrow.setObjectName("eyebrow")
        heading.addWidget(self.eyebrow)
        self.title = W.QLabel("仿真控制")
        self.title.setObjectName("title")
        heading.addWidget(self.title)
        header.addLayout(heading)
        header.addStretch()
        self.model_label = W.QLabel("二维 / 三维 · 热流耦合\nBGK / MRT / TRT · Float64")
        self.model_label.setObjectName("modelLabel")
        self.model_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        header.addWidget(self.model_label)
        content_layout.addLayout(header)
        self.overview = W.QWidget()
        self.overview.setObjectName("overview")
        overview = W.QHBoxLayout(self.overview)
        overview.setContentsMargins(0, 9, 0, 9)
        overview.setSpacing(0)
        self.overview_values = {}
        for key, label in (("data", "数据档案"), ("active", "运行 / 暂停"),
                           ("queued", "等待计算"), ("converged", "已收敛")):
            item = W.QWidget()
            item.setObjectName("overviewItem")
            row = W.QHBoxLayout(item)
            row.setContentsMargins(12, 0, 16, 0)
            caption = W.QLabel(label)
            caption.setObjectName("muted")
            row.addWidget(caption)
            value = W.QLabel("0")
            value.setObjectName("overviewValue")
            value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            row.addWidget(value, 1)
            overview.addWidget(item, 1)
            self.overview_values[key] = value
        content_layout.addWidget(self.overview)
        self.build_control()
        self.build_library()
        self.build_analysis()
        self.build_cpu()
        content_layout.addWidget(self.pages, 1)
        self.message = W.QLabel("就绪")
        self.message.setWordWrap(True)
        self.message.setObjectName("muted")
        content_layout.addWidget(self.message)
        body.addWidget(content, 1)
        layout.addWidget(container, 1)

    def build_control(self):
        page = W.QWidget()
        outer = W.QHBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        splitter = W.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter)
        parameters = W.QWidget()
        parameters.setMinimumWidth(265)
        parameters.setMaximumWidth(355)
        parameter_layout = W.QVBoxLayout(parameters)
        parameter_layout.setContentsMargins(0, 0, 17, 0)
        parameter_layout.setSpacing(12)
        form_scroll = W.QScrollArea()
        form_scroll.setWidgetResizable(True)
        form_scroll.setFrameShape(W.QFrame.NoFrame)
        panel = W.QWidget()
        panel.setObjectName("parameterPanel")
        form_scroll.setWidget(panel)
        form = W.QVBoxLayout(panel)
        form.setContentsMargins(0, 0, 5, 0)
        form.setSpacing(8)
        section = W.QLabel("新建仿真")
        section.setObjectName("section")
        form.addWidget(section)
        presets = W.QHBoxLayout()
        self.preset = W.QComboBox()
        self.preset.setObjectName("preset")
        self.preset.setSizeAdjustPolicy(W.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.preset.setMinimumContentsLength(8)
        self.preset.currentIndexChanged.connect(self.load_preset)
        presets.addWidget(self.preset, 1)
        presets.addWidget(self.tool(W.QStyle.SP_DialogSaveButton, "保存参数模板", self.save_preset))
        presets.addWidget(self.tool(W.QStyle.SP_TrashIcon, "删除选中模板", self.delete_preset))
        form.addLayout(presets)
        self.recipe = W.QComboBox()
        self.recipe.setSizeAdjustPolicy(W.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.recipe.setMinimumContentsLength(10)
        self.recipe.addItem("内置实验 · 选择后填入参数", None)
        for title in RECIPES:
            self.recipe.addItem(title, title)
        self.recipe.currentIndexChanged.connect(self.load_recipe)
        form.addWidget(self.recipe)
        grid = W.QFormLayout()
        grid.setRowWrapPolicy(W.QFormLayout.WrapLongRows)
        grid.setFieldGrowthPolicy(W.QFormLayout.AllNonFixedFieldsGrow)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        self.backend = W.QComboBox()
        self.backend.setObjectName("backend")
        for key in ("fused", "cpu", "array"):
            self.backend.addItem(BACKEND_NAMES[key], key)
        grid.addRow("计算后端", self.backend)
        self.extra_combos = {}
        for key, label, choices in (("scenario","物理场景",SCENARIOS),("lattice","速度模型",LATTICES),
                                    ("run_mode","停止方式",{"steady":"稳态收敛","transient":"瞬态 · 固定步数"})):
            combo = W.QComboBox()
            combo.setObjectName(key)
            combo.setSizeAdjustPolicy(W.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
            for value,title in choices.items(): combo.addItem(title,value)
            self.extra_combos[key] = combo
            grid.addRow(label,combo)
            combo.currentIndexChanged.connect(self.update_preview)
        self.collision = W.QComboBox()
        self.collision.setObjectName("collision")
        for key, label in COLLISION_NAMES.items():
            self.collision.addItem(label, key)
        grid.addRow("碰撞模型", self.collision)
        self.boundary = W.QComboBox()
        self.boundary.setObjectName("boundary")
        for key, label in BOUNDARY_NAMES.items():
            self.boundary.addItem(label, key)
        grid.addRow("壁面条件", self.boundary)
        self.boundary.setToolTip("NEE：壁面在最外层格点，L=N−1。HBB：全部格点为流体，壁面在外侧半格，L=N。")
        self.collision.currentIndexChanged.connect(self.update_preview)
        self.boundary.currentIndexChanged.connect(self.update_preview)
        self.name_input = W.QLineEdit()
        self.name_input.setObjectName("case_name")
        self.name_input.setMaxLength(120)
        grid.addRow("算例名称", self.name_input)
        self.group_input = W.QLineEdit("研究算例")
        self.group_input.setObjectName("case_group")
        self.group_input.setMaxLength(100)
        grid.addRow("分组", self.group_input)
        self.inputs = {}
        for key, label in (("grid", "网格 Nx"), ("grid_y", "Ny · 0 跟随 Nx"), ("grid_z", "Nz · 三维"),
                           ("re", "参考雷诺数 Re"), ("lid_speed", "参考速度 U"),
                           ("tol", "残差阈值"), ("max_iter", "最大步数"), ("min_iter", "最少步数"),
                           ("ramp_steps", "启动步数"), ("report_interval", "报告间隔"),
                           ("mrt_s_e", "MRT · s_e"), ("mrt_s_eps", "MRT · s_ε"), ("mrt_s_q", "MRT · s_q")):
            edit = W.QLineEdit()
            edit.setObjectName("param_" + key)
            edit.setMaxLength(64)
            edit.setAlignment(QtCore.Qt.AlignRight)
            edit.setAccessibleName(label)
            edit.textChanged.connect(self.update_preview)
            self.inputs[key] = edit
            grid.addRow(label, edit)
        self.backend.currentIndexChanged.connect(self.update_preview)
        form.addLayout(grid)
        advanced = W.QPushButton("展开物性、热流与障碍物参数")
        advanced.setCheckable(True)
        advanced_panel = W.QWidget()
        advanced_form = W.QFormLayout(advanced_panel)
        advanced_form.setRowWrapPolicy(W.QFormLayout.WrapLongRows)
        advanced_form.setFieldGrowthPolicy(W.QFormLayout.AllNonFixedFieldsGrow)
        advanced_form.setContentsMargins(0,0,0,0)
        for key,label in (("viscosity","ν · 0 由 Re 推算"),("rho0","初始密度 ρ₀"),("trt_magic","TRT · Λ"),
                          ("force_x","附加加速度 ax"),("force_y","附加加速度 ay"),("force_z","附加加速度 az"),
                          ("diffusivity","热扩散率 α"),("rayleigh","Rayleigh · Ra"),("prandtl","Prandtl · Pr"),
                          ("obstacle_x","障碍中心 x/Lx"),("obstacle_y","障碍中心 y/Ly"),("obstacle_z","障碍中心 z/Lz"),
                          ("obstacle_size","半径 / 最短边长"),("snapshot_interval","场快照间隔 · 0 关"),
                          ("checkpoint_interval","检查点间隔 · 0 关")):
            edit = W.QLineEdit()
            edit.setObjectName("param_"+key)
            edit.setAlignment(QtCore.Qt.AlignRight)
            edit.setMaxLength(64)
            edit.textChanged.connect(self.update_preview)
            self.inputs[key] = edit
            advanced_form.addRow(label,edit)
        obstacle = W.QComboBox()
        obstacle.addItem("圆 / 球", "circle")
        obstacle.addItem("正方形 / 立方体", "rectangle")
        obstacle.currentIndexChanged.connect(self.update_preview)
        self.extra_combos["obstacle"] = obstacle
        advanced_form.addRow("障碍形状",obstacle)
        resume = W.QLineEdit()
        resume.setPlaceholderText("空白表示重新初始化")
        resume.textChanged.connect(self.update_preview)
        self.inputs["resume_from"] = resume
        advanced_form.addRow("续算检查点",resume)
        advanced_form.addRow(self.button("从检查点读取参数",W.QStyle.SP_DirOpenIcon,self.load_checkpoint))
        advanced.toggled.connect(advanced_panel.setVisible)
        advanced_panel.setVisible(False)
        form.addWidget(advanced)
        form.addWidget(advanced_panel)
        self.build_experiment_parameters(form)
        self.sweep_enabled = W.QCheckBox("参数扫描")
        self.sweep_enabled.setObjectName("sweep_enabled")
        self.sweep_values = W.QLineEdit("100, 400, 1000")
        self.sweep_values.setObjectName("sweep_values")
        self.sweep_values.setEnabled(False)
        self.sweep_enabled.toggled.connect(self.sweep_values.setEnabled)
        self.sweep_enabled.toggled.connect(self.update_preview)
        self.sweep_values.textChanged.connect(self.update_preview)
        form.addWidget(self.sweep_enabled)
        self.sweep_parameter = W.QComboBox()
        for label,key in (("Re","re"),("Nx","grid"),("ν","viscosity"),("Ra","rayleigh"),("Pr","prandtl"),
                          ("α","diffusivity"),("ax","force_x"),("参考速度 U","lid_speed"),
                          ("Ny","grid_y"),("Nz","grid_z"),("驱动周期","drive_period"),
                          ("加速度幅值","force_amplitude"),("障碍尺寸","obstacle_size")):
            self.sweep_parameter.addItem(label,key)
        self.sweep_parameter.currentIndexChanged.connect(self.update_preview)
        form.addWidget(self.sweep_parameter)
        form.addWidget(self.sweep_values)
        self.build_second_scan(form)
        self.derived = W.QLabel()
        self.derived.setWordWrap(True)
        self.derived.setObjectName("derived")
        self.start_button = self.button("开始计算", W.QStyle.SP_MediaPlay, self.enqueue)
        self.start_button.setIcon(self.symbol(W.QStyle.SP_MediaPlay, "#ffffff"))
        self.start_button.setObjectName("primary")
        self.start_button.setMinimumHeight(42)
        form.addStretch()
        parameter_layout.addWidget(form_scroll, 1)
        parameter_layout.addWidget(self.derived)
        parameter_layout.addWidget(self.start_button)
        splitter.addWidget(parameters)

        right = W.QWidget()
        area = W.QVBoxLayout(right)
        area.setContentsMargins(17, 0, 0, 0)
        area.setSpacing(8)
        toolbar = W.QHBoxLayout()
        label = W.QLabel("任务与结果")
        label.setObjectName("section")
        toolbar.addWidget(label)
        toolbar.addStretch()
        self.queue_paused = W.QCheckBox("暂停调度")
        self.queue_paused.setObjectName("queue_paused")
        self.queue_paused.setToolTip("暂停后续任务启动，当前计算不受影响")
        self.queue_paused.toggled.connect(self.set_queue_paused)
        toolbar.addWidget(self.queue_paused)
        self.pause_button = self.tool(W.QStyle.SP_MediaPause, "暂停当前计算", lambda: self.control("pause"))
        self.resume_button = self.tool(W.QStyle.SP_MediaPlay, "继续当前计算", lambda: self.control("run"))
        self.cancel_button = self.tool(W.QStyle.SP_MediaStop, "取消当前任务并保存状态", lambda: self.control("cancel"))
        for button in (self.pause_button, self.resume_button, self.cancel_button):
            toolbar.addWidget(button)
        toolbar.addWidget(self.tool(W.QStyle.SP_ArrowUp, "排队任务前移", lambda: self.move_queue(-1)))
        toolbar.addWidget(self.tool(W.QStyle.SP_ArrowDown, "排队任务后移", lambda: self.move_queue(1)))
        toolbar.addWidget(self.tool(W.QStyle.SP_DirOpenIcon, "打开当前数据目录", self.open_current))
        area.addLayout(toolbar)
        vertical = W.QSplitter(QtCore.Qt.Vertical)
        vertical.setChildrenCollapsible(False)
        self.queue = W.QTreeWidget()
        self.queue.setObjectName("queue")
        self.queue.setHeaderLabels(["名称", "后端", "控制参数", "状态", "报告步数"])
        self.queue.setRootIsDecorated(False)
        self.queue.setMinimumHeight(100)
        self.queue.setAlternatingRowColors(True)
        self.queue.setUniformRowHeights(True)
        self.queue.header().setSectionResizeMode(W.QHeaderView.Interactive)
        self.queue.setColumnWidth(0, 220)
        self.queue.setColumnWidth(1, 72)
        self.queue.setColumnWidth(2, 75)
        self.queue.setColumnWidth(3, 120)
        self.queue.currentItemChanged.connect(self.select_monitor)
        vertical.addWidget(self.queue)
        observation = W.QWidget()
        observation_layout = W.QVBoxLayout(observation)
        observation_layout.setContentsMargins(0, 8, 0, 0)
        self.monitor_status = W.QLabel("尚未选择结果")
        self.monitor_status.setWordWrap(True)
        self.monitor_status.setObjectName("status")
        observation_layout.addWidget(self.monitor_status)
        metrics = W.QHBoxLayout()
        self.metrics = {}
        for key, label in (("iteration", "最近报告步数"), ("residual", "单步残差"), ("mass_drift", "质量守恒误差"), ("max_mach", "最大 Ma")):
            column = W.QVBoxLayout()
            heading = W.QLabel(label)
            heading.setObjectName("muted")
            value = W.QLabel("未记录")
            value.setObjectName("metric")
            column.addWidget(heading)
            column.addWidget(value)
            metrics.addLayout(column, 1)
            self.metrics[key] = value
        self.metrics_bar = W.QWidget()
        self.metrics_bar.setLayout(metrics)
        metrics.setContentsMargins(0, 0, 0, 0)
        observation_layout.addWidget(self.metrics_bar)
        self.budget = W.QProgressBar()
        self.budget.setRange(0, 1000)
        self.budget.setTextVisible(False)
        self.budget.setFixedHeight(4)
        observation_layout.addWidget(self.budget)
        self.monitor_tabs = W.QTabWidget()
        self.monitor_plot = Plot()
        self.monitor_plot.empty("尚无报告数据")
        self.monitor_tabs.addTab(self.monitor_plot, "诊断曲线")
        field = W.QWidget()
        field_layout = W.QVBoxLayout(field)
        field_layout.setContentsMargins(0, 0, 0, 0)
        field_controls = W.QGridLayout()
        self.field_controls = field_controls
        self.field_mode = W.QComboBox()
        for title, key in (("速度模 |u| / U", "speed"), ("水平速度 ux / U", "ux"), ("竖直速度 uy / U", "uy"), ("速度 uz / U", "uz"),
                           ("密度 rho", "rho"), ("温度 T", "T"), ("截面法向涡量", "vorticity"),
                           ("流线", "streamlines"), ("速度云图 + 流线", "overlay"), ("角涡四联图", "vortices")):
            self.field_mode.addItem(title, key)
        self.field_mode.setToolTip("流场图类型；流线沿当前速度方向，不是粒子轨迹")
        self.field_region = W.QComboBox()
        for key, title in REGIONS.items():
            self.field_region.addItem(title, key)
        self.field_region.setToolTip("绘图区域")
        self.field_density = W.QDoubleSpinBox()
        self.field_density.setRange(.6, 3)
        self.field_density.setSingleStep(.2)
        self.field_density.setDecimals(1)
        self.field_density.setValue(1.5)
        self.field_density.setToolTip("只改变流线绘图疏密，不改变网格或计算")
        self.field_extent = W.QSpinBox()
        self.field_extent.setRange(10, 50)
        self.field_extent.setValue(35)
        self.field_extent.setSuffix(" %")
        self.field_extent.setToolTip("角落视野占腔体边长的比例")
        field_controls.addWidget(self.field_mode, 0, 0, 1, 2)
        field_controls.addWidget(self.field_region, 0, 2, 1, 2)
        self.expand_field_button = self.tool(W.QStyle.SP_TitleBarMaxButton, "独立窗口查看当前流场快照", self.expand_field)
        self.expand_field_button.setIcon(self.style().standardIcon(W.QStyle.SP_TitleBarMaxButton))
        field_controls.addWidget(self.expand_field_button, 0, 4)
        field_controls.addWidget(W.QLabel("流线疏密"), 1, 0)
        field_controls.addWidget(self.field_density, 1, 1)
        field_controls.addWidget(W.QLabel("角落范围"), 1, 2)
        field_controls.addWidget(self.field_extent, 1, 3)
        self.field_plane = W.QComboBox()
        for plane in ("xy","xz","yz"): self.field_plane.addItem(plane+" 截面",plane)
        self.field_position = W.QDoubleSpinBox()
        self.field_position.setRange(0,1)
        self.field_position.setDecimals(3)
        self.field_position.setSingleStep(.05)
        self.field_position.setValue(.5)
        self.field_position.setToolTip("按归一化位置选择最近格点截面；完整三维数据保留在结果文件中")
        field_controls.addWidget(self.field_plane,2,0,1,2)
        field_controls.addWidget(W.QLabel("截面位置"),2,2)
        field_controls.addWidget(self.field_position,2,3)
        self.field_plane.currentIndexChanged.connect(self.field_options_changed)
        self.field_position.valueChanged.connect(self.field_options_changed)
        field_controls.setColumnStretch(1, 1)
        field_controls.setColumnStretch(3, 1)
        field_layout.addLayout(field_controls)
        self.build_timeline(field_layout)
        self.field_plot = Plot()
        self.field_plot.figure.set_layout_engine("compressed")
        self.field_plot.empty("尚无流场文件")
        field_scroll = W.QScrollArea()
        field_scroll.setWidgetResizable(True)
        field_scroll.setWidget(self.field_plot)
        field_layout.addWidget(field_scroll, 1)
        self.monitor_tabs.addTab(field, "流场")
        for control in (self.field_mode, self.field_region):
            control.currentIndexChanged.connect(self.field_options_changed)
        for control in (self.field_density, self.field_extent):
            control.valueChanged.connect(self.field_options_changed)
        self.monitor_tabs.currentChanged.connect(lambda index: self.render_field() if index == 1 else None)
        self.monitor_tabs.currentChanged.connect(self.adjust_field_layout)
        self.field_options_changed()
        self.details = W.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.monitor_tabs.addTab(self.details, "参数与日志")
        self.build_signal_tab()
        observation_layout.addWidget(self.monitor_tabs, 1)
        vertical.addWidget(observation)
        vertical.setSizes([145, 480])
        vertical.setStretchFactor(1, 1)
        area.addWidget(vertical, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 850])
        self.pages.addWidget(page)

    def build_library(self):
        page = W.QWidget()
        layout = W.QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 0)
        tools = W.QHBoxLayout()
        self.search = W.QLineEdit()
        self.search.setPlaceholderText("名称、分组、备注或 Re")
        self.search.textChanged.connect(lambda: self.rebuild_library(force=True))
        tools.addWidget(self.search, 1)
        self.status_filter = W.QComboBox()
        for text, key in (("全部状态", ""), ("已收敛", "converged"), ("未收敛 / 停止", "unfinished"), ("活动与排队", "active")):
            self.status_filter.addItem(text, key)
        self.status_filter.currentIndexChanged.connect(lambda: self.rebuild_library(force=True))
        tools.addWidget(self.status_filter)
        self.show_archived = W.QCheckBox("含已归档")
        self.show_archived.toggled.connect(lambda: self.rebuild_library(force=True))
        tools.addWidget(self.show_archived)
        tools.addWidget(self.tool(W.QStyle.SP_DirOpenIcon, "导入结果目录", self.import_results))
        tools.addWidget(self.tool(W.QStyle.SP_BrowserReload, "扫描已有 GPU 结果", self.discover_results))
        layout.addLayout(tools)
        actionbar = W.QHBoxLayout()
        self.selected_label = W.QLabel("已选 0 组")
        actionbar.addWidget(self.selected_label)
        actionbar.addWidget(self.tool(W.QStyle.SP_DialogApplyButton, "选择当前列表", self.select_visible))
        actionbar.addWidget(self.tool(W.QStyle.SP_DialogResetButton, "清除选择", self.clear_selection))
        actionbar.addStretch()
        for title, icon, callback in (("对比所选", W.QStyle.SP_FileDialogDetailedView, self.compare_selected),
                                       ("复用参数", W.QStyle.SP_FileDialogBack, self.reuse_selected),
                                       ("CSV", W.QStyle.SP_DialogSaveButton, self.export_csv),
                                       ("数据包", W.QStyle.SP_DriveFDIcon, self.export_bundle),
                                       ("归档 / 恢复", W.QStyle.SP_DirClosedIcon, self.archive_selected)):
            actionbar.addWidget(self.button(title, icon, callback))
        layout.addLayout(actionbar)
        self.library = W.QTreeWidget()
        self.library.setObjectName("library")
        self.library.setHeaderLabels(["数据名称", "分组", "后端", "网格", "控制参数", "状态", "步数", "残差", "质量漂移", "记录日期"])
        self.library.setRootIsDecorated(False)
        self.library.setAlternatingRowColors(True)
        self.library.setUniformRowHeights(True)
        self.library.setSelectionMode(W.QAbstractItemView.ExtendedSelection)
        for column, width in enumerate((240, 105, 75, 55, 85, 155, 90, 100, 100, 160)):
            self.library.setColumnWidth(column, width)
        self.library.itemChanged.connect(self.library_checked)
        self.library.currentItemChanged.connect(self.library_current)
        self.library.itemDoubleClicked.connect(lambda item, col: self.open_case(item.data(0, QtCore.Qt.UserRole)))
        layout.addWidget(self.library, 1)
        edit = W.QGridLayout()
        self.edit_name = W.QLineEdit()
        self.edit_group = W.QLineEdit()
        self.edit_notes = W.QLineEdit()
        self.edit_name.setMaxLength(120)
        self.edit_group.setMaxLength(100)
        self.edit_notes.setMaxLength(4000)
        edit.addWidget(W.QLabel("名称"), 0, 0)
        edit.addWidget(self.edit_name, 0, 1)
        edit.addWidget(W.QLabel("分组"), 0, 2)
        edit.addWidget(self.edit_group, 0, 3)
        edit.addWidget(self.tool(W.QStyle.SP_DialogSaveButton, "保存名称、分组与备注", self.save_metadata), 0, 4)
        edit.addWidget(W.QLabel("备注"), 1, 0)
        edit.addWidget(self.edit_notes, 1, 1, 1, 4)
        layout.addLayout(edit)
        self.source_label = W.QLabel()
        self.source_label.setWordWrap(True)
        self.source_label.setObjectName("muted")
        layout.addWidget(self.source_label)
        self.pages.addWidget(page)

    def build_analysis(self):
        page = W.QWidget()
        layout = W.QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 0)
        top = W.QHBoxLayout()
        self.analysis_selection = W.QLabel("尚未选择数据")
        self.analysis_selection.setWordWrap(True)
        top.addWidget(self.analysis_selection, 1)
        top.addWidget(self.button("选择数据", W.QStyle.SP_DirOpenIcon, lambda: self.show_page(1)))
        top.addWidget(self.tool(W.QStyle.SP_BrowserReload, "重新读取选中数据", self.load_analysis))
        layout.addLayout(top)
        controls = W.QHBoxLayout()
        self.analysis_mode = W.QComboBox()
        for name, key in (("诊断曲线叠加", "history"), ("归一化中心线", "centerlines"), ("参数扫描", "sweep"), ("两组流场差值", "difference")):
            self.analysis_mode.addItem(name, key)
        self.analysis_mode.currentIndexChanged.connect(self.render_analysis)
        controls.addWidget(self.analysis_mode)
        self.analysis_metric = W.QComboBox()
        for name, key in (("单步残差", "residual"), ("质量漂移", "mass_drift"), ("最大 Ma", "max_mach"),
                          ("温度残差", "thermal_residual"),("平均动能", "kinetic_energy"),
                          ("热壁 Nu", "nusselt_hot"),("冷壁 Nu", "nusselt_cold"),("最高温度", "temperature_max"),
                          ("记录耗时 / s", "elapsed_seconds"), ("迭代步数", "iteration")):
            self.analysis_metric.addItem(name, key)
        self.analysis_metric.currentIndexChanged.connect(self.render_analysis)
        controls.addWidget(self.analysis_metric)
        self.scan_axis = W.QComboBox()
        for name, key in (("扫描轴：Re", "re"), ("扫描轴：Nx", "grid"), ("扫描轴：U", "lid_speed"),
                          ("扫描轴：ν","viscosity"),("扫描轴：Ra","rayleigh"),("扫描轴：Pr","prandtl"),
                          ("扫描轴：α","diffusivity"),("扫描轴：ax","force_x")):
            self.scan_axis.addItem(name, key)
        self.scan_axis.currentIndexChanged.connect(self.render_analysis)
        controls.addWidget(self.scan_axis)
        self.reference_case = W.QComboBox()
        self.reference_case.setToolTip("流场差值与相对 L2 的基准数据")
        self.reference_case.setMaximumWidth(310)
        self.reference_case.setSizeAdjustPolicy(W.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.reference_case.setMinimumContentsLength(12)
        self.reference_case.currentIndexChanged.connect(self.render_analysis)
        controls.addWidget(self.reference_case)
        self.scan_axis.setVisible(False)
        self.reference_case.setVisible(False)
        controls.addStretch()
        layout.addLayout(controls)
        self.analysis_warning = W.QLabel()
        self.analysis_warning.setWordWrap(True)
        self.analysis_warning.setObjectName("warning")
        layout.addWidget(self.analysis_warning)
        self.comparison_plot = Plot()
        self.comparison_plot.empty("尚未选择数据")
        layout.addWidget(self.comparison_plot, 1)
        self.comparison_table = W.QTreeWidget()
        self.comparison_table.setHeaderLabels(["名称", "后端", "网格 / 控制参数", "状态", "步数", "残差", "质量漂移", "耗时 / s"])
        self.comparison_table.setRootIsDecorated(False)
        self.comparison_table.setAlternatingRowColors(True)
        self.comparison_table.setMaximumHeight(190)
        self.comparison_table.setMinimumHeight(90)
        for column, width in enumerate((230, 65, 110, 155, 85, 105, 100, 90)):
            self.comparison_table.setColumnWidth(column, width)
        layout.addWidget(self.comparison_table)
        self.pages.addWidget(page)

    def build_cpu(self):
        page = W.QWidget()
        layout = W.QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 0)
        self.cpu_status = W.QLabel()
        self.cpu_status.setObjectName("status")
        self.cpu_status.setWordWrap(True)
        layout.addWidget(self.cpu_status)
        self.cpu_plot = Plot()
        layout.addWidget(self.cpu_plot, 1)
        self.cpu_detail = W.QPlainTextEdit()
        self.cpu_detail.setReadOnly(True)
        self.cpu_detail.setMaximumHeight(150)
        layout.addWidget(self.cpu_detail)
        self.cpu_key = None
        self.pages.addWidget(page)

    def notice(self, text, error=False):
        self.message.setText(text)
        self.message.setStyleSheet("color:#b64039" if error else "color:#62746b")

    def error(self, text):
        self.notice(text, True)
        W.QMessageBox.warning(self, "操作未完成", text)

    def async_run(self, function, success, failure=None):
        self.task_serial += 1
        task = Task(self.task_serial, function)
        self.tasks[self.task_serial] = (task, success, failure)
        task.signals.completed.connect(self.async_finished)
        QtCore.QThreadPool.globalInstance().start(task)
        return self.task_serial

    @QtCore.Slot(int, object, object)
    def async_finished(self, token, result, error):
        entry = self.tasks.pop(token, None)
        if self.closing or entry is None:
            return
        _, success, failure = entry
        if error:
            (failure or self.error)(error)
        else:
            success(result)

    def show_page(self, index):
        self.pages.setCurrentIndex(index)
        self.navigation.button(index).setChecked(True)
        self.title.setText(PAGE_NAMES[index] + (" · 只读" if index == 3 else ""))
        self.breadcrumb.setText("研究空间  /  " + PAGE_NAMES[index])
        self.eyebrow.setText(PAGE_ENGLISH[index] + " / LOCAL COMPUTE")
        if index == 1:
            self.rebuild_library(force=True)
        if index == 2 and self.checked:
            self.load_analysis()
        if index == 3:
            self.refresh_cpu()
        if index == 0:
            self.render_field()

    def form_params(self):
        values = {k: edit.text().strip() for k, edit in self.inputs.items()}
        if self.collision.currentData() != "MRT":
            # Disabled MRT inputs cannot prevent an otherwise valid BGK run.
            for key in ("mrt_s_e", "mrt_s_eps", "mrt_s_q"):
                try:
                    valid = 0 < float(values[key]) < 2
                except ValueError:
                    valid = False
                if not valid:
                    values[key] = DEFAULTS[key]
        return validate_parameters(dict(backend=self.backend.currentData(), collision=self.collision.currentData(),
                                        boundary=self.boundary.currentData(),
                                        **{k:combo.currentData() for k,combo in self.extra_combos.items()},
                                        **values))

    def form_requests(self):
        return self.experiment_requests()

    def apply_params(self, params):
        value = dict(DEFAULTS, **EXTRA_DEFAULTS)
        value.update({key: item for key, item in params.items() if key in value and item is not None})
        index = self.backend.findData(value["backend"])
        self.backend.setCurrentIndex(max(0, index))
        self.collision.setCurrentIndex(max(0, self.collision.findData(value["collision"])))
        self.boundary.setCurrentIndex(max(0, self.boundary.findData(value["boundary"])))
        for key,combo in self.extra_combos.items():
            combo.setCurrentIndex(max(0,combo.findData(value[key])))
        for key, edit in self.inputs.items():
            edit.setMinimumWidth(70)
            edit.setText(str(value[key]))
        self.update_preview()

    def load_recipe(self):
        name = self.recipe.currentData()
        if name and hasattr(self,"start_button"):
            self.apply_params(dict(RECIPES[name],backend=self.backend.currentData()))
            self.sweep_enabled.setChecked(False)
            self.name_input.setText(name)
            self.notice("已加载内置实验："+name)

    def load_checkpoint(self):
        source,_ = W.QFileDialog.getOpenFileName(self,"读取续算检查点",str(self.store.workspace),"NPZ (*.npz)")
        if not source: return
        try:
            with np.load(source,allow_pickle=False) as archive:
                meta = json.loads(str(archive["metadata"]))
                iteration = int(archive["iteration"])
            params = dict(meta["params"],resume_from=source,backend=self.backend.currentData())
            params["max_iter"] = max(params["max_iter"],iteration+1000)
            self.apply_params(params)
            self.notice(f"检查点在第 {iteration} 步；最大步数表示续算后的总步数，结果保存到新任务")
        except (OSError,ValueError,KeyError) as error:
            self.error(str(error))

    def update_preview(self, *args):
        if not hasattr(self, "start_button"):
            return
        for key in ("mrt_s_e", "mrt_s_eps", "mrt_s_q"):
            self.inputs[key].setEnabled(self.collision.currentData() == "MRT")
            self.inputs[key].setToolTip("非守恒模态松弛率，0 < s < 2；剪切模态由 Re、U、腔长自动确定")
        s = self.extra_combos["scenario"].currentData()
        d3 = self.extra_combos["lattice"].currentData().startswith("D3")
        for key in ("grid_z","force_z","obstacle_z"): self.inputs[key].setEnabled(d3)
        self.inputs["trt_magic"].setEnabled(self.collision.currentData()=="TRT")
        for key in ("rayleigh","prandtl"): self.inputs[key].setEnabled(s in BUOYANT)
        self.inputs["diffusivity"].setEnabled(s=="thermal_wave")
        for key in ("obstacle_x","obstacle_y","obstacle_size"): self.inputs[key].setEnabled(s in OBSTACLES)
        self.extra_combos["obstacle"].setEnabled(s in OBSTACLES)
        for axis in 'xyz':
            for stem in ('obstacle_count_','obstacle_spacing_'):
                self.inputs[stem+axis].setEnabled(s in OBSTACLES and (axis!='z' or d3))
        for key in ('drive_period','drive_phase'):self.inputs[key].setEnabled(s in {'oscillatory_channel','oscillating_couette'})
        self.inputs['force_amplitude'].setEnabled(s=='oscillatory_channel')
        self.inputs['thermal_perturbation'].setEnabled(s=='rayleigh_benard')
        self.inputs['outlet_rho'].setEnabled(s in {'open_channel','cylinder_wake'})
        self.extra_combos['inlet_profile'].setEnabled(s in {'open_channel','cylinder_wake'})
        try:
            requests = self.form_requests()
            params = requests[0]["params"]
            d = transport(params)
            text = f"{model_label(params)} · L = {d['length']}\ntau_ν = {d['tau']:.10g}\nnu = {d['nu']:.6g}  ·  Ma = {d['mach']:.6g}\n"
            text = f"{d['nx']} × {d['ny']}" + (f" × {d['nz']}" if d['dim']==3 else "") + " 格点\n" + text
            if d.get("thermal_tau"):
                text += f"α = {d['alpha']:.6g} · tau_T = {d['thermal_tau']:.6g}\n"
            if params["collision"] == "MRT":
                text += f"s_ν = {1 / d['tau']:.8g}\n"
            text += f"内存估算 {d['estimated_host_bytes']/1024**2:.0f} MB"
            if params["backend"] != "cpu":
                text += f" · 显存 {d['estimated_gpu_bytes']/1024**2:.0f} MB"
            warnings = []
            if params["min_iter"] > params["max_iter"]:
                warnings.append("本次预算小于最少收敛步数")
            if params["ramp_steps"] > params["max_iter"]:
                warnings.append("本次预算不足以完成启动过程")
            if d["tau"] < 0.5005:
                warnings.append("tau 非常接近 0.5，稳定性风险较高")
            if params["viscosity"]:
                warnings.append(f"使用指定 ν；有效 Re={params['lid_speed']*d['length']/d['nu']:.5g}")
            if s=="obstacle": warnings.append("周期障碍阵列；曲面按格点阶梯近似")
            if warnings:
                text += "\n" + "；".join(warnings)
            self.derived.setText(text)
            self.derived.setStyleSheet("color:#7e641d" if warnings else "color:#536c5e")
            self.start_button.setEnabled(True)
            self.start_button.setText("开始计算" if len(requests) == 1 else f"加入队列 · {len(requests)} 组")
        except (ValueError, TypeError, OverflowError) as error:
            self.derived.setText(str(error))
            self.derived.setStyleSheet("color:#b64039")
            self.start_button.setEnabled(False)

    def refresh_presets(self):
        self.preset.blockSignals(True)
        self.preset.clear()
        self.preset.addItem("参数模板", None)
        for name in self.store.presets():
            self.preset.addItem(name, name)
        self.preset.blockSignals(False)

    def load_preset(self):
        name = self.preset.currentData()
        if name:
            self.apply_params(self.store.presets()[name])
            self.sweep_enabled.setChecked(False)
            self.notice("已加载模板：" + name)

    def save_preset(self):
        try:
            params = self.form_params()
            name, accepted = W.QInputDialog.getText(self, "保存参数模板", "模板名称", text=self.name_input.text())
            if accepted and name.strip():
                if name in self.store.presets() and W.QMessageBox.question(self, "覆盖模板", f"覆盖已有模板“{name}”？") != W.QMessageBox.Yes:
                    return
                self.store.save_preset(name, params)
                self.refresh_presets()
                self.notice("模板已保存")
        except ValueError as error:
            self.error(str(error))

    def delete_preset(self):
        name = self.preset.currentData()
        if name and W.QMessageBox.question(self, "删除模板", f"删除参数模板“{name}”？已有结果不受影响。") == W.QMessageBox.Yes:
            self.store.delete_preset(name)
            self.refresh_presets()

    def enqueue(self):
        try:
            requests = self.form_requests()
            gates = receipt_status()
            if not all(gates[r["params"]["backend"]] for r in requests):
                raise ValueError("所选后端未通过当前源码核验，请重新执行对应验证")
            from research_validation import verified_research
            for request in requests:
                if generic_engine(request["params"]): verified_research(request["params"]["backend"])
            ids = self.store.enqueue(requests)
            self.store.set_setting("last_parameters", self.form_params())
            ensure_service(self.store)
            self.monitor_id = ids[0]
            self.monitor_key = None
            self.refresh(force=True)
            self.notice(f"已提交 {len(ids)} 组算例，按队列顺序运行")
        except (ValueError, OSError, RuntimeError) as error:
            self.error(str(error))

    def control(self, action):
        if not self.monitor_id:
            return
        try:
            if action == "cancel" and W.QMessageBox.question(self, "取消任务", "停止此任务并保存已计算状态？") != W.QMessageBox.Yes:
                return
            self.store.control(self.monitor_id, action)
            ensure_service(self.store)
            self.notice({"pause": "已提交暂停请求", "run": "已提交继续请求", "cancel": "已提交取消请求"}[action])
            self.refresh(force=True)
        except ValueError as error:
            self.error(str(error))

    def move_queue(self, direction):
        if self.monitor_id:
            self.store.move_queued(self.monitor_id, direction)
            self.refresh(force=True)

    def set_queue_paused(self, paused):
        self.store.set_setting("queue_paused", bool(paused))
        if not paused:
            ensure_service(self.store)
        self.notice("调度已暂停，当前计算状态不变" if paused else "调度已恢复")

    def refresh_validation(self):
        try:
            gates = receipt_status()
            from research_validation import verified_research
            for backend in ("cpu","array","fused"): verified_research(backend)
            self.validation_label.setText("CPU / GPU 源码核验通过" if all(gates.values()) else "存在未验证后端")
        except (OSError, ValueError, RuntimeError) as error:
            self.validation_label.setText("源码核验失败")
            self.notice(str(error), True)

    def manual_refresh(self):
        self.refresh_validation()
        self.monitor_key = None
        self.refresh(force=True)

    def refresh(self, force=False):
        if self.closing:
            return
        try:
            self.cases = self.store.cases(archived=True)
            active = sum(case["status"] in ACTIVE for case in self.cases)
            queued = sum(case["status"] == "queued" for case in self.cases)
            self.side_count.setText(f"数据 {len(self.cases)} 组\n运行 {active} · 排队 {queued}")
            summary = dict(data=sum(not c["archived"] for c in self.cases), active=active, queued=queued,
                           converged=sum(c["status"] == "converged" and not c["archived"] for c in self.cases))
            for key, value in summary.items():
                self.overview_values[key].setText(str(value))
            self.queue_paused.blockSignals(True)
            paused = self.store.setting("queue_paused", False)
            self.queue_paused.setChecked(paused)
            self.queue_paused.blockSignals(False)
            if queued and not paused and time.monotonic() - self.last_service_check > 5:
                ensure_service(self.store)
                self.last_service_check = time.monotonic()
            visible = [case for case in self.cases if not case["archived"]]
            live = sorted([c for c in visible if c["status"] in (*ACTIVE, "queued")], key=lambda c: (c["status"] == "queued", c["position"] or 0))
            recent = [case for case in visible if case["status"] not in (*ACTIVE, "queued")][:30]
            rows = live + recent
            if self.monitor_id is None and rows:
                self.monitor_id = rows[0]["id"]
            queue_key = (self.monitor_id, tuple((c["id"], c["name"], c["status"], c["metrics"].get("iteration")) for c in rows))
            if force or queue_key != self.queue_key:
                self.queue_key = queue_key
                scroll = self.queue.verticalScrollBar().value()
                self.queue.blockSignals(True)
                self.queue.clear()
                for case in rows:
                    params, metrics = case["params"], case["metrics"]
                    item = W.QTreeWidgetItem([case["name"], backend_label(params.get("backend")), physical_label(params),
                                             STATUS_NAMES.get(case["status"], case["status"]), fmt(metrics.get("iteration"), 10)])
                    item.setData(0, QtCore.Qt.UserRole, case["id"])
                    item.setToolTip(0, case["name"])
                    item.setForeground(3, QtGui.QColor("#008875" if case["status"] == "converged" else "#926629"))
                    self.queue.addTopLevelItem(item)
                    if case["id"] == self.monitor_id:
                        self.queue.setCurrentItem(item)
                self.queue.blockSignals(False)
                if not force:
                    self.queue.verticalScrollBar().setValue(scroll)
            self.refresh_monitor(force)
            if self.pages.currentIndex() == 1:
                self.rebuild_library(force)
            if self.pages.currentIndex() == 3:
                self.refresh_cpu()
        except (OSError, ValueError) as error:
            self.notice(str(error), True)

    def select_monitor(self, item, previous=None):
        if item:
            self.monitor_id = item.data(0, QtCore.Qt.UserRole)
            self.monitor_key = None
            self.refresh_monitor(True)

    def refresh_monitor(self, force=False):
        case = next((c for c in self.cases if c["id"] == self.monitor_id), None)
        for button in (self.pause_button, self.resume_button, self.cancel_button):
            button.setEnabled(False)
        if case is None:
            return
        self.pause_button.setEnabled(case["status"] in ("running", "starting") and case.get("desired") != "pause")
        self.resume_button.setEnabled(case["status"] == "paused" or case.get("desired") == "pause")
        self.cancel_button.setEnabled(case["status"] in (*ACTIVE, "queued"))
        text = f"{case['name']}  |  {STATUS_NAMES.get(case['status'], case['status'])}"
        if case["status"] in ACTIVE and case["metrics"].get("preview_warning"):
            text += "\n流场预览暂缓更新；计算未因预览错误中断"
        if case.get("error"):
            text += "\n" + case["error"]
        self.monitor_status.setText(text)
        for key, widget in self.metrics.items():
            widget.setText(fmt(case["metrics"].get(key), 8 if key == "iteration" else 5))
        budget = case["params"].get("max_iter") or 1
        self.budget.setValue(int(min(1000, 1000 * (case["metrics"].get("iteration") or 0) / budget)))
        key = (case["id"], case["updated_at"], encode(case["metrics"]), case["status"])
        if not self.monitor_busy and (force or key != self.monitor_key):
            self.monitor_busy = True
            self.monitor_key = key
            self.async_run(lambda: analysis.load_case(case, fields=True, preview=True), self.monitor_loaded, self.monitor_failed)

    def monitor_failed(self, message):
        self.monitor_busy = False
        self.notice(message, True)
        self.monitor_key = None

    def monitor_loaded(self, case):
        self.monitor_busy = False
        if case["id"] != self.monitor_id:
            self.refresh_monitor(True)
            return
        self.loaded_monitor = case
        figure = self.monitor_plot.figure
        figure.clear()
        for axis, metric, title in zip(figure.subplots(1, 2), ("residual", "mass_drift"), ("单步速度残差", "质量守恒相对误差")):
            x, y = analysis.curve(case["records"], metric)
            axis.plot(x, y, color=COLORS[0 if metric == "residual" else 1], lw=1.4)
            scale_values = [y]
            if metric == "residual" and case["params"].get("tol"):
                scale_values.append(np.array([case["params"]["tol"]]))
            if metric=="residual" and case["params"].get("scenario") in THERMAL:
                tx,ty = analysis.curve(case["records"],"thermal_residual")
                axis.plot(tx,ty,color=COLORS[2],lw=1.2,label="温度残差")
                scale_values.append(ty)
                axis.legend(fontsize=7)
            diagnostic_scale(axis, scale_values)
            axis.set_xlabel("迭代步数")
            axis.set_title(title)
            axis.grid(alpha=0.5)
            if metric == "residual" and case["params"].get("tol"):
                axis.axhline(case["params"]["tol"], color="#8a958d", ls="--", lw=0.8)
        self.monitor_plot.canvas.draw_idle()
        if self.monitor_tabs.currentIndex() == 1 and self.pages.currentIndex() == 0:
            delay = max(0, 3 - (time.monotonic() - self.last_field_render)) if case["status"] in ACTIVE else 0
            if not self.field_render_timer.isActive():
                self.field_render_timer.start(int(delay * 1000))
        detail = {key: case.get(key) for key in ("name", "status", "origin", "source_path", "data_dir", "params", "metrics", "error", "warnings")}
        log = Path(case["data_dir"]) / "worker.log"
        tail = ""
        if log.is_file():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 16000))
                tail = stream.read().decode("utf-8", errors="replace")
        self.details.setPlainText(json.dumps(detail, ensure_ascii=False, indent=2) + ("\n\n" + tail if tail else ""))
        self.refresh_experiment_views(case)
        self.field_options_changed(schedule=False)

    def field_options_changed(self, *args, schedule=True):
        mode = self.field_mode.currentData()
        self.field_density.setEnabled(mode in FLOW_MODES)
        self.field_region.setEnabled(mode != "vortices")
        self.field_extent.setEnabled(mode == "vortices" or self.field_region.currentData() != "full")
        show_flow=mode in FLOW_MODES
        show_extent=mode=='vortices' or self.field_region.currentData()!='full'
        for col in (0,1):self.field_controls.itemAtPosition(1,col).widget().setVisible(show_flow)
        for col in (2,3):self.field_controls.itemAtPosition(1,col).widget().setVisible(show_extent)
        case=self.loaded_monitor
        show_plane=bool(case and case['params'].get('lattice','').startswith('D3'))
        for col in (0,2,3):self.field_controls.itemAtPosition(2,col).widget().setVisible(show_plane)
        self.field_plot.canvas.setMinimumHeight(620 if mode == "vortices" else 360)
        if schedule:self.field_render_timer.start(120)

    def adjust_field_layout(self, *args):
        field_active = self.monitor_tabs.currentIndex() == 1
        self.metrics_bar.setVisible(not field_active)
        self.budget.setVisible(not field_active)
        if field_active != getattr(self, "field_focus", False):
            splitter = self.queue.parentWidget()
            if field_active:
                self.field_previous_sizes = splitter.sizes()
            self.queue.setMaximumHeight(110 if field_active else 16777215)
            if not field_active:
                splitter.setSizes(self.field_previous_sizes)
            self.field_focus = field_active

    def render_field(self, *args):
        if not hasattr(self, "field_plot"):
            return
        self.field_render_timer.stop()
        if self.closing or self.monitor_tabs.currentIndex() != 1 or self.pages.currentIndex() != 0:
            return
        case = self.loaded_monitor
        fields=self.current_fields()
        if not case or case["id"] != self.monitor_id or fields is None:
            self.field_plot.empty("尚无当前算例的流场文件")
            return
        data = analysis.slice_fields(fields,self.field_plane.currentData(),self.field_position.value())
        key = self.field_mode.currentData()
        speed = case["params"].get("lid_speed")
        label = "保存快照" if data.get('snapshot') else ("采样预览" if data["preview"] else "最终输出")
        iteration = data["preview_iteration"] if data["preview"] or data.get('snapshot') else case["metrics"].get("iteration")
        caption = f"{self.field_mode.currentText()} · {label} · {STATUS_NAMES.get(case['status'], case['status'])}\n{model_label(case['params'])} · {physical_label(case['params'])} · step {fmt(iteration, 10)}"
        if data.get('slice_caption'):caption+='\n'+data['slice_caption'].strip(' ·')
        try:
            draw_field(self.field_plot.figure, data, key, speed=speed, caption=caption,
                       region=self.field_region.currentData(), density=self.field_density.value(),
                       fraction=self.field_extent.value() / 100)
            if not data["finite"]:
                self.field_plot.figure.suptitle(caption + " · 含非有限值", fontsize=9)
        except ValueError as error:
            self.field_plot.empty(str(error))
        self.field_plot.toolbar.update()
        self.last_field_render = time.monotonic()
        self.field_plot.canvas.draw_idle()

    def expand_field(self):
        case = self.loaded_monitor
        fields=self.current_fields()
        if not case or case["id"] != self.monitor_id or fields is None:
            self.notice("尚无当前算例的流场文件")
            return
        data = analysis.slice_fields(fields,self.field_plane.currentData(),self.field_position.value())
        iteration = data["preview_iteration"] if data["preview"] or data.get('snapshot') else case["metrics"].get("iteration")
        label='保存快照' if data.get('snapshot') else ('采样预览' if data['preview'] else '最终输出')
        caption = f"{self.field_mode.currentText()} · {label} · {STATUS_NAMES.get(case['status'], case['status'])}\n{model_label(case['params'])} · {physical_label(case['params'])} · step {fmt(iteration, 10)}"
        if data.get('slice_caption'):caption+='\n'+data['slice_caption'].strip(' ·')
        dialog = W.QDialog(self)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        dialog.setWindowTitle(f"流场快照 · {case['name']} · step {fmt(iteration, 10)}")
        dialog.resize(1080, 850)
        dialog.setMinimumSize(640, 550)
        layout = W.QVBoxLayout(dialog)
        plot = Plot(dialog)
        layout.addWidget(plot)
        try:
            draw_field(plot.figure, data, self.field_mode.currentData(), speed=case["params"].get("lid_speed"),
                       caption=caption, region=self.field_region.currentData(), density=self.field_density.value(),
                       fraction=self.field_extent.value() / 100)
        except ValueError as error:
            dialog.deleteLater()
            self.notice(str(error), True)
            return
        self.field_windows.append(dialog)
        dialog.destroyed.connect(lambda: self.field_windows.remove(dialog))
        dialog.show()
        plot.canvas.draw_idle()

    def rebuild_library(self, force=False):
        key = (tuple((c["id"], c["updated_at"], c["status"], encode(c["metrics"])) for c in self.cases),
               self.search.text(), self.status_filter.currentData(), self.show_archived.isChecked())
        if not force and key == self.library_key:
            return
        self.library_key = key
        selected = self.library.currentItem()
        current_id = selected.data(0, QtCore.Qt.UserRole) if selected else None
        scroll = self.library.verticalScrollBar().value()
        self.library.blockSignals(True)
        self.library.clear()
        query = self.search.text().strip().lower()
        status_filter = self.status_filter.currentData()
        for case in self.cases:
            if case["archived"] and not self.show_archived.isChecked():
                continue
            if query and query not in " ".join(str(case.get(k, "")) for k in ("name", "group_name", "notes", "params")).lower():
                continue
            if status_filter == "converged" and case["status"] != "converged":
                continue
            if status_filter == "unfinished" and case["status"] in (*ACTIVE, "queued", "converged"):
                continue
            if status_filter == "active" and case["status"] not in (*ACTIVE, "queued"):
                continue
            p, m = case["params"], case["metrics"]
            item = W.QTreeWidgetItem([case["name"] + (" [归档]" if case["archived"] else ""), case["group_name"],
                                     backend_label(p.get("backend")), grid_label(p), physical_label(p),
                                     STATUS_NAMES.get(case["status"], case["status"]), fmt(m.get("iteration"), 10),
                                     fmt(m.get("residual")), fmt(m.get("mass_drift")), case["created_at"][:19].replace("T", " ")])
            item.setData(0, QtCore.Qt.UserRole, case["id"])
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(0, QtCore.Qt.Checked if case["id"] in self.checked else QtCore.Qt.Unchecked)
            item.setToolTip(0, case["name"] + "\n" + case["data_dir"])
            self.library.addTopLevelItem(item)
            if case["id"] == current_id:
                self.library.setCurrentItem(item)
        self.library.blockSignals(False)
        self.library.verticalScrollBar().setValue(scroll)
        self.selected_label.setText(f"已选 {len(self.checked)} 组")

    def library_checked(self, item, column):
        case_id = item.data(0, QtCore.Qt.UserRole)
        if item.checkState(0) == QtCore.Qt.Checked:
            self.checked.add(case_id)
        else:
            self.checked.discard(case_id)
        self.selected_label.setText(f"已选 {len(self.checked)} 组")

    def library_current(self, item, previous=None):
        if not item:
            return
        case = self.store.get(item.data(0, QtCore.Qt.UserRole))
        if case:
            self.edit_name.setText(case["name"])
            self.edit_group.setText(case["group_name"])
            self.edit_notes.setText(case["notes"])
            self.source_label.setText(f"{case['origin']} · {case['data_dir']}")

    def selected_cases(self):
        identifiers = self.checked or {item.data(0, QtCore.Qt.UserRole) for item in self.library.selectedItems()}
        return [case for case in self.cases if case["id"] in identifiers]

    def select_visible(self):
        for index in range(self.library.topLevelItemCount()):
            self.checked.add(self.library.topLevelItem(index).data(0, QtCore.Qt.UserRole))
        self.rebuild_library(force=True)

    def clear_selection(self):
        self.checked.clear()
        self.library.clearSelection()
        self.rebuild_library(force=True)

    def save_metadata(self):
        item = self.library.currentItem()
        if not item:
            return
        if not self.edit_name.text().strip():
            self.error("名称不能为空")
            return
        self.store.update_case(item.data(0, QtCore.Qt.UserRole), name=self.edit_name.text().strip(),
                               group_name=self.edit_group.text().strip(), notes=self.edit_notes.text())
        self.refresh(force=True)
        self.notice("元数据已保存，原始计算文件未修改")

    def reuse_selected(self):
        item = self.library.currentItem()
        if not item:
            self.error("先选择一组数据")
            return
        case = self.store.get(item.data(0, QtCore.Qt.UserRole))
        self.apply_params(case["params"])
        self.name_input.setText(case["name"] + " · 复算")
        self.group_input.setText(case["group_name"])
        self.sweep_enabled.setChecked(False)
        self.show_page(0)
        if case["params"].get("backend") not in ("cpu", "array", "fused"):
            self.notice("来源后端未标注；当前计算后端：" + self.backend.currentText())

    def archive_selected(self):
        cases = self.selected_cases()
        if not cases:
            self.error("先选择数据")
            return
        if any(case["status"] in (*ACTIVE, "queued") for case in cases):
            self.error("活动或排队任务不能归档")
            return
        restore = all(case["archived"] for case in cases)
        for case in cases:
            self.store.update_case(case["id"], archived=0 if restore else 1)
        self.checked.clear()
        self.refresh(force=True)
        self.notice("已恢复所选记录" if restore else "所选记录已归档，数据文件保留")

    def discover_results(self):
        self.async_run(self.store.discover, lambda result: (self.refresh(force=True), self.notice(f"扫描完成，登记 {len(result[0])} 组；异常 {len(result[1])} 项")))

    def import_results(self):
        path = W.QFileDialog.getExistingDirectory(self, "导入结果目录", str(ROOT))
        if not path:
            return
        def perform():
            folder = Path(path)
            candidates = [folder] if (folder / "config.json").is_file() else [p.parent for p in folder.glob("*/config.json")]
            if not candidates:
                raise ValueError("没有找到包含 config.json 的结果目录")
            successes, errors = [], []
            for item in candidates:
                try:
                    data = analysis.load_fields(item)
                    if data is None:
                        raise ValueError("缺少最终流场 NPZ")
                    successes.append(self.store.register_result(item, copy_files=True, group="导入数据"))
                except (OSError, ValueError, KeyError) as error:
                    errors.append(f"{item.name}: {error}")
            return successes, errors
        def done(result):
            self.refresh(force=True)
            self.notice(f"已导入 {len(result[0])} 组；未导入 {len(result[1])} 组", bool(result[1]))
            if result[1]:
                self.error("\n".join(result[1][:10]))
        self.async_run(perform, done)

    def export_csv(self):
        cases = self.selected_cases()
        if not cases:
            self.error("先选择要导出的数据")
            return
        path, _ = W.QFileDialog.getSaveFileName(self, "导出指标", str(self.store.workspace / "comparison.csv"), "CSV (*.csv)")
        if path:
            self.async_run(lambda: analysis.export_csv(cases, path), lambda _: self.notice("已导出：" + path))

    def export_bundle(self):
        cases = self.selected_cases()
        if not cases:
            self.error("先选择要导出的数据")
            return
        if any(case["status"] in (*ACTIVE, "queued") for case in cases):
            self.error("数据包只导出已结束的记录，避免混入写入中的文件")
            return
        path, _ = W.QFileDialog.getSaveFileName(self, "导出数据包", str(self.store.workspace / "lbm_results.zip"), "ZIP (*.zip)")
        if path:
            self.async_run(lambda: analysis.export_bundle(cases, path), lambda _: self.notice("已导出：" + path))

    def compare_selected(self):
        cases = self.selected_cases()
        if not cases:
            self.error("先选择用于对比的数据")
            return
        if len(cases) > 8:
            self.error("同一张对比图最多选择 8 组；更多记录可导出 CSV")
            return
        self.checked = {case["id"] for case in cases}
        self.show_page(2)

    def load_analysis(self):
        cases = [case for case in self.cases if case["id"] in self.checked]
        if not cases or len(cases) > 8:
            self.comparison_plot.empty("请选择 1 至 8 组数据")
            return
        self.analysis_version += 1
        version = self.analysis_version
        self.analysis_selection.setText(f"读取 {len(cases)} 组数据…")
        def done(data):
            if version != self.analysis_version:
                return
            self.analysis_data = data
            self.reference_case.blockSignals(True)
            previous_reference = self.reference_case.currentData()
            self.reference_case.clear()
            for case in data:
                self.reference_case.addItem("基准：" + case["name"], case["id"])
            preferred = previous_reference or next((case["id"] for case in data if case["params"].get("backend") == "cpu"), data[0]["id"])
            self.reference_case.setCurrentIndex(max(0, self.reference_case.findData(preferred)))
            self.reference_case.blockSignals(False)
            self.analysis_selection.setText(f"{len(data)} 组数据 · {datetime.now():%H:%M:%S}")
            self.comparison_table.clear()
            for case in data:
                p, m = case["params"], case["metrics"]
                item = W.QTreeWidgetItem([case["name"], backend_label(p.get("backend")), f"{grid_label(p)} / {physical_label(p)}",
                                         STATUS_NAMES.get(case["status"], case["status"]), fmt(m.get("iteration"), 10), fmt(m.get("residual")),
                                         fmt(m.get("mass_drift")), fmt(m.get("elapsed_seconds"))])
                item.setToolTip(0, short_label(case))
                self.comparison_table.addTopLevelItem(item)
            self.render_analysis()
        self.async_run(lambda: [analysis.load_case(case, fields=True) for case in cases], done)

    def render_analysis(self, *args):
        if not hasattr(self, "comparison_plot"):
            return
        mode = self.analysis_mode.currentData()
        self.analysis_metric.setEnabled(mode in ("history", "sweep"))
        self.scan_axis.setVisible(mode == "sweep")
        self.reference_case.setVisible(mode == "difference")
        data = self.analysis_data
        if not data:
            self.comparison_plot.empty("尚未选择数据")
            return
        figure = self.comparison_plot.figure
        figure.clear()
        warnings = []
        if any(case["status"] != "converged" for case in data):
            warnings.append("包含未收敛或未完成结果")
        if len({analysis.physical_signature(case) for case in data}) > 1:
            warnings.append("物理参数或网格不同")
        for case in data:
            warnings.extend(case["warnings"])
        try:
            if mode == "history":
                axis = figure.add_subplot(111)
                metric = self.analysis_metric.currentData()
                series = []
                for index, case in enumerate(data):
                    x, y = analysis.curve(case["records"], metric)
                    series.append(y)
                    axis.plot(x, y, color=COLORS[index], lw=1.4, label=short_label(case))
                if metric in ("residual", "mass_drift"):
                    diagnostic_scale(axis, series)
                axis.set_xlabel("迭代步数")
                axis.set_ylabel(self.analysis_metric.currentText())
                axis.legend(fontsize=8)
                axis.grid(alpha=0.5)
            elif mode == "centerlines":
                left, right = figure.subplots(1, 2)
                for index, case in enumerate(data):
                    if case["fields"] is None:
                        continue
                    if case["fields"]["rho"].ndim==3:
                        warnings.append("三维中心线取最接近 z/Lz=0.5 的 xy 截面")
                    values = analysis.centerlines(case["fields"], case["params"].get("lid_speed"))
                    left.plot(values["u"], values["y"], color=COLORS[index], label=short_label(case))
                    right.plot(values["x"], values["v"], color=COLORS[index])
                left.set(xlabel="ux / U", ylabel="y / Ly", title="x / Lx = 0.5")
                right.set(xlabel="x / Lx", ylabel="uy / U", title="y / Ly = 0.5")
                left.legend(fontsize=7)
                left.grid(alpha=0.5)
                right.grid(alpha=0.5)
            elif mode == "sweep":
                axis = figure.add_subplot(111)
                metric, scan = self.analysis_metric.currentData(), self.scan_axis.currentData()
                for index, case in enumerate(data):
                    x, y = case["params"].get(scan), case["metrics"].get(metric)
                    if x is None or y is None:
                        continue
                    axis.scatter(x, y, s=64, edgecolors=COLORS[index], facecolors=COLORS[index] if case["status"] == "converged" else "none",
                                 marker="o" if case["status"] == "converged" else "s", label=short_label(case))
                axis.set_xlabel(self.scan_axis.currentText())
                axis.set_ylabel(self.analysis_metric.currentText())
                if metric in ("residual", "mass_drift"):
                    axis.set_yscale("symlog", linthresh=1e-12)
                axis.legend(fontsize=8)
                axis.grid(alpha=0.5)
            else:
                if len(data) != 2:
                    raise ValueError("流场差值需要恰好两组数据")
                first = next((case for case in data if case["id"] == self.reference_case.currentData()), data[0])
                second = next(case for case in data if case["id"] != first["id"])
                result = analysis.difference(first, second)
                warnings.extend(result["warnings"])
                axis = figure.add_subplot(111)
                field = analysis.slice_fields(dict(first["fields"],difference=result["values"]))
                image = axis.pcolormesh(field["x"], field["y"], field["difference"], shading="auto", cmap="magma")
                if first["fields"]["rho"].ndim==3:
                    warnings.append("三维差值指标覆盖全部体积；图示为 z 中截面")
                figure.set_layout_engine("compressed")
                figure.colorbar(image, ax=axis, pad=0.04, panchor=False)
                axis.set_aspect("equal")
                axis.set(xlabel="x / Lx", ylabel="y / Ly", title="|u对照 − u基准| · 格子单位")
                relative = fmt(result['relative_velocity_l2']) if result['relative_velocity_l2'] is not None else "未定义"
                warnings.append(f"max |Δu| = {fmt(result['max_velocity_difference'])}；相对 L2 = {relative}；max |Δrho| = {fmt(result['max_density_difference'])}")
            self.comparison_plot.canvas.draw_idle()
        except (ValueError, KeyError) as error:
            self.comparison_plot.empty(str(error))
            warnings.append(str(error))
        self.analysis_warning.setText("；".join(dict.fromkeys(warnings)) if warnings else "所选结果均已满足各自速度残差条件；质量漂移单独列示")

    def refresh_cpu(self):
        root = ROOT.parent / "BGK_NEE_Windows_20260928_191621/simulation"
        state = read_json(root / "search_state.json", {})
        control = read_json(root / "dashboard_control/state.json", {})
        current = state.get("current") or {}
        target = current.get("path")
        info = []
        for role in ("supervisor", "child"):
            identity = control.get("target", {}).get(role, {})
            try:
                process = psutil.Process(identity["pid"])
                if abs(process.create_time() - identity["started"]) > 0.001:
                    raise ValueError("PID identity changed")
                info.append(f"{role}: PID {process.pid} · {process.status()}")
            except (psutil.Error, KeyError, ValueError):
                info.append(f"{role}: 当前身份未核验")
        self.cpu_status.setText("原 CPU 搜索 · " + " | ".join(info))
        self.cpu_detail.setPlainText(json.dumps(dict(root=str(root), current=current, last_control=control.get("action")), ensure_ascii=False, indent=2))
        if target:
            path = Path(target) / "progress.jsonl"
            key = (str(path), path.stat().st_mtime_ns if path.is_file() else 0)
            if key != self.cpu_key:
                self.cpu_key = key
                def done(result):
                    records, warnings = result
                    self.cpu_plot.figure.clear()
                    axis = self.cpu_plot.figure.add_subplot(111)
                    x, y = analysis.curve(records, "residual")
                    axis.plot(x, y, color=COLORS[0])
                    axis.set_yscale("symlog", linthresh=1e-12)
                    axis.set(xlabel="迭代步数", ylabel="单步残差", title=f"原 CPU · Re {current.get('Re', '')}")
                    axis.grid(alpha=0.5)
                    self.cpu_plot.canvas.draw_idle()
                self.async_run(lambda: analysis.history(target), done)
        else:
            self.cpu_plot.empty("未找到原 CPU 当前任务记录")

    def open_current(self):
        if self.monitor_id:
            case = self.store.get(self.monitor_id)
            folder = Path(case["data_dir"])
            if folder.is_dir():
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))
            else:
                self.notice("任务尚未创建结果目录")

    def open_case(self, case_id):
        self.monitor_id = case_id
        self.monitor_key = None
        self.show_page(0)
        self.refresh(force=True)

    def closeEvent(self, event):
        self.timer.stop()
        self.snapshot_timer.stop()
        self.field_render_timer.stop()
        for dialog in list(self.field_windows):
            dialog.close()
        self.closing = True
        try:
            self.store.set_setting("last_parameters", self.form_params())
        except ValueError:
            pass
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not hasattr(self, "sidebar"):
            return
        compact = self.width() < 1160
        self.sidebar.setFixedWidth(84 if compact else 226)
        self.brand.setText("LBM" if compact else "LBM Atlas")
        self.sidebar.layout().setContentsMargins(12 if compact else 20, 29, 12 if compact else 20, 22)
        for widget in (self.side_title, self.nav_label, self.side_model, self.side_count):
            widget.setVisible(not compact)
        for index, label in enumerate(PAGE_NAMES):
            button = self.navigation.button(index)
            button.setText("" if compact else label)
            button.setIconSize(QtCore.QSize(20 if compact else 17, 20 if compact else 17))
            button.setAccessibleName(label)
        self.atlas_link.setText("" if compact else "学术资料站")
        self.overview.setVisible(self.height() >= 820)
        self.topbar.setFixedHeight(52 if compact else 67)
        margin = 20 if compact else 30
        self.content_layout.setContentsMargins(margin, 12 if compact else 19, margin, 10)
        self.adjust_field_layout()


STYLE = """
QWidget {font-family:'Source Sans 3','Noto Sans SC','Microsoft YaHei UI';font-size:13px;color:#202d2b;}
QWidget#shell,QWidget#parameterPanel {background:#f7f8f6;}
QWidget#sidebar {background:#ffffff;border-right:1px solid #dbe2dd;}
QWidget#topbar {background:#f7f8f6;border-bottom:1px solid #dbe2dd;}
QLabel {background:transparent;}
QLabel#brand {font-family:'Source Serif 4','Noto Serif SC';font-size:28px;font-weight:600;color:#202d2b;}
QLabel#sideTitle {font-size:10px;color:#61706b;}
QLabel#sideInfo {font-size:11px;color:#78857f;line-height:1.7;}
QLabel#navLabel {font-size:12px;color:#78857f;padding-left:12px;}
QLabel#title {font-family:'Noto Serif SC','Source Serif 4';font-size:30px;font-weight:600;color:#202d2b;}
QLabel#eyebrow {font-size:10px;color:#78857f;}
QLabel#modelLabel {font-size:11px;color:#61706b;}
QWidget#overview {border-top:1px solid #202d2b;border-bottom:1px solid #dbe2dd;}
QWidget#overviewItem {border-right:1px solid #dbe2dd;}
QLabel#overviewValue {font-family:'Source Serif 4';font-size:28px;color:#202d2b;}
QLabel#section {font-family:'Noto Serif SC','Source Serif 4';font-size:15px;font-weight:600;}
QLabel#muted {font-size:12px;color:#61706b;}
QLabel#derived {background:#f0f3f1;border-left:2px solid #dbe2dd;padding:10px;font-size:11px;}
QLabel#status {padding:6px 10px;background:transparent;border-left:2px solid #23594b;}
QLabel#metric {font-family:'Source Serif 4';font-size:22px;color:#202d2b;}
QLabel#warning {color:#936b31;font-size:12px;}
QPushButton,QToolButton {background:#ffffff;border:1px solid #dbe2dd;border-radius:3px;padding:6px 10px;min-height:18px;}
QToolButton {padding:3px;}
QPushButton:hover,QToolButton:hover {background:#e9f0eb;border-color:#9eb3a8;}
QPushButton:pressed,QToolButton:pressed {background:#dce7df;}
QPushButton:disabled,QToolButton:disabled {color:#9aa79f;background:#f0f3f1;border-color:#e4e8e5;}
QPushButton#primary {background:#23594b;color:white;border:1px solid #23594b;font-weight:500;}
QPushButton#primary:hover {background:#174839;}
QPushButton#primary:disabled {background:#a5b4ad;color:#edf1ee;border-color:#a5b4ad;}
QPushButton#nav {text-align:left;background:transparent;border:none;border-left:2px solid transparent;border-radius:0;color:#61706b;padding:9px 11px;}
QPushButton#nav:hover {background:#f0f3f1;color:#23594b;}
QPushButton#nav:checked {background:#e9f0eb;color:#23594b;border-left:2px solid #23594b;}
QLineEdit,QComboBox,QPlainTextEdit,QSpinBox,QDoubleSpinBox {background:white;border:1px solid #dbe2dd;border-radius:3px;padding:5px 7px;min-height:21px;selection-background-color:#e0ebe4;selection-color:#202d2b;}
QLineEdit:focus,QComboBox:focus {border-color:#23594b;}
QLineEdit:disabled {color:#9aa79f;background:#f0f3f1;}
QComboBox QAbstractItemView {background:#ffffff;selection-background-color:#e9f0eb;selection-color:#23594b;border:1px solid #dbe2dd;}
QTreeWidget {background:#f7f8f6;alternate-background-color:#f0f3f1;border:none;border-top:1px solid #dbe2dd;border-bottom:1px solid #dbe2dd;}
QTreeWidget::item {min-height:29px;padding:2px 4px;}
QTreeWidget::item:selected {background:#e3ece5;color:#23594b;}
QTreeWidget::item:hover {background:#ecf1ed;}
QHeaderView::section {background:#f7f8f6;color:#61706b;border:none;border-bottom:1px solid #dbe2dd;padding:8px 6px;font-size:12px;}
QTabWidget::pane {border:none;border-top:1px solid #dbe2dd;background:#f7f8f6;}
QTabBar::tab {padding:8px 16px;background:transparent;border:none;border-bottom:2px solid transparent;color:#61706b;}
QTabBar::tab:selected {background:transparent;color:#23594b;border-bottom:2px solid #23594b;}
QTabBar::tab:hover {background:#e9f0eb;}
QSplitter::handle {background:#dbe2dd;}
QSplitter::handle:horizontal {width:1px;}
QSplitter::handle:vertical {height:2px;}
QProgressBar {background:#e4e9e5;border:none;}
QProgressBar::chunk {background:#23594b;}
QScrollArea {border:none;background:transparent;}
QScrollBar:vertical {width:8px;background:#f0f3f1;}
QScrollBar::handle:vertical {background:#c3cec7;border-radius:3px;min-height:24px;}
QScrollBar:horizontal {height:8px;background:#f0f3f1;}
QScrollBar::handle:horizontal {background:#c3cec7;border-radius:3px;min-width:24px;}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical {height:0;}
QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal {width:0;}
QToolBar {background:#f7f8f6;border:none;spacing:3px;}
QToolTip {background:#ffffff;color:#202d2b;border:1px solid #dbe2dd;padding:5px;}
"""
