"""Local Qt dashboard: read files and explicitly control verified processes in a worker."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets as W
from matplotlib import rcParams, font_manager
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from reader import collect
from control import execute

ROOT = Path(__file__).resolve().parent.parent / 'simulation'
FONT_DIR = Path(__file__).resolve().parent / 'fonts'
for font_path in FONT_DIR.glob('*-Regular.ttf'):
    font_manager.fontManager.addfont(str(font_path))

rcParams.update({'font.family': ['Source Sans 3', 'Noto Sans SC', 'DejaVu Sans'], 'font.size': 10,
                 'axes.unicode_minus': False, 'axes.labelcolor': '#657987', 'text.color': '#344f5e',
                 'xtick.color': '#657987', 'ytick.color': '#657987', 'axes.edgecolor': '#d9e2e7',
                 'axes.titlesize': 10, 'axes.titlepad': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                 'axes.labelsize': 8, 'grid.color': '#dce4e9', 'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False})


def fmt(value: object) -> str:
    if value is None:
        return '未提供'
    try:
        n = float(value)
        return f'{n:.7g}' if math.isfinite(n) else str(value) + '（非有限）'
    except (ValueError, TypeError):
        return str(value)


class ReadWorker(QtCore.QObject):
    ready = QtCore.Signal(object)
    controlled = QtCore.Signal(object)

    @QtCore.Slot(str, str, object)
    def control(self, root: str, action: str, token: dict) -> None:
        try:
            self.controlled.emit(execute(Path(root), action, token))
        except Exception as exc:
            self.controlled.emit({"error": str(exc)})

    @QtCore.Slot(str, str)
    def read(self, root: str, selected: str) -> None:
        try:
            self.ready.emit(collect(Path(root), selected))
        except Exception as exc:
            self.ready.emit({'error': f'{type(exc).__name__}: {exc}'})


class Dashboard(W.QMainWindow):
    read_requested = QtCore.Signal(str, str)
    control_requested = QtCore.Signal(str, str, object)
    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root, self.busy, self.data = root, False, None
        self.plot_key = None
        self.control_pending = False
        self.confirmation = None
        self.setWindowTitle('LBM · 后台计算观测台')
        self.resize(1280, 880)
        self.setMinimumSize(1040, 760)
        main = W.QWidget(); main.setObjectName('shell'); self.setCentralWidget(main)
        shell = W.QHBoxLayout(main); shell.setContentsMargins(0, 0, 0, 0); shell.setSpacing(0)
        sidebar = W.QFrame(); sidebar.setObjectName('sidebar'); sidebar.setFixedWidth(176)
        side = W.QVBoxLayout(sidebar); side.setContentsMargins(22, 30, 20, 24); side.setSpacing(12)
        brand = W.QLabel('LBM'); brand.setObjectName('brand'); side.addWidget(brand)
        lab = W.QLabel('COMPUTE OBSERVATORY'); lab.setObjectName('eyebrow'); side.addWidget(lab)
        side.addSpacing(36)
        section = W.QLabel('工作空间'); section.setObjectName('sideMuted'); side.addWidget(section)
        current_nav = W.QLabel('  ◉   计算观测'); current_nav.setObjectName('sideActive'); side.addWidget(current_nav)
        self.side_case = W.QLabel('等待运行参数'); self.side_case.setWordWrap(True); self.side_case.setObjectName('sideInfo'); side.addWidget(self.side_case)
        side.addSpacing(20)
        label = W.QLabel('后台计算控制'); label.setObjectName('sideMuted'); side.addWidget(label)
        self.control_buttons = {}
        for action, text in [('pause', 'Ⅱ  暂停计算'), ('resume', '▷  继续计算'), ('terminate', '□  终止任务')]:
            button = W.QPushButton(text); button.setObjectName('danger' if action == 'terminate' else 'controlButton')
            button.setEnabled(False); button.clicked.connect(lambda checked=False, a=action: self.request_control(a))
            self.control_buttons[action] = button; side.addWidget(button)
        self.control_note = W.QLabel('核验进程后可用'); self.control_note.setObjectName('sideMuted'); self.control_note.setWordWrap(True); side.addWidget(self.control_note)
        side.addStretch()
        tag = W.QLabel('本地数据 / 进程控制'); tag.setObjectName('sideTag'); side.addWidget(tag)
        note = W.QLabel('5 秒刷新 / 本地文件\n关闭看板不影响计算'); note.setObjectName('sideMuted'); side.addWidget(note)
        shell.addWidget(sidebar)
        content = W.QWidget(); shell.addWidget(content, 1)
        layout = W.QVBoxLayout(content); layout.setContentsMargins(26, 24, 26, 18); layout.setSpacing(13)
        heading = W.QHBoxLayout(); layout.addLayout(heading)
        title = W.QLabel('后台计算观测台'); title.setObjectName('title'); heading.addWidget(title); heading.addStretch()
        badge = W.QLabel(' LOCAL  /  D2Q9 · BGK '); badge.setObjectName('badge'); heading.addWidget(badge)
        self.subtitle = W.QLabel('正在读取磁盘记录'); self.subtitle.setObjectName('muted'); layout.addWidget(self.subtitle)
        bar = W.QHBoxLayout(); bar.setSpacing(10); layout.addLayout(bar)
        self.selector = W.QComboBox(); self.selector.addItem('跟随当前计算', ''); self.selector.setMinimumWidth(280)
        self.selector.setSizePolicy(W.QSizePolicy.Expanding, W.QSizePolicy.Fixed)
        self.selector.currentIndexChanged.connect(self.selection_changed); bar.addWidget(self.selector, 1)
        self.pause = W.QCheckBox('冻结画面'); self.pause.toggled.connect(self.pause_changed); bar.addWidget(self.pause)
        button = W.QPushButton('刷新'); button.setObjectName('primary'); button.clicked.connect(self.refresh); bar.addWidget(button)
        open_button = W.QPushButton('数据目录 ↗'); open_button.clicked.connect(self.open_folder); bar.addWidget(open_button)
        self.status = W.QLabel('正在读取真实数据…'); self.status.setObjectName('status'); self.status.setWordWrap(True); layout.addWidget(self.status)
        cards = W.QHBoxLayout(); cards.setSpacing(12); layout.addLayout(cards); self.cards = []
        for label, caption in [('已记录步数', 'ITERATIONS'), ('单步相对 L2 残差', 'VELOCITY RESIDUAL'),
                               ('质量相对漂移', '|M − M₀| / M₀'), ('空间最大 Ma', 'MAXIMUM MACH')]:
            frame = W.QFrame(); frame.setObjectName('card'); box = W.QVBoxLayout(frame)
            box.setContentsMargins(17, 15, 17, 14); box.setSpacing(7)
            name = W.QLabel(label); name.setObjectName('cardLabel'); box.addWidget(name)
            value = W.QLabel('—'); value.setObjectName('value'); box.addWidget(value)
            hint = W.QLabel(caption); hint.setObjectName('caption'); box.addWidget(hint)
            self.cards.append(value); cards.addWidget(frame, 1)
        budget_box = W.QVBoxLayout(); budget_box.setSpacing(7); layout.addLayout(budget_box)
        self.budget_label = W.QLabel('步数预算'); self.budget_label.setObjectName('muted'); budget_box.addWidget(self.budget_label)
        self.budget = W.QProgressBar(); self.budget.setRange(0, 10000); self.budget.setTextVisible(False); self.budget.setFixedHeight(5); budget_box.addWidget(self.budget)
        self.parameters = W.QLabel(); self.parameters.setObjectName('parameters'); self.parameters.setWordWrap(True); layout.addWidget(self.parameters)
        self.tabs = W.QTabWidget(); self.tabs.setDocumentMode(True); layout.addWidget(self.tabs, 1)
        plot_page = W.QWidget(); pv = W.QVBoxLayout(plot_page); pv.setContentsMargins(14, 10, 14, 10); pv.setSpacing(5)
        self.fig = Figure(figsize=(11, 5), constrained_layout=True, facecolor='#ffffff')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setMinimumHeight(300)
        tools_row = W.QHBoxLayout(); chart_title = W.QLabel('诊断趋势'); chart_title.setObjectName('chartTitle'); tools_row.addWidget(chart_title); tools_row.addStretch()
        toolbar = NavigationToolbar2QT(self.canvas, self); toolbar.setIconSize(QtCore.QSize(16, 16)); toolbar.setMaximumHeight(29)
        tools_row.addWidget(toolbar); pv.addLayout(tools_row); pv.addWidget(self.canvas, 1)
        chart_note = W.QLabel('实际报告点 · 无平滑 / 插值 · 残差 symlog，线性区 ±10⁻¹²，保留零值 · 缺失值留空')
        chart_note.setObjectName('chartNote'); chart_note.setWordWrap(True); pv.addWidget(chart_note)
        plot_scroll = W.QScrollArea(); plot_scroll.setWidgetResizable(True)
        plot_scroll.setFrameShape(W.QFrame.NoFrame); plot_scroll.setWidget(plot_page)
        self.tabs.addTab(plot_scroll, '运行曲线')
        field_page = W.QWidget(); fv = W.QVBoxLayout(field_page); fv.setContentsMargins(18, 16, 18, 16)
        self.field_note = W.QLabel(); self.field_note.setWordWrap(True); fv.addWidget(self.field_note)
        self.field_fig = Figure(figsize=(8, 5), constrained_layout=True); self.field_canvas = FigureCanvasQTAgg(self.field_fig); fv.addWidget(self.field_canvas)
        self.tabs.addTab(field_page, '流场文件')
        self.table = W.QTableWidget(); self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(['步数', '记录耗时 / s', '残差', '质量漂移', '最大 Ma', '记录状态'])
        self.table.setEditTriggers(W.QAbstractItemView.NoEditTriggers); self.table.horizontalHeader().setSectionResizeMode(W.QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True); self.table.setShowGrid(False); self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(W.QAbstractItemView.SelectRows); self.table.verticalHeader().setDefaultSectionSize(32)
        self.tabs.addTab(self.table, '原始报告值')
        self.sources = W.QPlainTextEdit(); self.sources.setReadOnly(True); self.tabs.addTab(self.sources, '来源与搜索结果')
        self.foot = W.QLabel('速度达标不等于质量守恒、基准验证或物理结论成立。'); self.foot.setObjectName('muted'); self.foot.setWordWrap(True); layout.addWidget(self.foot)
        self.setStyleSheet('''QWidget {color:#263848;font-family:"Source Sans 3","Noto Sans SC","PingFang SC";font-size:12px;}
        QWidget#shell {background:#f3f5f7;} QLabel {background:transparent;}
        QFrame#sidebar {background:#172e3b;} QLabel#brand {font-family:"Source Serif 4","Noto Serif SC";color:#f1f7f8;font-size:36px;font-weight:600;letter-spacing:3px;}
        QLabel#eyebrow {color:#9fb7c3;font-size:8px;letter-spacing:1px;}
        QLabel#sideMuted {color:#adc0ca;font-size:11px;line-height:1.6;}
        QLabel#sideActive {background:#294956;color:#e2f5f1;padding:13px 0;border-radius:6px;font-weight:600;}
        QLabel#sideInfo {color:#c3d2d9;font-size:12px;padding-top:10px;}
        QLabel#sideTag {color:#9bddcf;font-size:12px;}
        QLabel#title {font-family:"Source Serif 4","Noto Serif SC";font-size:25px;font-weight:600;color:#172e3b;}
        QLabel#badge {color:#526b79;background:#e7edf1;border-radius:4px;padding:5px 8px;font-size:10px;letter-spacing:1px;}
        QLabel#muted,QLabel#chartNote {color:#657987;font-size:11px;}
        QLabel#status {background:#e6efed;color:#265f55;border:1px solid #d5e5e0;padding:9px 12px;border-radius:6px;}
        QFrame#card {background:white;border:1px solid #e0e6eb;border-radius:9px;}
        QLabel#cardLabel {color:#566c7a;font-size:12px;}
        QLabel#value {font-family:"Source Serif 4","Noto Serif SC";font-size:25px;font-weight:500;color:#173d4b;}
        QLabel#caption {color:#758591;font-size:9px;letter-spacing:1px;}
        QLabel#parameters {color:#566d7b;font-size:11px;padding:4px 0;}
        QPushButton,QComboBox {min-height:20px;padding:7px 12px;background:white;border:1px solid #d7e0e6;border-radius:6px;}
        QPushButton:disabled {color:#87949d;background:#e9eef1;border-color:#d7e0e6;}
        QPushButton#controlButton {background:#294956;color:#e5f2ef;border-color:#43616b;}
        QPushButton#controlButton:disabled {color:#738a95;background:#223c49;border-color:#304b58;}
        QPushButton#danger {color:#9b443c;background:#faefed;border-color:#e7c9c4;}
        QPushButton#danger:disabled {color:#938d8c;background:#293e47;border-color:#3e535b;}
        QLabel#status[tone="warning"] {background:#f5eddd;color:#805c20;border-color:#e7d7b7;}
        QLabel#status[tone="error"] {background:#f7e8e5;color:#923e37;border-color:#e5c6c0;}
        QPushButton:hover {background:#edf4f4;border-color:#9bbeb8;} QPushButton:pressed {background:#deebea;}
        QPushButton#primary {background:#266b62;color:white;border:1px solid #266b62;padding-left:18px;padding-right:18px;}
        QPushButton#primary:hover {background:#1d5a53;} QComboBox::drop-down {border:0;width:24px;}
        QComboBox QAbstractItemView {background:white;selection-background-color:#e1efec;selection-color:#174c45;padding:6px;}
        QCheckBox {spacing:6px;color:#627785;} QCheckBox::indicator {width:14px;height:14px;}
        QTabWidget::pane {border:1px solid #e0e6eb;background:white;border-radius:8px;top:0px;}
        QTabBar::tab {padding:11px 18px;color:#657987;border-bottom:2px solid transparent;margin-right:10px;}
        QTabBar::tab:selected {color:#22675e;border-bottom:2px solid #267a6d;font-weight:600;}
        QTabBar::tab:hover {color:#245f57;background:#edf2f3;}
        QToolBar {background:transparent;border:0;spacing:3px;} QToolButton {background:transparent;border:0;border-radius:4px;padding:3px;}
        QToolButton:hover {background:#e7efee;} QLabel#chartTitle {font-family:"Source Serif 4","Noto Serif SC";color:#364f5d;font-weight:600;}
        QProgressBar {border:0;background:#dfe7eb;border-radius:2px;} QProgressBar::chunk {background:#458b80;border-radius:2px;}
        QPlainTextEdit,QTableWidget {background:white;alternate-background-color:#f6f8fa;border:0;selection-background-color:#dbece8;selection-color:#173d4b;padding:9px;}
        QHeaderView::section {background:#f1f5f7;color:#586f7e;border:0;border-bottom:1px solid #e0e6eb;padding:10px;font-weight:600;}
        QScrollBar:vertical {width:8px;background:transparent;} QScrollBar::handle:vertical {background:#c3d0d8;border-radius:4px;min-height:24px;}
        QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical {height:0;}
        ''')
        self.thread = QtCore.QThread(self); self.worker = ReadWorker(); self.worker.moveToThread(self.thread)
        self.read_requested.connect(self.worker.read); self.worker.ready.connect(self.render)
        self.control_requested.connect(self.worker.control); self.worker.controlled.connect(self.control_finished)
        self.thread.finished.connect(self.worker.deleteLater); self.thread.start()
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(5000)
        self.refresh()

    def selection_changed(self, *_: object) -> None:
        for button in self.control_buttons.values(): button.setEnabled(False)
        self.refresh()

    def update_controls(self) -> None:
        data = self.data or {}; token = data.get('controls', {})
        current_view = (not self.selector.currentData() or self.selector.currentData() == token.get('folder'))
        enabled = token.get('available', False) and current_view and data.get('folder') == token.get('folder') and not self.control_pending
        for action, button in self.control_buttons.items():
            allowed = enabled
            if action == 'pause': allowed = allowed and not (token.get('child_paused') and token.get('supervisor_paused'))
            if action == 'resume': allowed = allowed and (token.get('child_paused') or token.get('supervisor_paused'))
            button.setEnabled(bool(allowed))
        if self.control_pending:
            text = '正在核验并执行…'
        elif not current_view:
            text = '历史运行不提供进程控制'
        elif enabled:
            text = '控制当前计算及调度器\n暂停保留内存；终止不可续算'
        else:
            text = token.get('reason', '等待当前运行核验')
        self.control_note.setText(text)

    def request_control(self, action: str) -> None:
        if self.control_pending or not self.control_buttons[action].isEnabled(): return
        token = dict((self.data or {}).get('controls', {}))
        if not token.get('available'): return
        if action == 'terminate':
            if self.confirmation is not None: return
            box = W.QMessageBox(self); box.setWindowTitle('终止当前计算与队列')
            box.setIcon(W.QMessageBox.Warning)
            box.setText('终止后不能从当前步继续。')
            box.setInformativeText(f"运行：{Path(token['folder']).name}\n计算 PID：{token['child']['pid']}\n\n将停止当前计算和后续队列。已写入的日志与结果保留；未落盘的流场和内存状态不会保存。\n如需稍后续算，请选择“暂停计算”。")
            cancel = box.addButton('取消', W.QMessageBox.RejectRole)
            stop = box.addButton('终止计算与队列', W.QMessageBox.DestructiveRole)
            box.setDefaultButton(cancel)
            self.confirmation = box
            def finished(_: int) -> None:
                self.confirmation = None
                if box.clickedButton() is stop: self.dispatch_control(action, token)
                box.deleteLater()
            box.finished.connect(finished); box.open()
        else:
            self.dispatch_control(action, token)

    def dispatch_control(self, action: str, token: dict) -> None:
        self.control_pending = True; self.update_controls()
        self.control_requested.emit(str(self.root), action, token)

    @QtCore.Slot(object)
    def control_finished(self, result: dict) -> None:
        self.control_pending = False
        if 'error' in result:
            self.control_note.setText('控制未完成：' + result['error'])
            W.QMessageBox.warning(self, '操作未确认完成', result['error'])
        else:
            self.control_note.setText(result['message'])
            self.statusBar().showMessage(result['message'], 15000)
        # Force a fresh process observation even if the picture is frozen.
        self.refresh()

    def pause_changed(self, paused: bool) -> None:
        if paused:
            self.subtitle.setText('看板刷新已暂停 · 当前显示为最后一次读取的快照 · 后台计算不受影响')
        else:
            self.refresh()

    def tick(self) -> None:
        if not self.pause.isChecked(): self.refresh()

    def refresh(self, *_: object) -> None:
        if self.busy or self.control_pending: return
        self.busy = True
        self.read_requested.emit(str(self.root), self.selector.currentData() or '')

    def open_folder(self) -> None:
        folder = (self.data or {}).get('folder') or str(self.root)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(folder))

    @QtCore.Slot(object)
    def render(self, data: dict) -> None:
        self.busy = False
        if 'error' in data:
            self.data = None; self.update_controls()
            self.status.setText('读取失败 · 上次显示值已过期：' + data['error'])
            self.subtitle.setText('未获得新快照；所有保留内容均为上次成功读取的数据。')
            return
        self.data = data
        self.update_controls()
        selected = self.selector.currentData()
        folders = data['folders']
        known = [self.selector.itemData(i) for i in range(1, self.selector.count())]
        if folders != known:
            self.selector.blockSignals(True); self.selector.clear(); self.selector.addItem('跟随当前计算', '')
            for folder in folders: self.selector.addItem(Path(folder).name, folder)
            self.selector.setCurrentIndex(max(0, self.selector.findData(selected))); self.selector.blockSignals(False)
        latest, cfg = data['latest'], data['config']
        stamp = dt.datetime.fromtimestamp(data['read_at']).strftime('%H:%M:%S')
        age = '无记录' if data['age'] is None else f"{data['age']:.0f} 秒前写入"
        self.subtitle.setText(f'本地连接 · 每 5 秒刷新 · 本次读取 {stamp} · {age}')
        self.status.setProperty('tone', 'warning' if ('暂停' in data['status'] or '延迟' in data['status'] or '未核验' in data['status']) else 'normal')
        self.status.style().unpolish(self.status); self.status.style().polish(self.status)
        queue_status = '已暂停' if data.get('controls', {}).get('supervisor_paused') else data['state'].get('status', '未知')
        if '已由看板终止' in data['status']: queue_status = '已终止'
        self.status.setText(data['status'] + f"  ·  调度器：{queue_status}  ·  已入账 {len(data['state'].get('results', []))} 组")
        if self.pause.isChecked():
            self.subtitle.setText(f'看板刷新已暂停 · 最后读取 {stamp} · 后台计算不受影响')
        for widget, key in zip(self.cards, ('iteration', 'residual', 'mass_drift', 'max_mach')):
            value = latest.get(key)
            widget.setText(f'{value:,}' if key == 'iteration' and isinstance(value, int) else fmt(value))
        conv = cfg.get('convergence', {}); total = conv.get('max_iter'); step = latest.get('iteration', 0)
        percent = step / total * 100 if total else 0
        self.budget.setValue(min(10000, int(percent * 100)))
        self.budget_label.setText(f'步数预算已使用 {percent:.2f}%  ·  {step:,} / {total or "未知"}（不是收敛完成率）')
        derived = cfg.get('derived', {}); grid = cfg.get('grid', {}); flow = cfg.get('flow', {})
        self.side_case.setText(f"Re  {fmt(flow.get('Re'))}\n{grid.get('NX', '?')} × {grid.get('NY', '?')} 节点\n{cfg.get('collision_model', '未知模型')}\n\n{cfg.get('boundary_scheme_default', '未知边界').replace('non_equilibrium_extrapolation', '非平衡外推边界')}")
        self.parameters.setText(f"Re {fmt(flow.get('Re'))}  ·  {grid.get('NX','?')} × {grid.get('NY','?')}  ·  U={fmt(flow.get('U_ref'))}  ·  L={fmt(derived.get('L_ref'))}  ·  τ={fmt(derived.get('tau'))}\n"
                                f"残差阈值 {fmt(conv.get('tol'))}  ·  最早判断 {conv.get('min_iter','?')} 步  ·  Ma 保护 {fmt(conv.get('max_mach'))}  ·  质量参与停止判据：{'是' if conv.get('mass_criterion') else '否'}")
        self.foot.setText('速度达标不等于质量守恒、基准验证或物理结论成立。' + ('  数据提示：' + '；'.join(data['warnings']) if data['warnings'] else ''))
        evidence = {k: v for k, v in data.items() if k not in ('records', 'fields')}
        self.sources.setPlainText('当前显示来源：' + data['folder'] + '\n\n' +
            '读取协议：run_request.json / progress.jsonl / run_summary.json / validation.json / results.npz\n'
            '控制器更新与运行记录是独立快照，可能相差一次写入。耗时沿用原记录，可能包含暂停等待，不是纯 CPU 时间。\n'
            '历史 Re100 运行与当前搜索分开选择；搜索端点不等于物理临界 Re。\n\n' + json.dumps(evidence, ensure_ascii=False, indent=2))
        key = (data['folder'], len(data['records']), repr(latest), repr(data['summary']), data['fields'] is not None)
        if key != self.plot_key:
            self.plot_key = key; self.draw_plots(data); self.fill_table(data)

    def draw_plots(self, data: dict) -> None:
        self.fig.clear(); axes = self.fig.subplots(2, 2)
        rows = data['records']; x = [r['iteration'] for r in rows]; conv = data['config'].get('convergence', {})
        for ax, key, title, color in zip(axes.flat, ('residual', 'mass_drift', 'max_mach', 'elapsed_seconds'),
                  ('单步相对 L2 残差（内部节点）', '质量相对漂移（独立诊断）', '空间最大 Ma', '求解器记录耗时 / s'),
                  ('#277b83', '#b08044', '#518b76', '#7783a0')):
            y = [r.get(key) if isinstance(r.get(key), (float, int)) and math.isfinite(r[key]) else np.nan for r in rows]
            ax.plot(x, y, linestyle='none', marker='.', markersize=2.7, color=color)
            ax.set_title(title, loc='left', fontfamily=['Source Serif 4', 'Noto Serif SC']); ax.set_xlabel('迭代步数'); ax.grid(alpha=.6, linewidth=.6)
            ax.tick_params(length=0, pad=6); ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False)
            if key == 'residual':
                ax.set_yscale('symlog', linthresh=1e-12)
                ax.set_yticks([0, 1e-9, 1e-6, 1e-3]); ax.set_yticklabels(['0', '10⁻⁹', '10⁻⁶', '10⁻³'])
                if conv.get('tol'): ax.axhline(conv['tol'], color='#a45b19', linestyle='--', linewidth=1, label='速度阈值'); ax.legend(fontsize=8)
            if key == 'max_mach' and conv.get('max_mach'):
                ax.axhline(conv['max_mach'], color='#a45b19', linestyle='--', linewidth=1, label='保护上限'); ax.legend(fontsize=8)
            if not rows: ax.text(.5, .5, '暂无已落盘报告', ha='center', transform=ax.transAxes)
        self.canvas.draw_idle()
        self.field_fig.clear(); fields = data['fields']
        if fields is None:
            self.field_note.setText('当前所选运行尚无可读取的最终流场。求解器只在结束时写 results.npz；此处不显示其他运行的替代图。')
            ax = self.field_fig.subplots(); ax.set_axis_off(); ax.text(.5, .5, '等待当前运行的真实流场文件', ha='center', va='center', color='#61718a', fontsize=16)
        else:
            self.field_note.setText(f"最终文件流场 · 第 {data['summary'].get('iteration', '?')} 步 · {data['folder']}/results.npz\n颜色按本次文件自动范围；历史图不可用颜色深浅直接跨工况比较。灰色为固体或非有限值。")
            ax = self.field_fig.subplots(); speed = np.ma.masked_invalid(np.ma.array(fields['speed'], mask=fields['solid_mask']))
            cmap = __import__('matplotlib').colormaps['viridis'].copy(); cmap.set_bad('#a0a0a0')
            im = ax.pcolormesh(fields['x'], fields['y'], speed, shading='nearest', cmap=cmap, vmin=0)
            ax.set(xlabel='x / L（文件原值）', ylabel='y / H（文件原值）', title='速度模 |u|（格子单位）', aspect='equal')
            self.field_fig.colorbar(im, ax=ax, label='|u| / lattice units')
        self.field_canvas.draw_idle()

    def fill_table(self, data: dict) -> None:
        rows = data['records']; self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, key in enumerate(('iteration', 'elapsed_seconds', 'residual', 'mass_drift', 'max_mach', 'stop_reason')):
                value = row.get(key)
                text = repr(value) if isinstance(value, float) else str(value) if value is not None else '缺失'
                self.table.setItem(i, j, W.QTableWidgetItem(text))
        self.table.scrollToBottom()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.timer.stop(); self.thread.quit(); self.thread.wait(); event.accept()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--capture', type=Path, help='save a genuine Qt window capture after load')
    args = parser.parse_args(); app = W.QApplication(sys.argv); app.setStyle('Fusion'); app.setApplicationName('LBM 后台计算观测台')
    for font_path in FONT_DIR.glob('*-Regular.ttf'):
        if QtGui.QFontDatabase.addApplicationFont(str(font_path)) < 0:
            raise RuntimeError(f'无法加载字体：{font_path}')
    window = Dashboard(args.root); window.show()
    if args.capture:
        def capture() -> None:
            if window.data:
                window.grab().save(str(args.capture))
                audit = {'visible': window.isVisible(), 'exposed': window.windowHandle().isExposed(), 'platform': app.platformName(), 'source': window.data['folder'], 'latest': window.data['latest']}
                args.capture.with_suffix('.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
            else: QtCore.QTimer.singleShot(500, capture)
        QtCore.QTimer.singleShot(2500, capture)
    sys.exit(app.exec())

if __name__ == '__main__': main()
