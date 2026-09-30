"""Qt experiment controls; rendering reads saved fields and report samples."""
import json
from pathlib import Path
import numpy as np
from PySide6 import QtCore, QtWidgets as W
from . import experiments
from research_options import BUOYANT, OBSTACLES, THERMAL, transport
from advanced_physics import geometry_mask


class ExperimentToolsMixin:
    def build_experiment_parameters(self, form):
        toggle=W.QPushButton('展开时变驱动、阵列与探针')
        toggle.setCheckable(True)
        panel=W.QWidget();grid=W.QFormLayout(panel)
        grid.setContentsMargins(0,0,0,0)
        grid.setRowWrapPolicy(W.QFormLayout.WrapLongRows)
        grid.setFieldGrowthPolicy(W.QFormLayout.AllNonFixedFieldsGrow)
        for key,label in (('drive_period','驱动周期 · 步'),('drive_phase','初相位 · rad'),
                          ('force_amplitude','周期加速度幅值'),('thermal_perturbation','初始温度扰动'),
                          ('outlet_rho','出口密度'),('obstacle_count_x','阵列数量 x'),
                          ('obstacle_count_y','阵列数量 y'),('obstacle_count_z','阵列数量 z'),
                          ('obstacle_spacing_x','中心间距 / Lx'),('obstacle_spacing_y','中心间距 / Ly'),
                          ('obstacle_spacing_z','中心间距 / Lz'),('probes','探针归一化坐标')):
            edit=W.QLineEdit();edit.setObjectName('param_'+key)
            edit.setMaxLength(1024 if key=='probes' else 64)
            edit.setAlignment(QtCore.Qt.AlignRight)
            edit.textChanged.connect(self.update_preview)
            self.inputs[key]=edit;grid.addRow(label,edit)
        self.inputs['probes'].setPlaceholderText('0.5,0.5; 0.75,0.5')
        self.inputs['probes'].setToolTip('每点 x,y（3D: x,y,z），分号分隔，范围 0–1；最多 8 点，取最近格点。固体内点输出空值。')
        combo=W.QComboBox();combo.addItem('抛物线 · U 为峰值','parabolic');combo.addItem('均匀 · U 为入口速度','uniform')
        combo.currentIndexChanged.connect(self.update_preview)
        self.extra_combos['inlet_profile']=combo;grid.addRow('入口剖面',combo)
        toggle.toggled.connect(panel.setVisible);panel.hide();form.addWidget(toggle);form.addWidget(panel)
        row=W.QHBoxLayout()
        for title,action in (('导入配置',self.import_configuration),('导出配置',self.export_configuration),('几何预览',self.preview_geometry)):
            button=W.QPushButton(title);button.clicked.connect(action);row.addWidget(button)
        form.addLayout(row)

    def build_second_scan(self, form):
        self.sweep_second_enabled=W.QCheckBox('增加第二扫描轴 · 笛卡尔积')
        self.sweep_second_parameter=W.QComboBox()
        for i in range(self.sweep_parameter.count()):
            self.sweep_second_parameter.addItem(self.sweep_parameter.itemText(i),self.sweep_parameter.itemData(i))
        self.sweep_second_parameter.setCurrentIndex(2)
        self.sweep_second_values=W.QLineEdit('0.02, 0.04')
        self.sweep_second_values.setEnabled(False)
        self.sweep_second_enabled.toggled.connect(self.sweep_second_values.setEnabled)
        self.sweep_second_enabled.toggled.connect(self.update_preview)
        self.sweep_second_parameter.currentIndexChanged.connect(self.update_preview)
        self.sweep_second_values.textChanged.connect(self.update_preview)
        form.addWidget(self.sweep_second_enabled);form.addWidget(self.sweep_second_parameter);form.addWidget(self.sweep_second_values)

    def experiment_requests(self):
        axes=[]
        if self.sweep_enabled.isChecked():
            axes.append((self.sweep_parameter.currentData(),self.sweep_values.text()))
            if self.sweep_second_enabled.isChecked():
                axes.append((self.sweep_second_parameter.currentData(),self.sweep_second_values.text()))
        return experiments.scan_requests(self.form_params(),axes,self.name_input.text().strip(),self.group_input.text())

    def import_configuration(self):
        path,_=W.QFileDialog.getOpenFileName(self,'导入实验配置',str(self.store.workspace),'JSON (*.json)')
        if not path:return
        try:
            data=experiments.read_configuration(path)
            self.apply_params(data['params']);self.name_input.setText(str(data.get('name','')))
            self.group_input.setText(str(data.get('group','')));self.sweep_enabled.setChecked(False)
            self.notice('已导入实验参数，尚未入队')
        except (OSError,ValueError,KeyError,TypeError) as error:self.error(str(error))

    def export_configuration(self):
        try:p=self.form_params()
        except ValueError as error:self.error(str(error));return
        path,_=W.QFileDialog.getSaveFileName(self,'导出实验配置',str(self.store.workspace/'experiment.json'),'JSON (*.json)')
        if path:
            try:
                experiments.save_configuration(path,p,self.name_input.text(),self.group_input.text())
                self.notice('已导出实验配置：'+path)
            except (OSError,ValueError) as error:self.error(str(error))

    def preview_geometry(self):
        from .app import Plot
        try:
            p=self.form_params();d=transport(p)
            sizes=[d['nx'],d['ny'],d['nz']][:d['dim']]
            x=(np.arange(min(d['nx'],256))+.5)*d['nx']/min(d['nx'],256)-.5
            y=(np.arange(min(d['ny'],192))+.5)*d['ny']/min(d['ny'],192)-.5
            xx,yy=np.meshgrid(x,y);coords=[xx,yy]
            if d['dim']==3:coords.append(np.full(xx.shape,(d['nz']-1)/2))
            mask=geometry_mask(np,coords,sizes,p)
            dialog=W.QDialog(self);dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose);dialog.resize(860,570)
            dialog.setWindowTitle('几何预览 · xy 中截面' if d['dim']==3 else '几何预览')
            layout=W.QVBoxLayout(dialog);plot=Plot(dialog);layout.addWidget(plot)
            axis=plot.figure.subplots();axis.imshow(mask,origin='lower',extent=(0,d['nx'],0,d['ny']),vmin=0,vmax=1,cmap='Greys',interpolation='nearest')
            axis.set_xlabel('x · lattice units');axis.set_ylabel('y · lattice units')
            axis.set_title('障碍物及流体区域 · 预览采样，不改变实际计算网格',fontsize=11)
            self.field_windows.append(dialog);dialog.destroyed.connect(lambda:self.field_windows.remove(dialog))
            dialog.show();plot.canvas.draw_idle()
        except (ValueError,TypeError) as error:self.error(str(error))

    def build_timeline(self, layout):
        self.snapshot_data=None;self.snapshot_owner=None;self.snapshot_token=0;self.snapshot_files=[];self.snapshot_busy=False
        row=W.QHBoxLayout();self.snapshot_frame=W.QComboBox();self.snapshot_frame.addItem('当前 / 最终场',None)
        self.snapshot_frame.setSizeAdjustPolicy(W.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.snapshot_frame.setMinimumContentsLength(10)
        self.snapshot_slider=W.QSlider(QtCore.Qt.Horizontal);self.snapshot_slider.setRange(0,0)
        self.snapshot_play=W.QPushButton('播放');self.snapshot_play.setCheckable(True)
        row.addWidget(self.snapshot_frame);row.addWidget(self.snapshot_slider,1);row.addWidget(self.snapshot_play);layout.addLayout(row)
        self.snapshot_timer=QtCore.QTimer(self);self.snapshot_timer.setInterval(700)
        self.snapshot_timer.timeout.connect(self.next_snapshot)
        self.snapshot_play.toggled.connect(lambda playing:self.snapshot_timer.start() if playing else self.snapshot_timer.stop())
        self.snapshot_frame.currentIndexChanged.connect(self.snapshot_changed)
        self.snapshot_slider.valueChanged.connect(self.snapshot_frame.setCurrentIndex)

    def refresh_experiment_views(self, case):
        files=experiments.snapshots(case['data_dir'])
        changed=self.snapshot_owner!=case['id']
        if changed or files!=self.snapshot_files:
            old=self.snapshot_frame.currentData() if not changed else None
            if changed:
                self.snapshot_play.setChecked(False);self.snapshot_data=None;self.snapshot_token+=1;self.snapshot_busy=False
            self.snapshot_owner=case['id'];self.snapshot_files=files
            self.snapshot_frame.blockSignals(True);self.snapshot_frame.clear();self.snapshot_frame.addItem('当前 / 最终场',None)
            for step,path in files:self.snapshot_frame.addItem(f'快照 · step {step}',path)
            self.snapshot_frame.setCurrentIndex(max(0,self.snapshot_frame.findData(old)));self.snapshot_frame.blockSignals(False)
            self.snapshot_slider.setRange(0,len(files));self.snapshot_slider.setValue(self.snapshot_frame.currentIndex())
            self.snapshot_play.setEnabled(bool(files));self.snapshot_slider.setEnabled(bool(files))
        self.render_signals()

    def refresh_signal_choices(self,case):
        labels=(('drag_coefficient','阻力系数 Cd'),('lift_coefficient','升力系数 Cl'),('force_x','障碍合力 Fx'),
                ('force_y','障碍合力 Fy'),('nusselt_hot','热壁 Nu'),('nusselt_cold','冷壁 Nu'),
                ('flux_imbalance','进出口质量通量差'),('mass_change','域内质量相对变化'),
                ('boundary_mass_exchange','累计边界质量交换'),('outlet_backflow_fraction','出口回流格点比例'))
        last=case['records'][-1] if case['records'] else {}
        choices=[(label,(key,None)) for key,label in labels if key in last]
        p=case['params'];count=len(p.get('probes',[]))
        components=[('ux','ux'),('uy','uy')]
        if p.get('lattice','').startswith('D3'):components.append(('uz','uz'))
        components.append(('pressure','压力'))
        if p.get('scenario') in THERMAL:components.append(('T','温度'))
        choices += [(f'探针 {i+1} · {label}',(key,i)) for i in range(count) for key,label in components]
        signature=(case['id'],tuple(choices))
        if signature==getattr(self,'signal_options_signature',None):return
        self.signal_options_signature=signature;old=self.signal_source.currentData()
        self.signal_source.blockSignals(True);self.signal_source.clear()
        for title,(key,probe) in choices:
            self.signal_source.addItem(title,key if probe is None else f'{key}@{probe}')
        if not choices:self.signal_source.addItem('尚无探针或积分诊断数据','')
        self.signal_source.setCurrentIndex(max(0,self.signal_source.findData(old)))
        self.signal_source.blockSignals(False)

    def snapshot_changed(self, index):
        self.snapshot_slider.blockSignals(True);self.snapshot_slider.setValue(index);self.snapshot_slider.blockSignals(False)
        self.snapshot_token+=1;token=self.snapshot_token;self.snapshot_data=None;self.snapshot_busy=False
        path=self.snapshot_frame.currentData();case=self.loaded_monitor
        if not path or not case:self.render_field();return
        owner=case['id'];self.snapshot_busy=True
        def ready(data):
            if token!=self.snapshot_token or owner!=self.monitor_id:return
            self.snapshot_busy=False;self.snapshot_data=data;self.render_field()
        def failed(error):
            if token!=self.snapshot_token:return
            self.snapshot_busy=False;self.snapshot_play.setChecked(False);self.notice(error,True)
        self.async_run(lambda:experiments.load_snapshot(path,case['data_dir']),ready,failed)

    def next_snapshot(self):
        if self.snapshot_busy:return
        count=self.snapshot_frame.count()
        if count<2:self.snapshot_play.setChecked(False);return
        index=self.snapshot_frame.currentIndex()+1
        self.snapshot_frame.setCurrentIndex(index if index<count else 1)

    def current_fields(self):
        if self.snapshot_frame.currentData():
            return self.snapshot_data if self.snapshot_owner==self.monitor_id else None
        return self.loaded_monitor.get('fields') if self.loaded_monitor else None

    def build_signal_tab(self):
        from .app import Plot
        panel=W.QWidget();layout=W.QVBoxLayout(panel);row=W.QHBoxLayout()
        self.signal_source=W.QComboBox()
        for key,label in (('drag_coefficient','阻力系数 Cd'),('lift_coefficient','升力系数 Cl'),('force_x','障碍合力 Fx'),
                          ('force_y','障碍合力 Fy'),('nusselt_hot','热壁 Nu'),('nusselt_cold','冷壁 Nu'),
                          ('flux_imbalance','进出口质量通量差'),('mass_change','域内质量相对变化'),
                          ('boundary_mass_exchange','累计边界质量交换'),('outlet_backflow_fraction','出口回流格点比例')):
            self.signal_source.addItem(label,key)
        for index in range(8):
            for key,label in (('ux','ux'),('uy','uy'),('uz','uz'),('pressure','压力'),('T','温度')):
                self.signal_source.addItem(f'探针 {index+1} · {label}',f'{key}@{index}')
        self.signal_start=W.QSpinBox();self.signal_start.setRange(0,90);self.signal_start.setSuffix(' %');self.signal_start.setValue(0)
        self.signal_start.setToolTip('FFT 跳过记录开头的比例；时间曲线仍显示全部记录')
        row.addWidget(self.signal_source,1);row.addWidget(W.QLabel('频谱跳过起始'));row.addWidget(self.signal_start)
        layout.addLayout(row)
        self.signal_info=W.QLabel('选择有记录的探针或积分诊断量');self.signal_info.setWordWrap(True);layout.addWidget(self.signal_info)
        self.signal_plot=Plot();layout.addWidget(self.signal_plot,1)
        self.monitor_tabs.addTab(panel,'探针与频谱')
        self.signal_source.currentIndexChanged.connect(self.render_signals);self.signal_start.valueChanged.connect(self.render_signals)
        self.monitor_tabs.currentChanged.connect(self.render_signals)

    def render_signals(self,*args):
        if not hasattr(self,'signal_plot') or not self.loaded_monitor:return
        self.refresh_signal_choices(self.loaded_monitor)
        if self.monitor_tabs.currentWidget()!=self.signal_plot.parentWidget():return
        identity=self.signal_source.currentData()
        if not identity:
            self.signal_plot.empty('此记录没有探针或积分诊断信号')
            self.signal_info.setText('新建任务时可在参数区配置点探针，障碍与热对流场景另有积分诊断量')
            return
        parts=identity.split('@')
        key,probe=parts[0],int(parts[1]) if len(parts)>1 else None
        times,values=experiments.signal_series(self.loaded_monitor['records'],key,probe)
        figure=self.signal_plot.figure;figure.clear();left,right=figure.subplots(2,1)
        left.plot(times,values,lw=1.2);left.set_xlabel('迭代步数');left.set_ylabel(self.signal_source.currentText());left.grid(alpha=.3)
        try:
            data=experiments.spectrum(times,values,self.signal_start.value()/100)
            right.plot(data['frequency'],data['amplitude'],lw=1.2)
            self.signal_info.setText(f"Hann 窗、去均值 · {data['samples']} 点 · Δf={data['resolution']:.5g} · Nyquist={data['nyquist']:.5g}\n峰值 f={data['peak_frequency']:.6g} /步，幅值={data['peak_amplitude']:.6g}；剔除首尾不规则点 {data['trimmed']} 个")
            if not data['resolved']:self.signal_info.setText('信号为常量或幅值处于舍入误差水平，未检出可解析的周期峰值')
        except ValueError as error:self.signal_info.setText(str(error))
        right.set_xlabel('频率 · 1 / lattice step');right.set_ylabel('单边幅值');right.grid(alpha=.3)
        figure.set_layout_engine('constrained');self.signal_plot.canvas.draw_idle()
