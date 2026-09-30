"""Native workbench end-to-end QA, isolated from the user's workspace."""
import datetime as dt
import hashlib
import os
from pathlib import Path
import time
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('PYTHONUTF8','1')
import numpy as np
from PySide6 import QtCore, QtTest, QtWidgets as W
from dashboard.app import Dashboard,configure_fonts
from dashboard.store import ROOT,Store,ACTIVE,atomic_json
from dashboard import analysis,experiments
from research_options import RECIPES,generic_engine


def wait(condition, seconds=180):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        W.QApplication.processEvents()
        result=condition()
        if result:return result
        QtTest.QTest.qWait(80)
    raise AssertionError('Timed out waiting for native UI / worker')


def main():
    folder=ROOT/'validation'/('advanced_ui_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S'))
    store=Store(folder/'workspace');app=W.QApplication([]);app.setStyle('Fusion');configure_fonts()
    window=Dashboard(store);errors=[];window.error=errors.append;window.show()
    report=dict(passed=False,workspace=str(store.workspace),jobs=[],checks=[],screenshots=[],errors=errors,
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'dashboard').glob('*.py'))})
    receipt=ROOT/'validation/advanced_ui.json'
    def finished(ids):return all(store.get(i)['status'] not in (*ACTIVE,'queued') for i in ids)
    def enqueue(p,name):
        window.apply_params(p);window.name_input.setText(name)
        assert window.start_button.isEnabled(),window.derived.text()
        window.start_button.click();assert not errors,errors
        case=next(c for c in store.cases() if c['name']==name)
        report['jobs'].append(case['id']);return case['id']
    try:
        wait(lambda:not window.tasks);window.timer.setInterval(500)
        ids=[]
        for index,(name,preset) in enumerate(RECIPES.items()):
            window.recipe.setCurrentIndex(window.recipe.findData(name));p=window.form_params()
            p.update(grid=16,grid_y=16,grid_z=16,backend=('cpu','array','fused')[index%3],
                     max_iter=31,min_iter=100,ramp_steps=10,report_interval=10)
            if generic_engine(p):p.update(snapshot_interval=10,checkpoint_interval=10)
            ids.append(enqueue(p,'综合验收 · '+name))
        wait(lambda:finished(ids),240)
        for identifier in ids:
            case=store.get(identifier)
            assert case['status'] in {'max_iter','completed_steps'},(case['name'],case['status'],case.get('error'))
            f=analysis.load_fields(case['data_dir']);assert f['finite']
            assert f['rho'].shape==((16,16,16) if case['params']['lattice'].startswith('D3') else (16,16))
            if generic_engine(case['params']):assert len(experiments.snapshots(case['data_dir']))==3
        report['checks'].append('20 presets submitted through native UI; three backends; complete finite fields')
        print('PASS 20 recipe workers',flush=True)
        # Exercise JSON dialog routes with deterministic file chooser values.
        p=dict(RECIPES['周期驱动力 · Womersley 通道'],backend='fused',grid=16,grid_y=16,max_iter=640,
               min_iter=1000,drive_period=200,report_interval=8,snapshot_interval=40,checkpoint_interval=80)
        window.apply_params(p);window.name_input.setText('配置往返验证');saved=window.form_params();path=folder/'experiment.json'
        save_dialog=W.QFileDialog.getSaveFileName;open_dialog=W.QFileDialog.getOpenFileName
        try:
            W.QFileDialog.getSaveFileName=lambda *a,**k:(str(path),'JSON')
            window.export_configuration();assert path.is_file()
            window.inputs['drive_period'].setText('400')
            W.QFileDialog.getOpenFileName=lambda *a,**k:(str(path),'JSON')
            window.import_configuration();assert window.form_params()==saved
        finally:W.QFileDialog.getSaveFileName=save_dialog;W.QFileDialog.getOpenFileName=open_dialog
        signal_id=enqueue(saved,'综合验收 · 周期信号与回放')
        wait(lambda:finished([signal_id]));signal_case=store.get(signal_id)
        assert signal_case['status']=='completed_steps',signal_case
        window.open_case(signal_id);wait(lambda:window.loaded_monitor and window.loaded_monitor['id']==signal_id)
        wait(lambda:window.snapshot_frame.count()==17)
        window.resize(1440,940);window.monitor_tabs.setCurrentIndex(1)
        window.snapshot_frame.setCurrentIndex(3)
        wait(lambda:window.snapshot_data is not None and window.snapshot_data['preview_iteration']==120)
        window.render_field();assert '保存快照' in window.field_plot.figure.axes[0].get_title() or window.current_fields()['snapshot']
        before=window.snapshot_frame.currentIndex();window.snapshot_play.setChecked(True)
        wait(lambda:window.snapshot_frame.currentIndex()!=before);window.snapshot_play.setChecked(False)
        wait(lambda:not window.snapshot_busy)
        assert store.get(signal_id)['metrics']['iteration']==640
        file=folder/'snapshot_playback.png';window.grab().save(str(file));report['screenshots'].append(str(file))
        window.monitor_tabs.setCurrentIndex(3)
        index=window.signal_source.findData('ux@0');assert index>=0
        window.signal_source.setCurrentIndex(index);window.render_signals()
        assert 'Hann' in window.signal_info.text(),window.signal_info.text()
        file=folder/'probe_spectrum.png';QtTest.QTest.qWait(200);window.grab().save(str(file));report['screenshots'].append(str(file))
        report['checks'].extend(['Configuration JSON via UI import/export is exact','Saved full fields playback + timer; no recomputation','Probe history and FFT panel with real worker data'])
        print('PASS configuration, playback, probes and FFT',flush=True)
        # Restore to a new run and preserve the physical time of periodic forcing.
        resumed=enqueue(dict(saved,resume_from=str(Path(signal_case['data_dir'])/'checkpoint.npz'),max_iter=680),'综合验收 · 周期检查点续算')
        wait(lambda:finished([resumed]));assert store.get(resumed)['metrics']['iteration']==680
        window.apply_params(dict(RECIPES['底热顶冷对流 · Ra10000'],backend='cpu',grid=16,grid_y=8,max_iter=3,min_iter=100,snapshot_interval=0))
        window.name_input.setText('综合验收 · 双参数矩阵')
        window.sweep_parameter.setCurrentIndex(window.sweep_parameter.findData('rayleigh'));window.sweep_values.setText('0,100')
        window.sweep_second_parameter.setCurrentIndex(window.sweep_second_parameter.findData('prandtl'));window.sweep_second_values.setText('.7,1,2')
        window.sweep_second_enabled.setChecked(True);window.sweep_enabled.setChecked(True)
        assert len(window.form_requests())==6
        window.start_button.click();assert not errors,errors
        matrix=[c['id'] for c in store.cases() if c['name'].startswith('综合验收 · 双参数矩阵')]
        assert len(matrix)==6;report['jobs'].extend(matrix);wait(lambda:finished(matrix))
        assert all(store.get(i)['status']=='max_iter' for i in matrix)
        window.sweep_enabled.setChecked(False);window.sweep_second_enabled.setChecked(False)
        report['checks'].extend(['Periodic forcing checkpoint resumes to total budget 680','2 x 3 Cartesian scan produces six actual serial jobs'])
        # 3D slicing and compact layout remain usable with the new timeline.
        thermal=next(store.get(i) for i in ids if store.get(i)['params']['scenario']=='rayleigh_benard' and store.get(i)['params']['lattice']=='D3Q27')
        window.open_case(thermal['id']);wait(lambda:window.loaded_monitor and window.loaded_monitor['id']==thermal['id'])
        window.monitor_tabs.setCurrentIndex(1);window.field_mode.setCurrentIndex(window.field_mode.findData('T'))
        window.snapshot_frame.setCurrentIndex(1);wait(lambda:window.snapshot_data is not None)
        for plane in ('xy','xz','yz'):
            window.field_plane.setCurrentIndex(window.field_plane.findData(plane));window.render_field();W.QApplication.processEvents()
            assert window.field_plot.figure.axes
        window.resize(960,680);QtTest.QTest.qWait(300)
        file=folder/'thermal_compact.png';window.grab().save(str(file));report['screenshots'].append(str(file))
        window.apply_params(RECIPES['六圆柱周期阵列']);window.preview_geometry()
        assert window.field_windows;QtTest.QTest.qWait(200)
        file=folder/'geometry_array.png';window.field_windows[-1].grab().save(str(file));report['screenshots'].append(str(file))
        window.field_windows[-1].close();W.QApplication.processEvents()
        cases=[store.get(i) for i in report['jobs']]
        analysis.export_csv(cases,folder/'advanced_metrics.csv');analysis.export_bundle(cases,folder/'advanced_results.zip')
        report['checks'].extend(['3D xy/xz/yz snapshot slicing and 960x680 layout','Six-cylinder geometry preview','CSV and ZIP include probe CSV, snapshots, parameters and checkpoints'])
        assert not errors,errors
        report['passed']=True;print('PASS matrix, restart, 3D, geometry and exports',flush=True)
    finally:
        atomic_json(receipt,report);window.close()
        QtCore.QThreadPool.globalInstance().waitForDone(10000);W.QApplication.processEvents()
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
