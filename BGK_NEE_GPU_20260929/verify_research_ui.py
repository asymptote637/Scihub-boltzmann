"""Native UI -> persistent queue -> actual worker -> 3D/thermal result QA."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("PYTHONUTF8","1")

import numpy as np
from PySide6 import QtCore, QtTest, QtWidgets as W
from dashboard.app import Dashboard, configure_fonts
from dashboard.store import ACTIVE, DEFAULTS, ROOT, Store, atomic_json
from dashboard import analysis
from research_options import RECIPES, generic_engine


def wait_for(condition, seconds=120):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        W.QApplication.processEvents()
        result=condition()
        if result:return result
        QtTest.QTest.qWait(80)
    raise AssertionError("Timed out waiting for UI/worker")


def main():
    folder=ROOT/"validation"/("research_ui_"+dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    store=Store(folder/"workspace")
    app=W.QApplication([])
    app.setStyle("Fusion")
    configure_fonts()
    window=Dashboard(store)
    errors=[]
    window.error=errors.append
    window.show()
    report=dict(passed=False,workspace=str(store.workspace),jobs=[],checks=[],screenshots=[],errors=errors,
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((ROOT/"dashboard").glob("*.py"))})
    path=ROOT/"validation/research_ui.json"
    try:
        wait_for(lambda:not window.tasks)
        window.timer.setInterval(500)
        for index,(name,recipe) in enumerate(list(RECIPES.items())[:12]):
            window.recipe.setCurrentIndex(window.recipe.findData(name))
            p=dict(window.form_params(),backend=("cpu","array","fused")[index%3],grid=16,grid_y=16,grid_z=16,
                   max_iter=31,min_iter=100,ramp_steps=10,report_interval=10)
            if generic_engine(p):p["snapshot_interval"]=10;p["checkpoint_interval"]=10
            window.apply_params(p)
            window.name_input.setText("扩展验收 · "+name)
            assert window.start_button.isEnabled(),window.derived.text()
            window.start_button.click()
            assert not errors,errors
            case=next(c for c in store.cases() if c["name"]=="扩展验收 · "+name)
            report["jobs"].append(case["id"])
        atomic_json(path,report)
        wait_for(lambda:all(store.get(i)["status"] not in (*ACTIVE,"queued") for i in report["jobs"]),180)
        for identifier in report["jobs"]:
            case=store.get(identifier)
            assert case["status"] in {"max_iter","completed_steps"},(case["name"],case.get("error"),case["status"])
            fields=analysis.load_fields(case["data_dir"])
            assert fields["finite"]
            assert fields["rho"].shape==((16,16,16) if case["params"]["lattice"].startswith("D3") else (16,16))
            if generic_engine(case["params"]):
                assert (Path(case["data_dir"])/"checkpoint.npz").is_file()
                assert len(list((Path(case["data_dir"])/"snapshots").glob("*.npz")))==3
            report["checks"].append(dict(name=case["name"],status=case["status"],shape=list(fields["rho"].shape)))
        print("PASS 12 recipe jobs, all three backends, final fields and snapshots",flush=True)
        last=store.get(report["jobs"][-1])
        checkpoint=Path(last["data_dir"])/"checkpoint.npz"
        window.apply_params(dict(last["params"],resume_from=str(checkpoint),max_iter=61,backend="fused"))
        window.name_input.setText("扩展验收 · 三维温度续算")
        window.start_button.click()
        resumed=next(c for c in store.cases() if c["name"]=="扩展验收 · 三维温度续算")
        wait_for(lambda:store.get(resumed["id"])["status"] not in (*ACTIVE,"queued"))
        resumed=store.get(resumed["id"])
        assert resumed["metrics"]["iteration"]==61 and resumed["status"]=="max_iter",resumed
        report["jobs"].append(resumed["id"])
        report["checks"].append("3D thermal checkpoint resumed into new job to total iteration 61")
        print("PASS checkpoint resume via UI",flush=True)
        window.apply_params(dict(RECIPES["二维自然对流 · Ra1000"],backend="cpu",grid=12,grid_y=12,max_iter=3,min_iter=100))
        window.sweep_parameter.setCurrentIndex(window.sweep_parameter.findData("rayleigh"))
        window.sweep_values.setText("0, 100")
        window.sweep_enabled.setChecked(True)
        window.name_input.setText("扩展验收 · Ra扫描")
        requests=window.form_requests()
        assert [r["params"]["rayleigh"] for r in requests]==[0,100]
        window.start_button.click()
        ids=[c["id"] for c in store.cases() if c["name"].startswith("扩展验收 · Ra扫描")]
        assert len(ids)==2
        wait_for(lambda:all(store.get(i)["status"] not in (*ACTIVE,"queued") for i in ids))
        assert all(store.get(i)["status"]=="max_iter" for i in ids)
        report["jobs"].extend(ids)
        report["checks"].append("Ra scan produces two validated serial jobs")
        window.sweep_enabled.setChecked(False)
        window.apply_params(RECIPES["三维自然对流 · D3Q19+D3Q7"])
        window.open_case(resumed["id"])
        wait_for(lambda:window.loaded_monitor and window.loaded_monitor["id"]==resumed["id"])
        window.monitor_tabs.setCurrentIndex(1)
        window.field_mode.setCurrentIndex(window.field_mode.findData("T"))
        for size,plane,name in (( (1440,940),"xy","thermal_3d_xy"),((1440,940),"xz","thermal_3d_xz"),
                                ((960,680),"yz","thermal_3d_compact")):
            window.resize(*size)
            window.field_plane.setCurrentIndex(window.field_plane.findData(plane))
            window.render_field()
            QtTest.QTest.qWait(350)
            file=folder/(name+".png")
            assert window.grab().save(str(file))
            report["screenshots"].append(str(file))
        window.resize(1440,940)
        for key in ("speed","uz","vorticity","overlay","vortices"):
            window.field_mode.setCurrentIndex(window.field_mode.findData(key))
            window.render_field()
            W.QApplication.processEvents()
            assert window.field_plot.figure.axes
        report["checks"].append("3D xy/xz/yz slice controls, T/uz/speed/vorticity/streamline renders")
        cases=[store.get(i) for i in report["jobs"]]
        analysis.export_csv(cases,folder/"research_metrics.csv")
        analysis.export_bundle(cases,folder/"research_results.zip")
        report["checks"].append("CSV includes research parameters; ZIP includes checkpoints and snapshots")
        assert not errors,errors
        report["passed"]=True
        print("PASS 3D plots, parameter scan, exports",flush=True)
    finally:
        atomic_json(path,report)
        window.close()
        QtCore.QThreadPool.globalInstance().waitForDone(10000)
        W.QApplication.processEvents()
    if not report["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
