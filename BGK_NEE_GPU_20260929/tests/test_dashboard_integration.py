import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from dashboard.store import ACTIVE, DEFAULTS, ROOT, Store


def until(predicate, seconds=60):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.1)
    raise AssertionError("Timed out waiting for a worker state")


@pytest.fixture
def service(tmp_path):
    store = Store(tmp_path / "workspace")
    log = (store.workspace / "service_test.log").open("wb")
    process = subprocess.Popen([sys.executable, "-X", "utf8", "-u", "-m", "dashboard.service", "--workspace",
                                str(store.workspace), "--idle-seconds", "5"], cwd=ROOT, stdout=log, stderr=log,
                               env=dict(os.environ, PYTHONUTF8="1", OPENBLAS_NUM_THREADS="1"),
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    yield store
    for case in store.cases(archived=True):
        if case["status"] in (*ACTIVE, "queued"):
            store.control(case["id"], "cancel")
    process.wait(timeout=45)
    log.close()
    assert process.returncode == 0


def test_cpu_and_two_gpu_workers_match_frozen_reference(service):
    params = dict(DEFAULTS, grid=16, re=100, max_iter=301, report_interval=100)
    ids = service.enqueue([dict(name=backend, params=dict(params, backend=backend)) for backend in ("cpu", "array", "fused")])
    until(lambda: all(service.get(identifier)["status"] not in (*ACTIVE, "queued") for identifier in ids), 60)
    from reference import CPUSolver, make_config
    cpu = CPUSolver(make_config(**{k: v for k, v in params.items() if k != "backend"}))
    reports = list(cpu.run())
    for identifier in ids:
        case = service.get(identifier)
        assert case["status"] == "max_iter", case
        assert case["metrics"]["iteration"] == 301
        assert case["metrics"]["wall_seconds"] >= case["metrics"]["elapsed_seconds"]
        folder = Path(case["data_dir"])
        summary = json.loads((folder / "run_summary.json").read_text(encoding="utf-8"))
        assert summary["iteration"] == 301 and summary["dtype"] == "float64"
        rows = [json.loads(line) for line in (folder / "progress.jsonl").read_text().splitlines()]
        assert [r["iteration"] for r in rows] == [r.iteration for r in reports]
        with np.load(folder / "results.npz", allow_pickle=False) as fields:
            for key in ("rho", "ux", "uy"):
                np.testing.assert_array_equal(fields[key], getattr(cpu, key))


def test_pause_resume_cancel_and_persistent_queue(service):
    identifier = service.enqueue([dict(name="control", params=dict(DEFAULTS, backend="cpu", grid=64, re=100,
                                                                     max_iter=1000000, report_interval=20))])[0]
    queued = service.enqueue([dict(name="cancel-before-start", params=dict(DEFAULTS, backend="cpu", grid=16, max_iter=1))])[0]
    until(lambda: service.get(identifier)["status"] == "running")
    service.control(identifier, "pause")
    until(lambda: service.get(identifier)["status"] == "paused")
    step = service.get(identifier)["metrics"].get("iteration")
    time.sleep(0.6)
    reopened = Store(service.workspace)
    assert reopened.get(identifier)["status"] == "paused"
    assert reopened.get(identifier)["metrics"].get("iteration") == step
    reopened.control(queued, "cancel")
    reopened.control(identifier, "run")
    until(lambda: (reopened.get(identifier)["metrics"].get("iteration") or 0) > (step or 0))
    reopened.control(identifier, "cancel")
    until(lambda: reopened.get(identifier)["status"] == "cancelled")
    case = reopened.get(identifier)
    assert 0 < case["metrics"]["iteration"] < 1000000
    assert case["metrics"]["paused_seconds"] >= 0.6
    assert reopened.get(queued)["status"] == "cancelled"
    assert not Path(reopened.get(queued)["data_dir"]).exists()
    summary = json.loads((Path(case["data_dir"]) / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["stop_reason"] == "cancelled" and summary["converged"] is False
    assert (Path(case["data_dir"]) / "results.npz").is_file()


def test_optimized_3d_worker_controls_and_final_fields(service):
    from research_options import EXTRA_DEFAULTS
    p=DEFAULTS|EXTRA_DEFAULTS|dict(backend='fused',lattice='D3Q19',collision='TRT',scenario='cavity',
        boundary='halfway',grid=32,grid_y=32,grid_z=32,run_mode='transient',max_iter=1000000,
        min_iter=1000000,ramp_steps=100,report_interval=20)
    identifier=service.enqueue([dict(name='optimized-3d-controls',params=p)])[0]
    until(lambda:service.get(identifier)['status']=='running')
    until(lambda:service.get(identifier)['metrics'].get('iteration',0)>=20)
    service.control(identifier,'pause')
    until(lambda:service.get(identifier)['status']=='paused')
    paused=service.get(identifier)['metrics']['iteration']
    time.sleep(.6)
    assert service.get(identifier)['metrics']['iteration']==paused
    service.control(identifier,'run')
    until(lambda:service.get(identifier)['metrics'].get('iteration',0)>paused)
    service.control(identifier,'cancel')
    until(lambda:service.get(identifier)['status']=='cancelled')
    case=service.get(identifier);folder=Path(case['data_dir'])
    assert case['metrics']['paused_seconds']>=.6
    assert case['metrics']['preview_iteration']==case['metrics']['iteration']
    with np.load(folder/'results.npz',allow_pickle=False) as fields,np.load(folder/'checkpoint.npz',allow_pickle=False) as checkpoint:
        assert fields['rho'].shape==(32,32,32)
        assert int(fields['iteration'])==int(checkpoint['iteration'])==case['metrics']['iteration']
        for key in ('rho','ux','uy','uz'):
            assert np.isfinite(fields[key]).all()
            np.testing.assert_array_equal(fields[key],checkpoint[key])
    timing=json.loads((folder/'timing_and_population.json').read_text(encoding='utf-8'))
    assert timing['population_check']['finite']
