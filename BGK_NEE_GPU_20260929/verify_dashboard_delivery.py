"""Create labelled CPU/GPU acceptance records and audit unchanged frozen sources."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import psutil

from dashboard.service import ensure_service, process_matches
from dashboard.store import ACTIVE, DEFAULTS, ROOT, Store, atomic_json, read_json, receipt_status


def original_audit():
    original = ROOT.parent / "BGK_NEE_Windows_20260928_191621/simulation"
    plan = read_json(ROOT / "reference_cpu/plan.json")
    for name, expected in plan["source_sha256"].items():
        assert hashlib.sha256((original / name).read_bytes()).hexdigest() == expected, name
    controls = read_json(original / "dashboard_control/state.json", {})
    processes = []
    for role in ("supervisor", "child"):
        identity = controls.get("target", {}).get(role, {})
        try:
            process = psutil.Process(identity["pid"])
            assert abs(process.create_time() - identity["started"]) < 0.001
            processes.append(dict(role=role, pid=process.pid, status=process.status(), cpu_seconds=sum(process.cpu_times()[:2])))
        except (psutil.Error, KeyError):
            processes.append(dict(role=role, unavailable=True))
    return dict(frozen_files=len(plan["source_sha256"]), processes=processes)


def main():
    store = Store()
    store.discover()
    report = dict(passed=False, started_at=datetime.now().astimezone().isoformat(), original_before=original_audit(), cases=[])
    path = ROOT / "validation/dashboard_delivery.json"
    atomic_json(path, report)
    identifiers = []
    superseded = []
    worker_hash = hashlib.sha256((ROOT / "dashboard/worker.py").read_bytes()).hexdigest()
    for backend in ("cpu", "fused"):
        name = ("CPU" if backend == "cpu" else "GPU") + " 完整收敛对照 · N64 Re100"
        existing = next((case for case in store.cases(archived=True) if case["name"] == name and case["group_name"] == "工作台验收"), None)
        existing_summary = read_json(Path(existing["data_dir"]) / "run_summary.json", {}) if existing else {}
        if existing and existing_summary.get("worker_source_sha256") == worker_hash:
            identifiers.append(existing["id"])
        else:
            if existing:
                superseded.append(existing["id"])
            identifiers.extend(store.enqueue([dict(name=name, group="工作台验收", params=dict(DEFAULTS, backend=backend,
                                                    grid=64, re=100, max_iter=50000))]))
    ensure_service(store)
    deadline = time.monotonic() + 180
    last = None
    while time.monotonic() < deadline:
        cases = [store.get(case_id) for case_id in identifiers]
        state = [(case["params"]["backend"], case["status"], case["metrics"].get("iteration")) for case in cases]
        if state != last:
            print(state, flush=True)
            last = state
        if all(case["status"] not in (*ACTIVE, "queued") for case in cases):
            break
        time.sleep(1)
    else:
        raise AssertionError("Acceptance cases exceeded the bounded validation time")
    fields = []
    for case in cases:
        assert case["status"] == "converged", case
        assert case["metrics"]["iteration"] == 16148
        assert case["metrics"]["wall_seconds"] >= case["metrics"]["elapsed_seconds"]
        with np.load(Path(case["data_dir"]) / "results.npz", allow_pickle=False) as archive:
            fields.append({key: archive[key].copy() for key in ("rho", "ux", "uy")})
        report["cases"].append({key: case[key] for key in ("id", "name", "params", "metrics", "data_dir", "status")})
    errors = {key: float(np.max(np.abs(fields[0][key] - fields[1][key]))) for key in fields[0]}
    assert max(errors.values()) == 0
    for identifier in superseded:
        store.update_case(identifier, archived=1)
    report.update(passed=True, field_max_errors=errors, original_after=original_audit(), backend_gates=receipt_status(),
                  completed_at=datetime.now().astimezone().isoformat(),
                  source_sha256={str(source.relative_to(ROOT)): hashlib.sha256(source.read_bytes()).hexdigest()
                                 for source in sorted((ROOT / "dashboard").glob("*.py"))})
    atomic_json(path, report)
    identity = store.setting("service", {})
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline and process_matches(identity.get("pid"), identity.get("started"), "dashboard.service"):
        time.sleep(1)
    assert not process_matches(identity.get("pid"), identity.get("started"), "dashboard.service")
    print("PASS: " + str(path), flush=True)


if __name__ == "__main__":
    main()
