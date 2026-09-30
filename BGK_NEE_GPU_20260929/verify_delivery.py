"""Exercise the CLI, exported artifacts, validation gate, and read-only CPU audit."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import psutil
import run_gpu
from reference import ROOT, PLAN, CPUSolver, make_config, verify_reference


def cpu_audit():
    original = ROOT.parent / "BGK_NEE_Windows_20260928_191621/simulation"
    fingerprints = {}
    for name, expected in PLAN["source_sha256"].items():
        digest = hashlib.sha256((original / name).read_bytes()).hexdigest()
        assert digest == expected, name
        fingerprints[name] = digest
    control = json.loads((original / "dashboard_control/state.json").read_text(encoding="utf-8"))
    processes = []
    for role in ("supervisor", "child"):
        identity = control["target"][role]
        process = psutil.Process(identity["pid"])
        assert abs(process.create_time() - identity["started"]) < 1e-3
        processes.append(dict(role=role, pid=process.pid, status=process.status(),
                              cpu_seconds=sum(process.cpu_times()[:2]), command=process.cmdline()))
    return dict(timestamp=dt.datetime.now().astimezone().isoformat(), verified_files=len(fingerprints),
                original_sha256=fingerprints, processes=processes,
                last_dashboard_action=control["action"], last_action_at=control["at"],
                last_action_outcome=control["outcome"], operations="read only; no process signals")


def gate_tests():
    assert json.dumps(run_gpu.json_safe({"values": [float("nan"), float("inf"), 1.0]}),
                      allow_nan=False) == '{"values": [null, null, 1.0]}'
    names = ["gpu_environment.py", "reference.py", "gpu_solver.py", "gpu_fused.py", "kernels.cu"]
    with tempfile.TemporaryDirectory(prefix="gate_", dir=ROOT / "validation") as tmp:
        folder = Path(tmp)
        (folder / "validation").mkdir()
        for name in names:
            shutil.copy2(ROOT / name, folder / name)
        receipt = ROOT / "validation/correctness_fused.json"
        target = folder / "validation/correctness_fused.json"
        shutil.copy2(receipt, target)
        with patch.object(run_gpu, "ROOT", folder):
            run_gpu.verified_backend("fused")
            kernel = folder / "kernels.cu"
            kernel.write_text(kernel.read_text(encoding="utf-8") + "\n// gate test\n", encoding="utf-8")
            try:
                run_gpu.verified_backend("fused")
            except RuntimeError as error:
                assert "Backend changed" in str(error)
            else:
                raise AssertionError("Changed backend was not rejected")
            record = json.loads(receipt.read_text(encoding="utf-8"))
            record["passed"] = False
            target.write_text(json.dumps(record), encoding="utf-8")
            try:
                run_gpu.verified_backend("fused")
            except RuntimeError as error:
                assert "has not passed" in str(error)
            else:
                raise AssertionError("Failed correctness receipt was not rejected")
    return ["strict_json_nonfinite_to_null", "valid_receipt_accepted", "modified_kernel_rejected", "failed_receipt_rejected"]


def cli_case(backend, grid, re, steps, compare_cpu=True):
    command = [sys.executable, "-X", "utf8", "-u", str(ROOT / "run_gpu.py"),
               "--backend", backend, "--grid", str(grid), "--re", str(re), "--max-iter", str(steps)]
    start = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", check=True, timeout=300)
    elapsed = time.perf_counter() - start
    (ROOT / f"validation/cli_{backend}_N{grid}_{steps}.log").write_text(result.stdout, encoding="utf-8")
    lines = [line.removeprefix("OUTPUT ") for line in result.stdout.splitlines() if line.startswith("OUTPUT ")]
    assert len(lines) == 1
    folder = Path(lines[0])
    assert folder.is_relative_to(ROOT / "results")
    summary = json.loads((folder / "run_summary.json").read_text(encoding="utf-8"))
    timing = json.loads((folder / "timing_and_population.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (folder / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["stop_reason"] == "max_iter" and summary["iteration"] == steps
    assert summary["dtype"] == "float64"
    assert rows[-1]["iteration"] == steps and rows[-1]["stop_reason"] == "max_iter"
    assert summary["gpu_source_sha256"] == run_gpu.verified_backend(backend)
    assert timing["population_check"]["finite"]
    reference = None
    if compare_cpu:
        reference = CPUSolver(make_config(grid=grid, re=re, max_iter=steps))
        list(reference.run())
    differences = {}
    with np.load(folder / "results.npz", allow_pickle=False) as archive:
        for name in ("rho", "ux", "uy"):
            field = archive[name]
            assert field.shape == (grid, grid) and field.dtype == np.float64
            assert np.isfinite(field).all()
            if reference is not None:
                differences[name] = float(np.max(np.abs(field - getattr(reference, name))))
                assert differences[name] <= 2e-12
    output = dict(backend=backend, grid=grid, Re=re, steps=steps, path=str(folder),
                  elapsed_with_process_start_and_export=elapsed,
                  compute_seconds=timing["compute_seconds"],
                  milliseconds_per_step=timing["compute_seconds"] * 1000 / steps,
                  final_diagnostics=summary["final_diagnostics"],
                  stop_reason=summary["stop_reason"], cpu_compared=compare_cpu,
                  field_max_errors=differences, population_check=timing["population_check"])
    print(f"CLI PASS {backend} N{grid} Re{re}, {steps} steps, {elapsed:.2f}s", flush=True)
    return output


def main():
    report = dict(passed=False, reference_files=verify_reference(), before=cpu_audit(),
                  source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                 for name in ("run_gpu.py", "verify_delivery.py")}, cli_cases=[])
    target = ROOT / "validation/delivery_checks.json"
    def save():
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    save()
    try:
        report["gate_tests"] = gate_tests()
        for case in (("array", 16, 100, 301, True), ("fused", 16, 100, 301, True),
                     ("fused", 256, 5468, 1000, True), ("fused", 256, 5468, 20000, False)):
            report["cli_cases"].append(cli_case(*case))
            save()
        report["after"] = cpu_audit()
        report["passed"] = True
    finally:
        save()
    print("PASS: " + str(target), flush=True)


if __name__ == "__main__":
    main()
