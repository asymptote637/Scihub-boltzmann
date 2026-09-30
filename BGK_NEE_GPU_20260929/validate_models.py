"""Validate the extended model family and write its source-bound receipt.

Does not launch production jobs or alter the default dashboard database.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

from datetime import datetime
from itertools import product
import json
import time

import numpy as np
import pytest
import gpu_environment
import cupy as cp

from reference import ROOT, REFERENCE, make_config, verify_reference
from cavity_models import CavityCPUSolver
from gpu_solver import ArrayGPUSolver
from gpu_fused import FusedGPUSolver
from config import domain_axis
from model_validation import fingerprints


def converged_backend_comparison(collision, boundary):
    cfg = make_config(grid=24, re=100, collision=collision, boundary=boundary, max_iter=30000)
    solvers = [CavityCPUSolver(cfg), ArrayGPUSolver(cfg), FusedGPUSolver(cfg)]
    for step in range(1, 30001):
        for solver in solvers:
            solver.step()
        if any(s.stop_reason != "running" for s in solvers):
            break
    assert all(s.stop_reason == "converged" for s in solvers), [(s.stop_reason, s.iteration) for s in solvers]
    assert len({s.iteration for s in solvers}) == 1
    cpu = solvers[0]
    errors = {}
    for gpu in solvers[1:]:
        differences = {k:float(np.max(np.abs(getattr(cpu,k)-cp.asnumpy(getattr(gpu,k))))) for k in ("f","rho","ux","uy")}
        assert max(differences.values()) < 5e-11, differences
        errors[gpu.backend] = differences
    drift = abs(cpu.rho.sum()/cpu.mass0-1)
    if boundary == "halfway":
        assert drift < 1e-10
    record = dict(collision=collision, boundary=boundary, grid=24, re=100, iteration=step,
                  stop_reason="converged", max_field_errors=errors, mass_drift=float(drift))
    print(json.dumps(record), flush=True)
    return record


def cavity_benchmark(collision, boundary):
    cfg = make_config(grid=64, re=100, collision=collision, boundary=boundary, max_iter=50000)
    s = FusedGPUSolver(cfg)
    for report in s.run():
        pass
    assert s.stop_reason == "converged", (collision, boundary, s.iteration, s.stop_reason)
    xo, width = domain_axis(cfg, "x")
    yo, height = domain_axis(cfg, "y")
    x, y = (np.arange(s.nx)+xo)/width, (np.arange(s.ny)+yo)/height
    u = np.array([np.interp(.5, x, row) for row in s.host.ux])/cfg.flow.U_ref
    v = np.array([np.interp(.5, y, col) for col in s.host.uy.T])/cfg.flow.U_ref
    errors = {}
    for axis, coordinates, values in (("u", y, u), ("v", x, v)):
        data = np.loadtxt(REFERENCE / f"calculation/source_snapshot/references/ghia_{axis}_source.txt")
        # Compare only tabulated interior points inside the fluid-node span.
        # HBB has no stored wall nodes; never clamp wall data to fluid endpoints.
        selected = data[(data[:,0] > 0) & (data[:,0] < 1) & (data[:,0] >= coordinates[0]) & (data[:,0] <= coordinates[-1])]
        error = np.interp(selected[:,0], coordinates, values) - selected[:,1]
        errors[axis] = dict(points=len(error), max_abs=float(np.abs(error).max()), rms=float(np.sqrt(np.mean(error**2))))
        assert errors[axis]["max_abs"] < .025, (collision, boundary, errors)
    mass = abs(float(s.host.rho.sum())/s.mass0 - 1)
    if boundary == "halfway":
        assert mass < 1e-10
    np.savez_compressed(ROOT / f"validation/model_{collision}_{boundary}_N64_Re100.npz",
                        x=x, y=y, rho=s.host.rho, ux=s.host.ux, uy=s.host.uy)
    result = dict(collision=collision, boundary=boundary, grid=64, re=100, iteration=s.iteration,
                  stop_reason=s.stop_reason, mass_drift=mass, residual=report.residual,
                  ghia_interior_errors=errors, criterion_max_abs=.025)
    print(json.dumps(result), flush=True)
    return result


def main():
    started = time.perf_counter()
    receipt = dict(passed=False, started_at=datetime.now().astimezone().isoformat(),
                   source_sha256=fingerprints(), frozen_sources_verified=verify_reference(),
                   scope="BGK/MRT x NEE/HBB, float64 CPU/array/CUDA; low-Re cavity only",
                   backend_comparisons=[], benchmarks=[])
    path = ROOT / "validation/correctness_models.json"
    def save():
        path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    save()
    try:
        code = pytest.main([str(ROOT / "tests/test_cavity_models.py"), "-q", "-p", "no:cacheprovider",
                            "--basetemp", str(ROOT / "validation/pytest-model-verification"),
                            "--junitxml", str(ROOT / "validation/model_unit_tests.xml")])
        receipt["unit_test_exit_code"] = int(code)
        assert code == 0
        for collision, boundary in product(("BGK", "MRT"), ("nee", "halfway")):
            receipt["backend_comparisons"].append(converged_backend_comparison(collision, boundary))
            save()
            receipt["benchmarks"].append(cavity_benchmark(collision, boundary))
            save()
        assert receipt["source_sha256"] == fingerprints(), "Sources changed during verification"
        receipt["passed"] = True
    except Exception as error:
        receipt["error"] = repr(error)
        raise
    finally:
        receipt["seconds"] = time.perf_counter() - started
        save()
    print(f"PASS: {path}", flush=True)


if __name__ == "__main__":
    main()
