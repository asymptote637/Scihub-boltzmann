"""Cooperative CPU/GPU worker for validated cavity models."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import time
import traceback

import numpy as np
import psutil

from .store import DEFAULT_WORKSPACE, ROOT, Store, atomic_json, encode, now, read_json, transport, validate_parameters


def write_preview(solver, folder):
    if hasattr(solver, "field_arrays"):
        from research_solver import atomic_npz
        atomic_npz(folder / "preview.npz", **solver.field_arrays(preview=True))
        return
    host = getattr(solver, "host", solver)
    stride = max(1, (host.nx + 255) // 256)
    xs = np.unique(np.r_[np.arange(0, host.nx, stride), host.nx - 1])
    ys = np.unique(np.r_[np.arange(0, host.ny, stride), host.ny - 1])
    target = folder / "preview.npz"
    temporary = folder / "preview.tmp"
    if hasattr(host, "config"):
        from config import domain_axis
        xo, width = domain_axis(host.config, "x")
        yo, height = domain_axis(host.config, "y")
    else:
        xo, width, yo, height = 0, host.nx - 1, 0, host.ny - 1
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, x=(xs + xo) / width, y=(ys + yo) / height,
                                rho=host.rho[np.ix_(ys, xs)], ux=host.ux[np.ix_(ys, xs)], uy=host.uy[np.ix_(ys, xs)],
                                iteration=np.int64(host.iteration))
        # Windows readers or sync clients can briefly deny replacement of a complete preview.
        for attempt, delay in enumerate((0, 0.01, 0.03, 0.06)):
            if delay:
                time.sleep(delay)
            try:
                temporary.replace(target)
                break
            except PermissionError:
                if attempt == 3:
                    raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def run_case(store, case_id):
    case = store.get(case_id)
    if case is None or case["status"] != "starting":
        raise ValueError("Only a scheduler-claimed job can be started")
    folder = Path(case["data_dir"]).resolve()
    if folder != store.workspace / "runs" / case_id:
        raise ValueError("Invalid owned output directory")
    folder.mkdir(parents=True, exist_ok=True)
    store.update_job(case_id, pid=os.getpid(), process_started=psutil.Process().create_time(), heartbeat=now())
    try:
        params = validate_parameters(case["params"])
        from reference import PLAN, config_to_dict, make_config, verify_reference
        from cavity_models import CavityCPUSolver
        from model_validation import verified_models
        verify_reference()
        model_sources = verified_models()
        from research_options import generic_engine
        from research_runtime import build_config, solver_class
        config = build_config(params)
        config.output.output_dir = str(folder)
        config.output.save_png = False
        estimates = transport(params)
        if estimates["estimated_host_bytes"] > psutil.virtual_memory().available * 0.8:
            raise ValueError("预计内存需求超过当前可用内存的 80%，未分配计算数组")
        solver_type = solver_class(params)
        metadata = dict(backend="cpu", dtype="float64", initialized_from="rest", frozen_source_sha256=PLAN["source_sha256"],
                        worker_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        model_source_sha256=model_sources, collision=params["collision"], boundary=params["boundary"])
        if generic_engine(params):
            from research_validation import verified_research
            metadata["research_source_sha256"] = verified_research(params["backend"])
            metadata["initialized_from"] = params.get("resume_from") or params["scenario"]
            metadata["lattice"] = params["lattice"]
        if params["backend"] != "cpu":
            from run_gpu import verified_backend
            import cupy as cp
            metadata["gpu_source_sha256"] = verified_backend(params["backend"])
            free, total = cp.cuda.runtime.memGetInfo()
            if estimates["estimated_gpu_bytes"] > free * 0.8:
                raise ValueError("预计显存需求超过当前可用显存的 80%，未分配计算数组")
            metadata.update(backend=solver_type.backend, gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                            cupy_version=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion())
        solver = solver_type(config)
        host = getattr(solver, "host", solver)
        config_dict = config_to_dict(solver.config, solver.transport)
        atomic_json(folder / "run_request.json", dict(created_at=now(), config=config_dict, **metadata))
        store.update_case(case_id, config_json=encode(config_dict), status="running")
        original_step = solver.step
        started = time.perf_counter()
        paused_seconds = 0.0
        last_control = -1.0
        last_preview = -1.0
        last_preview_warning = -float("inf")
        preview_status = dict(skipped_updates=0, last_written_iteration=None, warning=None)
        latest_metrics = {}

        def cooperative_step():
            nonlocal last_control, paused_seconds
            current_time = time.perf_counter()
            if current_time - last_control >= 0.1:
                job = store.get(case_id)
                if job["desired"] == "pause":
                    pause_start = time.perf_counter()
                    store.update_case(case_id, status="paused")
                    while job["desired"] == "pause":
                        store.update_job(case_id, heartbeat=now())
                        time.sleep(0.2)
                        job = store.get(case_id)
                    paused_seconds += time.perf_counter() - pause_start
                    store.update_case(case_id, status="running")
                if job["desired"] == "cancel":
                    host.cancel()
                    store.update_case(case_id, status="cancelling")
                    return
                store.update_job(case_id, heartbeat=now())
                last_control = time.perf_counter()
            original_step()

        # Control runs between complete steps; collision and stopping logic are unchanged.
        solver.step = cooperative_step
        with (folder / "progress.jsonl").open("w", encoding="utf-8") as progress:
            try:
                for report in solver.run():
                    latest_metrics = dict(iteration=report.iteration, residual=report.residual, mass_drift=report.mass_drift,
                                          max_mach=report.max_mach, min_density=report.min_density,
                                          elapsed_seconds=time.perf_counter() - started - paused_seconds,
                                          wall_seconds=time.perf_counter() - started, stop_reason=report.stop_reason)
                    if generic_engine(params) and solver.history:
                        latest_metrics.update(solver.history[-1])
                    if params["backend"] != "cpu":
                        latest_metrics["gpu_decision_residual"] = solver.decision_residual
                    if time.perf_counter() - last_preview > 2 or report.stop_reason != "running":
                        try:
                            write_preview(solver, folder)
                        except OSError as error:
                            preview_status["skipped_updates"] += 1
                            preview_status["warning"] = f"{type(error).__name__}: {error}"
                            if time.perf_counter() - last_preview_warning >= 30 or report.stop_reason != "running":
                                print(f"WARNING preview update skipped at step {report.iteration}; "
                                      f"calculation continues: {preview_status['warning']}", flush=True)
                                last_preview_warning = time.perf_counter()
                        else:
                            if preview_status["warning"]:
                                print(f"INFO preview update recovered at step {report.iteration}", flush=True)
                            preview_status.update(last_written_iteration=int(host.iteration), warning=None)
                        last_preview = time.perf_counter()
                    latest_metrics.update(preview_warning=preview_status["warning"],
                                          preview_skipped_updates=preview_status["skipped_updates"],
                                          preview_iteration=preview_status["last_written_iteration"])
                    progress.write(encode(latest_metrics) + "\n")
                    progress.flush()
                    store.update_case(case_id, metrics_json=encode(latest_metrics))
                    store.update_job(case_id, heartbeat=now())
            except KeyboardInterrupt:
                host.cancel()
                solver.report()
        active_seconds = time.perf_counter() - started - paused_seconds
        solver.finalize()
        summary = read_json(folder / "run_summary.json", {})
        summary.update(metadata, active_compute_seconds=active_seconds, paused_seconds=paused_seconds,
                       preview_status=preview_status,
                       validation_scope=("Expanded lattice/force/thermal/backend tests; see correctness_research.json; "
                                         "user cases still require time and grid convergence" if generic_engine(params) else
                                         "Validated BGK/MRT x NEE/HBB; low-Re checks do not establish high-Re or grid independence"))
        atomic_json(folder / "run_summary.json", summary)
        population = host.f
        timing = dict(iterations=solver.iteration, stop_reason=solver.stop_reason, compute_seconds=active_seconds,
                      paused_seconds=paused_seconds, total_seconds=time.perf_counter() - started,
                      population_check=dict(finite=bool(np.isfinite(population).all()), min=float(np.min(population)),
                                            max=float(np.max(population)), negative_count=int((population < 0).sum())), **metadata)
        atomic_json(folder / "timing_and_population.json", timing)
        latest_metrics.update(summary.get("final_diagnostics", {}), iteration=solver.iteration,
                              elapsed_seconds=active_seconds, wall_seconds=timing["total_seconds"], paused_seconds=paused_seconds)
        store.update_case(case_id, status=solver.stop_reason, metrics_json=encode(latest_metrics))
        store.update_job(case_id, heartbeat=now())
        print(f"FINISHED {case_id} {solver.stop_reason}", flush=True)
    except Exception as error:
        traceback.print_exc()
        store.update_case(case_id, status="failed")
        store.update_job(case_id, error=f"{type(error).__name__}: {error}", heartbeat=now())
        atomic_json(folder / "worker_error.json", dict(error=str(error), type=type(error).__name__, at=now()))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    run_case(Store(args.workspace), args.case)


if __name__ == "__main__":
    main()
