"""Warm, repeated CPU/array/fused timings including every-step safety checks."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import gpu_environment
import cupy as cp
import numpy as np
import psutil
from reference import ROOT, CPUSolver, make_config
from gpu_solver import ArrayGPUSolver
from run_gpu import verified_backend
from validate_gpu import source_fingerprints


def gpu_telemetry():
    command = ["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,utilization.gpu,power.draw,clocks.sm,clocks.mem",
               "--format=csv,noheader"]
    return subprocess.run(command, capture_output=True, text=True, check=True, timeout=10).stdout.strip()


def advance(solver, count):
    for _ in range(count):
        solver.step()
        if solver.iteration == 1 or solver.iteration % solver.config.convergence.report_interval == 0:
            solver.report()
        if solver.stop_reason != "running":
            raise RuntimeError(f"Benchmark stopped early at {solver.iteration}: {solver.stop_reason}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--grid", type=int, default=256)
    args = parser.parse_args()
    if min(args.steps, args.warmup, args.repeats) < 1:
        raise ValueError("Positive timing counts are required")
    verified_backend("array")
    verified_backend("fused")
    from gpu_fused import FusedGPUSolver
    factories = dict(cpu=CPUSolver, array=ArrayGPUSolver, fused=FusedGPUSolver)
    config = make_config(grid=args.grid, re=5468, max_iter=args.steps + args.warmup)
    production_commands = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = process.info["cmdline"] or []
            if any("BGK_NEE_Windows_20260928_191621" in arg and "run_baseline.py" in arg for arg in cmd):
                production_commands.append(process.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    report = dict(started_at=dt.datetime.now().astimezone().isoformat(), passed=False,
                  grid=args.grid, Re=5468, dtype="float64", steps=args.steps, warmup=args.warmup,
                  repetitions=args.repeats, periodic_reports_included=True,
                  every_step_residual_and_protection_included=True,
                  excludes="initialization, first compilation, final output compression",
                  background_cpu_processes=production_commands,
                  gpu_before=gpu_telemetry(), source_sha256=source_fingerprints(), trials=[])
    output = ROOT / "validation/performance.json"
    reference_fields = None
    def save():
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save()
    for repeat in range(args.repeats):
        order = ["cpu", "array", "fused"]
        order = order[repeat % 3:] + order[:repeat % 3]
        for backend in order:
            solver = factories[backend](config)
            advance(solver, args.warmup)
            cp.cuda.get_current_stream().synchronize()
            start_event, end_event = cp.cuda.Event(), cp.cuda.Event()
            if backend != "cpu":
                start_event.record()
            start = time.perf_counter()
            advance(solver, args.steps)
            if backend != "cpu":
                end_event.record()
                end_event.synchronize()
            elapsed = time.perf_counter() - start
            fields = {key: getattr(solver, key).copy() if backend == "cpu" else cp.asnumpy(getattr(solver, key))
                      for key in ("f", "rho", "ux", "uy")}
            if reference_fields is None:
                reference_fields = fields
            max_difference = max(float(np.max(np.abs(fields[k] - reference_fields[k]))) for k in fields)
            assert max_difference < 2e-12, (backend, max_difference)
            row = dict(backend=backend, repetition=repeat + 1, seconds=elapsed,
                       milliseconds_per_step=elapsed * 1000 / args.steps,
                       steps_per_second=args.steps / elapsed,
                       mlups=args.grid ** 2 * args.steps / elapsed / 1e6,
                       final_iteration=solver.iteration, max_field_difference=max_difference,
                       cuda_event_span_ms=None if backend == "cpu" else cp.cuda.get_elapsed_time(start_event, end_event))
            report["trials"].append(row)
            save()
            print(f"{backend} trial {repeat+1}: {row['milliseconds_per_step']:.6f} ms/step, "
                  f"{row['steps_per_second']:.2f} steps/s", flush=True)
            del solver
    summary = {}
    for backend in factories:
        times = [r["milliseconds_per_step"] for r in report["trials"] if r["backend"] == backend]
        summary[backend] = dict(median_ms=statistics.median(times), min_ms=min(times), max_ms=max(times))
    for backend in summary:
        summary[backend]["speedup_vs_cpu"] = summary["cpu"]["median_ms"] / summary[backend]["median_ms"]
        summary[backend]["speedup_vs_array"] = summary["array"]["median_ms"] / summary[backend]["median_ms"]
    report.update(summary=summary, gpu_after=gpu_telemetry(), passed=True,
                  completed_at=dt.datetime.now().astimezone().isoformat())
    save()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
