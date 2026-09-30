"""Run one independently initialized case using a validated GPU backend."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import gpu_environment
import cupy as cp
import numpy as np
from gpu_solver import ArrayGPUSolver
from reference import ROOT, make_config, config_to_dict, verify_reference


def json_safe(item):
    if isinstance(item, dict):
        return {k: json_safe(v) for k, v in item.items()}
    if isinstance(item, list):
        return [json_safe(v) for v in item]
    if isinstance(item, float) and not np.isfinite(item):
        return None
    return item


def json_write(path, value):
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def verified_backend(name):
    verify_reference()
    from model_validation import verified_models
    verified_models()
    receipt_path = ROOT / f"validation/correctness_{name}.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"Run validate_gpu.py --backend {name} --full first")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not receipt.get("passed") or not receipt.get("low_re", {}).get("passed"):
        raise RuntimeError("Complete correctness verification has not passed")
    files = ["gpu_environment.py", "reference.py", "gpu_solver.py", "cavity_models.py"]
    if name == "fused":
        files += ["gpu_fused.py", "kernels.cu"]
    for file in files:
        actual = hashlib.sha256((ROOT / file).read_bytes()).hexdigest()
        if actual != receipt["source_sha256"].get(file):
            raise RuntimeError("Backend changed since validation: " + file)
    return {file: receipt["source_sha256"][file] for file in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("array", "fused"), default="fused")
    parser.add_argument("--collision", choices=("BGK", "MRT"), default="BGK")
    parser.add_argument("--boundary", choices=("nee", "halfway"), default="nee")
    parser.add_argument("--mrt-s-e", type=float, default=1.64)
    parser.add_argument("--mrt-s-eps", type=float, default=1.54)
    parser.add_argument("--mrt-s-q", type=float, default=1.9)
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--re", type=float, default=5468)
    parser.add_argument("--max-iter", type=int, default=1000000)
    parser.add_argument("--min-iter", type=int, default=2000)
    parser.add_argument("--ramp-steps", type=int, default=500)
    parser.add_argument("--report-interval", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--lid-speed", type=float, default=0.04)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    fingerprints = verified_backend(args.backend)
    config = make_config(**{key: getattr(args, key) for key in (
        "grid", "re", "max_iter", "min_iter", "ramp_steps", "report_interval", "tol", "lid_speed",
        "collision", "boundary", "mrt_s_e", "mrt_s_eps", "mrt_s_q")})
    solver_type = ArrayGPUSolver
    if args.backend == "fused":
        from gpu_fused import FusedGPUSolver
        solver_type = FusedGPUSolver
    solver = solver_type(config)
    if args.dry_run:
        print(json.dumps(dict(backend=solver.backend, dtype="float64", gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                              config=config_to_dict(config, solver.transport)), indent=2))
        return 0
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = ROOT / "results" / f"{solver.backend}_{args.collision}_{args.boundary}_N{args.grid}_Re{args.re:g}_{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    solver.config.output.output_dir = str(folder)
    solver.config.output.save_png = False
    metadata = dict(backend=solver.backend, collision=args.collision, boundary=args.boundary, dtype="float64", initialized_from="rest",
                    gpu_source_sha256=fingerprints, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                    cupy_version=cp.__version__, gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
    json_write(folder / "run_request.json", dict(created_at=dt.datetime.now().astimezone().isoformat(),
               config=config_to_dict(solver.config, solver.transport), residual_steps=1, **metadata))
    print(f"OUTPUT {folder}", flush=True)
    print(f"{solver.backend}, float64, N={args.grid}, Re={args.re:g}, tau={solver.transport['tau']:.16g}", flush=True)
    start = time.perf_counter()
    with (folder / "progress.jsonl").open("w", encoding="utf-8") as log:
        def save_report(report):
            row = dict(iteration=report.iteration, elapsed_seconds=time.perf_counter()-start,
                       residual=report.residual, gpu_decision_residual=solver.decision_residual,
                       mass_drift=report.mass_drift, max_mach=report.max_mach,
                       stop_reason=report.stop_reason)
            log.write(json.dumps(json_safe(row), allow_nan=False) + "\n")
            log.flush()
            print(f"{row['iteration']:>8} | {row['elapsed_seconds']:.3f}s | residual {row['residual']:.8e} "
                  f"| mass {row['mass_drift']:.8e} | Ma {row['max_mach']:.6f} | {row['stop_reason']}", flush=True)
        try:
            for report in solver.run():
                save_report(report)
        except KeyboardInterrupt:
            solver.host.cancel()
            save_report(solver.report())
    cp.cuda.get_current_stream().synchronize()
    compute = time.perf_counter() - start
    solver.finalize()
    summary = json.loads((folder / "run_summary.json").read_text(encoding="utf-8"))
    summary.update(metadata)
    summary["gpu_decision_residual"] = solver.decision_residual
    summary["validation_scope"] = "CPU equivalence tests; not a claim of high-Re or grid-independent physical validation"
    json_write(folder / "run_summary.json", summary)
    population = solver.host.f
    json_write(folder / "timing_and_population.json", dict(
        iterations=solver.iteration, stop_reason=solver.stop_reason, compute_seconds=compute,
        total_seconds=time.perf_counter()-start,
        population_check=dict(finite=bool(np.isfinite(population).all()), min=float(population.min()),
                              max=float(population.max()), negative_count=int((population < 0).sum()), dtype=str(population.dtype)),
        **metadata))
    json_write(ROOT / "latest_gpu_run.json", dict(path=str(folder), stop_reason=solver.stop_reason,
                                                 iteration=solver.iteration, backend=solver.backend))
    print(f"FINISHED {solver.stop_reason}: {folder}", flush=True)
    return 1 if solver.stop_reason == "diverged" else 0


if __name__ == "__main__":
    raise SystemExit(main())
