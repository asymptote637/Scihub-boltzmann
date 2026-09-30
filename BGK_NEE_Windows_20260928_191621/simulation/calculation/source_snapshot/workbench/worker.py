"""One isolated run. Stdout is bounded NDJSON; files carry durable results."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from lbm_solver import LBMSolver
from workbench.project import atomic_json, clean_json, config_from_dict, load_case, now
from workbench.validation import evaluate


def emit(event: str, **data) -> None:
    print(json.dumps(clean_json(dict(event=event, **data)), ensure_ascii=False, allow_nan=False), flush=True)


def run_job(root: Path) -> int:
    # Only a freshly prepared run may be executed; never overwrite a prior run.
    if not (root / "request.json").is_file():
        emit("failed", error="运行目录缺少 request.json")
        return 2
    try:
        with (root / "worker.lock").open("x", encoding="utf-8") as handle:
            handle.write(now())
    except FileExistsError:
        emit("failed", error="此运行已启动过；请创建新的运行目录。")
        return 2
    started = time.monotonic()
    job = dict(state="starting", started_at=now())
    try:
        document = load_case(root / "request.json")
        config = config_from_dict(document["config"])
        # Request files cannot redirect output into unrelated directories.
        config.output.output_dir = str(root / "result")
        config.output.save_npz = config.output.save_csv = True
        job.update(name=document["name"], state="running")
        atomic_json(root / "job.json", job)
        emit("state", state="running")
        solver = LBMSolver(config)
        last_preview = -float("inf")
        for report in solver.run(should_stop=lambda: (root / "cancel.request").exists()):
            timestamp = time.monotonic()
            if timestamp-last_preview >= .75 or report.stop_reason != "running":
                temporary = root / ".preview.tmp.npz"
                snapshot = solver.field_snapshot()
                # Preview is sampled to bound GUI memory; final output is full resolution.
                stride = max(1, int(np.ceil(max(config.grid.NX, config.grid.NY)/256)))
                np.savez_compressed(temporary, **{k: v[::stride, ::stride] for k, v in snapshot.items()},
                                    stride=stride, iteration=solver.iteration)
                temporary.replace(root / "preview.npz")
                emit("progress", report=asdict(report), elapsed_seconds=timestamp-started,
                     max_iter=config.convergence.max_iter)
                last_preview = timestamp
        job["state"] = "finalizing"
        atomic_json(root / "job.json", job)
        emit("state", state="finalizing")
        result = solver.finalize()
        from workbench.project import read_json
        summary = read_json(root / "result/run_summary.json")
        validation = evaluate(result.config, summary,
                              dict(rho=result.rho, ux=result.ux, uy=result.uy, solid_mask=result.solid_mask))
        atomic_json(root / "result/validation.json", validation)
        if config.output.save_png and result.stop_reason != "diverged":
            from matplotlib.figure import Figure
            from workbench.plots import draw_fields, draw_validation
            figures = root / "result/figures"
            figures.mkdir(exist_ok=True)
            figure = Figure(figsize=(11, 7), constrained_layout=True)
            draw_fields(figure, result.config, solver.field_snapshot(), result.history)
            figure.savefig(figures / "workbench_fields.png", dpi=160, facecolor="white")
            if validation["profiles"]:
                figure = Figure(figsize=(10, 4), constrained_layout=True)
                draw_validation(figure, validation)
                figure.savefig(figures / "benchmark.png", dpi=160, facecolor="white")
                figure.savefig(figures / "benchmark.svg", facecolor="white")
        # Solver summary hashes the core; also record this application's sources.
        job.update(state=result.stop_reason, finished_at=now(), elapsed_seconds=time.monotonic()-started,
                   validation_status=validation["status"],
                   application_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in Path(__file__).parent.glob("*.py")})
        atomic_json(root / "job.json", job)
        emit("finished", state=result.stop_reason, validation=validation["status"])
        return 1 if result.stop_reason == "diverged" else 0
    except Exception as exc:
        detail = traceback.format_exc()
        with (root / "worker.log").open("a", encoding="utf-8") as handle:
            handle.write(detail)
        job.update(state="failed", error=str(exc), finished_at=now(), elapsed_seconds=time.monotonic()-started)
        atomic_json(root / "job.json", job)
        emit("failed", error=str(exc))
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    return run_job(args.run_dir.expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main())
