"""Command-line entry point for the customized 2D LBM simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import BoundaryConfig, ObstacleConfig, OutputConfig, SolverConfig, case_preset, derived_parameters, validate_config
from lbm_solver import LBMSolver
from postprocess import save_results


def load_json_config(path: str | Path) -> SolverConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = case_preset(data.get("case_type", "lid_driven_cavity"), data.get("nx", 256), data.get("ny", 256))
    for key, value in data.items():
        if key in {"left", "right", "bottom", "top"}:
            setattr(cfg, key, BoundaryConfig(**value))
        elif key == "obstacle":
            cfg.obstacle = ObstacleConfig(**value)
        elif key == "output":
            cfg.output = OutputConfig(**value)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 2D D2Q9-BGK LBM simulation.")
    parser.add_argument("--case", default="lid_driven_cavity")
    parser.add_argument("--config-json", help="Optional JSON config exported by the UI.")
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--re", type=float, default=100.0)
    parser.add_argument("--u-ref", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--min-iter", type=int, default=200)
    parser.add_argument("--report-interval", type=int, default=100)
    args = parser.parse_args()

    if args.config_json:
        cfg = load_json_config(args.config_json)
    else:
        cfg = case_preset(args.case, nx=args.nx, ny=args.ny)
        cfg.reynolds = args.re
        cfg.u_ref = args.u_ref
        cfg.max_iter = args.max_iter
        cfg.min_iter = args.min_iter
        cfg.report_interval = args.report_interval
        if cfg.case_type == "lid_driven_cavity":
            cfg.top.ux = cfg.u_ref

    errors, warnings = validate_config(cfg)
    if errors:
        raise SystemExit("Invalid configuration:\n" + "\n".join(f"- {err}" for err in errors))
    for warning in warnings:
        print(f"Warning: {warning}")
    print("Derived:", json.dumps(derived_parameters(cfg), indent=2))

    solver = LBMSolver(cfg)
    for report in solver.run():
        print(
            f"iter={report.iteration} residual={report.residual:.3e} "
            f"q={report.q if report.q is not None else 'NA'} "
            f"mass_drift={report.mass_drift:.3e} status={report.status}"
        )

    run_dir = save_results(solver, cfg)
    print(f"Saved results to: {run_dir}")


if __name__ == "__main__":
    main()

