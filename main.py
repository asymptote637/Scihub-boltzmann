"""Command-line entry point for the customized 2D LBM simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import (
    SUPPORTED_CASE_TYPES,
    SUPPORTED_COLLISION_MODELS,
    SUPPORTED_MRT_PRESETS,
    SUPPORTED_PARAMETER_MODES,
    SUPPORTED_RAMP_PROFILES,
    BoundaryConfig,
    ObstacleConfig,
    OutputConfig,
    SolverConfig,
    ThermalBoundaryConfig,
    case_preset,
    derived_parameters,
    validate_config,
)
from lbm_solver import LBMSolver
from postprocess import save_results


def load_json_config(path: str | Path) -> SolverConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = case_preset(data.get("case_type", "lid_driven_cavity"), data.get("nx", 256), data.get("ny", 256))
    for key, value in data.items():
        if key in {"left", "right", "bottom", "top"}:
            setattr(cfg, key, BoundaryConfig(**value))
        elif key in {"thermal_left", "thermal_right", "thermal_bottom", "thermal_top"}:
            setattr(cfg, key, ThermalBoundaryConfig(**value))
        elif key == "obstacle":
            cfg.obstacle = ObstacleConfig(**value)
        elif key == "output":
            cfg.output = OutputConfig(**value)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 2D D2Q9 LBM simulation.")
    parser.add_argument("--case", choices=SUPPORTED_CASE_TYPES, default="lid_driven_cavity")
    parser.add_argument("--config-json", help="Optional JSON config exported by the UI.")
    parser.add_argument("--collision-model", choices=SUPPORTED_COLLISION_MODELS, default="BGK")
    parser.add_argument("--trt-lambda", type=float, default=3.0 / 16.0)
    parser.add_argument("--mrt-preset", choices=SUPPORTED_MRT_PRESETS, default="lallemand_luo")
    parser.add_argument("--mrt-s-e", type=float, default=1.64)
    parser.add_argument("--mrt-s-epsilon", type=float, default=1.54)
    parser.add_argument("--mrt-s-q", type=float, default=1.90)
    parser.add_argument("--parameter-mode", choices=SUPPORTED_PARAMETER_MODES)
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--length-phys", type=float)
    parser.add_argument("--velocity-phys", type=float)
    parser.add_argument("--nu-phys", type=float)
    parser.add_argument("--prandtl", type=float)
    parser.add_argument("--rayleigh", type=float)
    parser.add_argument("--temperature-hot", type=float)
    parser.add_argument("--temperature-cold", type=float)
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--re", type=float, default=100.0)
    parser.add_argument("--u-ref", type=float, default=0.05)
    parser.add_argument("--body-force-x", type=float)
    parser.add_argument("--body-force-y", type=float)
    parser.add_argument("--ramp-profile", choices=SUPPORTED_RAMP_PROFILES, default="smoothstep")
    parser.add_argument("--ramp-steps", type=int, default=1000)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--min-iter", type=int, default=200)
    parser.add_argument("--report-interval", type=int, default=100)
    args = parser.parse_args()

    if args.config_json:
        cfg = load_json_config(args.config_json)
    else:
        cfg = case_preset(args.case, nx=args.nx, ny=args.ny, u_ref=args.u_ref)
        cfg.collision_model = args.collision_model
        cfg.trt_magic_parameter = args.trt_lambda
        cfg.mrt_preset = args.mrt_preset
        cfg.mrt_s_e = args.mrt_s_e
        cfg.mrt_s_epsilon = args.mrt_s_epsilon
        cfg.mrt_s_q = args.mrt_s_q
        if args.parameter_mode is not None:
            cfg.parameter_mode = args.parameter_mode
        cfg.tau_target = args.tau
        if args.length_phys is not None:
            cfg.length_phys = args.length_phys
        if args.velocity_phys is not None:
            cfg.velocity_phys = args.velocity_phys
        if args.nu_phys is not None:
            cfg.nu_phys = args.nu_phys
        if args.prandtl is not None:
            cfg.prandtl = args.prandtl
        if args.rayleigh is not None:
            cfg.rayleigh = args.rayleigh
        if args.temperature_hot is not None:
            cfg.temperature_hot = args.temperature_hot
        if args.temperature_cold is not None:
            cfg.temperature_cold = args.temperature_cold
        if cfg.case_type == "natural_convection_cavity":
            cfg.thermal_left.temperature = cfg.temperature_hot
            cfg.thermal_right.temperature = cfg.temperature_cold
        elif cfg.case_type in {"rayleigh_benard_convection", "heated_channel_flow"}:
            cfg.thermal_bottom.temperature = cfg.temperature_hot
            cfg.thermal_top.temperature = cfg.temperature_cold
        cfg.reynolds = args.re
        if args.body_force_x is not None:
            cfg.body_force_x = args.body_force_x
        if args.body_force_y is not None:
            cfg.body_force_y = args.body_force_y
        cfg.ramp_profile = args.ramp_profile
        cfg.ramp_steps = args.ramp_steps
        cfg.max_iter = args.max_iter
        cfg.min_iter = args.min_iter
        cfg.report_interval = args.report_interval

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
