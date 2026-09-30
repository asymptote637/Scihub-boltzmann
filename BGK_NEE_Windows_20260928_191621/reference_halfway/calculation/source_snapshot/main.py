"""Command-line entry point for the 2D D2Q9-BGK LBM simulator."""

from __future__ import annotations

import argparse
import sys

from config import (
    CASE_TYPES,
    BoundaryConfig,
    ConvergenceConfig,
    FlowConfig,
    GridConfig,
    OutputConfig,
    SimulationConfig,
    default_boundaries_for_case,
    validate_config,
)
from lbm_solver import LBMSolver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a 2D D2Q9-BGK LBM simulation.")
    parser.add_argument("--case", choices=CASE_TYPES, default="lid_driven_cavity", dest="case_type")
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--re", type=float, default=100.0)
    parser.add_argument("--u-ref", type=float, default=0.05)
    parser.add_argument("--rho0", type=float, default=1.0)
    parser.add_argument("--l-ref", type=float, help="Effective reference length; defaults to boundary geometry.")
    parser.add_argument("--nu", type=float, help="Explicit lattice viscosity; overrides --re.")
    parser.add_argument("--force-x", type=float, default=0.0, help="Lattice acceleration ax.")
    parser.add_argument("--force-y", type=float, default=0.0, help="Lattice acceleration ay.")
    parser.add_argument("--initial", choices=["rest", "uniform", "couette"], default="rest")
    parser.add_argument("--boundary-scheme", choices=["halfway_bounce_back", "non_equilibrium_extrapolation", "no_slip_bounce_back"], default="halfway_bounce_back")
    parser.add_argument("--steady-windows", type=int, default=3)
    parser.add_argument("--fixed-steps", action="store_true", help="Transient run; do not stop on steady residual.")
    parser.add_argument("--mass-tol", type=float, default=1e-4)
    parser.add_argument("--max-mach", type=float, default=0.3)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--min-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--ramp-steps", type=int, default=500)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--obstacle", choices=["none", "circle", "rectangle"], default="none")
    parser.add_argument("--no-png", action="store_true", help="Do not save PNG figures.")
    return parser


def config_from_args(args: argparse.Namespace) -> SimulationConfig:
    grid = GridConfig(NX=args.nx, NY=args.ny, L_ref=args.l_ref, obstacle_type=args.obstacle)
    flow = FlowConfig(rho0=args.rho0, U_ref=args.u_ref, Re=args.re,
        nu_lattice=args.nu, body_force_x=args.force_x, body_force_y=args.force_y,
        initial_velocity=args.initial)
    boundaries = default_boundaries_for_case(args.case_type, args.u_ref, args.boundary_scheme)
    convergence = ConvergenceConfig(
        tol=args.tol,
        max_iter=args.max_iter,
        min_iter=args.min_iter,
        report_interval=args.report_interval,
        ramp_steps=args.ramp_steps,
        consecutive_reports=args.steady_windows,
        steady=not args.fixed_steps,
        mass_tolerance=args.mass_tol,
        max_mach=args.max_mach,
    )
    output = OutputConfig(output_dir=args.output_dir, save_png=not args.no_png)
    return SimulationConfig(
        case_type=args.case_type,
        grid=grid,
        flow=flow,
        convergence=convergence,
        output=output,
        boundaries=boundaries,
        boundary_scheme_default=args.boundary_scheme,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    errors, warnings, transport = validate_config(config)
    if errors:
        print("Configuration errors:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(
        "derived: "
        f"Re={transport['Re']:.6g}, Ma={transport['Ma']:.6g}, "
        f"nu={transport['nu_lattice']:.6g}, tau={transport['tau']:.6g}, omega={transport['omega']:.6g}"
    )

    solver = LBMSolver(config)
    last = None
    for report in solver.run():
        last = report
        print(
            f"iter={report.iteration:8d} residual={report.residual:.6e} "
            f"q={report.q:.4g} q_avg={report.q_avg:.4g} "
            f"mass_drift={report.mass_drift:.3e} max_u={report.max_velocity:.5f} "
            f"rho_min={report.min_density:.6g} Ma_max={report.max_mach:.4g} "
            f"{report.message}"
        )
    result = solver.finalize(save_outputs=True)
    print(f"finished: {result.message}")
    if last is not None and last.diverged:
        return 1
    for name, path in result.output_files.items():
        print(f"{name}: {path}")
    return 0 if result.stop_reason in {"converged", "completed_steps"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
