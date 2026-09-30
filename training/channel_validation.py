"""Training 02: start-up Couette/force-driven Poiseuille with halfway walls.

Run ``python -m training.channel_validation`` from the project root.
Refinement keeps Re and tau fixed (diffusive scaling), with all nodes fluid
and y_j = j + 1/2, H = ny. Each run starts from rest, not the analytic answer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np

from config import CS, SUPPORTED_COLLISION_MODELS, BoundaryConfig, case_preset, validate_config
from lbm_solver import LBMSolver

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_channel_config(
    case: str, ny: int, *, nx: int = 8, reynolds: float = 3.2,
    tau: float = 0.8, collision_model: str = "BGK", rho0: float = 1.0,
):
    """Re = U H / nu; U is lid speed or analytic Poiseuille maximum speed.

    body_force_x is FORCE DENSITY F_x, not acceleration. g = F_x/rho0.
    """
    if case not in {"couette", "poiseuille"}:
        raise ValueError("case must be couette or poiseuille")
    if not isinstance(ny, int) or ny < 8 or not isinstance(nx, int) or nx < 8:
        raise ValueError("nx and ny must be integers >= 8")
    if not math.isfinite(tau) or not 0.5 < tau < 2.0:
        raise ValueError("this exercise requires 0.5 < tau < 2")
    if not math.isfinite(reynolds) or reynolds <= 0:
        raise ValueError("Re must be positive and finite")
    if not math.isfinite(rho0) or rho0 <= 0:
        raise ValueError("rho0 must be positive and finite")
    nu = (tau - 0.5) / 3.0
    u_ref = reynolds * nu / ny
    if u_ref / CS > 0.1:
        raise ValueError("this benchmark requires Ma <= 0.1; lower Re or increase ny")
    # 'custom' starts from rest; presets may initialize a developed flow.
    cfg = case_preset("custom", nx=nx, ny=ny, u_ref=u_ref)
    cfg.parameter_mode = "tau"
    cfg.tau_target = tau
    cfg.reynolds = reynolds
    cfg.rho0 = rho0
    cfg.collision_model = collision_model
    cfg.ramp_profile = "instant"
    cfg.left = BoundaryConfig("periodic")
    cfg.right = BoundaryConfig("periodic")
    cfg.bottom = BoundaryConfig("halfway_bounce_back")
    cfg.top = BoundaryConfig("halfway_bounce_back")
    if case == "couette":
        cfg.top = BoundaryConfig("moving_halfway_bounce_back", ux=u_ref)
    else:
        cfg.body_force_x = rho0 * 8.0 * nu * u_ref / ny**2
    errors, _ = validate_config(cfg)
    if errors:
        raise ValueError("; ".join(errors))
    return cfg


def analytical_profile(case: str, y_over_h: np.ndarray, u_ref: float) -> np.ndarray:
    if case == "couette":
        return u_ref * y_over_h
    if case == "poiseuille":
        return 4.0 * u_ref * y_over_h * (1.0 - y_over_h)
    raise ValueError("unknown channel case")


def run_channel_case(
    case: str, ny: int, *, nx: int = 8, reynolds: float = 3.2, tau: float = 0.8,
    collision_model: str = "BGK", rho0: float = 1.0,
    tolerance: float = 1e-9, max_diffusion_times: float = 3.0,
) -> dict:
    """Advance from rest; require three small changes after one diffusion time.

    Residual = RMS(u(t)-u(t-dt_report))/U, sampled every ~0.01 H^2/nu.
    It is a steady-state change diagnostic, not a PDE residual.
    """
    if not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("tolerance must be finite and in (0, 1)")
    if not math.isfinite(max_diffusion_times) or max_diffusion_times <= 0:
        raise ValueError("max_diffusion_times must be positive and finite")
    cfg = build_channel_config(
        case, ny, nx=nx, reynolds=reynolds, tau=tau,
        collision_model=collision_model, rho0=rho0,
    )
    diffusion_time = ny**2 / cfg.nu_lattice
    cfg.max_iter = max(1, math.ceil(max_diffusion_times * diffusion_time))
    cfg.min_iter = math.ceil(diffusion_time)
    cfg.report_interval = max(1, round(0.01 * diffusion_time))
    cfg.tol = tolerance
    solver = LBMSolver(cfg)
    y_over_h = (np.arange(ny) + 0.5) / ny
    exact = analytical_profile(case, y_over_h, cfg.u_ref)
    reference_norm = np.linalg.norm(exact)
    previous_ux = solver.ux.copy()
    previous_uy = solver.uy.copy()
    history = []
    status = "max_iter"
    small_changes = 0
    mass_drift_max = 0.0
    density_deviation_max = 0.0
    transverse_velocity_max = 0.0
    started = time.perf_counter()

    for iteration in range(cfg.max_iter + 1):
        if iteration:
            solver.step()
        if iteration % cfg.report_interval and iteration != cfg.max_iter:
            continue
        finite = np.isfinite(solver.f).all() and np.all(solver.rho > 0)
        if not finite:
            raise RuntimeError(f"{case} ny={ny}: invalid populations/density at {iteration}")
        profile = solver.ux.mean(axis=1)
        residual = float(np.sqrt(np.mean(
            (solver.ux - previous_ux)**2 + (solver.uy - previous_uy)**2
        )) / cfg.u_ref)
        mass_drift = float(abs(solver.rho.sum() / solver.mass0 - 1.0))
        mass_drift_max = max(mass_drift_max, mass_drift)
        density_deviation_max = max(
            density_deviation_max, float(np.max(np.abs(solver.rho / rho0 - 1.0)))
        )
        transverse_velocity_max = max(
            transverse_velocity_max, float(np.max(np.abs(solver.uy)))
        )
        max_speed = float(np.max(np.hypot(solver.ux, solver.uy)))
        history.append({
            "iteration": iteration, "diffusion_time": iteration / diffusion_time,
            "velocity_change_over_u": residual,
            "relative_l2_error": float(np.linalg.norm(profile - exact) / reference_norm),
            "relative_mass_drift": mass_drift, "max_speed": max_speed,
        })
        if mass_drift > cfg.mass_drift_limit or max_speed > cfg.max_velocity_limit:
            status = "diverged"
            break
        if iteration >= cfg.min_iter and residual < tolerance:
            small_changes += 1
            if small_changes >= 3:
                status = "converged"
                break
        else:
            small_changes = 0
        previous_ux = solver.ux.copy()
        previous_uy = solver.uy.copy()

    profile = solver.ux.mean(axis=1)
    error = profile - exact
    relative_l2 = float(np.linalg.norm(error) / reference_norm)
    flux = float(profile.sum())  # midpoint quadrature; dx_y = 1
    flux_exact = cfg.u_ref * ny * (0.5 if case == "couette" else 2.0 / 3.0)
    relative_flux_error = (flux - flux_exact) / flux_exact
    # A quadratic extrapolation estimates slip at the actual half-grid wall.
    lower = np.polyfit(y_over_h[:3], profile[:3] / cfg.u_ref, 2)
    upper = np.polyfit(y_over_h[-3:], profile[-3:] / cfg.u_ref, 2)
    lower_slip = float(np.polyval(lower, 0.0))
    upper_slip = float(np.polyval(upper, 1.0) - (1.0 if case == "couette" else 0.0))
    checks = {
        "reached_steady_state": status == "converged",
        "profile_l2": relative_l2 < (1e-6 if case == "couette" else 5e-3),
        "flux": abs(relative_flux_error) < (1e-6 if case == "couette" else 5e-3),
        "mass": mass_drift_max < 1e-9,
        "transverse_velocity": transverse_velocity_max / cfg.u_ref < 1e-8,
        "streamwise_invariance": float(np.max(np.abs(solver.ux - profile[:, None])))
        / cfg.u_ref < 1e-8,
    }
    summary = {
        "case": case, "collision_model": collision_model, "nx": nx, "ny": ny,
        "wall_height": ny, "rho0": rho0, "tau": tau, "nu": cfg.nu_lattice,
        "u_ref": cfg.u_ref, "reynolds": cfg.effective_reynolds, "mach_ref": cfg.mach,
        "force_density_x": cfg.body_force_x, "acceleration_x": cfg.body_force_x / rho0,
        "status": status, "iteration": solver.iteration,
        "diffusion_times_elapsed": solver.iteration / diffusion_time,
        "report_interval": cfg.report_interval, "residual": history[-1]["velocity_change_over_u"],
        "relative_l2_error": relative_l2, "max_error_over_u": float(np.max(np.abs(error))
        / cfg.u_ref), "flux_midpoint": flux, "flux_exact": flux_exact,
        "relative_flux_error": relative_flux_error,
        "lower_wall_extrapolated_slip_over_u": lower_slip,
        "upper_wall_extrapolated_slip_over_u": upper_slip,
        "mass_drift_max_sampled": mass_drift_max,
        "density_deviation_max_sampled": density_deviation_max,
        "transverse_velocity_max_sampled": transverse_velocity_max,
        "elapsed_seconds": time.perf_counter() - started,
        "checks": checks, "passed": all(checks.values()),
    }
    return {
        "summary": summary, "config": cfg.to_dict(), "history": history,
        "y_over_h": y_over_h, "velocity": profile, "analytical_velocity": exact,
        "fields": {name: value.copy() for name, value in solver.fields().items()},
    }


def observed_orders(results: list[dict]) -> list[dict]:
    orders = []
    for case in sorted({r["summary"]["case"] for r in results}):
        group = sorted(
            (r["summary"] for r in results if r["summary"]["case"] == case),
            key=lambda r: r["ny"],
        )
        for coarse, fine in pairwise(group):
            ec, ef = coarse["relative_l2_error"], fine["relative_l2_error"]
            # Couette is linear and exactly representable; its remaining error
            # is chiefly the stop tolerance/roundoff, so no spatial order claim.
            resolved = case == "poiseuille" and min(ec, ef) > 1e-7
            order = math.log(ec / ef) / math.log(fine["ny"] / coarse["ny"]) if resolved else None
            orders.append({
                "case": case, "coarse_ny": coarse["ny"], "fine_ny": fine["ny"],
                "l2_order": order,
                "note": "measured" if resolved else "not resolved above numerical floor",
            })
    return orders


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def source_hashes() -> dict[str, str]:
    return {
        p: hashlib.sha256((PROJECT_ROOT / p).read_bytes()).hexdigest()
        for p in ["config.py", "boundary_conditions.py", "lbm_solver.py",
                  "training/channel_validation.py"]
    }


def save_experiment(
    output_dir: Path, results: list[dict], arguments: dict,
    initial_source_hashes: dict[str, str] | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    orders = observed_orders(results)
    resolved_orders = [o["l2_order"] for o in orders if o["l2_order"] is not None]
    order_passed = all(1.8 <= order <= 2.2 for order in resolved_orders)
    summary = {
        "experiment": "halfway_channel_validation",
        "created_utc": datetime.now(UTC).isoformat(),
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "arguments": arguments,
        "source_sha256": initial_source_hashes or source_hashes(),
        "source_files_changed_during_run": initial_source_hashes is not None
        and initial_source_hashes != source_hashes(),
        "refinement": "fixed Re and tau, U proportional to 1/H, force density proportional to 1/H^3",
        "runs": [r["summary"] for r in results], "orders": orders,
        "grid_order_checked": bool(resolved_orders),
        "grid_order_passed": order_passed if resolved_orders else None,
        "passed": all(r["summary"]["passed"] for r in results) and order_passed,
    }
    for result in results:
        s = result["summary"]
        subdir = output_dir / f'{s["case"]}_n{s["ny"]}'
        subdir.mkdir()
        write_json(subdir / "config.json", result["config"])
        write_json(subdir / "metrics.json", s)
        write_csv(subdir / "history.csv", result["history"])
        write_csv(subdir / "profile.csv", [
            {"y_over_h": float(y), "u": float(u), "u_exact": float(ue),
             "error": float(u - ue)}
            for y, u, ue in zip(result["y_over_h"], result["velocity"],
                                result["analytical_velocity"], strict=True)
        ])
        np.savez_compressed(subdir / "fields.npz", **result["fields"])
    write_json(output_dir / "summary.json", summary)
    plot_results(output_dir, results)
    return summary


def plot_results(output_dir: Path, results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = sorted({r["summary"]["case"] for r in results})
    fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 4.5), squeeze=False)
    for axis, case in zip(axes[0], cases, strict=True):
        eta = np.linspace(0, 1, 201)
        axis.plot(analytical_profile(case, eta, 1.0), eta, "k-", label="analytical")
        for r in results:
            s = r["summary"]
            if s["case"] == case:
                axis.plot(r["velocity"] / s["u_ref"], r["y_over_h"], "o",
                          ms=3, fillstyle="none", label=f'H={s["ny"]}')
        axis.set(xlabel="u / U", ylabel="y / H", title=case.capitalize())
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "profiles.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for case in cases:
        group = [r["summary"] for r in results if r["summary"]["case"] == case]
        heights = np.array([s["ny"] for s in group])
        errors = np.array([s["relative_l2_error"] for s in group])
        axes[0].loglog(heights, errors, "o-", label=case)
        if case == "poiseuille":
            axes[0].loglog(heights, errors[0] * (heights[0] / heights)**2,
                          "k--", label="second-order reference")
    for r in results:
        s = r["summary"]
        h = r["history"][1:]
        if h:
            axes[1].semilogy([a["diffusion_time"] for a in h],
                             [max(a["velocity_change_over_u"], 1e-16) for a in h],
                             label=f'{s["case"]} H={s["ny"]}')
    axes[0].set(xlabel="channel height H (lattice cells)", ylabel="relative L2 error",
                title="Spatial refinement (Couette: stopping floor)")
    axes[1].set(xlabel="t nu / H^2", ylabel="RMS velocity change / U",
                title="Start-up convergence")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "convergence.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["both", "couette", "poiseuille"], default="both")
    parser.add_argument("--grids", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--re", type=float, default=3.2)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--rho0", type=float, default=1.0)
    parser.add_argument("--collision-model", choices=SUPPORTED_COLLISION_MODELS, default="BGK")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--max-diffusion-times", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/training/02_channels"))
    args = parser.parse_args()
    if len(set(args.grids)) != len(args.grids):
        parser.error("grid sizes must be unique")
    cases = ["couette", "poiseuille"] if args.case == "both" else [args.case]
    # Validate every requested case before starting the longer simulations.
    for case in cases:
        for ny in args.grids:
            build_channel_config(case, ny, nx=args.nx, reynolds=args.re, tau=args.tau,
                                 collision_model=args.collision_model, rho0=args.rho0)
    results = []
    initial_source_hashes = source_hashes()
    for case in cases:
        for ny in sorted(args.grids):
            print(f"Running {case}, H={ny}, {args.collision_model} ...", flush=True)
            r = run_channel_case(
                case, ny, nx=args.nx, reynolds=args.re, tau=args.tau,
                collision_model=args.collision_model, rho0=args.rho0,
                tolerance=args.tolerance, max_diffusion_times=args.max_diffusion_times,
            )
            results.append(r)
            s = r["summary"]
            print(f'  {s["status"]}: L2={s["relative_l2_error"]:.5e}, '
                  f'mass={s["mass_drift_max_sampled"]:.3e}, steps={s["iteration"]}', flush=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = args.output_dir / run_id
    summary = save_experiment(
        output_dir, results,
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        initial_source_hashes=initial_source_hashes,
    )
    print(f'Acceptance: {"PASS" if summary["passed"] else "FAIL"}', flush=True)
    print(f"Saved: {output_dir.resolve()}", flush=True)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
