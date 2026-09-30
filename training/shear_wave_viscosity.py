"""Quantitatively validate D2Q9 isotropy and the BGK viscosity relation.

Run from the repository root with::

    python -m training.shear_wave_viscosity

The experiment uses a periodic transverse shear wave

    u_x(y, t) = U_0 sin(k y) exp(-nu k^2 t)

so that viscosity can be measured without introducing wall-boundary errors.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from boundary_conditions import E, W, equilibrium
from config import CS, SUPPORTED_COLLISION_MODELS, case_preset
from lbm_solver import LBMSolver


@dataclass(frozen=True)
class Sample:
    tau: float
    time: int
    amplitude: float
    theory_amplitude: float
    total_mass: float
    density_deviation_max: float


@dataclass(frozen=True)
class FitResult:
    collision_model: str
    tau: float
    omega: float
    nu_theory: float
    nu_fitted: float
    relative_viscosity_error: float
    fitted_decay_rate: float
    theoretical_decay_rate: float
    log_fit_rmse: float
    amplitude_initial: float
    amplitude_final: float
    relative_mass_drift_max: float
    density_deviation_max: float
    mach: float
    nx: int
    ny: int
    steps: int
    fit_start: int


def d2q9_isotropy_errors() -> dict[str, float]:
    """Return maximum absolute residuals of the D2Q9 isotropy identities."""
    identity = np.eye(2)
    cs2 = CS**2
    zeroth = float(abs(np.sum(W) - 1.0))
    first = np.einsum("q,qi->i", W, E)
    second = np.einsum("q,qi,qj->ij", W, E, E)
    fourth = np.einsum("q,qi,qj,qk,ql->ijkl", W, E, E, E, E)
    fourth_target = np.zeros((2, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for ell in range(2):
                    fourth_target[i, j, k, ell] = cs2**2 * (
                        identity[i, j] * identity[k, ell]
                        + identity[i, k] * identity[j, ell]
                        + identity[i, ell] * identity[j, k]
                    )
    return {
        "zeroth_order": zeroth,
        "first_order": float(np.max(np.abs(first))),
        "second_order": float(np.max(np.abs(second - cs2 * identity))),
        "third_order": float(
            np.max(np.abs(np.einsum("q,qi,qj,qk->ijk", W, E, E, E)))
        ),
        "fourth_order": float(np.max(np.abs(fourth - fourth_target))),
    }


def equilibrium_moment_errors() -> dict[str, float]:
    """Check the density, momentum, and momentum-flux moments of f_eq."""
    rho = np.array([[0.97, 1.00], [1.03, 1.01]])
    ux = np.array([[0.01, -0.02], [0.03, -0.01]])
    uy = np.array([[-0.02, 0.01], [0.00, 0.02]])
    feq = equilibrium(rho, ux, uy)
    density = np.sum(feq, axis=-1)
    momentum = np.einsum("...q,qi->...i", feq, E)
    flux = np.einsum("...q,qi,qj->...ij", feq, E, E)
    velocity = np.stack((ux, uy), axis=-1)
    flux_target = rho[..., None, None] * (
        CS**2 * np.eye(2) + velocity[..., :, None] * velocity[..., None, :]
    )
    return {
        "density": float(np.max(np.abs(density - rho))),
        "momentum": float(np.max(np.abs(momentum - rho[..., None] * velocity))),
        "momentum_flux": float(np.max(np.abs(flux - flux_target))),
    }


def project_fundamental_amplitude(ux: np.ndarray) -> float:
    """Project the x velocity onto the fundamental sine mode in y."""
    ny = ux.shape[0]
    y_phase = 2.0 * np.pi * np.arange(ny) / ny
    mode = np.sin(y_phase)
    row_mean = np.mean(ux, axis=1)
    return float(np.dot(row_mean, mode) / np.dot(mode, mode))


def run_viscosity_case(
    *,
    tau: float,
    nx: int = 64,
    ny: int = 64,
    amplitude: float = 0.02,
    steps: int = 400,
    sample_every: int = 5,
    fit_start: int = 20,
    collision_model: str = "BGK",
) -> tuple[FitResult, list[Sample]]:
    """Run one shear-wave decay case and infer viscosity from its decay rate."""
    if not 0.5 < tau < 2.0:
        raise ValueError("tau must satisfy 0.5 < tau < 2.0 for this exercise")
    if nx < 8 or ny < 16:
        raise ValueError("use nx >= 8 and ny >= 16")
    if steps <= fit_start:
        raise ValueError("steps must be larger than fit_start")
    if sample_every < 1:
        raise ValueError("sample_every must be positive")
    if collision_model not in SUPPORTED_COLLISION_MODELS:
        raise ValueError(f"unsupported collision model: {collision_model}")

    cfg = case_preset("shear_wave_decay", nx=nx, ny=ny, u_ref=amplitude)
    cfg.parameter_mode = "tau"
    cfg.tau_target = tau
    cfg.collision_model = collision_model
    solver = LBMSolver(cfg)

    wave_number = 2.0 * np.pi / ny
    nu_theory = (tau - 0.5) / 3.0
    initial_amplitude = project_fundamental_amplitude(solver.ux)
    mass_initial = float(np.sum(solver.rho))
    samples: list[Sample] = []

    for time in range(steps + 1):
        if time % sample_every == 0 or time == steps:
            measured = project_fundamental_amplitude(solver.ux)
            samples.append(
                Sample(
                    tau=tau,
                    time=time,
                    amplitude=measured,
                    theory_amplitude=initial_amplitude
                    * np.exp(-nu_theory * wave_number**2 * time),
                    total_mass=float(np.sum(solver.rho)),
                    density_deviation_max=float(np.max(np.abs(solver.rho - cfg.rho0))),
                )
            )
        if time < steps:
            solver.step()

    fit_samples = [sample for sample in samples if sample.time >= fit_start]
    fit_times = np.asarray([sample.time for sample in fit_samples], dtype=float)
    fit_amplitudes = np.asarray([sample.amplitude for sample in fit_samples])
    if np.any(fit_amplitudes <= 0.0):
        raise RuntimeError("the fitted shear-wave amplitude changed sign or vanished")
    slope, intercept = np.polyfit(fit_times, np.log(fit_amplitudes), 1)
    fitted_log_amplitudes = intercept + slope * fit_times
    nu_fitted = -float(slope) / wave_number**2
    relative_mass_drift = max(
        abs(sample.total_mass - mass_initial) / mass_initial for sample in samples
    )
    result = FitResult(
        collision_model=collision_model,
        tau=tau,
        omega=1.0 / tau,
        nu_theory=nu_theory,
        nu_fitted=nu_fitted,
        relative_viscosity_error=(nu_fitted - nu_theory) / nu_theory,
        fitted_decay_rate=-float(slope),
        theoretical_decay_rate=nu_theory * wave_number**2,
        log_fit_rmse=float(
            np.sqrt(np.mean((np.log(fit_amplitudes) - fitted_log_amplitudes) ** 2))
        ),
        amplitude_initial=initial_amplitude,
        amplitude_final=samples[-1].amplitude,
        relative_mass_drift_max=relative_mass_drift,
        density_deviation_max=max(sample.density_deviation_max for sample in samples),
        mach=amplitude / CS,
        nx=nx,
        ny=ny,
        steps=steps,
        fit_start=fit_start,
    )
    return result, samples


def write_outputs(
    output_dir: Path,
    results: list[FitResult],
    all_samples: list[Sample],
    isotropy_errors: dict[str, float],
    moment_errors: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "amplitude_decay.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(all_samples[0])))
        writer.writeheader()
        writer.writerows(asdict(sample) for sample in all_samples)

    summary = {
        "experiment": "periodic_shear_wave_viscosity",
        "isotropy_errors": isotropy_errors,
        "equilibrium_moment_errors": moment_errors,
        "fits": [asdict(result) for result in results],
    }
    (output_dir / "fit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for result in results:
        samples = [sample for sample in all_samples if sample.tau == result.tau]
        times = np.asarray([sample.time for sample in samples])
        measured = np.asarray([sample.amplitude for sample in samples])
        theory = np.asarray([sample.theory_amplitude for sample in samples])
        axis.semilogy(times, measured, "o", ms=3, label=fr"measured $\tau={result.tau:g}$")
        axis.semilogy(times, theory, "-", lw=1.5, label=fr"theory $\tau={result.tau:g}$")
    axis.set_xlabel("lattice time step")
    axis.set_ylabel("fundamental shear-wave amplitude")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "amplitude_decay.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training 01: validate D2Q9 moments and measure viscosity from shear-wave decay."
    )
    parser.add_argument("--tau", type=float, nargs="+", default=[0.6, 0.8, 1.0])
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--amplitude", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--fit-start", type=int, default=20)
    parser.add_argument(
        "--collision-model", choices=SUPPORTED_COLLISION_MODELS, default="BGK"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/training/01_shear_wave")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    isotropy_errors = d2q9_isotropy_errors()
    moment_errors = equilibrium_moment_errors()
    results: list[FitResult] = []
    all_samples: list[Sample] = []
    for tau in args.tau:
        result, samples = run_viscosity_case(
            tau=tau,
            nx=args.nx,
            ny=args.ny,
            amplitude=args.amplitude,
            steps=args.steps,
            sample_every=args.sample_every,
            fit_start=args.fit_start,
            collision_model=args.collision_model,
        )
        results.append(result)
        all_samples.extend(samples)

    write_outputs(args.output_dir, results, all_samples, isotropy_errors, moment_errors)
    print("D2Q9 isotropy residuals:", json.dumps(isotropy_errors))
    print("Equilibrium-moment residuals:", json.dumps(moment_errors))
    print("tau       nu_theory     nu_fitted      rel_error      max_mass_drift")
    for result in results:
        print(
            f"{result.tau:4.2f}  {result.nu_theory:13.8f}  {result.nu_fitted:13.8f}  "
            f"{result.relative_viscosity_error:12.3e}  "
            f"{result.relative_mass_drift_max:14.3e}"
        )
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

