"""Post-processing, plotting, and result export for LBM simulations."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

Path("logs/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("logs") / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import SolverConfig, save_config
from lbm_solver import LBMSolver, Report


def speed(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    return np.sqrt(ux**2 + uy**2)


def vorticity(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    vort = np.zeros_like(ux)
    vort[1:-1, 1:-1] = (
        (uy[1:-1, 2:] - uy[1:-1, :-2]) * 0.5
        - (ux[2:, 1:-1] - ux[:-2, 1:-1]) * 0.5
    )
    return vort


def normalized_grid(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    return np.meshgrid(x, y)


def write_history_csv(path: str | Path, history: list[Report]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iteration",
        "residual",
        "q",
        "q_avg",
        "mass_drift",
        "max_velocity",
        "tau",
        "omega",
        "Ma",
        "status",
        "temperature_residual",
        "temperature_min",
        "temperature_max",
        "nusselt_average",
    ]
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for report in history:
            writer.writerow(report.as_dict())


def save_figures(
    output_dir: str | Path,
    rho: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    solid_mask: np.ndarray,
    history: list[Report],
    temperature: np.ndarray | None = None,
) -> None:
    out = Path(output_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    spd = speed(ux, uy)
    vort = vorticity(ux, uy)
    masked_speed = np.ma.masked_where(solid_mask, spd)
    masked_vort = np.ma.masked_where(solid_mask, vort)
    x, y = normalized_grid(*ux.shape)

    plt.figure(figsize=(7, 5))
    plt.imshow(masked_speed, origin="lower", cmap="viridis")
    plt.colorbar(label="|u|")
    plt.tight_layout()
    plt.savefig(out / "velocity_magnitude.png", dpi=180)
    plt.close()

    if temperature is not None:
        plt.figure(figsize=(7, 5))
        plt.imshow(np.ma.masked_where(solid_mask, temperature), origin="lower", cmap="inferno")
        plt.colorbar(label="temperature")
        plt.tight_layout()
        plt.savefig(out / "temperature.png", dpi=180)
        plt.close()

    plt.figure(figsize=(7, 5))
    plt.imshow(masked_vort, origin="lower", cmap="coolwarm")
    plt.colorbar(label="vorticity")
    plt.tight_layout()
    plt.savefig(out / "vorticity.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.streamplot(x, y, np.ma.masked_where(solid_mask, ux), np.ma.masked_where(solid_mask, uy), density=1.4)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out / "streamlines.png", dpi=180)
    plt.close()

    stride = max(1, min(ux.shape) // 32)
    plt.figure(figsize=(7, 5))
    plt.quiver(x[::stride, ::stride], y[::stride, ::stride], ux[::stride, ::stride], uy[::stride, ::stride])
    plt.tight_layout()
    plt.savefig(out / "quiver.png", dpi=180)
    plt.close()

    if history:
        plt.figure(figsize=(7, 5))
        it = [r.iteration for r in history]
        residual = [r.residual for r in history]
        plt.semilogy(it, residual)
        plt.xlabel("iteration")
        plt.ylabel("residual")
        plt.tight_layout()
        plt.savefig(out / "residual_history.png", dpi=180)
        plt.close()


def save_results(solver: LBMSolver, cfg: SolverConfig, output_root: str | Path | None = None) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = Path(output_root or cfg.output.output_dir)
    run_dir = root / f"{cfg.case_type}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    fields = solver.fields()
    vort = vorticity(fields["ux"], fields["uy"])
    fields["vorticity"] = vort

    save_config(cfg, run_dir / "config.json")
    (run_dir / "summary.json").write_text(json.dumps(solver.summary(), indent=2), encoding="utf-8")

    if cfg.output.save_csv:
        write_history_csv(run_dir / "residual_history.csv", solver.residual_history)
    if cfg.output.save_npz:
        archive = {
            "rho": fields["rho"],
            "ux": fields["ux"],
            "uy": fields["uy"],
            "vorticity": vort,
            "speed": fields["speed"],
            "solid_mask": fields["solid_mask"],
            "config": json.dumps(cfg.to_dict()),
        }
        if "temperature" in fields:
            archive["temperature"] = fields["temperature"]
        np.savez_compressed(run_dir / "results.npz", **archive)
    if cfg.output.save_png:
        save_figures(
            run_dir,
            fields["rho"],
            fields["ux"],
            fields["uy"],
            fields["solid_mask"],
            solver.residual_history,
            fields.get("temperature"),
        )
    return run_dir
