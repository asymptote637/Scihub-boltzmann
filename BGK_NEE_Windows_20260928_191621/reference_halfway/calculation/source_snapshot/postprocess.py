"""Post-processing, plotting, and result export for 2D LBM fields."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import SimulationConfig, config_to_dict, domain_axis


def speed(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    return np.sqrt(ux * ux + uy * uy)


def vorticity(ux: np.ndarray, uy: np.ndarray, solid_mask: np.ndarray | None = None) -> np.ndarray:
    vort = np.zeros_like(ux)
    vort[1:-1, 1:-1] = 0.5 * (
        uy[1:-1, 2:] - uy[1:-1, :-2] - ux[2:, 1:-1] + ux[:-2, 1:-1]
    )
    vort[:, 0] = vort[:, 1]
    vort[:, -1] = vort[:, -2]
    vort[0, :] = vort[1, :]
    vort[-1, :] = vort[-2, :]
    if solid_mask is not None and np.any(solid_mask):
        invalid = solid_mask.copy()
        invalid[1:, :] |= solid_mask[:-1, :]
        invalid[:-1, :] |= solid_mask[1:, :]
        invalid[:, 1:] |= solid_mask[:, :-1]
        invalid[:, :-1] |= solid_mask[:, 1:]
        vort[invalid] = np.nan  # derivative stencil crosses solid: undefined
    return vort


def save_history_csv(history: list[dict[str, float]], output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / "residual_history.csv"
    fields = list(history[0]) if history else ["iteration", "residual", "mass_drift"]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({name: row.get(name, "") for name in fields})
    return target


def save_npz(
    rho: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    solid_mask: np.ndarray,
    config: SimulationConfig,
    transport: dict[str, float],
    output_dir: str | Path,
) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    vort = vorticity(ux, uy, solid_mask)
    spd = speed(ux, uy)
    x_offset, width = domain_axis(config, "x")
    y_offset, height = domain_axis(config, "y")
    target = path / "results.npz"
    np.savez_compressed(
        target,
        x=(np.arange(rho.shape[1]) + x_offset) / width,
        y=(np.arange(rho.shape[0]) + y_offset) / height,
        rho=rho,
        ux=ux,
        uy=uy,
        vorticity=vort,
        speed=spd,
        solid_mask=solid_mask,
        config=json.dumps(config_to_dict(config, transport), ensure_ascii=False),
    )
    return target


def _add_mask(ax: plt.Axes, solid_mask: np.ndarray) -> None:
    if np.any(solid_mask):
        masked = np.ma.masked_where(~solid_mask, solid_mask)
        ax.imshow(masked, origin="lower", cmap="gray_r", vmin=0, vmax=1, alpha=0.85, interpolation="nearest")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="0.15", label="solid")], loc="lower right")


def plot_velocity_magnitude(
    ux: np.ndarray,
    uy: np.ndarray,
    solid_mask: np.ndarray,
    output_dir: str | Path,
) -> Path:
    path = Path(output_dir) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    target = path / "velocity_magnitude.png"
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    im = ax.imshow(speed(ux, uy), origin="lower", cmap="viridis", interpolation="nearest")
    _add_mask(ax, solid_mask)
    ax.set_title("Velocity magnitude")
    ax.set_xlabel("x (lattice units)")
    ax.set_ylabel("y (lattice units)")
    fig.colorbar(im, ax=ax, label="speed (lattice units)")
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def plot_vorticity(ux: np.ndarray, uy: np.ndarray, solid_mask: np.ndarray, output_dir: str | Path) -> Path:
    path = Path(output_dir) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    target = path / "vorticity.png"
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    vort = vorticity(ux, uy, solid_mask)
    finite = vort[np.isfinite(vort)]
    vmax = max(float(np.max(np.abs(finite))) if finite.size else 0.0, 1.0e-12)
    im = ax.imshow(vort, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    _add_mask(ax, solid_mask)
    ax.set_title("Vorticity")
    ax.set_xlabel("x (lattice units)")
    ax.set_ylabel("y (lattice units)")
    fig.colorbar(im, ax=ax, label="vorticity (1 / lattice time)")
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def plot_streamlines(ux: np.ndarray, uy: np.ndarray, solid_mask: np.ndarray, output_dir: str | Path) -> Path:
    path = Path(output_dir) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    target = path / "streamlines.png"
    ny, nx = ux.shape
    x = np.arange(nx)
    y = np.arange(ny)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    stream = ax.streamplot(x, y, np.ma.masked_where(solid_mask, ux),
        np.ma.masked_where(solid_mask, uy), density=1.3,
        color=speed(ux, uy), cmap="viridis", linewidth=1.0)
    fig.colorbar(stream.lines, ax=ax, label="speed (lattice units)")
    ax.set_aspect("equal")
    _add_mask(ax, solid_mask)
    ax.set_title("Streamlines")
    ax.set_xlabel("x (lattice units)")
    ax.set_ylabel("y (lattice units)")
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def plot_quiver(ux: np.ndarray, uy: np.ndarray, solid_mask: np.ndarray, output_dir: str | Path) -> Path:
    path = Path(output_dir) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    target = path / "quiver.png"
    ny, nx = ux.shape
    step = max(1, min(nx, ny) // 32)
    yy, xx = np.mgrid[0:ny:step, 0:nx:step]
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    q = ax.quiver(xx, yy, np.ma.masked_where(solid_mask, ux)[::step, ::step],
        np.ma.masked_where(solid_mask, uy)[::step, ::step], speed(ux[::step, ::step], uy[::step, ::step]))
    fig.colorbar(q, ax=ax, label="speed (lattice units)")
    ax.set_aspect("equal")
    _add_mask(ax, solid_mask)
    ax.set_title("Velocity vectors")
    ax.set_xlabel("x (lattice units)")
    ax.set_ylabel("y (lattice units)")
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def plot_residual_history(history: list[dict[str, float]], output_dir: str | Path) -> Path:
    path = Path(output_dir) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    target = path / "residual_history.png"
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    if history:
        it = [row["iteration"] for row in history]
        residual = [row["residual"] for row in history]
        mass = [row["mass_drift"] for row in history]
        ax.semilogy(it, np.ma.masked_less_equal(residual, 0), label="residual")
        ax.semilogy(it, np.ma.masked_less_equal(mass, 0), label="mass drift")
        ax.legend()
    ax.set_title("Residual history (zeros omitted on log axes)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target



def save_centerlines(ux, uy, solid_mask, config, output_dir) -> Path:
    """Interpolate at the physical midplanes, not an arbitrary central index."""
    xo, width = domain_axis(config, "x")
    yo, height = domain_axis(config, "y")
    x = (np.arange(ux.shape[1]) + xo) / width
    y = (np.arange(ux.shape[0]) + yo) / height
    field_u = np.where(solid_mask, np.nan, ux)
    field_v = np.where(solid_mask, np.nan, uy)
    u_vertical = np.array([np.interp(0.5, x, row) for row in field_u])
    v_horizontal = np.array([np.interp(0.5, y, col) for col in field_v.T])
    path = Path(output_dir) / "centerlines.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["line", "coordinate_normalized", "velocity_lattice", "velocity_over_Uref"])
        for name, coords, vals in (("x_mid_ux", y, u_vertical), ("y_mid_uy", x, v_horizontal)):
            for coord, value in zip(coords, vals):
                w.writerow([name, coord, value, value / config.flow.U_ref if config.flow.U_ref else ""])
    return path

def save_all_outputs(
    rho: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    solid_mask: np.ndarray,
    history: list[dict[str, float]],
    config: SimulationConfig,
    transport: dict[str, float],
) -> dict[str, Path]:
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    if config.output.save_npz:
        files["npz"] = save_npz(rho, ux, uy, solid_mask, config, transport, output_dir)
    if config.output.save_csv:
        files["history_csv"] = save_history_csv(history, output_dir)
        files["centerlines"] = save_centerlines(ux, uy, solid_mask, config, output_dir)
    if config.output.save_png:
        files["velocity"] = plot_velocity_magnitude(ux, uy, solid_mask, output_dir)
        files["streamlines"] = plot_streamlines(ux, uy, solid_mask, output_dir)
        files["vorticity"] = plot_vorticity(ux, uy, solid_mask, output_dir)
        files["quiver"] = plot_quiver(ux, uy, solid_mask, output_dir)
        files["residual"] = plot_residual_history(history, output_dir)
    return files


def latest_status(history: list[dict[str, float]]) -> dict[str, Any]:
    if not history:
        return {
            "iteration": 0,
            "residual": float("nan"),
            "q": float("nan"),
            "q_avg": float("nan"),
            "mass_drift": 0.0,
            "max_velocity": 0.0,
        }
    return history[-1]
