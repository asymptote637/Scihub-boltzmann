"""Lid-driven cavity example using D2Q9 BGK dynamics."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Iterator

import numpy as np

from lbm_lab.config import SimulationConfig
from lbm_lab.lbm.d2q9 import C, OPPOSITE, equilibrium, macroscopic, stream


@dataclass(frozen=True)
class CavityState:
    step: int
    rho: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    f: np.ndarray


@dataclass(frozen=True)
class MetricRow:
    step: int
    max_speed: float
    mean_density: float
    mass_error: float


def viscosity(config: SimulationConfig) -> float:
    length = max(config.domain.nx, config.domain.ny) - 1
    return config.physics.u_lid * length / config.physics.reynolds


def relaxation_omega(config: SimulationConfig) -> float:
    nu = viscosity(config)
    tau = 3.0 * nu + 0.5
    return 1.0 / tau


def initialize(config: SimulationConfig) -> CavityState:
    shape = (config.domain.ny, config.domain.nx)
    rho = np.ones(shape, dtype=np.float64)
    ux = np.zeros(shape, dtype=np.float64)
    uy = np.zeros(shape, dtype=np.float64)
    ux[-1, :] = config.physics.u_lid
    f = equilibrium(rho, ux, uy)
    return CavityState(step=0, rho=rho, ux=ux, uy=uy, f=f)


def apply_bounce_back(f: np.ndarray) -> None:
    """Bounce back on left, right, and bottom walls."""
    left = f[:, :, 0].copy()
    right = f[:, :, -1].copy()
    bottom = f[:, 0, :].copy()

    f[:, :, 0] = left[OPPOSITE]
    f[:, :, -1] = right[OPPOSITE]
    f[:, 0, :] = bottom[OPPOSITE]


def apply_lid_boundary(f: np.ndarray, rho: np.ndarray, u_lid: float) -> None:
    """Simple moving-lid reconstruction on the top row."""
    ux = np.full_like(rho[-1, :], u_lid)
    uy = np.zeros_like(ux)
    f[:, -1, :] = equilibrium(rho[-1, :], ux, uy)


def step(state: CavityState, config: SimulationConfig, omega: float) -> CavityState:
    rho, ux, uy = macroscopic(state.f)
    ux[-1, :] = config.physics.u_lid
    uy[-1, :] = 0.0

    collided = state.f - omega * (state.f - equilibrium(rho, ux, uy))
    streamed = stream(collided)
    apply_bounce_back(streamed)

    rho, ux, uy = macroscopic(streamed)
    apply_lid_boundary(streamed, rho, config.physics.u_lid)
    rho, ux, uy = macroscopic(streamed)
    ux[-1, :] = config.physics.u_lid
    uy[-1, :] = 0.0

    return CavityState(step=state.step + 1, rho=rho, ux=ux, uy=uy, f=streamed)


def metrics(state: CavityState, initial_mass: float) -> MetricRow:
    speed = np.sqrt(state.ux**2 + state.uy**2)
    mass = float(np.sum(state.rho))
    return MetricRow(
        step=state.step,
        max_speed=float(np.max(speed)),
        mean_density=float(np.mean(state.rho)),
        mass_error=float((mass - initial_mass) / initial_mass),
    )


def run(config: SimulationConfig) -> Iterator[tuple[CavityState, MetricRow]]:
    omega = relaxation_omega(config)
    if not 0.0 < omega < 2.0:
        raise ValueError(f"Unstable relaxation omega={omega:.6g}; adjust Re, grid, or velocity.")

    state = initialize(config)
    initial_mass = float(np.sum(state.rho))
    yield state, metrics(state, initial_mass)

    for _ in range(config.time.steps):
        state = step(state, config, omega)
        if state.step % config.time.report_interval == 0 or state.step == config.time.steps:
            yield state, metrics(state, initial_mass)


def write_metrics_csv(path: str | Path, rows: list[MetricRow]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["step", "max_speed", "mean_density", "mass_error"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "step": row.step,
                    "max_speed": row.max_speed,
                    "mean_density": row.mean_density,
                    "mass_error": row.mass_error,
                }
            )

