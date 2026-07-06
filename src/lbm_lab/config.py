"""Configuration loading for reproducible simulation runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class DomainConfig:
    nx: int
    ny: int


@dataclass(frozen=True)
class PhysicsConfig:
    reynolds: float
    u_lid: float


@dataclass(frozen=True)
class TimeConfig:
    steps: int
    report_interval: int
    snapshot_interval: int = 0


@dataclass(frozen=True)
class OutputConfig:
    database: Path
    results_dir: Path
    save_final_fields: bool = True


@dataclass(frozen=True)
class SimulationConfig:
    name: str
    kind: str
    seed: int
    domain: DomainConfig
    physics: PhysicsConfig
    time: TimeConfig
    output: OutputConfig
    raw: dict[str, Any]


def load_config(path: str | Path) -> SimulationConfig:
    config_path = Path(path)
    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)

    sim = raw.get("simulation", {})
    domain = raw.get("domain", {})
    physics = raw.get("physics", {})
    time = raw.get("time", {})
    output = raw.get("output", {})

    return SimulationConfig(
        name=str(sim.get("name", config_path.stem)),
        kind=str(sim.get("kind", "lid_driven_cavity")),
        seed=int(sim.get("seed", 0)),
        domain=DomainConfig(nx=int(domain["nx"]), ny=int(domain["ny"])),
        physics=PhysicsConfig(
            reynolds=float(physics["reynolds"]),
            u_lid=float(physics["u_lid"]),
        ),
        time=TimeConfig(
            steps=int(time["steps"]),
            report_interval=int(time.get("report_interval", 100)),
            snapshot_interval=int(time.get("snapshot_interval", 0)),
        ),
        output=OutputConfig(
            database=Path(output.get("database", "database/lbm_runs.sqlite")),
            results_dir=Path(output.get("results_dir", "results")),
            save_final_fields=bool(output.get("save_final_fields", True)),
        ),
        raw=raw,
    )

