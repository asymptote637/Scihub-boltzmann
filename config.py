"""Configuration and stability checks for the 2D D2Q9-BGK LBM app."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math
from typing import Any


CS = 1.0 / math.sqrt(3.0)
CORE_BOUNDARY_TYPES = {
    "periodic",
    "no_slip_bounce_back",
    "moving_wall_bounce_back",
    "non_equilibrium_extrapolation",
    "full_developed_outlet",
}

FUTURE_BOUNDARY_TYPES = {
    "velocity_zou_he",
    "pressure_zou_he",
    "symmetry_slip",
    "specular_reflection",
    "mixed_bounce_specular",
    "mass_corrected_outlet",
}


@dataclass
class BoundaryConfig:
    type: str = "no_slip_bounce_back"
    ux: float = 0.0
    uy: float = 0.0
    rho: float = 1.0
    rb: float = 1.0


@dataclass
class ObstacleConfig:
    type: str = "none"
    cx: float = 0.25
    cy: float = 0.5
    radius: float = 0.08
    width: float = 0.12
    height: float = 0.20
    mask_path: str = ""


@dataclass
class OutputConfig:
    output_dir: str = "results/ui_runs"
    save_npz: bool = True
    save_csv: bool = True
    save_png: bool = True
    save_animation: bool = False
    plot_interval: int = 100


@dataclass
class SolverConfig:
    case_type: str = "lid_driven_cavity"
    collision_model: str = "BGK"
    boundary_scheme_default: str = "non_equilibrium_extrapolation"
    nx: int = 256
    ny: int = 256
    rho0: float = 1.0
    u_ref: float = 0.05
    reynolds: float = 1000.0
    l_ref: float | None = None
    tol: float = 1e-6
    max_iter: int = 100000
    min_iter: int = 1000
    report_interval: int = 100
    ramp_steps: int = 1000
    body_force_x: float = 0.0
    left: BoundaryConfig = field(default_factory=BoundaryConfig)
    right: BoundaryConfig = field(default_factory=BoundaryConfig)
    bottom: BoundaryConfig = field(default_factory=BoundaryConfig)
    top: BoundaryConfig = field(default_factory=BoundaryConfig)
    obstacle: ObstacleConfig = field(default_factory=ObstacleConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def characteristic_length(self) -> float:
        if self.l_ref is not None and self.l_ref > 0:
            return float(self.l_ref)
        if self.case_type == "cylinder_flow":
            return max(3.0, 2.0 * self.obstacle.radius * min(self.nx, self.ny))
        if self.case_type in {"poiseuille_channel", "periodic_channel", "couette_flow"}:
            return float(self.ny - 1)
        return float(min(self.nx, self.ny) - 1)

    @property
    def nu_lattice(self) -> float:
        return self.u_ref * self.characteristic_length / self.reynolds

    @property
    def tau(self) -> float:
        return 3.0 * self.nu_lattice + 0.5

    @property
    def omega(self) -> float:
        return 1.0 / self.tau

    @property
    def mach(self) -> float:
        return self.u_ref / CS

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["derived"] = derived_parameters(self)
        return data


def case_preset(case_type: str, nx: int = 256, ny: int = 256) -> SolverConfig:
    cfg = SolverConfig(case_type=case_type, nx=nx, ny=ny)
    if case_type == "lid_driven_cavity":
        cfg.top = BoundaryConfig("non_equilibrium_extrapolation", ux=cfg.u_ref, uy=0.0)
        cfg.bottom = BoundaryConfig("non_equilibrium_extrapolation", ux=0.0, uy=0.0)
        cfg.left = BoundaryConfig("non_equilibrium_extrapolation", ux=0.0, uy=0.0)
        cfg.right = BoundaryConfig("non_equilibrium_extrapolation", ux=0.0, uy=0.0)
    elif case_type == "poiseuille_channel":
        cfg.left = BoundaryConfig("non_equilibrium_extrapolation", ux=cfg.u_ref, uy=0.0)
        cfg.right = BoundaryConfig("full_developed_outlet")
        cfg.top = BoundaryConfig("no_slip_bounce_back")
        cfg.bottom = BoundaryConfig("no_slip_bounce_back")
    elif case_type == "couette_flow":
        cfg.left = BoundaryConfig("periodic")
        cfg.right = BoundaryConfig("periodic")
        cfg.top = BoundaryConfig("moving_wall_bounce_back", ux=cfg.u_ref, uy=0.0)
        cfg.bottom = BoundaryConfig("no_slip_bounce_back")
    elif case_type == "periodic_channel":
        cfg.left = BoundaryConfig("periodic")
        cfg.right = BoundaryConfig("periodic")
        cfg.top = BoundaryConfig("no_slip_bounce_back")
        cfg.bottom = BoundaryConfig("no_slip_bounce_back")
        cfg.body_force_x = 1e-7
    elif case_type == "cylinder_flow":
        cfg.left = BoundaryConfig("non_equilibrium_extrapolation", ux=cfg.u_ref, uy=0.0)
        cfg.right = BoundaryConfig("full_developed_outlet")
        cfg.top = BoundaryConfig("non_equilibrium_extrapolation", ux=cfg.u_ref, uy=0.0)
        cfg.bottom = BoundaryConfig("non_equilibrium_extrapolation", ux=cfg.u_ref, uy=0.0)
        cfg.obstacle = ObstacleConfig(type="cylinder", cx=0.25, cy=0.5, radius=0.08)
    else:
        cfg.left = BoundaryConfig("no_slip_bounce_back")
        cfg.right = BoundaryConfig("no_slip_bounce_back")
        cfg.top = BoundaryConfig("no_slip_bounce_back")
        cfg.bottom = BoundaryConfig("no_slip_bounce_back")
    return cfg


def derived_parameters(cfg: SolverConfig) -> dict[str, float | str]:
    tau = cfg.tau
    ma = cfg.mach
    if tau <= 0.5 or ma > 0.3:
        level = "blocked"
    elif tau < 0.53 or ma > 0.15:
        level = "high risk"
    elif tau > 1.5:
        level = "diffusive"
    elif 0.55 <= tau <= 1.2 and ma <= 0.1:
        level = "good"
    else:
        level = "acceptable"
    return {
        "Re": cfg.reynolds,
        "Ma": ma,
        "cs": CS,
        "L_ref": cfg.characteristic_length,
        "nu_lattice": cfg.nu_lattice,
        "tau": tau,
        "omega": cfg.omega,
        "stability_level": level,
    }


def validate_config(cfg: SolverConfig) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if cfg.nx < 8 or cfg.ny < 8:
        errors.append("NX and NY must both be at least 8.")
    if cfg.reynolds <= 0:
        errors.append("Reynolds number must be positive.")
    if cfg.rho0 <= 0:
        errors.append("Initial density rho0 must be positive.")
    if cfg.u_ref < 0:
        errors.append("U_ref must be non-negative.")

    for side in ("left", "right", "bottom", "top"):
        bc = getattr(cfg, side)
        if bc.type in FUTURE_BOUNDARY_TYPES:
            errors.append(f"{side} uses {bc.type}, which is reserved for version 2.")
        elif bc.type not in CORE_BOUNDARY_TYPES:
            errors.append(f"{side} boundary type is unknown: {bc.type}")
        if not 0.0 <= bc.rb <= 1.0:
            errors.append(f"{side} rb must be in [0, 1].")

    if (cfg.left.type == "periodic") != (cfg.right.type == "periodic"):
        errors.append("left and right must both be periodic or both be non-periodic.")
    if (cfg.bottom.type == "periodic") != (cfg.top.type == "periodic"):
        errors.append("bottom and top must both be periodic or both be non-periodic.")

    if cfg.tau <= 0.5:
        errors.append("tau <= 0.5, BGK LBM unstable.")
    if cfg.tau < 0.53:
        warnings.append("tau is very close to 0.5; increase grid resolution, lower Re, or raise U_ref carefully.")
    if cfg.tau > 1.5:
        warnings.append("tau is large; the simulation may be overly diffusive and converge slowly.")
    if cfg.mach > 0.3:
        errors.append("Mach number too high for incompressible LBM.")
    elif cfg.mach > 0.15:
        warnings.append("Mach number > 0.15; compressibility error may be significant.")

    if cfg.reynolds > 3000:
        warnings.append("Re > 3000: single-relaxation BGK may be unstable; consider finer grids or future MRT/TRT.")
    if cfg.reynolds > 1000 and min(cfg.nx, cfg.ny) < 128:
        warnings.append("Grid may be too coarse for this Reynolds number.")
    if cfg.left.type == "non_equilibrium_extrapolation" and cfg.right.type == "no_slip_bounce_back":
        warnings.append("A velocity-like left boundary with a right wall can accumulate mass.")

    all_wall = all(
        getattr(cfg, side).type in {"no_slip_bounce_back", "non_equilibrium_extrapolation"}
        and abs(getattr(cfg, side).ux) < 1e-15
        and abs(getattr(cfg, side).uy) < 1e-15
        for side in ("left", "right", "bottom", "top")
    )
    if all_wall and abs(cfg.body_force_x) < 1e-20:
        warnings.append("All boundaries are stationary and no body force is set; the flow will remain nearly static.")

    return errors, warnings


def recommend_u_ref(
    *,
    reynolds: float,
    nx: int,
    ny: int,
    case_type: str,
    desired_ma_max: float = 0.1,
    desired_tau_target: float = 0.6,
) -> dict[str, float | str]:
    cfg = case_preset(case_type, nx=nx, ny=ny)
    cfg.reynolds = reynolds
    l_ref = cfg.characteristic_length
    u_max_by_ma = desired_ma_max * CS
    u_by_tau_target = (desired_tau_target - 0.5) * reynolds / (3.0 * l_ref)
    u_ref = min(u_max_by_ma, u_by_tau_target)
    n_required = (desired_tau_target - 0.5) * reynolds / (3.0 * u_max_by_ma)
    message = "Current grid can satisfy the selected low-Mach and tau target."
    if u_by_tau_target > u_max_by_ma:
        message = (
            "Current grid is too coarse for this Re under the low-Mach constraint. "
            f"Recommended L_ref >= {n_required:.1f}."
        )
    return {
        "U_ref": u_ref,
        "U_max_by_Ma": u_max_by_ma,
        "U_by_tau_target": u_by_tau_target,
        "recommended_L_ref_min": n_required,
        "message": message,
    }


def save_config(cfg: SolverConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

