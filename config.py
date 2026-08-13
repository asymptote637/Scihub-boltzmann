"""Configuration and stability checks for the 2D D2Q9 LBM app."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CS = 1.0 / math.sqrt(3.0)
SUPPORTED_COLLISION_MODELS = ("BGK", "TRT", "MRT")
SUPPORTED_MRT_PRESETS = ("lallemand_luo", "bgk_equivalent", "custom")
SUPPORTED_PARAMETER_MODES = ("reynolds", "tau", "physical", "rayleigh")
SUPPORTED_RAMP_PROFILES = ("linear", "smoothstep", "exponential", "instant")
SUPPORTED_THERMAL_MODELS = ("D2Q5_BGK",)
SUPPORTED_THERMAL_BOUNDARIES = ("isothermal", "adiabatic", "periodic", "outlet")
SUPPORTED_CASE_TYPES = (
    "lid_driven_cavity",
    "double_lid_cavity",
    "poiseuille_channel",
    "couette_flow",
    "periodic_channel",
    "open_channel_flow",
    "natural_convection_cavity",
    "rayleigh_benard_convection",
    "heated_channel_flow",
    "cylinder_flow",
    "square_cylinder_flow",
    "backward_facing_step",
    "taylor_green_vortex",
    "shear_wave_decay",
    "custom",
)
CORE_BOUNDARY_TYPES = {
    "periodic",
    "no_slip_bounce_back",
    "moving_wall_bounce_back",
    "non_equilibrium_extrapolation",
    "full_developed_outlet",
    "specular_reflection",
    "mixed_bounce_specular",
}

FUTURE_BOUNDARY_TYPES = {
    "velocity_zou_he",
    "pressure_zou_he",
    "symmetry_slip",
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
class ThermalBoundaryConfig:
    type: str = "adiabatic"
    temperature: float = 0.0


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
    trt_magic_parameter: float = 3.0 / 16.0
    mrt_preset: str = "lallemand_luo"
    mrt_s_e: float = 1.64
    mrt_s_epsilon: float = 1.54
    mrt_s_q: float = 1.90
    boundary_scheme_default: str = "non_equilibrium_extrapolation"
    parameter_mode: str = "reynolds"
    nx: int = 256
    ny: int = 256
    rho0: float = 1.0
    u_ref: float = 0.05
    reynolds: float = 1000.0
    tau_target: float = 0.6
    length_phys: float = 1.0
    velocity_phys: float = 1.0
    nu_phys: float = 1e-6
    l_ref: float | None = None
    thermal_enabled: bool = False
    thermal_model: str = "D2Q5_BGK"
    thermal_buoyancy: bool = False
    prandtl: float = 0.71
    rayleigh: float = 1e4
    temperature_hot: float = 1.0
    temperature_cold: float = 0.0
    temperature_initial: float = 0.5
    temperature_reference: float = 0.5
    gravity_x: float = 0.0
    gravity_y: float = -1.0
    thermal_tol: float = 1e-6
    tol: float = 1e-6
    max_iter: int = 100000
    min_iter: int = 1000
    report_interval: int = 100
    ramp_steps: int = 1000
    ramp_profile: str = "smoothstep"
    body_force_x: float = 0.0
    body_force_y: float = 0.0
    mass_drift_warning: float = 1e-4
    mass_drift_limit: float = 1e-3
    residual_limit: float = 1e3
    max_velocity_limit: float = 0.3
    left: BoundaryConfig = field(default_factory=BoundaryConfig)
    right: BoundaryConfig = field(default_factory=BoundaryConfig)
    bottom: BoundaryConfig = field(default_factory=BoundaryConfig)
    top: BoundaryConfig = field(default_factory=BoundaryConfig)
    thermal_left: ThermalBoundaryConfig = field(default_factory=ThermalBoundaryConfig)
    thermal_right: ThermalBoundaryConfig = field(default_factory=ThermalBoundaryConfig)
    thermal_bottom: ThermalBoundaryConfig = field(default_factory=ThermalBoundaryConfig)
    thermal_top: ThermalBoundaryConfig = field(default_factory=ThermalBoundaryConfig)
    obstacle: ObstacleConfig = field(default_factory=ObstacleConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def characteristic_length(self) -> float:
        if self.l_ref is not None and self.l_ref > 0:
            return float(self.l_ref)
        if self.case_type == "cylinder_flow":
            return max(3.0, 2.0 * self.obstacle.radius * min(self.nx, self.ny))
        if self.case_type == "square_cylinder_flow":
            width = self.obstacle.width * self.nx
            height = self.obstacle.height * self.ny
            return max(3.0, width, height)
        if self.case_type in {
            "poiseuille_channel",
            "periodic_channel",
            "open_channel_flow",
            "heated_channel_flow",
            "couette_flow",
            "backward_facing_step",
        }:
            return float(self.ny - 1)
        return float(min(self.nx, self.ny) - 1)

    @property
    def nu_lattice(self) -> float:
        if self.parameter_mode in {"tau", "rayleigh"}:
            return (self.tau_target - 0.5) / 3.0
        if self.parameter_mode == "physical":
            if self.length_phys <= 0.0 or self.velocity_phys <= 0.0:
                return math.nan
            return self.nu_phys * self.dt_phys / self.dx_phys**2
        return self.u_ref * self.characteristic_length / self.reynolds

    @property
    def tau(self) -> float:
        if self.parameter_mode in {"tau", "rayleigh"}:
            return self.tau_target
        return 3.0 * self.nu_lattice + 0.5

    @property
    def dx_phys(self) -> float:
        if self.length_phys <= 0.0:
            return math.nan
        return self.length_phys / self.characteristic_length

    @property
    def dt_phys(self) -> float:
        if self.velocity_phys <= 0.0 or not math.isfinite(self.dx_phys):
            return math.nan
        return self.u_ref * self.dx_phys / self.velocity_phys

    @property
    def thermal_diffusivity(self) -> float:
        return self.nu_lattice / self.prandtl if self.prandtl > 0.0 else math.nan

    @property
    def thermal_tau(self) -> float:
        return 0.5 + 3.0 * self.thermal_diffusivity

    @property
    def thermal_omega(self) -> float:
        return 1.0 / self.thermal_tau if self.thermal_tau != 0.0 else math.inf

    @property
    def temperature_delta(self) -> float:
        return abs(self.temperature_hot - self.temperature_cold)

    @property
    def buoyancy_per_temperature(self) -> float:
        denominator = self.temperature_delta * self.characteristic_length**3
        if denominator <= 0.0:
            return 0.0
        return self.rayleigh * self.nu_lattice * self.thermal_diffusivity / denominator

    @property
    def omega(self) -> float:
        return math.inf if self.tau == 0.0 else 1.0 / self.tau

    @property
    def trt_tau_minus(self) -> float:
        shear_offset = self.tau - 0.5
        if shear_offset <= 0.0:
            return math.inf
        return 0.5 + self.trt_magic_parameter / shear_offset

    @property
    def trt_omega_minus(self) -> float:
        return 0.0 if not math.isfinite(self.trt_tau_minus) else 1.0 / self.trt_tau_minus

    @property
    def mrt_relaxation_rates(self) -> tuple[float, ...]:
        if self.mrt_preset == "lallemand_luo":
            s_e, s_epsilon, s_q = 1.64, 1.54, 1.90
        elif self.mrt_preset == "bgk_equivalent":
            s_e = s_epsilon = s_q = self.omega
        else:
            s_e, s_epsilon, s_q = self.mrt_s_e, self.mrt_s_epsilon, self.mrt_s_q
        return (
            0.0,
            s_e,
            s_epsilon,
            0.0,
            s_q,
            0.0,
            s_q,
            self.omega,
            self.omega,
        )

    @property
    def effective_reynolds(self) -> float:
        if self.nu_lattice <= 0.0:
            return math.inf
        return self.u_ref * self.characteristic_length / self.nu_lattice

    @property
    def mach(self) -> float:
        return self.u_ref / CS

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["derived"] = derived_parameters(self)
        return data


def case_preset(
    case_type: str,
    nx: int = 256,
    ny: int = 256,
    *,
    u_ref: float | None = None,
) -> SolverConfig:
    cfg = SolverConfig(case_type=case_type, nx=nx, ny=ny)
    if u_ref is not None:
        cfg.u_ref = u_ref
    if case_type == "lid_driven_cavity":
        cfg.top = BoundaryConfig("non_equilibrium_extrapolation", ux=cfg.u_ref, uy=0.0)
        cfg.bottom = BoundaryConfig("non_equilibrium_extrapolation", ux=0.0, uy=0.0)
        cfg.left = BoundaryConfig("non_equilibrium_extrapolation", ux=0.0, uy=0.0)
        cfg.right = BoundaryConfig("non_equilibrium_extrapolation", ux=0.0, uy=0.0)
    elif case_type == "double_lid_cavity":
        cfg.top = BoundaryConfig("non_equilibrium_extrapolation", ux=cfg.u_ref, uy=0.0)
        cfg.bottom = BoundaryConfig("non_equilibrium_extrapolation", ux=-cfg.u_ref, uy=0.0)
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
    elif case_type == "open_channel_flow":
        cfg.left = BoundaryConfig("periodic")
        cfg.right = BoundaryConfig("periodic")
        cfg.top = BoundaryConfig("specular_reflection")
        cfg.bottom = BoundaryConfig("no_slip_bounce_back")
        cfg.body_force_x = 1e-7
    elif case_type == "natural_convection_cavity":
        cfg.parameter_mode = "rayleigh"
        cfg.thermal_enabled = True
        cfg.thermal_buoyancy = True
        cfg.left = BoundaryConfig("non_equilibrium_extrapolation")
        cfg.right = BoundaryConfig("non_equilibrium_extrapolation")
        cfg.top = BoundaryConfig("non_equilibrium_extrapolation")
        cfg.bottom = BoundaryConfig("non_equilibrium_extrapolation")
        cfg.thermal_left = ThermalBoundaryConfig("isothermal", cfg.temperature_hot)
        cfg.thermal_right = ThermalBoundaryConfig("isothermal", cfg.temperature_cold)
        cfg.thermal_top = ThermalBoundaryConfig("adiabatic")
        cfg.thermal_bottom = ThermalBoundaryConfig("adiabatic")
    elif case_type == "rayleigh_benard_convection":
        cfg.parameter_mode = "rayleigh"
        cfg.thermal_enabled = True
        cfg.thermal_buoyancy = True
        cfg.left = BoundaryConfig("non_equilibrium_extrapolation")
        cfg.right = BoundaryConfig("non_equilibrium_extrapolation")
        cfg.top = BoundaryConfig("non_equilibrium_extrapolation")
        cfg.bottom = BoundaryConfig("non_equilibrium_extrapolation")
        cfg.thermal_left = ThermalBoundaryConfig("adiabatic")
        cfg.thermal_right = ThermalBoundaryConfig("adiabatic")
        cfg.thermal_top = ThermalBoundaryConfig("isothermal", cfg.temperature_cold)
        cfg.thermal_bottom = ThermalBoundaryConfig("isothermal", cfg.temperature_hot)
    elif case_type == "heated_channel_flow":
        cfg.thermal_enabled = True
        cfg.thermal_buoyancy = False
        cfg.reynolds = 100.0
        cfg.left = BoundaryConfig("periodic")
        cfg.right = BoundaryConfig("periodic")
        cfg.top = BoundaryConfig("no_slip_bounce_back")
        cfg.bottom = BoundaryConfig("no_slip_bounce_back")
        cfg.body_force_x = 1e-7
        cfg.thermal_left = ThermalBoundaryConfig("periodic")
        cfg.thermal_right = ThermalBoundaryConfig("periodic")
        cfg.thermal_top = ThermalBoundaryConfig("isothermal", cfg.temperature_cold)
        cfg.thermal_bottom = ThermalBoundaryConfig("isothermal", cfg.temperature_hot)
    elif case_type == "cylinder_flow":
        cfg.left = BoundaryConfig("non_equilibrium_extrapolation", ux=cfg.u_ref, uy=0.0)
        cfg.right = BoundaryConfig("full_developed_outlet")
        cfg.top = BoundaryConfig("non_equilibrium_extrapolation", ux=cfg.u_ref, uy=0.0)
        cfg.bottom = BoundaryConfig("non_equilibrium_extrapolation", ux=cfg.u_ref, uy=0.0)
        cfg.obstacle = ObstacleConfig(type="cylinder", cx=0.25, cy=0.5, radius=0.08)
    elif case_type == "square_cylinder_flow":
        cfg.left = BoundaryConfig("non_equilibrium_extrapolation", ux=cfg.u_ref, uy=0.0)
        cfg.right = BoundaryConfig("full_developed_outlet")
        cfg.top = BoundaryConfig("non_equilibrium_extrapolation", ux=cfg.u_ref, uy=0.0)
        cfg.bottom = BoundaryConfig("non_equilibrium_extrapolation", ux=cfg.u_ref, uy=0.0)
        cfg.obstacle = ObstacleConfig(
            type="rectangle", cx=0.25, cy=0.5, width=0.10, height=0.10
        )
    elif case_type == "backward_facing_step":
        cfg.left = BoundaryConfig("non_equilibrium_extrapolation", ux=cfg.u_ref, uy=0.0)
        cfg.right = BoundaryConfig("full_developed_outlet")
        cfg.top = BoundaryConfig("no_slip_bounce_back")
        cfg.bottom = BoundaryConfig("no_slip_bounce_back")
        cfg.obstacle = ObstacleConfig(
            type="rectangle", cx=0.15, cy=0.125, width=0.30, height=0.25
        )
    elif case_type in {"taylor_green_vortex", "shear_wave_decay"}:
        cfg.left = BoundaryConfig("periodic")
        cfg.right = BoundaryConfig("periodic")
        cfg.top = BoundaryConfig("periodic")
        cfg.bottom = BoundaryConfig("periodic")
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
    derived: dict[str, float | str] = {
        "collision_model": cfg.collision_model,
        "parameter_mode": cfg.parameter_mode,
        "Re": cfg.effective_reynolds,
        "Re_input": cfg.reynolds,
        "Ma": ma,
        "cs": CS,
        "L_ref": cfg.characteristic_length,
        "nu_lattice": cfg.nu_lattice,
        "tau": tau,
        "omega": cfg.omega,
        "stability_level": level,
    }
    if cfg.parameter_mode == "physical":
        derived.update(
            {
                "dx_phys": cfg.dx_phys,
                "dt_phys": cfg.dt_phys,
                "nu_phys": cfg.nu_phys,
            }
        )
    if cfg.thermal_enabled:
        derived.update(
            {
                "thermal_model": cfg.thermal_model,
                "Pr": cfg.prandtl,
                "Ra": cfg.rayleigh,
                "alpha_lattice": cfg.thermal_diffusivity,
                "tau_thermal": cfg.thermal_tau,
                "omega_thermal": cfg.thermal_omega,
                "buoyancy_per_temperature": cfg.buoyancy_per_temperature,
            }
        )
    if cfg.collision_model == "TRT":
        derived.update(
            {
                "trt_magic_parameter": cfg.trt_magic_parameter,
                "tau_minus": cfg.trt_tau_minus,
                "omega_minus": cfg.trt_omega_minus,
            }
        )
    elif cfg.collision_model == "MRT":
        rates = cfg.mrt_relaxation_rates
        derived.update(
            {
                "mrt_preset": cfg.mrt_preset,
                "mrt_s_e": rates[1],
                "mrt_s_epsilon": rates[2],
                "mrt_s_q": rates[4],
            }
        )
    return derived


def validate_config(cfg: SolverConfig) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if cfg.nx < 8 or cfg.ny < 8:
        errors.append("NX and NY must both be at least 8.")
    if cfg.case_type not in SUPPORTED_CASE_TYPES:
        errors.append(f"Unknown case type: {cfg.case_type}.")
    if cfg.reynolds <= 0:
        errors.append("Reynolds number must be positive.")
    if cfg.rho0 <= 0:
        errors.append("Initial density rho0 must be positive.")
    if cfg.u_ref < 0:
        errors.append("U_ref must be non-negative.")
    if cfg.collision_model not in SUPPORTED_COLLISION_MODELS:
        errors.append(
            f"Unknown collision model: {cfg.collision_model}. "
            f"Choose one of {', '.join(SUPPORTED_COLLISION_MODELS)}."
        )

    if cfg.parameter_mode not in SUPPORTED_PARAMETER_MODES:
        errors.append(f"Unknown parameter mode: {cfg.parameter_mode}.")
    if cfg.parameter_mode in {"tau", "rayleigh"} and cfg.tau_target <= 0.5:
        errors.append("Direct tau must be greater than 0.5.")
    if cfg.parameter_mode == "physical":
        if cfg.length_phys <= 0.0:
            errors.append("length_phys must be positive in physical mode.")
        if cfg.velocity_phys <= 0.0:
            errors.append("velocity_phys must be positive in physical mode.")
        if cfg.nu_phys <= 0.0:
            errors.append("nu_phys must be positive in physical mode.")
        if cfg.u_ref <= 0.0:
            errors.append("U_ref lattice velocity must be positive in physical mode.")

    if not math.isfinite(cfg.tau):
        errors.append("Derived tau is not finite; check the parameter mode inputs.")

    if cfg.ramp_profile not in SUPPORTED_RAMP_PROFILES:
        errors.append(f"Unknown ramp profile: {cfg.ramp_profile}.")
    if cfg.ramp_steps < 1:
        errors.append("ramp_steps must be at least 1.")
    if not math.isfinite(cfg.body_force_x) or not math.isfinite(cfg.body_force_y):
        errors.append("Body-force components must be finite.")
    if cfg.mass_drift_warning <= 0.0 or cfg.mass_drift_limit <= 0.0:
        errors.append("Mass-drift thresholds must be positive.")
    elif cfg.mass_drift_warning >= cfg.mass_drift_limit:
        errors.append("mass_drift_warning must be smaller than mass_drift_limit.")
    if cfg.residual_limit <= 0.0:
        errors.append("residual_limit must be positive.")
    if cfg.max_velocity_limit <= 0.0:
        errors.append("max_velocity_limit must be positive.")
    elif cfg.u_ref >= cfg.max_velocity_limit:
        warnings.append("U_ref is at or above max_velocity_limit; the run may stop immediately.")

    if cfg.collision_model == "TRT":
        if cfg.trt_magic_parameter <= 0.0:
            errors.append("TRT magic parameter Lambda must be positive.")
        elif cfg.tau > 0.5 and not 0.0 < cfg.trt_omega_minus < 2.0:
            errors.append("TRT odd relaxation rate omega_minus must be in (0, 2).")
    elif cfg.collision_model == "MRT":
        if cfg.mrt_preset not in SUPPORTED_MRT_PRESETS:
            errors.append(f"Unknown MRT preset: {cfg.mrt_preset}.")
        effective_rates = cfg.mrt_relaxation_rates
        for name, rate in zip(
            ("mrt_s_e", "mrt_s_epsilon", "mrt_s_q"),
            (effective_rates[1], effective_rates[2], effective_rates[4]),
            strict=True,
        ):
            if not 0.0 < rate < 2.0:
                errors.append(f"{name} must be in (0, 2).")

    if cfg.parameter_mode == "rayleigh" and not cfg.thermal_enabled:
        errors.append("Rayleigh parameter mode requires the thermal model.")
    if cfg.thermal_enabled:
        if cfg.thermal_model not in SUPPORTED_THERMAL_MODELS:
            errors.append(f"Unknown thermal model: {cfg.thermal_model}.")
        if cfg.prandtl <= 0.0:
            errors.append("Prandtl number must be positive.")
        if cfg.rayleigh < 0.0:
            errors.append("Rayleigh number must be non-negative.")
        if cfg.thermal_tol <= 0.0:
            errors.append("thermal_tol must be positive.")
        if not math.isfinite(cfg.thermal_tau) or cfg.thermal_tau <= 0.5:
            errors.append("Thermal tau must be finite and greater than 0.5.")
        elif cfg.thermal_tau < 0.53:
            warnings.append(
                "Thermal tau is close to 0.5; increase thermal resolution or adjust Pr/tau."
            )
        elif cfg.thermal_tau > 2.0:
            warnings.append("Thermal tau is large; temperature diffusion may be excessive.")
        if cfg.thermal_buoyancy and cfg.temperature_delta <= 0.0:
            errors.append("Buoyancy requires different hot and cold temperatures.")
        if cfg.thermal_buoyancy and math.hypot(cfg.gravity_x, cfg.gravity_y) <= 0.0:
            errors.append("Buoyancy requires a non-zero gravity direction.")
        for side in ("left", "right", "bottom", "top"):
            thermal_bc = getattr(cfg, f"thermal_{side}")
            if thermal_bc.type not in SUPPORTED_THERMAL_BOUNDARIES:
                errors.append(f"Unknown {side} thermal boundary: {thermal_bc.type}.")
        if (cfg.thermal_left.type == "periodic") != (cfg.thermal_right.type == "periodic"):
            errors.append("Thermal left and right boundaries must both be periodic or non-periodic.")
        if (cfg.thermal_bottom.type == "periodic") != (cfg.thermal_top.type == "periodic"):
            errors.append("Thermal bottom and top boundaries must both be periodic or non-periodic.")

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

    if math.isfinite(cfg.tau) and cfg.tau <= 0.5:
        errors.append("tau <= 0.5 gives non-positive lattice viscosity.")
    if cfg.tau < 0.53:
        warnings.append(
            "Shear tau is very close to 0.5; increase grid resolution, lower Re, "
            "or raise U_ref carefully."
        )
    if cfg.tau > 1.5:
        warnings.append("tau is large; the simulation may be overly diffusive and converge slowly.")
    if cfg.mach > 0.3:
        errors.append("Mach number too high for incompressible LBM.")
    elif cfg.mach > 0.15:
        warnings.append("Mach number > 0.15; compressibility error may be significant.")

    if cfg.effective_reynolds > 3000 and cfg.collision_model == "BGK":
        warnings.append("Re > 3000: BGK may be unstable; use TRT/MRT and refine the grid.")
    if cfg.effective_reynolds > 1000 and min(cfg.nx, cfg.ny) < 128:
        warnings.append("Grid may be too coarse for this Reynolds number.")
    if cfg.case_type == "taylor_green_vortex" and cfg.nx != cfg.ny:
        warnings.append("Taylor-Green reference comparisons are simplest on a square grid.")
    if cfg.left.type == "non_equilibrium_extrapolation" and cfg.right.type == "no_slip_bounce_back":
        warnings.append("A velocity-like left boundary with a right wall can accumulate mass.")

    all_wall = all(
        getattr(cfg, side).type in {"no_slip_bounce_back", "non_equilibrium_extrapolation"}
        and abs(getattr(cfg, side).ux) < 1e-15
        and abs(getattr(cfg, side).uy) < 1e-15
        for side in ("left", "right", "bottom", "top")
    )
    if (
        all_wall
        and math.hypot(cfg.body_force_x, cfg.body_force_y) < 1e-20
        and not (cfg.thermal_enabled and cfg.thermal_buoyancy)
    ):
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
