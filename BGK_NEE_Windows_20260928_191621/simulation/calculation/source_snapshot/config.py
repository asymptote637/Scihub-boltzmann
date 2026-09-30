"""Configuration objects and parameter diagnostics for the LBM solver."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import math


CS = 1.0 / math.sqrt(3.0)
CS2 = 1.0 / 3.0


BOUNDARY_TYPES = (
    "periodic",
    "halfway_bounce_back",
    "halfway_moving_wall",
    "no_slip_bounce_back",
    "moving_wall_bounce_back",
    "non_equilibrium_extrapolation",
    "full_developed_outlet",
    "velocity_zou_he",
    "pressure_zou_he",
)


CASE_TYPES = (
    "lid_driven_cavity",
    "poiseuille_channel",
    "couette_flow",
    "periodic_channel",
    "force_poiseuille",
    "cylinder_flow",
    "custom",
)


@dataclass
class BoundaryConfig:
    """Boundary type and scalar parameters for one rectangular-domain side."""

    type: str = "no_slip_bounce_back"
    ux: float = 0.0
    uy: float = 0.0
    rho: float = 1.0
    rb: float = 1.0


@dataclass
class GridConfig:
    NX: int = 256
    NY: int = 256
    L_ref: float | None = None
    obstacle_type: str = "none"
    obstacle_x: float = 0.33
    obstacle_y: float = 0.5
    obstacle_radius: float = 0.08
    obstacle_width: float = 0.12
    obstacle_height: float = 0.20


@dataclass
class FlowConfig:
    rho0: float = 1.0
    U_ref: float = 0.05
    Re: float = 1000.0
    body_force_x: float = 0.0
    physical_mode: bool = False
    L_phys: float = 1.0
    U_phys: float = 1.0
    nu_phys: float = 1.0e-6
    # body_force_* are accelerations in lattice units, NOT force densities.
    body_force_y: float = 0.0
    nu_lattice: float | None = None
    initial_velocity: str = "rest"


@dataclass
class ConvergenceConfig:
    tol: float = 1.0e-6
    max_iter: int = 100000
    min_iter: int = 1000
    report_interval: int = 100
    ramp_steps: int = 1000
    consecutive_reports: int = 3
    steady: bool = True
    mass_tolerance: float = 1.0e-4
    mass_criterion: bool = True
    max_mach: float = 0.3


@dataclass
class OutputConfig:
    output_dir: str = "outputs"
    save_npz: bool = True
    save_csv: bool = True
    save_png: bool = True
    save_animation: bool = False
    plot_interval: int = 500


@dataclass
class SimulationConfig:
    case_type: str = "lid_driven_cavity"
    collision_model: str = "BGK"
    boundary_scheme_default: str = "halfway_bounce_back"
    grid: GridConfig = field(default_factory=GridConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    boundaries: dict[str, BoundaryConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.boundaries:
            self.boundaries = default_boundaries_for_case(
                self.case_type, self.flow.U_ref, self.boundary_scheme_default
            )


def default_boundaries_for_case(
    case_type: str, u_ref: float = 0.05, scheme: str = "halfway_bounce_back"
) -> dict[str, BoundaryConfig]:
    wall = BoundaryConfig("halfway_bounce_back")
    moving = BoundaryConfig("halfway_moving_wall", ux=u_ref)
    if scheme == "no_slip_bounce_back":
        wall = BoundaryConfig("no_slip_bounce_back")
        moving = BoundaryConfig("moving_wall_bounce_back", ux=u_ref)
    ne_wall = BoundaryConfig("non_equilibrium_extrapolation")
    ne_lid = BoundaryConfig("non_equilibrium_extrapolation", ux=u_ref)

    if case_type == "lid_driven_cavity":
        if scheme == "non_equilibrium_extrapolation":
            return {
                "left": BoundaryConfig(**asdict(ne_wall)),
                "right": BoundaryConfig(**asdict(ne_wall)),
                "bottom": BoundaryConfig(**asdict(ne_wall)),
                "top": BoundaryConfig(**asdict(ne_lid)),
            }
        return {
            "left": BoundaryConfig(**asdict(wall)),
            "right": BoundaryConfig(**asdict(wall)),
            "bottom": BoundaryConfig(**asdict(wall)),
            "top": BoundaryConfig(**asdict(moving)),
        }
    if case_type == "poiseuille_channel":
        return {
            "left": BoundaryConfig("non_equilibrium_extrapolation", ux=u_ref),
            "right": BoundaryConfig("full_developed_outlet"),
            "bottom": BoundaryConfig(**asdict(wall)),
            "top": BoundaryConfig(**asdict(wall)),
        }
    if case_type in {"couette_flow", "periodic_channel", "force_poiseuille"}:
        return {
            "left": BoundaryConfig("periodic"),
            "right": BoundaryConfig("periodic"),
            "bottom": BoundaryConfig(**asdict(wall)),
            "top": BoundaryConfig(**asdict(moving if case_type == "couette_flow" else wall)),
        }
    if case_type == "cylinder_flow":
        return {
            "left": BoundaryConfig("non_equilibrium_extrapolation", ux=u_ref),
            "right": BoundaryConfig("full_developed_outlet"),
            "bottom": BoundaryConfig(**asdict(wall)),
            "top": BoundaryConfig(**asdict(wall)),
        }
    return {
        "left": BoundaryConfig(**asdict(wall)),
        "right": BoundaryConfig(**asdict(wall)),
        "bottom": BoundaryConfig(**asdict(wall)),
        "top": BoundaryConfig(**asdict(wall)),
    }


def characteristic_length(config: SimulationConfig) -> float:
    grid = config.grid
    if grid.L_ref is not None and grid.L_ref > 0:
        return float(grid.L_ref)
    if config.case_type == "cylinder_flow":
        return float(2 * max(1, round(grid.obstacle_radius * min(grid.NX, grid.NY))))
    if config.case_type in {"poiseuille_channel", "periodic_channel", "force_poiseuille", "couette_flow"}:
        return domain_axis(config, "y")[1]
    return min(domain_axis(config, "x")[1], domain_axis(config, "y")[1])


def domain_axis(config: SimulationConfig, axis: str) -> tuple[float, float]:
    """Return (first node distance from lower wall, physical lattice span).

    Halfway walls lie half a lattice spacing outside the array. On-node BCs
    lie on the first/last nodes; a periodic axis has N cells.
    """
    low, high, n = (("left", "right", config.grid.NX) if axis == "x"
                    else ("bottom", "top", config.grid.NY))
    a = config.boundaries.get(low, BoundaryConfig()).type
    b = config.boundaries.get(high, BoundaryConfig()).type
    if a == b == "periodic":
        return 0.0, float(n)
    start = 0.5 if a.startswith("halfway_") else 0.0
    end = 0.5 if b.startswith("halfway_") else 0.0
    return start, float(n - 1) + start + end


def lattice_transport(config: SimulationConfig) -> dict[str, float]:
    flow = config.flow
    grid = config.grid
    l_ref = characteristic_length(config)

    if flow.physical_mode:
        dx_phys = flow.L_phys / l_ref
        dt_phys = flow.U_ref * dx_phys / flow.U_phys
        nu = flow.nu_phys * dt_phys / (dx_phys * dx_phys)
        re = flow.U_phys * flow.L_phys / flow.nu_phys
    else:
        nu = flow.nu_lattice if flow.nu_lattice is not None else flow.U_ref * l_ref / flow.Re
        re = flow.U_ref * l_ref / nu if nu > 0 else float("nan")

    tau = 3.0 * nu + 0.5
    omega = 1.0 / tau
    ma = flow.U_ref / CS
    return {
        "L_ref": l_ref,
        "Re": re,
        "nu_lattice": nu,
        "tau": tau,
        "omega": omega,
        "Ma": ma,
        "cs": CS,
    }


def recommend_parameters(
    case_type: str,
    nx: int,
    ny: int,
    re: float,
    desired_ma_max: float = 0.1,
    desired_tau_min: float = 0.55,
    desired_tau_target: float = 0.6,
    l_ref: float | None = None,
) -> dict[str, float | str]:
    if l_ref is None:
        l_ref = characteristic_length(SimulationConfig(case_type=case_type, grid=GridConfig(NX=nx, NY=ny)))
    u_max_by_ma = desired_ma_max * CS
    u_by_tau_target = (desired_tau_target - 0.5) * re / (3.0 * l_ref)
    u_ref = min(u_max_by_ma, u_by_tau_target)
    nu = u_ref * l_ref / re
    tau = 3.0 * nu + 0.5
    n_required = (desired_tau_target - 0.5) * re / (3.0 * u_max_by_ma)
    status = "ok"
    if tau < desired_tau_min:
        status = "grid_too_coarse_for_tau_min"
    if u_by_tau_target > u_max_by_ma:
        status = "grid_too_coarse_under_low_mach"
    return {
        "U_ref": u_ref,
        "nu_lattice": nu,
        "tau": tau,
        "omega": 1.0 / tau,
        "Ma": u_ref / CS,
        "L_ref": l_ref,
        "N_required": n_required,
        "status": status,
    }


def validate_config(config: SimulationConfig) -> tuple[list[str], list[str], dict[str, float]]:
    """Validate before dividing or allocating; return actionable errors for UI/CLI."""
    errors: list[str] = []
    warnings: list[str] = []
    g, f, c = config.grid, config.flow, config.convergence
    if type(c.mass_criterion) is not bool:
        errors.append("mass_criterion must be a boolean.")
    if config.case_type not in CASE_TYPES:
        errors.append(f"Unsupported case_type: {config.case_type}")
    if config.collision_model != "BGK":
        errors.append("Only BGK collision is implemented.")
    for name, value, minimum in (("NX", g.NX, 8), ("NY", g.NY, 8),
            ("max_iter", c.max_iter, 1), ("min_iter", c.min_iter, 0),
            ("report_interval", c.report_interval, 1), ("ramp_steps", c.ramp_steps, 0),
            ("consecutive_reports", c.consecutive_reports, 1)):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            errors.append(f"{name} must be an integer >= {minimum}.")
    for name, value in (("rho0", f.rho0), ("Re", f.Re), ("tol", c.tol),
            ("mass_tolerance", c.mass_tolerance), ("max_mach", c.max_mach)):
        if not math.isfinite(value) or value <= 0:
            errors.append(f"{name} must be finite and positive.")
    for name, value in (("U_ref", f.U_ref), ("body_force_x", f.body_force_x),
            ("body_force_y", f.body_force_y)):
        if not math.isfinite(value): errors.append(f"{name} must be finite.")
    if f.U_ref < 0: errors.append("U_ref must be nonnegative.")
    for name, value in (("L_ref", g.L_ref), ("nu_lattice", f.nu_lattice)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            errors.append(f"{name} must be finite and positive when specified.")
    if f.U_ref == 0 and f.nu_lattice is None:
        errors.append("Zero U_ref requires explicit nu_lattice > 0.")
    if f.initial_velocity not in {"rest", "uniform", "couette"}:
        errors.append("initial_velocity must be rest, uniform, or couette.")
    if f.physical_mode:
        for name in ("L_phys", "U_phys", "nu_phys"):
            v = getattr(f, name)
            if not math.isfinite(v) or v <= 0: errors.append(f"{name} must be finite and positive.")
        if f.U_ref <= 0: errors.append("physical_mode requires positive U_ref.")
        if f.nu_lattice is not None: errors.append("Choose physical_mode OR nu_lattice.")
    if g.obstacle_type not in {"none", "circle", "rectangle"}:
        errors.append("Unsupported obstacle_type.")
    for name in ("obstacle_x", "obstacle_y", "obstacle_radius", "obstacle_width", "obstacle_height"):
        v = getattr(g, name)
        if not math.isfinite(v) or not 0 <= v <= 1:
            errors.append(f"{name} must be finite and in [0, 1].")
    sides = ("left", "right", "bottom", "top")
    for side in sides:
        bc = config.boundaries.get(side)
        if bc is None:
            errors.append(f"Missing boundary: {side}")
            continue
        if bc.type not in BOUNDARY_TYPES:
            errors.append(f"Boundary {side}: {bc.type} is not implemented; no silent fallback is allowed.")
        if not all(math.isfinite(v) for v in (bc.ux, bc.uy, bc.rho, bc.rb)) or bc.rho <= 0:
            errors.append(f"Boundary {side}: finite velocities and positive finite rho are required.")
        if bc.type == "halfway_moving_wall":
            normal = bc.ux if side in {"left", "right"} else bc.uy
            if abs(normal) > 1e-14: errors.append(f"{side}: halfway wall velocity must be tangential.")
        if bc.type == "velocity_zou_he" and math.hypot(bc.ux, bc.uy) >= CS * c.max_mach:
            errors.append(f"{side}: prescribed velocity exceeds the Mach limit.")
    for a, b in (("left", "right"), ("bottom", "top")):
        ba, bb = config.boundaries.get(a), config.boundaries.get(b)
        if ba is not None and bb is not None and ((ba.type == "periodic") != (bb.type == "periodic")):
            errors.append(f"{a} and {b} must both be periodic.")
    forcing = abs(f.body_force_x) + abs(f.body_force_y) > 0
    if forcing and any(bc.type not in {"periodic", "halfway_bounce_back", "halfway_moving_wall"}
                       for bc in config.boundaries.values()):
        errors.append("Forcing currently requires periodic or halfway boundaries; forced on-node/open BCs are not validated.")
    if config.case_type == "force_poiseuille" and not forcing:
        warnings.append("force_poiseuille has zero acceleration; the flow will remain at rest.")
    if c.min_iter > c.max_iter:
        warnings.append("min_iter > max_iter: this run cannot be labelled converged.")
    if errors:
        return errors, warnings, {k: float("nan") for k in ("Re", "Ma", "nu_lattice", "tau", "omega", "L_ref", "cs")}
    transport = lattice_transport(config)
    tau, ma = transport["tau"], transport["Ma"]
    if not math.isfinite(tau) or tau <= 0.5:
        errors.append("Positive viscosity requires tau > 0.5.")
    elif tau < 0.53:
        warnings.append("tau is close to 0.5; positive viscosity is not a stability guarantee.")
    elif tau > 1.5:
        warnings.append("Large tau: check resolution and viscosity sensitivity.")
    if ma > c.max_mach:
        errors.append("Design Mach exceeds max_mach.")
    elif ma > 0.1:
        warnings.append("Design Ma > 0.1: quantify compressibility error.")
    if f.nu_lattice is not None:
        warnings.append(f"Explicit nu_lattice sets transport; resulting reference Re={transport['Re']:.6g} (input Re ignored).")
    if any(bc.type == "non_equilibrium_extrapolation" for bc in config.boundaries.values()):
        warnings.append("On-node non-equilibrium extrapolation: monitor mass separately from residual.")
    return errors, warnings, transport


def config_to_dict(config: SimulationConfig, transport: dict[str, float] | None = None) -> dict[str, Any]:
    data = asdict(config)
    if transport is not None:
        data["derived"] = transport
    return data


def save_config(config: SimulationConfig, output_dir: str | Path, transport: dict[str, float]) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / "config.json"
    target.write_text(
        json.dumps(config_to_dict(config, transport), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target
