"""Core D2Q9 solver for two-dimensional low-Mach incompressible flow."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from boundary_conditions import OPP, E, W, apply_boundaries, equilibrium, macroscopic
from config import SolverConfig, derived_parameters, validate_config
from thermal_lbm import ThermalField

MRT_MATRIX = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [-4, -1, -1, -1, -1, 2, 2, 2, 2],
        [4, -2, -2, -2, -2, 1, 1, 1, 1],
        [0, 1, 0, -1, 0, 1, -1, -1, 1],
        [0, -2, 0, 2, 0, 1, -1, -1, 1],
        [0, 0, 1, 0, -1, 1, 1, -1, -1],
        [0, 0, -2, 0, 2, 1, 1, -1, -1],
        [0, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 1, -1],
    ],
    dtype=np.float64,
)
MRT_MATRIX_INV = np.linalg.inv(MRT_MATRIX)


@dataclass
class Report:
    iteration: int
    residual: float
    q: float | None
    q_avg: float | None
    mass_drift: float
    max_velocity: float
    tau: float
    omega: float
    mach: float
    status: str
    temperature_residual: float | None = None
    temperature_min: float | None = None
    temperature_max: float | None = None
    nusselt_average: float | None = None

    def as_dict(self) -> dict[str, float | int | str | None]:
        return {
            "iteration": self.iteration,
            "residual": self.residual,
            "q": self.q,
            "q_avg": self.q_avg,
            "mass_drift": self.mass_drift,
            "max_velocity": self.max_velocity,
            "tau": self.tau,
            "omega": self.omega,
            "Ma": self.mach,
            "status": self.status,
            "temperature_residual": self.temperature_residual,
            "temperature_min": self.temperature_min,
            "temperature_max": self.temperature_max,
            "nusselt_average": self.nusselt_average,
        }


def build_solid_mask(cfg: SolverConfig) -> np.ndarray:
    yy, xx = np.mgrid[0 : cfg.ny, 0 : cfg.nx]
    mask = np.zeros((cfg.ny, cfg.nx), dtype=bool)
    obs = cfg.obstacle
    if obs.type == "none":
        return mask
    if obs.type == "cylinder":
        cx = obs.cx * (cfg.nx - 1)
        cy = obs.cy * (cfg.ny - 1)
        radius = obs.radius * min(cfg.nx, cfg.ny)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    elif obs.type == "rectangle":
        cx = obs.cx * (cfg.nx - 1)
        cy = obs.cy * (cfg.ny - 1)
        width = obs.width * cfg.nx
        height = obs.height * cfg.ny
        mask = (np.abs(xx - cx) <= width / 2.0) & (np.abs(yy - cy) <= height / 2.0)
    elif obs.type == "mask_image" and obs.mask_path:
        mask = _load_mask_image(obs.mask_path, cfg.nx, cfg.ny)
    return mask


def _load_mask_image(path: str | Path, nx: int, ny: int) -> np.ndarray:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Pillow is required for mask_image obstacles.") from exc
    image = Image.open(path).convert("L").resize((nx, ny))
    values = np.asarray(image)
    return values < 128


class LBMSolver:
    """D2Q9 solver with BGK, TRT, MRT, core boundaries, and solid masks."""

    def __init__(self, cfg: SolverConfig):
        errors, warnings = validate_config(cfg)
        if errors:
            raise ValueError("; ".join(errors))
        self.cfg = cfg
        self.warnings = warnings
        self.solid_mask = build_solid_mask(cfg)
        self.iteration = 0
        self.rho = np.full((cfg.ny, cfg.nx), cfg.rho0, dtype=np.float64)
        self.ux = np.zeros((cfg.ny, cfg.nx), dtype=np.float64)
        self.uy = np.zeros((cfg.ny, cfg.nx), dtype=np.float64)
        self._initialize_case_velocity()
        self.f = equilibrium(self.rho, self.ux, self.uy)
        self.thermal = ThermalField(cfg, self.solid_mask) if cfg.thermal_enabled else None
        self.mass0 = float(np.sum(self.rho[~self.solid_mask]))
        self.previous_ux = self.ux.copy()
        self.previous_uy = self.uy.copy()
        self.previous_residual: float | None = None
        self.previous_temperature = (
            self.thermal.temperature.copy() if self.thermal is not None else None
        )
        self.residual_history: list[Report] = []
        self.current_ramp = 1.0

    @property
    def periodic_x(self) -> bool:
        return self.cfg.left.type == "periodic" and self.cfg.right.type == "periodic"

    @property
    def periodic_y(self) -> bool:
        return self.cfg.bottom.type == "periodic" and self.cfg.top.type == "periodic"

    def _initialize_case_velocity(self) -> None:
        if self.cfg.case_type in {
            "poiseuille_channel",
            "heated_channel_flow",
            "cylinder_flow",
            "square_cylinder_flow",
            "backward_facing_step",
        }:
            self.ux[:, :] = self.cfg.u_ref
        elif self.cfg.case_type == "couette_flow":
            y = np.linspace(0.0, 1.0, self.cfg.ny)[:, None]
            self.ux[:, :] = y * self.cfg.u_ref
        elif self.cfg.case_type in {"periodic_channel", "open_channel_flow"}:
            self.ux[:, :] = min(self.cfg.u_ref, 0.01)
        elif self.cfg.case_type == "taylor_green_vortex":
            x = 2.0 * np.pi * np.arange(self.cfg.nx) / self.cfg.nx
            y = 2.0 * np.pi * np.arange(self.cfg.ny) / self.cfg.ny
            phase_x, phase_y = np.meshgrid(x, y)
            kx = 2.0 * np.pi / self.cfg.nx
            ky = 2.0 * np.pi / self.cfg.ny
            amplitude = self.cfg.u_ref / max(kx, ky)
            self.ux[:, :] = amplitude * ky * np.cos(phase_x) * np.sin(phase_y)
            self.uy[:, :] = -amplitude * kx * np.sin(phase_x) * np.cos(phase_y)
            if self.cfg.nx == self.cfg.ny:
                pressure_density = 0.75 * self.cfg.u_ref**2 * (
                    np.cos(2.0 * phase_x) + np.cos(2.0 * phase_y)
                )
                self.rho[:, :] = self.cfg.rho0 * (1.0 - pressure_density)
        elif self.cfg.case_type == "shear_wave_decay":
            y = 2.0 * np.pi * np.arange(self.cfg.ny) / self.cfg.ny
            self.ux[:, :] = self.cfg.u_ref * np.sin(y)[:, None]
        self.ux[self.solid_mask] = 0.0
        self.uy[self.solid_mask] = 0.0

    def collide(self) -> np.ndarray:
        feq = equilibrium(self.rho, self.ux, self.uy)
        has_force = (
            math.hypot(self.cfg.body_force_x, self.cfg.body_force_y) > 0.0
            or self.cfg.thermal_enabled
            and self.cfg.thermal_buoyancy
        )
        source = self._guo_force_source() if has_force else None
        if self.cfg.collision_model == "BGK":
            return self._collide_bgk(feq, source)
        if self.cfg.collision_model == "TRT":
            return self._collide_trt(feq, source)
        if self.cfg.collision_model == "MRT":
            return self._collide_mrt(feq, source)
        raise ValueError(f"Unsupported collision model: {self.cfg.collision_model}")

    def _collide_bgk(self, feq: np.ndarray, source: np.ndarray | None) -> np.ndarray:
        f_post = self.f - self.cfg.omega * (self.f - feq)
        if source is not None:
            f_post += (1.0 - 0.5 * self.cfg.omega) * source
        return f_post

    def _collide_trt(self, feq: np.ndarray, source: np.ndarray | None) -> np.ndarray:
        delta_even, delta_odd = self._even_odd(self.f - feq)
        omega_plus = self.cfg.omega
        omega_minus = self.cfg.trt_omega_minus
        f_post = self.f - omega_plus * delta_even - omega_minus * delta_odd
        if source is not None:
            source_even, source_odd = self._even_odd(source)
            f_post += (1.0 - 0.5 * omega_plus) * source_even
            f_post += (1.0 - 0.5 * omega_minus) * source_odd
        return f_post

    def _collide_mrt(self, feq: np.ndarray, source: np.ndarray | None) -> np.ndarray:
        moments = self.f @ MRT_MATRIX.T
        moments_eq = feq @ MRT_MATRIX.T
        rates = np.asarray(self.cfg.mrt_relaxation_rates)
        moments_post = moments - rates * (moments - moments_eq)
        if source is not None:
            source_moments = source @ MRT_MATRIX.T
            moments_post += (1.0 - 0.5 * rates) * source_moments
        return moments_post @ MRT_MATRIX_INV.T

    @staticmethod
    def _even_odd(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        opposite = values[..., OPP]
        return 0.5 * (values + opposite), 0.5 * (values - opposite)

    def _guo_force_source(self) -> np.ndarray:
        source = np.zeros_like(self.f)
        fx, fy = self._force_components()
        for i, (ex, ey) in enumerate(E):
            eu = ex * self.ux + ey * self.uy
            force_projection = (ex - self.ux) * fx + (ey - self.uy) * fy
            lattice_projection = ex * fx + ey * fy
            source[..., i] = W[i] * (3.0 * force_projection + 9.0 * eu * lattice_projection)
        source[self.solid_mask, :] = 0.0
        return source

    def _force_components(self) -> tuple[np.ndarray, np.ndarray]:
        fx = np.full_like(self.rho, self.cfg.body_force_x * self.current_ramp)
        fy = np.full_like(self.rho, self.cfg.body_force_y * self.current_ramp)
        if self.thermal is not None and self.cfg.thermal_buoyancy:
            gravity_norm = math.hypot(self.cfg.gravity_x, self.cfg.gravity_y)
            temperature_offset = self.thermal.temperature - self.cfg.temperature_reference
            buoyancy = (
                self.rho
                * self.cfg.buoyancy_per_temperature
                * temperature_offset
                * self.current_ramp
            )
            fx -= self.cfg.gravity_x / gravity_norm * buoyancy
            fy -= self.cfg.gravity_y / gravity_norm * buoyancy
        fx[self.solid_mask] = 0.0
        fy[self.solid_mask] = 0.0
        return fx, fy

    def stream(self, f_post: np.ndarray) -> np.ndarray:
        streamed = np.zeros_like(f_post)
        for i, (cx, cy) in enumerate(E):
            arr = f_post[..., i]
            if self.periodic_x and self.periodic_y:
                moved = np.roll(np.roll(arr, shift=cy, axis=0), shift=cx, axis=1)
                source_solid = np.roll(
                    np.roll(self.solid_mask, shift=cy, axis=0), shift=cx, axis=1
                )
                streamed[..., i] = moved
            elif self.periodic_x:
                moved = np.roll(arr, shift=cx, axis=1)
                source_solid = np.roll(self.solid_mask, shift=cx, axis=1)
                y_src, y_dst = self._axis_slices(cy, self.cfg.ny)
                streamed[y_dst, :, i] = moved[y_src, :]
                solid_dest = np.zeros_like(self.solid_mask)
                solid_dest[y_dst, :] = source_solid[y_src, :]
                source_solid = solid_dest
            elif self.periodic_y:
                moved = np.roll(arr, shift=cy, axis=0)
                source_solid = np.roll(self.solid_mask, shift=cy, axis=0)
                x_src, x_dst = self._axis_slices(cx, self.cfg.nx)
                streamed[:, x_dst, i] = moved[:, x_src]
                solid_dest = np.zeros_like(self.solid_mask)
                solid_dest[:, x_dst] = source_solid[:, x_src]
                source_solid = solid_dest
            else:
                y_src, y_dst = self._axis_slices(cy, self.cfg.ny)
                x_src, x_dst = self._axis_slices(cx, self.cfg.nx)
                streamed[y_dst, x_dst, i] = arr[y_src, x_src]
                source_solid = np.zeros_like(self.solid_mask)
                source_solid[y_dst, x_dst] = self.solid_mask[y_src, x_src]

            bounce = source_solid & ~self.solid_mask
            streamed[bounce, i] = f_post[bounce, OPP[i]]

        streamed[self.solid_mask, :] = f_post[self.solid_mask][:, OPP]
        return streamed

    @staticmethod
    def _axis_slices(shift: int, size: int) -> tuple[slice, slice]:
        if shift > 0:
            return slice(0, size - shift), slice(shift, size)
        if shift < 0:
            return slice(-shift, size), slice(0, size + shift)
        return slice(0, size), slice(0, size)

    def step(self) -> None:
        self.iteration += 1
        ramp = self.ramp_factor()
        self.current_ramp = ramp
        f_post = self.collide()
        self.f = self.stream(f_post)
        apply_boundaries(self.f, self.cfg, ramp)
        self.rho, self.ux, self.uy = macroscopic(self.f, self.solid_mask)
        has_force = (
            math.hypot(self.cfg.body_force_x, self.cfg.body_force_y) > 0.0
            or self.cfg.thermal_enabled
            and self.cfg.thermal_buoyancy
        )
        if has_force:
            fluid = ~self.solid_mask
            fx, fy = self._force_components()
            self.ux[fluid] += 0.5 * fx[fluid] / self.rho[fluid]
            self.uy[fluid] += 0.5 * fy[fluid] / self.rho[fluid]
        self.rho[self.solid_mask] = self.cfg.rho0
        self.ux[self.solid_mask] = 0.0
        self.uy[self.solid_mask] = 0.0
        if self.thermal is not None:
            self.thermal.step(self.ux, self.uy)

    def ramp_factor(self) -> float:
        if self.cfg.ramp_profile == "instant":
            return 1.0
        x = min(1.0, self.iteration / max(1, self.cfg.ramp_steps))
        if self.cfg.ramp_profile == "smoothstep":
            return x * x * (3.0 - 2.0 * x)
        if self.cfg.ramp_profile == "exponential":
            return (1.0 - math.exp(-5.0 * x)) / (1.0 - math.exp(-5.0))
        return x

    def make_report(self, status: str = "running") -> Report:
        residual = self._residual()
        mass = float(np.sum(self.rho[~self.solid_mask]))
        mass_drift = abs(mass - self.mass0) / max(abs(self.mass0), 1e-30)
        speed = np.sqrt(self.ux**2 + self.uy**2)
        max_velocity = float(np.max(speed[~self.solid_mask])) if np.any(~self.solid_mask) else 0.0
        temperature_residual = self._temperature_residual()
        temperature_min = None
        temperature_max = None
        nusselt_average = None
        if self.thermal is not None:
            fluid_temperature = self.thermal.temperature[~self.solid_mask]
            temperature_min = float(np.min(fluid_temperature))
            temperature_max = float(np.max(fluid_temperature))
            nusselt_average = self.thermal.average_nusselt()
        q = None if self.previous_residual in (None, 0.0) else residual / self.previous_residual
        ratios = [
            self.residual_history[k].residual / self.residual_history[k - 1].residual
            for k in range(max(1, len(self.residual_history) - 9), len(self.residual_history))
            if self.residual_history[k - 1].residual > 0
        ]
        if q is not None and q > 0:
            ratios.append(q)
        q_avg = math.exp(float(np.mean(np.log(ratios)))) if ratios else None

        thermal_is_finite = temperature_residual is None or np.isfinite(temperature_residual)
        if (
            not np.isfinite(residual)
            or not np.isfinite(max_velocity)
            or not thermal_is_finite
            or residual > self.cfg.residual_limit
            or max_velocity > self.cfg.max_velocity_limit
            or mass_drift > self.cfg.mass_drift_limit
        ):
            status = "diverged"
        elif mass_drift > self.cfg.mass_drift_warning:
            status = "mass drift warning"
        elif (
            self.iteration >= self.cfg.min_iter
            and residual < self.cfg.tol
            and (temperature_residual is None or temperature_residual < self.cfg.thermal_tol)
        ):
            status = "converged"

        report = Report(
            iteration=self.iteration,
            residual=residual,
            q=q,
            q_avg=q_avg,
            mass_drift=mass_drift,
            max_velocity=max_velocity,
            tau=self.cfg.tau,
            omega=self.cfg.omega,
            mach=self.cfg.mach,
            status=status,
            temperature_residual=temperature_residual,
            temperature_min=temperature_min,
            temperature_max=temperature_max,
            nusselt_average=nusselt_average,
        )
        self.previous_residual = residual
        self.previous_ux = self.ux.copy()
        self.previous_uy = self.uy.copy()
        if self.thermal is not None:
            self.previous_temperature = self.thermal.temperature.copy()
        self.residual_history.append(report)
        return report

    def _residual(self) -> float:
        dux = self.ux - self.previous_ux
        duy = self.uy - self.previous_uy
        fluid = ~self.solid_mask
        numerator = np.sqrt(np.sum(dux[fluid] ** 2 + duy[fluid] ** 2))
        denominator = np.sqrt(np.sum(self.ux[fluid] ** 2 + self.uy[fluid] ** 2) + 1e-30)
        return float(numerator / denominator)

    def _temperature_residual(self) -> float | None:
        if self.thermal is None or self.previous_temperature is None:
            return None
        fluid = ~self.solid_mask
        difference = self.thermal.temperature - self.previous_temperature
        numerator = np.sqrt(np.sum(difference[fluid] ** 2))
        denominator = np.sqrt(np.sum(self.thermal.temperature[fluid] ** 2) + 1e-30)
        return float(numerator / denominator)

    def run(self) -> Iterator[Report]:
        yield self.make_report("initialized")
        while self.iteration < self.cfg.max_iter:
            self.step()
            if self.iteration % self.cfg.report_interval == 0 or self.iteration == self.cfg.max_iter:
                report = self.make_report()
                yield report
                if report.status in {"converged", "diverged"}:
                    break

    def fields(self) -> dict[str, np.ndarray]:
        speed = np.sqrt(self.ux**2 + self.uy**2)
        fields = {
            "rho": self.rho,
            "ux": self.ux,
            "uy": self.uy,
            "speed": speed,
            "solid_mask": self.solid_mask,
        }
        if self.thermal is not None:
            fields["temperature"] = self.thermal.temperature
        return fields

    def summary(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "derived": derived_parameters(self.cfg),
            "warnings": self.warnings,
            "history": [r.as_dict() for r in self.residual_history],
        }
