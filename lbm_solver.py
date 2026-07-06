"""Core D2Q9-LBGK solver for two-dimensional low-Mach incompressible flow."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterator

import numpy as np

from boundary_conditions import E, OPP, apply_boundaries, equilibrium, macroscopic
from config import SolverConfig, derived_parameters, validate_config


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
    """First-version solver with D2Q9, BGK collision, core boundaries, and masks."""

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
        self.mass0 = float(np.sum(self.rho[~self.solid_mask]))
        self.previous_ux = self.ux.copy()
        self.previous_uy = self.uy.copy()
        self.previous_residual: float | None = None
        self.residual_history: list[Report] = []

    @property
    def periodic_x(self) -> bool:
        return self.cfg.left.type == "periodic" and self.cfg.right.type == "periodic"

    @property
    def periodic_y(self) -> bool:
        return self.cfg.bottom.type == "periodic" and self.cfg.top.type == "periodic"

    def _initialize_case_velocity(self) -> None:
        if self.cfg.case_type in {"poiseuille_channel", "cylinder_flow"}:
            self.ux[:, :] = self.cfg.u_ref
        elif self.cfg.case_type == "couette_flow":
            y = np.linspace(0.0, 1.0, self.cfg.ny)[:, None]
            self.ux[:, :] = y * self.cfg.u_ref
        elif self.cfg.case_type == "periodic_channel":
            self.ux[:, :] = min(self.cfg.u_ref, 0.01)
        self.ux[self.solid_mask] = 0.0
        self.uy[self.solid_mask] = 0.0

    def collide(self) -> np.ndarray:
        feq = equilibrium(self.rho, self.ux, self.uy)
        f_post = self.f - self.cfg.omega * (self.f - feq)
        if abs(self.cfg.body_force_x) > 0.0:
            f_post += self._guo_force_x()
        return f_post

    def _guo_force_x(self) -> np.ndarray:
        force = np.zeros_like(self.f)
        fx = self.cfg.body_force_x
        for i, (ex, ey) in enumerate(E):
            eu = ex * self.ux + ey * self.uy
            term = (ex - self.ux) / 3.0 + eu * ex
            force[..., i] = (1.0 - 0.5 * self.cfg.omega) * 3.0 * fx * term
        force[self.solid_mask, :] = 0.0
        return force

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
        f_post = self.collide()
        self.f = self.stream(f_post)
        ramp = min(1.0, self.iteration / max(1, self.cfg.ramp_steps))
        apply_boundaries(self.f, self.cfg, ramp)
        self.rho, self.ux, self.uy = macroscopic(self.f, self.solid_mask)
        self.rho[self.solid_mask] = self.cfg.rho0
        self.ux[self.solid_mask] = 0.0
        self.uy[self.solid_mask] = 0.0

    def make_report(self, status: str = "running") -> Report:
        residual = self._residual()
        mass = float(np.sum(self.rho[~self.solid_mask]))
        mass_drift = abs(mass - self.mass0) / max(abs(self.mass0), 1e-30)
        speed = np.sqrt(self.ux**2 + self.uy**2)
        max_velocity = float(np.max(speed[~self.solid_mask])) if np.any(~self.solid_mask) else 0.0
        q = None if self.previous_residual in (None, 0.0) else residual / self.previous_residual
        ratios = [
            self.residual_history[k].residual / self.residual_history[k - 1].residual
            for k in range(max(1, len(self.residual_history) - 9), len(self.residual_history))
            if self.residual_history[k - 1].residual > 0
        ]
        if q is not None and q > 0:
            ratios.append(q)
        q_avg = math.exp(float(np.mean(np.log(ratios)))) if ratios else None

        if not np.isfinite(residual) or not np.isfinite(max_velocity):
            status = "diverged"
        elif mass_drift > 1e-3:
            status = "mass drift warning"
        elif self.iteration >= self.cfg.min_iter and residual < self.cfg.tol:
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
        )
        self.previous_residual = residual
        self.previous_ux = self.ux.copy()
        self.previous_uy = self.uy.copy()
        self.residual_history.append(report)
        return report

    def _residual(self) -> float:
        dux = self.ux - self.previous_ux
        duy = self.uy - self.previous_uy
        fluid = ~self.solid_mask
        numerator = np.sqrt(np.sum(dux[fluid] ** 2 + duy[fluid] ** 2))
        denominator = np.sqrt(np.sum(self.ux[fluid] ** 2 + self.uy[fluid] ** 2) + 1e-30)
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
        return {
            "rho": self.rho,
            "ux": self.ux,
            "uy": self.uy,
            "speed": speed,
            "solid_mask": self.solid_mask,
        }

    def summary(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "derived": derived_parameters(self.cfg),
            "warnings": self.warnings,
            "history": [r.as_dict() for r in self.residual_history],
        }

