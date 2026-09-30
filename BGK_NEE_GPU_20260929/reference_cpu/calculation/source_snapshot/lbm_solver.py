"""D2Q9-LBGK lattice Boltzmann solver for rectangular 2D domains."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator
import math
import copy
import json
import time
import datetime
import hashlib
import platform
import sys

import numpy as np

from boundary_conditions import E, W, OPP, MacroFields, apply_boundary, apply_solid_bounce_back, non_equilibrium_cavity
from config import BoundaryConfig, SimulationConfig, lattice_transport, save_config, validate_config, domain_axis
from postprocess import save_all_outputs, speed, vorticity


@dataclass
class StepReport:
    iteration: int
    residual: float
    q: float
    q_avg: float
    mass_drift: float
    max_velocity: float
    converged: bool
    diverged: bool
    message: str
    min_density: float = 0.0
    max_density: float = 0.0
    max_mach: float = 0.0
    mass_balance_error: float = 0.0
    stop_reason: str = "running"


@dataclass
class LBMResult:
    rho: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    solid_mask: np.ndarray
    history: list[dict[str, float]]
    config: SimulationConfig
    transport: dict[str, float]
    output_files: dict[str, Path] = field(default_factory=dict)
    converged: bool = False
    message: str = ""
    stop_reason: str = "running"


class LBMSolver:
    def __init__(self, config: SimulationConfig, solid_mask: np.ndarray | None = None) -> None:
        errors, warnings, transport = validate_config(config)
        if errors:
            raise ValueError("\n".join(errors))
        config = copy.deepcopy(config)
        self.config = config
        self.warnings = warnings
        self.transport = transport
        self.ny = int(config.grid.NY)
        self.nx = int(config.grid.NX)
        self.omega = transport["omega"]
        self.rho = np.full((self.ny, self.nx), config.flow.rho0, dtype=np.float64)
        self.ux = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.uy = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.solid_mask = self._build_solid_mask() if solid_mask is None else solid_mask.astype(bool)
        if self.solid_mask.shape != (self.ny, self.nx):
            raise ValueError("solid_mask shape must be (NY, NX).")
        if np.all(self.solid_mask):
            raise ValueError("Geometry contains no fluid nodes.")
        self._residual_mask = ~self.solid_mask.copy()
        self.residual_domain = "all_fluid_nodes"
        if config.case_type == "lid_driven_cavity" and all(
                bc.type == "non_equilibrium_extrapolation" for bc in config.boundaries.values()):
            # Appendix D p.222 sums only interior nodes; wall values are prescribed.
            self._residual_mask[[0, -1], :] = False
            self._residual_mask[:, [0, -1]] = False
            self.residual_domain = "interior_without_on_node_walls"
        self._initialize_velocity()
        # Guo forcing: physical velocity is raw momentum/rho + acceleration/2.
        self.f = self.equilibrium(self.rho,
            self.ux - 0.5 * config.flow.body_force_x,
            self.uy - 0.5 * config.flow.body_force_y)
        self.iteration = 0
        self.mass0 = float(np.sum(self.rho[~self.solid_mask]))
        self.history: list[dict[str, float]] = []
        self._prev_residual: float | None = None
        self._recent_q: list[float] = []
        self._step_ux = self.ux.copy()
        self._step_uy = self.uy.copy()
        self.message = "initialized"
        self.stop_reason = "running"
        self._failure = ""
        self._stable_reports = 0
        self._last_report = None
        self._last_report_iteration = 0
        self._last_mass = self.mass0
        self._boundary_exchange_sum = 0.0
        self._started = time.perf_counter()
        self._final_result = None
        self._closed = all(bc.type in {"periodic", "halfway_bounce_back", "halfway_moving_wall",
                          "no_slip_bounce_back", "moving_wall_bounce_back"}
                           for bc in config.boundaries.values()) or config.case_type == "lid_driven_cavity"

    @staticmethod
    def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
        eu = (
            E[:, 0] * ux[..., None]
            + E[:, 1] * uy[..., None]
        )
        uu = ux * ux + uy * uy
        return W * rho[..., None] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uu[..., None])

    def _initialize_velocity(self) -> None:
        case = self.config.case_type
        u = self.config.flow.U_ref
        if self.config.flow.initial_velocity == "uniform":
            self.ux[:, :] = u
            self.uy[:, :] = 0.0
        elif self.config.flow.initial_velocity == "couette":
            offset, height = domain_axis(self.config, "y")
            y = (np.arange(self.ny) + offset) / height
            self.ux[:, :] = y[:, None] * u
        else:
            self.ux[:, :] = 0.0
            self.uy[:, :] = 0.0
        self.ux[self.solid_mask] = 0.0
        self.uy[self.solid_mask] = 0.0

    def _build_solid_mask(self) -> np.ndarray:
        mask = np.zeros((self.ny, self.nx), dtype=bool)
        grid = self.config.grid
        if grid.obstacle_type == "circle" or self.config.case_type == "cylinder_flow":
            cx = int(round(grid.obstacle_x * (self.nx - 1)))
            cy = int(round(grid.obstacle_y * (self.ny - 1)))
            radius = max(1, int(round(grid.obstacle_radius * min(self.nx, self.ny))))
            yy, xx = np.ogrid[: self.ny, : self.nx]
            mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
        elif grid.obstacle_type == "rectangle":
            cx = int(round(grid.obstacle_x * (self.nx - 1)))
            cy = int(round(grid.obstacle_y * (self.ny - 1)))
            half_w = max(1, int(round(0.5 * grid.obstacle_width * self.nx)))
            half_h = max(1, int(round(0.5 * grid.obstacle_height * self.ny)))
            y0, y1 = max(0, cy - half_h), min(self.ny, cy + half_h + 1)
            x0, x1 = max(0, cx - half_w), min(self.nx, cx + half_w + 1)
            mask[y0:y1, x0:x1] = True
        return mask

    def collide(self) -> np.ndarray:
        feq = self.equilibrium(self.rho, self.ux, self.uy)
        post = self.f - self.omega * (self.f - feq)
        ax, ay = self.config.flow.body_force_x, self.config.flow.body_force_y
        if ax or ay:
            eu = self.ux[..., None] * E[:, 0] + self.uy[..., None] * E[:, 1]
            ea = E[:, 0] * ax + E[:, 1] * ay
            ua = self.ux * ax + self.uy * ay
            source = W * self.rho[..., None] * (1 - self.omega / 2) * (
                3 * (ea - ua[..., None]) + 9 * eu * ea)
            source[self.solid_mask] = 0.0
            post += source
        return post

    def stream(self, f_post: np.ndarray) -> np.ndarray:
        f_streamed = np.zeros_like(f_post)
        periodic_x = self.config.boundaries["left"].type == "periodic" and self.config.boundaries["right"].type == "periodic"
        periodic_y = self.config.boundaries["bottom"].type == "periodic" and self.config.boundaries["top"].type == "periodic"

        for i, (ex, ey) in enumerate(E):
            src = f_post[..., i]
            if periodic_x and periodic_y:
                f_streamed[..., i] = np.roll(src, shift=(ey, ex), axis=(0, 1))
                continue
            if periodic_x and ey == 0:
                f_streamed[..., i] = np.roll(src, shift=ex, axis=1)
                continue
            if periodic_y and ex == 0:
                f_streamed[..., i] = np.roll(src, shift=ey, axis=0)
                continue

            y_src0 = max(0, -ey)
            y_src1 = min(self.ny, self.ny - ey)
            x_src0 = max(0, -ex)
            x_src1 = min(self.nx, self.nx - ex)
            y_dst0 = y_src0 + ey
            y_dst1 = y_src1 + ey
            x_dst0 = x_src0 + ex
            x_dst1 = x_src1 + ex
            f_streamed[y_dst0:y_dst1, x_dst0:x_dst1, i] = src[y_src0:y_src1, x_src0:x_src1]

            if periodic_x and ex != 0:
                if ex > 0:
                    f_streamed[y_dst0:y_dst1, 0, i] = src[y_src0:y_src1, -1]
                else:
                    f_streamed[y_dst0:y_dst1, -1, i] = src[y_src0:y_src1, 0]
            if periodic_y and ey != 0:
                if ey > 0:
                    f_streamed[0, x_dst0:x_dst1, i] = src[-1, x_src0:x_src1]
                else:
                    f_streamed[-1, x_dst0:x_dst1, i] = src[0, x_src0:x_src1]

        return f_streamed

    def apply_boundaries(self, f_post: np.ndarray) -> None:
        apply_solid_bounce_back(self.f, f_post, self.solid_mask)
        # Interior moments must belong to the streamed time level for NEE.
        density = np.sum(self.f, axis=-1)
        safe = np.where(density > 0, density, 1.0)
        ux = np.sum(self.f * E[:, 0], axis=-1) / safe
        uy = np.sum(self.f * E[:, 1], axis=-1) / safe
        macros = MacroFields(density, ux, uy, f_post)
        if self.config.case_type == "lid_driven_cavity" and all(
                bc.type == "non_equilibrium_extrapolation" for bc in self.config.boundaries.values()):
            if self.config.flow.body_force_x or self.config.flow.body_force_y or np.any(self.solid_mask):
                raise ValueError("The comparison NEE cavity requires zero force and no obstacles.")
            non_equilibrium_cavity(self.f,
                {side: self._ramped_boundary(bc) for side, bc in self.config.boundaries.items()},
                macros, self.equilibrium)
            return
        # Halfway moving-wall correction uses density before streaming.
        wall_macros = MacroFields(self.rho, self.ux, self.uy, f_post)
        # Reconstruct wall populations before open faces use them at corners.
        # Preserve left/right/bottom/top ordering within each group.
        sides = sorted(("left", "right", "bottom", "top"),
            key=lambda side: not self.config.boundaries[side].type.startswith("halfway_"))
        for side in sides:
            bc = self._ramped_boundary(self.config.boundaries[side])
            apply_boundary(self.f, side, bc, wall_macros if bc.type.startswith("halfway_") else macros, self.equilibrium)

    def _ramped_boundary(self, bc: BoundaryConfig) -> BoundaryConfig:
        ramp_steps = self.config.convergence.ramp_steps
        if ramp_steps <= 0:
            return bc
        factor = min(1.0, (self.iteration + 1) / float(ramp_steps))
        return BoundaryConfig(type=bc.type, ux=bc.ux * factor, uy=bc.uy * factor, rho=bc.rho, rb=bc.rb)

    def update_macroscopic(self) -> None:
        self.rho = np.sum(self.f, axis=2)
        momentum_x = np.sum(self.f * E[:, 0][None, None, :], axis=2)
        momentum_y = np.sum(self.f * E[:, 1][None, None, :], axis=2)
        safe_rho = np.where(self.rho > 1.0e-14, self.rho, 1.0)
        self.ux = momentum_x / safe_rho + 0.5 * self.config.flow.body_force_x
        self.uy = momentum_y / safe_rho + 0.5 * self.config.flow.body_force_y
        self.rho[self.solid_mask] = self.config.flow.rho0
        self.ux[self.solid_mask] = 0.0
        self.uy[self.solid_mask] = 0.0

    def _check_fields(self) -> None:
        fluid = ~self.solid_mask
        if not (np.isfinite(self.f[fluid]).all() and np.isfinite(self.rho[fluid]).all()
                and np.isfinite(self.ux[fluid]).all() and np.isfinite(self.uy[fluid]).all()):
            self._failure = "Non-finite population or macroscopic field"
        elif np.min(self.rho[fluid]) <= 0:
            self._failure = "Non-positive density"
        elif np.max(speed(self.ux[fluid], self.uy[fluid])) / self.transport["cs"] > self.config.convergence.max_mach:
            self._failure = "Actual maximum Mach exceeded configured limit"
        if self._failure:
            self.stop_reason = "diverged"

    def _open_exchange(self, f_post: np.ndarray) -> float:
        # Signed lattice populations crossing OPEN faces. This diagnostic does
        # not hide extra mass introduced by on-node reconstruction of known f.
        total = 0.0
        wall_types = {"periodic", "halfway_bounce_back", "halfway_moving_wall",
                      "no_slip_bounce_back", "moving_wall_bounce_back"}
        if self._closed:
            return total
        from boundary_conditions import SIDE_SLICES, NORMAL_REFLECTION
        for side, bc in self.config.boundaries.items():
            if bc.type in wall_types: continue
            idx = SIDE_SLICES[side]
            fluid = ~self.solid_mask[idx]
            for incoming, outgoing in NORMAL_REFLECTION[side]:
                total += float(np.sum((self.f[idx + (incoming,)] - f_post[idx + (outgoing,)])[fluid]))
        return total

    def step(self) -> None:
        if self.stop_reason != "running": return
        self._check_fields()
        if self._failure: return
        self._step_ux, self._step_uy = self.ux.copy(), self.uy.copy()
        f_post = self.collide()
        self.f = self.stream(f_post)
        self.apply_boundaries(f_post)
        self._boundary_exchange_sum += self._open_exchange(f_post)
        self.iteration += 1
        self.update_macroscopic()
        self._check_fields()
        residual = self._velocity_residual()
        mass = float(np.sum(self.rho[~self.solid_mask]))
        mass_drift = abs(mass - self.mass0) / self.mass0
        balance = abs(mass - self._last_mass - self._boundary_exchange_sum) / self.mass0
        c = self.config.convergence
        ready = (self.iteration >= max(c.min_iter, c.ramp_steps)
                 and residual < c.tol and np.isfinite(residual) and not self._failure
                 and (not c.mass_criterion or
                      (mass_drift if self._closed else balance) <= c.mass_tolerance))
        self._stable_reports = self._stable_reports + 1 if ready else 0
        if c.steady and self._stable_reports >= c.consecutive_reports:
            self.stop_reason = "converged"

    def _velocity_residual(self, include_on_node_walls: bool = False) -> float:
        """Relative velocity change over ONE lattice step (textbook criterion).

        Exactly stationary zero flow has residual 0; nonzero change ending at
        zero velocity has residual infinity. No velocity-scale floor is used.
        """
        fluid = ~self.solid_mask if include_on_node_walls else self._residual_mask
        du2 = (self.ux - self._step_ux) ** 2 + (self.uy - self._step_uy) ** 2
        u2 = self.ux ** 2 + self.uy ** 2
        numerator = float(np.sqrt(np.sum(du2[fluid])))
        denominator = float(np.sqrt(np.sum(u2[fluid])))
        if denominator == 0.0:
            return 0.0 if numerator == 0.0 else float("inf")
        return numerator / denominator

    def report(self) -> StepReport:
        if self._last_report is not None and self._last_report.iteration == self.iteration:
            return self._last_report
        fluid = ~self.solid_mask
        self._check_fields()
        residual = self._velocity_residual()
        q = residual / self._prev_residual if self._prev_residual and np.isfinite(self._prev_residual) else float("nan")
        if np.isfinite(q) and q > 0:
            self._recent_q.append(q)
            self._recent_q = self._recent_q[-10:]
        q_avg = float(np.exp(np.mean(np.log(self._recent_q)))) if self._recent_q else float("nan")
        mass = float(np.sum(self.rho[fluid]))
        mass_drift = abs(mass - self.mass0) / self.mass0
        balance = abs(mass - self._last_mass - self._boundary_exchange_sum) / self.mass0
        max_velocity = float(np.max(speed(self.ux[fluid], self.uy[fluid])))
        rho_min, rho_max = float(np.min(self.rho[fluid])), float(np.max(self.rho[fluid]))
        max_ma = max_velocity / self.transport["cs"]
        c = self.config.convergence
        if self._failure: self.stop_reason = "diverged"
        self.message = self._failure or self.stop_reason
        if self.stop_reason == "running":
            self.message = f"running; steady steps {self._stable_reports}/{c.consecutive_reports}"
            if self._closed and mass_drift > c.mass_tolerance:
                self.message += "; mass conservation criterion failed"
            if not self._closed and balance > c.mass_tolerance:
                self.message += "; discrete open-boundary mass balance warning"
        row = dict(iteration=float(self.iteration), residual=residual, q=q, q_avg=q_avg,
            residual_all_nodes=self._velocity_residual(include_on_node_walls=True),
            mass_drift=mass_drift, max_velocity=max_velocity, min_density=rho_min,
            max_density=rho_max, max_mach=max_ma, mass_balance_error=balance,
            mean_ux=float(np.mean(self.ux[fluid])), mean_uy=float(np.mean(self.uy[fluid])),
            max_transverse_velocity=float(np.max(np.abs(self.uy[fluid]))),
            report_steps=float(self.iteration-self._last_report_iteration), residual_steps=1.0)
        self.history.append(row)
        self._prev_residual = residual
        self._last_mass, self._boundary_exchange_sum = mass, 0.0
        self._last_report_iteration = self.iteration
        self._last_report = StepReport(self.iteration, residual, q, q_avg, mass_drift,
            max_velocity, self.stop_reason == "converged", self.stop_reason == "diverged",
            self.message, rho_min, rho_max, max_ma, balance, self.stop_reason)
        return self._last_report

    def run(self, should_stop: Callable[[], bool] | None = None) -> Iterator[StepReport]:
        c = self.config.convergence
        for _ in range(c.max_iter):
            if should_stop is not None and should_stop():
                self.cancel()
                return
            self.step()
            if self._failure or self.stop_reason != "running" or self.iteration % c.report_interval == 0 or self.iteration == 1:
                report = self.report()
                if self.iteration >= c.max_iter and self.stop_reason == "running":
                    self.stop_reason = "max_iter" if c.steady else "completed_steps"
                    self.message = self.stop_reason
                    report.stop_reason = self.stop_reason
                    report.message = self.message
                yield report
                if self.stop_reason != "running": return
        if self.stop_reason == "running":
            self.stop_reason = "max_iter" if c.steady else "completed_steps"
            report = self.report()
            report.stop_reason = self.stop_reason
            report.message = self.message = self.stop_reason
            yield report

    def cancel(self) -> None:
        if self.stop_reason == "running":
            self.stop_reason = "cancelled"
            self.message = "cancelled"

    def finalize(self, save_outputs: bool = True) -> LBMResult:
        if self._final_result is not None: return self._final_result
        if self.stop_reason == "running": self.cancel()
        if not self.history or self.history[-1]["iteration"] != self.iteration:
            self.report()
        files = {}
        if save_outputs:
            path = Path(self.config.output.output_dir)
            if any((path / name).exists() for name in ("config.json", "results.npz", "run_summary.json")):
                path = path / datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
            self.config.output.output_dir = str(path)
            path.mkdir(parents=True, exist_ok=True)
            files["config"] = save_config(self.config, path, self.transport)
            # Invalid states are saved as raw data/diagnostics; do not plot them.
            output_config = copy.deepcopy(self.config)
            if self.stop_reason == "diverged": output_config.output.save_png = False
            files.update(save_all_outputs(self.rho, self.ux, self.uy, self.solid_mask,
                self.history, output_config, self.transport))
            summary = dict(stop_reason=self.stop_reason, converged=self.stop_reason == "converged",
                iteration=self.iteration, elapsed_seconds=time.perf_counter()-self._started,
                final_diagnostics=self.history[-1] if self.history else {},
                boundary_mass_diagnostic="closed total mass" if self._closed else "open lattice-face exchange; on-node corner quadrature may contribute",
                convergence_policy="velocity_and_mass" if self.config.convergence.mass_criterion else "velocity_only_mass_reported",
                residual_domain=self.residual_domain,
                warnings=self.warnings, error=self._failure,
                python=sys.version, executable=sys.executable, numpy=np.__version__, platform=platform.platform(),
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path(__file__).parent.glob("*.py")})
            def clean(obj):
                if isinstance(obj, dict): return {k:clean(v) for k,v in obj.items()}
                if isinstance(obj, list): return [clean(v) for v in obj]
                if isinstance(obj, float) and not math.isfinite(obj): return None
                return obj
            summary_path = path / "run_summary.json"
            summary_path.write_text(json.dumps(clean(summary), ensure_ascii=False, indent=2, allow_nan=False))
            files["summary"] = summary_path
        result = LBMResult(self.rho, self.ux, self.uy, self.solid_mask, self.history,
            self.config, self.transport, files, self.stop_reason == "converged", self.message, self.stop_reason)
        self._final_result = result
        return result

    def field_snapshot(self) -> dict[str, np.ndarray]:
        return {
            "rho": self.rho.copy(),
            "ux": self.ux.copy(),
            "uy": self.uy.copy(),
            "speed": speed(self.ux, self.uy),
            "vorticity": vorticity(self.ux, self.uy, self.solid_mask),
            "solid_mask": self.solid_mask.copy(),
        }
