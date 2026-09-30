"""Double-precision GPU backends for BGK/MRT and NEE/halfway cavities."""
import copy
import math

import gpu_environment
import cupy as cp
import numpy as np

from cavity_models import CavityCPUSolver, M, MINV, require_supported
from boundary_conditions import E, W, OPP


class ArrayGPUSolver:
    backend = "cupy-array"

    def __init__(self, config):
        require_supported(config)
        self.host = CavityCPUSolver(copy.deepcopy(config))
        self.config = self.host.config
        self.transport = self.host.transport
        self.nx, self.ny = self.host.nx, self.host.ny
        self.e = cp.asarray(E, dtype=cp.float64)
        self.w = cp.asarray(W, dtype=cp.float64)
        self._m, self._minv = cp.asarray(M, order="C"), cp.asarray(MINV, order="C")
        self._rates = cp.asarray(self.host.rates)
        self._links = [(k, cp.asarray(missing), cp.asarray(moving)) for k, missing, moving in self.host.links]
        for name in ("f", "rho", "ux", "uy", "_step_ux", "_step_uy"):
            setattr(self, name, cp.asarray(getattr(self.host, name)))
        self._decision_residual = 0.0
        self._last_stats = None
        self._state_validated = False

    def __getattr__(self, name):
        return getattr(self.host, name)

    @property
    def decision_residual(self):
        return self._decision_residual

    def equilibrium(self, rho, ux, uy):
        eu = self.e[:, 0] * ux[..., None] + self.e[:, 1] * uy[..., None]
        uu = ux * ux + uy * uy
        return self.w * rho[..., None] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uu[..., None])

    def _evolve(self):
        feq = self.equilibrium(self.rho, self.ux, self.uy)
        if self.config.collision_model == "MRT":
            # Explicit reductions keep the reference backend independent of the
            # optional cuBLAS DLL (not installed in the bundled CUDA runtime).
            delta = self.f - feq
            dm = cp.stack([(delta * self._m[a]).sum(axis=-1) * self._rates[a] for a in range(9)], axis=-1)
            post = self.f - cp.stack([(dm * self._minv[k]).sum(axis=-1) for k in range(9)], axis=-1)
        else:
            post = self.f - self.transport["omega"] * (self.f - feq)
        streamed = cp.zeros_like(post)
        for k, (ex, ey) in enumerate(E):
            x0, x1 = max(0, -int(ex)), min(self.nx, self.nx - int(ex))
            y0, y1 = max(0, -int(ey)), min(self.ny, self.ny - int(ey))
            streamed[y0 + ey:y1 + ey, x0 + ex:x1 + ex, k] = post[y0:y1, x0:x1, k]
        if self.halfway:
            ramp = self.config.convergence.ramp_steps
            factor = min(1.0, (self.iteration + 1) / ramp) if ramp else 1.0
            wall_u = self.config.boundaries["top"].ux * factor
            for k, missing, moving in self._links:
                reflected = post[..., OPP[k]] + cp.where(moving, 6 * W[k] * self.rho * E[k, 0] * wall_u, 0)
                streamed[..., k] = cp.where(missing, reflected, streamed[..., k])
            self._set_macroscopic(streamed)
            return
        density = streamed.sum(axis=-1)
        safe = cp.where(density > 0, density, 1.0)
        ux = (streamed * self.e[:, 0]).sum(axis=-1) / safe
        uy = (streamed * self.e[:, 1]).sum(axis=-1) / safe
        ramp = self.config.convergence.ramp_steps
        factor = min(1.0, (self.iteration + 1) / ramp) if ramp > 0 else 1.0

        def reconstruct(dst, donor, side):
            boundary = self.config.boundaries[side]
            rd = density[donor]
            wall_eq = self.equilibrium(rd, cp.full_like(rd, boundary.ux * factor),
                                       cp.full_like(rd, boundary.uy * factor))
            donor_eq = self.equilibrium(rd, ux[donor], uy[donor])
            streamed[dst] = wall_eq + (streamed[donor] - donor_eq)

        for side, dst, donor in (
            ("left", (slice(1, -1), 0), (slice(1, -1), 1)),
            ("right", (slice(1, -1), -1), (slice(1, -1), -2)),
            ("bottom", (0, slice(1, -1)), (1, slice(1, -1))),
            ("top", (-1, slice(1, -1)), (-2, slice(1, -1))),
            ("bottom", (0, 0), (1, 1)),
            ("bottom", (0, -1), (1, -2)),
            ("top", (-1, 0), (-2, 1)),
            ("top", (-1, -1), (-2, -2)),
        ):
            reconstruct(dst, donor, side)
        self._set_macroscopic(streamed)

    def _set_macroscopic(self, streamed):
        self.f = streamed
        self.rho = streamed.sum(axis=-1)
        safe = cp.where(self.rho > 1e-14, self.rho, 1.0)
        self.ux = (streamed * self.e[:, 0]).sum(axis=-1) / safe
        self.uy = (streamed * self.e[:, 1]).sum(axis=-1) / safe

    def _statistics(self):
        du2 = (self.ux - self._step_ux) ** 2 + (self.uy - self._step_uy) ** 2
        u2 = self.ux ** 2 + self.uy ** 2
        bad = (~cp.isfinite(self.f)).any() | (~cp.isfinite(self.rho)).any()
        bad |= (~cp.isfinite(self.ux)).any() | (~cp.isfinite(self.uy)).any()
        region = np.s_[:, :] if self.halfway else np.s_[1:-1, 1:-1]
        return cp.stack((du2[region].sum(), u2[region].sum(),
                         self.rho.sum(), u2.max(), self.rho.min(), bad.astype(cp.float64))).get()

    def _check_statistics(self, stats):
        failure = ""
        if stats[5]:
            failure = "Non-finite population or macroscopic field"
        elif stats[4] <= 0:
            failure = "Non-positive density"
        elif math.sqrt(stats[3]) / self.transport["cs"] > self.config.convergence.max_mach:
            failure = "Actual maximum Mach exceeded configured limit"
        if failure:
            self.host._failure = failure
            self.host.stop_reason = "diverged"
        return not failure

    def step(self):
        if self.stop_reason != "running":
            return
        # Previous post-step validation also validates the next input state.
        if not self._state_validated:
            if not self._check_statistics(self._statistics()):
                return
            self._state_validated = True
        self._step_ux, self._step_uy = self.ux, self.uy
        self._evolve()
        self.host.iteration += 1
        stats = self._statistics()
        self._last_stats = stats
        self._check_statistics(stats)
        num, den = math.sqrt(stats[0]), math.sqrt(stats[1])
        self._decision_residual = num / den if den else (0.0 if not num else math.inf)
        c = self.config.convergence
        mass_drift = abs(stats[2] - self.mass0) / self.mass0
        ready = (self.iteration >= max(c.min_iter, c.ramp_steps)
                 and self._decision_residual < c.tol and math.isfinite(self._decision_residual)
                 and not self._failure and (not c.mass_criterion or mass_drift <= c.mass_tolerance))
        self.host._stable_reports = self._stable_reports + 1 if ready else 0
        if c.steady and self._stable_reports >= c.consecutive_reports:
            self.host.stop_reason = "converged"

    def load_state(self, *, f, rho, ux, uy, iteration=0):
        """Explicit test/checkpoint loading; callers must not mutate device arrays directly."""
        shape = (self.ny, self.nx)
        for name, array in (("f", f), ("rho", rho), ("ux", ux), ("uy", uy)):
            expected = (*shape, 9) if name == "f" else shape
            if array.shape != expected:
                raise ValueError(f"{name} shape must be {expected}")
            setattr(self, name, cp.asarray(array, dtype=cp.float64).copy())
        self._step_ux, self._step_uy = self.ux.copy(), self.uy.copy()
        self.host.iteration = iteration
        self.host.stop_reason = "running"
        self.host._failure = ""
        self.host._stable_reports = 0
        self.host._last_report = None
        self.host.history.clear()
        self._last_stats = None
        self._state_validated = False

    def sync_host(self):
        for name in ("f", "rho", "ux", "uy", "_step_ux", "_step_uy"):
            setattr(self.host, name, cp.asnumpy(getattr(self, name)))

    def report(self):
        self.sync_host()
        report = self.host.report()
        self.host.history[-1]["gpu_decision_residual"] = self.decision_residual
        return report

    def run(self):
        c = self.config.convergence
        for _ in range(c.max_iter):
            self.step()
            should_report = (self._failure or self.stop_reason != "running"
                             or self.iteration % c.report_interval == 0 or self.iteration == 1)
            if should_report:
                report = self.report()
                if self.iteration >= c.max_iter and self.stop_reason == "running":
                    self.host.stop_reason = "max_iter" if c.steady else "completed_steps"
                    report.stop_reason = self.stop_reason
                    report.message = self.stop_reason
                yield report
                if self.stop_reason != "running":
                    return
        if self.stop_reason == "running":
            self.host.stop_reason = "max_iter" if c.steady else "completed_steps"
            report = self.report()
            report.stop_reason = self.stop_reason
            report.message = self.stop_reason
            yield report

    def finalize(self, save_outputs=True):
        self.sync_host()
        return self.host.finalize(save_outputs=save_outputs)

    def field_snapshot(self):
        self.sync_host()
        return self.host.field_snapshot()
