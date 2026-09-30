"""CUDA BGK/MRT collision, NEE/halfway walls, and float64 diagnostics."""
from pathlib import Path

import gpu_environment
import cupy as cp
import numpy as np
from gpu_solver import ArrayGPUSolver


class FusedGPUSolver(ArrayGPUSolver):
    backend = "cuda-fused"

    def __init__(self, config):
        super().__init__(config)
        source = Path(__file__).with_name("kernels.cu").read_text(encoding="utf-8")
        self.module = cp.RawModule(code=source, options=("--std=c++11", "--fmad=false"))
        self.pull = self.module.get_function("pull_bgk")
        self.collide_kernel = self.module.get_function("collide_models")
        self.pull_models = self.module.get_function("pull_models")
        self.walls = self.module.get_function("nee_walls")
        self.block_reduce = self.module.get_function("diagnostic_blocks")
        self.finish_reduce = self.module.get_function("diagnostic_finish")
        self._blocks = (self.nx * self.ny + 255) // 256
        self._partial = cp.empty((self._blocks, 6), dtype=cp.float64)
        self._totals = cp.empty(6, dtype=cp.float64)
        self._f_spare = cp.empty_like(self.f)
        self._post = cp.empty_like(self.f)
        self._r_spare = cp.empty_like(self.rho)
        self._u_spare = cp.empty_like(self.ux)
        self._v_spare = cp.empty_like(self.uy)
        self._donor_u = cp.empty_like(self.ux)
        self._donor_v = cp.empty_like(self.uy)
        self._wall_velocity = cp.asarray([
            value for side in ("left", "right", "bottom", "top")
            for value in (self.config.boundaries[side].ux, self.config.boundaries[side].uy)
        ], dtype=cp.float64)

    def _evolve(self):
        grid, block = (self._blocks,), (256,)
        nx, ny = np.int32(self.nx), np.int32(self.ny)
        ramp_steps = self.config.convergence.ramp_steps
        ramp = min(1.0, (self.iteration + 1) / ramp_steps) if ramp_steps else 1.0
        if self.config.collision_model == "BGK" and not self.halfway:
            self.pull(grid, block, (self.f, self.rho, self.ux, self.uy,
                      self._f_spare, self._r_spare, self._u_spare, self._v_spare,
                      self._donor_u, self._donor_v, nx, ny, np.float64(self.transport["omega"])))
        else:
            self.collide_kernel(grid, block, (self.f, self.rho, self.ux, self.uy, self._post,
                self._m, self._minv, self._rates, nx, ny, np.float64(self.transport["omega"]),
                np.int32(self.config.collision_model == "MRT")))
            self.pull_models(grid, block, (self._post, self.rho, self._f_spare, self._r_spare,
                self._u_spare, self._v_spare, self._donor_u, self._donor_v, nx, ny,
                np.int32(self.halfway), np.float64(self.config.boundaries["top"].ux * ramp)))
        if not self.halfway:
            self.walls(grid, block, (self._f_spare, self._r_spare, self._u_spare, self._v_spare,
                       self._donor_u, self._donor_v, self._wall_velocity, np.float64(ramp), nx, ny))
        self.f, self._f_spare = self._f_spare, self.f
        self.rho, self._r_spare = self._r_spare, self.rho
        self.ux, self._u_spare = self._u_spare, self.ux
        self.uy, self._v_spare = self._v_spare, self.uy

    def _statistics(self):
        self.block_reduce((self._blocks,), (256,), (
            self.f, self.rho, self.ux, self.uy, self._step_ux, self._step_uy,
            self._partial, np.int32(self.nx), np.int32(self.ny), np.int32(self.halfway)))
        self.finish_reduce((1,), (256,), (self._partial, self._totals, np.int32(self._blocks)))
        return self._totals.get()
