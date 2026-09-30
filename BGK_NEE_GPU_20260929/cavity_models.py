"""Workbench cavity models; the archived CPU source remains a frozen reference.

MRT uses the orthogonal D2Q9 raw-moment basis of Lallemand & Luo (2000),
https://doi.org/10.1103/PhysRevE.61.6546, in this project's E ordering.
Halfway walls lie half a link outside the array; all stored nodes are fluid.
"""
import copy
from dataclasses import dataclass
import math

import numpy as np

from reference import CPUSolver
from boundary_conditions import E, W, OPP
from config import SimulationConfig, validate_config

COLLISIONS = {"BGK": "BGK · 单松弛", "MRT": "MRT · 多松弛"}
BOUNDARIES = {"nee": "NEE · 非平衡外推", "halfway": "HBB · 半格点反弹"}
M = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [-4, -1, -1, -1, -1, 2, 2, 2, 2],
    [4, -2, -2, -2, -2, 1, 1, 1, 1],
    [0, 1, 0, -1, 0, 1, -1, -1, 1],
    [0, -2, 0, 2, 0, 1, -1, -1, 1],
    [0, 0, 1, 0, -1, 1, 1, -1, -1],
    [0, 0, -2, 0, 2, 1, 1, -1, -1],
    [0, 1, -1, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, -1, 1, -1],
], dtype=np.float64)
MINV = M.T / np.sum(M * M, axis=1)


@dataclass
class CavityConfig(SimulationConfig):
    mrt_s_e: float = 1.64
    mrt_s_eps: float = 1.54
    mrt_s_q: float = 1.9
    halfway_corner_policy: str = "horizontal_wall_priority"


def boundary_key(config):
    types = {b.type for b in config.boundaries.values()}
    if types == {"non_equilibrium_extrapolation"}:
        return "nee"
    if types and types <= {"halfway_bounce_back", "halfway_moving_wall"}:
        return "halfway"
    raise ValueError("Use four NEE walls or four halfway walls; mixed walls are unsupported")


def require_supported(config):
    if config.case_type != "lid_driven_cavity" or config.collision_model not in COLLISIONS:
        raise ValueError("Only BGK/MRT lid-driven cavities are supported")
    if set(config.boundaries) != {"left", "right", "bottom", "top"}:
        raise ValueError("Four named cavity walls are required")
    boundary = boundary_key(config)
    if config.flow.body_force_x or config.flow.body_force_y or config.grid.obstacle_type != "none":
        raise ValueError("Body forces and obstacles are not supported")
    if config.flow.initial_velocity != "rest":
        raise ValueError("Production initialization must be at rest")
    if boundary == "halfway":
        # Limit moving walls to the lid so the double-hit corner convention is unique.
        if any(config.boundaries[s].ux or config.boundaries[s].uy for s in ("left", "right", "bottom")):
            raise ValueError("Halfway cavity side and bottom walls must be stationary")
        if config.boundaries["top"].uy:
            raise ValueError("Halfway lid velocity must be tangential")
        if getattr(config, "halfway_corner_policy", "horizontal_wall_priority") != "horizontal_wall_priority":
            raise ValueError("Unsupported halfway corner policy")
    for key, default in (("mrt_s_e", 1.64), ("mrt_s_eps", 1.54), ("mrt_s_q", 1.9)):
        value = getattr(config, key, default)
        if not math.isfinite(value) or not 0 < value < 2:
            raise ValueError(f"{key} must be finite and strictly between 0 and 2")
    # Reuse archived physical/grid validation without falsely passing MRT into BGK.
    base = copy.deepcopy(config)
    base.collision_model = "BGK"
    errors, _, _ = validate_config(base)
    if errors:
        raise ValueError("; ".join(errors))


def relaxation_rates(config, omega):
    return np.array([0, getattr(config, "mrt_s_e", 1.64), getattr(config, "mrt_s_eps", 1.54),
                     0, getattr(config, "mrt_s_q", 1.9), 0, getattr(config, "mrt_s_q", 1.9),
                     omega, omega], dtype=np.float64)


def halfway_links(nx, ny):
    """Missing incoming links and wall identity, including double-hit corners."""
    y, x = np.indices((ny, nx))
    for k, (ex, ey) in enumerate(E):
        sx, sy = x - ex, y - ey
        missing = (sx < 0) | (sx >= nx) | (sy < 0) | (sy >= ny)
        # Horizontal wall takes precedence for diagonal rays hitting a corner.
        moving = missing & (sy >= ny)
        yield k, missing, moving


class CavityCPUSolver(CPUSolver):
    def __init__(self, config):
        require_supported(config)
        base = copy.deepcopy(config)
        base.collision_model = "BGK"
        super().__init__(base)
        self.config = copy.deepcopy(config)
        self.halfway = boundary_key(config) == "halfway"
        self.rates = relaxation_rates(config, self.omega)
        self.links = list(halfway_links(self.nx, self.ny)) if self.halfway else []

    def collide(self):
        if self.config.collision_model == "BGK":
            return super().collide()
        delta = self.f - self.equilibrium(self.rho, self.ux, self.uy)
        return self.f - ((delta @ M.T) * self.rates) @ MINV.T

    def apply_boundaries(self, f_post):
        if not self.halfway:
            return super().apply_boundaries(f_post)
        wall = self._ramped_boundary(self.config.boundaries["top"])
        for k, missing, moving in self.links:
            self.f[..., k][missing] = f_post[..., OPP[k]][missing]
            # Incoming direction convention: + 2*w*rho*(e_in.u_wall)/cs^2.
            self.f[..., k][moving] += 6 * W[k] * self.rho[moving] * E[k, 0] * wall.ux
