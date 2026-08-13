"""Boundary-condition registry for first-version D2Q9 LBM simulations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config import BoundaryConfig, SolverConfig


E = np.array(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ],
    dtype=np.int64,
)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)

INCOMING = {
    "left": np.array([1, 5, 8]),
    "right": np.array([3, 6, 7]),
    "bottom": np.array([2, 5, 6]),
    "top": np.array([4, 7, 8]),
}

SPECULAR_SOURCE = {
    "left": {1: 3, 5: 6, 8: 7},
    "right": {3: 1, 6: 5, 7: 8},
    "bottom": {2: 4, 5: 8, 6: 7},
    "top": {4: 2, 7: 6, 8: 5},
}

EDGE = {
    "left": (slice(None), 0),
    "right": (slice(None), -1),
    "bottom": (0, slice(None)),
    "top": (-1, slice(None)),
}

INSIDE = {
    "left": (slice(None), 1),
    "right": (slice(None), -2),
    "bottom": (1, slice(None)),
    "top": (-2, slice(None)),
}


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """D2Q9 equilibrium distribution with arrays shaped as (NY, NX, 9)."""
    rho = np.asarray(rho)
    ux = np.asarray(ux)
    uy = np.asarray(uy)
    uu = ux**2 + uy**2
    feq = np.empty((*rho.shape, 9), dtype=np.float64)
    for i, (ex, ey) in enumerate(E):
        eu = ex * ux + ey * uy
        feq[..., i] = W[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * uu)
    return feq


def macroscopic(f: np.ndarray, solid_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = np.sum(f, axis=-1)
    rho_safe = np.where(rho <= 1e-14, 1e-14, rho)
    ux = np.tensordot(f, E[:, 0], axes=([-1], [0])) / rho_safe
    uy = np.tensordot(f, E[:, 1], axes=([-1], [0])) / rho_safe
    if solid_mask is not None:
        ux = ux.copy()
        uy = uy.copy()
        ux[solid_mask] = 0.0
        uy[solid_mask] = 0.0
    return rho, ux, uy


def _edge_arrays(f: np.ndarray, side: str) -> tuple[np.ndarray, tuple]:
    selector = EDGE[side]
    return f[selector], selector


def no_slip_bounce_back(
    f: np.ndarray,
    side: str,
    bc: BoundaryConfig,
    cfg: SolverConfig,
    ramp: float,
) -> None:
    edge, selector = _edge_arrays(f, side)
    incoming = INCOMING[side]
    for i in incoming:
        edge[..., i] = edge[..., OPP[i]]
    f[selector] = edge


def moving_wall_bounce_back(
    f: np.ndarray,
    side: str,
    bc: BoundaryConfig,
    cfg: SolverConfig,
    ramp: float,
) -> None:
    edge, selector = _edge_arrays(f, side)
    rho_edge = np.sum(edge, axis=-1)
    ux_wall = bc.ux * ramp
    uy_wall = bc.uy * ramp
    incoming = INCOMING[side]
    for i in incoming:
        euw = E[i, 0] * ux_wall + E[i, 1] * uy_wall
        edge[..., i] = edge[..., OPP[i]] - 6.0 * W[i] * rho_edge * euw
    f[selector] = edge


def specular_reflection(
    f: np.ndarray,
    side: str,
    bc: BoundaryConfig,
    cfg: SolverConfig,
    ramp: float,
) -> None:
    edge, selector = _edge_arrays(f, side)
    source = edge.copy()
    for incoming, reflected in SPECULAR_SOURCE[side].items():
        edge[..., incoming] = source[..., reflected]
    f[selector] = edge


def mixed_bounce_specular(
    f: np.ndarray,
    side: str,
    bc: BoundaryConfig,
    cfg: SolverConfig,
    ramp: float,
) -> None:
    edge, selector = _edge_arrays(f, side)
    source = edge.copy()
    for incoming, specular in SPECULAR_SOURCE[side].items():
        bounced = source[..., OPP[incoming]]
        mirrored = source[..., specular]
        edge[..., incoming] = bc.rb * bounced + (1.0 - bc.rb) * mirrored
    f[selector] = edge


def non_equilibrium_extrapolation(
    f: np.ndarray,
    side: str,
    bc: BoundaryConfig,
    cfg: SolverConfig,
    ramp: float,
) -> None:
    selector = EDGE[side]
    neighbor_selector = INSIDE[side]
    neighbor = f[neighbor_selector]
    rho_n, ux_n, uy_n = macroscopic(neighbor)
    rho_b = np.full_like(rho_n, bc.rho if bc.rho > 0 else cfg.rho0)
    if abs(bc.ux) < 1e-15 and abs(bc.uy) < 1e-15:
        ux_b = np.zeros_like(ux_n)
        uy_b = np.zeros_like(uy_n)
    else:
        ux_b = np.full_like(ux_n, bc.ux * ramp)
        uy_b = np.full_like(uy_n, bc.uy * ramp)
    feq_b = equilibrium(rho_b, ux_b, uy_b)
    feq_n = equilibrium(rho_n, ux_n, uy_n)
    f[selector] = feq_b + (neighbor - feq_n)


def full_developed_outlet(
    f: np.ndarray,
    side: str,
    bc: BoundaryConfig,
    cfg: SolverConfig,
    ramp: float,
) -> None:
    selector = EDGE[side]
    neighbor_selector = INSIDE[side]
    neighbor = f[neighbor_selector]
    rho_n, ux_n, uy_n = macroscopic(neighbor)
    feq_n = equilibrium(rho_n, ux_n, uy_n)
    f[selector] = equilibrium(rho_n, ux_n, uy_n) + (neighbor - feq_n)


def periodic(
    f: np.ndarray,
    side: str,
    bc: BoundaryConfig,
    cfg: SolverConfig,
    ramp: float,
) -> None:
    """Periodic transport is handled by the streaming step."""


BOUNDARY_REGISTRY: dict[str, Callable[[np.ndarray, str, BoundaryConfig, SolverConfig, float], None]] = {
    "periodic": periodic,
    "no_slip_bounce_back": no_slip_bounce_back,
    "moving_wall_bounce_back": moving_wall_bounce_back,
    "specular_reflection": specular_reflection,
    "mixed_bounce_specular": mixed_bounce_specular,
    "non_equilibrium_extrapolation": non_equilibrium_extrapolation,
    "full_developed_outlet": full_developed_outlet,
}


def apply_boundaries(f: np.ndarray, cfg: SolverConfig, ramp: float) -> None:
    for side in ("left", "right", "bottom", "top"):
        bc = getattr(cfg, side)
        BOUNDARY_REGISTRY[bc.type](f, side, bc, cfg, ramp)
