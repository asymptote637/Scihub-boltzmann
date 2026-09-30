"""Boundary-condition registry and implementations for the D2Q9 solver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import numpy as np

from config import BoundaryConfig


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


SIDE_SLICES = {
    "left": (slice(None), 0),
    "right": (slice(None), -1),
    "bottom": (0, slice(None)),
    "top": (-1, slice(None)),
}


INSIDE_SLICES = {
    "left": (slice(None), 1),
    "right": (slice(None), -2),
    "bottom": (1, slice(None)),
    "top": (-2, slice(None)),
}


NORMAL_REFLECTION = {
    "left": ((1, 3), (5, 7), (8, 6)),
    "right": ((3, 1), (6, 8), (7, 5)),
    "bottom": ((2, 4), (5, 7), (6, 8)),
    "top": ((4, 2), (7, 5), (8, 6)),
}


@dataclass
class MacroFields:
    rho: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    post_collision: np.ndarray | None = None


BoundaryFunction = Callable[[np.ndarray, str, BoundaryConfig, MacroFields, Callable], None]
REGISTRY: dict[str, BoundaryFunction] = {}


def register(name: str) -> Callable[[BoundaryFunction], BoundaryFunction]:
    def decorator(func: BoundaryFunction) -> BoundaryFunction:
        REGISTRY[name] = func
        return func

    return decorator


def apply_boundary(
    f: np.ndarray,
    side: str,
    bc: BoundaryConfig,
    macros: MacroFields,
    equilibrium: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> None:
    bc_type = fallback_boundary_type(bc.type)
    REGISTRY[bc_type](f, side, bc, macros, equilibrium)


def fallback_boundary_type(name: str) -> str:
    """Compatibility entry point: unsupported names now fail explicitly."""

    if name in REGISTRY:
        return name
    raise ValueError(f"Boundary {name!r} is not implemented.")


@register("halfway_bounce_back")
@register("halfway_moving_wall")
def halfway_wall(f, side, bc, macros, equilibrium) -> None:
    """Reflect post-collision populations at the same fluid node (wall at 1/2).

    At rectangular corners the top/bottom tangential lid correction is applied
    last; each lost diagonal is still reflected only into its opposite slot.
    """
    del equilibrium
    if macros.post_collision is None:
        raise ValueError("Halfway bounce-back requires post-collision populations.")
    idx = SIDE_SLICES[side]
    for target, source in NORMAL_REFLECTION[side]:
        correction = 0.0
        if bc.type == "halfway_moving_wall":
            correction = 6 * W[target] * macros.rho[idx] * (E[target, 0] * bc.ux + E[target, 1] * bc.uy)
        f[idx + (target,)] = macros.post_collision[idx + (source,)] + correction


@register("velocity_zou_he")
@register("pressure_zou_he")
def zou_he(f, side, bc, macros, equilibrium) -> None:
    """On-node Zou/He reconstruction, rotated to each inward normal (no force).

    Pressure sets rho and tangential velocity; the normal velocity is inferred.
    """
    del macros, equilibrium
    nx, ny = {"left": (1, 0), "right": (-1, 0), "bottom": (0, 1), "top": (0, -1)}[side]
    tx, ty = -ny, nx
    local = {(int(ex * nx + ey * ny), int(ex * tx + ey * ty)): i
             for i, (ex, ey) in enumerate(E)}
    i0, i1, i2, i3, i4, i5, i6, i7, i8 = [local[tuple(e)] for e in E]
    a = f[SIDE_SLICES[side]].copy()
    un, ut = bc.ux * nx + bc.uy * ny, bc.ux * tx + bc.uy * ty
    known = a[..., i0] + a[..., i2] + a[..., i4] + 2 * (a[..., i3] + a[..., i6] + a[..., i7])
    if bc.type == "velocity_zou_he":
        rho = known / (1 - un)
    else:
        rho = bc.rho
        un = 1 - known / rho
    a[..., i1] = a[..., i3] + 2 / 3 * rho * un
    a[..., i5] = a[..., i7] + 0.5 * (a[..., i4] - a[..., i2]) + rho * un / 6 + rho * ut / 2
    a[..., i8] = a[..., i6] + 0.5 * (a[..., i2] - a[..., i4]) + rho * un / 6 - rho * ut / 2
    f[SIDE_SLICES[side]] = a


@register("periodic")
def periodic_boundary(
    f: np.ndarray,
    side: str,
    bc: BoundaryConfig,
    macros: MacroFields,
    equilibrium: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> None:
    del f, side, bc, macros, equilibrium
    return


@register("no_slip_bounce_back")
def no_slip_bounce_back(
    f: np.ndarray,
    side: str,
    bc: BoundaryConfig,
    macros: MacroFields,
    equilibrium: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> None:
    del bc, macros, equilibrium
    idx = SIDE_SLICES[side]
    old = f[idx].copy()
    for target, source in NORMAL_REFLECTION[side]:
        f[idx + (target,)] = old[..., source]


@register("moving_wall_bounce_back")
def moving_wall_bounce_back(
    f: np.ndarray,
    side: str,
    bc: BoundaryConfig,
    macros: MacroFields,
    equilibrium: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> None:
    del equilibrium
    idx = SIDE_SLICES[side]
    old = f[idx].copy()
    rho_wall = macros.rho[idx]
    for target, source in NORMAL_REFLECTION[side]:
        euw = E[target, 0] * bc.ux + E[target, 1] * bc.uy
        f[idx + (target,)] = old[..., source] + 6.0 * W[target] * rho_wall * euw


@register("non_equilibrium_extrapolation")
def non_equilibrium_extrapolation(
    f: np.ndarray,
    side: str,
    bc: BoundaryConfig,
    macros: MacroFields,
    equilibrium: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> None:
    bidx = SIDE_SLICES[side]
    nidx = INSIDE_SLICES[side]
    rho_b = macros.rho[nidx]
    ux_b = np.full_like(rho_b, bc.ux)
    uy_b = np.full_like(rho_b, bc.uy)
    feq_b = equilibrium(rho_b, ux_b, uy_b)
    feq_n = equilibrium(macros.rho[nidx], macros.ux[nidx], macros.uy[nidx])
    f[bidx] = feq_b + (f[nidx] - feq_n)


@register("full_developed_outlet")
def full_developed_outlet(
    f: np.ndarray,
    side: str,
    bc: BoundaryConfig,
    macros: MacroFields,
    equilibrium: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> None:
    del bc
    bidx = SIDE_SLICES[side]
    nidx = INSIDE_SLICES[side]
    feq_b = equilibrium(macros.rho[nidx], macros.ux[nidx], macros.uy[nidx])
    feq_n = equilibrium(macros.rho[nidx], macros.ux[nidx], macros.uy[nidx])
    f[bidx] = feq_b + (f[nidx] - feq_n)


def apply_solid_bounce_back(f_streamed: np.ndarray, f_post: np.ndarray, solid_mask: np.ndarray) -> None:
    """Link-wise bounce-back for fluid nodes adjacent to solid nodes."""

    if not np.any(solid_mask):
        return

    ny, nx = solid_mask.shape
    for i, (ex, ey) in enumerate(E):
        if i == 0:
            continue
        source = f_post[..., i]
        bounced = f_streamed[..., OPP[i]]

        ys0 = max(0, -ey)
        ys1 = min(ny, ny - ey)
        xs0 = max(0, -ex)
        xs1 = min(nx, nx - ex)
        fluid_slice = (slice(ys0, ys1), slice(xs0, xs1))
        neigh_slice = (slice(ys0 + ey, ys1 + ey), slice(xs0 + ex, xs1 + ex))
        hit_solid = (~solid_mask[fluid_slice]) & solid_mask[neigh_slice]
        bounced_view = bounced[fluid_slice]
        source_view = source[fluid_slice]
        bounced_view[hit_solid] = source_view[hit_solid]

    f_streamed[solid_mask, :] = f_post[solid_mask][:, OPP]
