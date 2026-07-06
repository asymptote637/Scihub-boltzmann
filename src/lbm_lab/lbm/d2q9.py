"""D2Q9 lattice constants and BGK helper functions."""

from __future__ import annotations

import numpy as np


C = np.array(
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
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Return the D2Q9 equilibrium distribution."""
    u_sq = ux**2 + uy**2
    feq = np.empty((9, *rho.shape), dtype=np.float64)
    for i, (cx, cy) in enumerate(C):
        cu = cx * ux + cy * uy
        feq[i] = W[i] * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u_sq)
    return feq


def macroscopic(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute density and velocity from distributions."""
    rho = np.sum(f, axis=0)
    ux = np.tensordot(C[:, 0], f, axes=(0, 0)) / rho
    uy = np.tensordot(C[:, 1], f, axes=(0, 0)) / rho
    return rho, ux, uy


def stream(f: np.ndarray) -> np.ndarray:
    """Stream distributions to neighboring nodes with periodic shifts."""
    streamed = np.empty_like(f)
    for i, (cx, cy) in enumerate(C):
        streamed[i] = np.roll(np.roll(f[i], shift=cx, axis=1), shift=cy, axis=0)
    return streamed

