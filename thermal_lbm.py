"""D2Q5 temperature solver for double-distribution thermal LBM."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config import SolverConfig, ThermalBoundaryConfig


E_T = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int64)
W_T = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6], dtype=np.float64)
OPP_T = np.array([0, 3, 4, 1, 2], dtype=np.int64)

INCOMING_T = {
    "left": (1,),
    "right": (3,),
    "bottom": (2,),
    "top": (4,),
}
EDGE_T = {
    "left": (slice(None), 0),
    "right": (slice(None), -1),
    "bottom": (0, slice(None)),
    "top": (-1, slice(None)),
}
INSIDE_T = {
    "left": (slice(None), 1),
    "right": (slice(None), -2),
    "bottom": (1, slice(None)),
    "top": (-2, slice(None)),
}


def thermal_equilibrium(
    temperature: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
) -> np.ndarray:
    geq = np.empty((*temperature.shape, 5), dtype=np.float64)
    for i, (ex, ey) in enumerate(E_T):
        eu = ex * ux + ey * uy
        geq[..., i] = W_T[i] * temperature * (1.0 + 3.0 * eu)
    return geq


class ThermalField:
    """Passive-temperature D2Q5 field with optional Boussinesq coupling."""

    def __init__(self, cfg: SolverConfig, solid_mask: np.ndarray):
        self.cfg = cfg
        self.solid_mask = solid_mask
        self.temperature = self._initial_temperature()
        zeros = np.zeros_like(self.temperature)
        self.g = thermal_equilibrium(self.temperature, zeros, zeros)

    @property
    def periodic_x(self) -> bool:
        return (
            self.cfg.thermal_left.type == "periodic"
            and self.cfg.thermal_right.type == "periodic"
        )

    @property
    def periodic_y(self) -> bool:
        return (
            self.cfg.thermal_bottom.type == "periodic"
            and self.cfg.thermal_top.type == "periodic"
        )

    def _initial_temperature(self) -> np.ndarray:
        temperature = np.full(
            (self.cfg.ny, self.cfg.nx), self.cfg.temperature_initial, dtype=np.float64
        )
        hot = self.cfg.temperature_hot
        cold = self.cfg.temperature_cold
        if self.cfg.case_type == "natural_convection_cavity":
            x = np.linspace(0.0, 1.0, self.cfg.nx)[None, :]
            temperature[:, :] = hot + (cold - hot) * x
        elif self.cfg.case_type == "rayleigh_benard_convection":
            y = np.linspace(0.0, 1.0, self.cfg.ny)[:, None]
            temperature[:, :] = hot + (cold - hot) * y
            x = np.linspace(0.0, 2.0 * np.pi, self.cfg.nx, endpoint=False)[None, :]
            perturbation = 1e-4 * max(abs(hot - cold), 1.0) * np.sin(x) * np.sin(np.pi * y)
            temperature += perturbation
        elif self.cfg.case_type == "heated_channel_flow":
            temperature[:, :] = cold
        return temperature

    def step(self, ux: np.ndarray, uy: np.ndarray) -> None:
        geq = thermal_equilibrium(self.temperature, ux, uy)
        g_post = self.g - self.cfg.thermal_omega * (self.g - geq)
        self.g = self._stream(g_post)
        self._apply_boundaries(ux, uy)
        self.temperature = np.sum(self.g, axis=-1)

    def _stream(self, g_post: np.ndarray) -> np.ndarray:
        streamed = np.zeros_like(g_post)
        for i, (cx, cy) in enumerate(E_T):
            values = g_post[..., i]
            if self.periodic_x and self.periodic_y:
                moved = np.roll(np.roll(values, shift=cy, axis=0), shift=cx, axis=1)
                source_solid = np.roll(
                    np.roll(self.solid_mask, shift=cy, axis=0), shift=cx, axis=1
                )
                streamed[..., i] = moved
            elif self.periodic_x:
                moved = np.roll(values, shift=cx, axis=1)
                source_solid = np.roll(self.solid_mask, shift=cx, axis=1)
                y_src, y_dst = self._axis_slices(cy, self.cfg.ny)
                streamed[y_dst, :, i] = moved[y_src, :]
                solid_dest = np.zeros_like(self.solid_mask)
                solid_dest[y_dst, :] = source_solid[y_src, :]
                source_solid = solid_dest
            elif self.periodic_y:
                moved = np.roll(values, shift=cy, axis=0)
                source_solid = np.roll(self.solid_mask, shift=cy, axis=0)
                x_src, x_dst = self._axis_slices(cx, self.cfg.nx)
                streamed[:, x_dst, i] = moved[:, x_src]
                solid_dest = np.zeros_like(self.solid_mask)
                solid_dest[:, x_dst] = source_solid[:, x_src]
                source_solid = solid_dest
            else:
                y_src, y_dst = self._axis_slices(cy, self.cfg.ny)
                x_src, x_dst = self._axis_slices(cx, self.cfg.nx)
                streamed[y_dst, x_dst, i] = values[y_src, x_src]
                source_solid = np.zeros_like(self.solid_mask)
                source_solid[y_dst, x_dst] = self.solid_mask[y_src, x_src]

            bounce = source_solid & ~self.solid_mask
            streamed[bounce, i] = g_post[bounce, OPP_T[i]]

        streamed[self.solid_mask, :] = g_post[self.solid_mask][:, OPP_T]
        return streamed

    @staticmethod
    def _axis_slices(shift: int, size: int) -> tuple[slice, slice]:
        if shift > 0:
            return slice(0, size - shift), slice(shift, size)
        if shift < 0:
            return slice(-shift, size), slice(0, size + shift)
        return slice(0, size), slice(0, size)

    def _apply_boundaries(self, ux: np.ndarray, uy: np.ndarray) -> None:
        for side in ("left", "right", "bottom", "top"):
            bc = getattr(self.cfg, f"thermal_{side}")
            if bc.type == "periodic":
                continue
            if bc.type == "isothermal":
                self._isothermal(side, bc, ux, uy)
            elif bc.type == "adiabatic":
                self._adiabatic(side)
            elif bc.type == "outlet":
                self.g[EDGE_T[side]] = self.g[INSIDE_T[side]]

    def _isothermal(
        self,
        side: str,
        bc: ThermalBoundaryConfig,
        ux: np.ndarray,
        uy: np.ndarray,
    ) -> None:
        edge = EDGE_T[side]
        temperature_wall = np.full_like(self.temperature[edge], bc.temperature)
        geq_wall = thermal_equilibrium(temperature_wall, ux[edge], uy[edge])
        self.g[edge] = geq_wall

    def _adiabatic(self, side: str) -> None:
        self.g[EDGE_T[side]] = self.g[INSIDE_T[side]]

    def average_nusselt(self) -> float | None:
        delta = self.cfg.temperature_delta
        if delta <= 0.0:
            return None
        for side in ("left", "right", "bottom", "top"):
            bc = getattr(self.cfg, f"thermal_{side}")
            if bc.type != "isothermal":
                continue
            wall = self.temperature[EDGE_T[side]]
            neighbor = self.temperature[INSIDE_T[side]]
            if abs(bc.temperature - self.cfg.temperature_hot) <= 1e-12:
                return float(
                    self.cfg.characteristic_length * np.mean(np.abs(wall - neighbor)) / delta
                )
        return None
