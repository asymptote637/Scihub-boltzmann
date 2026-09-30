"""Qt-independent, explicitly labelled plots, usable in the desktop or exports."""
from __future__ import annotations

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from config import SimulationConfig, domain_axis
from workbench.validation import center_profiles


def draw_fields(figure, config: SimulationConfig, fields: dict, history: list,
                stride: int = 1, mode: str = "all") -> None:
    figure.clear()
    if mode == "all":
        a, b, residual, center = figure.subplots(2, 2).flat
    elif mode == "fields":
        a, b = figure.subplots(1, 2)
    elif mode == "diagnostics":
        residual, center = figure.subplots(1, 2)
    else:
        raise ValueError("Unknown plot mode")
    xo, width = domain_axis(config, "x")
    yo, height = domain_axis(config, "y")
    rows, cols = fields["rho"].shape
    extent = [xo-stride/2, xo+(cols-.5)*stride, yo-stride/2, yo+(rows-.5)*stride]
    solid = fields["solid_mask"].astype(bool)
    if mode != "diagnostics":
        # A periodic axis has no wall at zero: show all N cells centred at 0..N-1.
        # Clipping this axis to [0,N] would invent a blank half-cell at the right.
        xlim = (-.5, width-.5) if config.boundaries["left"].type == "periodic" else (0, width)
        ylim = (-.5, height-.5) if config.boundaries["bottom"].type == "periodic" else (0, height)
        _draw_maps(figure, a, b, fields, solid, extent, xlim, ylim)
    if mode == "fields":
        return
    if history:
        iterations = [r["iteration"] for r in history]
        for key, style in (("residual", "-"), ("mass_drift", "--"), ("mass_balance_error", ":")):
            values = np.array([r.get(key, np.nan) if r.get(key) is not None else np.nan for r in history])
            residual.semilogy(iterations, np.ma.masked_invalid(np.ma.masked_less_equal(values, 0)),
                              style, label=key)
        residual.legend(fontsize=8)
    residual.set(xlabel="iteration", ylabel="relative value", title="Diagnostics (log; zeros omitted)")
    residual.grid(alpha=.2)
    x = (np.arange(cols)*stride+xo) / width
    y = (np.arange(rows)*stride+yo) / height
    ux = np.where(solid, np.nan, fields["ux"])
    uy = np.where(solid, np.nan, fields["uy"])
    center.plot(y, [np.interp(.5, x, row) for row in ux], "-", label="u at x/L=0.5")
    center.plot(x, [np.interp(.5, y, col) for col in uy.T], "--", label="v at y/H=0.5")
    center.set(xlabel="y/H or x/L", ylabel="velocity (lattice)",
               title=f"Centerlines (sampling stride={stride})")
    center.legend(fontsize=8)
    center.grid(alpha=.2)


def _draw_maps(figure, a, b, fields, solid, extent, xlim, ylim) -> None:
    velocity = np.ma.masked_where(solid, fields["speed"])
    finite_speed = velocity.compressed()
    finite_speed = finite_speed[np.isfinite(finite_speed)]
    spmax = max(float(finite_speed.max()) if finite_speed.size else 0, 1e-12)
    im = a.imshow(velocity, origin="lower", extent=extent, interpolation="nearest",
                  cmap="viridis", vmin=0, vmax=spmax)
    figure.colorbar(im, ax=a, label="speed (lattice)", shrink=.85)
    a.set_title("Velocity magnitude")
    vort = np.ma.masked_invalid(np.ma.masked_where(solid, fields["vorticity"]))
    finite = vort.compressed()
    vmax = max(float(np.abs(finite).max()) if finite.size else 0, 1e-12)
    cmap = __import__("matplotlib").colormaps["RdBu_r"].copy()
    cmap.set_bad("#d1d5db")
    im = b.imshow(vort, origin="lower", extent=extent, interpolation="nearest",
                  cmap=cmap, vmin=-vmax, vmax=vmax)
    figure.colorbar(im, ax=b, label="vorticity (1 / step)", shrink=.85)
    b.set_title("Vorticity (grey = undefined)")
    for ax in (a, b):
        if solid.any():
            ax.imshow(np.ma.masked_where(~solid, solid), origin="lower", extent=extent,
                      cmap=ListedColormap(["#202938"]), vmin=0, vmax=1, interpolation="nearest")
            ax.legend(handles=[Patch(facecolor="#202938", label="solid")], fontsize=8)
        ax.set(xlabel="x (lattice)", ylabel="y (lattice)",
               xlim=xlim, ylim=ylim, aspect="equal")


def draw_validation(figure, validation: dict) -> None:
    figure.clear()
    profiles = validation.get("profiles", [])
    if not profiles:
        ax = figure.add_subplot(111)
        ax.text(.5, .5, "No applicable reference profile", ha="center", va="center")
        ax.set_axis_off()
        return
    axes = np.atleast_1d(figure.subplots(1, len(profiles)))
    for ax, p in zip(axes, profiles):
        scale = p["scale"]
        ax.plot(p["coordinate"], np.asarray(p["observed"])/scale, "-", label="LBM")
        coords = p.get("reference_coordinate", p["coordinate"])
        ax.plot(coords, np.asarray(p["reference"])/scale,
                "o" if "reference_coordinate" in p else "--", label="reference")
        ax.set(xlabel=p["coordinate_label"], ylabel=f"{p['component']} / reference speed",
               title=f"{validation['reference']} | {validation['status']}")
        ax.grid(alpha=.2)
        ax.legend()


def draw_comparison(figure, entries: list) -> None:
    figure.clear()
    axes = figure.subplots(1, 2)
    for i, (config, fields, label) in enumerate(entries):
        if config.flow.U_ref <= 0:
            raise ValueError("归一化比较需要各算例 U_ref > 0")
        mask = fields["solid_mask"].astype(bool)
        p = center_profiles(config, np.where(mask, np.nan, fields["ux"]),
                            np.where(mask, np.nan, fields["uy"]))
        for ax, axis, coord in zip(axes, ("u", "v"), ("y", "x")):
            ax.plot(p[coord], p[axis]/config.flow.U_ref, ("-", "--", "-.", ":")[i % 4], label=label)
    for ax, axis, coord in zip(axes, ("u", "v"), ("y/H", "x/L")):
        ax.set(xlabel=coord, ylabel=f"{axis} / U_ref", title="Normalized centerline comparison")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
