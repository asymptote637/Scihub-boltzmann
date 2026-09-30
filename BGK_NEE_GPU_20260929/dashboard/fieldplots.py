"""Read-only field visualization; interpolated grids are never solver inputs."""
from __future__ import annotations

import numpy as np
from matplotlib.ticker import MaxNLocator

FLOW_MODES = {"streamlines", "overlay", "vortices"}
REGIONS = {"full": "全场", "bottom_left": "左下角", "bottom_right": "右下角",
           "top_left": "左上角", "top_right": "右上角"}


def region_bounds(data, region="full", fraction=.35):
    if region not in REGIONS or not .1 <= fraction <= .5:
        raise ValueError("流场区域或角落范围无效")
    x0, x1 = float(data["x"][0]), float(data["x"][-1])
    y0, y1 = float(data["y"][0]), float(data["y"][-1])
    dx, dy = (x1 - x0) * fraction, (y1 - y0) * fraction
    if region.endswith("left"):
        x1 = x0 + dx
    elif region.endswith("right"):
        x0 = x1 - dx
    if region.startswith("bottom"):
        y1 = y0 + dy
    elif region.startswith("top"):
        y0 = y1 - dy
    return x0, x1, y0, y1


def streamline_grid(data, bounds, limit=257):
    """Use actual coordinates, including an irregular last preview interval."""
    x, y = np.asarray(data["x"]), np.asarray(data["y"])
    x0, x1, y0, y1 = bounds
    if limit < 2 or not (x[0] <= x0 < x1 <= x[-1] and y[0] <= y0 < y1 <= y[-1]):
        raise ValueError("流线范围超出原始数据")
    if any(not np.isfinite(data[key]).all() for key in ("ux", "uy", "x", "y")):
        raise ValueError("速度或坐标含非有限值，不能绘制可靠流线")
    if not ((np.diff(x) > 0).all() and (np.diff(y) > 0).all()):
        raise ValueError("流场坐标必须严格递增")
    xx = np.linspace(x0, x1, min(limit, max(2, np.count_nonzero((x >= x0) & (x <= x1)))))
    yy = np.linspace(y0, y1, min(limit, max(2, np.count_nonzero((y >= y0) & (y <= y1)))))
    # Restrict source rows first so a corner view preserves its native resolution.
    low = max(0, np.searchsorted(y, y0, side="right") - 1)
    high = min(len(y), np.searchsorted(y, y1, side="left") + 1)
    result = []
    for key in ("ux", "uy"):
        rows = np.array([np.interp(xx, x, row) for row in data[key][low:high]])
        result.append(np.array([np.interp(yy, y[low:high], col) for col in rows.T]).T)
    return xx, yy, *result


def draw_field(figure, data, mode, *, speed, caption="", region="full", density=1.5, fraction=.35):
    if mode not in FLOW_MODES | {"speed", "ux", "uy", "uz", "rho", "T", "vorticity"}:
        raise ValueError("未知流场图类型")
    if mode not in {"rho","T","vorticity"} and (not speed or not np.isfinite(speed) or speed <= 0):
        raise ValueError("没有有效的顶盖速度，不能归一化流场")
    if not np.isfinite(density) or not .6 <= density <= 3:
        raise ValueError("流线疏密必须在 0.6 到 3 之间")
    region_bounds(data, region, fraction)
    if mode in {"uz","T"} and mode not in data:
        raise ValueError("此算例未计算该场量")
    flow_data = dict(data,ux=data.get("plot_ux",data["ux"]),uy=data.get("plot_uy",data["uy"]))
    lengths = data.get("axis_lengths",dict(x=len(data["x"]),y=len(data["y"])))
    # The axes are x/Lx and y/Ly. Transport vectors in those coordinates are
    # (ux/Lx, uy/Ly); using raw velocities would distort rectangular streamlines.
    stream_data = dict(flow_data,ux=flow_data["ux"]/lengths["x"],uy=flow_data["uy"]/lengths["y"])
    figure.clear()
    figure.set_layout_engine("compressed")
    if mode == "vortices":
        axes = figure.subplots(2, 2)
        panels = [(axes[0, 0], "speed", "full", "速度模 |u| / U"),
                  (axes[0, 1], "streamlines", "full", "全场流线"),
                  (axes[1, 0], "streamlines", "bottom_left", "左下角流线"),
                  (axes[1, 1], "streamlines", "bottom_right", "右下角流线")]
    else:
        panels = [(figure.add_subplot(111), mode, region, REGIONS[region])]
    for axis, kind, location, title in panels:
        bounds = region_bounds(data, location, fraction)
        if kind != "streamlines":
            key = "speed" if kind == "overlay" else kind
            if key=="speed":
                values = np.sqrt(data["ux"]**2+data["uy"]**2+data.get("uz",0)**2)/speed
            elif key=="vorticity":
                values = np.gradient(flow_data["uy"],data["x"]*lengths["x"],axis=1)-np.gradient(flow_data["ux"],data["y"]*lengths["y"],axis=0)
            else:
                values = data[key]/(1 if key in {"rho","T"} else speed)
            if "solid_mask" in data: values = np.ma.masked_where(data["solid_mask"],values)
            mesh = axis.pcolormesh(data["x"], data["y"], np.ma.masked_invalid(values), shading="auto",
                                   cmap="viridis" if key in ("speed", "rho") else "coolwarm")
            figure.colorbar(mesh, ax=axis, shrink=.9, pad=.04, panchor=False)
        if kind in ("streamlines", "overlay"):
            x, y, u, v = streamline_grid(stream_data, bounds)
            if "solid_mask" in data:
                mask = np.array([np.interp(x,data["x"],row.astype(float)) for row in data["solid_mask"]])
                mask = np.array([np.interp(y,data["y"],col) for col in mask.T]).T > .1
                u,v = np.ma.masked_where(mask,u),np.ma.masked_where(mask,v)
            if np.any(u) or np.any(v):
                axis.streamplot(x, y, u, v, density=density, linewidth=.65, arrowsize=.8,
                                color="#ffffff" if kind == "overlay" else "#23594b")
            else:
                axis.text(.5, .5, "零速度场", ha="center", va="center", transform=axis.transAxes)
        axis.set(xlim=bounds[:2], ylim=bounds[2:], aspect=lengths["y"]/lengths["x"], xlabel=data.get("xlabel","x / Lx"), ylabel=data.get("ylabel","y / Ly"), title=title)
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.yaxis.set_major_locator(MaxNLocator(4))
    if caption:
        figure.suptitle(caption, fontsize=9, wrap=True)
    return [panel[0] for panel in panels]
