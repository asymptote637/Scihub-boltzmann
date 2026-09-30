"""Post-run checks restricted to the physical assumptions of each reference."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

from config import SimulationConfig, domain_axis, lattice_transport


def _zero(value: float) -> bool:
    return abs(value) < 1e-14


def _stationary(bc, kind: str = "halfway_bounce_back") -> bool:
    return bc.type == kind and _zero(bc.ux) and _zero(bc.uy)


def reference_kind(config: SimulationConfig) -> tuple[str, str]:
    """Match actual geometry/BCs; a case label alone never enables acceptance."""
    c, b, f = config, config.boundaries, config.flow
    if c.grid.obstacle_type != "none" or c.case_type == "cylinder_flow":
        return "", "含障碍物：当前解析/方腔基准不适用。"
    if not _zero(f.body_force_y):
        return "", "横向体力不符合当前基准假设。"
    periodic_x = all(b[s].type == "periodic" for s in ("left", "right"))
    bottom = _stationary(b["bottom"])
    top_static = _stationary(b["top"])
    top_moving = b["top"].type == "halfway_moving_wall" and _zero(b["top"].uy) and b["top"].ux > 0
    if periodic_x and bottom:
        if top_moving and _zero(f.body_force_x):
            return "couette", "稳态平行板 Couette；半格壁面，流向周期。"
        if top_static and not _zero(f.body_force_x):
            return "force", "稳态恒加速度 Poiseuille；半格壁面，流向周期。"
    no_force = _zero(f.body_force_x)
    if bottom and top_static and no_force and all(
            b[s].type == "pressure_zou_he" and _zero(b[s].uy) for s in ("left", "right")):
        if not _zero(b["left"].rho - b["right"].rho):
            return "pressure", "低 Ma 常密度 Poiseuille 近似；压力 p=ρ/3，入口/出口间距 NX−1。"
    if bottom and top_moving and no_force and all(_stationary(b[s]) for s in ("left", "right")):
        nu = lattice_transport(c)["nu_lattice"]
        actual_re = b["top"].ux * c.grid.NY / nu
        if c.grid.NX == c.grid.NY and abs(actual_re-100) < 1e-6:
            return "cavity", "Ghia et al. Re=100 中心线二次转录；原论文表格未重新目视核对。"
    return "", "实际边界、体力、网格或 Re 不满足现有参考解的假设。"


def center_profiles(config: SimulationConfig, ux: np.ndarray, uy: np.ndarray) -> dict:
    xo, width = domain_axis(config, "x")
    yo, height = domain_axis(config, "y")
    x = (np.arange(config.grid.NX) + xo) / width
    y = (np.arange(config.grid.NY) + yo) / height
    return {"x": x, "y": y,
            "u": np.array([np.interp(.5, x, row) for row in ux]),
            "v": np.array([np.interp(.5, y, col) for col in uy.T])}


def evaluate(config: SimulationConfig, summary: dict, fields: dict) -> dict:
    kind, note = reference_kind(config)
    report = dict(schema_version=1, reference=kind, reference_note=note, status="not_applicable",
                  stop_reason=summary.get("stop_reason", "unknown"), metrics={}, thresholds={},
                  checks={}, profiles=[], scope="训练基准检查，不证明网格无关或任意工况精度。")
    if not kind:
        return report
    rho, ux, uy, solid = (fields[k] for k in ("rho", "ux", "uy", "solid_mask"))
    expected_shape = (config.grid.NY, config.grid.NX)
    if any(a.shape != expected_shape for a in (rho, ux, uy, solid)):
        raise ValueError("结果场尺寸与配置不一致")
    if np.any(solid):
        report.update(reference="", reference_note="结果包含固体掩膜，当前基准不适用。")
        return report
    if not all(np.isfinite(a).all() for a in (rho, ux, uy)) or np.min(rho) <= 0:
        report.update(status="failed", checks={"finite_positive_fields": False})
        return report
    nu = lattice_transport(config)["nu_lattice"]
    h = float(config.grid.NY)
    y = np.arange(config.grid.NY) + .5
    diag = summary.get("final_diagnostics", {})
    checks, metrics, thresholds = report["checks"], report["metrics"], report["thresholds"]
    checks["converged"] = summary.get("stop_reason") == "converged" and summary.get("converged") is True
    checks["finite_positive_fields"] = True
    metrics["actual_max_mach"] = float(np.hypot(ux, uy).max() * np.sqrt(3))
    thresholds["actual_max_mach"] = min(config.convergence.max_mach, .1)
    checks["actual_max_mach"] = metrics["actual_max_mach"] <= thresholds["actual_max_mach"]
    conservation = "mass_balance_error" if kind == "pressure" else "mass_drift"
    metrics[conservation] = diag.get(conservation)
    thresholds[conservation] = 1e-8 if kind == "pressure" else 1e-9
    checks[conservation] = (metrics[conservation] is not None
                            and np.isfinite(metrics[conservation])
                            and metrics[conservation] < thresholds[conservation])
    if kind in {"couette", "force", "pressure"}:
        if kind == "couette":
            exact = config.boundaries["top"].ux * y / h
        else:
            acceleration = config.flow.body_force_x
            if kind == "pressure":
                left, right = config.boundaries["left"].rho, config.boundaries["right"].rho
                acceleration = (left-right) / (3*(config.grid.NX-1)*((left+right)/2))
            exact = acceleration * y * (h-y) / (2*nu)
        observed = ux.mean(axis=1) if kind != "pressure" else center_profiles(config, ux, uy)["u"]
        metrics["relative_L2"] = float(np.linalg.norm(observed-exact)/np.linalg.norm(exact))
        thresholds["relative_L2"] = .01
        checks["relative_L2"] = metrics["relative_L2"] < .01
        # A mean profile alone could hide spurious transverse or streamwise modes.
        metrics["transverse_max"] = float(np.abs(uy).max())
        thresholds["transverse_max"] = 1e-4 if kind == "pressure" else 1e-8
        checks["transverse_max"] = metrics["transverse_max"] < thresholds["transverse_max"]
        if kind != "pressure":
            metrics["streamwise_variation"] = float(np.max(np.abs(ux-observed[:, None])) / np.max(np.abs(exact)))
            thresholds["streamwise_variation"] = .01
            checks["streamwise_variation"] = metrics["streamwise_variation"] < .01
        report["profiles"] = [dict(component="u", coordinate=(y/h).tolist(),
                                   observed=observed.tolist(), reference=exact.tolist(),
                                   scale=float(np.max(np.abs(exact))), coordinate_label="y / H")]
    else:
        p = center_profiles(config, ux, uy)
        lid = config.boundaries["top"].ux
        ref_root = Path(__file__).resolve().parents[1] / "references"
        hashes = {}
        for axis, coordinate, endpoints in (("u", p["y"], [0., 1.]), ("v", p["x"], [0., 0.])):
            path = ref_root / f"ghia_re100_{axis}.csv"
            data = np.loadtxt(path, delimiter=",", skiprows=1)
            coords = np.r_[0., coordinate, 1.]
            values = np.r_[endpoints[0], p[axis]/lid, endpoints[1]]
            prediction = np.interp(data[:, 0], coords, values)
            interior = (data[:, 0] > 0) & (data[:, 0] < 1)
            metric = f"{axis}_normalized_RMSE"
            metrics[metric] = float(np.sqrt(np.mean((prediction[interior]-data[interior, 1])**2)))
            thresholds[metric] = .02
            checks[metric] = metrics[metric] < .02
            order = np.argsort(data[:, 0])
            report["profiles"].append(dict(component=axis, coordinate=coords.tolist(),
                observed=(values*lid).tolist(), reference_coordinate=data[order, 0].tolist(),
                reference=(data[order, 1]*lid).tolist(), scale=lid,
                coordinate_label="y / H" if axis == "u" else "x / L"))
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        report["reference_sha256"] = hashes
    report["checks"] = {k: bool(v) for k, v in checks.items()}
    report["status"] = ("passed" if all(checks.values()) else
                        "not_converged" if not checks["converged"] else "failed")
    return report
