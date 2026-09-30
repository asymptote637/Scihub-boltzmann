"""Result readers and explicitly qualified comparisons; no solver execution."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from .store import BACKEND_NAMES, STATUS_NAMES, encode, now, read_json

PREVIEW_BYTES_LIMIT = 16 * 1024 * 1024


class PreviewUnavailable(ValueError):
    """An optional live preview could not be read; final results remain strict."""


def history(folder):
    folder = Path(folder)
    records, warnings = [], []
    source = folder / "progress.jsonl"
    if source.is_file():
        with source.open("rb") as stream:
            if source.stat().st_size > 64 * 1024 * 1024:
                stream.seek(-64 * 1024 * 1024, 2)
                stream.readline()
                warnings.append("曲线仅含日志末尾 64 MB；完整记录仍保留在数据目录")
            for number, raw in enumerate(stream, 1):
                if not raw.endswith(b"\n"):
                    warnings.append("最后一条记录尚未写完")
                    break
                try:
                    row = json.loads(raw)
                    if not isinstance(row, dict) or not isinstance(row.get("iteration"), (int, float)):
                        raise ValueError("missing iteration")
                    records.append(row)
                except (ValueError, TypeError):
                    records.append({"iteration": None})
                    warnings.append(f"日志第 {number} 条损坏，曲线保留断点")
    elif (folder / "residual_history.csv").is_file():
        with (folder / "residual_history.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                parsed = {}
                for key, value in row.items():
                    try:
                        parsed[key] = float(value) if value else None
                    except (TypeError, ValueError):
                        parsed[key] = None
                records.append(parsed)
    else:
        warnings.append("尚无诊断曲线记录")
    return records, warnings


def load_fields(folder, preview=False):
    folder = Path(folder)
    source = folder / "results.npz"
    is_preview = False
    if not source.is_file() and preview:
        source = folder / "preview.npz"
        is_preview = True
    if not source.is_file():
        return None
    if is_preview:
        try:
            # Release the Windows file handle before ZIP validation and decompression.
            with source.open("rb") as stream:
                snapshot = stream.read(PREVIEW_BYTES_LIMIT + 1)
            if len(snapshot) > PREVIEW_BYTES_LIMIT:
                raise ValueError("预览文件超过 16 MB")
            return _load_field_archive(io.BytesIO(snapshot), folder, True)
        except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile) as error:
            raise PreviewUnavailable(f"流场预览暂不可读，等待下次刷新：{error}") from error
    return _load_field_archive(source, folder, False)


def _load_field_archive(source, folder, is_preview):
    with zipfile.ZipFile(source) as archive:
        expanded = sum(item.file_size for item in archive.infolist())
        if is_preview and expanded > PREVIEW_BYTES_LIMIT:
            raise ValueError("预览解压后超过 16 MB")
        if not is_preview and expanded > 2 * 1024 ** 3:
            raise ValueError("流场解压后超过 2 GB，当前交互分析不加载此文件")
    if is_preview:
        source.seek(0)
    with np.load(source, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in ("rho", "ux", "uy")}
        shape = data["rho"].shape
        for key in ("uz","T","solid_mask"):
            if key in archive: data[key] = np.asarray(archive[key])
        if len(shape) not in (2,3) or min(shape) < 2 or any(
                array.shape != shape or array.dtype.kind not in (("b","i","u") if key=="solid_mask" else ("f",))
                for key,array in data.items()):
            raise ValueError("NPZ 中密度和速度场的形状或数值类型不一致")
        if "solid_mask" in data:
            if not np.isin(data["solid_mask"],[0,1]).all(): raise ValueError("固体掩膜必须只有 0 和 1")
            data["solid_mask"] = data["solid_mask"].astype(bool)
        if len(shape)==3 and "uz" not in data:
            raise ValueError("三维流场缺少 uz")
        for axis, size in zip("xyz",reversed(shape)):
            data[axis] = np.asarray(archive[axis]) if axis in archive else np.linspace(0,1,size)
            coords = data[axis]
            if coords.shape != (size,) or not np.isfinite(coords).all() or not (np.diff(coords) > 0).all():
                raise ValueError("流场坐标无效")
        data["preview_iteration"] = int(archive["iteration"]) if "iteration" in archive else None
    config = read_json(folder / "config.json", {}) or read_json(folder / "run_request.json", {}).get("config", {})
    if not is_preview and config:
        expected = (config.get("grid", {}).get("NY"), config.get("grid", {}).get("NX"))
        if len(shape)==3: expected = (config.get("grid",{}).get("NZ"),)+expected
        if expected != shape:
            raise ValueError("NPZ 网格形状与 config.json 不一致")
    data["preview"] = is_preview
    data["finite"] = all(np.isfinite(data[k]).all() for k in ("rho", "ux", "uy", "uz", "T") if k in data)
    on_node = int(config.get("boundary_scheme_default")=="non_equilibrium_extrapolation")
    data["axis_lengths"] = {axis:config.get("grid",{}).get("N"+axis.upper(),size)-on_node for axis,size in zip("xyz",reversed(shape))}
    if config.get('research',{}).get('boundary')=='inlet_outlet':
        data['axis_lengths']['x']=config['grid']['NX']-1
    return data


def slice_fields(data,plane="xy",position=.5):
    """Nearest coordinate plane, with full-vector speed and correct in-plane flow."""
    if data["rho"].ndim==2:
        return data
    if plane not in {"xy","xz","yz"} or not 0<=position<=1:
        raise ValueError("三维截面或位置无效")
    fixed = ({"xy":"z","xz":"y","yz":"x"})[plane]
    dimension = "zyx".index(fixed)
    index = int(np.argmin(abs(data[fixed]-position)))
    result = dict(data)
    for key in ("rho","ux","uy","uz","T","solid_mask","difference"):
        if key in data: result[key] = np.take(data[key],index,axis=dimension)
    result["x"],result["y"] = data[plane[0]],data[plane[1]]
    result["plot_ux"],result["plot_uy"] = result["u"+plane[0]],result["u"+plane[1]]
    result["xlabel"],result["ylabel"] = plane[0]+" / L"+plane[0],plane[1]+" / L"+plane[1]
    result["axis_lengths"] = dict(x=data.get("axis_lengths",{}).get(plane[0],1),y=data.get("axis_lengths",{}).get(plane[1],1))
    result["slice_caption"] = f" · {plane} 截面，{fixed}/L{fixed}={data[fixed][index]:.4g}（最近格点）"
    return result


def curve(records, metric, x_key="iteration"):
    def number(value):
        try:
            result = float(value)
            return result if math.isfinite(result) else np.nan
        except (ValueError, TypeError):
            return np.nan
    return np.asarray([number(r.get(x_key)) for r in records]), np.asarray([number(r.get(metric)) for r in records])


def centerlines(fields, speed):
    if not speed or not math.isfinite(speed):
        raise ValueError("没有有效 U_ref，不能归一化速度")
    fields = slice_fields(fields)
    x, y = fields["x"], fields["y"]
    if not (x[0] <= 0.5 <= x[-1] and y[0] <= 0.5 <= y[-1]):
        raise ValueError("坐标范围不包含中截面 0.5")
    u = np.array([np.interp(0.5, x, row) for row in fields["ux"]]) / speed
    v = np.array([np.interp(0.5, y, col) for col in fields["uy"].T]) / speed
    return dict(y=y, u=u, x=x, v=v)


def load_case(case, fields=False, preview=False):
    result = dict(case)
    result["records"], result["warnings"] = history(case["data_dir"])
    try:
        result["fields"] = load_fields(case["data_dir"], preview) if fields else None
    except PreviewUnavailable as error:
        result["fields"] = None
        result["warnings"].append(str(error))
    if case.get("metrics", {}).get("preview_warning"):
        result["warnings"].append("预览更新曾被跳过：" + case["metrics"]["preview_warning"])
    if fields and result["fields"] is None:
        result["warnings"].append("没有可读取的流场文件")
    return result


def physical_signature(case):
    config = case.get("config", {})
    def immutable(item):
        if isinstance(item, dict):
            return tuple(sorted((key, immutable(value)) for key, value in item.items()))
        if isinstance(item, list):
            return tuple(immutable(value) for value in item)
        return item
    # A collision model changes the numerical scheme, not physical node locations.
    # BGK/MRT on the same geometry can be compared; NEE/HBB nodes cannot.
    physical = {key: config.get(key) for key in ("case_type", "boundary_scheme_default", "grid", "flow", "boundaries")}
    physical["grid"] = dict(config.get("grid",{}),NZ=config.get("grid",{}).get("NZ",1))
    p = config.get("research") or {}
    physical["research"] = dict(scenario=p.get("scenario","cavity"),force_z=p.get("force_z",0))
    keys = []
    if p.get("scenario") in {"natural_convection","mixed_convection","rayleigh_benard"}: keys += ["rayleigh","prandtl"]
    if p.get('scenario')=='rayleigh_benard':keys += ['thermal_perturbation']
    if p.get("scenario")=="thermal_wave": keys += ["diffusivity"]
    if p.get("scenario") in {"obstacle","cylinder_wake"}:
        keys += ["obstacle","obstacle_x","obstacle_y","obstacle_z","obstacle_size"]
        keys += [stem+a for stem in ('obstacle_count_','obstacle_spacing_') for a in 'xyz']
    if p.get('scenario') in {'open_channel','cylinder_wake'}:keys += ['inlet_profile','outlet_rho']
    if p.get('scenario') in {'oscillatory_channel','oscillating_couette'}:keys += ['drive_period','drive_phase']
    if p.get('scenario')=='oscillatory_channel':keys += ['force_amplitude']
    from research_options import EXTRA_DEFAULTS
    physical["research"].update({k:p.get(k,EXTRA_DEFAULTS.get(k)) for k in keys})
    return immutable(physical)


def difference(first, second):
    a, b = first["fields"], second["fields"]
    if a is None or b is None or a["preview"] or b["preview"]:
        raise ValueError("差值分析需要两组完整的最终流场")
    if not first.get("config") or not second.get("config") or physical_signature(first) != physical_signature(second):
        raise ValueError("两组物理参数、网格或边界不一致，不能直接作节点差值")
    if a["rho"].shape != b["rho"].shape or not np.array_equal(a["x"], b["x"]) or not np.array_equal(a["y"], b["y"]):
        raise ValueError("两组网格坐标不同，未进行插值")
    if "z" in a and not np.array_equal(a["z"],b.get("z")):
        raise ValueError("两组三维 z 坐标不同，未进行插值")
    if not a["finite"] or not b["finite"]:
        raise ValueError("场量含非有限值，不能给出有效误差")
    delta_u, delta_v = b["ux"] - a["ux"], b["uy"] - a["uy"]
    delta_w = b.get("uz",0)-a.get("uz",0)
    magnitude = np.sqrt(delta_u**2+delta_v**2+delta_w**2)
    denominator = np.linalg.norm(np.sqrt(a["ux"]**2+a["uy"]**2+a.get("uz",0)**2))
    norm = np.linalg.norm(magnitude)
    relative = norm / denominator if denominator else (0.0 if norm == 0 else None)
    warnings = []
    model_keys = ("collision_model", "mrt_s_e", "mrt_s_eps", "mrt_s_q")
    if any(first["config"].get(k) != second["config"].get(k) for k in model_keys):
        warnings.append("碰撞模型或松弛率不同；此差值用于方案比较，不是后端等价性误差")
    if relative is None:
        warnings.append("基准速度范数为零，相对 L2 无定义")
    if first["status"] != "converged" or second["status"] != "converged":
        warnings.append("包含未收敛结果；差值不代表稳态误差")
    if first["metrics"].get("iteration") != second["metrics"].get("iteration"):
        warnings.append("终止步数不同；本图不是同一步 CPU/GPU 等价性对照")
    return dict(values=magnitude, max_velocity_difference=float(magnitude.max()),
                relative_velocity_l2=float(relative) if relative is not None else None,
                max_density_difference=float(np.abs(b["rho"] - a["rho"]).max()), warnings=warnings)


def safe_csv_text(value):
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def export_csv(cases, target):
    columns = ["id", "name", "group", "status", "backend", "collision", "boundary", "mrt_s_e", "mrt_s_eps", "mrt_s_q", "grid", "Re", "U", "tol", "max_iter", "iteration",
               "residual", "mass_drift", "max_mach", "elapsed_seconds", "notes", "data_dir", "source_path"]
    from research_options import EXTRA_DEFAULTS
    extra_metrics = ("thermal_residual","kinetic_energy","temperature_min","temperature_max","nusselt_hot","nusselt_cold",
                     "drag_coefficient","lift_coefficient","inlet_flux","outlet_flux","flux_imbalance","mass_change","boundary_mass_exchange")
    columns += list(EXTRA_DEFAULTS)+list(extra_metrics)+["effective_re","nu_lattice","scalar_lattice"]
    with Path(target).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for case in cases:
            p, m = case["params"], case["metrics"]
            row = dict(id=case["id"], name=case["name"], group=case["group_name"], status=case["status"],
                       collision=p.get("collision", "BGK"), boundary=p.get("boundary", "nee"),
                       **{k: p.get(k) for k in ("mrt_s_e", "mrt_s_eps", "mrt_s_q")},
                       backend=p.get("backend"), grid=p.get("grid"), Re=p.get("re"), U=p.get("lid_speed"), tol=p.get("tol"),
                       max_iter=p.get("max_iter"), notes=case["notes"], data_dir=case["data_dir"], source_path=case.get("source_path"))
            row.update({k: m.get(k) for k in ("iteration", "residual", "mass_drift", "max_mach", "elapsed_seconds")})
            row.update({k:p.get(k) for k in EXTRA_DEFAULTS})
            row.update({k:m.get(k) for k in extra_metrics})
            if p.get("lattice"):
                from research_options import transport
                d = transport(p)
                row.update(effective_re=d["effective_re"],nu_lattice=d["nu"],
                           scalar_lattice=("D2Q5" if d["dim"]==2 else "D3Q7") if d["thermal_tau"] else "")
            writer.writerow({k: safe_csv_text(v) for k, v in row.items()})


def export_bundle(cases, target):
    manifest = dict(exported_at=now(), cases=[])
    allowed = ("config.json", "run_request.json", "run_summary.json", "results.npz", "progress.jsonl", "residual_history.csv",
               "centerlines.csv", "timing_and_population.json", "import_provenance.json", "worker_error.json", "checkpoint.npz", "probes.csv")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for case in cases:
            record = {k: case.get(k) for k in ("id", "name", "group_name", "notes", "status", "params", "origin", "source_path")}
            record["sha256"] = {}
            for name in allowed:
                source = Path(case["data_dir"]) / name
                if source.is_file():
                    archive.write(source, f"{case['id']}/{name}")
                    with source.open("rb") as stream:
                        record["sha256"][name] = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest["cases"].append(record)
            for snapshot in (Path(case["data_dir"])/"snapshots").glob("fields_*.npz"):
                archive.write(snapshot,f"{case['id']}/snapshots/{snapshot.name}")
                with snapshot.open("rb") as stream:
                    record["sha256"]["snapshots/"+snapshot.name] = hashlib.file_digest(stream,"sha256").hexdigest()
        archive.writestr("catalog.json", encode(manifest))
    return str(target)
