"""Versioned case files and append-only run directories (no GUI dependency)."""
from __future__ import annotations

from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import uuid
from typing import Any

from config import (BoundaryConfig, ConvergenceConfig, FlowConfig, GridConfig,
                    OutputConfig, SimulationConfig, validate_config)

SCHEMA_VERSION = 1
PRESETS = {
    "couette": ("T07 · Couette 剪切流", "移动上壁、静止下壁，检查线性速度剖面与守恒。"),
    "force": ("T08 · 体力 Poiseuille", "恒定格子加速度驱动周期通道，检查抛物线剖面。"),
    "pressure": ("T09 · 压力通道", "Zou–He 密度入口/出口，检查低 Ma 压差驱动与质量收支。"),
    "cavity": ("T10 · Re=100 方腔", "移动顶盖产生回流，对照 Ghia 中心线二次转录数据。"),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def unique_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(clean_json(value), ensure_ascii=False,
                                        indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON 不允许非有限值：{value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _typed_values(cls: type, data: dict, label: str) -> dict:
    if not isinstance(data, dict):
        raise ValueError(f"{label} 必须为 JSON 对象")
    defaults = asdict(cls())
    unknown = set(data) - set(defaults)
    if unknown:
        raise ValueError(f"{label} 未知字段：{', '.join(sorted(unknown))}")
    for key, value in data.items():
        default = defaults[key]
        if default is None:
            valid = value is None or (type(value) in (int, float) and math.isfinite(value))
        elif type(default) is bool:
            valid = type(value) is bool
        elif type(default) is int:
            valid = type(value) is int
        elif type(default) is float:
            valid = type(value) in (int, float) and math.isfinite(value)
        else:
            valid = isinstance(value, type(default))
        if not valid:
            raise ValueError(f"{label}.{key} 类型不正确或不是有限数值")
    return data


def config_from_dict(data: dict) -> SimulationConfig:
    if not isinstance(data, dict):
        raise ValueError("配置必须为 JSON 对象")
    data = dict(data)
    # Solver exports add derived values; recompute these instead of trusting them.
    data.pop("derived", None)
    allowed = {f.name for f in fields(SimulationConfig)}
    if set(data) - allowed:
        raise ValueError(f"未知配置字段：{sorted(set(data) - allowed)}")
    values = {}
    for group, cls in (("grid", GridConfig), ("flow", FlowConfig),
                       ("convergence", ConvergenceConfig), ("output", OutputConfig)):
        values[group] = cls(**_typed_values(cls, data.pop(group, {}), group))
    boundaries = data.pop("boundaries", None)
    if boundaries is not None:
        if not isinstance(boundaries, dict) or set(boundaries) != {"left", "right", "bottom", "top"}:
            raise ValueError("boundaries 必须完整包含 left/right/bottom/top")
        values["boundaries"] = {side: BoundaryConfig(**_typed_values(BoundaryConfig, bc, side))
                                for side, bc in boundaries.items()}
    for name, value in data.items():
        if not isinstance(value, str):
            raise ValueError(f"{name} 必须是字符串")
    config = SimulationConfig(**data, **values)
    errors, _, _ = validate_config(config)
    if config.output.save_animation:
        errors.append("当前工作台未实现动画导出，请将 save_animation 设为 false。")
    if errors:
        raise ValueError("\n".join(errors))
    return config


def preset(key: str) -> SimulationConfig:
    if key not in PRESETS:
        raise ValueError(f"未知预设：{key}")
    cases = {"couette": "couette_flow", "force": "force_poiseuille",
             "pressure": "poiseuille_channel", "cavity": "lid_driven_cavity"}
    n = 64 if key == "cavity" else (16 if key == "pressure" else 32)
    c = SimulationConfig(
        case_type=cases[key], grid=GridConfig(NX=64 if key in {"cavity", "pressure"} else 8, NY=n),
        flow=FlowConfig(U_ref=.04 if key == "cavity" else .02, Re=100,
                        nu_lattice=None if key == "cavity" else .1,
                        body_force_x=8*.1*.02/n**2 if key == "force" else 0),
        convergence=ConvergenceConfig(max_iter=50000 if key == "cavity" else 30000,
                                      min_iter=2000 if key == "cavity" else 1000,
                                      report_interval=200, ramp_steps=500 if key == "cavity" else 0,
                                      tol=1e-7))
    if key == "pressure":
        delta = 3 * (8*.1*.02/n**2) * (c.grid.NX-1)
        c.boundaries["left"] = BoundaryConfig("pressure_zou_he", rho=1+delta/2)
        c.boundaries["right"] = BoundaryConfig("pressure_zou_he", rho=1-delta/2)
    return c


def case_document(config: SimulationConfig, name: str, notes: str = "") -> dict:
    # Strict roundtrip prevents malformed in-memory objects being saved.
    config = config_from_dict(asdict(config))
    if not isinstance(name, str) or not name.strip():
        raise ValueError("请填写算例名称")
    if not isinstance(notes, str):
        raise ValueError("备注必须是文本")
    return dict(schema_version=SCHEMA_VERSION, name=name.strip(), notes=notes,
                created_at=now(), config=asdict(config))


def load_case(path: Path) -> dict:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("算例文件必须为 JSON 对象")
    if "schema_version" in data:
        if type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
            raise ValueError("不支持该算例版本")
        unknown = set(data) - {"schema_version", "name", "notes", "created_at", "config"}
        if unknown or "config" not in data or "name" not in data:
            raise ValueError("算例封装字段不正确")
        document = case_document(config_from_dict(data["config"]), data["name"], data.get("notes", ""))
        if "created_at" in data:
            if not isinstance(data["created_at"], str):
                raise ValueError("created_at 必须是时间文本")
            document["created_at"] = data["created_at"]
        return document
    config = config_from_dict(data)
    return case_document(config, config.case_type, f"导入自 {path}")


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        for folder in ("cases", "runs"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)

    def save_case(self, config: SimulationConfig, name: str, notes: str = "") -> Path:
        path = self.root / "cases" / f"{unique_id('case')}.json"
        atomic_json(path, case_document(config, name, notes))
        return path

    def cases(self) -> list:
        result = []
        for path in sorted((self.root / "cases").glob("*.json"), reverse=True):
            try:
                result.append((path, load_case(path), None))
            except (OSError, ValueError, TypeError) as exc:
                result.append((path, None, str(exc)))
        return result

    def create_run(self, config: SimulationConfig, name: str, notes: str = "") -> Path:
        document = case_document(config, name, notes)
        path = self.root / "runs" / unique_id("run")
        path.mkdir()
        # Always retain macro fields and diagnostic tables for archive/validation.
        document["config"]["output"].update(output_dir=str(path / "result"),
                                           save_npz=True, save_csv=True)
        atomic_json(path / "request.json", document)
        atomic_json(path / "job.json", dict(state="created", name=name, created_at=now()))
        return path

    def runs(self) -> list:
        records = []
        for path in sorted((self.root / "runs").iterdir(), reverse=True):
            if not path.is_dir():
                continue
            try:
                request = load_case(path / "request.json")
                job = read_json(path / "job.json")
                summary = read_json(path / "result/run_summary.json") if (path / "result/run_summary.json").exists() else {}
                validation = read_json(path / "result/validation.json") if (path / "result/validation.json").exists() else {}
                if not all(isinstance(v, dict) for v in (job, summary, validation)):
                    raise ValueError("运行状态、摘要和验证文件必须为 JSON 对象")
                records.append(dict(path=path, request=request, job=job, summary=summary,
                                    validation=validation, error=None))
            except (OSError, ValueError, TypeError) as exc:
                records.append(dict(path=path, request={}, job={}, summary={}, validation={}, error=str(exc)))
        return records
