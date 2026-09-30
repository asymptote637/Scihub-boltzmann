"""Durable parameter templates, serial jobs, and non-destructive result catalog."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = ROOT / "workspace"
DEFAULTS = dict(backend="fused", grid=256, re=5468.0, lid_speed=0.04, max_iter=1000000,
                min_iter=2000, ramp_steps=500, report_interval=200, tol=1e-6,
                collision="BGK", boundary="nee", mrt_s_e=1.64, mrt_s_eps=1.54, mrt_s_q=1.9)
ACTIVE = ("starting", "running", "paused", "cancelling")
TERMINAL = ("converged", "max_iter", "completed_steps", "cancelled", "diverged", "failed", "interrupted", "incomplete")
STATUS_NAMES = dict(queued="排队", starting="初始化", running="计算中", paused="已暂停",
                    cancelling="正在停止", converged="已收敛", max_iter="步数用尽 · 未收敛",
                    completed_steps="指定步数完成", cancelled="已取消", diverged="数值保护停止",
                    failed="运行失败", interrupted="进程中断", incomplete="记录不完整")
BACKEND_NAMES = dict(cpu="CPU · NumPy", fused="GPU · 融合 CUDA", array="GPU · CuPy 对照")
COLLISION_NAMES = {"BGK": "BGK · 单松弛", "MRT": "MRT · 多松弛", "TRT": "TRT · 双松弛"}
BOUNDARY_NAMES = {"nee": "NEE · 非平衡外推", "halfway": "HBB · 半格点反弹", "periodic": "全周期", "inlet_outlet":"入口 / 出口 + HBB"}


def model_label(params):
    boundary = {"halfway":"HBB", "nee":"NEE", "periodic":"周期", "inlet_outlet":"入口出口"}.get(params.get("boundary", "nee"), "?")
    prefix = params.get("lattice", "D2Q9") + " · " if "lattice" in params else ""
    return f"{prefix}{params.get('collision', 'BGK')} / {boundary}"


def backend_label(key):
    return {"cpu": "CPU", "fused": "GPU·融合", "array": "GPU·数组"}.get(key, "未标注")


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_value(value):
    if isinstance(value, dict):
        return {str(k): safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def encode(value):
    return json.dumps(safe_value(value), ensure_ascii=False, allow_nan=False)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(encode(value), encoding="utf-8")
    temporary.replace(path)


def validate_parameters(raw):
    from research_options import is_extended, validate
    return validate(raw) if is_extended(raw) else validate_legacy_parameters(raw)


def validate_legacy_parameters(raw):
    params = dict(DEFAULTS)
    params.update({k: raw[k] for k in DEFAULTS if k in raw})
    if params["backend"] not in ("cpu", "array", "fused"):
        raise ValueError("计算后端无效")
    if params["collision"] not in {"BGK", "MRT"} or params["boundary"] not in {"nee", "halfway"}:
        raise ValueError("碰撞模型或边界条件无效")
    for key in ("mrt_s_e", "mrt_s_eps", "mrt_s_q"):
        try:
            params[key] = float(params[key])
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f"{key} 必须是 0 与 2 之间的有限数") from None
        if not math.isfinite(params[key]) or not 0 < params[key] < 2:
            raise ValueError(f"{key} 必须严格满足 0 < s < 2")
    limits = dict(grid=8, max_iter=1, min_iter=0, ramp_steps=0, report_interval=1)
    for key, minimum in limits.items():
        try:
            number = Decimal(str(params[key]))
        except InvalidOperation:
            raise ValueError(f"{key} 必须是整数") from None
        if not number.is_finite() or number != number.to_integral_value() or number < minimum or number > Decimal(sys.float_info.max):
            raise ValueError(f"{key} 必须是至少为 {minimum}、数量级不超出双精度范围的整数")
        params[key] = int(number)
    for key in ("re", "lid_speed", "tol"):
        try:
            value = float(params[key])
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f"{key} 必须是正数") from None
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} 必须是有限正数")
        params[key] = value
    if params["lid_speed"] * math.sqrt(3) > 0.1:
        raise ValueError("顶盖速度对应 Ma 超过模型上限 0.1")
    if params["grid"] ** 2 > 2147483647:
        raise ValueError("网格总节点数超出当前索引支持范围")
    derived = transport(params)
    if not math.isfinite(derived["tau"]) or derived["tau"] <= 0.5:
        raise ValueError("当前参数使双精度 tau 不大于 0.5，不能运行")
    return params


def transport(params):
    from research_options import is_extended, transport as research_transport
    if is_extended(params):
        return research_transport(params)
    length = params["grid"] if params.get("boundary", "nee") == "halfway" else params["grid"] - 1
    nu = params["lid_speed"] * length / params["re"]
    return dict(length=length, nu=nu, tau=0.5 + 3 * nu, mach=params["lid_speed"] * math.sqrt(3),
                estimated_host_bytes=params["grid"] ** 2 * 1024,
                estimated_gpu_bytes=params["grid"] ** 2 * (2048 if params["backend"] == "array" else 512))


def parameters_from_config(config, backend="cpu"):
    if config.get("research"):
        return dict(config["research"], backend=backend)
    grid, flow, convergence = (config.get(k, {}) for k in ("grid", "flow", "convergence"))
    types = {b.get("type") for b in config.get("boundaries", {}).values()}
    boundary = "halfway" if types and types <= {"halfway_bounce_back", "halfway_moving_wall"} else "nee"
    return dict(backend=backend, collision=config.get("collision_model", "BGK"), boundary=boundary,
                **{k: config.get(k, DEFAULTS[k]) for k in ("mrt_s_e", "mrt_s_eps", "mrt_s_q")},
                grid=grid.get("NX"), re=flow.get("Re"), lid_speed=flow.get("U_ref"),
                **{key: convergence.get(key) for key in ("max_iter", "min_iter", "ramp_steps", "report_interval", "tol")})


def receipt_status():
    result = {}
    for backend in ("array", "fused"):
        receipt = read_json(ROOT / f"validation/correctness_{backend}.json", {})
        names = ["gpu_environment.py", "reference.py", "gpu_solver.py", "cavity_models.py"]
        if backend == "fused":
            names += ["gpu_fused.py", "kernels.cu"]
        valid = bool(receipt.get("passed") and receipt.get("low_re", {}).get("passed"))
        valid = valid and all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() ==
                              receipt.get("source_sha256", {}).get(name) for name in names)
        result[backend] = valid
    plan = read_json(ROOT / "reference_cpu/plan.json", {})
    result["cpu"] = bool(plan.get("source_sha256")) and all(
        hashlib.sha256((ROOT / "reference_cpu" / name).read_bytes()).hexdigest() == digest
        for name, digest in plan.get("source_sha256", {}).items())
    from model_validation import verified_models
    try:
        verified_models()
    except RuntimeError:
        return {key: False for key in result}
    return result


class Store:
    def __init__(self, workspace=DEFAULT_WORKSPACE):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.path = self.workspace / "dashboard.sqlite3"
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS cases (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, group_name TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '', params_json TEXT NOT NULL, config_json TEXT NOT NULL DEFAULT '{}',
                    metrics_json TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL, data_dir TEXT UNIQUE,
                    origin TEXT NOT NULL, source_path TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    archived INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS jobs (
                    case_id TEXT PRIMARY KEY REFERENCES cases(id), desired TEXT NOT NULL DEFAULT 'run',
                    pid INTEGER, process_started REAL, error TEXT NOT NULL DEFAULT '', position INTEGER NOT NULL,
                    heartbeat TEXT, launcher_pid INTEGER, launcher_started REAL);
                CREATE TABLE IF NOT EXISTS presets (name TEXT PRIMARY KEY, params_json TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=15000")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def decode(row):
        if row is None:
            return None
        value = dict(row)
        for name in ("params", "config", "metrics"):
            value[name] = json.loads(value.pop(name + "_json"))
        return value

    def cases(self, archived=False):
        with self.connection() as db:
            rows = db.execute("SELECT c.*,j.desired,j.pid,j.process_started,j.error,j.position,j.heartbeat "
                              "FROM cases c LEFT JOIN jobs j ON c.id=j.case_id "
                              "WHERE (? OR c.archived=0) ORDER BY c.created_at DESC,c.rowid DESC", (int(archived),)).fetchall()
        return [self.decode(row) for row in rows]

    def get(self, case_id):
        with self.connection() as db:
            return self.decode(db.execute("SELECT c.*,j.desired,j.pid,j.process_started,j.error,j.position,j.heartbeat,"
                                          "j.launcher_pid,j.launcher_started FROM cases c LEFT JOIN jobs j ON c.id=j.case_id "
                                          "WHERE c.id=?", (case_id,)).fetchone())

    def enqueue(self, requests):
        prepared = [(str(r.get("name", "")).strip(), str(r.get("group", "")).strip(),
                     validate_parameters(r["params"])) for r in requests]
        if not prepared or len(prepared) > 100:
            raise ValueError("每批需要 1 至 100 组参数")
        identifiers = []
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            position = db.execute("SELECT COALESCE(MAX(position),0) FROM jobs").fetchone()[0]
            for name, group, params in prepared:
                case_id = uuid.uuid4().hex
                folder = self.workspace / "runs" / case_id
                title = name or f"{backend_label(params['backend'])} · {model_label(params)} · N{params['grid']} · Re{params['re']:g}"
                db.execute("INSERT INTO cases(id,name,group_name,params_json,status,data_dir,origin,created_at,updated_at) "
                           "VALUES(?,?,?,?,?,?,?,?,?)", (case_id, title, group, encode(params), "queued", str(folder), "computed", now(), now()))
                position += 1
                db.execute("INSERT INTO jobs(case_id,position) VALUES(?,?)", (case_id, position))
                identifiers.append(case_id)
        return identifiers

    def update_case(self, case_id, **values):
        allowed = {"status", "name", "group_name", "notes", "archived", "params_json", "config_json", "metrics_json"}
        if not set(values) <= allowed:
            raise ValueError("Invalid catalog update")
        values["updated_at"] = now()
        with self.connection() as db:
            db.execute("UPDATE cases SET " + ",".join(f"{k}=?" for k in values) + " WHERE id=?", (*values.values(), case_id))

    def update_job(self, case_id, **values):
        allowed = {"pid", "process_started", "desired", "error", "heartbeat", "launcher_pid", "launcher_started"}
        if not values or not set(values) <= allowed:
            raise ValueError("Invalid job update")
        with self.connection() as db:
            db.execute("UPDATE jobs SET " + ",".join(f"{k}=?" for k in values) + " WHERE case_id=?", (*values.values(), case_id))

    def control(self, case_id, action):
        if action not in ("run", "pause", "cancel"):
            raise ValueError("无效控制命令")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM cases WHERE id=?", (case_id,)).fetchone()
            if row is None or row["status"] not in (*ACTIVE, "queued"):
                raise ValueError("此记录没有可控制的任务")
            if row["status"] == "queued":
                if action != "cancel":
                    raise ValueError("排队中的任务只能调整顺序或取消")
                db.execute("UPDATE cases SET status='cancelled',updated_at=? WHERE id=?", (now(), case_id))
            db.execute("UPDATE jobs SET desired=? WHERE case_id=?", (action, case_id))

    def claim_next(self):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            paused = db.execute("SELECT value FROM settings WHERE key='queue_paused'").fetchone()
            if paused and json.loads(paused[0]):
                return None
            if db.execute("SELECT 1 FROM cases WHERE status IN ('starting','running','paused','cancelling') LIMIT 1").fetchone():
                return None
            row = db.execute("SELECT c.id FROM cases c JOIN jobs j ON c.id=j.case_id "
                             "WHERE c.status='queued' AND j.desired='run' ORDER BY j.position LIMIT 1").fetchone()
            if not row:
                return None
            db.execute("UPDATE cases SET status='starting',updated_at=? WHERE id=?", (now(), row[0]))
        return self.get(row[0])

    def move_queued(self, case_id, direction):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT j.case_id,j.position FROM jobs j JOIN cases c ON c.id=j.case_id "
                              "WHERE c.status='queued' ORDER BY j.position").fetchall()
            ids = [r[0] for r in rows]
            if case_id not in ids:
                return
            index = ids.index(case_id)
            target = index + (-1 if direction < 0 else 1)
            if 0 <= target < len(ids):
                db.execute("UPDATE jobs SET position=? WHERE case_id=?", (rows[target][1], case_id))
                db.execute("UPDATE jobs SET position=? WHERE case_id=?", (rows[index][1], ids[target]))

    def save_preset(self, name, params):
        if not name.strip():
            raise ValueError("模板名称不能为空")
        with self.connection() as db:
            db.execute("INSERT OR REPLACE INTO presets VALUES(?,?,?)", (name.strip(), encode(validate_parameters(params)), now()))

    def presets(self):
        with self.connection() as db:
            return {r["name"]: json.loads(r["params_json"]) for r in db.execute("SELECT * FROM presets ORDER BY name")}

    def delete_preset(self, name):
        with self.connection() as db:
            db.execute("DELETE FROM presets WHERE name=?", (name,))

    def setting(self, key, default=None):
        with self.connection() as db:
            row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key, value):
        with self.connection() as db:
            db.execute("INSERT OR REPLACE INTO settings VALUES(?,?)", (key, encode(value)))

    def register_result(self, source, copy_files=False, group="历史结果"):
        source = Path(source).resolve()
        if source.is_file():
            source = source.parent
        config = read_json(source / "config.json")
        summary = read_json(source / "run_summary.json", {})
        if not isinstance(config, dict) or not all(k in config for k in ("grid", "flow", "convergence")):
            raise ValueError("结果目录缺少有效的 config.json")
        if not (source / "results.npz").is_file():
            raise ValueError("结果目录缺少 results.npz，不能作为完整流场导入")
        with self.connection() as db:
            existing = db.execute("SELECT id FROM cases WHERE source_path=? OR data_dir=?", (str(source), str(source))).fetchone()
        if existing:
            return existing[0]
        case_id = uuid.uuid4().hex
        backend = {"cpu": "cpu", "cuda-fused": "fused", "cupy-array": "array"}.get(summary.get("backend"), "unknown")
        if source == (ROOT / "reference_cpu/historical_Re100_N64").resolve():
            backend = "cpu"
        params = parameters_from_config(config, backend)
        if not isinstance(params.get("re"), (int, float)) or not math.isfinite(params["re"]):
            raise ValueError("结果文件缺少有效的 Re 参数")
        from .analysis import load_fields
        load_fields(source)
        status = summary.get("stop_reason", "incomplete")
        if status not in TERMINAL:
            status = "incomplete"
        folder = source
        if copy_files:
            folder = self.workspace / "imports" / case_id
            folder.mkdir(parents=True, exist_ok=False)
            digests = {}
            for name in ("config.json", "run_summary.json", "results.npz", "progress.jsonl", "residual_history.csv",
                         "centerlines.csv", "run_request.json", "timing_and_population.json", "checkpoint.npz"):
                item = source / name
                if item.is_file():
                    shutil.copy2(item, folder / name)
                    with (folder / name).open("rb") as stream:
                        digests[name] = hashlib.file_digest(stream, "sha256").hexdigest()
            atomic_json(folder / "import_provenance.json", dict(source=str(source), imported_at=now(), sha256=digests))
        title = f"{backend_label(backend)} · {model_label(params)} · N{params['grid']} · Re{params['re']:g}"
        metrics = dict(summary.get("final_diagnostics", {}))
        timing = read_json(source / "timing_and_population.json", {})
        metrics.update(iteration=summary.get("iteration", metrics.get("iteration")),
                       elapsed_seconds=timing.get("compute_seconds", summary.get("elapsed_seconds")))
        with self.connection() as db:
            db.execute("INSERT INTO cases(id,name,group_name,params_json,config_json,metrics_json,status,data_dir,origin,"
                       "source_path,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                       (case_id, title, group, encode(params), encode(config), encode(metrics), status, str(folder),
                        "imported" if copy_files else "discovered", str(source), now(), now()))
        return case_id

    def discover(self):
        found, errors = [], []
        folders = sorted((ROOT / "results").glob("*/run_summary.json"))
        historical = ROOT / "reference_cpu/historical_Re100_N64/run_summary.json"
        if historical.is_file():
            folders.append(historical)
        for summary in folders:
            try:
                found.append(self.register_result(summary.parent, group="CPU 基准" if "historical" in str(summary) else "历史 GPU"))
            except (OSError, ValueError, TypeError) as error:
                errors.append(f"{summary.parent.name}: {error}")
        return found, errors
