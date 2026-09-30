"""One detached serial scheduler per workspace; never signals external CPU jobs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from .store import ACTIVE, DEFAULT_WORKSPACE, ROOT, Store, now, read_json


@contextmanager
def exclusive_lock(path):
    handle = Path(path).open("a+b")
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    locked = False
    try:
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                locked = True
            except OSError:
                pass
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError:
                pass
        yield locked
    finally:
        if locked and os.name == "nt":
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def process_matches(pid, started, token):
    if not pid or not started:
        return False
    try:
        process = psutil.Process(pid)
        return (abs(process.create_time() - started) < 0.001
                and token in process.cmdline() and process.is_running())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def launch_hidden(arguments, log_path):
    environment = dict(os.environ, PYTHONUTF8="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with Path(log_path).open("ab", buffering=0) as stream:
        return subprocess.Popen([sys.executable, "-X", "utf8", "-u", *arguments], cwd=ROOT,
                                stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, env=environment,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                                start_new_session=os.name != "nt")


def ensure_service(store):
    identity = store.setting("service", {})
    if process_matches(identity.get("pid"), identity.get("started"), "dashboard.service"):
        return identity["pid"]
    process = launch_hidden(["-m", "dashboard.service", "--workspace", str(store.workspace)], store.workspace / "service.log")
    return process.pid


def reconcile_missing(store, case, grace=15):
    if process_matches(case.get("pid"), case.get("process_started"), case["id"]):
        return True
    if process_matches(case.get("launcher_pid"), case.get("launcher_started"), case["id"]):
        return True
    from datetime import datetime
    age = time.time() - datetime.fromisoformat(case["updated_at"]).timestamp()
    if age < grace:
        return True
    summary = read_json(Path(case["data_dir"]) / "run_summary.json", {})
    if summary.get("stop_reason") in ("converged", "max_iter", "completed_steps", "cancelled", "diverged"):
        store.update_case(case["id"], status=summary["stop_reason"])
    else:
        store.update_case(case["id"], status="interrupted")
        store.update_job(case["id"], error="工作进程已退出，未找到完整结束记录；未自动重跑")
    return False


def run_service(store, idle_seconds=60):
    with exclusive_lock(store.workspace / "service.lock") as locked:
        if not locked:
            return
        store.set_setting("service", dict(pid=os.getpid(), started=psutil.Process().create_time(), heartbeat=now()))
        child = None
        child_id = None
        idle_since = time.monotonic()
        last_heartbeat = 0.0
        while True:
            if time.monotonic() - last_heartbeat > 3:
                store.set_setting("service", dict(pid=os.getpid(), started=psutil.Process().create_time(), heartbeat=now()))
                last_heartbeat = time.monotonic()
            if child is not None:
                exit_code = child.poll()
                if exit_code is not None:
                    case = store.get(child_id)
                    if case["status"] in ACTIVE:
                        store.update_case(child_id, status="failed")
                        store.update_job(child_id, error=f"工作进程退出码 {exit_code}，详见 worker.log")
                    child = None
                    child_id = None
            active = [case for case in store.cases(archived=True) if case["status"] in ACTIVE]
            for case in active:
                if case["id"] != child_id:
                    reconcile_missing(store, store.get(case["id"]))
            if child is None:
                case = store.claim_next()
                if case:
                    child_id = case["id"]
                    folder = Path(case["data_dir"])
                    folder.mkdir(parents=True, exist_ok=True)
                    try:
                        child = launch_hidden(["-m", "dashboard.worker", "--workspace", str(store.workspace), "--case", child_id],
                                              folder / "worker.log")
                        store.update_job(child_id, launcher_pid=child.pid,
                                         launcher_started=psutil.Process(child.pid).create_time())
                    except Exception as error:
                        store.update_case(child_id, status="failed")
                        store.update_job(child_id, error=str(error))
                        child = None
            outstanding = any(case["status"] in (*ACTIVE, "queued") for case in store.cases(archived=True))
            if child is not None or outstanding:
                idle_since = time.monotonic()
            elif time.monotonic() - idle_since >= idle_seconds:
                break
            time.sleep(0.25)
        store.set_setting("service", dict(pid=None, started=None, heartbeat=now()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--idle-seconds", type=float, default=60)
    args = parser.parse_args()
    run_service(Store(args.workspace), args.idle_seconds)


if __name__ == "__main__":
    main()
