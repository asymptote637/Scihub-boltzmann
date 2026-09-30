import errno
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from dashboard import analysis, worker
from dashboard.store import DEFAULTS, Store


def sample(iteration=7):
    xy = np.linspace(0, 1, 16)
    x, y = np.meshgrid(xy, xy)
    return SimpleNamespace(nx=16, ny=16, rho=np.ones_like(x), ux=x * 0.04,
                           uy=y * -0.01, iteration=iteration)


def test_preview_retries_transient_lock_and_publishes_atomically(tmp_path, monkeypatch):
    worker.write_preview(sample(7), tmp_path)
    previous = (tmp_path / "preview.npz").read_bytes()
    replace = Path.replace
    attempts, delays = [], []

    def transient(source, target):
        if Path(target).name == "preview.npz":
            attempts.append(1)
            if len(attempts) <= 2:
                assert Path(target).read_bytes() == previous
                raise PermissionError(errno.EACCES, "preview held open")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", transient)
    monkeypatch.setattr(worker.time, "sleep", delays.append)
    host = sample(23)
    before = host.ux.copy()
    worker.write_preview(host, tmp_path)
    assert len(attempts) == 3 and delays == [0.01, 0.03]
    assert not (tmp_path / "preview.tmp").exists()
    with np.load(tmp_path / "preview.npz") as archive:
        assert int(archive["iteration"]) == 23
    np.testing.assert_array_equal(host.ux, before)


def test_preview_persistent_lock_is_bounded_and_preserves_last_file(tmp_path, monkeypatch):
    worker.write_preview(sample(), tmp_path)
    previous = (tmp_path / "preview.npz").read_bytes()
    attempts, delays = [], []

    def locked(source, target):
        attempts.append(1)
        raise PermissionError(errno.EACCES, "preview held open")

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr(worker.time, "sleep", delays.append)
    with pytest.raises(PermissionError):
        worker.write_preview(sample(23), tmp_path)
    assert len(attempts) == 4 and sum(delays) == pytest.approx(0.1)
    assert (tmp_path / "preview.npz").read_bytes() == previous
    assert not (tmp_path / "preview.tmp").exists()


def test_preview_temp_write_failure_preserves_last_file(tmp_path, monkeypatch):
    worker.write_preview(sample(), tmp_path)
    previous = (tmp_path / "preview.npz").read_bytes()
    original = np.savez_compressed

    def full_disk(*args, **kwargs):
        raise OSError(errno.ENOSPC, "preview staging write failed")

    monkeypatch.setattr(np, "savez_compressed", full_disk)
    with pytest.raises(OSError, match="staging"):
        worker.write_preview(sample(23), tmp_path)
    assert (tmp_path / "preview.npz").read_bytes() == previous
    assert not (tmp_path / "preview.tmp").exists()
    monkeypatch.setattr(np, "savez_compressed", original)
    worker.write_preview(sample(24), tmp_path)
    assert analysis.load_fields(tmp_path, preview=True)["preview_iteration"] == 24


def test_preview_file_is_closed_before_decompression(tmp_path, monkeypatch):
    worker.write_preview(sample(), tmp_path)
    load = np.load

    def replace_during_decode(source, *args, **kwargs):
        assert isinstance(source, io.BytesIO)
        # On Windows this rename fails if our reader still holds a normal file handle.
        (tmp_path / "preview.npz").replace(tmp_path / "previous.npz")
        worker.write_preview(sample(23), tmp_path)
        return load(source, *args, **kwargs)

    monkeypatch.setattr(np, "load", replace_during_decode)
    fields = analysis.load_fields(tmp_path, preview=True)
    assert fields["preview_iteration"] == 7 and fields["finite"]
    monkeypatch.setattr(np, "load", load)
    assert analysis.load_fields(tmp_path, preview=True)["preview_iteration"] == 23


def test_unreadable_preview_keeps_diagnostics_and_reports_warning(tmp_path, monkeypatch):
    worker.write_preview(sample(), tmp_path)
    (tmp_path / "progress.jsonl").write_text('{"iteration":23,"residual":0.001}\n', encoding="utf-8")
    open_file = Path.open

    def locked(path, *args, **kwargs):
        if path.name == "preview.npz":
            raise PermissionError(errno.EACCES, "preview reader denied")
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", locked)
    case = analysis.load_case(dict(data_dir=str(tmp_path)), fields=True, preview=True)
    assert case["fields"] is None
    assert case["records"][0]["iteration"] == 23
    assert any("preview reader denied" in warning for warning in case["warnings"])


def test_corrupt_preview_is_optional_but_corrupt_final_result_is_not(tmp_path):
    (tmp_path / "preview.npz").write_bytes(b"incomplete ZIP")
    case = dict(data_dir=str(tmp_path))
    result = analysis.load_case(case, fields=True, preview=True)
    assert result["fields"] is None and result["warnings"]
    (tmp_path / "results.npz").write_bytes(b"incomplete ZIP")
    with pytest.raises(zipfile.BadZipFile):
        analysis.load_case(case, fields=True, preview=True)


def test_preview_snapshot_has_bounded_memory(tmp_path, monkeypatch):
    worker.write_preview(sample(), tmp_path)
    monkeypatch.setattr(analysis, "PREVIEW_BYTES_LIMIT", 16)
    with pytest.raises(analysis.PreviewUnavailable, match="16 MB"):
        analysis.load_fields(tmp_path, preview=True)


def prepare_job(tmp_path, backend="cpu"):
    store = Store(tmp_path / "workspace")
    params = dict(DEFAULTS, backend=backend, grid=16, re=100, max_iter=301, report_interval=100)
    identifier = store.enqueue([dict(name="preview-fault-isolation", params=params)])[0]
    assert store.claim_next()["id"] == identifier
    folder = Path(store.get(identifier)["data_dir"])
    folder.mkdir(parents=True)
    return store, identifier, folder, params


def assert_complete_and_equal(store, identifier, folder, params):
    from reference import CPUSolver, make_config
    cpu = CPUSolver(make_config(**{key: value for key, value in params.items() if key != "backend"}))
    list(cpu.run())
    case = store.get(identifier)
    assert case["status"] == "max_iter" and case["error"] == ""
    assert case["metrics"]["iteration"] == 301
    assert case["metrics"]["preview_skipped_updates"] >= 1
    assert not (folder / "worker_error.json").exists()
    summary = json.loads((folder / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["stop_reason"] == "max_iter"
    assert summary["preview_status"]["skipped_updates"] >= 1
    assert (folder / "timing_and_population.json").is_file()
    assert (folder / "residual_history.csv").is_file()
    with np.load(folder / "results.npz") as fields:
        for key in ("rho", "ux", "uy"):
            np.testing.assert_array_equal(fields[key], getattr(cpu, key))


@pytest.mark.skipif(os.name != "nt", reason="Native Windows file sharing regression")
@pytest.mark.parametrize("backend", ["cpu", "array", "fused"])
def test_real_windows_preview_lock_does_not_abort_worker(tmp_path, backend, capsys):
    store, identifier, folder, params = prepare_job(tmp_path, backend)
    worker.write_preview(sample(), folder)
    previous = (folder / "preview.npz").read_bytes()
    with (folder / "preview.npz").open("rb"):
        worker.run_case(store, identifier)
    assert_complete_and_equal(store, identifier, folder, params)
    assert (folder / "preview.npz").read_bytes() == previous
    assert "preview update skipped" in capsys.readouterr().out


@pytest.mark.parametrize("failure", [PermissionError(errno.EACCES, "preview write denied"),
                                    OSError(errno.ENOSPC, "preview staging full")])
def test_worker_skips_optional_preview_io_errors(tmp_path, monkeypatch, failure):
    store, identifier, folder, params = prepare_job(tmp_path)

    def unavailable(*args):
        raise failure

    monkeypatch.setattr(worker, "write_preview", unavailable)
    worker.run_case(store, identifier)
    assert_complete_and_equal(store, identifier, folder, params)


def test_worker_preview_recovers_and_clears_live_warning(tmp_path, monkeypatch, capsys):
    store, identifier, folder, params = prepare_job(tmp_path)
    publish = worker.write_preview
    attempts = []

    def first_fails(*args):
        attempts.append(1)
        if len(attempts) == 1:
            raise PermissionError(errno.EACCES, "temporary preview lock")
        return publish(*args)

    monkeypatch.setattr(worker, "write_preview", first_fails)
    worker.run_case(store, identifier)
    assert_complete_and_equal(store, identifier, folder, params)
    assert store.get(identifier)["metrics"]["preview_warning"] is None
    assert store.get(identifier)["metrics"]["preview_iteration"] == 301
    assert "preview update recovered" in capsys.readouterr().out


def test_final_result_io_failure_is_not_silenced(tmp_path, monkeypatch):
    from reference import CPUSolver
    store, identifier, folder, params = prepare_job(tmp_path)

    def failed_export(self):
        raise PermissionError(errno.EACCES, "final result export denied")

    monkeypatch.setattr(CPUSolver, "finalize", failed_export)
    with pytest.raises(PermissionError, match="final result export denied"):
        worker.run_case(store, identifier)
    assert store.get(identifier)["status"] == "failed"
    assert "final result export denied" in store.get(identifier)["error"]
    assert (folder / "worker_error.json").is_file()


def test_unexpected_preview_programming_error_is_not_silenced(tmp_path, monkeypatch):
    store, identifier, folder, params = prepare_job(tmp_path)

    def bug(*args):
        raise RuntimeError("unexpected preview bug")

    monkeypatch.setattr(worker, "write_preview", bug)
    with pytest.raises(RuntimeError, match="unexpected preview bug"):
        worker.run_case(store, identifier)
    assert store.get(identifier)["status"] == "failed"
