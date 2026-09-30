import csv
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from dashboard import analysis
from dashboard.service import reconcile_missing
from dashboard.store import DEFAULTS, Store, atomic_json, encode, receipt_status, transport, validate_parameters


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "workspace")


def request(**changes):
    return dict(name="case", group="test", params=dict(DEFAULTS, **changes))


@pytest.fixture
def result_folder(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    config = dict(case_type="lid_driven_cavity", collision_model="BGK", boundary_scheme_default="non_equilibrium_extrapolation",
                  grid=dict(NX=8, NY=8), flow=dict(Re=100, U_ref=0.04), convergence=dict(max_iter=300, min_iter=2000,
                  ramp_steps=500, report_interval=100, tol=1e-6), boundaries={})
    atomic_json(folder / "config.json", config)
    atomic_json(folder / "run_summary.json", dict(backend="cuda-fused", iteration=300, stop_reason="max_iter",
                                                final_diagnostics=dict(iteration=300, residual=0.002, mass_drift=0.0001)))
    xy = np.linspace(0, 1, 8)
    x, y = np.meshgrid(xy, xy)
    np.savez_compressed(folder / "results.npz", rho=np.ones_like(x), ux=x + 2 * y, uy=3 * x - y, x=xy, y=xy)
    return folder


def test_parameter_scientific_notation_and_transport():
    value = validate_parameters(dict(DEFAULTS, grid="2.56e2", max_iter="1e8", tol="1e-9"))
    assert value["grid"] == 256 and value["max_iter"] == 100000000
    assert transport(value)["tau"] == pytest.approx(0.5055961960497439)


@pytest.mark.parametrize("changes", [dict(grid="15.5"), dict(grid=7), dict(re="nan"), dict(re=0), dict(tol="inf"),
                                    dict(lid_speed=0.1), dict(report_interval=0), dict(backend="invalid"), dict(re=1e300)])
def test_invalid_parameters_rejected(changes):
    with pytest.raises(ValueError):
        validate_parameters(dict(DEFAULTS, **changes))


def test_batch_is_atomic(store):
    with pytest.raises(ValueError):
        store.enqueue([request(re=100), request(re=-1)])
    assert store.cases() == []


def test_queue_claim_control_and_reordering(store):
    a, b, c = store.enqueue([request(re=100), request(re=200), request(re=300)])
    store.move_queued(c, -1)
    assert store.claim_next()["id"] == a
    assert store.claim_next() is None
    store.control(a, "pause")
    assert store.get(a)["desired"] == "pause"
    store.update_case(a, status="cancelled")
    assert store.claim_next()["id"] == c
    store.control(b, "cancel")
    assert store.get(b)["status"] == "cancelled"
    assert store.get(b)["metrics"] == {}
    with pytest.raises(ValueError):
        store.control(b, "run")


def test_persistence_and_metadata(store):
    identifier = store.enqueue([request()])[0]
    store.save_preset("模板", dict(DEFAULTS, backend="cpu"))
    store.update_case(identifier, name="精度验证", group_name="对照组", notes="备注", archived=1)
    reopened = Store(store.workspace)
    assert reopened.cases() == []
    assert reopened.get(identifier)["notes"] == "备注"
    assert reopened.presets()["模板"]["backend"] == "cpu"
    reopened.delete_preset("模板")
    assert reopened.presets() == {}


def test_pause_queue_does_not_start_or_cancel_jobs(store):
    identifier = store.enqueue([request()])[0]
    store.set_setting("queue_paused", True)
    assert store.claim_next() is None
    assert store.get(identifier)["status"] == "queued"
    store.set_setting("queue_paused", False)
    assert store.claim_next()["id"] == identifier


def test_import_copies_and_deduplicates(store, result_folder):
    case_id = store.register_result(result_folder, copy_files=True)
    assert store.register_result(result_folder, copy_files=True) == case_id
    case = store.get(case_id)
    assert Path(case["data_dir"]) != result_folder
    assert case["status"] == "max_iter"
    assert case["metrics"].get("elapsed_seconds") is None
    copied = Path(case["data_dir"]) / "results.npz"
    assert copied.read_bytes() == (result_folder / "results.npz").read_bytes()
    assert (Path(case["data_dir"]) / "import_provenance.json").is_file()


def test_history_partial_and_invalid_rows(tmp_path):
    (tmp_path / "progress.jsonl").write_bytes(b'{"iteration":1,"residual":0}\ninvalid\n{"iteration":2')
    rows, warnings = analysis.history(tmp_path)
    assert len(rows) == 2 and len(warnings) == 2
    x, y = analysis.curve(rows, "residual")
    assert y[0] == 0 and np.isnan(y[1]) and np.isnan(x[1])


def test_fields_shape_and_centerline_interpolation(result_folder):
    fields = analysis.load_fields(result_folder)
    lines = analysis.centerlines(fields, 2.0)
    np.testing.assert_allclose(lines["u"], 0.25 + fields["y"])
    np.testing.assert_allclose(lines["v"], 1.5 * fields["x"] - 0.25)
    config = json.loads((result_folder / "config.json").read_text())
    config["grid"]["NX"] = 9
    atomic_json(result_folder / "config.json", config)
    with pytest.raises(ValueError, match="形状"):
        analysis.load_fields(result_folder)


def test_difference_and_configuration_guard(store, result_folder):
    case_id = store.register_result(result_folder)
    first = analysis.load_case(store.get(case_id), fields=True)
    second = analysis.load_case(store.get(case_id), fields=True)
    result = analysis.difference(first, second)
    assert result["max_velocity_difference"] == result["relative_velocity_l2"] == 0
    assert any("未收敛" in value for value in result["warnings"])
    second["config"]["flow"]["Re"] = 200
    with pytest.raises(ValueError, match="物理参数"):
        analysis.difference(first, second)


def test_signature_ignores_dictionary_order_and_equal_numeric_types():
    first = dict(config=dict(flow=dict(Re=100, U_ref=0.04), grid=dict(NX=8, NY=8)))
    second = dict(config=dict(grid=dict(NY=8, NX=8), flow=dict(U_ref=0.04, Re=100.0)))
    assert analysis.physical_signature(first) == analysis.physical_signature(second)


def test_zero_baseline_has_undefined_relative_error(store, result_folder):
    identifier = store.register_result(result_folder)
    first = analysis.load_case(store.get(identifier), fields=True)
    second = analysis.load_case(store.get(identifier), fields=True)
    first["fields"]["ux"] = np.zeros_like(first["fields"]["ux"])
    first["fields"]["uy"] = np.zeros_like(first["fields"]["uy"])
    result = analysis.difference(first, second)
    assert result["relative_velocity_l2"] is None
    assert any("无定义" in value for value in result["warnings"])


def test_exports_preserve_status_nulls_and_escape_formula(store, result_folder, tmp_path):
    case_id = store.register_result(result_folder)
    store.update_case(case_id, name="=HYPERLINK(1)")
    cases = [store.get(case_id)]
    output = tmp_path / "metrics.csv"
    analysis.export_csv(cases, output)
    with output.open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["name"].startswith("'=") and row["status"] == "max_iter"
    assert row["elapsed_seconds"] == ""
    bundle = tmp_path / "bundle.zip"
    analysis.export_bundle(cases, bundle)
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("catalog.json"))
        assert "results.npz" in manifest["cases"][0]["sha256"]
        assert archive.read(f"{case_id}/results.npz") == (result_folder / "results.npz").read_bytes()


def test_missing_worker_not_silently_restarted(store):
    case_id = store.enqueue([request()])[0]
    case = store.claim_next()
    assert not reconcile_missing(store, case, grace=0)
    assert store.get(case_id)["status"] == "interrupted"
    assert store.claim_next() is None


def test_frozen_and_gpu_gates():
    assert receipt_status() == {"cpu": True, "array": True, "fused": True}


def test_strict_json_cleaning():
    assert json.loads(encode(dict(nan=float("nan"), inf=float("inf")))) == dict(nan=None, inf=None)
