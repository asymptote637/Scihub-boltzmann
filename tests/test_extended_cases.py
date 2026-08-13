import numpy as np
import pytest

from config import SUPPORTED_CASE_TYPES, case_preset, save_config, validate_config
from lbm_solver import LBMSolver
from main import load_json_config


@pytest.mark.parametrize("case_type", SUPPORTED_CASE_TYPES)
def test_all_registered_cases_build_valid_configs(case_type):
    cfg = case_preset(case_type, nx=48, ny=48)

    errors, _ = validate_config(cfg)

    assert not errors


@pytest.mark.parametrize("case_type", ["taylor_green_vortex", "shear_wave_decay"])
def test_periodic_decay_cases_lose_kinetic_energy(case_type):
    cfg = case_preset(case_type, nx=32, ny=32, u_ref=0.03)
    cfg.reynolds = 100.0
    solver = LBMSolver(cfg)
    initial_energy = float(np.mean(solver.ux**2 + solver.uy**2))

    for _ in range(40):
        solver.step()

    final_energy = float(np.mean(solver.ux**2 + solver.uy**2))
    assert 0.0 < final_energy < initial_energy
    assert np.isfinite(solver.f).all()


def test_physical_parameter_mode_matches_unit_conversion():
    cfg = case_preset("lid_driven_cavity", nx=101, ny=101, u_ref=0.05)
    cfg.parameter_mode = "physical"
    cfg.length_phys = 0.1
    cfg.velocity_phys = 2.0
    cfg.nu_phys = 1e-6

    assert cfg.dx_phys == pytest.approx(1e-3)
    assert cfg.dt_phys == pytest.approx(2.5e-5)
    assert cfg.nu_lattice == pytest.approx(2.5e-5)
    assert cfg.effective_reynolds == pytest.approx(2.0e5)


@pytest.mark.parametrize(
    "case_type, hot_side, cold_side",
    [
        ("natural_convection_cavity", "left", "right"),
        ("rayleigh_benard_convection", "bottom", "top"),
        ("heated_channel_flow", "bottom", "top"),
    ],
)
def test_thermal_cases_hold_isothermal_walls(case_type, hot_side, cold_side):
    cfg = case_preset(case_type, nx=32, ny=32)
    solver = LBMSolver(cfg)

    for _ in range(10):
        solver.step()

    temperature = solver.fields()["temperature"]
    edge = {
        "left": temperature[:, 0],
        "right": temperature[:, -1],
        "bottom": temperature[0, :],
        "top": temperature[-1, :],
    }
    np.testing.assert_allclose(edge[hot_side], cfg.temperature_hot, atol=1e-12)
    np.testing.assert_allclose(edge[cold_side], cfg.temperature_cold, atol=1e-12)


def test_natural_convection_couples_temperature_to_velocity():
    cfg = case_preset("natural_convection_cavity", nx=48, ny=48)
    cfg.ramp_steps = 20
    solver = LBMSolver(cfg)

    for _ in range(40):
        solver.step()

    assert float(np.max(np.abs(solver.uy))) > 0.0
    report = solver.make_report()
    assert report.temperature_residual is not None
    assert report.nusselt_average is not None
    assert np.isfinite(solver.fields()["temperature"]).all()


def test_thermal_config_json_round_trip(tmp_path):
    cfg = case_preset("natural_convection_cavity", nx=36, ny=40)
    cfg.prandtl = 7.0
    cfg.rayleigh = 5e4
    path = tmp_path / "thermal_config.json"
    save_config(cfg, path)

    loaded = load_json_config(path)

    assert loaded.thermal_enabled
    assert loaded.prandtl == pytest.approx(7.0)
    assert loaded.rayleigh == pytest.approx(5e4)
    assert loaded.thermal_left.type == "isothermal"
    assert loaded.thermal_left.temperature == pytest.approx(cfg.temperature_hot)
