import numpy as np
import pytest

from boundary_conditions import (
    INCOMING,
    E,
    equilibrium,
    moving_wall_bounce_back,
)
from config import BoundaryConfig, case_preset, validate_config
from lbm_solver import LBMSolver
from training.channel_validation import (
    build_channel_config,
    observed_orders,
    run_channel_case,
    save_experiment,
)


@pytest.mark.parametrize("side", ["left", "right", "bottom", "top"])
@pytest.mark.parametrize("velocity", [-0.02, 0.02])
def test_legacy_moving_wall_preserves_tangential_equilibrium(side, velocity):
    """Uniform fluid moving with a wall is an invariant state (sign + density)."""
    cfg = case_preset("custom", nx=8, ny=8)
    ux, uy = (0.0, velocity) if side in {"left", "right"} else (velocity, 0.0)
    f_eq = equilibrium(np.full((8, 8), 1.7), np.full((8, 8), ux), np.full((8, 8), uy))
    f = f_eq.copy()
    from boundary_conditions import EDGE
    selector = EDGE[side]
    f[selector][..., INCOMING[side]] = 0.0
    moving_wall_bounce_back(f, side, BoundaryConfig(ux=ux, uy=uy), cfg, 1.0)
    np.testing.assert_allclose(f, f_eq, rtol=0, atol=1e-14)


@pytest.mark.parametrize("model", ["BGK", "TRT", "MRT"])
def test_guo_force_density_gives_exact_uniform_acceleration(model):
    cfg = case_preset("custom", nx=8, ny=8)
    cfg.collision_model = model
    cfg.parameter_mode = "tau"
    cfg.tau_target = 0.8
    cfg.rho0 = 1.7
    cfg.body_force_x, cfg.body_force_y = 2e-6, -1e-6
    cfg.ramp_profile = "instant"
    for side in ["left", "right", "bottom", "top"]:
        setattr(cfg, side, BoundaryConfig("periodic"))
    solver = LBMSolver(cfg)
    for step in range(11):
        force = np.array([cfg.body_force_x, cfg.body_force_y])
        momentum = np.einsum("...q,qi->...i", solver.f, E) + force / 2
        np.testing.assert_allclose(momentum[..., 0] / solver.rho, solver.ux, atol=1e-14)
        np.testing.assert_allclose(momentum[..., 1] / solver.rho, solver.uy, atol=1e-14)
        np.testing.assert_allclose(solver.ux, step * force[0] / cfg.rho0, atol=1e-14)
        np.testing.assert_allclose(solver.uy, step * force[1] / cfg.rho0, atol=1e-14)
        solver.step()


def test_refinement_preserves_re_and_diffusive_scaling():
    coarse = build_channel_config("poiseuille", 16)
    fine = build_channel_config("poiseuille", 32)
    assert coarse.effective_reynolds == pytest.approx(fine.effective_reynolds)
    assert coarse.nu_lattice == pytest.approx(fine.nu_lattice)
    assert fine.u_ref == pytest.approx(coarse.u_ref / 2)
    assert fine.body_force_x == pytest.approx(coarse.body_force_x / 8)
    assert fine.characteristic_length == 32


@pytest.mark.parametrize("case", ["couette", "poiseuille"])
def test_channels_start_from_rest_and_recover_profile(case):
    result = run_channel_case(case, 16)
    assert result["summary"]["passed"]
    assert result["history"][0]["relative_l2_error"] == pytest.approx(1)
    assert np.all(result["velocity"] > 0)


def test_poiseuille_refinement_is_second_order():
    # Smaller grids bound test time. Acceptance at ny=8 is deliberately not
    # asserted; its error is expected to exceed the default 0.5 percent gate.
    results = [run_channel_case("poiseuille", ny) for ny in (8, 16)]
    order = observed_orders(results)[0]["l2_order"]
    assert 1.9 < order < 2.1


@pytest.mark.parametrize("model,ny,expected_pass", [
    ("TRT", 16, True), ("MRT", 16, False), ("MRT", 32, True),
])
def test_other_collisions_with_guo_and_halfway_walls(model, ny, expected_pass):
    result = run_channel_case("poiseuille", ny, collision_model=model, rho0=1.7)
    summary = result["summary"]
    assert summary["status"] == "converged"
    assert summary["passed"] is expected_pass
    if not expected_pass:
        # Keep the common acceptance gate: default MRT at H=16 has greater
        # wall error. H=32 passes without changing any threshold/rate.
        assert 0.005 < summary["relative_l2_error"] < 0.0053
        assert not summary["checks"]["profile_l2"]


def test_rotated_halfway_channel_recovers_vertical_poiseuille():
    cfg = build_channel_config("poiseuille", 8)
    cfg.body_force_y = cfg.body_force_x
    cfg.body_force_x = 0.0
    cfg.left = cfg.right = BoundaryConfig("halfway_bounce_back")
    cfg.top = cfg.bottom = BoundaryConfig("periodic")
    solver = LBMSolver(cfg)
    for _ in range(1600):
        solver.step()
    eta = (np.arange(8) + 0.5) / 8
    exact = 4 * cfg.u_ref * eta * (1 - eta)
    assert np.linalg.norm(solver.uy.mean(axis=0) - exact) / np.linalg.norm(exact) < 0.012
    assert np.max(np.abs(solver.ux)) < 1e-13


def test_invalid_halfway_topology_and_wall_normal_velocity_rejected():
    cfg = build_channel_config("couette", 16)
    cfg.top.uy = 0.01
    assert any("normal velocity" in error for error in validate_config(cfg)[0])
    cfg.top.uy = 0.0
    cfg.right.type = "halfway_bounce_back"
    assert any("opposite walls" in error for error in validate_config(cfg)[0])


def test_exhausted_budget_is_not_marked_converged(tmp_path):
    import json
    result = run_channel_case("couette", 8, max_diffusion_times=0.01)
    assert result["summary"]["status"] == "max_iter"
    assert not result["summary"]["passed"]
    output = tmp_path / "failed_run"
    summary = save_experiment(output, [result], {})
    assert not summary["passed"]
    assert json.loads((output / "summary.json").read_text())["passed"] is False
    with np.load(output / "couette_n8" / "fields.npz") as fields:
        assert fields["ux"].shape == (8, 8)
    assert (output / "profiles.png").is_file()


@pytest.mark.parametrize("kwargs", [{"tau": 0.5}, {"reynolds": float("nan")}, {"rho0": 0}])
def test_invalid_experiment_parameters(kwargs):
    with pytest.raises(ValueError):
        build_channel_config("poiseuille", 16, **kwargs)
