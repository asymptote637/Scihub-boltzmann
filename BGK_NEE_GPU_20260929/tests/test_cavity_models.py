import copy
from itertools import product
import json

import numpy as np
import pytest

from reference import CPUSolver, make_config, config_to_dict, verify_reference
from cavity_models import CavityCPUSolver, M, MINV
from boundary_conditions import E, W, OPP
from dashboard.store import DEFAULTS, Store, parameters_from_config, transport, validate_parameters
from dashboard.worker import write_preview


@pytest.mark.parametrize("collision,boundary", product(("BGK", "MRT"), ("nee", "halfway")))
def test_gpu_trajectories_match_cpu_with_nonunit_density_and_nonequilibrium(collision, boundary):
    from gpu_solver import ArrayGPUSolver
    from gpu_fused import FusedGPUSolver
    import cupy as cp
    cfg = make_config(grid=17, re=100, collision=collision, boundary=boundary,
                      mrt_s_e=1.23, mrt_s_eps=1.41, mrt_s_q=1.72, ramp_steps=37)
    cfg.grid.NY = 19
    cfg.flow.rho0 = 1.7
    cpu = CavityCPUSolver(cfg)
    rng = np.random.default_rng(29629)
    cpu.f += rng.uniform(-1e-5, 1e-5, cpu.f.shape)
    cpu.update_macroscopic()
    gpus = [ArrayGPUSolver(cfg), FusedGPUSolver(cfg)]
    for gpu in gpus:
        gpu.load_state(f=cpu.f, rho=cpu.rho, ux=cpu.ux, uy=cpu.uy)
    initial_mass = cpu.rho.sum()
    for step in range(1, 302):
        cpu.step()
        for gpu in gpus:
            gpu.step()
        if step in (1, 2, 37, 38, 100, 301):
            for gpu in gpus:
                for key in ("f", "rho", "ux", "uy"):
                    np.testing.assert_allclose(cp.asnumpy(getattr(gpu, key)), getattr(cpu, key), atol=3e-12, rtol=0)
                assert gpu.stop_reason == cpu.stop_reason == "running"
                np.testing.assert_allclose(gpu.decision_residual, cpu._velocity_residual(), atol=2e-11, rtol=1e-8)
    if boundary == "halfway":
        assert abs(cpu.rho.sum() / initial_mass - 1) < 5e-13


def test_original_bgk_nee_matches_frozen_solver_exactly():
    cfg = make_config(grid=16, re=100)
    frozen, new = CPUSolver(cfg), CavityCPUSolver(cfg)
    for _ in range(100):
        frozen.step()
        new.step()
    np.testing.assert_array_equal(frozen.f, new.f)
    assert verify_reference() == 18


def test_mrt_conservation_equilibrium_and_bgk_limit():
    cfg = make_config(grid=8, re=20, collision="MRT")
    cfg.flow.rho0 = 1.7
    solver = CavityCPUSolver(cfg)
    np.testing.assert_allclose(solver.collide(), solver.f, atol=2e-16)
    rng = np.random.default_rng(28)
    solver.f += rng.uniform(-0.001, 0.001, solver.f.shape)
    solver.update_macroscopic()
    post = solver.collide()
    np.testing.assert_allclose(post.sum(axis=-1), solver.f.sum(axis=-1), atol=8e-16)
    np.testing.assert_allclose(post @ E, solver.f @ E, atol=3e-16)
    for a in (1, 2, 4, 6, 7, 8):
        solver.rates[a] = solver.omega
    bgk = solver.f - solver.omega * (solver.f - solver.equilibrium(solver.rho, solver.ux, solver.uy))
    np.testing.assert_allclose(solver.collide(), bgk, atol=6e-16, rtol=0)


def test_each_mrt_mode_relaxes_at_its_own_rate():
    cfg = make_config(grid=8, re=20, collision="MRT")
    solver = CavityCPUSolver(cfg)
    equilibrium = solver.f.copy()
    for a in (1, 2, 4, 6, 7, 8):
        solver.f = equilibrium + 1e-4 * MINV[:, a]
        actual = (solver.collide() - equilibrium) @ M.T
        expected = np.zeros(9)
        expected[a] = 1e-4 * (1 - solver.rates[a])
        np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), atol=6e-16)


@pytest.mark.parametrize("collision", ("BGK", "MRT"))
def test_shear_wave_measures_prescribed_viscosity(collision):
    # Periodic streaming is test-only: isolate production collision from walls.
    # u_x=A*sin(2*pi*y/N) decays as exp(-nu*k^2*t) at low Mach.
    errors = []
    for n in (32, 64):
        cfg = make_config(grid=n, re=100, collision=collision)
        cfg.grid.NX = 8
        cfg.flow.nu_lattice = 0.1
        s = CavityCPUSolver(cfg)
        wave = np.sin(2 * np.pi * np.arange(n) / n)[:, None]
        s.ux[:] = 1e-4 * wave
        s.f = s.equilibrium(s.rho, s.ux, s.uy)
        samples = []
        for step in range(401):
            if step >= 100 and step % 20 == 0:
                samples.append((step, float(np.sum(s.ux * wave) / np.sum(np.broadcast_to(wave**2, s.ux.shape)))))
            post = s.collide()
            s.f = np.stack([np.roll(post[..., k], (int(ey), int(ex)), (0, 1)) for k, (ex, ey) in enumerate(E)], axis=-1)
            s.update_macroscopic()
        times, amplitudes = np.asarray(samples).T
        measured = -np.polyfit(times, np.log(amplitudes), 1)[0] / (2 * np.pi / n)**2
        errors.append(abs(measured / 0.1 - 1))
    assert errors[1] < 0.003, errors
    assert errors[1] < errors[0] / 3, errors


def test_halfway_link_oracle_corners_and_mass():
    cfg = make_config(grid=8, re=100, boundary="halfway", ramp_steps=0)
    s = CavityCPUSolver(cfg)
    rng = np.random.default_rng(12)
    post = rng.uniform(0.01, 0.1, s.f.shape)
    s.rho[:] = rng.uniform(0.9, 1.9, s.rho.shape)
    s.f = s.stream(post)
    s.apply_boundaries(post)
    # Independent scalar pull oracle: every missing link reflects once.
    for y, x, k in product(range(8), range(8), range(9)):
        ex, ey = E[k]
        sx, sy = x - ex, y - ey
        if 0 <= sx < 8 and 0 <= sy < 8:
            expected = post[sy, sx, k]
        else:
            expected = post[y, x, OPP[k]]
            if sy >= 8:
                expected += 6 * W[k] * s.rho[y, x] * ex * cfg.boundaries["top"].ux
        assert s.f[y, x, k] == pytest.approx(expected, abs=1e-16)
    assert s.f.sum() == pytest.approx(post.sum(), abs=1e-13)


@pytest.mark.parametrize("boundary", ("nee", "halfway"))
def test_geometry_persistence_preview_and_final_export(tmp_path, boundary):
    p = validate_parameters(dict(DEFAULTS, backend="cpu", grid=16, re=100, collision="MRT", boundary=boundary,
                                 mrt_s_e=1.2, max_iter=2))
    s = CavityCPUSolver(make_config(**{k:v for k,v in p.items() if k != "backend"}))
    s.config.output.output_dir = str(tmp_path)
    s.config.output.save_png = False
    expected = ((np.arange(16) + 0.5) / 16 if boundary == "halfway" else np.arange(16) / 15)
    assert s.transport["L_ref"] == (16 if boundary == "halfway" else 15)
    assert s.transport["tau"] == transport(p)["tau"]
    list(s.run())
    write_preview(s, tmp_path)
    s.finalize()
    for filename in ("preview.npz", "results.npz"):
        with np.load(tmp_path / filename) as data:
            np.testing.assert_array_equal(data["x"], expected)
            np.testing.assert_array_equal(data["y"], expected)
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert parameters_from_config(config, "cpu") == p
    store = Store(tmp_path / "workspace")
    store.save_preset("MRT", p)
    assert Store(store.workspace).presets()["MRT"] == p
    identifier = store.register_result(tmp_path)
    assert store.get(identifier)["params"]["collision"] == "MRT"
    assert store.get(identifier)["params"]["boundary"] == boundary


@pytest.mark.parametrize("change", [dict(collision="TRT"), dict(boundary="fullway"), dict(mrt_s_e=0),
                                   dict(mrt_s_eps=2), dict(mrt_s_q=float("nan"))])
def test_bad_models_and_rates_rejected(change):
    with pytest.raises(ValueError):
        validate_parameters(dict(DEFAULTS, **change))
    with pytest.raises(ValueError):
        make_config(**change)


def test_model_comparison_allows_collision_change_but_rejects_different_wall_nodes():
    from dashboard.analysis import difference
    config = config_to_dict(make_config(grid=8, re=100))
    fields = dict(x=np.arange(8)/7, y=np.arange(8)/7, rho=np.ones((8,8)), ux=np.ones((8,8)) * .01,
                  uy=np.zeros((8,8)), preview=False, finite=True)
    first = dict(config=config, fields=fields, metrics=dict(iteration=10), status="max_iter")
    second = copy.deepcopy(first)
    second["config"]["collision_model"] = "MRT"
    assert any("碰撞模型" in v for v in difference(first, second)["warnings"])
    second["config"] = config_to_dict(make_config(grid=8, re=100, boundary="halfway"))
    with pytest.raises(ValueError, match="边界"):
        difference(first, second)


@pytest.mark.parametrize("collision,boundary", product(("BGK", "MRT"), ("nee", "halfway")))
def test_new_modes_stop_budget_cancel_and_guard_nonfinite(collision, boundary):
    from gpu_solver import ArrayGPUSolver
    from gpu_fused import FusedGPUSolver
    for solver_type in (CavityCPUSolver, ArrayGPUSolver, FusedGPUSolver):
        cfg = make_config(grid=8, re=100, collision=collision, boundary=boundary,
                          max_iter=5, min_iter=0, ramp_steps=0, report_interval=2)
        cfg.boundaries["top"].ux = 0
        cfg.convergence.consecutive_reports = 3
        s = solver_type(cfg)
        list(s.run())
        assert s.stop_reason == "converged" and s.iteration == 3
        cfg = make_config(grid=8, re=100, collision=collision, boundary=boundary,
                          max_iter=5, report_interval=2)
        s = solver_type(cfg)
        assert [r.iteration for r in s.run()] == [1, 2, 4, 5]
        assert s.stop_reason == "max_iter"
        s = solver_type(cfg)
        host = getattr(s, "host", s)
        if hasattr(s, "load_state"):
            f = host.f.copy()
            f[0, 0, 0] = np.nan
            s.load_state(f=f, rho=host.rho, ux=host.ux, uy=host.uy)
        else:
            s.f[0, 0, 0] = np.nan
        s.step()
        assert s.stop_reason == "diverged" and s.iteration == 0
        s = solver_type(cfg)
        getattr(s, "host", s).cancel()
        s.step()
        assert s.stop_reason == "cancelled" and s.iteration == 0
