import numpy as np
import pytest

from boundary_conditions import E, macroscopic, mixed_bounce_specular, specular_reflection
from config import BoundaryConfig, case_preset, recommend_u_ref, validate_config
from lbm_solver import LBMSolver


@pytest.mark.parametrize("collision_model", ["BGK", "TRT", "MRT"])
def test_lid_driven_cavity_core_smoke(collision_model):
    cfg = case_preset("lid_driven_cavity", nx=24, ny=24)
    cfg.collision_model = collision_model
    cfg.reynolds = 50
    cfg.u_ref = 0.03
    cfg.top.ux = cfg.u_ref
    cfg.max_iter = 20
    cfg.min_iter = 10
    cfg.report_interval = 10

    errors, _warnings = validate_config(cfg)
    assert not errors

    solver = LBMSolver(cfg)
    reports = list(solver.run())

    assert reports[-1].iteration == 20
    assert solver.f.shape == (24, 24, 9)
    assert solver.rho.shape == (24, 24)


@pytest.mark.parametrize("collision_model", ["BGK", "TRT", "MRT"])
def test_collision_conserves_local_mass_and_momentum(collision_model):
    cfg = case_preset("custom", nx=12, ny=10)
    cfg.collision_model = collision_model
    cfg.reynolds = 20.0
    cfg.u_ref = 0.02
    solver = LBMSolver(cfg)

    rng = np.random.default_rng(7)
    solver.f *= 1.0 + rng.normal(0.0, 1e-3, solver.f.shape)
    solver.rho, solver.ux, solver.uy = macroscopic(solver.f, solver.solid_mask)
    before = np.stack(
        (
            np.sum(solver.f, axis=-1),
            np.tensordot(solver.f, E[:, 0], axes=([-1], [0])),
            np.tensordot(solver.f, E[:, 1], axes=([-1], [0])),
        )
    )

    post_collision = solver.collide()
    after = np.stack(
        (
            np.sum(post_collision, axis=-1),
            np.tensordot(post_collision, E[:, 0], axes=([-1], [0])),
            np.tensordot(post_collision, E[:, 1], axes=([-1], [0])),
        )
    )

    np.testing.assert_allclose(after, before, rtol=0.0, atol=1e-13)


def test_trt_reduces_to_bgk_when_relaxation_rates_match():
    bgk_cfg = case_preset("custom", nx=12, ny=10)
    bgk_cfg.reynolds = 20.0
    bgk_cfg.u_ref = 0.02
    trt_cfg = case_preset("custom", nx=12, ny=10)
    trt_cfg.reynolds = bgk_cfg.reynolds
    trt_cfg.u_ref = bgk_cfg.u_ref
    trt_cfg.collision_model = "TRT"
    trt_cfg.trt_magic_parameter = (trt_cfg.tau - 0.5) ** 2

    bgk = LBMSolver(bgk_cfg)
    trt = LBMSolver(trt_cfg)
    rng = np.random.default_rng(11)
    populations = bgk.f * (1.0 + rng.normal(0.0, 1e-3, bgk.f.shape))
    for solver in (bgk, trt):
        solver.f = populations.copy()
        solver.rho, solver.ux, solver.uy = macroscopic(solver.f, solver.solid_mask)

    np.testing.assert_allclose(trt.collide(), bgk.collide(), rtol=0.0, atol=1e-13)


def test_collision_parameter_validation():
    cfg = case_preset("custom", nx=24, ny=24)
    cfg.collision_model = "MRT"
    cfg.mrt_preset = "custom"
    cfg.mrt_s_q = 2.0

    errors, _ = validate_config(cfg)

    assert any("mrt_s_q" in error for error in errors)


def test_mrt_bgk_equivalent_preset_matches_bgk():
    bgk_cfg = case_preset("custom", nx=12, ny=10)
    bgk_cfg.reynolds = 20.0
    bgk_cfg.u_ref = 0.02
    mrt_cfg = case_preset("custom", nx=12, ny=10)
    mrt_cfg.reynolds = bgk_cfg.reynolds
    mrt_cfg.u_ref = bgk_cfg.u_ref
    mrt_cfg.collision_model = "MRT"
    mrt_cfg.mrt_preset = "bgk_equivalent"

    bgk = LBMSolver(bgk_cfg)
    mrt = LBMSolver(mrt_cfg)
    rng = np.random.default_rng(19)
    populations = bgk.f * (1.0 + rng.normal(0.0, 1e-3, bgk.f.shape))
    for solver in (bgk, mrt):
        solver.f = populations.copy()
        solver.rho, solver.ux, solver.uy = macroscopic(solver.f, solver.solid_mask)

    np.testing.assert_allclose(mrt.collide(), bgk.collide(), rtol=0.0, atol=1e-13)


def test_direct_tau_mode_controls_viscosity_and_effective_reynolds():
    cfg = case_preset("lid_driven_cavity", nx=40, ny=40)
    cfg.parameter_mode = "tau"
    cfg.tau_target = 0.72
    cfg.u_ref = 0.04

    assert cfg.tau == pytest.approx(0.72)
    assert cfg.nu_lattice == pytest.approx((0.72 - 0.5) / 3.0)
    assert cfg.effective_reynolds == pytest.approx(
        cfg.u_ref * cfg.characteristic_length / cfg.nu_lattice
    )


@pytest.mark.parametrize("profile", ["linear", "smoothstep", "exponential"])
def test_ramp_profiles_reach_one(profile):
    cfg = case_preset("custom", nx=12, ny=10)
    cfg.ramp_profile = profile
    cfg.ramp_steps = 20
    solver = LBMSolver(cfg)

    solver.iteration = 5
    early = solver.ramp_factor()
    solver.iteration = 20
    final = solver.ramp_factor()

    assert 0.0 < early < 1.0
    assert final == pytest.approx(1.0)


def test_mixed_boundary_rb_endpoints_match_bounce_and_specular():
    cfg = case_preset("custom", nx=5, ny=4)
    original = np.arange(4 * 5 * 9, dtype=float).reshape(4, 5, 9)

    pure_bounce = original.copy()
    mixed_bounce_specular(pure_bounce, "left", BoundaryConfig(rb=1.0), cfg, 1.0)
    np.testing.assert_array_equal(pure_bounce[:, 0, 5], original[:, 0, 7])
    np.testing.assert_array_equal(pure_bounce[:, 0, 8], original[:, 0, 6])

    pure_specular = original.copy()
    mixed_bounce_specular(pure_specular, "left", BoundaryConfig(rb=0.0), cfg, 1.0)
    expected_specular = original.copy()
    specular_reflection(expected_specular, "left", BoundaryConfig(), cfg, 1.0)
    np.testing.assert_array_equal(pure_specular[:, 0, :], expected_specular[:, 0, :])


def test_y_body_force_drives_periodic_flow():
    cfg = case_preset("custom", nx=12, ny=10)
    for side in ("left", "right", "bottom", "top"):
        getattr(cfg, side).type = "periodic"
    cfg.body_force_y = 1e-6
    cfg.ramp_profile = "instant"
    solver = LBMSolver(cfg)

    for _ in range(5):
        solver.step()

    assert np.mean(solver.uy) > 0.0
    assert np.isfinite(solver.f).all()


def test_mass_drift_limit_marks_run_as_diverged():
    cfg = case_preset("custom", nx=12, ny=10)
    solver = LBMSolver(cfg)
    solver.rho *= 1.0 + 2.0 * cfg.mass_drift_limit

    report = solver.make_report()

    assert report.status == "diverged"


def test_periodic_pair_check():
    cfg = case_preset("custom", nx=24, ny=24)
    cfg.left.type = "periodic"
    cfg.right.type = "no_slip_bounce_back"

    errors, _ = validate_config(cfg)

    assert any("left and right" in err for err in errors)


def test_parameter_recommendation():
    rec = recommend_u_ref(reynolds=1000, nx=128, ny=128, case_type="lid_driven_cavity")

    assert rec["U_ref"] > 0
    assert rec["recommended_L_ref_min"] > 0
