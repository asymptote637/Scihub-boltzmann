from config import case_preset, recommend_u_ref, validate_config
from lbm_solver import LBMSolver


def test_lid_driven_cavity_core_smoke():
    cfg = case_preset("lid_driven_cavity", nx=24, ny=24)
    cfg.reynolds = 50
    cfg.u_ref = 0.03
    cfg.top.ux = cfg.u_ref
    cfg.max_iter = 20
    cfg.min_iter = 10
    cfg.report_interval = 10

    errors, warnings = validate_config(cfg)
    assert not errors

    solver = LBMSolver(cfg)
    reports = list(solver.run())

    assert reports[-1].iteration == 20
    assert solver.f.shape == (24, 24, 9)
    assert solver.rho.shape == (24, 24)


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

