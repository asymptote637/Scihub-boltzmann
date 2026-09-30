"""Load and verify the unchanged CPU reference included in this experiment."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "reference_cpu"
PLAN = json.loads((REFERENCE / "plan.json").read_text(encoding="utf-8"))


def verify_reference():
    for name, expected in PLAN["source_sha256"].items():
        actual = hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("Frozen CPU reference changed: " + name)
    return len(PLAN["source_sha256"])


verify_reference()
sys.path.insert(0, str(REFERENCE / "calculation"))
import run_baseline as frozen_runner
from lbm_solver import LBMSolver as CPUSolver
from config import config_to_dict, validate_config


def make_config(grid=256, re=5468, max_iter=1000000, min_iter=2000,
                ramp_steps=500, report_interval=200, tol=1e-6, lid_speed=0.04,
                collision="BGK", boundary="nee", mrt_s_e=1.64, mrt_s_eps=1.54, mrt_s_q=1.9):
    from cavity_models import CavityConfig, COLLISIONS, BOUNDARIES, require_supported
    from config import default_boundaries_for_case
    if collision not in COLLISIONS or boundary not in BOUNDARIES:
        raise ValueError("Unsupported collision model or boundary scheme")
    args = frozen_runner.build_parser().parse_args([
        "--boundary", "nee", "--mass-policy", "diagnostic", "--grid", str(grid),
        "--re", str(re), "--max-iter", str(max_iter), "--min-iter", str(min_iter),
        "--ramp-steps", str(ramp_steps), "--report-interval", str(report_interval),
        "--tol", str(tol), "--lid-speed", str(lid_speed),
    ])
    base = frozen_runner.configure(args)
    config = CavityConfig(**base.__dict__, mrt_s_e=mrt_s_e, mrt_s_eps=mrt_s_eps, mrt_s_q=mrt_s_q)
    config.collision_model = collision
    config.boundary_scheme_default = ("non_equilibrium_extrapolation" if boundary == "nee" else "halfway_bounce_back")
    config.boundaries = default_boundaries_for_case(config.case_type, lid_speed, config.boundary_scheme_default)
    # Let actual wall geometry determine L and therefore nu=U*L/Re.
    config.grid.L_ref = None
    require_supported(config)
    return config
