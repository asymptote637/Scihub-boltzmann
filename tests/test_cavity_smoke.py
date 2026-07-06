from pathlib import Path

from lbm_lab.config import load_config
from lbm_lab.simulations.cavity import run


def test_cavity_smoke_runs():
    config = load_config(Path("configs/lid_driven_cavity.toml"))
    rows = list(run(config))

    assert rows
    state, metric = rows[-1]
    assert state.step == config.time.steps
    assert metric.max_speed > 0.0

