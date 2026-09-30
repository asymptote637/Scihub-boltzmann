import pytest

from training.shear_wave_viscosity import (
    d2q9_isotropy_errors,
    equilibrium_moment_errors,
    run_viscosity_case,
)


def test_d2q9_isotropy_and_equilibrium_moments():
    assert max(d2q9_isotropy_errors().values()) < 1e-14
    assert max(equilibrium_moment_errors().values()) < 1e-14


def test_shear_wave_recovers_bgk_viscosity():
    result, samples = run_viscosity_case(
        tau=0.8,
        nx=32,
        ny=48,
        amplitude=0.02,
        steps=180,
        sample_every=5,
        fit_start=20,
    )

    assert result.nu_theory == pytest.approx(0.1)
    assert abs(result.relative_viscosity_error) < 3e-3
    assert result.relative_mass_drift_max < 1e-12
    assert samples[-1].amplitude < samples[0].amplitude

