import json
from pathlib import Path
import numpy as np
import pytest

from dashboard import analysis
from dashboard.store import DEFAULTS, Store, atomic_json, parameters_from_config
from research_options import EXTRA_DEFAULTS, make_config, validate, RECIPES
from research_runtime import solver_class
from reference import config_to_dict


def parameters(**values):
    return validate(DEFAULTS|EXTRA_DEFAULTS|dict(backend="cpu",scenario="thermal_wave",lattice="D3Q19",
                boundary="periodic",run_mode="transient",grid=8,grid_y=10,grid_z=12,max_iter=10,min_iter=0,
                report_interval=5,ramp_steps=0,viscosity=.03,lid_speed=.01)|values)


def test_full_volume_roundtrip_and_axis_slices(tmp_path):
    p=parameters()
    s=solver_class(p)(make_config(p))
    s.config.output.output_dir=str(tmp_path)
    list(s.run());s.finalize()
    fields=analysis.load_fields(tmp_path)
    assert fields["rho"].shape==(12,10,8) and fields["T"].shape==(12,10,8)
    assert fields["finite"] and not fields["preview"]
    assert s.stop_reason=="completed_steps" and s.iteration==10
    for plane,shape,normal,indexaxis in (("xy",(10,8),"z",0),("xz",(12,8),"y",1),("yz",(12,10),"x",2)):
        sliced=analysis.slice_fields(fields,plane,.51)
        index=np.argmin(abs(fields[normal]-.51))
        assert sliced["rho"].shape==shape
        np.testing.assert_array_equal(sliced["T"],np.take(fields["T"],index,axis=indexaxis))
        np.testing.assert_array_equal(sliced["plot_uy"],sliced["u"+plane[1]])
        assert "最近格点" in sliced["slice_caption"]
    restored=parameters_from_config(config_to_dict(s.config,s.transport),"cpu")
    assert restored==p


def test_checkpoint_guards_and_total_budget(tmp_path):
    p=parameters()
    s=solver_class(p)(make_config(p))
    for _ in range(4):s.step()
    path=tmp_path/"checkpoint.npz"
    s.save_checkpoint(path)
    resumed=solver_class(p)(make_config(p|dict(resume_from=str(path),max_iter=7)))
    list(resumed.run())
    assert resumed.iteration==7 and resumed.stop_reason=="completed_steps"
    with pytest.raises(ValueError,match="物理参数"):
        solver_class(p)(make_config(p|dict(resume_from=str(path),diffusivity=.03)))
    with pytest.raises(ValueError,match="总步数"):
        solver_class(p)(make_config(p|dict(resume_from=str(path),max_iter=4)))


def test_thermal_physical_signature_rejects_different_diffusivity():
    a=dict(config=config_to_dict(make_config(parameters(diffusivity=.02))))
    b=dict(config=config_to_dict(make_config(parameters(diffusivity=.03))))
    assert analysis.physical_signature(a)!=analysis.physical_signature(b)
    c=dict(config=config_to_dict(make_config(parameters(backend="fused",collision="TRT"))))
    assert analysis.physical_signature(a)==analysis.physical_signature(c)


def test_explicit_viscosity_has_correct_effective_reynolds_number():
    cfg=make_config(parameters(re=1e4,viscosity=.03))
    assert cfg.flow.Re==pytest.approx(.01*10/.03)
    assert cfg.research["re"]==1e4


def test_invalid_batch_is_atomic(tmp_path):
    store=Store(tmp_path)
    good=parameters()
    bad=dict(good,lattice="D3Q19",collision="MRT")
    with pytest.raises(ValueError):store.enqueue([dict(params=good),dict(params=bad)])
    assert store.cases()==[]


@pytest.mark.parametrize("name",RECIPES)
def test_recipe_config_roundtrip(name):
    p=validate(DEFAULTS|RECIPES[name])
    cfg=make_config(p)
    assert parameters_from_config(config_to_dict(cfg),p["backend"])==p


def test_3d_difference_accounts_for_spanwise_velocity():
    p=parameters()
    grid=np.zeros((12,10,8))
    fields=dict(rho=grid+1,ux=grid.copy(),uy=grid.copy(),uz=grid.copy(),
                x=(np.arange(8)+.5)/8,y=(np.arange(10)+.5)/10,z=(np.arange(12)+.5)/12,preview=False,finite=True)
    a=dict(config=config_to_dict(make_config(p)),fields=fields,status="completed_steps",metrics=dict(iteration=10))
    b=dict(a,fields=dict(fields,uz=grid+.1))
    result=analysis.difference(a,b)
    assert result["max_velocity_difference"]==.1 and result["relative_velocity_l2"] is None


def test_3d_preview_has_bounded_size_and_coordinates(tmp_path):
    from dashboard.worker import write_preview
    p=parameters()
    s=solver_class(p)(make_config(p))
    write_preview(s,tmp_path)
    fields=analysis.load_fields(tmp_path,preview=True)
    assert fields["preview"] and fields["rho"].shape==(12,10,8)
    np.testing.assert_allclose(fields["z"],(np.arange(12)+.5)/12)


def test_rectangular_streamlines_use_physical_direction(monkeypatch):
    from dashboard.fieldplots import draw_field
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    observed={}
    def capture(self,x,y,u,v,**kwargs):
        observed.update(u=u,v=v)
    monkeypatch.setattr(Axes,"streamplot",capture)
    fields=dict(x=(np.arange(16)+.5)/16,y=(np.arange(8)+.5)/8,
                rho=np.ones((8,16)),ux=np.ones((8,16))*.01,uy=np.ones((8,16))*.01,
                axis_lengths=dict(x=16,y=8))
    axes=draw_field(Figure(),fields,"streamlines",speed=.01)
    np.testing.assert_allclose(observed["v"]/observed["u"],2)
    assert axes[0].get_aspect()==.5
