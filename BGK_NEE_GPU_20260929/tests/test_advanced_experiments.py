import json
import numpy as np
import pytest
from dashboard import analysis, experiments
from dashboard.store import Store
from research_options import validate, make_config
from research_solver import ResearchSolver
from reference import config_to_dict
from validate_research import params


def test_fft_recovers_known_frequency_and_amplitude_with_edge_reports():
    times=np.r_[1,np.arange(20,20*257,20),5141]
    omega=2*np.pi*13/(256*20)
    signal=3.7+2.25*np.cos(omega*times+.42)
    result=experiments.spectrum(times,signal)
    assert result['samples']==256 and result['trimmed']==2
    assert result['peak_frequency']==pytest.approx(13/(256*20))
    assert result['peak_amplitude']==pytest.approx(2.25,rel=2e-4)
    assert result['nyquist']==.025 and result['resolved']


def test_fft_rejects_interior_gap_and_invalid_probe():
    t=np.arange(128)*10;y=np.sin(t)
    with pytest.raises(ValueError,match='间隔不均匀'):
        experiments.spectrum(np.delete(t,50),np.delete(y,50))
    with pytest.raises(ValueError,match='有效'):
        experiments.spectrum(t,np.full_like(t,np.nan,dtype=float))
    assert not experiments.spectrum(t,np.ones_like(t))['resolved']


def test_two_axis_matrix_order_and_atomic_validation(tmp_path):
    p=params(scenario='rayleigh_benard',boundary='halfway')
    requests=experiments.scan_requests(p,[('rayleigh','0,1000'),('prandtl','0.7,1,2')],'matrix','group')
    assert [(r['params']['rayleigh'],r['params']['prandtl']) for r in requests]==[(0,.7),(0,1),(0,2),(1000,.7),(1000,1),(1000,2)]
    store=Store(tmp_path)
    assert len(store.enqueue(requests))==6
    with pytest.raises(ValueError):experiments.scan_requests(p,[('rayleigh','0,1000'),('prandtl','0.7,-1')])
    assert len(store.cases())==6
    with pytest.raises(ValueError,match='不同'):experiments.scan_requests(p,[('rayleigh','1'),('rayleigh','2')])
    with pytest.raises(ValueError,match='100'):experiments.scan_requests(p,[('rayleigh',','.join(map(str,range(11)))),('prandtl',','.join(['1']*10))])


def test_configuration_roundtrip_and_schema_guard(tmp_path):
    p=params(scenario='oscillatory_channel',boundary='halfway',probes='0.4,0.5;0.8,0.7',drive_phase=.7)
    path=tmp_path/'实验配置.json';experiments.save_configuration(path,p,'周期流','验证')
    restored=experiments.read_configuration(path)
    assert restored['params']==p and restored['name']=='周期流'
    path.write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='版本 1'):experiments.read_configuration(path)


def test_point_probe_solid_null_and_saved_snapshots(tmp_path):
    p=params(scenario='obstacle',boundary='halfway',grid=16,grid_y=16,collision='TRT',max_iter=6,
             snapshot_interval=2,probes=[[.35,.5],[.8,.5]])
    cfg=make_config(p);cfg.output.output_dir=str(tmp_path)
    s=ResearchSolver(cfg);list(s.run());s.finalize()
    probes=s.history[-1]['probes']
    assert not probes[0]['valid'] and probes[0]['ux'] is None and probes[0]['pressure'] is None
    assert probes[1]['valid'] and np.isfinite(probes[1]['pressure'])
    assert (tmp_path/'probes.csv').is_file()
    frames=experiments.snapshots(tmp_path)
    assert [f[0] for f in frames]==[2,4,6]
    fields=experiments.load_snapshot(frames[0][1],tmp_path)
    assert fields['snapshot'] and fields['preview_iteration']==2 and fields['rho'].shape==(16,16)
    with pytest.raises(ValueError,match='当前算例'):experiments.load_snapshot(tmp_path/'results.npz',tmp_path)


@pytest.mark.parametrize('override',[
    dict(scenario='open_channel',boundary='inlet_outlet',lattice='D3Q19'),
    dict(scenario='open_channel',boundary='inlet_outlet',force_x=1e-6),
    dict(scenario='oscillatory_channel',boundary='halfway',run_mode='steady'),
    dict(scenario='obstacle',boundary='halfway',obstacle_count_x=4,obstacle_size=.2),
    dict(scenario='obstacle',boundary='halfway',obstacle_count_x=2,obstacle_spacing_x=.01),
    dict(probes=[[.5,.5,.5]]),dict(probes=[[.5,float('nan')]]),dict(drive_period=0)
])
def test_incompatible_advanced_requests_fail(override):
    with pytest.raises(ValueError):params(**override)


@pytest.mark.parametrize('scenario,key,value',[
    ('rayleigh_benard','rayleigh',100),('mixed_convection','prandtl',1),
    ('oscillatory_channel','drive_period',100),('cylinder_wake','inlet_profile','uniform'),
    ('obstacle','obstacle_count_x',2)
])
def test_comparison_rejects_changed_experiment_physics(scenario,key,value):
    p=params(scenario=scenario,boundary='inlet_outlet' if scenario=='cylinder_wake' else 'halfway',
             obstacle_x=.5,obstacle_size=.05,obstacle_spacing_x=.4)
    a=dict(config=config_to_dict(make_config(p)))
    b=dict(config=config_to_dict(make_config(p|{key:value})))
    assert analysis.physical_signature(a)!=analysis.physical_signature(b)


def test_open_channel_coordinates_and_mass_accounting(tmp_path):
    p=params(scenario='open_channel',boundary='inlet_outlet',grid=24,grid_y=16,probes=[[0,.5],[1,.5]],max_iter=20)
    cfg=make_config(p);cfg.output.output_dir=str(tmp_path)
    s=ResearchSolver(cfg);list(s.run());s.finalize()
    f=analysis.load_fields(tmp_path)
    assert f['x'][0]==0 and f['x'][-1]==1 and f['axis_lengths']==dict(x=23,y=16)
    assert s.history[-1]['mass_drift']<1e-13 and s.history[-1]['mass_change']>0
    assert s.history[-1]['probes'][0]['coordinates'][0]==0
    assert s.history[-1]['probes'][1]['coordinates'][0]==1


def test_old_single_obstacle_config_keeps_comparison_compatibility():
    import copy
    p=params(scenario='obstacle',boundary='halfway')
    config=config_to_dict(make_config(p));old=copy.deepcopy(config)
    for key in list(old['research']):
        if key.startswith(('obstacle_count_','obstacle_spacing_')):old['research'].pop(key)
    assert analysis.physical_signature(dict(config=config))==analysis.physical_signature(dict(config=old))
