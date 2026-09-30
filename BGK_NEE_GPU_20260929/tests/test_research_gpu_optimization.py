"""Contract tests for the optimized device path, including stopping and restart."""
import math
import numpy as np
import pytest

import gpu_environment
cp=pytest.importorskip('cupy')
from validate_research import params, solver
from research_runtime import build_config, solver_class


@pytest.mark.parametrize('lattice,model,scenario',[
    ('D2Q9','TRT','open_channel'),('D2Q9','MRT','rayleigh_benard'),
    ('D2Q25','TRT','thermal_wave'),('D3Q19','TRT','obstacle'),
    ('D3Q27','BGK','mixed_convection')])
def test_diagnostics_match_independent_full_field_calculation(lattice,model,scenario):
    kw=dict(lattice=lattice,collision=model,scenario=scenario,grid=9,grid_y=11,grid_z=8,
            boundary='inlet_outlet' if scenario=='open_channel' else 'periodic' if scenario=='thermal_wave' else 'halfway',
            lid_speed=.003,viscosity=.06,rayleigh=100,ramp_steps=5,rho0=1.3,outlet_rho=1.3,
            probes=[[.7,.4,.5][:3 if lattice.startswith('D3') else 2]])
    cpu=solver(**kw);gpu=solver(**kw,backend='fused')
    for _ in range(7):
        previous=[cp.asnumpy(u) for u in gpu.u]
        oldT=cp.asnumpy(gpu.T) if gpu.T is not None else None
        cpu.step();gpu.step()
        fluid=cp.asnumpy(gpu.fluid)
        u=[cp.asnumpy(a) for a in gpu.u]
        rho=cp.asnumpy(gpu.rho)
        numerator=sum(((a-b)[fluid]**2).sum() for a,b in zip(u,previous))
        denominator=sum((a[fluid]**2).sum() for a in u)
        expected=math.sqrt(numerator/max(denominator,1e-30))
        assert gpu._last_residual==pytest.approx(expected,rel=3e-13,abs=3e-14)
        if oldT is not None:
            T=cp.asnumpy(gpu.T)
            thermal=math.sqrt(((T-oldT)[fluid]**2).sum()/max((T[fluid]**2).sum(),1e-30))
            assert gpu.thermal_residual==pytest.approx(thermal,rel=3e-13,abs=3e-14)
        gpu.report();cpu.report()
        row=gpu.history[-1]
        speed=sum(a*a for a in u)
        assert row['max_velocity']==pytest.approx(math.sqrt(speed[fluid].max()),rel=2e-14)
        assert row['kinetic_energy']==pytest.approx(.5*speed[fluid].mean(),rel=2e-14)
        assert row['mass_change']==pytest.approx((rho[fluid].sum()-gpu.mass0)/gpu.mass0,abs=5e-15)
        if scenario=='open_channel':
            assert row['outlet_backflow_fraction']==float(np.mean(u[0][:,-1]<0))
            # The strict sign of an initially motionless outlet is undefined at
            # the roundoff scale; compare signs only for resolved velocities.
            resolved=abs(cpu.ux[:,-1])>3e-13
            np.testing.assert_array_equal(u[0][:,-1][resolved]<0,cpu.ux[:,-1][resolved]<0)
        for key,value in cpu.history[-1].items():
            if isinstance(value,(int,float)) and key not in {'residual','thermal_residual','outlet_backflow_fraction'}:
                # Coefficients divide force cancellation noise by rho*U^2*A/2.
                atol=3e-12
                if key.endswith('_coefficient'):
                    atol/=(.5*kw['rho0']*kw['lid_speed']**2*row['force_reference_area'])
                assert row[key]==pytest.approx(value,rel=1e-9,abs=atol),(key,row[key],value)
        for a,b in zip(row['probes'],cpu.history[-1]['probes']):
            assert a['valid']==b['valid']
            for key in ('ux','uy','uz','rho','T','pressure'):
                assert a[key]==pytest.approx(b[key],abs=3e-12) if b[key] is not None else a[key] is None


@pytest.mark.parametrize('mode',['convergence','mass_guard','mach','negative_density','nan_fluid','nan_solid'])
def test_every_step_stop_guards(mode):
    scenario='obstacle' if mode=='nan_solid' else 'couette'
    kw=dict(lattice='D3Q19',collision='TRT',scenario=scenario,boundary='halfway',
            grid=9,grid_y=11,grid_z=8,lid_speed=.003,viscosity=.06,run_mode='steady',min_iter=5,ramp_steps=0,tol=.5)
    states=[]
    for backend in ('cpu','fused'):
        s=solver(**kw,backend=backend)
        s.config.convergence.consecutive_reports=3
        if mode=='mass_guard':s.mass0*=1.01
        if mode=='mach':s.config.convergence.max_mach=.0001
        if mode=='negative_density':s.f[0,0,0,:]=-1
        if mode in {'nan_fluid','nan_solid'}:
            mask=np.asarray(s.solid_mask) if backend=='cpu' else cp.asnumpy(s.solid_mask)
            where=tuple(np.argwhere(mask if mode=='nan_solid' else ~mask)[0])
            s.f[where+(0,)]=float('nan')
        with np.errstate(invalid='ignore',divide='ignore'):
            for _ in range(10):
                s.step()
                if s.stop_reason!='running':break
        states.append((s.stop_reason,s.iteration,s._failure))
    assert states[0]==states[1],states
    if mode=='convergence':assert states[0][:2]==('converged',7)
    elif mode=='mass_guard':assert states[0][0]=='running'
    else:assert states[0][0]=='diverged'


@pytest.mark.parametrize('scenario,lattice',[
    ('thermal_wave','D3Q19'),('open_channel','D2Q9'),('oscillatory_channel','D3Q27'),('rayleigh_benard','D2Q9')])
def test_device_checkpoint_restart_preserves_phase_and_fields(tmp_path,scenario,lattice):
    kw=dict(backend='fused',lattice=lattice,collision='TRT',scenario=scenario,grid=9,grid_y=11,grid_z=8,
            boundary='inlet_outlet' if scenario=='open_channel' else 'periodic' if scenario=='thermal_wave' else 'halfway',
            lid_speed=.003,viscosity=.06,rayleigh=100,drive_period=40,drive_phase=.73,force_amplitude=1e-5)
    a=solver(**kw)
    for _ in range(13):a.step()
    file=tmp_path/'checkpoint.npz';a.save_checkpoint(file)
    b=solver(**kw,resume_from=str(file));b.report()
    for _ in range(17):a.step();b.step()
    for field in ('f','g','rho','ux','uy','uz','T'):
        if getattr(a,field) is not None:
            np.testing.assert_allclose(cp.asnumpy(getattr(a,field)),cp.asnumpy(getattr(b,field)),rtol=0,atol=3e-13)
    assert a.boundary_exchange==pytest.approx(b.boundary_exchange,abs=3e-12)


def test_progress_only_transfers_compact_statistics(monkeypatch,tmp_path):
    s=solver(backend='fused',lattice='D3Q19',collision='TRT',scenario='cavity',boundary='halfway',grid=16,grid_y=16,grid_z=16)
    original=cp.asnumpy;transfers=[]
    def tracked(a,*args,**kwargs):
        transfers.append(a.nbytes)
        return original(a,*args,**kwargs)
    monkeypatch.setattr(cp,'asnumpy',tracked)
    for _ in range(4):s.step();s.report()
    assert transfers==[120]*4
    s.config.output.output_dir=str(tmp_path)
    result=s.finalize(save_outputs=False)
    np.testing.assert_array_equal(result.ux,original(s.ux))
    np.testing.assert_array_equal(s.host.f,original(s.f))


def test_cancel_does_not_advance_or_lose_final_fields():
    s=solver(backend='fused',lattice='D3Q19',collision='TRT',scenario='cavity',boundary='halfway',grid=8,grid_y=8)
    s.step();state=cp.asnumpy(s.f);s.cancel();s.step()
    assert s.iteration==1 and s.report().stop_reason=='cancelled'
    np.testing.assert_array_equal(cp.asnumpy(s.f),state)


def test_explicit_host_reads_remain_current_without_reporting():
    s=solver(backend='fused',lattice='D3Q19',collision='TRT',scenario='thermal_wave',boundary='periodic',grid=8,grid_y=8)
    s.step()
    np.testing.assert_array_equal(s.host.T,cp.asnumpy(s.T))
    s.sync_host();previous=s.host.ux.copy()
    s.step()
    np.testing.assert_array_equal(s.host.ux,cp.asnumpy(s.ux))
    np.testing.assert_array_equal(s.host.T,cp.asnumpy(s.T))
