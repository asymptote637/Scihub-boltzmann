"""Analytic and conservation checks for open, unsteady and buoyant experiments."""
import math
from pathlib import Path
import tempfile
import numpy as np
from validate_research import solver, advance, params


def open_channel():
    results={}
    for model in ('BGK','TRT','MRT'):
        s=solver(scenario='open_channel',boundary='inlet_outlet',collision=model,backend='fused',
                 grid=64,grid_y=16,viscosity=.08,lid_speed=.015,ramp_steps=100)
        advance(s,6000);s.report();h=s.host
        y=(np.arange(s.ny)+.5)/s.ny;exact=.015*4*y*(1-y)
        error=float(np.linalg.norm(h.ux[:,32]-exact)/np.linalg.norm(exact))
        row=s.history[-1]
        assert error<.015 and row['mass_drift']<2e-11 and row['mass_change']>1e-4,(model,error,row)
        assert abs(row['flux_imbalance'])/row['inlet_flux']<.003,row
        np.testing.assert_allclose(h.ux[:,0],exact,atol=3e-15)
        np.testing.assert_allclose(h.rho[:,-1],1,atol=3e-15)
        results[model]=dict(relative_l2=error,mass_balance_error=row['mass_drift'],
                            physical_mass_change=row['mass_change'],relative_flux_imbalance=abs(row['flux_imbalance'])/row['inlet_flux'])
    return results


def harmonic_flows():
    result={}
    H,nu,period,A,U=16,.1,800,1e-5,.01
    omega=2*math.pi/period;k=np.sqrt(1j*omega/nu);y=np.arange(H)+.5
    for case in ('oscillatory_channel','oscillating_couette'):
        s=solver(scenario=case,boundary='halfway',collision='TRT',backend='fused',grid=8,grid_y=H,
                 viscosity=nu,lid_speed=U,drive_period=period,force_amplitude=A,drive_phase=.31)
        profile=(A/(1j*omega)*(1-np.cosh(k*(y-H/2))/np.cosh(k*H/2)) if case=='oscillatory_channel'
                 else U*np.sinh(k*y)/np.sinh(k*H))
        advance(s,4*period)
        actual=[];expected=[]
        for _ in range(16):
            advance(s,period//16);s.sync_host()
            actual.append(s.host.ux.mean(axis=1))
            expected.append(np.real(profile*np.exp(1j*(omega*s.iteration+.31))))
        error=float(np.linalg.norm(np.array(actual)-expected)/np.linalg.norm(expected))
        assert error<.015,(case,error)
        # Complex first harmonic checks phase as well as a coincidental profile.
        angles=omega*(np.arange(16)+1)*period/16+.31
        measured=2*np.mean(np.asarray(actual)*np.exp(-1j*angles[:,None]),axis=0)
        complex_error=float(np.linalg.norm(measured-profile)/np.linalg.norm(profile))
        assert complex_error<.015,(case,complex_error)
        result[case]=dict(space_time_relative_l2=error,complex_harmonic_relative_l2=complex_error,
                          height=H,period=period,viscosity=nu)
    return result


def rb_conduction():
    results={}
    for lat in ('D2Q9','D3Q19','D3Q27'):
        s=solver(scenario='rayleigh_benard',lattice=lat,collision='TRT',boundary='halfway',grid=16,grid_y=8,grid_z=8,
                 backend='fused',rayleigh=0,thermal_perturbation=0,viscosity=.02)
        advance(s,150);s.report()
        target=1-(np.arange(s.ny)+.5)/s.ny
        target=np.broadcast_to(target[:,None],s.shape)
        err=float(np.max(abs(s.host.T-target)));row=s.history[-1]
        assert err<1e-13 and abs(row['nusselt_hot']-1)<1e-12 and abs(row['nusselt_cold']-1)<1e-12
        results[lat]=dict(temperature_error=err,nu_hot=row['nusselt_hot'],nu_cold=row['nusselt_cold'])
    return results


def rb_growth():
    results={}
    for ra in (0,5000):
        s=solver(scenario='rayleigh_benard',lattice='D2Q9',collision='MRT',boundary='halfway',backend='fused',
                 grid=32,grid_y=16,viscosity=.02,rayleigh=ra,thermal_perturbation=.005)
        advance(s,18000);s.report();r=s.history[-1]
        results[str(ra)]=dict(nu_hot=r['nusselt_hot'],nu_cold=r['nusselt_cold'],max_mach=r['max_mach'],
                              kinetic_energy=r['kinetic_energy'],thermal_residual=r['thermal_residual'])
    assert abs(results['0']['nu_hot']-1)<1e-6,results
    assert results['5000']['nu_hot']>1.1 and results['5000']['max_mach']<.1,results
    assert abs(results['5000']['nu_hot']-results['5000']['nu_cold'])<.02,results
    return results


def momentum_exchange():
    s=solver(scenario='obstacle',boundary='halfway',collision='TRT',grid=32,grid_y=24,obstacle_size=.1,
             obstacle_x=.5,obstacle_y=.5,obstacle_count_x=2,obstacle_spacing_x=.4)
    # Remove outer walls for this exact momentum budget limit: fluid + bodies.
    s.periodic=[True,True];s.links=s.build_links(s.lat)
    errors=[]
    for _ in range(20):
        before=np.array([(s.rho*s.u[a])[s.fluid].sum() for a in range(2)])
        drive=np.array([(s.rho*s.a[a])[s.fluid].sum() for a in range(2)])
        s.step()
        after=np.array([(s.rho*s.u[a])[s.fluid].sum() for a in range(2)])
        force=s.obstacle_force();body=np.array([force['force_x'],force['force_y']])
        errors.append(float(np.max(abs(after-before-drive+body))))
    assert max(errors)<2e-12,errors
    return dict(max_momentum_budget_error=max(errors),obstacles=2,steps=20)


def gpu_new_cases():
    import gpu_environment
    import cupy as cp
    results={}
    cases=[]
    for scenario in ('open_channel','cylinder_wake','oscillatory_channel','oscillating_couette','rayleigh_benard','mixed_convection'):
        for model in ('BGK','TRT','MRT'):
            cases.append(dict(scenario=scenario,collision=model,lattice='D2Q9'))
    cases.extend(dict(scenario=s,lattice=lat,collision='TRT') for lat in ('D3Q19','D3Q27')
                 for s in ('oscillatory_channel','oscillating_couette','rayleigh_benard','mixed_convection','obstacle'))
    for settings in cases:
        sname=settings['scenario'];dim=3 if settings['lattice'].startswith('D3') else 2
        kwargs=dict(grid=24,grid_y=16,grid_z=16,lid_speed=.003,rho0=1.3,viscosity=.05,drive_period=40,drive_phase=.73,
                    force_amplitude=1e-5,rayleigh=100,obstacle_size=.06,obstacle_x=.5,obstacle_y=.5,
                    obstacle_count_x=2,obstacle_spacing_x=.4,probes=[[.7,.4,.5][:dim]],outlet_rho=1.3,
                    boundary='inlet_outlet' if sname in {'open_channel','cylinder_wake'} else 'halfway',**settings)
        cpu=solver(**kwargs)
        advance(cpu,33);cpu.report()
        for backend in ('array','fused'):
            gpu=solver(**kwargs,backend=backend);advance(gpu,33);gpu.report()
            keys=['f','rho','ux','uy','uz']+(['g','T'] if cpu.T is not None else [])
            error=max(float(np.max(abs(cp.asnumpy(getattr(gpu,k))-getattr(cpu,k)))) for k in keys)
            if sname in {'obstacle','cylinder_wake'}:
                for key in ('force_x','force_y','force_z'):
                    error=max(error,abs(cpu.history[-1][key]-gpu.history[-1][key]))
            assert error<5e-12,(settings,backend,error)
            results['/'.join([settings['lattice'],settings['collision'],sname,backend])]=error
    return results


def advanced_restart():
    from research_validation import ROOT
    results={}
    with tempfile.TemporaryDirectory(dir=ROOT/'validation') as folder:
        for scenario in ('open_channel','oscillatory_channel','oscillating_couette','rayleigh_benard'):
            kw=dict(scenario=scenario,boundary='inlet_outlet' if scenario=='open_channel' else 'halfway',
                    collision='TRT',grid=16,grid_y=16,rayleigh=100,probes=[[.5,.5]])
            a=solver(**kw);advance(a,17);path=Path(folder)/'checkpoint.npz';a.save_checkpoint(path)
            b=solver(**kw,resume_from=str(path));advance(a,24);advance(b,24)
            err=max(float(np.max(abs(a.f-b.f))),abs(a.boundary_exchange-b.boundary_exchange))
            if a.T is not None:err=max(err,float(np.max(abs(a.T-b.T))))
            assert err<1e-14,(scenario,err)
            results[scenario]=err
    return results


CHECKS=(('open_channel',open_channel),('harmonic_flows',harmonic_flows),('rb_conduction',rb_conduction),
        ('rb_growth',rb_growth),('momentum_exchange',momentum_exchange),('advanced_gpu',gpu_new_cases),
        ('advanced_restart',advanced_restart))

if __name__=='__main__':
    import json
    for name,check in CHECKS:
        print(name,json.dumps(check()),flush=True)
