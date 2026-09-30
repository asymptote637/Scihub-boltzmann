"""Reproducible quadrature, analytic-flow, heat, GPU, and restart acceptance."""
import json
import math
from pathlib import Path
import tempfile
import time
import sys
import numpy as np

from dashboard.store import DEFAULTS, atomic_json, validate_parameters, read_json
from lattices import lattice
from research_options import EXTRA_DEFAULTS, RECIPES
from research_runtime import build_config, solver_class
from research_validation import ROOT, fingerprints


def params(**overrides):
    return validate_parameters(DEFAULTS|EXTRA_DEFAULTS|dict(backend="cpu",scenario="shear_wave",lattice="D2Q9",
             boundary="periodic",collision="BGK",grid=16,grid_y=16,grid_z=8,viscosity=.05,ramp_steps=0,
             max_iter=100000,min_iter=100000,lid_speed=.01,run_mode="transient",report_interval=500)|overrides)


def solver(**overrides):
    p = params(**overrides)
    return solver_class(p)(build_config(p))


def advance(s,n):
    for _ in range(n):
        s.step()
        if s.stop_reason!="running": break
    if s.stop_reason=="diverged": raise AssertionError(s._failure)
    return s


def quadratures():
    errors = {}
    for name in ("D2Q9","D2Q25","D3Q19","D3Q27","D2Q5","D3Q7"):
        lat = lattice(name)
        assert lat.w.min()>0
        np.testing.assert_allclose(lat.e[lat.opp],-lat.e,atol=0)
        error = abs(lat.w.sum()-1)
        for a in range(lat.dim):
            error = max(error,abs(np.sum(lat.w*lat.e[:,a])))
            for b in range(lat.dim):
                error = max(error,abs(np.sum(lat.w*lat.e[:,a]*lat.e[:,b])-lat.cs2*(a==b)))
                if name not in {"D2Q5","D3Q7"}:
                    for c in range(lat.dim):
                        for d in range(lat.dim):
                            target = lat.cs2**2*((a==b)*(c==d)+(a==c)*(b==d)+(a==d)*(b==c))
                            error = max(error,abs(np.sum(lat.w*lat.e[:,a]*lat.e[:,b]*lat.e[:,c]*lat.e[:,d])-target))
        if name=="D2Q25":
            error=max(error,abs(np.sum(lat.w*lat.e[:,0]**6)-15*lat.cs2**3))
        assert error<2e-14,(name,error)
        errors[name]=float(error)
    return errors


def exact_force():
    errors = {}
    for lat in ("D2Q9","D2Q25","D3Q19","D3Q27"):
        for model in ("BGK","TRT","MRT") if lat=="D2Q9" else ("BGK","TRT"):
            s=solver(lattice=lat,collision=model,grid=8,grid_y=8,rho0=1.7,force_x=1e-6,force_y=-2e-6,
                     force_z=3e-6 if lat.startswith("D3") else 0)
            for u in s.u: u.fill(0)
            s.f=s.equilibrium()-.5*s.source()
            advance(s,25)
            error=max(float(np.max(abs(u-25*s.p["force_"+"xyz"[a]]))) for a,u in enumerate(s.u))
            mass=abs(float(s.rho.sum())/s.mass0-1)
            assert error<1e-13 and mass<1e-13,(lat,model,error,mass)
            errors[lat+"/"+model]=dict(velocity=error,mass=mass)
    return errors


def bgk_trt_limit():
    a=solver(lattice="D3Q27",grid=8,grid_y=8,force_x=1e-6)
    b=solver(lattice="D3Q27",grid=8,grid_y=8,force_x=1e-6,collision="TRT",trt_magic=(a.transport["tau"]-.5)**2)
    advance(a,30); advance(b,30)
    error=float(np.max(abs(a.f-b.f)))
    assert error<3e-15,error
    return dict(max_population_error=error)


def shear_decay():
    errors={}
    for lat in ("D2Q9","D2Q25","D3Q19","D3Q27"):
        s=solver(lattice=lat,grid=8,grid_y=32,lid_speed=.001)
        advance(s,200)
        exact=.001*np.sin(2*np.pi*s.fractions[1])*np.exp(-s.transport["nu"]*(2*np.pi/32)**2*200)
        error=float(np.linalg.norm(s.ux-exact)/np.linalg.norm(exact))
        assert error<.008,(lat,error)
        errors[lat]=error
    return errors


def channels():
    errors={}
    for case,lat,model in (("couette","D3Q19","TRT"),("poiseuille","D2Q9","MRT"),("poiseuille","D3Q27","TRT")):
        s=solver(scenario=case,lattice=lat,collision=model,boundary="halfway",grid=8,grid_y=16,viscosity=.1,lid_speed=.01)
        advance(s,3000)
        yy=s.fractions[1]
        exact=.01*yy if case=="couette" else 4*.01*yy*(1-yy)
        error=float(np.linalg.norm(s.ux-exact)/np.linalg.norm(exact))
        drift=abs(float(s.rho[~s.solid_mask].sum())/s.mass0-1)
        assert error<.009 and drift<2e-12,(case,lat,error,drift)
        errors[case+"/"+lat]=dict(relative_l2=error,mass_drift=drift)
    return errors


def thermal_tests():
    errors={}
    for lat in ("D2Q9","D3Q19"):
        s=solver(scenario="thermal_wave",lattice=lat,grid=32,grid_y=8,grid_z=8,diffusivity=.04,lid_speed=.01)
        advance(s,200)
        exact=.5+.1*np.sin(2*np.pi*(s.fractions[0]-.01*200/32))*np.exp(-.04*(2*np.pi/32)**2*200)
        error=float(np.linalg.norm(s.T-exact)/np.linalg.norm(exact-.5))
        assert error<.012,(lat,error)
        errors[lat+"_advection_diffusion_l2"]=error
        s=solver(scenario="natural_convection",lattice=lat,collision="TRT",boundary="halfway",grid=12,grid_y=12,
                 grid_z=12,rayleigh=0,viscosity=.02)
        advance(s,100)
        error=float(np.max(abs(s.T-(1-s.fractions[0]))))
        s.report()
        assert error<1e-13 and abs(s.history[-1]["nusselt_hot"]-1)<1e-12
        errors[lat+"_conduction_error"]=error
    return errors


def gpu_agreement():
    import gpu_environment
    import cupy as cp
    cases = [(lat,collision,scenario) for lat in ("D2Q9","D2Q25","D3Q19","D3Q27")
             for collision in (("BGK","TRT","MRT") if lat=="D2Q9" else ("BGK","TRT"))
             for scenario in (("shear_wave",) if lat=="D2Q25" else ("obstacle","natural_convection"))]
    cases += [(lat,model,scenario) for lat in ("D3Q19","D3Q27") for model in ("BGK","TRT")
              for scenario in ("cavity","couette")]
    cases += [("D2Q9","TRT","cavity"),("D2Q25","BGK","thermal_wave")]
    errors={}
    for lat,collision,scenario in cases:
        kwargs=dict(lattice=lat,collision=collision,scenario=scenario,boundary="periodic" if lat=="D2Q25" else "halfway",
                    grid=9,grid_y=11,grid_z=8,lid_speed=.003,force_x=1e-7,rho0=1.7,rayleigh=100)
        cpu=solver(**kwargs)
        rng=np.random.default_rng(930)
        cpu.f+=rng.uniform(-1e-7,1e-7,cpu.f.shape)
        cpu.macros()
        mass_before=float(cpu.rho[cpu.fluid].sum())
        gpus=[]
        for backend in ("array","fused"):
            gpu=solver(**kwargs,backend=backend)
            gpu.f=cp.asarray(cpu.f)
            gpu.macros()
            gpus.append(gpu)
        advance(cpu,21)
        assert abs(float(cpu.rho[cpu.fluid].sum())/mass_before-1)<3e-13
        for gpu in gpus:
            advance(gpu,21)
            error=max(float(np.max(abs(cp.asnumpy(getattr(gpu,k))-getattr(cpu,k)))) for k in ("f","rho","ux","uy","uz"))
            if cpu.T is not None:
                error=max(error,float(np.max(abs(cp.asnumpy(gpu.T)-cpu.T))))
            assert error<3e-12,(lat,collision,scenario,gpu.backend,error)
            errors["/".join((lat,collision,scenario,gpu.backend))]=error
    return errors


def natural_convection():
    s=solver(scenario="natural_convection",lattice="D2Q9",collision="MRT",boundary="halfway",backend="fused",
             grid=24,grid_y=24,rayleigh=1000,prandtl=.71,viscosity=.02,lid_speed=.01,run_mode="steady",
             min_iter=2000,max_iter=50000,tol=1e-8)
    for report in s.run():
        if report.iteration%5000==0: print("natural_convection",report.iteration,report.residual,flush=True)
    row=s.history[-1]
    assert s.stop_reason=="converged",(s.stop_reason,row)
    assert abs(row["nusselt_hot"]-1.118)<.035,row
    assert abs(row["nusselt_hot"]-row["nusselt_cold"])<.002,row
    return dict(iteration=s.iteration,nusselt_hot=row["nusselt_hot"],nusselt_cold=row["nusselt_cold"],
                reference_nusselt=1.118,absolute_tolerance=.035,grid="24x24",max_mach=row["max_mach"],
                velocity_residual=row["residual"],thermal_residual=row["thermal_residual"])


def restart():
    with tempfile.TemporaryDirectory(dir=ROOT/"validation") as folder:
        s=solver(lattice="D3Q19",scenario="thermal_wave",grid=8,grid_y=8)
        advance(s,20)
        s.save_checkpoint(Path(folder)/"checkpoint.npz")
        restored=solver(lattice="D3Q19",scenario="thermal_wave",grid=8,grid_y=8,resume_from=str(Path(folder)/"checkpoint.npz"))
        advance(s,30); advance(restored,30)
        errors={k:float(np.max(abs(getattr(s,k)-getattr(restored,k)))) for k in ("f","g","T")}
        assert max(errors.values())<1e-14,errors
        return errors


def compatibility():
    invalid=[dict(lattice="D2Q25",scenario="cavity",boundary="halfway"),dict(lattice="D3Q19",collision="MRT"),
             dict(scenario="thermal_wave",run_mode="steady"),dict(lattice="D2Q9",force_z=1e-7),
             dict(scenario="obstacle",boundary="halfway",obstacle_x=.05,obstacle_size=.4)]
    for overrides in invalid:
        try: params(**overrides)
        except ValueError: pass
        else: raise AssertionError(overrides)
    for recipe in RECIPES.values(): validate_parameters(DEFAULTS|recipe)
    return dict(rejected=len(invalid),recipes=len(RECIPES))


def rectangular_obstacles():
    errors={}
    for lat in ("D2Q9","D3Q19"):
        cpu=solver(lattice=lat,scenario="obstacle",boundary="halfway",obstacle="rectangle",grid=16,grid_y=16,grid_z=16)
        advance(cpu,20)
        for backend in ("array","fused"):
            gpu=solver(lattice=lat,scenario="obstacle",boundary="halfway",obstacle="rectangle",grid=16,grid_y=16,grid_z=16,backend=backend)
            advance(gpu,20)
            error=float(np.max(abs(gpu.xp.asnumpy(gpu.f)-cpu.f)))
            assert error<3e-12
            errors[lat+"/"+backend]=error
    return errors


def main():
    from validate_advanced import CHECKS
    started=time.perf_counter()
    receipt=dict(passed=False,source_sha256=fingerprints(),backends={},checks={})
    path=ROOT/"validation/correctness_research.json"
    if '--resume' in sys.argv:
        previous=read_json(path,{})
        if previous.get('source_sha256')==receipt['source_sha256']:
            receipt['checks']={name:value for name,value in previous.get('checks',{}).items() if value.get('passed')}
            receipt['resumed']=True
    checks=(("quadratures",quadratures),("exact_force",exact_force),("bgk_trt_limit",bgk_trt_limit),
                       ("shear_decay",shear_decay),("channels",channels),("thermal",thermal_tests),
                       ("compatibility",compatibility),("gpu_agreement",gpu_agreement),
                       ("rectangular_obstacles",rectangular_obstacles),
                       ("restart",restart),("natural_convection",natural_convection))+CHECKS
    for name,check in checks:
        if name in receipt['checks']:
            print(name,'REUSED current-source passed check',flush=True)
            continue
        begin=time.perf_counter()
        try:
            result=check()
            receipt["checks"][name]=dict(passed=True,result=result,seconds=time.perf_counter()-begin)
            print(name,"PASS",json.dumps(result),flush=True)
        except Exception as error:
            import traceback
            traceback.print_exc()
            receipt["checks"][name]=dict(passed=False,error=str(error),seconds=time.perf_counter()-begin)
        atomic_json(path,receipt)
    receipt["passed"]=all(c["passed"] for c in receipt["checks"].values())
    receipt["backends"]={k:receipt["passed"] for k in ("cpu","array","fused")}
    receipt["seconds"]=sum(c['seconds'] for c in receipt['checks'].values())
    receipt['current_invocation_seconds']=time.perf_counter()-started
    atomic_json(path,receipt)
    print("ACCEPTANCE",receipt["passed"],receipt["seconds"],flush=True)
    if not receipt["passed"]: raise SystemExit(1)


if __name__=="__main__": main()
