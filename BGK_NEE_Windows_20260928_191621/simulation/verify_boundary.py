"""Small reproducible checks for NEE moments, corners, and unchanged halfway path."""
from pathlib import Path
import sys,json,hashlib,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'calculation'))
import run_baseline as runner
from config import default_boundaries_for_case,domain_axis
from boundary_conditions import E,MacroFields,non_equilibrium_cavity
from lbm_solver import LBMSolver

def config(*args):
    return runner.configure(runner.build_parser().parse_args(list(args)))

checks={}
rng=np.random.default_rng(7)
rho=1+rng.uniform(-.01,.01,(9,11))
u=rng.uniform(-.02,.02,(9,11));v=rng.uniform(-.02,.02,(9,11))
f=LBMSolver.equilibrium(rho,u,v)
f+=rng.uniform(-1e-3,1e-3,(9,11,1))*np.array([0,1,-1,1,-1,0,0,0,0])
original=f.copy()
b=default_boundaries_for_case('lid_driven_cavity',.04,'non_equilibrium_extrapolation')
non_equilibrium_cavity(f,b,MacroFields(rho,u,v),LBMSolver.equilibrium)
r=f.sum(-1); ux=(f*E[:,0]).sum(-1)/r;uy=(f*E[:,1]).sum(-1)/r
assert np.array_equal(f[1:-1,1:-1],original[1:-1,1:-1])
assert np.max(np.abs(ux[-1,1:-1]-.04))<1e-14
assert np.max(np.abs(uy[[0,-1],:]))<1e-14
assert np.max(np.abs(ux[:-1,[0,-1]]))<1e-14
assert np.max(np.abs(ux[-1,:]-.04))<1e-14
assert np.max(np.abs(uy[:,[0,-1]]))<1e-14
assert np.max(np.abs(ux[0,:]))<1e-14
for dst,donor in [((0,0),(1,1)),((0,-1),(1,-2)),((-1,0),(-2,1)),((-1,-1),(-2,-2))]:
    assert abs(r[dst]-rho[donor])<1e-14
    neq_d=original[donor]-LBMSolver.equilibrium(rho[donor],u[donor],v[donor])
    neq_b=f[dst]-LBMSolver.equilibrium(r[dst],ux[dst],uy[dst])
    assert np.max(np.abs(neq_d-neq_b))<1e-14
checks['face_velocity_corner_velocity_and_non_equilibrium_moments']='pass'
c=config('--defaults','--grid','16','--re','100')
assert domain_axis(c,'x')==(0.,15.)
s=LBMSolver(c)
for i in range(20):
    s.step()
    assert np.max(np.abs(s.ux[-1,1:-1]-.04*(i+1)/500))<1e-14
    assert np.max(np.abs(s.ux[-1,:]-.04*(i+1)/500))<1e-14
    numerator=np.linalg.norm(np.stack([s.ux[1:-1,1:-1]-s._step_ux[1:-1,1:-1],s.uy[1:-1,1:-1]-s._step_uy[1:-1,1:-1]]))
    denominator=np.linalg.norm(np.stack([s.ux[1:-1,1:-1],s.uy[1:-1,1:-1]]))
    expected=numerator/denominator if denominator else 0.
    assert abs(s._velocity_residual()-expected)<1e-14
checks['lid_ramp_and_wall_length']='pass'
checks['NEE_N16_Re100_after20_mass_drift']=abs(s.rho.sum()-s.mass0)/s.mass0
# Exact steady Couette profile is an independent boundary check with no corners.
c=config('--defaults','--grid','16','--re','10')
c.case_type='couette_flow';c.flow.initial_velocity='couette';c.convergence.ramp_steps=0
c.convergence.steady=False
c.boundaries=default_boundaries_for_case('couette_flow',.04)
for side in ('bottom','top'): c.boundaries[side].type='non_equilibrium_extrapolation'
s=LBMSolver(c)
for _ in range(3000): s.step()
exact=np.arange(16)[:,None]/15*.04
err=float(np.linalg.norm(s.ux-exact)/np.linalg.norm(np.broadcast_to(exact,s.ux.shape)))
assert err<1e-4,err
assert np.max(np.abs(s.uy))<1e-12
assert np.isfinite(s.f).all() and np.min(s.rho)>0
checks['couette_relative_L2']=err
checks['couette_mass_drift']=abs(s.rho.sum()-s.mass0)/s.mass0
# The halfway branch must reproduce the previous snapshot bit for bit.
c=config('--defaults','--boundary','halfway','--grid','16','--re','100')
s=LBMSolver(c)
for _ in range(100):s.step()
new_hash=hashlib.sha256(s.f.tobytes()).hexdigest()
old=ROOT.parent/'reference_halfway/calculation'
code="import sys,hashlib; sys.path.insert(0,sys.argv[1]); import run_baseline as r; from lbm_solver import LBMSolver; c=r.configure(r.build_parser().parse_args(['--defaults','--grid','16','--re','100'])); s=LBMSolver(c); [s.step() for _ in range(100)]; print(hashlib.sha256(s.f.tobytes()).hexdigest())"
old_hash=subprocess.check_output([sys.executable,'-c',code,str(old)],text=True).strip()
assert old_hash==new_hash
checks['halfway_100step_bitwise_regression']=new_hash
# With unchanged rest populations but a deliberately offset reference mass,
# strict gating rejects the state while diagnostic mode accepts zero residual.
for strict in (True,False):
    c=config('--defaults','--grid','16','--re','100')
    c.flow.U_ref=0.;c.flow.nu_lattice=.03;c.boundaries['top'].ux=0.
    c.convergence.min_iter=1;c.convergence.ramp_steps=0
    c.convergence.mass_criterion=strict
    s=LBMSolver(c);s.mass0*=1.01;s.step()
    assert (s.stop_reason=='converged') == (not strict)
    assert abs(s.rho.sum()-s.mass0)/s.mass0>1e-9
checks['strict_vs_diagnostic_mass_gating']='pass'
# Independent pull update + Appendix D face ordering, including top corners.
c=config('--defaults','--grid','16','--re','100')
s=LBMSolver(c);fref=s.f.copy();rr=s.rho.copy();uu=s.ux.copy();vv=s.uy.copy()
for step in range(12):
    eq=LBMSolver.equilibrium(rr,uu,vv);out=np.zeros_like(fref)
    for k,(ex,ey) in enumerate(E):
        src=(slice(1-ey,15-ey),slice(1-ex,15-ex),k)
        out[1:-1,1:-1,k]=fref[src]+(eq[src]-fref[src])/s.transport['tau']
    rr[1:-1,1:-1]=out[1:-1,1:-1].sum(-1)
    uu[1:-1,1:-1]=(out[1:-1,1:-1]*E[:,0]).sum(-1)/rr[1:-1,1:-1]
    vv[1:-1,1:-1]=(out[1:-1,1:-1]*E[:,1]).sum(-1)/rr[1:-1,1:-1]
    for dst,donor,wall_u in [((slice(1,-1),0),(slice(1,-1),1),0.),((slice(1,-1),-1),(slice(1,-1),-2),0.),((0,slice(None)),(1,slice(None)),0.),((-1,slice(None)),(-2,slice(None)),.04*(step+1)/500)]:
        rr[dst]=rr[donor];uu[dst]=wall_u;vv[dst]=0.
        out[dst]=LBMSolver.equilibrium(rr[dst],uu[dst],vv[dst])+out[donor]-LBMSolver.equilibrium(rr[donor],uu[donor],vv[donor])
    fref=out;s.step()
    assert np.max(np.abs(s.f-fref))<2e-14
checks['book_pull_and_ordered_corner_equivalence_12steps']='pass'
(ROOT/'boundary_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
