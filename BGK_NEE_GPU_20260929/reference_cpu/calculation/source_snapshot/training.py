"""Reproducible isothermal training benchmarks, without opening a GUI."""
from __future__ import annotations
import argparse
import csv
import datetime
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from config import (SimulationConfig, GridConfig, FlowConfig, ConvergenceConfig,
                    OutputConfig, domain_axis)
from lbm_solver import LBMSolver


def run_case(config):
    solver=LBMSolver(config)
    for report in solver.run():
        if report.iteration % 5000 == 0:
            print(config.case_type, config.grid.NY, report.iteration,
                  f"residual={report.residual:.3e}", flush=True)
    result=solver.finalize()
    print(config.case_type, config.grid.NY, result.stop_reason,
          f"mass={report.mass_drift:.3e}", flush=True)
    return solver, result


def channel(root, case, heights):
    rows=[]
    fig, axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for index, n in enumerate(heights):
        # Diffusive scaling: fixed nu=.1, U proportional to 1/H, fixed Re.
        target_speed=.32/n
        acceleration=8*.1*target_speed/n**2 if case=="force_poiseuille" else 0.
        c=SimulationConfig(case_type=case,grid=GridConfig(NX=8,NY=n),
            flow=FlowConfig(U_ref=target_speed,nu_lattice=.1,body_force_x=acceleration),
            convergence=ConvergenceConfig(max_iter=int(6500*(n/16)**2),min_iter=500,
                report_interval=max(100,int(100*(n/16)**2)),ramp_steps=0,tol=1e-8),
            output=OutputConfig(output_dir=str(root/f'{case}_H{n}'),save_png=False))
        s,result=run_case(c)
        y=np.arange(n)+.5
        exact=(acceleration*y*(n-y)/(.2) if acceleration else target_speed*y/n)
        observed=s.ux.mean(axis=1)
        error=float(np.linalg.norm(observed-exact)/np.linalg.norm(exact))
        flow_error=abs(float(observed.sum()-exact.sum()))/float(exact.sum())
        row=dict(case=case,H=n,Re_reference=s.transport['Re'],
            Re_mean=2*target_speed*n/(3*.1) if acceleration else None,
            relative_L2=error,discrete_flow_error=flow_error,
            mass_drift=s.history[-1]['mass_drift'],transverse_max=float(np.abs(s.uy).max()),
            converged=result.converged,iteration=s.iteration,
            passed=bool(result.converged and error<.01 and s.history[-1]['mass_drift']<1e-9))
        rows.append(row)
        np.savetxt(Path(c.output.output_dir)/'analytic_profile.csv',
            np.column_stack([y/n,observed,exact]),delimiter=',',
            header='y_over_H,ux_lattice,analytic_ux_lattice',comments='')
        style = ['-', '--', '-.'][index % 3]
        axes[0].plot(observed/target_speed,y/n,style,label=f'H={n}')
        axes[1].plot((observed-exact)/target_speed,y/n,style,label=f'H={n}')
    yy=np.linspace(0,1,101)
    axes[0].plot(4*yy*(1-yy) if case=='force_poiseuille' else yy,yy,'k--',label='analytic')
    axes[0].set(xlabel='u / reference speed',ylabel='y / H',title=case)
    axes[1].set(xlabel='(u - analytic) / reference speed',ylabel='y / H',title='Profile error')
    axes[1].xaxis.set_major_locator(MaxNLocator(4))
    axes[1].ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.savefig(root/f'{case}_validation.png',dpi=160);plt.close(fig)
    return rows


def cavity(root, n=64, u=.04):
    c=SimulationConfig(case_type='lid_driven_cavity',grid=GridConfig(NX=n,NY=n),
        flow=FlowConfig(U_ref=u,Re=100),
        convergence=ConvergenceConfig(max_iter=int(50000*.04/u*(n/64)),min_iter=2000,
            report_interval=max(100,int(200*.04/u)),ramp_steps=500,tol=1e-7),
        output=OutputConfig(output_dir=str(root/f'cavity_Re100_N{n}_U{u:g}'),save_png=True))
    s,result=run_case(c)
    coordinate=(np.arange(n)+.5)/n
    vertical=np.array([np.interp(.5,coordinate,r) for r in s.ux])/u
    horizontal=np.array([np.interp(.5,coordinate,r) for r in s.uy.T])/u
    fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    errors={}
    for i,(axis,profile,endpoints) in enumerate((('u',vertical,[0,1]),('v',horizontal,[0,0]))):
        data=np.loadtxt(Path(__file__).parent/'references'/f'ghia_re100_{axis}.csv',delimiter=',',skiprows=1)
        coords=np.r_[0,coordinate,1]; values=np.r_[endpoints[0],profile,endpoints[1]]
        predicted=np.interp(data[:,0],coords,values)
        # Exclude imposed wall endpoints from error statistics.
        interior=(data[:,0]>0)&(data[:,0]<1)
        errors[f'{axis}_normalized_RMSE']=float(np.sqrt(np.mean((predicted[interior]-data[interior,1])**2)))
        axs[i].plot(coords,values,label='LBM');axs[i].plot(data[:,0],data[:,1],'o',label='Ghia Re=100 (transcription)')
        axs[i].set(xlabel='y / H' if axis=='u' else 'x / L',ylabel=f'{axis} / lid speed')
        axs[i].legend();axs[i].grid(alpha=.2)
    fig.savefig(root/f'cavity_Re100_N{n}_U{u:g}_validation.png',dpi=160);plt.close(fig)
    return dict(case='lid_driven_cavity',H=n,U=u,converged=result.converged,
        mass_drift=s.history[-1]['mass_drift'],iteration=s.iteration,**errors,
        passed=bool(result.converged and max(errors.values())<.02 and s.history[-1]['mass_drift']<1e-9),
        reference_note='Secondary table transcription, sources in references/sources.json; original paper not visually rechecked.')


def pressure_channel(root):
    from config import BoundaryConfig
    n, nx, nu, target_speed = 16, 64, .1, .02
    acceleration = 8 * nu * target_speed / n**2
    delta_rho = 3 * acceleration * (nx-1)
    c = SimulationConfig(case_type='poiseuille_channel',grid=GridConfig(NX=nx,NY=n),
        flow=FlowConfig(U_ref=target_speed,nu_lattice=nu),
        convergence=ConvergenceConfig(max_iter=15000,min_iter=1000,report_interval=200,ramp_steps=0,tol=1e-7),
        output=OutputConfig(output_dir=str(root/'pressure_channel'),save_png=False))
    c.boundaries['left']=BoundaryConfig('pressure_zou_he',rho=1+delta_rho/2)
    c.boundaries['right']=BoundaryConfig('pressure_zou_he',rho=1-delta_rho/2)
    s,result=run_case(c)
    y=np.arange(n)+.5
    # Low-Ma reference at rho0=1. Pressure-driven weakly compressible LBM
    # differs slightly from constant-density incompressible Poiseuille.
    exact=acceleration*y*(n-y)/(2*nu)
    error=float(np.linalg.norm(s.ux[:,nx//2]-exact)/np.linalg.norm(exact))
    transverse=float(np.abs(s.uy).max())
    return dict(case='pressure_channel',H=n,relative_L2=error,
        transverse_max=transverse,converged=result.converged,iteration=s.iteration,
        mass_drift=s.history[-1]['mass_drift'],mass_balance_error=s.history[-1]['mass_balance_error'],
        passed=bool(result.converged and error<.01 and transverse<1e-4
                    and s.history[-1]['mass_balance_error']<1e-8))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['channels','cavity','core'],default='core')
    p.add_argument('--heights',type=int,nargs='+',default=[16,32,64])
    p.add_argument('--cavity-n',type=int,default=64)
    p.add_argument('--cavity-u',type=float,default=.04)
    p.add_argument('--output-dir',default='training_runs')
    args=p.parse_args(argv)
    if any(h < 8 for h in args.heights) or args.cavity_n < 8 or not np.isfinite(args.cavity_u) or args.cavity_u <= 0:
        p.error('Grid sizes must be >= 8; cavity-u must be finite and positive.')
    root=Path(args.output_dir)/datetime.datetime.now().strftime('validation_%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True)
    rows=[]
    if args.suite in {'channels','core'}:
        for case in ['couette_flow','force_poiseuille']:rows.extend(channel(root,case,args.heights))
        rows.append(pressure_channel(root))
    if args.suite in {'cavity','core'}:rows.append(cavity(root,args.cavity_n,args.cavity_u))
    (root/'validation.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False,allow_nan=False))
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (root/'validation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2),flush=True);print('OUTPUT',root,flush=True)
    return 0 if all(r['passed'] for r in rows) else 1


if __name__=='__main__':raise SystemExit(main())
