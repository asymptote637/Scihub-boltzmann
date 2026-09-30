"""Isothermal D2/D3 BGK/TRT/MRT with Guo forcing and passive/Boussinesq heat.

Arrays use (y,x,q) or (z,y,x,q). Walls are halfway fluid links; x is fastest.
NumPy and CuPy deliberately share the equations. The fused subclass replaces
collision and pull streaming with CUDA kernels, without changing the model.
"""
from __future__ import annotations

from dataclasses import asdict
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from reference import config_to_dict
from lbm_solver import StepReport
from cavity_models import M, MINV, relaxation_rates
from lattices import lattice
from research_options import PERIODIC, CHANNEL, OPEN, OBSTACLES, BUOYANT, THERMAL, MOVING, transport
from advanced_physics import geometry_mask, obstacle_centers, sample_probes, zou_he


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def physical_parameters(params):
    runtime = {"backend", "max_iter", "min_iter", "tol", "report_interval", "snapshot_interval",
               "checkpoint_interval", "resume_from", "probes"}
    return {k:v for k,v in params.items() if k not in runtime}


HOST_FIELDS = ('rho','ux','uy','uz','f','solid_mask','T','g')


class DeviceHost(SimpleNamespace):
    """Compatibility view: explicit host-field reads transfer only that field."""
    def __getattr__(self,key):
        if key not in HOST_FIELDS:raise AttributeError(key)
        solver=self._solver
        value=getattr(solver,key)
        return solver.xp.asnumpy(value) if value is not None else None


class ResearchSolver:
    backend = "cpu"

    def __init__(self, config, xp=np):
        self.config = copy.deepcopy(config)
        from research_validation import fingerprints
        self.source_sha256 = fingerprints()
        self.p = self.config.research
        self.xp = xp
        self.lat = lattice(self.p["lattice"])
        self.dim, self.q = self.lat.dim, self.lat.q
        self.transport = transport(self.p)
        self.transport.update(cs=math.sqrt(self.lat.cs2), omega=1/self.transport["tau"])
        self.nx, self.ny, self.nz = (self.transport[k] for k in ("nx","ny","nz"))
        self.shape = (self.ny,self.nx) if self.dim == 2 else (self.nz,self.ny,self.nx)
        self.axes = tuple(range(self.dim))
        self.e, self.w, self.opp = xp.asarray(self.lat.e), xp.asarray(self.lat.w), xp.asarray(self.lat.opp)
        self.omega = 1/self.transport["tau"]
        self.omega_minus = 1/(.5+self.p["trt_magic"]/(self.transport["tau"]-.5))
        self.rates = relaxation_rates(config, self.omega)
        self.periodic = ([True]*self.dim if self.p["scenario"] in PERIODIC else
                         ([True,False]+([True] if self.dim==3 else [])
                          if self.p["scenario"] in CHANNEL else [False]*self.dim))
        self.open_boundary = self.p['scenario'] in OPEN
        self.boundary_exchange = 0.0
        self._last_post = None
        self.iteration = 0
        self.coords = list(reversed(xp.indices(self.shape, dtype=xp.float64)))
        sizes = [self.nx,self.ny,self.nz]
        self.fractions = [(a+.5)/sizes[i] for i,a in enumerate(self.coords)]
        self.solid_mask = geometry_mask(xp,self.coords,sizes[:self.dim],self.p,require_resolved=True)
        if self.p['scenario'] in OBSTACLES:
            if not bool(self.solid_mask.any()) or bool(self.solid_mask.all()):
                raise ValueError("障碍物必须占据部分格点；请增大网格或调整尺寸")
        self.fluid = ~self.solid_mask
        self.links = self.build_links(self.lat)
        self.rho = xp.full(self.shape,self.p["rho0"],dtype=xp.float64)
        self.u = [xp.zeros(self.shape,dtype=xp.float64) for _ in range(self.dim)]
        s, U = self.p["scenario"], self.p["lid_speed"]
        phase = [2*math.pi*a for a in self.fractions]
        if s == "shear_wave":
            self.u[0] = U*xp.sin(phase[1])
        elif s == "taylor_green":
            factor = xp.cos(phase[2]) if self.dim==3 else 1
            self.u[0] = U*xp.sin(phase[0])*xp.cos(phase[1])*factor
            self.u[1] = -U*xp.cos(phase[0])*xp.sin(phase[1])*factor
            pressure = U*U*(xp.cos(2*phase[0])+xp.cos(2*phase[1]))
            pressure *= ((xp.cos(2*phase[2])+2)/16 if self.dim==3 else .25)
            self.rho *= 1+pressure/self.lat.cs2
        elif s == "thermal_wave":
            self.u[0].fill(U)
        self.T = self.g = self.scalar = None
        self.heat_axis = 1 if s=='rayleigh_benard' else 0
        if s in THERMAL:
            self.scalar = lattice("D2Q5" if self.dim==2 else "D3Q7")
            self.T = (.5+.1*xp.sin(phase[0]) if s=="thermal_wave" else 1-self.fractions[self.heat_axis])
            if s=='rayleigh_benard':
                self.T += self.p['thermal_perturbation']*xp.cos(phase[0])*xp.sin(math.pi*self.fractions[1])
            self.scalar_links = self.build_links(self.scalar)
            self.g = self.scalar_equilibrium()
        self.a = self.acceleration()
        self.f = self.equilibrium() - .5*self.source()
        self.iteration, self.mass0 = 0, float(self.rho[self.fluid].sum())
        self.history, self._failure, self.stop_reason = [], "", "running"
        self._stable, self._last_report, self._last_residual = 0, None, 0.0
        self.thermal_residual = 0.0
        self.decision_residual = 0.0
        self._started = time.perf_counter()
        self._final_result = None
        self.host = self if xp is np else DeviceHost(_solver=self,cancel=self.cancel)
        self._bind_velocity()
        if self.p["resume_from"]:
            self.restore_checkpoint(self.p["resume_from"])
        self.sync_host(full=False)

    def _bind_velocity(self):
        self.ux,self.uy = self.u[:2]
        self.uz = self.u[2] if self.dim==3 else self.xp.zeros(self.shape)

    def build_links(self, lat):
        xp = self.xp
        links = []
        sizes = [self.nx,self.ny,self.nz]
        for velocity in lat.e:
            outside = xp.zeros(self.shape, dtype=bool)
            for axis,e in enumerate(velocity):
                if not self.periodic[axis]:
                    source = self.coords[axis]-int(e)
                    outside |= (source<0)|(source>=sizes[axis])
            solid = xp.roll(self.solid_mask, tuple(int(v) for v in reversed(velocity)), self.axes)
            top = (self.coords[1]-int(velocity[1]) >= self.ny) & outside
            left = (self.coords[0]-int(velocity[0]) < 0) & outside
            right = (self.coords[0]-int(velocity[0]) >= self.nx) & outside
            links.append((outside|solid, top, left, right))
        return links

    def acceleration(self):
        xp = self.xp
        a = [xp.full(self.shape, self.transport["force_x"] if i==0 else self.p["force_"+"xyz"[i]])
             for i in range(self.dim)]
        if self.p["scenario"] in BUOYANT:
            a[1] += self.transport["buoyancy"]*(self.T-.5)
        if self.p['scenario']=='oscillatory_channel':
            a[0] += self.p['force_amplitude']*math.cos(2*math.pi*self.iteration/self.p['drive_period']+self.p['drive_phase'])
        return a

    def equilibrium(self):
        xp,c = self.xp,self.lat.cs2
        eu = sum(self.u[a][...,None]*self.e[:,a] for a in range(self.dim))
        u2 = sum(u*u for u in self.u)[...,None]
        return self.w*self.rho[...,None]*(1+eu/c+eu*eu/(2*c*c)-u2/(2*c))

    def source(self):
        c = self.lat.cs2
        eu = sum(self.u[a][...,None]*self.e[:,a] for a in range(self.dim))
        ea = sum(self.a[a][...,None]*self.e[:,a] for a in range(self.dim))
        ua = sum(self.u[a]*self.a[a] for a in range(self.dim))[...,None]
        return self.w*self.rho[...,None]*((ea-ua)/c+eu*ea/(c*c))

    def collide(self):
        xp = self.xp
        delta, source = self.f-self.equilibrium(), self.source()
        model = self.p["collision"]
        if model == "BGK":
            return self.f-self.omega*delta+(1-.5*self.omega)*source
        if model == "TRT":
            dp, dm = .5*(delta+delta[...,self.opp]), .5*(delta-delta[...,self.opp])
            sp, sm = .5*(source+source[...,self.opp]), .5*(source-source[...,self.opp])
            return self.f-self.omega*dp-self.omega_minus*dm+(1-.5*self.omega)*sp+(1-.5*self.omega_minus)*sm
        # Avoid requiring cuBLAS for the array reference; this is a 9x9 transform.
        def transform(a, matrix):
            return xp.stack([sum(float(matrix[i,j])*a[...,j] for j in range(9) if matrix[i,j])
                             for i in range(9)], axis=-1)
        moments = transform(delta,M)*xp.asarray(self.rates)
        force = transform(source,M)*(1-.5*xp.asarray(self.rates))
        return self.f-transform(moments-force,MINV)

    def lid_velocity(self):
        if self.p["scenario"] not in MOVING: return 0.0
        ramp = self.p["ramp_steps"]
        value = self.p["lid_speed"]*min(1,(self.iteration+1)/ramp) if ramp else self.p["lid_speed"]
        if self.p['scenario']=='oscillating_couette':
            value *= math.cos(2*math.pi*(self.iteration+.5)/self.p['drive_period']+self.p['drive_phase'])
        return value

    def pull(self, post, lat, links, thermal=False):
        xp = self.xp
        result = xp.empty_like(post)
        lid = self.lid_velocity()
        for k,velocity in enumerate(lat.e):
            bounced,top,left,right = links[k]
            shifted = xp.roll(post[...,k], tuple(int(v) for v in reversed(velocity)), self.axes)
            reflected = post[...,int(lat.opp[k])]
            if thermal and self.p["scenario"] in BUOYANT:
                axis = self.heat_axis
                source = self.coords[axis]-int(velocity[axis])
                n = (self.nx,self.ny,self.nz)[axis]
                reflected = xp.where(source<0,-reflected+2*lat.w[k], xp.where(source>=n,-reflected,reflected))
            elif not thermal and lid:
                reflected = reflected+top*(2*lat.w[k]*self.rho*int(velocity[0])*lid/lat.cs2)
            result[...,k] = xp.where(self.solid_mask,post[...,k],xp.where(bounced,reflected,shifted))
        return result

    def scalar_equilibrium(self):
        xp = self.xp
        eu = sum(self.u[a][...,None]*xp.asarray(self.scalar.e[:,a]) for a in range(self.dim))
        return xp.asarray(self.scalar.w)*self.T[...,None]*(1+eu/self.scalar.cs2)

    def advance_populations(self):
        self._last_post = self.collide()
        self.f = self.pull(self._last_post,self.lat,self.links)
        if self.open_boundary: zou_he(self)
        if self.g is not None:
            rate = 1/self.transport["thermal_tau"]
            post = self.g-rate*(self.g-self.scalar_equilibrium())
            self.g = self.pull(post,self.scalar,self.scalar_links,thermal=True)
            self.T = self.g.sum(axis=-1)

    def macros(self):
        self.invalidate_host()
        xp = self.xp
        self.rho = self.f.sum(axis=-1)
        self.a = self.acceleration()
        denominator = xp.where(self.rho>0,self.rho,1)
        self.u = [xp.where(self.solid_mask,0,(self.f*self.e[:,i]).sum(axis=-1)/denominator+.5*self.a[i])
                  for i in range(self.dim)]
        self._bind_velocity()

    def step_diagnostics(self, old_u, old_T):
        """One-step residuals and safeguards; optimized engines may fuse reductions."""
        xp = self.xp
        fluid = self.fluid
        numerator = sum(((u-v)[fluid]**2).sum() for u,v in zip(self.u,old_u))
        denominator = sum((u[fluid]**2).sum() for u in self.u)
        self._last_residual = float(xp.sqrt(numerator/xp.maximum(denominator,1e-30)))
        self.decision_residual = self._last_residual
        if self.T is not None:
            self.thermal_residual = float(xp.sqrt(((self.T-old_T)[fluid]**2).sum()/xp.maximum((self.T[fluid]**2).sum(),1e-30)))
        speed2 = sum(u*u for u in self.u)
        return dict(finite=bool(xp.isfinite(self.f).all()) and (self.g is None or bool(xp.isfinite(self.g).all())),
                    min_density=float(self.rho[fluid].min()),max_speed2=float(speed2[fluid].max()),
                    mass=float(self.rho[fluid].sum()))

    def step(self):
        if self.stop_reason != "running": return
        old_u, old_T = self.u, self.T
        self.advance_populations()
        self.iteration += 1
        self.macros()
        diagnostics = self.step_diagnostics(old_u,old_T)
        if not diagnostics['finite'] or diagnostics['min_density']<=0:
            self._failure = "non-finite population or non-positive density"
        elif diagnostics['max_speed2'] > self.lat.cs2*self.config.convergence.max_mach**2:
            self._failure = "Mach limit exceeded"
        if self._failure:
            self.stop_reason = "diverged"
        elif self.p["run_mode"] == "steady" and self.iteration>=max(self.p["min_iter"],self.p["ramp_steps"]):
            mass_ok = abs(diagnostics['mass']-self.mass0-self.boundary_exchange)/self.mass0 <= self.config.convergence.mass_tolerance
            self._stable = self._stable+1 if (max(self._last_residual,self.thermal_residual)<self.p["tol"] and mass_ok) else 0
            if self._stable >= self.config.convergence.consecutive_reports:
                self.stop_reason = "converged"
        if self.iteration>=self.p["max_iter"] and self.stop_reason=="running":
            self.stop_reason = "max_iter" if self.p["run_mode"]=="steady" else "completed_steps"
        folder = Path(self.config.output.output_dir)
        interval = self.p["snapshot_interval"]
        if interval and self.iteration%interval==0:
            atomic_npz(folder/"snapshots"/f"fields_{self.iteration:09d}.npz",**self.field_arrays())
        interval = self.p["checkpoint_interval"]
        if interval and self.iteration%interval==0 and not self._failure:
            self.save_checkpoint(folder/"checkpoint.npz")
        self._last_report = None

    def invalidate_host(self):
        if self.xp is not np and hasattr(self,'host'):
            for key in HOST_FIELDS:self.host.__dict__.pop(key,None)

    def sync_host(self, full=True):
        if self.xp is np: return
        if full:
            for k in HOST_FIELDS:
                value = getattr(self,k)
                setattr(self.host,k,self.xp.asnumpy(value) if value is not None else None)
        for k in ("config","nx","ny","nz","iteration","dim"):
            setattr(self.host,k,getattr(self,k))

    def field_arrays(self, preview=False):
        xp = self.xp
        limit = 256 if self.dim==2 else 40
        samples = [np.unique(np.r_[np.arange(0,n,max(1,math.ceil(n/limit))),n-1]) if preview else np.arange(n)
                   for n in self.shape]
        selected = np.ix_(*samples)
        arrays = {}
        for key in ("rho","ux","uy","uz","T","solid_mask"):
            a = getattr(self,key)
            if a is not None:
                a = a[selected]
                arrays[key] = xp.asnumpy(a) if xp is not np else a
        arrays.update({axis:(sample+.5)/size for axis,sample,size in zip("xyz",reversed(samples),reversed(self.shape))})
        if self.open_boundary: arrays['x'] = samples[-1]/(self.nx-1)
        arrays["iteration"] = np.int64(self.iteration)
        return arrays

    def report(self):
        if self._last_report is not None: return self._last_report
        # Reduce on the owning device. Full host arrays are only needed for final
        # exports / explicit sync_host(), never for an ordinary progress report.
        xp = self.xp
        h,fluid = self,self.fluid
        velocities = (h.ux,h.uy,h.uz)[:self.dim]
        stats = self.report_statistics()
        max_velocity = math.sqrt(stats['max_speed2'])
        raw_mass_change = stats['mass']-self.mass0
        mass_drift = abs(raw_mass_change-self.boundary_exchange)/self.mass0
        row = dict(iteration=self.iteration,residual=self._last_residual,mass_drift=mass_drift,
                   max_mach=max_velocity/math.sqrt(self.lat.cs2),max_velocity=max_velocity,
                   min_density=stats['min_density'],max_density=stats['max_density'],
                   kinetic_energy=.5*stats['mean_speed2'],
                   mean_ux=stats['mean_ux'],mean_uy=stats['mean_uy'],
                   thermal_residual=self.thermal_residual,stop_reason=self.stop_reason)
        row.update(mass_change=raw_mass_change/self.mass0,boundary_mass_exchange=self.boundary_exchange,
                   probes=sample_probes(self))
        if self.open_boundary:
            inlet = float((h.rho[:,0]*h.ux[:,0]).sum())
            outlet = float((h.rho[:,-1]*h.ux[:,-1]).sum())
            row.update(inlet_flux=inlet,outlet_flux=outlet,flux_imbalance=inlet-outlet,
                       outlet_backflow_fraction=float(xp.mean(h.ux[:,-1]<0)))
        if self.p['scenario'] in OBSTACLES:
            row.update(self.obstacle_force())
        if h.T is not None:
            row.update(temperature_min=stats['temperature_min'],temperature_max=stats['temperature_max'],
                       temperature_mean=stats['temperature_mean'])
            if self.p["scenario"] in BUOYANT:
                axis = self.dim-1-self.heat_axis
                n = self.shape[axis]
                row["nusselt_hot"] = float(xp.mean(2*n*(1-xp.take(h.T,0,axis=axis))))
                row["nusselt_cold"] = float(xp.mean(2*n*xp.take(h.T,-1,axis=axis)))
        if self.host is not self: self.host.iteration = self.iteration
        self.history.append(row)
        self._last_report = StepReport(self.iteration,self._last_residual,float("nan"),float("nan"),mass_drift,
             max_velocity,self.stop_reason=="converged",self.stop_reason=="diverged",self._failure or self.stop_reason,
             row["min_density"],row["max_density"],row["max_mach"],mass_drift,self.stop_reason)
        return self._last_report

    def report_statistics(self):
        xp,fluid = self.xp,self.fluid
        speed2 = sum(u*u for u in self.u)
        values = dict(mass=self.rho[fluid].sum(),min_density=self.rho[fluid].min(),max_density=self.rho[fluid].max(),
                      max_speed2=speed2[fluid].max(),mean_speed2=speed2[fluid].mean(),
                      mean_ux=self.ux[fluid].mean(),mean_uy=self.uy[fluid].mean())
        if self.T is not None:
            values.update(temperature_min=self.T[fluid].min(),temperature_max=self.T[fluid].max(),
                          temperature_mean=self.T[fluid].mean())
        if xp is np: return {k:float(v) for k,v in values.items()}
        data = xp.asnumpy(xp.stack(list(values.values())))
        return dict(zip(values,map(float,data)))

    def obstacle_force(self):
        xp = self.xp
        force = [0.0]*self.dim
        if self._last_post is not None:
            for k,e in enumerate(self.lat.e):
                mask = xp.roll(self.solid_mask,tuple(int(v) for v in reversed(e)),self.axes)&self.fluid
                amount = float(self._last_post[...,int(self.lat.opp[k])][mask].sum())
                for a in range(self.dim): force[a] -= 2*int(e[a])*amount
        r = self.p['obstacle_size']*min(self.nx-int(self.open_boundary),self.ny,self.nz if self.dim==3 else self.ny)
        area = (2*r if self.dim==2 else (math.pi*r*r if self.p['obstacle']=='circle' else 4*r*r))
        area *= len(obstacle_centers(self.p,self.dim))
        denominator = .5*self.p['rho0']*self.p['lid_speed']**2*area
        return dict(force_x=force[0],force_y=force[1],force_z=force[2] if self.dim==3 else 0,
                    drag_coefficient=force[0]/denominator,lift_coefficient=force[1]/denominator,
                    force_available=self._last_post is not None,force_reference_area=area)

    def run(self):
        while self.iteration<self.p["max_iter"] and self.stop_reason=="running":
            self.step()
            if self.iteration==1 or self.iteration%self.p["report_interval"]==0 or self.stop_reason!="running":
                yield self.report()

    def cancel(self):
        if self.stop_reason=="running": self.stop_reason="cancelled"
        self._last_report = None

    def save_checkpoint(self,path):
        arrays = self.field_arrays()
        arrays["f"] = self.xp.asnumpy(self.f) if self.xp is not np else self.f
        if self.g is not None: arrays["g"] = self.xp.asnumpy(self.g) if self.xp is not np else self.g
        arrays["metadata"] = np.array(json.dumps(dict(version=1,params=self.p,mass0=self.mass0,stable=self._stable,
                    residual=self._last_residual,thermal_residual=self.thermal_residual,
                    boundary_exchange=self.boundary_exchange,source_sha256=self.source_sha256)))
        atomic_npz(path,**arrays)

    def restore_checkpoint(self,path):
        from research_validation import fingerprints
        with np.load(path,allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            if meta.get("version")!=1 or meta.get("source_sha256")!=fingerprints():
                raise ValueError("检查点版本或物理内核源码与当前程序不同")
            if physical_parameters(meta["params"])!=physical_parameters(self.p):
                raise ValueError("检查点物理参数不同；只允许修改后端、步数预算和报告/停止设置")
            for key,q in (("f",self.q),("g",self.scalar.q if self.scalar else 0)):
                if not q: continue
                data = archive[key]
                if data.shape!=self.shape+(q,) or data.dtype!=np.float64 or not np.isfinite(data).all():
                    raise ValueError("检查点分布函数形状、精度或数值无效")
                setattr(self,key,self.xp.asarray(data.copy()))
            self.iteration = int(archive["iteration"])
            if self.iteration>=self.p["max_iter"]:
                raise ValueError("续算总步数必须大于检查点步数")
            self.mass0 = float(meta["mass0"])
            self.boundary_exchange = float(meta.get('boundary_exchange',0))
            self._stable = 0  # New stop settings require fresh consecutive steps.
            if self.g is not None: self.T=self.g.sum(axis=-1)
            self.macros()
            self._last_residual,self.thermal_residual = meta["residual"],meta["thermal_residual"]

    def finalize(self,save_outputs=True):
        if self._final_result is not None: return self._final_result
        if self.stop_reason=="running": self.cancel()
        report = self.report()
        self.sync_host()
        if save_outputs:
            from dashboard.store import atomic_json
            folder = Path(self.config.output.output_dir)
            folder.mkdir(parents=True,exist_ok=True)
            atomic_json(folder/"config.json",config_to_dict(self.config,self.transport))
            atomic_npz(folder/"results.npz",**self.field_arrays())
            atomic_json(folder/"run_summary.json",dict(stop_reason=self.stop_reason,iteration=self.iteration,
                        final_diagnostics=self.history[-1],error=self._failure,backend=self.backend,
                        convergence_policy="velocity+temperature+mass" if self.g is not None else "velocity+mass",
                        residual_domain="all fluid nodes",elapsed_seconds=time.perf_counter()-self._started))
            with (folder/"residual_history.csv").open("w",encoding="utf-8",newline="") as stream:
                writer = csv.DictWriter(stream,fieldnames=list(self.history[-1]))
                writer.writeheader()
                writer.writerows(self.history)
            if self.p['probes']:
                with (folder/'probes.csv').open('w',encoding='utf-8',newline='') as stream:
                    fields = ['iteration','probe','valid','x','y','z','ux','uy','uz','rho','T','pressure']
                    writer = csv.DictWriter(stream,fieldnames=fields)
                    writer.writeheader()
                    for row in self.history:
                        for index,probe in enumerate(row['probes'],1):
                            out = {k:probe[k] for k in ('valid','ux','uy','uz','rho','T','pressure')}
                            out.update(iteration=row['iteration'],probe=index,**dict(zip('xyz',probe['coordinates'])))
                            writer.writerow(out)
            if not self._failure:
                self.save_checkpoint(folder/"checkpoint.npz")
        self._final_result = SimpleNamespace(rho=self.host.rho,ux=self.host.ux,uy=self.host.uy,uz=self.host.uz,
                    T=self.host.T,history=self.history,config=self.config,transport=self.transport,stop_reason=self.stop_reason)
        return self._final_result


class ResearchArraySolver(ResearchSolver):
    backend = "cupy-array"

    def __init__(self,config):
        import gpu_environment
        import cupy as cp
        super().__init__(config,cp)
