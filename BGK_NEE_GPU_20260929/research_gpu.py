"""Coalesced CUDA BGK/TRT/MRT, streaming, macros and every-step diagnostics."""
import math
import numpy as np
from research_solver import ResearchArraySolver
from research_options import BUOYANT
from advanced_physics import zou_he
from research_cuda import source, STAT_COUNT


class ResearchFusedSolver(ResearchArraySolver):
    backend = 'cuda-fused'

    def __init__(self,config):
        self._optimized_ready = False
        super().__init__(config)
        cp=self.xp
        names=['collide_fluid','pull_fluid','macros_stats','reduce_stats']
        if self.scalar:names+=['collide_scalar','pull_scalar']
        forcing=bool(self.transport['force_x'] or self.p['force_y'] or self.p['force_z'] or
                     self.transport['buoyancy'] or self.p['scenario']=='oscillatory_channel')
        self._module=cp.RawModule(code=source(self.lat,self.scalar,self.p['collision'],forcing),
                                 options=('--std=c++11','--fmad=false'),name_expressions=tuple(names))
        self._kernels={name:self._module.get_function(name) for name in names}
        self._rates=cp.asarray(self.rates)
        self._zero=cp.zeros(self.shape,dtype=cp.float64)
        self._solid=self.solid_mask.astype(cp.uint8)
        self._n=int(np.prod(self.shape))
        self._blocks=((self._n+127)//128,)
        self._partial=cp.empty((STAT_COUNT,self._blocks[0]),dtype=cp.float64)
        self._statistics=cp.empty(STAT_COUNT,dtype=cp.float64)
        self._statistics_host=None
        self._statistics_valid=False
        self._buffers={}
        self.f=self._soa(self.f)
        if self.g is not None:self.g=self._soa(self.g)
        self._optimized_ready=True

    def build_links(self,lat):
        # Streaming computes geometry directly; no 4 volume masks per velocity.
        return []

    def _soa(self,values):
        cp=self.xp
        return cp.moveaxis(cp.ascontiguousarray(cp.moveaxis(values,-1,0)),0,-1)

    def _ensure_layout(self):
        # Reference tests / callers may replace populations directly.
        for key in ('f','g'):
            value=getattr(self,key)
            if value is not None and (value.strides[-1]!=self._n*8 or value.strides[-2]!=8):
                setattr(self,key,self._soa(value))

    def _advance(self,f,rho,lat,scalar=False):
        cp,i=self.xp,np.int32
        suffix='scalar' if scalar else 'fluid'
        if suffix not in self._buffers:
            self._buffers[suffix]=cp.moveaxis(cp.empty((lat.q,)+self.shape,dtype=cp.float64),0,-1)
        post=self._buffers[suffix]
        u=self.u+([self.uz] if self.dim==2 else [])
        a=self.a+([self._zero] if self.dim==2 else [])
        rate=1/self.transport['thermal_tau'] if scalar else self.omega
        self._kernels['collide_'+suffix](self._blocks,(128,),
            (f,post,rho,*u,*a,self._rates,i(self._n),np.float64(rate),np.float64(self.omega_minus)))
        periodic=self.periodic+([True] if self.dim==2 else [])
        self._kernels['pull_'+suffix](self._blocks,(128,),
            (post,f,rho,self._solid,i(self.nx),i(self.ny),i(self.nz),*(i(v) for v in periodic),
             np.float64(0 if scalar else self.lid_velocity()),
             i(self.heat_axis+1 if scalar and self.p['scenario'] in BUOYANT else 0)))
        # Collision has finished before streaming starts on the same stream;
        # streaming reads only post, so the consumed f allocation can be reused.
        if not scalar:self._last_post=post
        return f

    def advance_populations(self):
        self._ensure_layout()
        self.f=self._advance(self.f,self.rho,self.lat)
        if self.open_boundary:zou_he(self)
        if self.g is not None:self.g=self._advance(self.g,self.T,self.scalar,True)

    def macros(self):
        if not self._optimized_ready:return super().macros()
        self.invalidate_host()
        self._ensure_layout()
        fx=self.transport['force_x']
        if self.p['scenario']=='oscillatory_channel':
            fx+=self.p['force_amplitude']*math.cos(2*math.pi*self.iteration/self.p['drive_period']+self.p['drive_phase'])
        buoyancy=self.transport['buoyancy'] if self.p['scenario'] in BUOYANT else 0.0
        a=self.a+([self._zero] if self.dim==2 else [])
        self._kernels['macros_stats'](self._blocks,(128,),
            (self.f,self.g if self.g is not None else self._zero,self.rho,self.ux,self.uy,self.uz,
             *a,self.T if self.T is not None else self._zero,self._solid,self._partial,np.int32(self._n),
             np.float64(fx),np.float64(self.p['force_y']),np.float64(self.p['force_z']),np.float64(buoyancy)))
        self._kernels['reduce_stats']((1,),(256,),(self._partial,self._statistics,np.int32(self._blocks[0])))
        self._statistics_host=None
        self._statistics_valid=True

    def _read_statistics(self):
        if self._statistics_host is None:
            # One 120-byte transfer synchronizes all per-step safeguards.
            self._statistics_host=self.xp.asnumpy(self._statistics)
        return self._statistics_host

    def step_diagnostics(self,old_u,old_T):
        s=self._read_statistics()
        self._last_residual=float(np.sqrt(s[0]/max(s[1],1e-30)))
        self.decision_residual=self._last_residual
        if self.T is not None:self.thermal_residual=float(np.sqrt(s[2]/max(s[3],1e-30)))
        return dict(finite=s[8]==0,min_density=float(s[5]),max_speed2=float(s[7]),mass=float(s[4]))

    def report_statistics(self):
        if not self._optimized_ready or not self._statistics_valid:return super().report_statistics()
        s=self._read_statistics()
        return dict(mass=float(s[4]),min_density=float(s[5]),max_density=float(s[6]),max_speed2=float(s[7]),
                    mean_speed2=float(s[1]/s[11]),mean_ux=float(s[9]/s[11]),mean_uy=float(s[10]/s[11]),
                    temperature_min=float(s[12]),temperature_max=float(s[13]),temperature_mean=float(s[14]/s[11]))
