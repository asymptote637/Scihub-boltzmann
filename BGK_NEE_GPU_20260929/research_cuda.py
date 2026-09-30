"""Specialize the existing double-precision equations for coalesced GPU storage.

Public populations retain (..., q) indexing, but their device backing is (q, ...).
Quadrature constants and opposite indices are compiled into the kernels; there
are no dynamically indexed per-thread population arrays (local-memory spills).
"""
from cavity_models import M, MINV

STAT_COUNT = 15


def number(value):
    return format(float(value), '.17e')


def dot(e, names):
    return '+'.join(f'({int(v)})*{n}' for v,n in zip(e,names) if v) or '0.0'


def collision(lat, model, suffix, scalar=False, forcing=True):
    out = [f'''extern "C" __global__ void collide_{suffix}(
      const double* f,double* post,const double* rho,
      const double* ux,const double* uy,const double* uz,
      const double* ax,const double* ay,const double* az,
      const double* rates,int n,double op,double om) {{
      int node=blockIdx.x*blockDim.x+threadIdx.x; if(node>=n)return;
      double u=ux[node],v=uy[node],z=uz[node],r=rho[node];
      double a=ax[node],b=ay[node],d=az[node];
      double u2=u*u+v*v+z*z,ua=u*a+v*b+z*d;
      const double ic={number(1/lat.cs2)},ic2={number(1/(2*lat.cs2**2))},ih={number(1/(2*lat.cs2))};''']
    def load(k):
        e,w=lat.e[k],number(lat.w[k])
        eq='1+eu*ic' if scalar else '1+eu*ic+eu*eu*ic2-u2*ih'
        src='0.0' if scalar or not forcing else f'{w}*r*((ea-ua)*ic+2*eu*ea*ic2)'
        return f'''double f{k}=f[(long long){k}*n+node],d{k},s{k};
          {{double eu={dot(e,('u','v','z'))},ea={dot(e,('a','b','d'))};
            d{k}=f{k}-{w}*r*({eq});s{k}={src};}}'''
    if model=='MRT':
        out += [load(k) for k in range(lat.q)]
        for j in range(9):
            dm='+'.join(f'({number(M[j,k])})*d{k}' for k in range(9) if M[j,k]) or '0.0'
            fm='+'.join(f'({number(M[j,k])})*s{k}' for k in range(9) if M[j,k]) or '0.0'
            out.append(f'double change{j}=rates[{j}]*({dm})-(1-.5*rates[{j}])*({fm});')
        for k in range(9):
            change='+'.join(f'({number(MINV[k,j])})*change{j}' for j in range(9) if MINV[k,j]) or '0.0'
            out.append(f'post[(long long){k}*n+node]=f{k}-({change});')
    elif model=='TRT':
        for k,j in enumerate(lat.opp):
            j=int(j)
            if j<k: continue
            # Opposite velocities have equal weights: compute the even/odd
            # equilibrium and Guo terms once, rather than twice per link pair.
            out.append(f'''{{double fk=f[(long long){k}*n+node],fj=f[(long long){j}*n+node];
              double eu={dot(lat.e[k],('u','v','z'))},wr={number(lat.w[k])}*r;
              double even=op*(.5*(fk+fj)-wr*(1+eu*eu*ic2-u2*ih));
              double odd=om*(.5*(fk-fj)-wr*eu*ic);''')
            if forcing:
                out.append(f'''double ea={dot(lat.e[k],('a','b','d'))};
                  even-=(1-.5*op)*wr*(-ua*ic+2*eu*ea*ic2);
                  odd-=(1-.5*om)*wr*ea*ic;''')
            out.append(f'post[(long long){k}*n+node]=fk-even-odd;')
            if j!=k:out.append(f'post[(long long){j}*n+node]=fj-even+odd;')
            out.append('}')
    else:
        for k in range(lat.q):
            out.extend(['{',load(k),f'post[(long long){k}*n+node]=f{k}-(op*d{k}-(1-.5*op)*s{k});','}'])
    return '\n'.join(out+['}'])


def pull(lat,suffix):
    out=[f'''extern "C" __global__ void pull_{suffix}(
      const double* post,double* f,const double* rho,const unsigned char* solid,
      int nx,int ny,int nz,int px,int py,int pz,double lid,int heat_wall) {{
      int node=blockIdx.x*blockDim.x+threadIdx.x,n=nx*ny*nz;if(node>=n)return;
      int x=node%nx,y=node/nx%ny,z=node/(nx*ny);''']
    for k,vel in enumerate(lat.e):
        ex,ey,ez=tuple(vel)+(0,)*(3-lat.dim)
        out.append(f'''{{
          if(solid[node]) f[(long long){k}*n+node]=post[(long long){k}*n+node];
          else {{
            int sx=x-({ex}),sy=y-({ey}),sz=z-({ez});
            bool outside=(!px&&(sx<0||sx>=nx))||(!py&&(sy<0||sy>=ny))||(!pz&&(sz<0||sz>=nz));
            int wx=sx<0?sx+nx:(sx>=nx?sx-nx:sx);
            int wy=sy<0?sy+ny:(sy>=ny?sy-ny:sy);
            int wz=sz<0?sz+nz:(sz>=nz?sz-nz:sz);
            int from=(wz*ny+wy)*nx+wx;
            if(outside||solid[from]) {{
              double reflected=post[(long long){int(lat.opp[k])}*n+node];
              if((heat_wall==1&&sx<0)||(heat_wall==2&&sy<0)) reflected=-reflected+2*{number(lat.w[k])};
              else if((heat_wall==1&&sx>=nx)||(heat_wall==2&&sy>=ny)) reflected=-reflected;
              else if(!heat_wall&&outside&&sy>=ny) reflected+=2*{number(lat.w[k])}*rho[node]*({ex})*lid/{number(lat.cs2)};
              f[(long long){k}*n+node]=reflected;
            }} else f[(long long){k}*n+node]=post[(long long){k}*n+from];
          }}
        }}''')
    return '\n'.join(out+['}'])


REDUCTION = r'''
// Statistics: du2,u2,dT2,T2,mass,minrho,maxrho,maxu2,bad,sumux,sumuy,count,minT,maxT,sumT.
__device__ double combine(double a,double b,int s) {
  return (s==5||s==12)?fmin(a,b):((s==6||s==7||s==13)?fmax(a,b):a+b);
}
__device__ double neutral(int s) {
  return (s==5||s==12)?INFINITY:((s==6||s==7||s==13)?-INFINITY:0.0);
}
__device__ void block_reduce(double* values,double* shared,double* out,int stride,int index) {
  int lane=threadIdx.x&31,warp=threadIdx.x/32,warps=blockDim.x/32;
  #pragma unroll
  for(int s=0;s<15;s++) {
    double v=values[s];
    for(int offset=16;offset;offset/=2)v=combine(v,__shfl_down_sync(0xffffffff,v,offset),s);
    if(lane==0)shared[s*warps+warp]=v;
  }
  __syncthreads();
  if(warp==0) {
    #pragma unroll
    for(int s=0;s<15;s++) {
      double v=lane<warps?shared[s*warps+lane]:neutral(s);
      for(int offset=16;offset;offset/=2)v=combine(v,__shfl_down_sync(0xffffffff,v,offset),s);
      if(lane==0)out[s*stride+index]=v;
    }
  }
}
extern "C" __global__ void reduce_stats(const double* partial,double* result,int blocks) {
  double values[15];
  #pragma unroll
  for(int s=0;s<15;s++) {
    double v=neutral(s);
    for(int b=threadIdx.x;b<blocks;b+=blockDim.x)v=combine(v,partial[s*blocks+b],s);
    values[s]=v;
  }
  __shared__ double shared[15*8];
  block_reduce(values,shared,result,1,0);
}
'''


def macros(lat,scalar):
    out=['''extern "C" __global__ void macros_stats(
      const double* f,const double* g,double* rho,double* ux,double* uy,double* uz,
      double* ax,double* ay,double* az,double* T,const unsigned char* solid,
      double* partial,int n,double fx,double fy,double fz,double buoyancy) {
      int node=blockIdx.x*blockDim.x+threadIdx.x;
      double values[15];
      #pragma unroll
      for(int s=0;s<15;s++)values[s]=neutral(s);
      if(node<n) {
        double r=0,mx=0,my=0,mz=0;bool finite=true;''']
    for k,e in enumerate(lat.e):
        out.append(f'{{double v=f[(long long){k}*n+node];finite=finite&&isfinite(v);r+=v;')
        for i,name in enumerate(('mx','my','mz')[:lat.dim]):
            if e[i]:out.append(f'{name}+=({int(e[i])})*v;')
        out.append('}')
    out.append('double oldT=T[node],t=0.0;')
    if scalar:
        for k in range(scalar.q):
            out.append(f'{{double v=g[(long long){k}*n+node];t+=v;finite=finite&&isfinite(v);}}')
        out.append('T[node]=t;')
    out.append('''double a=fx,b=fy+buoyancy*(t-.5),d=fz,den=r>0?r:1.0;
        double u=solid[node]?0.0:mx/den+.5*a;
        double v=solid[node]?0.0:my/den+.5*b;
        double z=solid[node]?0.0:mz/den+.5*d;
        double du=u-ux[node],dv=v-uy[node],dz=z-uz[node],dt=t-oldT;
        rho[node]=r;ux[node]=u;uy[node]=v;uz[node]=z;
        ax[node]=a;ay[node]=b;az[node]=d;
        values[8]=finite&&isfinite(r)&&isfinite(u)&&isfinite(v)&&isfinite(z)?0.0:1.0;
        if(!solid[node]) {
          double speed=u*u+v*v+z*z;
          values[0]=du*du+dv*dv+dz*dz;values[1]=speed;
          values[2]=dt*dt;values[3]=t*t;values[4]=r;values[5]=r;values[6]=r;values[7]=speed;
          values[9]=u;values[10]=v;values[11]=1.0;values[12]=t;values[13]=t;values[14]=t;
        }
      }
      __shared__ double shared[15*4];
      block_reduce(values,shared,partial,gridDim.x,blockIdx.x);
    }''')
    return '\n'.join(out)


def source(lat,scalar,model,forcing=True):
    code = ['#define INFINITY __longlong_as_double(0x7ff0000000000000LL)', REDUCTION,
            collision(lat,model,'fluid',forcing=forcing),pull(lat,'fluid'),macros(lat,scalar)]
    if scalar:code += [collision(scalar,'BGK','scalar',True),pull(scalar,'scalar')]
    return '\n'.join(code)
