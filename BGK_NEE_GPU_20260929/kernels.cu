__device__ __constant__ int EX[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
__device__ __constant__ int EY[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
__device__ __constant__ int OPP[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
__device__ __constant__ double WEIGHT[9] = {
    4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};

__device__ double sum9(const double* a) {
    return (((a[0]+a[1])+(a[2]+a[3]))+((a[4]+a[5])+(a[6]+a[7])))+a[8];
}

__device__ double equilibrium(int k, double r, double u, double v) {
    double eu = EX[k]*u + EY[k]*v;
    double uu = u*u + v*v;
    return (WEIGHT[k]*r) * (1.0+3.0*eu+4.5*eu*eu-1.5*uu);
}

__device__ void moments(const double* a, double* r, double* u, double* v, double cutoff) {
    double mx[9], my[9];
    #pragma unroll
    for (int k=0; k<9; ++k) {
        mx[k] = a[k]*EX[k];
        my[k] = a[k]*EY[k];
    }
    *r = sum9(a);
    double safe = *r > cutoff ? *r : 1.0;
    *u = sum9(mx)/safe;
    *v = sum9(my)/safe;
}

extern "C" __global__ void pull_bgk(
    const double* old_f, const double* old_r, const double* old_u, const double* old_v,
    double* next_f, double* next_r, double* next_u, double* next_v,
    double* donor_u, double* donor_v, int nx, int ny, double omega
) {
    int p = blockDim.x*blockIdx.x+threadIdx.x;
    if (p >= nx*ny) return;
    int x=p%nx, y=p/nx;
    if (x==0 || x==nx-1 || y==0 || y==ny-1) return;
    double a[9];
    #pragma unroll
    for (int k=0; k<9; ++k) {
        int source=(y-EY[k])*nx+x-EX[k];
        double population=old_f[9ULL*source+k];
        double eq=equilibrium(k, old_r[source], old_u[source], old_v[source]);
        a[k]=population-omega*(population-eq);
        next_f[9ULL*p+k]=a[k];
    }
    double r, u, v;
    moments(a, &r, &u, &v, 0.0);
    next_r[p]=r;
    donor_u[p]=u;
    donor_v[p]=v;
    if (r > 0.0 && r <= 1e-14) {
        moments(a, &r, &u, &v, 1e-14);
    }
    next_u[p]=u;
    next_v[p]=v;
}

extern "C" __global__ void nee_walls(
    double* f, double* r, double* u, double* v,
    const double* donor_u, const double* donor_v,
    const double* wall_velocity, double ramp, int nx, int ny
) {
    int p=blockDim.x*blockIdx.x+threadIdx.x;
    if (p>=nx*ny) return;
    int x=p%nx, y=p/nx;
    if (x>0 && x<nx-1 && y>0 && y<ny-1) return;
    int dx=x==0 ? 1 : (x==nx-1 ? nx-2 : x);
    int dy=y==0 ? 1 : (y==ny-1 ? ny-2 : y);
    int donor=dy*nx+dx;
    // Top/bottom take precedence at corners; every donor is an interior node.
    int side=y==0 ? 2 : (y==ny-1 ? 3 : (x==0 ? 0 : 1));
    double wall_u=wall_velocity[2*side]*ramp;
    double wall_v=wall_velocity[2*side+1]*ramp;
    double a[9];
    #pragma unroll
    for (int k=0; k<9; ++k) {
        double ew=equilibrium(k, r[donor], wall_u, wall_v);
        double ed=equilibrium(k, r[donor], donor_u[donor], donor_v[donor]);
        a[k]=ew+(f[9ULL*donor+k]-ed);
        f[9ULL*p+k]=a[k];
    }
    double rb, ub, vb;
    moments(a, &rb, &ub, &vb, 1e-14);
    r[p]=rb; u[p]=ub; v[p]=vb;
}

// Materialize post-collision populations once per cell for MRT. Pulling and
// redoing a full moment transform at every neighbour would multiply its cost.
extern "C" __global__ void collide_models(
    const double* f, const double* r, const double* u, const double* v, double* post,
    const double* matrix, const double* inverse, const double* rates,
    int nx, int ny, double omega, int mrt
) {
    int p=blockDim.x*blockIdx.x+threadIdx.x;
    if (p>=nx*ny) return;
    double delta[9], dm[9];
    #pragma unroll
    for (int k=0; k<9; ++k) delta[k]=f[9ULL*p+k]-equilibrium(k,r[p],u[p],v[p]);
    if (mrt) {
        #pragma unroll
        for (int a=0; a<9; ++a) {
            double total=0.0;
            #pragma unroll
            for (int k=0; k<9; ++k) total+=matrix[9*a+k]*delta[k];
            dm[a]=rates[a]*total;
        }
    }
    #pragma unroll
    for (int k=0; k<9; ++k) {
        double change=omega*delta[k];
        if (mrt) {
            change=0.0;
            #pragma unroll
            for (int a=0; a<9; ++a) change+=inverse[9*k+a]*dm[a];
        }
        post[9ULL*p+k]=f[9ULL*p+k]-change;
    }
}

extern "C" __global__ void pull_models(
    const double* post, const double* old_r,
    double* next_f, double* next_r, double* next_u, double* next_v,
    double* donor_u, double* donor_v, int nx, int ny, int halfway, double lid_u
) {
    int p=blockDim.x*blockIdx.x+threadIdx.x;
    if (p>=nx*ny) return;
    int x=p%nx, y=p/nx;
    if (!halfway && (x==0 || x==nx-1 || y==0 || y==ny-1)) return;
    double a[9];
    #pragma unroll
    for (int k=0; k<9; ++k) {
        int sx=x-EX[k], sy=y-EY[k];
        if (sx<0 || sx>=nx || sy<0 || sy>=ny) {
            a[k]=post[9ULL*p+OPP[k]];
            // Top/bottom priority at double-hit corners; static other walls.
            if (sy>=ny) a[k]+=6.0*WEIGHT[k]*old_r[p]*EX[k]*lid_u;
        } else {
            a[k]=post[9ULL*(sy*nx+sx)+k];
        }
        next_f[9ULL*p+k]=a[k];
    }
    double r,u,v;
    moments(a,&r,&u,&v,0.0);
    donor_u[p]=u; donor_v[p]=v;
    moments(a,&r,&u,&v,1e-14);
    next_r[p]=r; next_u[p]=u; next_v[p]=v;
}

extern "C" __global__ void diagnostic_blocks(
    const double* f, const double* r, const double* u, const double* v,
    const double* previous_u, const double* previous_v,
    double* blocks, int nx, int ny, int all_fluid
) {
    __shared__ double values[6][256];
    int t=threadIdx.x, p=blockIdx.x*blockDim.x+t;
    double du2=0.0, u2=0.0, mass=0.0, speed2=0.0, minimum=__longlong_as_double(0x7ff0000000000000LL), bad=0.0;
    if (p<nx*ny) {
        int x=p%nx, y=p/nx;
        mass=r[p]; minimum=r[p]; speed2=u[p]*u[p]+v[p]*v[p];
        bad=!(isfinite(r[p]) && isfinite(u[p]) && isfinite(v[p]));
        #pragma unroll
        for (int k=0; k<9; ++k) {
            if (!isfinite(f[9ULL*p+k])) bad=1.0;
        }
        if (all_fluid || (x>0 && x<nx-1 && y>0 && y<ny-1)) {
            double du=u[p]-previous_u[p], dv=v[p]-previous_v[p];
            du2=du*du+dv*dv; u2=speed2;
        }
    }
    values[0][t]=du2; values[1][t]=u2; values[2][t]=mass;
    values[3][t]=speed2; values[4][t]=minimum; values[5][t]=bad;
    __syncthreads();
    for (int offset=128; offset>0; offset/=2) {
        if (t<offset) {
            values[0][t]+=values[0][t+offset];
            values[1][t]+=values[1][t+offset];
            values[2][t]+=values[2][t+offset];
            values[3][t]=fmax(values[3][t], values[3][t+offset]);
            values[4][t]=fmin(values[4][t], values[4][t+offset]);
            values[5][t]=fmax(values[5][t], values[5][t+offset]);
        }
        __syncthreads();
    }
    if (t==0) {
        for (int j=0; j<6; ++j) blocks[6*blockIdx.x+j]=values[j][0];
    }
}

extern "C" __global__ void diagnostic_finish(const double* blocks, double* output, int count) {
    __shared__ double values[6][256];
    int t=threadIdx.x;
    double local[6]={0.0, 0.0, 0.0, 0.0, __longlong_as_double(0x7ff0000000000000LL), 0.0};
    for (int b=t; b<count; b+=blockDim.x) {
        local[0]+=blocks[6*b]; local[1]+=blocks[6*b+1]; local[2]+=blocks[6*b+2];
        local[3]=fmax(local[3], blocks[6*b+3]);
        local[4]=fmin(local[4], blocks[6*b+4]);
        local[5]=fmax(local[5], blocks[6*b+5]);
    }
    for (int j=0; j<6; ++j) values[j][t]=local[j];
    __syncthreads();
    for (int offset=128; offset>0; offset/=2) {
        if (t<offset) {
            values[0][t]+=values[0][t+offset];
            values[1][t]+=values[1][t+offset];
            values[2][t]+=values[2][t+offset];
            values[3][t]=fmax(values[3][t], values[3][t+offset]);
            values[4][t]=fmin(values[4][t], values[4][t+offset]);
            values[5][t]=fmax(values[5][t], values[5][t+offset]);
        }
        __syncthreads();
    }
    if (t==0) {
        for (int j=0; j<6; ++j) output[j]=values[j][0];
    }
}
