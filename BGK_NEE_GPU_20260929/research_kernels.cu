// Runtime quadratures (q <= 27), double precision, Guo force and link walls.
extern "C" __global__ void collide_general(
 const double* f, double* post, const double* rho,
 const double* ux, const double* uy, const double* uz,
 const double* ax, const double* ay, const double* az,
 const int* e, const double* w, const int* opp,
 const double* M, const double* MI, const double* rates,
 int n, int q, int model, int scalar, double c, double op, double om) {
 int node=blockDim.x*blockIdx.x+threadIdx.x;
 if(node>=n) return;
 double u=ux[node],v=uy[node],z=uz[node],r=rho[node];
 double a=scalar?0:ax[node],b=scalar?0:ay[node],d=scalar?0:az[node];
 double u2=u*u+v*v+z*z,ua=u*a+v*b+z*d;
 double delta[27],force[27],change[27];
 long long base=(long long)node*q;
 for(int k=0;k<q;k++) {
   double eu=e[3*k]*u+e[3*k+1]*v+e[3*k+2]*z;
   double ea=e[3*k]*a+e[3*k+1]*b+e[3*k+2]*d;
   double eq=w[k]*r*(1+eu/c+(scalar?0:eu*eu/(2*c*c)-u2/(2*c)));
   delta[k]=f[base+k]-eq;
   force[k]=scalar?0:w[k]*r*((ea-ua)/c+eu*ea/(c*c));
 }
 if(model==2) {
   for(int i=0;i<q;i++) {
     double dm=0,fm=0;
     for(int j=0;j<q;j++){dm+=M[i*q+j]*delta[j];fm+=M[i*q+j]*force[j];}
     change[i]=rates[i]*dm-(1-.5*rates[i])*fm;
   }
 }
 for(int k=0;k<q;k++) {
   double correction;
   if(model==2) {
     correction=0;
     for(int j=0;j<q;j++) correction+=MI[k*q+j]*change[j];
   } else if(model==1) {
     int j=opp[k];
     correction=.5*(op*(delta[k]+delta[j])+om*(delta[k]-delta[j])
                -(1-.5*op)*(force[k]+force[j])-(1-.5*om)*(force[k]-force[j]));
   } else correction=op*delta[k]-(1-.5*op)*force[k];
   post[base+k]=f[base+k]-correction;
 }
}

extern "C" __global__ void pull_general(
 const double* post,double* f,const double* rho,const unsigned char* solid,
 const int* e,const double* w,const int* opp,
 int nx,int ny,int nz,int q,int dim,int px,int py,int pz,
 double cs2,double lid,int heat_wall) {
 int node=blockDim.x*blockIdx.x+threadIdx.x;
 int n=nx*ny*nz;
 if(node>=n) return;
 int x=node%nx,y=(node/nx)%ny,z=node/(nx*ny);
 long long base=(long long)node*q;
 for(int k=0;k<q;k++) {
   if(solid[node]) {f[base+k]=post[base+k];continue;}
   int sx=x-e[3*k],sy=y-e[3*k+1],sz=z-e[3*k+2];
   bool outside=(!px&&(sx<0||sx>=nx))||(!py&&(sy<0||sy>=ny))
               ||(dim==3&&!pz&&(sz<0||sz>=nz));
   int wrapped=((sz%nz+nz)%nz*ny+(sy%ny+ny)%ny)*nx+(sx%nx+nx)%nx;
   if(outside||solid[wrapped]) {
     double reflected=post[base+opp[k]];
     if((heat_wall==1&&sx<0)||(heat_wall==2&&sy<0)) reflected=-reflected+2*w[k];
     else if((heat_wall==1&&sx>=nx)||(heat_wall==2&&sy>=ny)) reflected=-reflected;
     else if(!heat_wall&&outside&&sy>=ny) reflected+=2*w[k]*rho[node]*e[3*k]*lid/cs2;
     f[base+k]=reflected;
   } else f[base+k]=post[(long long)wrapped*q+k];
 }
}
