#include <metal_stdlib>
using namespace metal;
kernel void silu_mul(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]], device float* y [[buffer(2)]],constant uint& n [[buffer(3)]],constant uint& mode [[buffer(4)]],uint i [[thread_position_in_grid]]) {
 if(i<n){float v=g[i];float e=exp(-v);float s=1.0f/(1.0f+e);float a=mode==0?v*s:v/(1.0f+e);y[i]=a*u[i];}
}
kernel void rms(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device float* y [[buffer(2)]],constant uint& cols [[buffer(3)]],constant float& eps [[buffer(4)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint nt [[threads_per_threadgroup]]) {
 threadgroup float sum[1024];float s=0;
 for(uint j=tid;j<cols;j+=nt){float v=x[row*cols+j];s+=v*v;}
 sum[tid]=s;threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint stride=nt/2;stride>0;stride/=2){if(tid<stride)sum[tid]+=sum[tid+stride];threadgroup_barrier(mem_flags::mem_threadgroup);}
 float scale=rsqrt(sum[0]/float(cols)+eps);
 for(uint j=tid;j<cols;j+=nt)y[row*cols+j]=(x[row*cols+j]*scale)*w[j];
}
kernel void rope(device const float* q [[buffer(0)]],device const float* k [[buffer(1)]],device const float* c [[buffer(2)]],device const float* s [[buffer(3)]],device float* oq [[buffer(4)]],device float* ok [[buffer(5)]],constant uint* a [[buffer(6)]],uint i [[thread_position_in_grid]]) {
 uint B=a[0],L=a[1],Q=a[2],K=a[3],D=a[4],CB=a[5];uint nq=B*Q*L*D,nk=B*K*L*D;
 if(i>=nq+nk)return;bool isq=i<nq;uint z=isq?i:i-nq,H=isq?Q:K;
 uint j=z%D,t=(z/D)%L,h=(z/(D*L))%H,b=z/(D*L*H);uint src=((b*L+t)*H+h)*D;uint pos=((CB==1?0:b)*L+t)*D+j;
 uint other=j<D/2?j+D/2:j-D/2;float sign=j<D/2?-1.0f:1.0f;
 float v=isq?q[src+j]:k[src+j],r=isq?q[src+other]:k[src+other];float left=v*c[pos];float right=(r*sign)*s[pos];float val=left+right;
 if(isq)oq[z]=val;else ok[z]=val;
}
kernel void norm_scale(device const float* x [[buffer(0)]],device const float* scale [[buffer(1)]],device const float* w [[buffer(2)]],device float* y [[buffer(3)]],constant uint& n [[buffer(4)]],constant uint& cols [[buffer(5)]],uint i [[thread_position_in_grid]]) {
 if(i<n){float v=x[i]*scale[i/cols];y[i]=v*w[i%cols];}
}
kernel void norm_tail(device const float* x [[buffer(0)]],device const float* variance [[buffer(1)]],device const float* w [[buffer(2)]],device float* y [[buffer(3)]],constant uint& n [[buffer(4)]],constant uint& cols [[buffer(5)]],constant float& eps [[buffer(6)]],constant uint& mode [[buffer(7)]],uint i [[thread_position_in_grid]]) {
 if(i<n){float v=variance[i/cols]+eps;float scale=mode==0?rsqrt(v):(mode==2?fast::rsqrt(v):1.0f/sqrt(v));float z=x[i]*scale;y[i]=z*w[i%cols];}
}
kernel void add_square(device const float* x [[buffer(0)]],device const float* r [[buffer(1)]],device float* sum [[buffer(2)]],device float* squared [[buffer(3)]],constant uint& n [[buffer(4)]],uint i [[thread_position_in_grid]]) {
 if(i<n){float v=x[i]+r[i];sum[i]=v;squared[i]=v*v;}
}
kernel void silu_packed(device const float* p [[buffer(0)]],device float* y [[buffer(1)]],constant uint& n [[buffer(2)]],constant uint& cols [[buffer(3)]],constant uint& mode [[buffer(4)]],uint i [[thread_position_in_grid]]) {
 if(i<n){uint j=i%cols,row=i/cols;float g=p[row*cols*2+j],u=p[row*cols*2+cols+j];float e=exp(-g);float a=mode==0?g*(1.0f/(1.0f+e)):g/(1.0f+e);y[i]=a*u;}
}
kernel void qkv_rope(device const float* p [[buffer(0)]],device const float* c [[buffer(1)]],device const float* s [[buffer(2)]],device float* oq [[buffer(3)]],device float* ok [[buffer(4)]],device float* ov [[buffer(5)]],constant uint* a [[buffer(6)]],uint i [[thread_position_in_grid]]) {
 uint B=a[0],L=a[1],Q=a[2],K=a[3],D=a[4],CB=a[5],nq=B*Q*L*D,nk=B*K*L*D;if(i>=nq+2*nk)return;
 uint kind=i<nq?0:(i<nq+nk?1:2);uint z=kind==0?i:(kind==1?i-nq:i-nq-nk),H=kind==0?Q:K;
 uint j=z%D,t=(z/D)%L,h=(z/(D*L))%H,b=z/(D*L*H);uint start=kind==0?0:(kind==1?Q:Q+K);
 uint src=(b*L+t)*(Q+2*K)*D+(start+h)*D;
 if(kind==2){ov[z]=p[src+j];return;}
 uint pos=((CB==1?0:b)*L+t)*D+j,other=j<D/2?j+D/2:j-D/2;float sign=j<D/2?-1.0f:1.0f;
 float left=p[src+j]*c[pos],right=(p[src+other]*sign)*s[pos],value=left+right;
 if(kind==0)oq[z]=value;else ok[z]=value;
}
// Float32 dot products, one SIMD group per output; no precision conversion.
kernel void gemv_simd(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device float* y [[buffer(2)]],constant uint* d [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 uint N=d[1],K=d[2],n=group.x*4+tid/32,row=group.y,lane=tid%32;
 float acc=0.0f;
 if(n<N)for(uint k=lane;k<K;k+=32)acc=fma(x[row*K+k],w[n*K+k],acc);
 acc=simd_sum(acc);if(lane==0 && n<N)y[row*N+n]=acc;
}
// Shared-memory square tiles; explicit float32 FMA and bounded fringe accesses.
kernel void gemm_tile16(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device float* y [[buffer(2)]],constant uint* d [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 uint M=d[0],N=d[1],K=d[2],tx=tid%16,ty=tid/16,row=group.y*16+ty,col=group.x*16+tx;
 threadgroup float a[256];threadgroup float b[256];float acc=0.0f;
 for(uint base=0;base<K;base+=16){
  a[tid]=(row<M && base+tx<K)?x[row*K+base+tx]:0.0f;
  b[tid]=(col<N && base+ty<K)?w[col*K+base+ty]:0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=0;k<16;k++)acc=fma(a[ty*16+k],b[k*16+tx],acc);
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 if(row<M && col<N)y[row*N+col]=acc;
}
#include <metal_simdgroup_matrix>
kernel void gemm_simd8(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device float* y [[buffer(2)]],constant uint* d [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 uint M=d[0],N=d[1],K=d[2];threadgroup float aa[64],bb[64],cc[64];
 simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8>(0.0f);
 for(uint base=0;base<K;base+=8){
  for(uint j=tid;j<64;j+=32){uint row=group.y*8+j/8,col=group.x*8+j%8;
   aa[j]=(row<M && base+j%8<K)?x[row*K+base+j%8]:0.0f;
   bb[j]=(col<N && base+j/8<K)?w[col*K+base+j/8]:0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 a,b;simdgroup_load(a,aa,8);simdgroup_load(b,bb,8);
  simdgroup_multiply_accumulate(acc,a,b,acc);threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 simdgroup_store(acc,cc,8);threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint j=tid;j<64;j+=32){uint row=group.y*8+j/8,col=group.x*8+j%8;if(row<M && col<N)y[row*N+col]=cc[j];}
}
// Eight SIMD groups share a16x32 output tile and stage32 reduction elements.
kernel void gemm_simd32(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device float* y [[buffer(2)]],constant uint* d [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 uint M=d[0],N=d[1],K=d[2],sg=tid/32,sy=(sg/4)*8,sx=(sg%4)*8;
 threadgroup float aa[512],bb[1024],cc[512];simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8>(0.0f);
 for(uint base=0;base<K;base+=32){
  for(uint j=tid;j<512;j+=256){uint row=group.y*16+j/32,kk=base+j%32;aa[j]=(row<M && kk<K)?x[row*K+kk]:0.0f;}
  for(uint j=tid;j<1024;j+=256){uint col=group.x*32+j%32,kk=base+j/32;bb[j]=(col<N && kk<K)?w[col*K+kk]:0.0f;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint kk=0;kk<32;kk+=8){simdgroup_float8x8 a,b;simdgroup_load(a,aa+sy*32+kk,32);simdgroup_load(b,bb+kk*32+sx,32);simdgroup_multiply_accumulate(acc,a,b,acc);}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 simdgroup_store(acc,cc+sy*32+sx,32);threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint j=tid;j<512;j+=256){uint row=group.y*16+j/32,col=group.x*32+j%32;if(row<M && col<N)y[row*N+col]=cc[j];}
}
//32x64 output tile, four accumulators per SIMD group,12KiB staged input.
kernel void gemm_simd64(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device float* y [[buffer(2)]],constant uint* d [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 uint M=d[0],N=d[1],K=d[2],sg=tid/32,sy=(sg/4)*16,sx=(sg%4)*16;
 threadgroup float aa[1024],bb[2048],cc[2048];
 simdgroup_float8x8 c00=make_filled_simdgroup_matrix<float,8>(0.0f),c01=c00,c10=c00,c11=c00;
 for(uint base=0;base<K;base+=32){
  for(uint j=tid;j<1024;j+=256){uint row=group.y*32+j/32,kk=base+j%32;aa[j]=(row<M && kk<K)?x[row*K+kk]:0.0f;}
  for(uint j=tid;j<2048;j+=256){uint col=group.x*64+j%64,kk=base+j/64;bb[j]=(col<N && kk<K)?w[col*K+kk]:0.0f;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint kk=0;kk<32;kk+=8){
   simdgroup_float8x8 a0,a1,b0,b1;simdgroup_load(a0,aa+sy*32+kk,32);simdgroup_load(a1,aa+(sy+8)*32+kk,32);simdgroup_load(b0,bb+kk*64+sx,64);simdgroup_load(b1,bb+kk*64+sx+8,64);
   simdgroup_multiply_accumulate(c00,a0,b0,c00);simdgroup_multiply_accumulate(c01,a0,b1,c01);simdgroup_multiply_accumulate(c10,a1,b0,c10);simdgroup_multiply_accumulate(c11,a1,b1,c11);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 simdgroup_store(c00,cc+sy*64+sx,64);simdgroup_store(c01,cc+sy*64+sx+8,64);simdgroup_store(c10,cc+(sy+8)*64+sx,64);simdgroup_store(c11,cc+(sy+8)*64+sx+8,64);threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint j=tid;j<2048;j+=256){uint row=group.y*32+j/64,col=group.x*64+j%64;if(row<M && col<N)y[row*N+col]=cc[j];}
}

kernel void gemv_vec4_fma(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device float* y [[buffer(2)]],constant uint* d [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 uint N=d[1],K=d[2],n=group.x*4+tid/32,row=group.y,lane=tid%32;float acc=0.0f;uint k=lane*4;
 if(n<N){
  for(;k+388<=K;k+=512){
   float4 x0=*((device const float4*)(x+row*K+k)),x1=*((device const float4*)(x+row*K+k+128)),x2=*((device const float4*)(x+row*K+k+256)),x3=*((device const float4*)(x+row*K+k+384));
   float4 a0=*((device const float4*)(w+n*K+k)),a1=*((device const float4*)(w+n*K+k+128)),a2=*((device const float4*)(w+n*K+k+256)),a3=*((device const float4*)(w+n*K+k+384));
   for(uint i=0;i<4;i++){acc=fma(a0[i],x0[i],acc);acc=fma(a1[i],x1[i],acc);acc=fma(a2[i],x2[i],acc);acc=fma(a3[i],x3[i],acc);}
  }
  for(;k+4<=K;k+=128){float4 av=*((device const float4*)(w+n*K+k)),xv=*((device const float4*)(x+row*K+k));for(uint i=0;i<4;i++){acc=fma(av[i],xv[i],acc);}}
 }
 acc=simd_sum(acc);if(lane==0 && n<N)y[row*N+n]=acc;
}

kernel void gemv_vec4_separate(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device float* y [[buffer(2)]],constant uint* d [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 uint N=d[1],K=d[2],n=group.x*4+tid/32,row=group.y,lane=tid%32;float acc=0.0f;uint k=lane*4;
 if(n<N){
  for(;k+388<=K;k+=512){
   float4 x0=*((device const float4*)(x+row*K+k)),x1=*((device const float4*)(x+row*K+k+128)),x2=*((device const float4*)(x+row*K+k+256)),x3=*((device const float4*)(x+row*K+k+384));
   float4 a0=*((device const float4*)(w+n*K+k)),a1=*((device const float4*)(w+n*K+k+128)),a2=*((device const float4*)(w+n*K+k+256)),a3=*((device const float4*)(w+n*K+k+384));
   for(uint i=0;i<4;i++){float p0=a0[i]*x0[i];acc=acc+p0;float p1=a1[i]*x1[i];acc=acc+p1;float p2=a2[i]*x2[i];acc=acc+p2;float p3=a3[i]*x3[i];acc=acc+p3;}
  }
  for(;k+4<=K;k+=128){float4 av=*((device const float4*)(w+n*K+k)),xv=*((device const float4*)(x+row*K+k));for(uint i=0;i<4;i++){float product=av[i]*xv[i];acc=acc+product;}}
 }
 acc=simd_sum(acc);if(lane==0 && n<N)y[row*N+n]=acc;
}
kernel void gemv_silu_scalar(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device float* y [[buffer(2)]],constant uint* d [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 uint N=d[1],K=d[2],n=group.x*4+tid/32,row=group.y,lane=tid%32;float g=0.0f,u=0.0f;
 if(n<N)for(uint k=lane;k<K;k+=32){float v=x[row*K+k];g=fma(v,w[n*K+k],g);u=fma(v,w[(n+N)*K+k],u);}
 g=simd_sum(g);u=simd_sum(u);if(lane==0 && n<N)y[row*N+n]=(g/(1.0f+exp(-g)))*u;
}
kernel void gemv_silu_vec4(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device float* y [[buffer(2)]],constant uint* d [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 uint N=d[1],K=d[2],n=group.x*4+tid/32,row=group.y,lane=tid%32;float g=0.0f,u=0.0f;uint k=lane*4;
 if(n<N){
  for(;k+388<=K;k+=512){
   float4 xv[4],gv[4],uv[4];for(uint j=0;j<4;j++){xv[j]=*((device const float4*)(x+row*K+k+j*128));gv[j]=*((device const float4*)(w+n*K+k+j*128));uv[j]=*((device const float4*)(w+(n+N)*K+k+j*128));}
   for(uint i=0;i<4;i++)for(uint j=0;j<4;j++){g=fma(gv[j][i],xv[j][i],g);u=fma(uv[j][i],xv[j][i],u);}
  }
  for(;k+4<=K;k+=128){float4 xv=*((device const float4*)(x+row*K+k)),gv=*((device const float4*)(w+n*K+k)),uv=*((device const float4*)(w+(n+N)*K+k));for(uint i=0;i<4;i++){g=fma(gv[i],xv[i],g);u=fma(uv[i],xv[i],u);}}
 }
 g=simd_sum(g);u=simd_sum(u);if(lane==0 && n<N)y[row*N+n]=(g/(1.0f+exp(-g)))*u;
}

kernel void gemv_add_vec4(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device const float* residual [[buffer(2)]],device float* y [[buffer(3)]],constant uint* d [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 uint N=d[1],K=d[2],n=group.x*4+tid/32,row=group.y,lane=tid%32;float acc=0.0f;uint k=lane*4;
 if(n<N){
  for(;k+388<=K;k+=512){
   float4 x0=*((device const float4*)(x+row*K+k)),x1=*((device const float4*)(x+row*K+k+128)),x2=*((device const float4*)(x+row*K+k+256)),x3=*((device const float4*)(x+row*K+k+384));
   float4 a0=*((device const float4*)(w+n*K+k)),a1=*((device const float4*)(w+n*K+k+128)),a2=*((device const float4*)(w+n*K+k+256)),a3=*((device const float4*)(w+n*K+k+384));
   for(uint i=0;i<4;i++){acc=fma(a0[i],x0[i],acc);acc=fma(a1[i],x1[i],acc);acc=fma(a2[i],x2[i],acc);acc=fma(a3[i],x3[i],acc);}
  }
  for(;k+4<=K;k+=128){float4 av=*((device const float4*)(w+n*K+k)),xv=*((device const float4*)(x+row*K+k));for(uint i=0;i<4;i++){acc=fma(av[i],xv[i],acc);}}
 }
 acc=simd_sum(acc);if(lane==0 && n<N)y[row*N+n]=residual[row*N+n]+acc;
}
