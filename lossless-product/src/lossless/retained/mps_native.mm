#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fstream>
#include <sstream>
#include <map>
#include <array>
#include <torch/version.h>
#if TORCH_VERSION_MAJOR > 2 || TORCH_VERSION_MINOR >= 14
namespace lab_dispatch = at::mps;
#else
namespace lab_dispatch = at::native::mps;
#endif
static id<MTLComputePipelineState> pipeline(const std::string& name) {
 static id<MTLLibrary> lib=nil;
 static std::map<std::string,id<MTLComputePipelineState>> cache;
 if(!lib){
  std::ifstream f(LAB_METAL_SOURCE);std::stringstream ss;ss<<f.rdbuf();auto text=ss.str();
  NSError* error=nil;MTLCompileOptions* options=[MTLCompileOptions new];options.fastMathEnabled=NO;
  lib=[at::mps::getCurrentMPSStream()->device() newLibraryWithSource:[NSString stringWithUTF8String:text.c_str()] options:options error:&error];
  TORCH_CHECK(lib,"Metal compilation failed: ",error?[[error description] UTF8String]:"unknown");
 }
 auto it=cache.find(name);if(it!=cache.end())return it->second;
 NSError* error=nil;id<MTLFunction> function=[lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
 auto p=[at::mps::getCurrentMPSStream()->device() newComputePipelineStateWithFunction:function error:&error];
 TORCH_CHECK(p,"Metal pipeline failed: ",error?[[error description] UTF8String]:"unknown");cache[name]=p;return p;
}
static void check(const at::Tensor& x){TORCH_CHECK(x.device().is_mps() && x.scalar_type()==at::kFloat && x.is_contiguous(),"MPS contiguous float32 required");}
static void bind(id<MTLComputeCommandEncoder> e,const at::Tensor& x,int index){[e setBuffer:at::native::mps::getMTLBufferStorage(x) offset:x.storage_offset()*x.element_size() atIndex:index];}
static at::Tensor silu_mul(const at::Tensor& g,const at::Tensor& u,int64_t threads,int64_t mode){
 check(g);check(u);TORCH_CHECK(g.sizes()==u.sizes(),"shape mismatch");auto y=at::empty_like(g);
 auto stream=at::mps::getCurrentMPSStream();uint32_t n=g.numel(),m=mode;
 lab_dispatch::dispatch_sync_with_rethrow(stream->queue(),^{
  auto p=pipeline("silu_mul");TORCH_CHECK(threads>0 && threads<=p.maxTotalThreadsPerThreadgroup,"invalid thread count");
  auto e=stream->commandEncoder();[e setComputePipelineState:p];bind(e,g,0);bind(e,u,1);bind(e,y,2);
  [e setBytes:&n length:sizeof(n) atIndex:3];[e setBytes:&m length:sizeof(m) atIndex:4];
  [e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
 });return y;
}
TORCH_LIBRARY(lossless_mps,m){m.def("silu_mul(Tensor g, Tensor u, int threads, int mode) -> Tensor");}
TORCH_LIBRARY_IMPL(lossless_mps,MPS,m){m.impl("silu_mul",silu_mul);}
static at::Tensor rms(const at::Tensor& x,const at::Tensor& w,double eps,int64_t threads){
 check(x);check(w);TORCH_CHECK(w.numel()==x.size(-1),"RMS weight shape mismatch");auto y=at::empty_like(x);auto stream=at::mps::getCurrentMPSStream();uint32_t cols=x.size(-1);float epsi=eps;int64_t rows=x.numel()/cols;
 lab_dispatch::dispatch_sync_with_rethrow(stream->queue(),^{
  auto p=pipeline("rms");TORCH_CHECK(threads>0 && threads<=1024 && (threads&(threads-1))==0 && threads<=p.maxTotalThreadsPerThreadgroup,"invalid RMS threads");auto e=stream->commandEncoder();[e setComputePipelineState:p];bind(e,x,0);bind(e,w,1);bind(e,y,2);[e setBytes:&cols length:sizeof(cols) atIndex:3];[e setBytes:&epsi length:sizeof(epsi) atIndex:4];[e dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
 });return y;
}
TORCH_LIBRARY_FRAGMENT(lossless_mps,m){m.def("rms(Tensor x, Tensor w, float eps, int threads) -> Tensor");}
TORCH_LIBRARY_IMPL(lossless_mps,MPS,m){m.impl("rms",rms);}
static std::tuple<at::Tensor,at::Tensor> rope(const at::Tensor& q,const at::Tensor& k,const at::Tensor& c,const at::Tensor& s,int64_t qh,int64_t kh,int64_t threads){
 check(q);check(k);check(c);check(s);int64_t B=q.size(0),L=q.size(1),D=q.size(2)/qh;
 TORCH_CHECK(q.dim()==3 && k.dim()==3 && k.size(0)==B && k.size(1)==L && k.size(2)==kh*D && D%2==0,"invalid projections");
 TORCH_CHECK(c.dim()==3 && (c.size(0)==1 || c.size(0)==B) && c.size(1)==L && c.size(2)==D && s.sizes()==c.sizes(),"invalid rope tables");
 auto oq=at::empty({B,qh,L,D},q.options()),ok=at::empty({B,kh,L,D},k.options());std::array<uint32_t,6> a={(uint32_t)B,(uint32_t)L,(uint32_t)qh,(uint32_t)kh,(uint32_t)D,(uint32_t)c.size(0)};auto stream=at::mps::getCurrentMPSStream();int64_t n=oq.numel()+ok.numel();
 lab_dispatch::dispatch_sync_with_rethrow(stream->queue(),^{auto p=pipeline("rope");TORCH_CHECK(threads>0 && threads<=p.maxTotalThreadsPerThreadgroup,"invalid threads");auto e=stream->commandEncoder();[e setComputePipelineState:p];bind(e,q,0);bind(e,k,1);bind(e,c,2);bind(e,s,3);bind(e,oq,4);bind(e,ok,5);[e setBytes:a.data() length:sizeof(a) atIndex:6];[e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];});return {oq,ok};
}
TORCH_LIBRARY_FRAGMENT(lossless_mps,m){m.def("rope(Tensor q, Tensor k, Tensor c, Tensor s, int qh, int kh, int threads) -> (Tensor, Tensor)");}
TORCH_LIBRARY_IMPL(lossless_mps,MPS,m){m.impl("rope",rope);}
static at::Tensor norm_scale(const at::Tensor& x,const at::Tensor& scale,const at::Tensor& w,int64_t threads){
 check(x);check(scale);check(w);uint32_t cols=x.size(-1),n=x.numel();TORCH_CHECK(w.numel()==cols && scale.numel()==n/cols,"norm shape mismatch");auto y=at::empty_like(x);auto stream=at::mps::getCurrentMPSStream();
 lab_dispatch::dispatch_sync_with_rethrow(stream->queue(),^{auto p=pipeline("norm_scale");TORCH_CHECK(threads>0 && threads<=p.maxTotalThreadsPerThreadgroup,"invalid threads");auto e=stream->commandEncoder();[e setComputePipelineState:p];bind(e,x,0);bind(e,scale,1);bind(e,w,2);bind(e,y,3);[e setBytes:&n length:sizeof(n) atIndex:4];[e setBytes:&cols length:sizeof(cols) atIndex:5];[e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];});return y;
}
TORCH_LIBRARY_FRAGMENT(lossless_mps,m){m.def("norm_scale(Tensor x, Tensor scale, Tensor w, int threads) -> Tensor");}
TORCH_LIBRARY_IMPL(lossless_mps,MPS,m){m.impl("norm_scale",norm_scale);}
static at::Tensor norm_tail(const at::Tensor& x,const at::Tensor& var,const at::Tensor& w,double eps,int64_t mode,int64_t threads){
 check(x);check(var);check(w);uint32_t cols=x.size(-1),n=x.numel(),m=mode;float epsi=eps;TORCH_CHECK(w.numel()==cols && var.numel()==n/cols,"norm shape mismatch");auto y=at::empty_like(x);auto stream=at::mps::getCurrentMPSStream();
 lab_dispatch::dispatch_sync_with_rethrow(stream->queue(),^{auto p=pipeline("norm_tail");TORCH_CHECK(threads>0 && threads<=p.maxTotalThreadsPerThreadgroup,"invalid threads");auto e=stream->commandEncoder();[e setComputePipelineState:p];bind(e,x,0);bind(e,var,1);bind(e,w,2);bind(e,y,3);[e setBytes:&n length:sizeof(n) atIndex:4];[e setBytes:&cols length:sizeof(cols) atIndex:5];[e setBytes:&epsi length:sizeof(epsi) atIndex:6];[e setBytes:&m length:sizeof(m) atIndex:7];[e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];});return y;
}
TORCH_LIBRARY_FRAGMENT(lossless_mps,m){m.def("norm_tail(Tensor x, Tensor var, Tensor w, float eps, int mode, int threads) -> Tensor");}
TORCH_LIBRARY_IMPL(lossless_mps,MPS,m){m.impl("norm_tail",norm_tail);}
static std::tuple<at::Tensor,at::Tensor> add_square(const at::Tensor& x,const at::Tensor& r,int64_t threads){
 check(x);check(r);TORCH_CHECK(x.sizes()==r.sizes(),"add_square shape mismatch");uint32_t n=x.numel();auto sum=at::empty_like(x),squared=at::empty_like(x);auto stream=at::mps::getCurrentMPSStream();
 lab_dispatch::dispatch_sync_with_rethrow(stream->queue(),^{auto p=pipeline("add_square");TORCH_CHECK(threads>0 && threads<=p.maxTotalThreadsPerThreadgroup,"invalid threads");auto e=stream->commandEncoder();[e setComputePipelineState:p];bind(e,x,0);bind(e,r,1);bind(e,sum,2);bind(e,squared,3);[e setBytes:&n length:sizeof(n) atIndex:4];[e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];});return {sum,squared};
}
TORCH_LIBRARY_FRAGMENT(lossless_mps,m){m.def("add_square(Tensor x, Tensor r, int threads) -> (Tensor, Tensor)");}
TORCH_LIBRARY_IMPL(lossless_mps,MPS,m){m.impl("add_square",add_square);}
static at::Tensor silu_packed(const at::Tensor& x,int64_t mode,int64_t threads){
 check(x);TORCH_CHECK(x.dim()>0 && x.size(-1)>0 && x.size(-1)%2==0,"packed projection shape");auto sizes=x.sizes().vec();sizes.back()/=2;auto y=at::empty(sizes,x.options());uint32_t n=y.numel(),cols=sizes.back(),m=mode;auto stream=at::mps::getCurrentMPSStream();
 lab_dispatch::dispatch_sync_with_rethrow(stream->queue(),^{auto p=pipeline("silu_packed");TORCH_CHECK(threads>0 && threads<=p.maxTotalThreadsPerThreadgroup,"invalid threads");auto e=stream->commandEncoder();[e setComputePipelineState:p];bind(e,x,0);bind(e,y,1);[e setBytes:&n length:sizeof(n) atIndex:2];[e setBytes:&cols length:sizeof(cols) atIndex:3];[e setBytes:&m length:sizeof(m) atIndex:4];[e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];});return y;
}
TORCH_LIBRARY_FRAGMENT(lossless_mps,m){m.def("silu_packed(Tensor x, int mode, int threads) -> Tensor");}
TORCH_LIBRARY_IMPL(lossless_mps,MPS,m){m.impl("silu_packed",silu_packed);}
static std::tuple<at::Tensor,at::Tensor,at::Tensor> qkv_rope(const at::Tensor& p,const at::Tensor& c,const at::Tensor& s,int64_t qh,int64_t kh,int64_t threads){
 check(p);check(c);check(s);TORCH_CHECK(p.dim()==3 && qh>0 && kh>0,"invalid QKV projection");int64_t B=p.size(0),L=p.size(1),D=p.size(2)/(qh+2*kh);
 TORCH_CHECK(B>0 && L>0 && D>0 && D%2==0 && p.size(2)==(qh+2*kh)*D,"invalid QKV dimensions");TORCH_CHECK(c.dim()==3 && (c.size(0)==1 || c.size(0)==B) && c.size(1)==L && c.size(2)==D && s.sizes()==c.sizes(),"invalid rotary tables");
 auto q=at::empty({B,qh,L,D},p.options()),k=at::empty({B,kh,L,D},p.options()),v=at::empty({B,kh,L,D},p.options());std::array<uint32_t,6> a={(uint32_t)B,(uint32_t)L,(uint32_t)qh,(uint32_t)kh,(uint32_t)D,(uint32_t)c.size(0)};auto stream=at::mps::getCurrentMPSStream();int64_t n=p.numel();
 lab_dispatch::dispatch_sync_with_rethrow(stream->queue(),^{auto ps=pipeline("qkv_rope");TORCH_CHECK(threads>0 && threads<=ps.maxTotalThreadsPerThreadgroup,"invalid threads");auto e=stream->commandEncoder();[e setComputePipelineState:ps];bind(e,p,0);bind(e,c,1);bind(e,s,2);bind(e,q,3);bind(e,k,4);bind(e,v,5);[e setBytes:a.data() length:sizeof(a) atIndex:6];[e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];});return {q,k,v};
}
TORCH_LIBRARY_FRAGMENT(lossless_mps,m){m.def("qkv_rope(Tensor p, Tensor c, Tensor s, int qh, int kh, int threads) -> (Tensor, Tensor, Tensor)");}
TORCH_LIBRARY_IMPL(lossless_mps,MPS,m){m.impl("qkv_rope",qkv_rope);}
static at::Tensor gemm(const at::Tensor& x,const at::Tensor& w,int64_t mode){
 check(x);check(w);TORCH_CHECK(x.dim()>=2 && w.dim()==2 && x.size(-1)==w.size(1),"gemm shape mismatch");
 TORCH_CHECK(mode>=0 && mode<=6,"unknown GEMM mode");
 uint32_t K=x.size(-1),N=w.size(0),M=x.numel()/K;TORCH_CHECK(mode<5 || K%4==0,"vector kernel requires K divisible by4");TORCH_CHECK(K>0 && N>0 && M>0,"empty GEMM unsupported");
 auto sizes=x.sizes().vec();sizes.back()=N;auto y=at::empty(sizes,x.options());std::array<uint32_t,3> dims={M,N,K};auto stream=at::mps::getCurrentMPSStream();
 lab_dispatch::dispatch_sync_with_rethrow(stream->queue(),^{
  auto p=pipeline(mode>=5?(mode==5?"gemv_vec4_fma":"gemv_vec4_separate"):mode==0?"gemv_simd":(mode==1?"gemm_tile16":(mode==2?"gemm_simd8":(mode==3?"gemm_simd32":"gemm_simd64"))));auto e=stream->commandEncoder();[e setComputePipelineState:p];bind(e,x,0);bind(e,w,1);bind(e,y,2);[e setBytes:dims.data() length:sizeof(dims) atIndex:3];
  if(mode==0 || mode>=5)[e dispatchThreadgroups:MTLSizeMake((N+3)/4,M,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
  else if(mode==4)[e dispatchThreadgroups:MTLSizeMake((N+63)/64,(M+31)/32,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  else if(mode==3)[e dispatchThreadgroups:MTLSizeMake((N+31)/32,(M+15)/16,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  else if(mode==2)[e dispatchThreadgroups:MTLSizeMake((N+7)/8,(M+7)/8,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
  else [e dispatchThreadgroups:MTLSizeMake((N+15)/16,(M+15)/16,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
 });return y;
}
TORCH_LIBRARY_FRAGMENT(lossless_mps,m){m.def("gemm(Tensor x, Tensor w, int mode) -> Tensor");}
TORCH_LIBRARY_IMPL(lossless_mps,MPS,m){m.impl("gemm",gemm);}
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
static at::Tensor mps_gemm(const at::Tensor& x,const at::Tensor& w){
 check(x);check(w);TORCH_CHECK(x.dim()>=2 && w.dim()==2 && x.size(-1)==w.size(1),"MPS GEMM shape mismatch");
 int64_t K=w.size(1),N=w.size(0),M=x.numel()/K;auto sizes=x.sizes().vec();sizes.back()=N;auto y=at::empty(sizes,x.options());auto stream=at::mps::getCurrentMPSStream();
 lab_dispatch::dispatch_sync_with_rethrow(stream->queue(),^{@autoreleasepool{
  static std::map<std::array<int64_t,3>,MPSMatrixMultiplication*> cache;std::array<int64_t,3> key={M,N,K};auto it=cache.find(key);MPSMatrixMultiplication* op;
  if(it==cache.end()){op=[[MPSMatrixMultiplication alloc] initWithDevice:stream->device() transposeLeft:NO transposeRight:YES resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];cache[key]=op;}else op=it->second;
  auto ad=[MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:K*4 dataType:MPSDataTypeFloat32];auto bd=[MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K rowBytes:K*4 dataType:MPSDataTypeFloat32];auto cd=[MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N*4 dataType:MPSDataTypeFloat32];
  auto a=[[MPSMatrix alloc] initWithBuffer:at::native::mps::getMTLBufferStorage(x) offset:x.storage_offset()*4 descriptor:ad];auto b=[[MPSMatrix alloc] initWithBuffer:at::native::mps::getMTLBufferStorage(w) offset:w.storage_offset()*4 descriptor:bd];auto c=[[MPSMatrix alloc] initWithBuffer:at::native::mps::getMTLBufferStorage(y) offset:y.storage_offset()*4 descriptor:cd];
  stream->endKernelCoalescing();[op encodeToCommandBuffer:stream->commandBuffer() leftMatrix:a rightMatrix:b resultMatrix:c];[a release];[b release];[c release];
 }});return y;
}
TORCH_LIBRARY_FRAGMENT(lossless_mps,m){m.def("mps_gemm(Tensor x, Tensor w) -> Tensor");}
TORCH_LIBRARY_IMPL(lossless_mps,MPS,m){m.impl("mps_gemm",mps_gemm);}
static at::Tensor gemv_silu(const at::Tensor& x,const at::Tensor& w,int64_t mode){
 check(x);check(w);TORCH_CHECK(x.dim()>=2 && w.dim()==2 && x.size(-1)==w.size(1) && w.size(0)%2==0,"fused GEMV shape mismatch");
 uint32_t K=w.size(1),N=w.size(0)/2,M=x.numel()/K;TORCH_CHECK(K%4==0 && (mode==0 || mode==1),"fused GEMV configuration");auto sizes=x.sizes().vec();sizes.back()=N;auto y=at::empty(sizes,x.options());std::array<uint32_t,3> dims={M,N,K};auto stream=at::mps::getCurrentMPSStream();
 lab_dispatch::dispatch_sync_with_rethrow(stream->queue(),^{auto p=pipeline(mode==0?"gemv_silu_scalar":"gemv_silu_vec4");auto e=stream->commandEncoder();[e setComputePipelineState:p];bind(e,x,0);bind(e,w,1);bind(e,y,2);[e setBytes:dims.data() length:sizeof(dims) atIndex:3];[e dispatchThreadgroups:MTLSizeMake((N+3)/4,M,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];});return y;
}
TORCH_LIBRARY_FRAGMENT(lossless_mps,m){m.def("gemv_silu(Tensor x, Tensor w, int mode) -> Tensor");}
TORCH_LIBRARY_IMPL(lossless_mps,MPS,m){m.impl("gemv_silu",gemv_silu);}
static at::Tensor gemv_add(const at::Tensor& x,const at::Tensor& w,const at::Tensor& residual){
 check(x);check(w);check(residual);TORCH_CHECK(x.dim()>=2 && w.dim()==2 && x.size(-1)==w.size(1) && w.size(1)%4==0,"GEMV add shape mismatch");
 uint32_t K=w.size(1),N=w.size(0),M=x.numel()/K;auto sizes=x.sizes().vec();sizes.back()=N;auto y=at::empty(sizes,x.options());TORCH_CHECK(y.sizes()==residual.sizes(),"residual shape mismatch");std::array<uint32_t,3> dims={M,N,K};auto stream=at::mps::getCurrentMPSStream();
 lab_dispatch::dispatch_sync_with_rethrow(stream->queue(),^{auto p=pipeline("gemv_add_vec4");auto e=stream->commandEncoder();[e setComputePipelineState:p];bind(e,x,0);bind(e,w,1);bind(e,residual,2);bind(e,y,3);[e setBytes:dims.data() length:sizeof(dims) atIndex:4];[e dispatchThreadgroups:MTLSizeMake((N+3)/4,M,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];});return y;
}
TORCH_LIBRARY_FRAGMENT(lossless_mps,m){m.def("gemv_add(Tensor x, Tensor w, Tensor residual) -> Tensor");}
TORCH_LIBRARY_IMPL(lossless_mps,MPS,m){m.impl("gemv_add",gemv_add);}
