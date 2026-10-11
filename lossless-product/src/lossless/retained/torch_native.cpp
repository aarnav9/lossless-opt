#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <cmath>
#include <tuple>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

static void check(const at::Tensor& x, const at::Tensor& r, const at::Tensor& w) {
    TORCH_CHECK(x.device().is_cpu() && r.device().is_cpu() && w.device().is_cpu(), "CPU only");
    TORCH_CHECK(x.scalar_type()==at::kFloat && r.scalar_type()==at::kFloat && w.scalar_type()==at::kFloat, "float32 only");
    TORCH_CHECK(x.dim()==2 && r.sizes()==x.sizes() && w.dim()==1 && w.numel()==x.size(1), "shape mismatch");
    TORCH_CHECK(x.size(0)>0 && x.size(1)>0, "positive sizes required");
    TORCH_CHECK(x.is_contiguous() && r.is_contiguous() && w.is_contiguous(), "contiguous tensors required");
}

static std::tuple<at::Tensor,at::Tensor> rms_core(const at::Tensor& x, const at::Tensor& r,
                                               const at::Tensor& w, int64_t algorithm, int64_t grain, bool epilogue, bool weighted=true) {
    check(x,r,w);
    if(algorithm==2) {
        auto sum=at::add(x,r);
        auto variance=at::mean(at::pow(sum,2),{-1},true);
        auto y=at::mul(sum,at::rsqrt(variance+1e-5));
        if(weighted)y=at::mul(y,w);
        if(epilogue)y=at::add(y,sum);
        return {y,sum};
    }
    TORCH_CHECK(algorithm==0 || algorithm==1, "unknown algorithm");
    auto y=at::empty_like(x), sum=epilogue ? y : at::empty_like(x);
    const float* xp=x.data_ptr<float>(); const float* rp=r.data_ptr<float>(); const float* wp=w.data_ptr<float>();
    float* yp=y.data_ptr<float>(); float* sp=sum.data_ptr<float>();
    const int64_t cols=x.size(1), rows=x.size(0);
    auto work=[&](int64_t begin,int64_t end) {
        for(int64_t row=begin;row<end;++row) {
            const float* xx=xp+row*cols; const float* rr=rp+row*cols;
            float* yy=yp+row*cols; float* ss=sp+row*cols;
            int64_t j=0; double total=0;
#if defined(__aarch64__)
            float64x2_t lo=vdupq_n_f64(0), hi=vdupq_n_f64(0);
            float32x4_t fast=vdupq_n_f32(0);
            for(;j+4<=cols;j+=4) {
                float32x4_t v=vaddq_f32(vld1q_f32(xx+j),vld1q_f32(rr+j)); vst1q_f32(ss+j,v);
                float32x4_t q=vmulq_f32(v,v);
                if(algorithm==0) {
                    lo=vaddq_f64(lo,vcvt_f64_f32(vget_low_f32(q)));
                    hi=vaddq_f64(hi,vcvt_f64_f32(vget_high_f32(q)));
                } else fast=vaddq_f32(fast,q);
            }
            total=algorithm==0 ? vaddvq_f64(lo)+vaddvq_f64(hi) : (double)vaddvq_f32(fast);
#endif
            for(;j<cols;++j){float v=xx[j]+rr[j];ss[j]=v;float q=v*v;total+=(double)q;}
            float scale=(float)(1.0/std::sqrt(total/(double)cols+1e-5)); j=0;
#if defined(__aarch64__)
            float32x4_t scales=vdupq_n_f32(scale);
            for(;j+4<=cols;j+=4) {
                float32x4_t v=vmulq_f32(vld1q_f32(ss+j),scales);
                if(weighted) v=vmulq_f32(v,vld1q_f32(wp+j));
                if(epilogue) v=vaddq_f32(v,vld1q_f32(ss+j));
                vst1q_f32(yy+j,v);
            }
#endif
            for(;j<cols;++j) { float v=ss[j]*scale; if(weighted) v*=wp[j]; if(epilogue) v+=ss[j]; yy[j]=v; }
        }
    };
    if(grain>0) at::parallel_for(0,rows,grain,work); else work(0,rows);
    return {y,sum};
}

static std::tuple<at::Tensor,at::Tensor> rms_pre(const at::Tensor& x,const at::Tensor& r,
                                               const at::Tensor& w,int64_t algorithm,int64_t grain) {
    return rms_core(x,r,w,algorithm,grain,false);
}
static at::Tensor rms_region(const at::Tensor& x,const at::Tensor& r,const at::Tensor& w,int64_t algorithm,int64_t grain) {
    return std::get<0>(rms_core(x,r,w,algorithm,grain,true));
}
static at::Tensor region_meta(const at::Tensor& x,const at::Tensor& r,const at::Tensor& w,int64_t algorithm,int64_t grain) {
    return at::empty_like(x);
}
static at::Tensor rms_mlp_impl(const at::Tensor& x,const at::Tensor& r,const at::Tensor& w,
                         const at::Tensor& gw,const c10::optional<at::Tensor>& gb,const at::Tensor& uw,const c10::optional<at::Tensor>& ub,
                         const at::Tensor& dw,const c10::optional<at::Tensor>& db,int64_t algorithm,int64_t grain,bool folded) {
    auto normalized=rms_core(x,r,w,algorithm,grain,false,!folded);
    auto g=at::linear(std::get<0>(normalized),gw,gb);
    auto u=at::linear(std::get<0>(normalized),uw,ub);
    auto product=at::silu(g); product.mul_(u);
    auto out=at::linear(product,dw,db); out.add_(std::get<1>(normalized));
    return out;
}
static at::Tensor rms_mlp(const at::Tensor& x,const at::Tensor& r,const at::Tensor& w,
                         const at::Tensor& gw,const c10::optional<at::Tensor>& gb,const at::Tensor& uw,const c10::optional<at::Tensor>& ub,
                         const at::Tensor& dw,const c10::optional<at::Tensor>& db,int64_t algorithm,int64_t grain) {
    return rms_mlp_impl(x,r,w,gw,gb,uw,ub,dw,db,algorithm,grain,false);
}
static at::Tensor rms_mlp_folded(const at::Tensor& x,const at::Tensor& r,const at::Tensor& w,
                         const at::Tensor& gw,const c10::optional<at::Tensor>& gb,const at::Tensor& uw,const c10::optional<at::Tensor>& ub,
                         const at::Tensor& dw,const c10::optional<at::Tensor>& db,int64_t algorithm,int64_t grain) {
    return rms_mlp_impl(x,r,w,gw,gb,uw,ub,dw,db,algorithm,grain,true);
}
static at::Tensor rms_mlp_merged(const at::Tensor& x,const at::Tensor& r,const at::Tensor& w,
                         const at::Tensor& gw,const c10::optional<at::Tensor>& gb,const at::Tensor& uw,const c10::optional<at::Tensor>& ub,
                         const at::Tensor& dw,const c10::optional<at::Tensor>& db,int64_t algorithm,int64_t grain) {
    auto normalized=rms_core(x,r,w,algorithm,grain,false);
    auto projected=at::linear(std::get<0>(normalized),gw,gb);
    auto g=projected.narrow(-1,0,uw.size(0));
    auto u=projected.narrow(-1,uw.size(0),uw.size(0));
    auto product=at::silu(g); product.mul_(u);
    auto out=at::linear(product,dw,db);out.add_(std::get<1>(normalized));return out;
}

static at::Tensor rotate_half(const at::Tensor& t) {
    auto half=t.size(-1)/2;
    return at::cat({at::neg(t.narrow(-1,half,half)),t.narrow(-1,0,half)},-1);
}
static at::Tensor plain_norm(const at::Tensor& x,const at::Tensor& w,double eps) {
    return at::mul(w,at::mul(x,at::rsqrt(at::mean(at::pow(x,2),{-1},true)+eps)));
}
static at::Tensor plain_norm_meta(const at::Tensor& x,const at::Tensor& w,double eps) {return at::empty_like(x);}
static std::tuple<at::Tensor,at::Tensor> rope(const at::Tensor& x,const at::Tensor& inv,const at::Tensor& positions,double scaling) {
    auto frequencies=at::matmul(inv.unsqueeze(0).unsqueeze(-1).expand({positions.size(0),-1,1}).to(at::kFloat),
                               positions.unsqueeze(1).to(at::kFloat)).transpose(1,2);
    auto emb=at::cat({frequencies,frequencies},-1);
    return {at::mul(at::cos(emb),scaling).to(x.scalar_type()),at::mul(at::sin(emb),scaling).to(x.scalar_type())};
}
static std::tuple<at::Tensor,at::Tensor> rope_meta(const at::Tensor& x,const at::Tensor& inv,const at::Tensor& positions,double scaling) {
    auto shape=std::vector<int64_t>{positions.size(0),positions.size(1),inv.numel()*2};
    return {at::empty(shape,x.options()),at::empty(shape,x.options())};
}
static at::Tensor decoder(const at::Tensor& x,const at::Tensor& nw,
                         const at::Tensor& qw,const at::Tensor& kw,const at::Tensor& vw,const at::Tensor& ow,
                         const at::Tensor& cos,const at::Tensor& sin,const at::Tensor& mw,
                         const at::Tensor& gw,const at::Tensor& uw,const at::Tensor& dw,
                         int64_t heads,int64_t kvheads,int64_t algorithm,bool merged,const c10::optional<at::Tensor>& mask) {
    TORCH_CHECK(x.dim()==3 && x.is_contiguous() && x.scalar_type()==at::kFloat && x.device().is_cpu(),"CPU contiguous float32 decoder only");
    int64_t batch=x.size(0),seq=x.size(1),width=x.size(2),dim=width/heads;
    auto norm=at::mul(nw,at::mul(x,at::rsqrt(at::mean(at::pow(x,2),{-1},true)+1e-5)));
    auto q=at::linear(norm,qw).view({batch,seq,heads,dim}).transpose(1,2);
    auto k=at::linear(norm,kw).view({batch,seq,kvheads,dim}).transpose(1,2);
    auto v=at::linear(norm,vw).view({batch,seq,kvheads,dim}).transpose(1,2);
    auto c=cos.unsqueeze(1),s=sin.unsqueeze(1);
    q=at::add(at::mul(q,c),at::mul(rotate_half(q),s));
    k=at::add(at::mul(k,c),at::mul(rotate_half(k),s));
    int64_t groups=heads/kvheads;
    if(groups>1) {
        k=k.unsqueeze(2).expand({batch,kvheads,groups,seq,dim}).reshape({batch,heads,seq,dim});
        v=v.unsqueeze(2).expand({batch,kvheads,groups,seq,dim}).reshape({batch,heads,seq,dim});
    }
    c10::optional<at::Tensor> causal=mask.has_value() ? c10::optional<at::Tensor>(mask->slice(-1,0,seq)) : c10::nullopt;
    auto a=at::scaled_dot_product_attention(q,k,v,causal,0.0,!causal.has_value() && seq>1);
    a=a.transpose(1,2).contiguous().view({batch,seq,width});
    auto projected=at::linear(a,ow).reshape({batch*seq,width});
    auto residual=x.reshape({batch*seq,width});
    auto out=merged ? rms_mlp_merged(projected,residual,mw,gw,c10::nullopt,uw,c10::nullopt,dw,c10::nullopt,algorithm,0)
                    : rms_mlp(projected,residual,mw,gw,c10::nullopt,uw,c10::nullopt,dw,c10::nullopt,algorithm,0);
    return out.reshape({batch,seq,width});
}
static at::Tensor decoder_meta(const at::Tensor& x,const at::Tensor& nw,
                         const at::Tensor& qw,const at::Tensor& kw,const at::Tensor& vw,const at::Tensor& ow,
                         const at::Tensor& cos,const at::Tensor& sin,const at::Tensor& mw,
                         const at::Tensor& gw,const at::Tensor& uw,const at::Tensor& dw,
                         int64_t heads,int64_t kvheads,int64_t algorithm,bool merged,const c10::optional<at::Tensor>& mask) {return at::empty_like(x);}
static at::Tensor mlp_meta(const at::Tensor& x,const at::Tensor& r,const at::Tensor& w,
                          const at::Tensor& gw,const c10::optional<at::Tensor>& gb,const at::Tensor& uw,const c10::optional<at::Tensor>& ub,
                          const at::Tensor& dw,const c10::optional<at::Tensor>& db,int64_t algorithm,int64_t grain) {
    return at::empty_like(x);
}

static std::tuple<at::Tensor,at::Tensor> rms_meta(const at::Tensor& x,const at::Tensor& r,
                                                const at::Tensor& w,int64_t algorithm,int64_t grain) {
    return {at::empty_like(x),at::empty_like(x)};
}

TORCH_LIBRARY(lossless_retained, m) {
    m.def("plain_norm(Tensor x, Tensor w, float eps) -> Tensor");
    m.def("rope(Tensor x, Tensor inv, Tensor positions, float scaling) -> (Tensor, Tensor)");
    m.def("decoder(Tensor x, Tensor nw, Tensor qw, Tensor kw, Tensor vw, Tensor ow, Tensor cos, Tensor sin, Tensor mw, Tensor gw, Tensor uw, Tensor dw, int heads, int kvheads, int algorithm, bool merged, Tensor? mask) -> Tensor");
    m.def("rms_pre(Tensor x, Tensor residual, Tensor weight, int algorithm, int grain) -> (Tensor, Tensor)");
    m.def("rms_region(Tensor x, Tensor residual, Tensor weight, int algorithm, int grain) -> Tensor");
    m.def("rms_mlp(Tensor x, Tensor residual, Tensor weight, Tensor gw, Tensor? gb, Tensor uw, Tensor? ub, Tensor dw, Tensor? db, int algorithm, int grain) -> Tensor");
    m.def("rms_mlp_merged(Tensor x, Tensor residual, Tensor weight, Tensor gw, Tensor? gb, Tensor uw, Tensor? ub, Tensor dw, Tensor? db, int algorithm, int grain) -> Tensor");
    m.def("rms_mlp_folded(Tensor x, Tensor residual, Tensor weight, Tensor gw, Tensor? gb, Tensor uw, Tensor? ub, Tensor dw, Tensor? db, int algorithm, int grain) -> Tensor");
}
TORCH_LIBRARY_IMPL(lossless_retained, CPU, m) {m.impl("decoder",decoder);m.impl("plain_norm",plain_norm);m.impl("rope",rope);}
TORCH_LIBRARY_IMPL(lossless_retained, Meta, m) {m.impl("decoder",decoder_meta);m.impl("plain_norm",plain_norm_meta);m.impl("rope",rope_meta);}
TORCH_LIBRARY_IMPL(lossless_retained, CPU, m) { m.impl("rms_pre", rms_pre); m.impl("rms_region",rms_region); m.impl("rms_mlp",rms_mlp); m.impl("rms_mlp_merged",rms_mlp_merged); m.impl("rms_mlp_folded",rms_mlp_folded); }
TORCH_LIBRARY_IMPL(lossless_retained, Meta, m) { m.impl("rms_pre", rms_meta); m.impl("rms_region",region_meta); m.impl("rms_mlp",mlp_meta); m.impl("rms_mlp_merged",mlp_meta); m.impl("rms_mlp_folded",mlp_meta); }
