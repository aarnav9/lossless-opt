#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <arm_neon.h>

static inline float32x4_t load4(const float *p, size_t s) {
    if (s == 1) return vld1q_f32(p);
    return (float32x4_t){p[0],p[s],p[2*s],p[3*s]};
}

static inline void fill_row(float *o, size_t n, float value) {
    float32x4_t v = vdupq_n_f32(value);
    size_t j = 0;
    for (; n-j >= 16; j += 16) {
        vst1q_f32(o+j,v); vst1q_f32(o+j+4,v);
        vst1q_f32(o+j+8,v); vst1q_f32(o+j+12,v);
    }
    for (; n-j >= 4; j += 4) vst1q_f32(o+j,v);
    for (; j < n; ++j) o[j] = value;
}

static inline void scale_row(float *o, size_t n, float scale) {
    float32x4_t v = vdupq_n_f32(scale);
    size_t j = 0;
    for (; n-j >= 16; j += 16) {
        vst1q_f32(o+j,vmulq_f32(vld1q_f32(o+j),v));
        vst1q_f32(o+j+4,vmulq_f32(vld1q_f32(o+j+4),v));
        vst1q_f32(o+j+8,vmulq_f32(vld1q_f32(o+j+8),v));
        vst1q_f32(o+j+12,vmulq_f32(vld1q_f32(o+j+12),v));
    }
    for (; n-j >= 4; j += 4)
        vst1q_f32(o+j,vmulq_f32(vld1q_f32(o+j),v));
    for (; j < n; ++j) o[j] *= scale;
}

static inline double sum_row(const float *o, size_t n) {
    float64x2_t a = vdupq_n_f64(0.0), b = a, c = a, d = a;
    size_t j = 0;
    for (; n-j >= 8; j += 8) {
        float32x4_t u = vld1q_f32(o+j), v = vld1q_f32(o+j+4);
        a = vaddq_f64(a,vcvt_f64_f32(vget_low_f32(u)));
        b = vaddq_f64(b,vcvt_f64_f32(vget_high_f32(u)));
        c = vaddq_f64(c,vcvt_f64_f32(vget_low_f32(v)));
        d = vaddq_f64(d,vcvt_f64_f32(vget_high_f32(v)));
    }
    double sum = vaddvq_f64(vaddq_f64(vaddq_f64(a,b),vaddq_f64(c,d)));
    for (; j < n; ++j) sum += (double)o[j];
    return sum;
}

static inline void extrema(const float *p, size_t n, size_t s, float *lo, float *hi) {
    float32x4_t h0 = vdupq_n_f32(p[0]), h1 = h0, h2 = h0, h3 = h0;
    float32x4_t l0 = h0, l1 = h0, l2 = h0, l3 = h0;
    size_t j = 0;
    for (; n-j >= 16; j += 16) {
        float32x4_t a = load4(p+j*s,s), b = load4(p+(j+4)*s,s);
        float32x4_t c = load4(p+(j+8)*s,s), d = load4(p+(j+12)*s,s);
        h0 = vmaxq_f32(h0,a); l0 = vminq_f32(l0,a);
        h1 = vmaxq_f32(h1,b); l1 = vminq_f32(l1,b);
        h2 = vmaxq_f32(h2,c); l2 = vminq_f32(l2,c);
        h3 = vmaxq_f32(h3,d); l3 = vminq_f32(l3,d);
    }
    for (; n-j >= 4; j += 4) {
        float32x4_t v = load4(p+j*s,s);
        h0 = vmaxq_f32(h0,v); l0 = vminq_f32(l0,v);
    }
    float high = vmaxvq_f32(vmaxq_f32(vmaxq_f32(h0,h1),vmaxq_f32(h2,h3)));
    float low = vminvq_f32(vminq_f32(vminq_f32(l0,l1),vminq_f32(l2,l3)));
    for (; j < n; ++j) {
        float v = p[j*s];
        if (v > high) high = v;
        if (v < low) low = v;
    }
    *lo = low; *hi = high;
}

static inline int count_two(const float *p, size_t n, size_t s,
                            float lo, float hi, size_t *nh) {
    if (p[0] != lo && p[0] != hi) return 0;
    float32x4_t lv = vdupq_n_f32(lo), hv = vdupq_n_f32(hi);
    uint64x2_t count = vdupq_n_u64(0);
    size_t j = 0;
    for (; n-j >= 8; j += 8) {
        float32x4_t a = load4(p+j*s,s), b = load4(p+(j+4)*s,s);
        uint32x4_t ha = vceqq_f32(a,hv), hb = vceqq_f32(b,hv);
        uint32x4_t va = vorrq_u32(ha,vceqq_f32(a,lv));
        uint32x4_t vb = vorrq_u32(hb,vceqq_f32(b,lv));
        if (vminvq_u32(vandq_u32(va,vb)) != UINT32_MAX) return 0;
        count = vpadalq_u32(count,vaddq_u32(vshrq_n_u32(ha,31),vshrq_n_u32(hb,31)));
    }
    size_t total = (size_t)vaddvq_u64(count);
    for (; j < n; ++j) {
        float v = p[j*s];
        if (v == hi) ++total;
        else if (v != lo) return 0;
    }
    *nh = total;
    return 1;
}

static inline void write_two(const float *p, float *o, size_t n, size_t s,
                             float hi, float high_output, float low_output) {
    float32x4_t hv = vdupq_n_f32(hi);
    float32x4_t ho = vdupq_n_f32(high_output), lo = vdupq_n_f32(low_output);
    size_t j = 0;
    for (; n-j >= 4; j += 4) {
        uint32x4_t mask = vceqq_f32(load4(p+j*s,s),hv);
        vst1q_f32(o+j,vbslq_f32(mask,ho,lo));
    }
    for (; j < n; ++j) o[j] = p[j*s] == hi ? high_output : low_output;
}

static inline void exponentials(const float *p, float *o, size_t n, size_t s, float hi) {
    size_t j = 0;
    if (s == 1) {
        for (; n-j >= 4; j += 4) {
            o[j] = expf(p[j]-hi); o[j+1] = expf(p[j+1]-hi);
            o[j+2] = expf(p[j+2]-hi); o[j+3] = expf(p[j+3]-hi);
        }
    } else {
        for (; n-j >= 4; j += 4) {
            o[j] = expf(p[j*s]-hi); o[j+1] = expf(p[(j+1)*s]-hi);
            o[j+2] = expf(p[(j+2)*s]-hi); o[j+3] = expf(p[(j+3)*s]-hi);
        }
    }
    for (; j < n; ++j) o[j] = expf(p[j*s]-hi);
}

static void finish_row(const float *p, float *o, size_t n, size_t s,
                       float lo, float hi, float uniform) {
    if (lo == hi) { fill_row(o,n,uniform); return; }
    size_t nh;
    if (count_two(p,n,s,lo,hi,&nh)) {
        float e = expf(lo-hi);
        double sum = (double)nh + (double)(n-nh)*(double)e;
        float scale = (float)(1.0/sum);
        write_two(p,o,n,s,hi,scale,e*scale);
        return;
    }
    exponentials(p,o,n,s,hi);
    scale_row(o,n,(float)(1.0/sum_row(o,n)));
}

/* Input rows are adjacent; output rows remain C-contiguous. */
static void pack_four_rows(const float *p, float *o, size_t n, size_t cs,
                           float lows[4], float highs[4]) {
    float *o0 = o, *o1 = o+n, *o2 = o+2*n, *o3 = o+3*n;
    float32x4_t h0 = vdupq_n_f32(-INFINITY), h1 = h0, h2 = h0, h3 = h0;
    float32x4_t l0 = vdupq_n_f32(INFINITY), l1 = l0, l2 = l0, l3 = l0;
    size_t j = 0;
    for (; n-j >= 4; j += 4) {
        float32x4_t a = vld1q_f32(p+j*cs);
        float32x4_t b = vld1q_f32(p+(j+1)*cs);
        float32x4_t c = vld1q_f32(p+(j+2)*cs);
        float32x4_t d = vld1q_f32(p+(j+3)*cs);
        h0 = vmaxq_f32(h0,a); l0 = vminq_f32(l0,a);
        h1 = vmaxq_f32(h1,b); l1 = vminq_f32(l1,b);
        h2 = vmaxq_f32(h2,c); l2 = vminq_f32(l2,c);
        h3 = vmaxq_f32(h3,d); l3 = vminq_f32(l3,d);
        float32x4x2_t ab = vtrnq_f32(a,b), cd = vtrnq_f32(c,d);
        vst1q_f32(o0+j,vcombine_f32(vget_low_f32(ab.val[0]),vget_low_f32(cd.val[0])));
        vst1q_f32(o1+j,vcombine_f32(vget_low_f32(ab.val[1]),vget_low_f32(cd.val[1])));
        vst1q_f32(o2+j,vcombine_f32(vget_high_f32(ab.val[0]),vget_high_f32(cd.val[0])));
        vst1q_f32(o3+j,vcombine_f32(vget_high_f32(ab.val[1]),vget_high_f32(cd.val[1])));
    }
    for (; j < n; ++j) {
        float32x4_t v = vld1q_f32(p+j*cs);
        h0 = vmaxq_f32(h0,v); l0 = vminq_f32(l0,v);
        o0[j] = vgetq_lane_f32(v,0); o1[j] = vgetq_lane_f32(v,1);
        o2[j] = vgetq_lane_f32(v,2); o3[j] = vgetq_lane_f32(v,3);
    }
    vst1q_f32(highs,vmaxq_f32(vmaxq_f32(h0,h1),vmaxq_f32(h2,h3)));
    vst1q_f32(lows,vminq_f32(vminq_f32(l0,l1),vminq_f32(l2,l3)));
}

void kernel(const float *restrict x, const float *restrict r, const float *restrict w,
            float *restrict out, float *restrict inv, float *restrict px, float *restrict pr,
            size_t rows, size_t cols, size_t rs, size_t cs) {
    (void)r; (void)w; (void)inv; (void)px; (void)pr;
    if (rows == 0 || cols == 0) return;
    if (cols == 1) {
        for (size_t i = 0; i < rows; ++i) out[i] = 1.0f;
        return;
    }
    float uniform = (float)(1.0/(double)cols);
    size_t i = 0;
    if (rs == 1) {
        for (; rows-i >= 4; i += 4) {
            float lows[4], highs[4];
            float *o = out+i*cols;
            pack_four_rows(x+i,o,cols,cs,lows,highs);
            for (size_t q = 0; q < 4; ++q)
                finish_row(o+q*cols,o+q*cols,cols,1,lows[q],highs[q],uniform);
        }
    }
    for (; i < rows; ++i) {
        const float *p = x+i*rs;
        float *o = out+i*cols;
        float lo, hi;
        extrema(p,cols,cs,&lo,&hi);
        finish_row(p,o,cols,cs,lo,hi,uniform);
    }
}
