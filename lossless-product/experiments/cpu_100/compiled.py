"""Adaptive compiled-region hypotheses and reproducible application builds."""

import copy
import shutil
import subprocess
import sys

from common import HERE, read, sha, write
from lossless.applications import patches

BASE_C = r"""
#include <stddef.h>
#include <math.h>
#define RESTRICT @RESTRICT@
void clip(const double *RESTRICT s, const double *RESTRICT b, const double *RESTRICT w,
          double *RESTRICT y, double *RESTRICT z, size_t n, int unit) {
    size_t i=0;
    @CLIPLOOP@
}
void difference(const double *a, const double *b, double *z, size_t n) {
    for(size_t i=0;i<n;i++) z[i]=a[i]-b[i];
}
void norms(const double *a,const double *b,double *out,size_t n) {
    double nn=0,dd=0; @NORMS@
    out[0]=sqrt(nn); out[1]=sqrt(dd);
}
void gemv(const double *a, const double *x, double *y,size_t rows,size_t cols,
          size_t rs,size_t cs) {
    for(size_t i=0;i<rows;i++) {double v=0; for(size_t j=0;j<cols;j++) v+=a[i*rs+j*cs]*x[j]; y[i]=v;}
}
"""

ITERATE_C = r"""
typedef void (*gemv_fn)(int,int,long long,long long,double,const double*,long long,const double*,long long,double,double*,long long);
typedef double (*dot_fn)(long long,const double*,long long,const double*,long long);
int iterate(void *gemv_address,void *dot_address,const double *source,
            double *b0,double *b1,const double *weight,double *clipped,double *scratch,
            const double *pinv,const double *vander,double *coef,double *history,
            long long n,long long order,long long pinv_ld,long long vander_ld,
            int pinv_layout,int vander_layout,int unit,int maximum,double tol,double minimum_norm) {
    gemv_fn gemv=(gemv_fn)gemv_address; dot_fn dot=(dot_fn)dot_address;
    double *old=b0,*next=b1;
    for(int it=0;it<maximum;it++) {
        for(long long j=0;j<n;j++) {
            double v=source[j]<old[j]?source[j]:old[j];
            clipped[j]=v;
            if(!unit)scratch[j]=weight[j]*v;
        }
        gemv(pinv_layout,111,order,n,1.0,pinv,pinv_ld,unit?clipped:scratch,1,0.0,coef,1);
        gemv(vander_layout,111,n,order,1.0,vander,vander_ld,coef,1,0.0,next,1);
        for(long long j=0;j<n;j++)scratch[j]=next[j]-old[j];
        double numerator=sqrt(dot(n,scratch,1,scratch,1));
        double denominator=sqrt(dot(n,old,1,old,1));
        if(denominator<minimum_norm)denominator=minimum_norm;
        double change=numerator/denominator;
        history[it]=change;
        if(change<tol)return it+1;
        double *swap=old;old=next;next=swap;
    }
    return maximum;
}
"""

ITERATE_HELPER = r"""
# Match the loaded NumPy OpenBLAS ILP64 symbols rather than selecting another BLAS.
from threadpoolctl import threadpool_info
_blas_handles=[]
_gemv=_dot=None
for _pool in threadpool_info():
    if _pool.get('internal_api')=='openblas':
        _handle=ctypes.CDLL(_pool['filepath'])
        if hasattr(_handle,'cblas_dgemv64_') and hasattr(_handle,'cblas_ddot64_'):
            _blas_handles.append(_handle)
            _gemv=ctypes.cast(_handle.cblas_dgemv64_,ctypes.c_void_p).value
            _dot=ctypes.cast(_handle.cblas_ddot64_,ctypes.c_void_p).value
            break
_lib.iterate.argtypes=[ctypes.c_void_p]*12+[ctypes.c_longlong]*4+[ctypes.c_int]*4+[ctypes.c_double]*2
_lib.iterate.restype=ctypes.c_int
def can_iterate(a,b):
    return _gemv is not None and a.shape[0]>1 and all(x.dtype==np.float64 and (x.flags.c_contiguous or x.flags.f_contiguous) for x in (a,b))
def run_iterations(source,b0,b1,w,clipped,scratch,pinv,vander,coef,history,unit,maximum,tol,minimum_norm):
    def layout(a):return (101,a.strides[0]//8) if a.flags.c_contiguous else (102,a.strides[1]//8)
    pl,ps=layout(pinv);vl,vs=layout(vander)
    ptrs=[int(a.ctypes.data) for a in (source,b0,b1,w,clipped,scratch,pinv,vander,coef,history)]
    steps=_lib.iterate(_gemv,_dot,*ptrs,source.size,coef.size,ps,vs,pl,vl,unit,maximum,tol,minimum_norm)
    return (b1 if steps%2 else b0),steps-1
"""

HELPER = r"""
import ctypes
from pathlib import Path
import numpy as np
_lib=ctypes.CDLL(str(Path(__file__).with_name('cpu_region@SUFFIX@')))
P=ctypes.POINTER(ctypes.c_double)
_lib.clip.argtypes=[P]*5+[ctypes.c_size_t,ctypes.c_int]
_lib.difference.argtypes=[P]*3+[ctypes.c_size_t]
_lib.norms.argtypes=[P]*3+[ctypes.c_size_t]
_lib.gemv.argtypes=[P]*3+[ctypes.c_size_t]*4
for name in ('clip','difference','norms','gemv'):getattr(_lib,name).restype=None
def bind(s,b0,b1,w,y,z):
    ptr=lambda a:a.ctypes.data_as(P)
    sp,wp,yp,zp=map(ptr,(s,w,y,z))
    bp={id(b0):ptr(b0),id(b1):ptr(b1)}
    nn=np.empty(2,dtype=np.float64); nptr=ptr(nn); n=s.size
    def clip(b,unit):_lib.clip(sp,bp[id(b)],wp,yp,zp,n,unit)
    def difference(a,b):_lib.difference(bp[id(a)],bp[id(b)],zp,n)
    def norms(a,b):
        _lib.norms(bp[id(a)],bp[id(b)],nptr,n)
        return float(nn[0]),float(nn[1])
    return clip,difference,norms
def bind_gemv(a,x,y):
    ap,xp,yp=[v.ctypes.data_as(P) for v in (a,x,y)]
    rows,cols=a.shape;rs,cs=[s//8 for s in a.strides]
    return lambda:_lib.gemv(ap,xp,yp,rows,cols,rs,cs)
"""


def proposal(number, history):
    eligible = [r for r in history if r["decision"]["category"] == "exact"]
    # Once conformance has exposed a missing guard, only guarded incumbents may
    # seed later implementation choices; old timings remain in the ledger.
    parents = (
        [r for r in eligible if r["proposal"]["options"].get("guard_source")]
        if number >= 18
        else eligible
    )
    passing = [r for r in parents if r["decision"]["eligible"]]
    best = max(passing or parents, key=lambda r: r["speedup"]) if parents else None
    base = dict(
        clip=False,
        difference=False,
        norms=0,
        unroll=1,
        restrict=False,
        threshold=0,
        weighted_only=False,
        unit_only=False,
        gemv=False,
        flags=[],
        min_style="compare",
        hoist=number > 1,
        raw_pointers=False,
        lazy=False,
        full_loop=False,
        full_no_bind=False,
        split_unit=False,
        full_maximum=0,
        guard_source=False,
    )
    pointer_results = [r for r in eligible if r["proposal"]["options"].get("raw_pointers")]
    if pointer_results and max(r["speedup"] for r in pointer_results) > max(
        (r["speedup"] for r in eligible if not r["proposal"]["options"].get("raw_pointers")),
        default=0,
    ):
        base["raw_pointers"] = True
    if number > 6 and best and best["proposal"]["options"].get("lazy"):
        base.update(lazy=True, threshold=best["proposal"]["options"]["threshold"])
    changes = [
        (
            "Replace clipping with one bound native pass; preserve NumPy weighting.",
            dict(clip=True, clip_only=True),
        ),
        (
            "Experiment 1 lost heavily on first use: move the same clipping helper import into application setup.",
            dict(clip=True, clip_only=True),
        ),
        (
            "With helper import moved to setup, fuse clipping and weighting preserving double rounding.",
            dict(clip=True),
        ),
        (
            "Experiments 2/3 still lose on batches: remove per-fit ctypes pointer-owner construction from fused clipping.",
            dict(clip=True, raw_pointers=True),
        ),
        (
            "Combine clipping/weighting and difference helpers without changing reductions.",
            dict(clip=True, difference=True),
        ),
        (
            "Earlier helpers lost on small calls: avoid binding entirely below 2048 elements and unroll fusion four lanes.",
            dict(clip=True, unroll=4, threshold=2048, lazy=True),
        ),
        ("Test eight-lane unrolling after four-lane feedback.", dict(clip=True, unroll=8)),
        (
            "Small helpers did not win: compile the entire iteration loop while calling the same loaded NumPy BLAS routines.",
            dict(full_loop=True),
        ),
        (
            "Experiment 8 won overall: remove its redundant small-helper binding before whole-loop dispatch.",
            dict(full_loop=True, full_no_bind=True),
        ),
        (
            "Separate unit and weighted clipping loops inside the compiled iteration to remove an inner branch.",
            dict(full_loop=True, full_no_bind=True, split_unit=True),
        ),
        (
            "Guard whole-loop dispatch above 8192 elements after the long-spectrum regression.",
            dict(full_loop=True, full_no_bind=True, full_maximum=8192),
        ),
        (
            "Test a 4096-element whole-loop cutoff to preserve vectorized library performance on longer inputs.",
            dict(full_loop=True, full_no_bind=True, full_maximum=4096),
        ),
        (
            "Test loop unrolling in the whole compiled iteration with unchanged BLAS reductions.",
            dict(full_loop=True, full_no_bind=True, flags=["-funroll-loops"]),
        ),
        (
            "Conformance found strided-source misreads: guard contiguity/alignment and preserve the original loop fallback.",
            dict(full_loop=True, full_no_bind=True, guard_source=True),
        ),
        (
            "Fuse subtraction and two scalar norm reductions; numerical candidate, exactness independently tested.",
            dict(norms=1, lazy=False, threshold=0),
        ),
        (
            "Use four norm accumulators; explicitly test the changed reduction order.",
            dict(norms=4, lazy=False, threshold=0),
        ),
        (
            "Replace small coefficient GEMV with a native ordered loop; numerical contract remains fixed.",
            dict(gemv=True),
        ),
        (
            "Apply O2 to the best exact discovered helper; test whether O3 was useful.",
            {"adaptive": True, "flags": ["-O2"]},
        ),
        (
            "Disable loop and SLP vectorization in the best exact helper to test the measured mechanism.",
            {"adaptive": True, "flags": ["-fno-vectorize", "-fno-slp-vectorize"]},
        ),
        (
            "Neither compiler ablation passed the regression gate: combine the guarded whole loop with separate unit/weighted clipping passes from experiment 10.",
            {"adaptive": True, "split_unit": True},
        ),
    ]
    hypothesis, change = changes[number - 1]
    if change.pop("adaptive", False) and best:
        base.update(copy.deepcopy(best["proposal"]["options"]))
    if change.pop("compose", False):
        diffs = [
            r
            for r in eligible
            if r["proposal"]["options"].get("difference") and r["speedup"] > 1.02
        ]
        base["difference"] = bool(diffs)
        # A new guarded composition is selected on development only.
        base["threshold"] = 4096 if best and best["case_speedups"].get("small", 1) < 1 else 512
    base.update(change)
    if number >= 14:
        base["guard_source"] = True
    return dict(
        hypothesis=hypothesis,
        options=base,
        informed_by=[
            dict(experiment=r["number"], speedup=r["speedup"], category=r["decision"]["category"])
            for r in history[-2:]
        ],
        parent_experiment=best["number"] if best else None,
        rationale="Uses prior discovery outcomes only; final inputs remain closed.",
    )


def source(options):
    minimum = {
        "compare": "(s[k] <= b[k] ? s[k] : b[k])",
        "ternary": "(s[k] < b[k] ? s[k] : b[k])",
        "fmin": "fmin(s[k],b[k])",
    }[options["min_style"]]
    # Source and clipped output can alias for use_original=False, so restrict must remain off.
    statement = f"double v={minimum}; y[k]=v; if(!unit) z[k]=w[k]*v;"
    width = options["unroll"]
    loop = f"for(;i+{width}<=n;i+={width}) {{"
    loop += "".join("{size_t k=i+" + str(j) + ";" + statement + "}" for j in range(width)) + "}"
    loop += "for(;i<n;i++){size_t k=i;" + statement + "}"
    if options["norms"] == 4:
        norm = "double ns[4]={0},ds[4]={0}; for(size_t i=0;i<n;i++){double d=a[i]-b[i]; ns[i%4]+=d*d; ds[i%4]+=b[i]*b[i];} for(int i=0;i<4;i++){nn+=ns[i];dd+=ds[i];}"
    else:
        norm = "for(size_t i=0;i<n;i++){double d=a[i]-b[i];nn+=d*d;dd+=b[i]*b[i];}"
    return BASE_C.replace("@RESTRICT@", "").replace("@CLIPLOOP@", loop).replace("@NORMS@", norm)


def build(baseline, folder, plan):
    project = folder / "project"
    shutil.copytree(baseline, project, ignore=shutil.ignore_patterns("__pycache__"))
    options = plan["options"]
    iteration = ITERATE_C
    if options.get("split_unit"):
        original = """        for(long long j=0;j<n;j++) {
            double v=source[j]<old[j]?source[j]:old[j];
            clipped[j]=v;
            if(!unit)scratch[j]=weight[j]*v;
        }"""
        replacement = """        if(unit) {
            for(long long j=0;j<n;j++)clipped[j]=source[j]<old[j]?source[j]:old[j];
        } else {
            for(long long j=0;j<n;j++){double v=source[j]<old[j]?source[j]:old[j];clipped[j]=v;scratch[j]=weight[j]*v;}
        }"""
        assert original in iteration
        iteration = iteration.replace(original, replacement)
    c = project / "cpu_region.c"
    c.write_text(source(options) + (iteration if options.get("full_loop") else ""))
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    binary = project / ("cpu_region" + suffix)
    flags = [
        "-O3",
        "-std=c11",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-fPIC",
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "-Rpass=loop-vectorize",
        "-Rpass-missed=loop-vectorize",
        "-Rpass-analysis=loop-vectorize",
    ] + options["flags"]
    command = ["clang", *flags, str(c), "-o", str(binary), "-lm"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    write(
        folder / "build.json",
        dict(
            command=command,
            returncode=result.returncode,
            diagnostics=result.stderr,
            source_sha256=sha(c),
            binary_sha256=sha(binary) if binary.exists() else None,
        ),
    )
    if result.returncode:
        raise ValueError(result.stderr)
    helper = HELPER.replace("@SUFFIX@", suffix)
    if options.get("raw_pointers"):
        helper = helper.replace("P=ctypes.POINTER(ctypes.c_double)", "P=ctypes.c_void_p")
        helper = helper.replace(
            "ptr=lambda a:a.ctypes.data_as(P)", "ptr=lambda a:int(a.ctypes.data)"
        )
    if options.get("full_loop"):
        helper += ITERATE_HELPER
    (project / "cpu_helper.py").write_text(helper)
    body = read(HERE.parent / "application_intake/cpu-control-candidate.json")["edits"][0]["body"]
    condition = f"baseline.size >= {options['threshold']}"
    if options.get("guard_source"):
        condition += " and clip_source.flags.c_contiguous and clip_source.flags.aligned"
    if options["weighted_only"]:
        condition += " and not unit_weights"
    if options["unit_only"]:
        condition += " and unit_weights"
    setup = """    from cpu_helper import bind
    clip_native, difference_native, norms_native = bind(clip_source, baseline, baseline_buffer, sqrt_w, clipped, scratch)
    use_native = CONDITION
""".replace("CONDITION", condition)
    if options.get("lazy"):
        setup = """    from cpu_helper import bind
    use_native = CONDITION
    if use_native:
        clip_native, difference_native, norms_native = bind(clip_source, baseline, baseline_buffer, sqrt_w, clipped, scratch)
""".replace("CONDITION", condition)
    if options.get("full_no_bind") and options.get("full_loop"):
        setup = ""
    body = body.replace(
        "    vander_dot = vandermonde.dot\n", "    vander_dot = vandermonde.dot\n" + setup
    )
    if options["clip"]:
        original = """        minimum(clip_source, baseline, clipped)
        if unit_weights:
            pinv_dot(clipped, coef)
        else:
            multiply(sqrt_w, clipped, scratch)
            pinv_dot(scratch, coef)"""
        replacement = """        if use_native:
            clip_native(baseline, UNIT)
        else:
            minimum(clip_source, baseline, clipped)
        if unit_weights:
            pinv_dot(clipped, coef)
        else:
            if MULTIPLY:
                multiply(sqrt_w, clipped, scratch)
            pinv_dot(scratch, coef)""".replace(
            "UNIT", "True" if options.get("clip_only") else "unit_weights"
        ).replace("MULTIPLY", "True" if options.get("clip_only") else "not use_native")
        assert original in body
        body = body.replace(original, replacement)
    if options["difference"]:
        body = body.replace(
            "        subtract(baseline, baseline_old, scratch)",
            "        if use_native:\n            difference_native(baseline, baseline_old)\n        else:\n            subtract(baseline, baseline_old, scratch)",
        )
    if options["norms"]:
        body = body.replace(
            """        subtract(baseline, baseline_old, scratch)
        numerator = sqrt(scratch_dot(scratch))
        denominator = sqrt(baseline_old.dot(baseline_old))""",
            """        numerator, denominator = norms_native(baseline, baseline_old)""",
        )
    if options["gemv"]:
        body = body.replace(
            "    use_native = " + condition,
            "    use_native = "
            + condition
            + "\n    from cpu_helper import bind_gemv\n    coefficient_native = bind_gemv(pseudo_inverse, scratch, coef)\n    coefficient_unit_native = bind_gemv(pseudo_inverse, clipped, coef)",
        )
        body = body.replace(
            "            pinv_dot(clipped, coef)", "            coefficient_unit_native()"
        )
        body = body.replace(
            "            pinv_dot(scratch, coef)", "            coefficient_native()"
        )
    if options.get("full_loop"):
        # Rewrite only the scratch-path loop; fallback retains the original function body.
        import textwrap

        start = body.index("    for i in range(max_iter):")
        end = body.index("\nelse:\n    for i in range(max_iter):", start)
        old = body[start:end]
        new = """    from cpu_helper import can_iterate, run_iterations
    if can_iterate(pseudo_inverse, vandermonde):
        baseline, i = run_iterations(clip_source, baseline, baseline_buffer, sqrt_w,
            clipped, scratch, pseudo_inverse, vandermonde, coef, tol_history,
            unit_weights, max_iter, tol, minimum_norm)
    else:
""" + textwrap.indent(old, "    ")
        if options.get("full_maximum"):
            new = new.replace(
                "if can_iterate(pseudo_inverse, vandermonde):",
                f"if baseline.size <= {options['full_maximum']} and can_iterate(pseudo_inverse, vandermonde):",
            )
        if options.get("guard_source"):
            new = new.replace(
                "and can_iterate(pseudo_inverse, vandermonde):",
                "and can_iterate(pseudo_inverse, vandermonde) and clip_source.flags.c_contiguous and clip_source.flags.aligned:",
            )
            new = new.replace(
                "if can_iterate(pseudo_inverse, vandermonde):",
                "if can_iterate(pseudo_inverse, vandermonde) and clip_source.flags.c_contiguous and clip_source.flags.aligned:",
            )
        body = body[:start] + new + body[end:]
    proposal = dict(
        schema_version=1,
        hypothesis=plan["hypothesis"],
        edits=[dict(path="pybaselines/polynomial.py", symbol="_Polynomial.modpoly", body=body)],
    )
    diff, _ = patches.apply(
        project,
        project,
        proposal,
        [dict(path="pybaselines/polynomial.py", symbol="_Polynomial.modpoly")],
    )
    if options["hoist"]:
        polynomial = project / "pybaselines/polynomial.py"
        content = "".join(
            line
            for line in polynomial.read_text().splitlines(keepends=True)
            if line.strip() != "from cpu_helper import bind"
        )
        content = content.replace(
            "import numpy as np\n", "import numpy as np\nfrom cpu_helper import bind\n", 1
        )
        polynomial.write_text(content)
    (folder / "candidate.patch").write_text(diff)
    write(folder / "proposal.json", proposal)
    return dict(project=project, config={})
