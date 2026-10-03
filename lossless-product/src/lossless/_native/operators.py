"""Trusted numerical contracts, input generation, and reference baselines."""

import ctypes
import hashlib

import numpy as np

try:
    from scipy.special import softmax as scipy_softmax
except ImportError:
    scipy_softmax = None

GUARD = np.float32(-12345.5)
COMPARATORS = {
    "copy": ("numpy_copy", "native_baseline"),
    "softmax": ("numpy_buffered", "scipy_softmax_float64", "native_baseline"),
    "rmsnorm_residual": ("numpy_buffered", "native_baseline"),
}


def validate_comparator(operator, name):
    if name not in COMPARATORS[operator]:
        raise ValueError(f"unsupported deployment comparator {name!r} for {operator}")
    if name == "scipy_softmax_float64" and scipy_softmax is None:
        raise ValueError("scipy_softmax_float64 comparator requires the numerical extra")
    return name


ABI = "const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs"
CONTRACTS = {
    "copy": {
        "version": 1,
        "relation": "bitwise_equal",
        "formula": "out[row,col] = x[row,col]",
        "input": "float32 matrix with supported positive strides; all bit patterns allowed",
        "output": "C-contiguous independent float32 copy; inputs unchanged",
        "distributions": ["gaussian", "uniform", "constant", "special_bits"],
    },
    "softmax": {
        "version": 1,
        "formula": "exp(x-row_max) / sum(exp(x-row_max))",
        "input": "finite float32, abs(x)<=10000, positive nonoverlapping strides",
        "output": "C-contiguous float32, nonnegative, row sum within 2e-6 of 1",
        "atol": 2e-7,
        "rtol": 4e-5,
        "distributions": ["gaussian", "uniform", "extreme", "constant", "single_peak", "shifted"],
    },
    "rmsnorm_residual": {
        "version": 1,
        "formula": "x / sqrt(mean(x*x)+1e-5) * w + residual",
        "input": "finite float32, abs(x)<=1000, abs(residual)<=1000, abs(weight)<=2, nonoverlapping buffers",
        "output": "C-contiguous float32",
        "atol": 1e-5,
        "rtol": 5e-5,
        "distributions": ["gaussian", "uniform", "extreme", "constant", "single_peak", "tiny"],
    },
}


def input_arrays(operator, case, seed, distribution="gaussian"):
    rng = np.random.default_rng(seed)
    rows, cols = case["rows"], case["columns"]
    layout = case["layout"]

    def array():
        if layout == "c":
            return rng.standard_normal((rows, cols), dtype=np.float32)
        if layout == "f":
            return rng.standard_normal((cols, rows), dtype=np.float32).T
        if layout == "slice2":
            return rng.standard_normal((rows, cols * 2), dtype=np.float32)[:, ::2]
        raise ValueError("unknown layout")

    x, r = array(), array()
    w = rng.uniform(-2, 2, cols).astype(np.float32)
    if distribution == "uniform":
        x[:] = rng.uniform(-8, 8, x.shape)
    elif distribution == "extreme":
        x[:] = rng.uniform(
            -80 if operator == "softmax" else -1000, 80 if operator == "softmax" else 1000, x.shape
        )
    elif distribution == "constant":
        x[:] = np.arange(rows, dtype=np.float32)[:, None] % 7 - 3
    elif distribution == "single_peak":
        x[:] = -20
        x[np.arange(rows), np.arange(rows) % cols] = 20
    elif distribution == "shifted":
        x[:] = np.clip(x + np.float32(9990), -10000, 10000)
    elif distribution == "tiny":
        x *= np.float32(1e-7)
    if distribution == "special_bits":
        bits = np.resize(
            np.array(
                [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC01234, 0x7F800001, 0x3F800000],
                dtype=np.uint32,
            ),
            x.shape,
        )
        x.view(np.uint32)[:] = bits
    return x, r, w


def reference(operator, arrays):
    x, r, w = arrays
    if operator == "copy":
        return np.array(x, copy=True, order="C")
    d = x.astype(np.float64)
    if operator == "softmax":
        e = np.exp(d - np.max(d, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)
    return d / np.sqrt(np.mean(d * d, axis=1, keepdims=True) + 1e-5) * w.astype(
        np.float64
    ) + r.astype(np.float64)


def digest(arrays):
    result = hashlib.sha256()
    for array in arrays:
        result.update(array.tobytes())
    return result.hexdigest()


def load_library(path):
    lib = ctypes.CDLL(str(path))
    ptr = ctypes.POINTER(ctypes.c_float)
    lib.kernel.argtypes = [ptr] * 7 + [ctypes.c_size_t] * 4
    lib.kernel.restype = None
    return lib


def executors(operator, case, arrays, baseline, candidate):
    x, r, w = arrays
    rows, cols = x.shape
    stores = [np.full(rows * cols + 32, GUARD, dtype=np.float32) for _ in range(3)]
    inverse_store = np.full(rows + 32, GUARD, dtype=np.float32)
    stores.append(inverse_store)
    out, px, pr = [v[16:-16].reshape(rows, cols) for v in stores[:3]]
    inv = inverse_store[16:-16]
    ptr = ctypes.POINTER(ctypes.c_float)
    pointers = [a.ctypes.data_as(ptr) for a in (x, r, w, out, inv, px, pr)]
    rs, cs = (s // 4 for s in x.strides)

    def native(lib):
        def call():
            lib.kernel(*pointers, rows, cols, rs, cs)
            return out

        return call

    funcs = {"native_baseline": native(baseline), "proposal": native(candidate)}
    if operator == "copy":
        funcs["numpy_copy"] = lambda: np.copyto(out, x) or out
    elif operator == "softmax":
        maxima = np.empty((rows, 1), dtype=np.float32)
        sums = np.empty((rows, 1), dtype=np.float64)

        def numpy_call():
            np.max(x, axis=1, keepdims=True, out=maxima)
            np.subtract(x, maxima, out=out)
            np.exp(out, out=out)
            np.sum(out, axis=1, keepdims=True, dtype=np.float64, out=sums)
            np.divide(out, sums, out=out, casting="unsafe")
            return out

        funcs["numpy_buffered"] = numpy_call
        # Float32 SciPy accumulation can exceed this contract's row-sum bound
        # on Fortran inputs. Include both precision/layout conversions in timing.
        if scipy_softmax is not None:
            funcs["scipy_softmax_float64"] = lambda: np.ascontiguousarray(
                scipy_softmax(x.astype(np.float64), axis=1), dtype=np.float32
            )
    else:
        inverse64 = np.empty(rows, dtype=np.float64)

        def numpy_call():
            np.multiply(x, x, out=px)
            np.sum(px, axis=1, dtype=np.float64, out=inverse64)
            np.divide(inverse64, cols, out=inverse64)
            np.add(inverse64, 1e-5, out=inverse64)
            np.sqrt(inverse64, out=inverse64)
            np.reciprocal(inverse64, out=inverse64)
            np.copyto(inv, inverse64, casting="unsafe")
            np.multiply(x, inv[:, None], out=out)
            np.multiply(out, w, out=out)
            np.add(out, r, out=out)
            return out

        funcs["numpy_buffered"] = numpy_call

    def guards():
        return all(bool(np.all(s[:16] == GUARD) and np.all(s[-16:] == GUARD)) for s in stores)

    def poison():
        out.fill(np.nan)

    return funcs, guards, poison


def check(operator, actual, expected):
    if operator == "copy":
        return {
            "passed": actual.dtype == expected.dtype
            and actual.shape == expected.shape
            and actual.flags.c_contiguous
            and actual.tobytes() == expected.tobytes(),
            "relation": "bitwise_equal",
        }
    contract = CONTRACTS[operator]
    finite = bool(np.isfinite(actual).all())
    error = np.abs(actual.astype(np.float64) - expected)
    bound = contract["atol"] + contract["rtol"] * np.abs(expected)
    passed = (
        finite
        and actual.dtype == np.float32
        and actual.flags.c_contiguous
        and actual.shape == expected.shape
        and bool(np.all(error <= bound))
    )
    row_error = None
    if operator == "softmax":
        row_error = (
            float(np.max(np.abs(np.sum(actual, axis=1, dtype=np.float64) - 1))) if finite else None
        )
        passed = passed and bool(np.all(actual >= 0)) and row_error <= 2e-6
    return {
        "passed": bool(passed),
        "max_error_over_tolerance": float(np.max(error / bound)) if finite else None,
        "max_row_sum_error": row_error,
    }


def baseline_source(operator):
    # The adapter oracle is independent from compiled candidate code.
    if operator == "copy":
        return (
            "#include <stddef.h>\n#include <string.h>\nvoid kernel("
            + ABI
            + "){for(size_t i=0;i<rows;++i)for(size_t j=0;j<cols;++j)memcpy(out+i*cols+j,x+i*rs+j*cs,sizeof(float));}\n"
        )
    if operator == "softmax":
        body = """
        for(size_t row=0;row<rows;++row){
          float maximum=-INFINITY;for(size_t j=0;j<cols;++j)if(x[row*rs+j*cs]>maximum)maximum=x[row*rs+j*cs];
          double sums[8]={0};size_t j=0;
          for(;j+8<=cols;j+=8)for(size_t k=0;k<8;++k){float e=expf(x[row*rs+(j+k)*cs]-maximum);out[row*cols+j+k]=e;sums[k]+=(double)e;}
          for(;j<cols;++j){float e=expf(x[row*rs+j*cs]-maximum);out[row*cols+j]=e;sums[0]+=(double)e;}
          double sum=0;for(size_t k=0;k<8;++k)sum+=sums[k];float scale=(float)(1.0/sum);
          for(j=0;j<cols;++j)out[row*cols+j]*=scale;
        }"""
    else:
        body = """
        for(size_t row=0;row<rows;++row){double sums[8]={0};size_t j=0;
          for(;j+8<=cols;j+=8)for(size_t k=0;k<8;++k){float v=x[row*rs+(j+k)*cs];float q=v*v;sums[k]+=(double)q;}
          for(;j<cols;++j){float v=x[row*rs+j*cs];float q=v*v;sums[0]+=(double)q;}
          double sum=0;for(size_t k=0;k<8;++k)sum+=sums[k];float scale=(float)(1.0/sqrt(sum/(double)cols+1e-5));
          for(j=0;j<cols;++j){float a=x[row*rs+j*cs]*scale;float b=a*w[j];out[row*cols+j]=b+r[row*rs+j*cs];}
        }"""
    return "#include <stddef.h>\n#include <math.h>\nvoid kernel(" + ABI + "){" + body + "}\n"
