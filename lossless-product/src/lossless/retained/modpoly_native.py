"""Whole-loop C binding to the *loaded NumPy* ILP64 BLAS; otherwise decline."""

import ctypes
from pathlib import Path
import numpy as np

_handles = []
_gemv = _dot = None


def _locate():
    global _gemv, _dot
    from threadpoolctl import threadpool_info

    root = Path(np.__file__).resolve().parent
    for pool in threadpool_info():
        path = Path(pool.get("filepath", "")).resolve()
        if pool.get("internal_api") != "openblas" or not (
            path.is_relative_to(root) or path.is_relative_to(root.with_name("numpy.libs"))
        ):
            continue
        handle = ctypes.CDLL(str(path))
        if hasattr(handle, "cblas_dgemv64_") and hasattr(handle, "cblas_ddot64_"):
            _handles.append(handle)
            _gemv = ctypes.cast(handle.cblas_dgemv64_, ctypes.c_void_p).value
            _dot = ctypes.cast(handle.cblas_ddot64_, ctypes.c_void_p).value
            return


try:
    _locate()
except (ImportError, OSError):
    pass


def can_iterate(a, b):
    return (
        _gemv is not None
        and a.shape[0] > 1
        and all(
            x.dtype == np.float64
            and x.flags.aligned
            and (x.flags.c_contiguous or x.flags.f_contiguous)
            for x in (a, b)
        )
    )


def run_iterations(
    iterate,
    source,
    b0,
    b1,
    weight,
    clipped,
    scratch,
    pinv,
    vander,
    coef,
    history,
    unit,
    maximum,
    tol,
    minimum_norm,
):
    # This internal ABI is reached only by the frozen guarded source proposal.
    def layout(a):
        return (101, a.strides[0] // 8) if a.flags.c_contiguous else (102, a.strides[1] // 8)

    pl, ps = layout(pinv)
    vl, vs = layout(vander)
    owners = (source, b0, b1, weight, clipped, scratch, pinv, vander, coef, history)
    steps = iterate(
        _gemv,
        _dot,
        *(int(a.ctypes.data) for a in owners),
        source.size,
        coef.size,
        ps,
        vs,
        pl,
        vl,
        unit,
        maximum,
        tol,
        minimum_norm,
    )
    return (b1 if steps % 2 else b0), steps - 1
