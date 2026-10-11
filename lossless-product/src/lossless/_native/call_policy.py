"""Opt-in CPU-100 ordinary-call candidate; no native arithmetic is changed."""

import ctypes
from pathlib import Path
import tempfile

import numpy as np


class CallPolicy:
    def __init__(self):
        from ..applications.native import build

        self.folder = tempfile.TemporaryDirectory(prefix="lossless-call-policy-")
        root = Path(self.folder.name)
        source = """#include <stddef.h>
int valid(const float *x,size_t rows,size_t cols,size_t rs,size_t cs,float bound) {
 for(size_t i=0;i<rows;i++)for(size_t j=0;j<cols;j++){
  float v=x[i*rs+j*cs];if(!(v>=-bound && v<=bound))return 0;
 }return 1;
}
"""
        scope = [
            dict(
                name="domain",
                functions=[
                    dict(
                        name="valid",
                        returns="int",
                        arguments=["pointer"] + ["size_t"] * 4 + ["float"],
                    )
                ],
            )
        ]
        try:
            receipt = build(
                root,
                [dict(name="domain", source=source)],
                scope,
                timeout=30,
                output=root / "build.json",
            )
            binary = next(p for p in receipt["files"] if p.endswith((".so", ".dylib")))
            self.library = ctypes.CDLL(str(root / binary))
            self.valid = self.library.valid
            self.valid.argtypes = [ctypes.c_void_p] + [ctypes.c_size_t] * 4 + [ctypes.c_float]
            self.valid.restype = ctypes.c_int
            self.build_receipt = receipt
            # The loaded library handle owns the mapping. There is no persistent
            # generated binary/cache directory to leak from ordinary wrappers.
            self.folder.cleanup()
        except BaseException:
            self.folder.cleanup()
            raise
        self.ones = None

    def domain_valid(self, x, bound):
        if x.size > 8192 or not x.flags.aligned:
            return bool(((x >= -bound) & (x <= bound)).all())
        shape = (1, x.size) if x.ndim == 1 else x.shape
        strides = (0, x.strides[0] // 4) if x.ndim == 1 else tuple(s // 4 for s in x.strides)
        return bool(self.valid(int(x.ctypes.data), *shape, *strides, bound))

    def weight(self, columns):
        if self.ones is not None and self.ones.size == columns:
            return self.ones
        value = np.ones(columns, dtype=np.float32)
        value.flags.writeable = False
        if columns <= 65536:
            self.ones = value
        return value
