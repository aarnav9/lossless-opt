"""Shared native deployment path, also frozen into benchmark workers."""

import platform
import sys


class NativeOperation:
    @classmethod
    def from_loaded_library(cls, operator, library, cases, *, comparator=None):
        """Use a worker's already compiled library with the public call boundary."""
        operation = cls.__new__(cls)
        operation.operator = operator
        operation.library = library
        operation.cases = {(c["rows"], c["columns"], c["layout"]) for c in cases}
        operation.fallback_reason = None
        operation.manifest = {
            "selected": "reference" if comparator else "proposal",
            "comparator": comparator,
        }
        return operation

    def __init__(self, path, manifest):
        import numpy as np
        from .operators import load_library

        self.manifest = manifest
        self.operator = manifest["adapter"].split(".")[1]
        self.library = None
        self.fallback_reason = None
        self.cases = {(c["rows"], c["columns"], c["layout"]) for c in manifest["cases"]}
        compatible = (
            manifest["platform"] == sys.platform
            and manifest["machine"] == platform.machine()
            and manifest["numpy"] == np.__version__
        )
        binary = manifest.get("binary")
        if compatible and binary and binary in manifest["hashes"]:
            self.library = load_library(path / binary)
        else:
            self.fallback_reason = "artifact environment differs or compiled binary is absent"

    def bind(self, x, residual=None, weight=None):
        """Bind buffers once. Caller preserves their shape, strides and contract.

        Output/scratch ownership belongs to this bound callable. Its return value
        is overwritten by its next call; separate bindings have separate buffers.
        Input mutations must remain within the admitted domain. No concurrent calls.
        """
        import numpy as np
        from . import operators

        if not isinstance(x, np.ndarray) or x.dtype != np.float32 or x.ndim != 2 or not x.size:
            raise ValueError("x must be a nonempty rank-two float32 array")
        if any(s <= 0 or s % 4 for s in x.strides):
            raise ValueError("only positive aligned strides are supported")
        rows, cols = x.shape
        if x.strides == (cols * 4, 4):
            layout = "c"
        elif x.strides == (4, rows * 4):
            layout = "f"
        elif x.strides == (cols * 8, 8):
            layout = "slice2"
        else:
            layout = None
        if self.operator != "copy":
            bound = 10000 if self.operator == "softmax" else 1000
            if not np.isfinite(x).all() or np.max(np.abs(x)) > bound:
                raise ValueError("input lies outside the numerical contract")
        if self.operator == "rmsnorm_residual":
            if (
                not isinstance(residual, np.ndarray)
                or residual.dtype != np.float32
                or residual.shape != x.shape
                or residual.strides != x.strides
            ):
                raise ValueError("residual must match x shape, dtype and strides")
            if (
                not isinstance(weight, np.ndarray)
                or weight.dtype != np.float32
                or weight.shape != (cols,)
                or not weight.flags.c_contiguous
            ):
                raise ValueError("weight must be a contiguous float32 vector matching columns")
            if (
                not np.isfinite(residual).all()
                or np.max(np.abs(residual)) > 1000
                or not np.isfinite(weight).all()
                or np.max(np.abs(weight)) > 2
                or np.shares_memory(x, residual)
                or np.shares_memory(x, weight)
                or np.shares_memory(residual, weight)
            ):
                raise ValueError(
                    "RMSNorm buffers must be finite and nonoverlapping, with abs(residual)<=1000 and abs(weight)<=2"
                )
        else:
            residual = x  # unused by copy and softmax implementations
            weight = np.ones(cols, dtype=np.float32)
        arrays = (x, residual, weight)
        compatible = (
            layout is not None and (rows, cols, layout) in self.cases and self.library is not None
        )
        if not compatible:
            self.fallback_reason = (
                self.fallback_reason or "shape or layout outside validated artifact cases"
            )

            def reference():
                return np.asarray(
                    operators.reference(self.operator, arrays), dtype=np.float32, order="C"
                )

            return reference
        if self.manifest["selected"] == "reference":
            comparator = self.manifest.get("comparator", "native_baseline")
            if comparator == "scipy_softmax_float64" and operators.scipy_softmax is None:
                self.fallback_reason = "declared comparator unavailable; independent reference used"
                return lambda: np.asarray(
                    operators.reference(self.operator, arrays), dtype=np.float32, order="C"
                )
            return operators.bind_comparator(self.operator, arrays, self.library, comparator)
        return operators.bind_native(arrays, self.library)

    def __call__(self, x, residual=None, weight=None):
        return self.bind(x, residual, weight)()
