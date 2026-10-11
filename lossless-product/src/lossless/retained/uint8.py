"""Exact pointwise integer fusion/LUT candidates from the solver pipeline study."""

import ctypes
from pathlib import Path
import tempfile
import numpy as np


class Uint8Pipeline:
    """Compile a finite byte-operation sequence; no input-derived answer cache.

    ``operations`` are (name, integer) pairs: min, max, add (modulo 256), xor,
    right_shift (0..7), or invert with argument 0. LUT mode exhaustively derives
    all 256 scalar results from these operations. Outputs own independent storage.
    """

    def __init__(self, operations, *, strategy="native"):
        from ..applications.native import build

        self.operations = tuple(tuple(v) for v in operations)
        if not 1 <= len(self.operations) <= 64 or strategy not in {"native", "lookup"}:
            raise ValueError("require 1..64 operations and native/lookup strategy")
        expressions = []
        for operation in self.operations:
            if len(operation) != 2:
                raise ValueError("operations are (name, integer) pairs")
            name, value = operation
            if type(value) is not int or not 0 <= value <= 255:
                raise ValueError("byte operand must be in 0..255")
            if name == "min":
                expression = f"v > {value} ? {value} : v"
            elif name == "max":
                expression = f"v < {value} ? {value} : v"
            elif name in {"add", "xor"}:
                expression = f"v {'+' if name == 'add' else '^'} {value}"
            elif name == "right_shift" and value <= 7:
                expression = f"v >> {value}"
            elif name == "invert" and value == 0:
                expression = "255 - v"
            else:
                raise ValueError("unsupported byte operation")
            expressions.append("v = (uint8_t)(" + expression + ");")
        header = "#include <stdint.h>\n#include <stddef.h>\n"
        if strategy == "lookup":
            values = self.reference(np.arange(256, dtype=np.uint8))
            header += "static const uint8_t table[256]={" + ",".join(map(str, values)) + "};\n"
            body = "out[i]=table[x[i]];"
        else:
            body = "uint8_t v=x[i];" + "".join(expressions) + "out[i]=v;"
        source = (
            header
            + "void apply(const uint8_t*x,uint8_t*out,size_t n){for(size_t i=0;i<n;i++){"
            + body
            + "}}\n"
        )
        scope = [
            dict(
                name="byte",
                functions=[
                    dict(name="apply", returns="void", arguments=["pointer", "pointer", "size_t"])
                ],
            )
        ]
        with tempfile.TemporaryDirectory(prefix="lossless-byte-") as tmp:
            root = Path(tmp)
            self.build_receipt = build(
                root,
                [dict(name="byte", source=source)],
                scope,
                timeout=30,
                output=root / "build.json",
            )
            library = next(p for p in self.build_receipt["files"] if p.endswith((".dylib", ".so")))
            self._library = ctypes.CDLL(str(root / library))
            self._apply = self._library.apply
            self._apply.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            self._apply.restype = None

    def reference(self, x):
        x = np.array(x, copy=True)
        for name, value in self.operations:
            value = np.uint8(value)
            if name == "min":
                x = np.minimum(x, value)
            elif name == "max":
                x = np.maximum(x, value)
            elif name == "add":
                x = np.add(x, value)
            elif name == "xor":
                x = np.bitwise_xor(x, value)
            elif name == "right_shift":
                x = np.right_shift(x, value)
            elif name == "invert":
                x = np.subtract(np.uint8(255), x)
        return np.ascontiguousarray(x).reshape(x.shape)

    def __call__(self, x):
        if not isinstance(x, np.ndarray) or x.dtype != np.uint8 or not x.size:
            raise ValueError("nonempty uint8 array required")
        packed = np.ascontiguousarray(x)
        result = np.empty_like(packed, order="C")
        self._apply(int(packed.ctypes.data), int(result.ctypes.data), packed.size)
        return result.reshape(x.shape)
