"""Prepared metadata predicates; every changing floating-point array is still scanned."""

import struct

from ..jobs import identity
from .bundle import signature
from .worker import inputs, Observer


class RequestGuard:
    def __init__(self, cases, root, admitted, *, scratch_limit=2**20):
        import numpy as np

        self._patterns = []
        self._scratch = np.empty(0, dtype=np.bool_)
        self.scratch_limit = scratch_limit
        seen = set()
        for case in cases:
            values = inputs(case["calls"], root)
            try:
                key = identity(signature(values))
            except ValueError:
                continue
            if key in admitted and key not in seen:
                self._patterns.append(self._compile(values))
                seen.add(key)

    def _compile(self, value):
        import numpy as np

        if type(value) is np.ndarray:
            expected = (
                value.shape,
                value.dtype.str,
                value.strides,
                bool(value.flags.aligned),
                bool(value.flags.writeable),
            )
            return lambda x: (
                type(x) is np.ndarray
                and (
                    x.shape,
                    x.dtype.str,
                    x.strides,
                    bool(x.flags.aligned),
                    bool(x.flags.writeable),
                )
                == expected
            )
        if type(value) is list:
            children = [self._compile(x) for x in value]
            return lambda x: (
                type(x) is list
                and len(x) == len(children)
                and all(check(v) for check, v in zip(children, x))
            )
        if type(value) is dict:
            children = {k: self._compile(v) for k, v in value.items()}
            return lambda x: (
                type(x) is dict
                and x.keys() == children.keys()
                and all(check(x[k]) for k, check in children.items())
            )
        if type(value) is float:
            bits = struct.pack(">d", value)
            return lambda x: type(x) is float and struct.pack(">d", x) == bits
        if value is None or type(value) in {bool, int, str, bytes}:
            return lambda x: type(x) is type(value) and x == value
        expected = identity(Observer(1024 * 1024).encode(value))
        return lambda x: identity(Observer(1024 * 1024).encode(x)) == expected

    def _finite(self, value):
        import numpy as np

        if type(value) is np.ndarray:
            if value.dtype.kind in "bui":
                return True
            if value.dtype.kind not in "fc":
                return False
            if value.size > self.scratch_limit:
                return bool(np.isfinite(value).all())
            if value.size > self._scratch.size:
                self._scratch = np.empty(value.size, dtype=np.bool_)
            result = self._scratch[: value.size].reshape(value.shape)
            np.isfinite(value, out=result)
            return bool(result.all())
        if type(value) is list:
            return all(self._finite(v) for v in value)
        if type(value) is dict:
            return all(self._finite(v) for v in value.values())
        return True

    def admits(self, calls):
        try:
            return any(check(calls) for check in self._patterns) and self._finite(calls)
        except (ValueError, TypeError, OverflowError):
            return False
