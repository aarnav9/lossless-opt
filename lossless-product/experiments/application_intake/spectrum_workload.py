"""Study-owned spectrum-processing application, evaluated against pybaselines.

Input preparation, fitter/cache setup, correction and result construction are
inside the measured request. NumPy fixture decoding and result hashing are not.
The public/reuse comparison changes only MODE in separately frozen copies.
"""

import numpy as np
from pybaselines import Baseline
from pybaselines.polynomial import modpoly

MODE = "reuse"


class Processor:
    def __init__(self):
        self.fitters = {}

    def prepare(self, records):
        prepared = []
        for record in records:
            y = np.asarray(record["y"])
            x = np.asarray(record["x"])
            options = dict(record.get("options", {}))
            if "weights" in options:
                options["weights"] = np.asarray(options["weights"])
            prepared.append((x, y, options))
        return prepared

    def fit(self, x, y, options):
        if MODE == "public":
            return modpoly(y, x_data=x, **options)
        # Reuse the library's existing public Baseline facility. Include order
        # because changing order on an object can change its matrix layout.
        key = (x.dtype.str, x.shape, x.tobytes(), options.get("poly_order", 2))
        fitter = self.fitters.get(key)
        if fitter is None:
            fitter = Baseline(x_data=x)
            if len(self.fitters) >= 4:
                del self.fitters[next(iter(self.fitters))]
            self.fitters[key] = fitter
        return fitter.modpoly(y, **options)

    def __call__(self, records):
        results = []
        for x, y, options in self.prepare(records):
            baseline, parameters = self.fit(x, y, options)
            results.append(
                {"baseline": baseline, "corrected": y - baseline, "parameters": parameters}
            )
        return results


def make_processor():
    return Processor()


def observe_processor(processor, result, args, kwargs):
    """Read-only evaluator hook; never an author-editable function."""
    seen = set()

    def retained_bytes(item):
        if id(item) in seen:
            return 0
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            return item.nbytes
        if isinstance(item, (bytes, bytearray, str)):
            return len(item)
        if isinstance(item, dict):
            return sum(retained_bytes(k) + retained_bytes(v) for k, v in item.items())
        if isinstance(item, (tuple, list)):
            return sum(map(retained_bytes, item))
        if hasattr(item, "__dict__") and not isinstance(item, type) and not callable(item):
            return retained_bytes(vars(item))
        return 0

    if retained_bytes(vars(processor)) > 16 * 1024**2:
        raise ValueError("processor retained cache exceeds the frozen 16 MiB limit")
    if len(processor.fitters) > 4:
        raise ValueError("processor retains more than four grid/order fitters")
    return result
