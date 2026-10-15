"""Explicit copied-value request schema, separate from JSON fixture references."""

from .wire import MAX_MESSAGE


def normalize(request):
    import numpy as np

    if type(request) is not dict or set(request) != {"calls"}:
        raise ValueError("value requests require exactly one calls field")
    calls = request["calls"]
    if type(calls) is not list or not 1 <= len(calls) <= 100:
        raise ValueError("value requests require 1..100 calls")
    size = 0

    def check(value, depth=0):
        nonlocal size
        if depth > 40:
            raise ValueError("value request tree too deep")
        if type(value) is np.ndarray:
            if (
                value.dtype.kind not in "buifc"
                or not value.flags.aligned
                or not value.flags.writeable
                or not (value.flags.c_contiguous or value.flags.f_contiguous)
            ):
                raise ValueError(
                    "value inputs require aligned, writeable, contiguous numeric arrays"
                )
            size += value.nbytes
        elif value is None or type(value) in {bool, int, float, str, bytes}:
            size += len(value) if type(value) in {str, bytes} else 16
        elif type(value) in {list, tuple}:
            for item in value:
                check(item, depth + 1)
        elif type(value) is dict and all(type(k) is str for k in value):
            for k, item in value.items():
                size += len(k)
                check(item, depth + 1)
        else:
            raise ValueError("unsupported direct request value")
        if size > MAX_MESSAGE:
            raise ValueError("value request exceeds 64 MiB transport limit")

    result = []
    for call in calls:
        if type(call) is not dict or set(call) - {"args", "kwargs"}:
            raise ValueError("each value call accepts only args and kwargs")
        args, kwargs = call.get("args", []), call.get("kwargs", {})
        if type(args) is not list or type(kwargs) is not dict:
            raise ValueError("call args must be a list and kwargs a dict")
        check(args)
        check(kwargs)
        result.append(dict(args=args, kwargs=kwargs))
    return {"calls": result}
