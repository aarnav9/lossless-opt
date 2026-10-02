import numpy as np


def arrays(a):
    if "bfloat16" in str(a.dtype):
        import mlx.core as mx

        return np.array(a.astype(mx.float32)), np.array(a.view(mx.uint16)).tobytes()
    value = np.array(a)
    return value, value.tobytes()


def diff(a, b):
    aa, ab = arrays(a)
    bb, bbits = arrays(b)
    same_dtype = a.dtype == b.dtype
    if aa.shape != bb.shape:
        return {
            "shape_equal": False,
            "dtype_equal": same_dtype,
            "bitwise": False,
            "finite": False,
            "max_abs": None,
            "rms": None,
        }
    delta = aa.astype(np.float64) - bb.astype(np.float64)
    finite = bool(np.isfinite(aa).all() and np.isfinite(bb).all())
    return {
        "shape_equal": True,
        "dtype_equal": same_dtype,
        "bitwise": same_dtype and ab == bbits,
        "finite": finite,
        "max_abs": float(np.max(np.abs(delta))) if finite else None,
        "rms": float(np.sqrt(np.mean(delta * delta))) if finite else None,
    }


def cache_diff(a, b):
    bad = {
        "offset_equal": False,
        "dtype_equal": False,
        "bitwise": False,
        "finite": False,
        "max_abs": None,
    }
    if not a or len(a) != len(b):
        return bad
    rows = []
    for ca, cb in zip(a, b):
        if ca.offset != cb.offset:
            return bad
        av, bv = list(ca.keys_and_values()), list(cb.keys_and_values())
        if len(av) != 2 or len(bv) != 2:
            return bad
        rows.extend(diff(x, y) for x, y in zip(av, bv))
    finite = all(x["finite"] for x in rows)
    return {
        "offset_equal": True,
        "dtype_equal": all(x["dtype_equal"] for x in rows),
        "bitwise": all(x["bitwise"] for x in rows),
        "finite": finite,
        "max_abs": max(x["max_abs"] for x in rows) if finite else None,
    }


def prepare_prefix(model, ids, chunk=2048):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    for i in range(0, len(ids), chunk):
        model(mx.array([ids[i : i + chunk]]), cache=cache)
        mx.eval([c.state for c in cache])
    return cache
