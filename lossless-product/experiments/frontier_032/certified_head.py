"""CPU decision-certificate pilot on real hidden states; separate token-only reference."""

import argparse
import ctypes
from pathlib import Path
import random
import statistics
import subprocess
import time
import numpy as np
from lossless._native.common import stamp, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    stamp(f"START estimated=1-3min result={out} log={out.parent / (out.name + '.log')}")
    data = np.load(a.data)
    w = np.ascontiguousarray(data["head"], dtype=np.float32)
    hidden = np.ascontiguousarray(data["hidden"], dtype=np.float32)
    v, d = w.shape
    block = 32
    src = Path(__file__).with_suffix(".c")
    binary = out / "head.dylib"
    subprocess.run(
        [
            "clang",
            "-O3",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-dynamiclib",
            str(src),
            "-o",
            str(binary),
            "-lm",
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(binary))
    fp = ctypes.POINTER(ctypes.c_float)
    dp = ctypes.POINTER(ctypes.c_double)
    sz = ctypes.c_size_t
    lib.full_head.argtypes = [fp, fp, fp, sz, sz]
    lib.full_head.restype = ctypes.c_int
    lib.suffixes.argtypes = [fp, dp, sz, sz, sz]
    lib.certified_head.argtypes = [
        fp,
        fp,
        dp,
        fp,
        dp,
        dp,
        ctypes.POINTER(ctypes.c_uint8),
        fp,
        ctypes.POINTER(ctypes.c_uint64),
        sz,
        sz,
        sz,
    ]
    lib.certified_head.restype = ctypes.c_int
    suffix = np.empty((v, (d + block - 1) // block + 1))
    partial = np.empty(v, np.float32)
    lo = np.empty(v)
    hi = np.empty(v)
    alive = np.empty(v, np.uint8)
    scratch = np.empty(v, np.float32)
    stats = np.zeros(2, np.uint64)
    ptr = lambda x, t: x.ctypes.data_as(t)
    wp = ptr(w, fp)
    t = time.perf_counter()
    lib.suffixes(wp, ptr(suffix, dp), v, d, block)
    setup = time.perf_counter() - t
    cases = [(f"real_{i}", x) for i, x in enumerate(hidden)] + [
        ("zero_tie", np.zeros(d, np.float32)),
        ("tiny", hidden[0] * np.float32(1e-20)),
        ("large", hidden[0] * np.float32(100)),
    ]
    results = []
    for name, x in cases:
        x = np.ascontiguousarray(x)
        xp = ptr(x, fp)
        full = lambda: lib.full_head(wp, xp, ptr(scratch, fp), v, d)
        cert = lambda: lib.certified_head(
            wp,
            xp,
            ptr(suffix, dp),
            ptr(partial, fp),
            ptr(lo, dp),
            ptr(hi, dp),
            ptr(alive, ctypes.POINTER(ctypes.c_uint8)),
            ptr(scratch, fp),
            ptr(stats, ctypes.POINTER(ctypes.c_uint64)),
            v,
            d,
            block,
        )
        expected = full()
        actual = cert()
        assert expected == actual, (name, expected, actual)
        operations = int(stats[0])
        fallback = bool(stats[1])
        times = {"reference": [], "certificate": []}
        for rep in range(7):
            order = list(times)
            random.Random(123 + rep).shuffle(order)
            for arm in order:
                t = time.perf_counter()
                value = (full if arm == "reference" else cert)()
                times[arm].append(time.perf_counter() - t)
                assert value == expected
        results.append(
            {
                "case": name,
                "token": expected,
                "fallback": fallback,
                "computed_fraction": operations / (v * d),
                "speedup": statistics.median(times["reference"])
                / statistics.median(times["certificate"]),
                "samples": times,
            }
        )
        stamp(
            f"CERTIFICATE case={name} fallback={fallback} computed={operations / (v * d):.3f} speedup={results[-1]['speedup']:.3f}"
        )
    write(
        out / "summary.json",
        {
            "shape": [v, d],
            "precompute_seconds": setup,
            "suffix_bytes": suffix.nbytes,
            "results": results,
            "scope": "Conditional analytic dot-product bounds plus finite validation against the named scalar FP32 CPU projection/log-normalization reference, using real dequantized model weights and hidden states. Not a machine-checked certificate or certificate for MLX quantized arithmetic; full-model token/KV contract unchanged. Conservative normalization margin and finite libm semantics remain assumptions to formalize before production.",
        },
    )
    stamp(f"COMPLETE result={out / 'summary.json'} log={out.parent / (out.name + '.log')}")


if __name__ == "__main__":
    main()
