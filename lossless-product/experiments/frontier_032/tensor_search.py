"""Search a verified rank-seven decomposition family on measured model shapes."""

import argparse
from pathlib import Path
import random
import statistics
import time
import numpy as np
import sympy as sp
from lossless._native.common import stamp, write


def multiply(a, b, shear=0, order=tuple(range(7))):
    m, k = a.shape
    n = b.shape[1]
    # Include packing/padding and all addition/temporary costs in timings.
    if m % 2:
        a = np.pad(a, ((0, 1), (0, 0)))
    aa = np.split(a, 2, axis=0)
    bb = np.split(b, 2, axis=0)
    A, B = np.split(aa[0], 2, axis=1)
    C, D = np.split(aa[1], 2, axis=1)
    E, F = np.split(bb[0], 2, axis=1)
    G, H = np.split(bb[1], 2, axis=1)
    if shear:
        B = B + shear * A
        D = D + shear * C
        E = E - shear * G
        F = F - shear * H
    ops = [
        lambda: (A + D) @ (E + H),
        lambda: (C + D) @ E,
        lambda: A @ (F - H),
        lambda: D @ (G - E),
        lambda: (A + B) @ H,
        lambda: (C - A) @ (E + F),
        lambda: (B - D) @ (G + H),
    ]
    # Accumulate in varied algebraically equivalent orders; at most one product plus outputs live.
    blocks = [np.zeros((a.shape[0] // 2, n // 2), np.float32) for _ in range(4)]
    effects = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, -1)],
        [(1, 1), (3, 1)],
        [(0, 1), (2, 1)],
        [(0, -1), (1, 1)],
        [(3, 1)],
        [(0, 1)],
    ]
    for j in order:
        product = ops[j]()
        for dest, sign in effects[j]:
            if sign == 1:
                blocks[dest] += product
            else:
                blocks[dest] -= product
    return np.concatenate(
        [np.concatenate(blocks[:2], axis=1), np.concatenate(blocks[2:], axis=1)], axis=0
    )[:m]


def verify(shear):
    A, B, C, D, E, F, G, H = sp.symbols("A B C D E F G H", commutative=False)
    B1, D1, E1, F1 = B + shear * A, D + shear * C, E - shear * G, F - shear * H
    p = [
        (A + D1) * (E1 + H),
        (C + D1) * E1,
        A * (F1 - H),
        D1 * (G - E1),
        (A + B1) * H,
        (C - A) * (E1 + F1),
        (B1 - D1) * (G + H),
    ]
    actual = [p[0] + p[3] - p[4] + p[6], p[2] + p[4], p[1] + p[3], p[0] - p[1] + p[2] + p[5]]
    expected = [A * E + B * G, A * F + B * H, C * E + D * G, C * F + D * H]
    return all(sp.expand(x - y) == 0 for x, y in zip(actual, expected))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    stamp(f"START estimated=under2min result={out} log={out.parent / (out.name + '.log')}")
    data = np.load(a.data)
    hidden = data["hidden"].astype(np.float32)
    b = np.ascontiguousarray(data["q"].T, dtype=np.float32)
    configs = []
    for shear in [-2, -1, 0, 1, 2]:
        assert verify(shear)
        for seed in [0, 1, 2]:
            order = list(range(7))
            random.Random(seed).shuffle(order)
            configs.append({"shear": shear, "order": order})
    rows = []
    rng = np.random.default_rng(326)
    for m in [8, 33]:
        aa = np.resize(hidden, (m, hidden.shape[1])).copy()
        baseline = lambda: aa @ b
        timing = []
        for _ in range(15):
            t = time.perf_counter()
            baseline()
            timing.append(time.perf_counter() - t)
        scores = []
        for c in configs:
            times = []
            reference = aa.astype(np.float64) @ b.astype(np.float64)
            value = multiply(aa, b, **c)
            error = float(np.max(np.abs(value - reference)))
            allowed = 2e-4 + 2e-4 * np.abs(reference)
            valid = bool(np.all(np.abs(value - reference) <= allowed))
            for _ in range(9):
                t = time.perf_counter()
                multiply(aa, b, **c)
                times.append(time.perf_counter() - t)
            scores.append(
                {
                    "config": c,
                    "valid": valid,
                    "max_abs_error": error,
                    "median_seconds": statistics.median(times),
                }
            )
        eligible = [c for c in scores if c["valid"]]
        chosen = min(eligible, key=lambda x: x["median_seconds"]) if eligible else None
        # Freeze one discovery choice, then use new input perturbations and severe cancellation.
        cases = []
        for kind in ["fresh", "cancellation"]:
            x = aa + rng.normal(0, 0.01, aa.shape).astype(np.float32)
            y = b.copy()
            if kind == "cancellation":
                x[:, 1::2] = x[:, ::2]
                y[1::2] = -y[::2] + np.float32(1e-6)
            oracle = x.astype(np.float64) @ y.astype(np.float64)
            times = {"blas": [], "decomposition": []}
            valid = True
            errors = []
            if chosen:
                for rep in range(15):
                    order = list(times)
                    random.Random(329 + rep).shuffle(order)
                    for name in order:
                        t = time.perf_counter()
                        v = x @ y if name == "blas" else multiply(x, y, **chosen["config"])
                        times[name].append(time.perf_counter() - t)
                        if name == "decomposition":
                            errors.append(float(np.max(np.abs(v - oracle))))
                            valid &= bool(
                                np.all(np.abs(v - oracle) <= 2e-4 + 2e-4 * np.abs(oracle))
                            )
            cases.append(
                {
                    "kind": kind,
                    "valid": valid,
                    "max_abs_error": max(errors, default=None),
                    "samples": times,
                    "speedup": statistics.median(times["blas"])
                    / statistics.median(times["decomposition"])
                    if chosen
                    else None,
                }
            )
        rows.append(
            {
                "shape": [m, hidden.shape[1], b.shape[1]],
                "discovery": scores,
                "chosen": chosen,
                "evaluation": cases,
                "scalar_products_fraction": 7 / 8,
                "addition_and_padding_cost_included": True,
            }
        )
        stamp(
            f"TENSOR shape={rows[-1]['shape']} selected={None if chosen is None else chosen['config']}"
        )
    write(
        out / "summary.json",
        {
            "algebra": "Symbolically verified noncommutative 2x2 rank-7 family with five shear bases and three accumulation schedules, 15 candidates; not a novel rank-48 discovery.",
            "workloads": rows,
            "contract": {"atol": 2e-4, "rtol": 2e-4},
            "scope": "Real q-projection weights and recorded decode/prefill dimensions, dequantized FP32 CPU sandbox vs NumPy BLAS. This does not replace the MLX quantized kernel. Register pressure unavailable; allocations, additions and padding are timed.",
        },
    )
    stamp(f"COMPLETE result={out / 'summary.json'} log={out.parent / (out.name + '.log')}")


if __name__ == "__main__":
    main()
