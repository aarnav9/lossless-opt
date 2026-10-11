"""A bounded uint8 pointwise image pipeline, not an external application suite.

Every region has allocating NumPy, fused native arithmetic and native lookup
implementations. Timing includes packing, guards, allocation and ctypes dispatch.
"""

import ctypes
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import time
import tracemalloc

import numpy as np

STAGES = 6
OPERATIONS = ["clamp high at 224", "clamp low at 32", "invert", "shift right 1", "add 8", "xor 85"]
EXPRESSIONS = ["v > 224 ? 224 : v", "v < 32 ? 32 : v", "255 - v", "v >> 1", "v + 8", "v ^ 85"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def numpy_region(x, start, stop):
    for stage in range(start, stop):
        if stage == 0:
            x = np.minimum(x, np.uint8(224))
        elif stage == 1:
            x = np.maximum(x, np.uint8(32))
        elif stage == 2:
            x = np.subtract(np.uint8(255), x)
        elif stage == 3:
            x = np.right_shift(x, np.uint8(1))
        elif stage == 4:
            x = np.add(x, np.uint8(8))
        else:
            x = np.bitwise_xor(x, np.uint8(85))
    return np.ascontiguousarray(x)


def specs():
    return [
        dict(id=f"{backend}_{start}_{stop}", backend=backend, start=start, stop=stop)
        for start in range(STAGES)
        for stop in range(start + 1, STAGES + 1)
        for backend in ("numpy", "native", "lut")
    ]


def source():
    lines = ["#include <stdint.h>", "#include <stddef.h>"]
    for row in specs():
        if row["backend"] == "numpy":
            continue
        name, start, stop = row["id"], row["start"], row["stop"]
        if row["backend"] == "lut":
            values = numpy_region(np.arange(256, dtype=np.uint8), start, stop)
            lines.append(
                f"static const uint8_t table_{name}[256] = {{" + ",".join(map(str, values)) + "};"
            )
        lines.append(f"void {name}(const uint8_t *x, uint8_t *out, size_t n) {{")
        if row["backend"] == "lut":
            lines.append(f"for (size_t i=0; i<n; ++i) out[i] = table_{name}[x[i]];")
        else:
            lines.append("for (size_t i=0; i<n; ++i) { uint8_t v=x[i];")
            lines.extend(f"v = (uint8_t)({e});" for e in EXPRESSIONS[start:stop])
            lines.append("out[i] = v; }")
        lines.append("}")
    return "\n".join(lines) + "\n"


class Pipeline:
    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.source_path = directory / "kernels.c"
        self.source_path.write_text(source())
        compiler = shutil.which("clang") or shutil.which("cc")
        if not compiler:
            raise ValueError("a local C compiler is required")
        library = directory / ("kernels.dylib" if platform.system() == "Darwin" else "kernels.so")
        self.command = [
            compiler,
            "-O3",
            "-std=c11",
            "-shared",
            "-fPIC",
            str(self.source_path),
            "-o",
            str(library),
        ]
        started = time.perf_counter()
        subprocess.run(self.command, capture_output=True, text=True, check=True, timeout=60)
        self.compile_seconds = time.perf_counter() - started
        self.compiler_version = subprocess.check_output(
            [compiler, "--version"], text=True, timeout=5
        ).splitlines()[0]
        self.library = ctypes.CDLL(str(library))
        self.rows = {r["id"]: r for r in specs()}
        self.functions = {}
        for row in self.rows.values():
            if row["backend"] != "numpy":
                fn = getattr(self.library, row["id"])
                fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
                fn.restype = None
                self.functions[row["id"]] = fn

    def region(self, name, x):
        if not isinstance(x, np.ndarray) or x.dtype != np.uint8 or not x.size:
            raise ValueError("nonempty uint8 arrays required")
        row = self.rows[name]
        if row["backend"] == "numpy":
            return numpy_region(x, row["start"], row["stop"])
        packed = np.ascontiguousarray(x)
        out = np.empty(packed.shape, dtype=np.uint8)
        self.functions[name](packed.ctypes.data, out.ctypes.data, packed.size)
        return out

    def run(self, plan, x):
        for name in plan:
            x = self.region(name, x)
        return x

    def sequence(self, plan, inputs):
        return tuple(self.run(plan, x) for x in inputs)

    def validate_regions(self):
        checks = 0
        for row in self.rows.values():
            # All uint8 values, in both contiguous and strided input storage.
            for x in (
                np.arange(256, dtype=np.uint8),
                np.tile(np.arange(256, dtype=np.uint8), (2, 1)).T,
            ):
                before = x.copy()
                expected = numpy_region(x, row["start"], row["stop"])
                result = self.region(row["id"], x)
                saved = result.copy()
                second = self.region(row["id"], np.zeros_like(x))
                if not (
                    np.array_equal(result, expected)
                    and np.array_equal(x, before)
                    and np.array_equal(result, saved)
                    and result.flags.c_contiguous
                    and not np.shares_memory(result, x)
                    and not np.shares_memory(result, second)
                ):
                    raise ValueError(f"region correctness/ownership failed: {row['id']}")
                checks += 1
        return checks


REFERENCE = [f"numpy_{i}_{i + 1}" for i in range(STAGES)]
CONTROLS = {"reference": REFERENCE, "full_native": ["native_0_6"], "full_lut": ["lut_0_6"]}


def cases(split, seed):
    specifications = (
        [(16, 16, False), (96, 96, True), (384, 384, False)]
        if split == "discovery"
        else [(23, 31, True), (127, 111, False), (511, 509, False)]
    )
    rng = np.random.default_rng(seed + (10000 if split == "evaluation" else 0))
    return [
        rng.integers(0, 256, (h, w * (2 if strided else 1), 3), dtype=np.uint8)[
            :, :: 2 if strided else 1
        ]
        for h, w, strided in specifications
    ]


def measure(functions, *, repeats, target_seconds, seed):
    if repeats < 1 or target_seconds <= 0:
        raise ValueError("positive timing controls required")
    calibration = []
    for fn in functions.values():
        fn()
        start = time.perf_counter_ns()
        fn()
        calibration.append(max(1, time.perf_counter_ns() - start))
    # Same repetition count for every variant in the paired comparison.
    loops = max(1, min(1000, round(target_seconds * 1e9 / max(calibration))))
    samples = {name: [] for name in functions}
    orders = []
    rng = random.Random(seed)
    for _ in range(repeats):
        order = list(functions)
        rng.shuffle(order)
        orders.append(order)
        for name in order:
            fn = functions[name]
            start = time.perf_counter_ns()
            for _ in range(loops):
                fn()
            samples[name].append((time.perf_counter_ns() - start) / loops)
    return dict(
        samples_ns=samples,
        median_ns={k: statistics.median(v) for k, v in samples.items()},
        loops=loops,
        orders=orders,
    )


def measure_catalog(pipeline, inputs, *, repeats, target_seconds, seed, progress):
    rows = specs()
    raw = []
    for index, x in enumerate(inputs):
        intermediates = [x] + [numpy_region(x, 0, stop) for stop in range(1, STAGES)]
        timing = measure(
            {
                r["id"]: lambda r=r: pipeline.region(r["id"], intermediates[r["start"]])
                for r in rows
            },
            repeats=repeats,
            target_seconds=target_seconds,
            seed=seed + index,
        )
        raw.append(dict(case=index, shape=list(x.shape), strides=list(x.strides), timing=timing))
        progress(f"CATALOG case={index + 1}/{len(inputs)} regions={len(rows)}")
    for row in rows:
        row["measurement_ids"] = [f"catalog/case_{i}/{row['id']}" for i in range(len(raw))]
        row["cost_ns"] = sum(max(1, round(r["timing"]["median_ns"][row["id"]])) for r in raw)
        row["per_case_ns"] = [r["timing"]["median_ns"][row["id"]] for r in raw]
    return rows, raw


def peak_traced(fn):
    tracemalloc.start()
    try:
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
        del result
        return peak
    finally:
        tracemalloc.stop()


def evaluate(pipeline, plans, inputs, *, repeats, target_seconds, seed):
    # Identical plans share samples; labels cannot manufacture a speedup.
    unique, aliases, names = {}, {}, {}
    for name, plan in plans.items():
        canonical = names.setdefault(tuple(plan), name)
        aliases[name] = canonical
        unique[canonical] = plan

    def expand(timing):
        for field in ("samples_ns", "median_ns"):
            timing[field] = {name: timing[field][canonical] for name, canonical in aliases.items()}
        return timing

    for x in inputs:
        original = x.copy()
        expected = pipeline.run(REFERENCE, x)
        for name, plan in unique.items():
            result = pipeline.run(plan, x)
            saved = result.copy()
            second = pipeline.run(plan, np.zeros_like(x))
            if not (
                np.array_equal(result, expected)
                and np.array_equal(x, original)
                and np.array_equal(result, saved)
                and result.flags.c_contiguous
                and not np.shares_memory(result, x)
                and not np.shares_memory(result, second)
            ):
                raise ValueError(f"pipeline correctness/ownership failed: {name}")
    per_case = [
        expand(
            measure(
                {
                    name: lambda plan=plan, x=x: pipeline.run(plan, x)
                    for name, plan in unique.items()
                },
                repeats=repeats,
                target_seconds=target_seconds,
                seed=seed + i,
            )
        )
        for i, x in enumerate(inputs)
    ]
    funcs = {
        name: lambda plan=plan: pipeline.sequence(plan, inputs) for name, plan in unique.items()
    }
    sequence = expand(
        measure(funcs, repeats=repeats, target_seconds=target_seconds, seed=seed + 100)
    )
    peaks = {name: peak_traced(fn) for name, fn in funcs.items()}
    return dict(
        correctness="passed finite equality, input mutation and output ownership checks",
        sequence=sequence,
        per_case=per_case,
        traced_peak_bytes={name: peaks[canonical] for name, canonical in aliases.items()},
        plan_aliases=aliases,
    )


def verdict(evidence, name, *, minimum_speedup=1.02, max_regression=0.03):
    pairs = evidence["sequence"]["samples_ns"]
    ratios = [a / b for a, b in zip(pairs["reference"], pairs[name])]
    # An intentionally descriptive pilot gate; not a confidence interval.
    speedup = statistics.median(ratios)
    cases_speedup = [
        row["median_ns"]["reference"] / row["median_ns"][name] for row in evidence["per_case"]
    ]
    memory_ok = (
        evidence["traced_peak_bytes"][name]
        <= 2 * evidence["traced_peak_bytes"]["reference"] + 32768
    )
    eligible = (
        speedup >= minimum_speedup
        and all(r >= 1 / (1 + max_regression) for r in cases_speedup)
        and memory_ok
    )
    return dict(
        eligible=eligible,
        sequence_speedup=speedup,
        per_case_speedup=cases_speedup,
        memory_gate_passed=memory_ok,
    )
