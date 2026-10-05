"""Fixed synchronized MLX evaluator; model authors cannot modify this file."""

import argparse
import hashlib
import importlib.util
from pathlib import Path
import random
import time
import traceback

import numpy as np

from protocol import ATOL, RTOL, METHODS, digest, read, write
from lossless._native.common import stamp


def main(job, output):
    import mlx.core as mx
    import ports

    mx.set_memory_limit(3 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    originals = {
        n: getattr(mx, n)
        for n in [
            "eval",
            "synchronize",
            "get_peak_memory",
            "reset_peak_memory",
            "compile",
            "matmul",
            "conv2d",
        ]
    }

    def restored():
        assert all(getattr(mx, n) is v for n, v in originals.items()), (
            "modified evaluator/runtime global"
        )

    module = None
    import_seconds = 0
    if job.get("candidate"):
        begin = time.perf_counter()
        spec = importlib.util.spec_from_file_location("candidate_049", job["candidate"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import_seconds = time.perf_counter() - begin
        restored()

    def view(v, layout):
        if layout == "transposed_view" and v.ndim > 1:
            return mx.array(np.swapaxes(v, -1, -2).copy()).swapaxes(-1, -2)
        return mx.array(v)

    def identity(values):
        return hashlib.sha256(b"".join(np.array(v).tobytes() for v in values)).hexdigest()

    def compare(value, expected):
        a = np.array(value)
        if a.shape != expected.shape or a.dtype != expected.dtype:
            return dict(
                pass_=False,
                error="shape/dtype mismatch",
                actual_shape=list(a.shape),
                expected_shape=list(expected.shape),
            )
        finite = bool(np.isfinite(a).all())
        err = np.abs(a.astype(np.float64) - expected.astype(np.float64))
        limit = ATOL + RTOL * np.abs(expected.astype(np.float64))
        return dict(
            pass_=finite and bool(np.all(err <= limit)),
            max_absolute=float(err.max()) if finite else None,
            max_tolerance_fraction=float((err / limit).max()) if finite else None,
        )

    def call(fn, inputs):
        restored()
        mx.synchronize()
        start = time.perf_counter()
        y = fn(*inputs)
        mx.eval(y)
        mx.synchronize()
        seconds = time.perf_counter() - start
        restored()
        return y, seconds

    oracle_root = Path(job["oracle"])
    oracle_index = {r["case"]["id"]: r for r in read(oracle_root / "index.json")["records"]}
    records = []
    for case in job["cases"]:
        mx.clear_cache()
        file = oracle_root / (case["id"] + ".npz")
        assert digest(file) == oracle_index[case["id"]]["sha256"]
        with np.load(file) as stored:
            arrays = {k: stored[k] for k in stored.files}
        params = {k[3:]: mx.array(v) for k, v in arrays.items() if k.startswith("p__")}
        pool = [
            [view(arrays[f"x{v}_{i}"], case["layout"]) for i in range(len(case["shapes"]))]
            for v in range(5)
        ]
        mx.eval(params, pool)
        protected = [*params.values(), *[x for xs in pool for x in xs]]
        before = identity(protected)
        eager = ports.make(case["task"], case["config"], params, "eager")
        eager_arrays = [np.array(call(eager, xs)[0]) for xs in pool]
        setup, runners, errors = {}, {}, {}
        modes = (
            METHODS if job["phase"] == "baseline" else [job["references"][case["task"]], "eager"]
        )
        for mode in dict.fromkeys(modes):
            begin = time.perf_counter()
            runners[mode] = ports.make(case["task"], case["config"], params, mode)
            setup[mode] = time.perf_counter() - begin
        if job["phase"] != "baseline":
            ref_name = job["references"][case["task"]]
            runners["reference"] = runners[ref_name]
            setup["reference"] = setup[ref_name]
            if ref_name != "eager":
                del runners[ref_name]
            begin = time.perf_counter()
            try:
                instance = module.Runner(
                    case["task"], dict(case["config"]), dict(params), runners["reference"]
                )
                runners["candidate"] = instance.run
                restored()
            except Exception:
                errors["candidate"] = traceback.format_exc(limit=5)
            setup["candidate"] = time.perf_counter() - begin
        checks, first = {}, {}
        for name, fn in runners.items():
            try:
                validations = []
                first[name] = {}
                previous = None
                for variant, xs in enumerate(pool):
                    mx.reset_peak_memory()
                    y, sec = call(fn, xs)
                    if not variant:
                        first[name] = dict(seconds=sec, peak_bytes=mx.get_peak_memory())
                    c = compare(y, arrays[f"y{variant}"])
                    c["mlx_port"] = compare(y, eager_arrays[variant])
                    validations.append(c)
                    if variant == 0:
                        previous = y, np.array(y).copy()
                unchanged = (
                    np.array_equal(np.array(previous[0]), previous[1])
                    and identity(protected) == before
                )
                checks[name] = {
                    "pass": unchanged
                    and all(v["pass_"] and v["mlx_port"]["pass_"] for v in validations),
                    "inputs_parameters_previous_output_unchanged": unchanged,
                    "variants": validations,
                }
                if not unchanged:
                    raise RuntimeError("input/parameter/prior-output mutation")
            except Exception:
                checks[name] = {"pass": False, "error": traceback.format_exc(limit=5)}
                restored()
                if identity(protected) != before:
                    raise RuntimeError("mutation contaminated worker")
        for name, err in errors.items():
            checks[name] = {"pass": False, "error": err}
        samples, peaks = {n: [] for n in runners}, {n: [] for n in runners}
        names = [n for n in runners if checks[n]["pass"]]
        # Same invocation count per method/case, calibrated once from the fixed eager reference.
        probe = min(call(eager, pool[1])[1] for _ in range(3))
        inner = min(32, max(1, round(0.005 / max(probe, 1e-6))))
        rng = random.Random(job.get("seed", 4900) + case["seed"])
        for block in range(job["repeats"]):
            order = list(names)
            rng.shuffle(order)
            for name in order:
                fn = runners[name]
                mx.reset_peak_memory()
                start = time.perf_counter()
                for j in range(inner):
                    xs = pool[[0, 1, 3][(block + j) % 3]]
                    y = fn(*xs)
                    mx.eval(y)
                    mx.synchronize()
                seconds = (time.perf_counter() - start) / inner
                samples[name].append(seconds)
                peaks[name].append(mx.get_peak_memory())
                restored()
                # Validate the timed final output; numpy conversion is outside timing.
                variant = [0, 1, 3][(block + inner - 1) % 3]
                if (
                    not compare(y, arrays[f"y{variant}"])["pass_"]
                    or not compare(y, eager_arrays[variant])["pass_"]
                ):
                    checks[name]["pass"] = False
                    checks[name]["timed_output_failure"] = True
        assert identity(protected) == before, "timed call mutated protected inputs/parameters"
        record = dict(
            case=case,
            checks=checks,
            samples=samples,
            peaks=peaks,
            setup_seconds=setup,
            first_call=first,
            inner_calls=inner,
        )
        records.append(record)
        write(output, dict(status="running", records=records))
        stamp(
            f"049 WORKER case={case['id']} checks=" + str({n: v["pass"] for n, v in checks.items()})
        )
        del runners, params, pool, protected, eager, eager_arrays, arrays
    write(
        output,
        dict(
            status="complete",
            records=records,
            import_seconds=import_seconds,
            hardware=mx.device_info(),
            mlx=mx.__version__,
            numpy=np.__version__,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        main(read(a.job), a.output)
    except Exception:
        stamp(traceback.format_exc())
        raise
