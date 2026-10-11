"""Ordinary-call and bound-call lifecycle checks, outside performance counts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def worker(project, cache_limit=0):
    sys.path.insert(0, str(project))
    import numpy as np
    from lossless._native import operators
    from lossless._native.deployment import NativeOperation

    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    observed = {}

    def digest(value):
        return dict(
            dtype=value.dtype.str,
            shape=list(value.shape),
            bits=hashlib.sha256(value.tobytes()).hexdigest(),
        )

    for operator in ("copy", "softmax", "rmsnorm_residual"):
        lib = operators.load_library(project / (operator + suffix))
        cases = [dict(rows=3, columns=7, layout=layout) for layout in ("c", "f", "slice2")]
        operation = NativeOperation.from_loaded_library(operator, lib, cases)
        for layout in ("c", "f", "slice2"):
            case = dict(rows=3, columns=7, layout=layout)
            arrays = operators.input_arrays(operator, case, 34119)
            args = arrays if operator == "rmsnorm_residual" else (arrays[0],)
            key = f"{operator}/{layout}"
            before = [digest(a) for a in args]
            first = operation(*args)
            saved = digest(first)
            assert before == [digest(a) for a in args], key + " input mutation"
            args[0][0, 0] += 0.125
            second = operation(*args)
            assert digest(first) == saved, key + " retained output overwritten"
            assert not np.shares_memory(first, second), key + " shared ordinary outputs"
            observed[key] = [saved, digest(second)]
            bound = operation.bind(*args)
            bound_first = bound()
            bound_bits = digest(bound_first)
            args[0][0, 1] += 0.125
            bound_second = bound()
            assert bound_first is bound_second, key + " bound output contract changed"
            observed[key + "/bound"] = [bound_bits, digest(bound_second)]
            import gc
            import weakref

            owner_ref = weakref.ref(args[0])
            del args, arrays
            gc.collect()
            assert owner_ref() is not None, key + " bound input owner lost"
            assert digest(bound()) == observed[key + "/bound"][1], (
                key + " bound result changed after caller released inputs"
            )
        for bad in (float("nan"), float("inf"), -float("inf"), 10001.0):
            arrays = operators.input_arrays(operator, cases[0], 1142)
            arrays[0][0, 0] = bad
            args = arrays if operator == "rmsnorm_residual" else (arrays[0],)
            try:
                result = operation(*args)
            except ValueError:
                observed[f"{operator}/invalid/{bad}"] = "ValueError"
            else:
                assert operator == "copy", "invalid numerical input admitted"
                observed[f"{operator}/invalid/{bad}"] = digest(result)
        # Same shape and pointer, newly invalid values must never hit stale admission.
        if operator != "copy":
            arrays = operators.input_arrays(operator, cases[0], 9152)
            args = arrays if operator == "rmsnorm_residual" else (arrays[0],)
            operation(*args)
            arrays[0][1, 1] = float("nan")
            try:
                operation(*args)
            except ValueError:
                observed[operator + "/mutation_revalidated"] = True
            else:
                raise AssertionError("mutable values bypassed validation")
        for dimension, layout in [(9, "c"), (11, "f")]:
            arrays = operators.input_arrays(
                operator, dict(rows=2, columns=dimension, layout=layout), 8112
            )
            args = arrays if operator == "rmsnorm_residual" else (arrays[0],)
            observed[f"{operator}/fallback/{dimension}"] = digest(operation(*args))
        arrays = operators.input_arrays(operator, cases[0], 1981)
        if operator == "rmsnorm_residual":
            for name, args in [
                ("alias", (arrays[0], arrays[0], arrays[2])),
                ("bad_weight", (arrays[0], arrays[1], np.full(7, 3, dtype=np.float32))),
            ]:
                try:
                    operation(*args)
                except ValueError:
                    observed[f"{operator}/{name}"] = "ValueError"
                else:
                    raise AssertionError(name + " admitted")
        else:
            arrays[0].flags.writeable = False
            observed[operator + "/readonly"] = digest(operation(arrays[0]))
    # Copy preserves exceptional payload bits, signed zero and subnormals.
    bits = np.array(
        [0, 0x80000000, 1, 0x80000001, 0x7FC00123, 0xFFC00456, 0x7F800000], dtype=np.uint32
    )
    x = np.tile(bits.view(np.float32), (3, 1))
    op = NativeOperation.from_loaded_library(
        "copy",
        operators.load_library(project / ("copy" + suffix)),
        [dict(rows=3, columns=7, layout="c")],
    )
    result = op(x)
    assert result.tobytes() == x.tobytes()
    observed["copy/payloads"] = digest(result)
    large = np.arange(65537, dtype=np.float32).reshape(1, -1)
    observed["copy/large_fallback"] = digest(op(large))
    if cache_limit:
        cached = getattr(op, "_ones", None)
        assert cached is None or cached.nbytes <= cache_limit * 4, (
            "oversized fallback retained in default-weight cache"
        )
        observed["copy/cache_budget_passed"] = True
    return observed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", type=int)
    p.add_argument("--worker", type=Path)
    p.add_argument("--result", type=Path)
    p.add_argument("--cache-limit", type=int, default=0)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.worker:
        a.result.write_text(json.dumps(worker(a.worker, a.cache_limit)))
        return
    from common import DEFAULT_RUN, write, stamp

    out = a.output or DEFAULT_RUN
    folder = out / f"conformance_f3_{a.candidate:02d}"
    folder.mkdir()
    results = {}
    plan = json.loads((out / f"f3_{a.candidate:02d}/plan.json").read_text())
    cache_limit = plan["options"].get("ones_limit", 0)
    for name, project in [
        ("reference", out / "native_baseline"),
        ("candidate", out / f"f3_{a.candidate:02d}/project"),
    ]:
        destination = folder / (name + ".json")
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                __file__,
                "--worker",
                str(project.resolve()),
                "--result",
                str(destination.resolve()),
                "--cache-limit",
                str(cache_limit),
            ],
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        (folder / (name + ".log")).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{name} conformance failed: {folder}/{name}.log")
        results[name] = json.loads(destination.read_text())
    same = results["reference"] == results["candidate"]
    write(
        folder / "report.json",
        dict(
            exact=same,
            observations=len(results["reference"]),
            reference=results["reference"],
            candidate=results["candidate"],
        ),
    )
    stamp(
        f"NATIVE CONFORMANCE exact={same} observations={len(results['reference'])} result={folder}/report.json"
    )
    if not same:
        raise ValueError("native lifecycle/guard conformance differs")


if __name__ == "__main__":
    main()
