"""Check the real exported recipe, its guards and deployment setup cost."""

import argparse
import copy
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import lossless
from lossless._native import operators
from lossless._native.common import read, sha, stamp, write
from lossless.artifacts import NativeOperation, export

import qualify


def check(run, out):
    out.mkdir(parents=True, exist_ok=False)
    report = read(run / "report.json")
    assert report["status"] == "accepted"
    assert report["selected"] == "softmax_fortran_046"
    expected_source = "34d962e0ebe3e012ddba475f224d3570114fa4f3f05f8847adbef31e0cd3235d"
    started = time.perf_counter()
    artifact = export(run, out / "artifact")
    export_seconds = time.perf_counter() - started
    assert sha(artifact / "implementation.c") == expected_source
    started = time.perf_counter()
    op = lossless.load(artifact)
    load_seconds = time.perf_counter() - started
    assert op.library is not None and op.fallback_reason is None
    cases = op.manifest["cases"]
    assert len(cases) == 19 and all(qualify.in_scope(c, "large_f") for c in cases)
    records, setup = [], []
    for index, case in enumerate(cases):
        for offset, dist in enumerate(
            [*operators.CONTRACTS["softmax"]["distributions"], *qualify.old.EXTRA]
        ):
            arrays = qualify.inputs(case, 469000 + index * 37 + offset, dist)
            expected = operators.reference("softmax", arrays)
            before = operators.digest(arrays)
            called = operators.bind_native
            with patch.object(operators, "bind_native", wraps=called) as native:
                ordinary = op(arrays[0])
                started = time.perf_counter()
                bound = op.bind(arrays[0])
                bind_seconds = time.perf_counter() - started
                started = time.perf_counter()
                result = bound()
                first_call_seconds = time.perf_counter() - started
                assert native.call_count == 2
            ordinary_check = operators.check("softmax", ordinary, expected)
            bound_check = operators.check("softmax", result, expected)
            assert ordinary_check["passed"] and bound_check["passed"]
            assert operators.digest(arrays) == before
            records.append(
                {
                    "case": case,
                    "distribution": dist,
                    "ordinary": ordinary_check,
                    "bound": bound_check,
                    "inputs_unchanged": True,
                    "native_dispatches": 2,
                }
            )
            if dist == "gaussian":
                setup.append(
                    {
                        "case": case,
                        "bind_seconds": bind_seconds,
                        "first_bound_call_seconds": first_call_seconds,
                    }
                )
        stamp(f"DEPLOY validated case={index + 1}/{len(cases)} shape={case}")
    fallbacks = []
    for rows, columns, layout in [
        (4, 33, "f"),
        (7, 255, "f"),
        (35, 1027, "c"),
        (35, 1027, "slice2"),
        (36, 1027, "f"),
    ]:
        case = {"rows": rows, "columns": columns, "layout": layout}
        arrays = operators.input_arrays("softmax", case, 469999)
        expected = np.asarray(operators.reference("softmax", arrays), dtype=np.float32)
        with patch.object(
            operators, "bind_native", side_effect=AssertionError("native route must be guarded")
        ):
            assert np.array_equal(op(arrays[0]), expected)
            assert np.array_equal(op.bind(arrays[0])(), expected)
        fallbacks.append({"case": case, "reference_bits_match": True})
    arrays = operators.input_arrays("softmax", cases[0], 469998)
    environment_checks = []
    for changed in ["numpy", "machine", "platform", "binary"]:
        manifest = copy.deepcopy(op.manifest)
        manifest[changed] = None if changed == "binary" else "different"
        changed_op = NativeOperation(artifact, manifest)
        assert changed_op.library is None
        expected = np.asarray(operators.reference("softmax", arrays), dtype=np.float32)
        assert np.array_equal(changed_op(arrays[0]), expected)
        environment_checks.append(changed)
    rejected = []
    for value in [float("nan"), float("inf"), 10001]:
        x = arrays[0].copy(order="F")
        x[0, 0] = value
        try:
            op(x)
        except ValueError:
            rejected.append(str(value))
        else:
            raise AssertionError("out-of-contract value admitted")
    result = {
        "status": "passed",
        "recipe_sha256": expected_source,
        "report": report,
        "export_seconds": export_seconds,
        "load_seconds": load_seconds,
        "setup": setup,
        "checks": records,
        "fallbacks": fallbacks,
        "environment_fallbacks": environment_checks,
        "invalid_values_rejected": rejected,
        "scope": "19 exact shapes and Fortran layout; numerical finite validation. Measured bind includes test instrumentation overhead. Default recipes unchanged.",
    }
    write(out / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    stamp(
        f"START expected=under-1min result={out / 'summary.json'} log={out.parent / (out.name + '.log')}"
    )
    check(args.run.resolve(), out)
    stamp(f"COMPLETE result={out / 'summary.json'} log={out.parent / (out.name + '.log')}")


if __name__ == "__main__":
    main()
