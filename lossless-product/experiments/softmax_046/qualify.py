"""Broad discovery, then untouched confirmation of one scoped softmax recipe."""

import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

from lossless._native import operators
from lossless._native.common import geomean, interval, measure, read, sha, stamp, write
from lossless._native.worker import compile_job
from lossless.artifacts import NativeOperation

HERE = Path(__file__).resolve().parent
PRODUCT = HERE.parents[1]
loader = importlib.util.spec_from_file_location(
    "qualification_039", HERE.parent / "softmax_039/qualify.py"
)
old = importlib.util.module_from_spec(loader)
loader.loader.exec_module(old)
TIMED = ["gaussian", "uniform", "extreme", "constant", "single_peak", "near_ties"]
FINAL_SHAPES = [
    (1, 17),
    (2, 129),
    (4, 255),
    (6, 33),
    (7, 257),
    (14, 509),
    (30, 1021),
    (34, 509),
    (35, 1027),
    (48, 521),
    (62, 2053),
    (66, 2039),
    (95, 313),
    (126, 4093),
    (130, 4099),
    (259, 1031),
]
SCOPES = ["all", "large", "large_c", "large_f", "large_slice2"]
CANDIDATES = ["prior_044", "first_045", "final_045"]


def in_scope(case, scope):
    if scope == "all":
        return True
    large = (
        case["rows"] >= 32 and case["columns"] >= 128 and case["rows"] * case["columns"] >= 16384
    )
    return large and (scope == "large" or case["layout"] == scope.removeprefix("large_"))


def inputs(case, seed, distribution):
    import numpy as np

    arrays = operators.input_arrays("softmax", case, seed, distribution)
    x = arrays[0]
    palettes = {
        "near_ties": [
            1,
            np.nextafter(np.float32(1), np.float32(2)),
            np.nextafter(np.float32(1), np.float32(0)),
        ],
        "alternating_bounds": [-10000, 10000],
        "underflow_steps": [0, -80, -90, -104, -10000],
    }
    if distribution in palettes:
        x[:] = np.resize(np.array(palettes[distribution], np.float32), x.shape)
    elif distribution == "negative_bound":
        x[:] = -10000
    elif distribution == "tiny_signed":
        x[:] *= np.float32(1e-40)
    return arrays


def numpy_allocating(x):
    import numpy as np

    out = np.array(x, dtype=np.float32, copy=True, order="C")
    out -= np.max(x, axis=1, keepdims=True)
    np.exp(out, out=out)
    out /= np.sum(out, axis=1, keepdims=True, dtype=np.float64)
    return out


def freeze(out):
    from lossless import recipes

    out.mkdir(parents=True, exist_ok=False)
    sources = out / "sources"
    sources.mkdir()
    bundle = json.loads(gzip.decompress((PRODUCT / "docs/assets/iterate-045.json.gz").read_bytes()))
    for name, source_name in [
        ("prior_044", "prior_044"),
        ("first_045", "round01_candidate1"),
        ("final_045", "round03_candidate0"),
    ]:
        source = bundle["sources"][f"<study>/iterative/proposals/{source_name}/kernel.c"]["source"]
        (sources / f"{name}.c").write_text(source)
    for proposal in recipes.native_candidates("softmax"):
        (sources / f"{proposal['id']}.c").write_text(proposal["source"])
    (sources / "baseline.c").write_text(operators.baseline_source("softmax"))
    files = [
        Path(__file__),
        Path(old.__file__),
        *[
            PRODUCT / "src/lossless" / p
            for p in [
                "artifacts.py",
                "_native/operators.py",
                "_native/common.py",
                "_native/worker.py",
            ]
        ],
    ]
    write(
        out / "plan.json",
        {
            "campaign": "046",
            "replicates": 3,
            "candidates": CANDIDATES,
            "scopes": SCOPES,
            "discovery_shapes": old.SHAPES,
            "confirmation_shapes": FINAL_SHAPES,
            "layouts": ["c", "f", "slice2"],
            "distributions": [*operators.CONTRACTS["softmax"]["distributions"], *old.EXTRA],
            "timed_distributions": TIMED,
            "screening_blocks": 5,
            "confirmation_blocks": 15,
            "target_ns": 300000,
            "sources": {p.name: sha(p) for p in sources.glob("*.c")},
            "harness": {str(p.resolve()): sha(p) for p in files},
            "source_evidence_sha256": sha(PRODUCT / "docs/assets/iterate-045.json.gz"),
            "rules": "Three fixed sources, no author calls or code repairs. Each case/distribution/interface screens the retained and applicable library references, freezes the fastest reference, then measures separate confirmation blocks. Ordinary calls include NativeOperation validation/allocation; NumPy allocating reference includes its own allocations without artificial validation overhead. All correctness checks must pass. A candidate/scope must gain >=1.02 geomean on ordinary and bound calls in each process, with no case/distribution/interface upper interval below .95 in two processes. Select one passing discovery pair by ordinary-call geomean, then open untouched confirmation shapes. Same gates there; no second-choice retry on final failure. Scope bounds alone are not universal validation; deployment retains exact shape/layout/environment guards and reference fallback. No automatic default promotion.",
        },
    )


def verify(out):
    plan = read(out / "plan.json")
    for name, digest in plan["sources"].items():
        assert sha(out / "sources" / name) == digest, f"source changed: {name}"
    for file, digest in plan["harness"].items():
        assert sha(file) == digest, f"harness changed: {file}"
    return plan


def operation(folder, name, cases, suffix):
    import numpy as np

    return NativeOperation(
        folder,
        {
            "adapter": "native.softmax",
            "selected": name,
            "cases": cases,
            "platform": sys.platform,
            "machine": platform.machine(),
            "numpy": np.__version__,
            "binary": name + suffix,
            "hashes": {name + suffix: sha(folder / (name + suffix))},
        },
    )


def worker(out, stage, replicate):
    import numpy as np

    plan = verify(out)
    selection = read(out / "selection.json") if stage == "confirmation" else None
    candidates = [selection["candidate"]] if selection else plan["candidates"]
    shapes = plan[stage + "_shapes"]
    cases = [
        {"rows": r, "columns": c, "layout": layout} for r, c in shapes for layout in plan["layouts"]
    ]
    if selection:
        cases = [c for c in cases if in_scope(c, selection["scope"])]
    folder = out / f"{stage}_{replicate}"
    folder.mkdir()
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    libraries, setup, ops = {}, {}, {}
    for name in ["baseline", "softmax_row_16", "softmax_column_4", *candidates]:
        started = time.perf_counter()
        compiled = compile_job(
            {
                "source": str(out / "sources" / (name + ".c")),
                "binary": str(folder / (name + suffix)),
            }
        )
        compile_seconds = time.perf_counter() - started
        assert compiled["status"] == "ok", compiled
        libraries[name] = operators.load_library(folder / (name + suffix))
        started = time.perf_counter()
        ops[name] = operation(folder, name, cases, suffix)
        setup[name] = {
            "compile_seconds": compile_seconds,
            "load_seconds": time.perf_counter() - started,
            "compile": compiled,
        }
    write(folder / "setup.json", setup)
    order = list(enumerate(cases))
    random.Random(4601 + replicate + (100 if selection else 0)).shuffle(order)
    for number, (case_id, case) in enumerate(order):
        seed = 460000 + replicate * 10000 + case_id * 37 + (100000 if selection else 0)
        checks, timings, bindings = [], [], []
        for offset, dist in enumerate(plan["distributions"]):
            arrays = inputs(case, seed + offset, dist)
            before = operators.digest(arrays)
            expected = operators.reference("softmax", arrays)
            for name, lib in libraries.items():
                funcs, guards, poison = operators.executors(
                    "softmax", case, arrays, libraries["baseline"], lib
                )
                poison()
                check = operators.check("softmax", funcs["proposal"](), expected)
                check.update(
                    guards_intact=guards(), inputs_unchanged=operators.digest(arrays) == before
                )
                checks.append({"implementation": name, "distribution": dist, **check})
            for name, result in [
                ("numpy_allocating", numpy_allocating(arrays[0])),
                *[(name + "_ordinary", op(arrays[0])) for name, op in ops.items()],
            ]:
                checks.append(
                    {
                        "implementation": name,
                        "distribution": dist,
                        "guards_intact": True,
                        "inputs_unchanged": operators.digest(arrays) == before,
                        **operators.check("softmax", result, expected),
                    }
                )
            references, guards, _ = operators.executors(
                "softmax", case, arrays, libraries["baseline"], libraries[candidates[0]]
            )
            for name, fn in references.items():
                if name.startswith(("numpy_", "scipy_")):
                    checks.append(
                        {
                            "implementation": name,
                            "distribution": dist,
                            **operators.check("softmax", fn(), expected),
                            "guards_intact": guards(),
                            "inputs_unchanged": operators.digest(arrays) == before,
                        }
                    )
            if dist not in TIMED:
                continue
            bound = {}
            for name in ["softmax_row_16", "softmax_column_4", *candidates]:
                started = time.perf_counter()
                bound[name] = ops[name].bind(arrays[0])
                bind_seconds = time.perf_counter() - started
                started = time.perf_counter()
                bound[name]()
                bindings.append(
                    {
                        "name": name,
                        "distribution": dist,
                        "bind_seconds": bind_seconds,
                        "first_call_seconds": time.perf_counter() - started,
                    }
                )
            refs, _, _ = operators.executors(
                "softmax", case, arrays, libraries["baseline"], libraries[candidates[0]]
            )
            bound.update(
                {name: fn for name, fn in refs.items() if name.startswith(("numpy_", "scipy_"))}
            )
            ordinary = {
                name: (lambda op=ops[name]: op(arrays[0]))
                for name in ["softmax_row_16", "softmax_column_4", *candidates]
            }
            ordinary["numpy_allocating"] = lambda: numpy_allocating(arrays[0])
            if operators.scipy_softmax is not None:
                ordinary["scipy_softmax_float64"] = refs["scipy_softmax_float64"]
            for interface, funcs in [("bound", bound), ("ordinary", ordinary)]:
                screening = measure(
                    funcs, seed + offset, plan["screening_blocks"], plan["target_ns"]
                )
                reference = min(
                    [name for name in funcs if name not in candidates],
                    key=screening["median_us"].get,
                )
                confirmation = measure(
                    funcs, seed + offset + 1, plan["confirmation_blocks"], plan["target_ns"]
                )
                timings.append(
                    {
                        "interface": interface,
                        "distribution": dist,
                        "reference": reference,
                        "screening": screening,
                        "confirmation": confirmation,
                        "comparisons": {
                            name: {
                                "speedup": confirmation["median_us"][reference]
                                / confirmation["median_us"][name],
                                "ci95": interval(
                                    confirmation["samples_us"][reference],
                                    confirmation["samples_us"][name],
                                    seed + offset,
                                ),
                            }
                            for name in candidates
                        },
                    }
                )
        correct = all(c["passed"] and c["guards_intact"] and c["inputs_unchanged"] for c in checks)
        write(
            folder / f"case_{case_id:03d}.json",
            {
                "case": case,
                "checks": checks,
                "correct": correct,
                "timings": timings,
                "bindings": bindings,
            },
        )
        stamp(
            f"QUALIFY stage={stage} process={replicate + 1}/3 case={number + 1}/{len(cases)} shape={case} correct={correct}"
        )
    cpu = (
        subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        if sys.platform == "darwin"
        else None
    )
    write(
        folder / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "cpu": cpu.stdout.strip()
            if cpu is not None and cpu.returncode == 0
            else platform.processor() or None,
            "cpu_probe_returncode": cpu.returncode if cpu is not None else None,
        },
    )


def assess(out, stage, candidate, scope):
    means, correctness, regressions, counts = [], True, {}, []
    for process in range(3):
        rows = [read(f) for f in sorted((out / f"{stage}_{process}").glob("case_*.json"))]
        rows = [r for r in rows if in_scope(r["case"], scope)]
        assert rows, "no evidence for scope"
        correctness &= all(r["correct"] for r in rows)
        counts.append(len(rows))
        gains = {interface: [] for interface in ["bound", "ordinary"]}
        for row in rows:
            for timing in row["timings"]:
                value = timing["comparisons"][candidate]
                gains[timing["interface"]].append(value["speedup"])
                key = json.dumps(
                    [row["case"], timing["interface"], timing["distribution"]], sort_keys=True
                )
                regressions[key] = regressions.get(key, 0) + (value["ci95"][1] < 0.95)
        means.append({k: geomean(v) for k, v in gains.items()})
    failures = [json.loads(k) for k, count in regressions.items() if count >= 2]
    return {
        "candidate": candidate,
        "scope": scope,
        "correct": correctness,
        "geomeans": means,
        "case_counts": counts,
        "repeated_regressions": failures,
        "passed": correctness
        and all(v >= 1.02 for row in means for v in row.values())
        and not failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--stage", choices=["discovery", "confirmation"], default="discovery")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.worker is not None:
        worker(out, args.stage, args.worker)
        return
    started = time.monotonic()
    stamp(f"START estimated=8-15min result={out} log={out.parent / (out.name + '.log')}")
    freeze(out)
    env = {
        **os.environ,
        **{
            key: "1"
            for key in [
                "PYTHONUNBUFFERED",
                "VECLIB_MAXIMUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
            ]
        },
    }
    for stage in ["discovery", "confirmation"]:
        stage_started = time.monotonic()
        for replicate in range(3):
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    __file__,
                    "--output",
                    str(out),
                    "--stage",
                    stage,
                    "--worker",
                    str(replicate),
                ],
                env=env,
                check=True,
                timeout=900,
            )
        if stage == "discovery":
            assessments = [
                assess(out, stage, candidate, scope) for candidate in CANDIDATES for scope in SCOPES
            ]
            write(
                out / "discovery_summary.json",
                {"assessments": assessments, "seconds": time.monotonic() - stage_started},
            )
            passing = [a for a in assessments if a["passed"]]
            if not passing:
                write(
                    out / "summary.json",
                    {
                        "status": "no_qualified_discovery_scope",
                        "discovery": assessments,
                        "confirmation_opened": False,
                        "seconds": time.monotonic() - started,
                    },
                )
                break
            winner = max(
                passing,
                key=lambda a: (
                    geomean([r["ordinary"] for r in a["geomeans"]]),
                    a["candidate"],
                    a["scope"],
                ),
            )
            write(
                out / "selection.json",
                {
                    "candidate": winner["candidate"],
                    "scope": winner["scope"],
                    "source_sha256": sha(out / "sources" / (winner["candidate"] + ".c")),
                    "discovery_summary_sha256": sha(out / "discovery_summary.json"),
                },
            )
            stamp(
                f"SEALED selection={winner['candidate']} scope={winner['scope']}; final cases now eligible"
            )
        else:
            selected = read(out / "selection.json")
            result = assess(out, stage, selected["candidate"], selected["scope"])
            write(
                out / "summary.json",
                {
                    "status": "qualified_for_scoped_deployment_check"
                    if result["passed"]
                    else "confirmation_failed",
                    "confirmation": result,
                    "confirmation_opened": True,
                    "seconds": time.monotonic() - started,
                },
            )
    stamp(f"COMPLETE result={out / 'summary.json'} log={out.parent / (out.name + '.log')}")


if __name__ == "__main__":
    main()
