"""Frozen, independent-process qualification of the campaign 033 softmax proposal."""

import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

from lossless._native.common import geomean, interval, measure, read, sha, stamp, write

SHAPES = [
    (1, 1),
    (1, 7),
    (1, 257),
    (1, 4097),
    (2, 3),
    (3, 31),
    (4, 33),
    (5, 127),
    (7, 255),
    (8, 256),
    (9, 257),
    (15, 511),
    (16, 512),
    (17, 513),
    (31, 1023),
    (32, 1024),
    (33, 1025),
    (63, 2047),
    (64, 2048),
    (65, 2049),
    (96, 311),
    (127, 4095),
    (128, 4096),
    (129, 4097),
    (257, 1024),
]
EXTRA = ["near_ties", "alternating_bounds", "negative_bound", "tiny_signed", "underflow_steps"]


def freeze(out):
    from lossless import recipes

    out.mkdir(parents=True, exist_ok=False)
    sources = out / "sources"
    sources.mkdir()
    for item in recipes.native_candidates("softmax"):
        (sources / (item["id"] + ".c")).write_text(item["source"])
    candidate = (
        Path(__file__).resolve().parents[1] / "frontier_032/manual_proposals/layout_dispatch.c"
    )
    (sources / "candidate.c").write_bytes(candidate.read_bytes())
    plan = {
        "cases": [
            {"rows": r, "columns": c, "layout": l} for r, c in SHAPES for l in ["c", "f", "slice2"]
        ],
        "replicates": 3,
        "screening_blocks": 5,
        "confirmation_blocks": 15,
        "target_ns": 500000,
        "extra_distributions": EXTRA,
        "source_hashes": {p.name: sha(p) for p in sources.glob("*.c")},
        "script_sha256": sha(__file__),
        "promotion_rule": "All finite-domain checks pass; each process geomean against screened retained winner >=1.02; no case's upper paired interval <0.95 in two or more processes. Any failure keeps the source experimental.",
        "scope": "One machine, three fresh processes; frozen candidate and test plan before timings, no adaptive refinement or universal equivalence claim.",
    }
    write(out / "plan.json", plan)


def worker(out, replicate):
    import numpy as np
    from lossless._native import operators
    from lossless._native.worker import compile_job

    plan = read(out / "plan.json")
    assert sha(__file__) == plan["script_sha256"]
    folder = out / f"process_{replicate}"
    folder.mkdir()
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    libraries = {}
    for name, digest in plan["source_hashes"].items():
        source = out / "sources" / name
        assert sha(source) == digest
        binary = folder / (source.stem + suffix)
        result = compile_job({"source": str(source), "binary": str(binary)})
        assert result["status"] == "ok", result
        libraries[source.stem] = operators.load_library(binary)
    (folder / "baseline.c").write_text(operators.baseline_source("softmax"))
    baseline_path = folder / ("baseline" + suffix)
    assert (
        compile_job({"source": str(folder / "baseline.c"), "binary": str(baseline_path)})["status"]
        == "ok"
    )
    baseline = operators.load_library(baseline_path)
    cases = list(enumerate(plan["cases"]))
    random.Random(7290 + replicate).shuffle(cases)
    records = []
    for index, (case_id, case) in enumerate(cases):
        seed = 91234 + replicate * 10000 + case_id * 19
        checks = []
        for offset, dist in enumerate([*operators.CONTRACTS["softmax"]["distributions"], *EXTRA]):
            arrays = operators.input_arrays("softmax", case, seed + offset, dist)
            x = arrays[0]
            if dist == "near_ties":
                x[:] = np.resize(
                    np.array(
                        [
                            1,
                            np.nextafter(np.float32(1), np.float32(2)),
                            np.nextafter(np.float32(1), np.float32(0)),
                        ],
                        np.float32,
                    ),
                    x.shape,
                )
            elif dist == "alternating_bounds":
                x[:] = np.resize(np.array([-10000, 10000], np.float32), x.shape)
            elif dist == "negative_bound":
                x[:] = -10000
            elif dist == "tiny_signed":
                x[:] *= np.float32(1e-40)
            elif dist == "underflow_steps":
                x[:] = np.resize(np.array([0, -80, -90, -104, -10000], np.float32), x.shape)
            before = operators.digest(arrays)
            expected = operators.reference("softmax", arrays)
            for name, lib in libraries.items():
                funcs, guards, poison = operators.executors("softmax", case, arrays, baseline, lib)
                names = ["proposal"] if name != "candidate" else list(funcs)
                for method in names:
                    poison()
                    value = operators.check("softmax", funcs[method](), expected)
                    value.update(
                        guards_intact=guards(), inputs_unchanged=operators.digest(arrays) == before
                    )
                    checks.append(
                        {
                            "distribution": dist,
                            "implementation": name if method == "proposal" else method,
                            **value,
                        }
                    )
        arrays = operators.input_arrays("softmax", case, seed + 4000)
        funcs = {}
        for name, lib in libraries.items():
            funcs[name] = operators.bind_native(arrays, lib)
        references, _, _ = operators.executors(
            "softmax", case, arrays, baseline, libraries["candidate"]
        )
        funcs.update({n: f for n, f in references.items() if n.startswith(("numpy_", "scipy_"))})
        screen = measure(funcs, seed, plan["screening_blocks"], plan["target_ns"])
        retained = min(["softmax_row_16", "softmax_column_4"], key=screen["median_us"].get)
        confirmation = measure(funcs, seed + 1, plan["confirmation_blocks"], plan["target_ns"])
        med = confirmation["median_us"]
        records.append(
            {
                "case_id": case_id,
                "case": case,
                "checks": checks,
                "correct": all(
                    c["passed"] and c["guards_intact"] and c["inputs_unchanged"] for c in checks
                ),
                "screening": screen,
                "retained_comparator": retained,
                "confirmation": confirmation,
                "speedup_vs_retained": med[retained] / med["candidate"],
                "ci95_vs_retained": interval(
                    confirmation["samples_us"][retained],
                    confirmation["samples_us"]["candidate"],
                    seed,
                ),
                "speedup_vs_libraries": {
                    n: v / med["candidate"]
                    for n, v in med.items()
                    if n.startswith(("numpy_", "scipy_"))
                },
            }
        )
        write(folder / "results.json", records)
        stamp(
            f"QUALIFY process={replicate + 1}/3 case={index + 1}/{len(cases)} shape={case} correct={records[-1]['correct']}"
        )
    write(
        folder / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": {n: importlib.metadata.version(n) for n in ["numpy", "scipy"]},
            "compiler": subprocess.check_output(["clang", "--version"], text=True),
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--worker", type=int)
    a = p.parse_args()
    out = Path(a.output).resolve()
    if a.worker is not None:
        worker(out, a.worker)
        return
    started = time.monotonic()
    stamp(f"START estimated=2-4min result={out} log={out.parent / (out.name + '.log')}")
    freeze(out)
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    for i in range(3):
        subprocess.run(
            [sys.executable, "-u", __file__, "--output", str(out), "--worker", str(i)],
            env=env,
            check=True,
            timeout=100,
        )
    runs = [read(out / f"process_{i}/results.json") for i in range(3)]
    means = [geomean([r["speedup_vs_retained"] for r in rows]) for rows in runs]
    regressions = [
        case
        for case in range(len(read(out / "plan.json")["cases"]))
        if sum(
            next(r for r in rows if r["case_id"] == case)["ci95_vs_retained"][1] < 0.95
            for rows in runs
        )
        >= 2
    ]
    correct = all(r["correct"] for rows in runs for r in rows)
    promoted = correct and all(v >= 1.02 for v in means) and not regressions
    result = {
        "plan": read(out / "plan.json"),
        "correct": correct,
        "promotion_qualified": promoted,
        "geomean_vs_retained_by_process": means,
        "repeated_regression_case_ids": regressions,
        "total_case_processes": sum(map(len, runs)),
        "checks": sum(len(r["checks"]) for rows in runs for r in rows),
        "library_geomeans": [
            {
                n: geomean([r["speedup_vs_libraries"][n] for r in rows])
                for n in rows[0]["speedup_vs_libraries"]
            }
            for rows in runs
        ],
        "elapsed_seconds": time.monotonic() - started,
        "source_unchanged": all(
            sha(out / "sources" / n) == h
            for n, h in read(out / "plan.json")["source_hashes"].items()
        ),
    }
    write(out / "summary.json", result)
    stamp(
        f"COMPLETE promoted={promoted} geomeans={means} repeated_regressions={len(regressions)} result={out / 'summary.json'} log={out.parent / (out.name + '.log')}"
    )


if __name__ == "__main__":
    main()
