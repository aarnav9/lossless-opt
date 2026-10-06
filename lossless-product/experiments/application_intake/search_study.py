"""Run the frozen pybaselines ModPoly source-authoring pilot through Lossless."""

import argparse
from pathlib import Path
import sys

import numpy as np

from lossless.applications import initialize, search
from lossless.applications.discovery import inventory
from lossless.applications.replay import copy_project

from study import REVISION, URL, git, log, sha, write


def make_cases(project):
    fixtures = project / "lossless_fixtures"
    fixtures.mkdir()
    specifications = [
        ("short", "discovery", 256, 7, {}),
        ("long", "discovery", 4096, 31, {}),
        ("batch_eight", "discovery", 1000, 19, {"batch": 8}),
        ("weighted", "discovery", 511, 89, {"weights": True, "return_coef": True}),
        ("mask", "discovery", 257, 91, {"mask_initial_peaks": True, "use_original": True}),
        ("flat", "discovery", 512, 0, {"flat": True}),
        ("odd_small", "evaluation", 127, 103, {"poly_order": 1}),
        ("odd_large", "evaluation", 2049, 107, {"poly_order": 5, "return_coef": True}),
        ("float32", "evaluation", 513, 109, {"dtype": "float32"}),
        ("batch_new", "evaluation", 1536, 113, {"batch": 8}),
        (
            "descending_weights",
            "evaluation",
            769,
            127,
            {"descending": True, "weights": True, "return_coef": True},
        ),
        ("mask_new", "evaluation", 1023, 131, {"mask_initial_peaks": True, "poly_order": 4}),
        ("original", "evaluation", 3072, 137, {"use_original": True, "weights": True}),
        ("zero", "evaluation", 255, 0, {"zero": True, "max_iter": 5}),
    ]
    result = []
    for name, split, size, seed, options in specifications:
        options = dict(options)
        batch = options.pop("batch", 1)
        dtype = options.pop("dtype", "float64")
        descending = options.pop("descending", False)
        weighted = options.pop("weights", False)
        flat, zero = options.pop("flat", False), options.pop("zero", False)
        calls = []
        for i in range(batch):
            rng = np.random.default_rng(seed + i)
            x = np.linspace(0, 1, size)
            y = 1 + 2 * np.exp(-3 * x) + 0.3 * x
            for center, width, height in [(0.17, 0.012, 3), (0.46, 0.027, 5), (0.79, 0.018, 2)]:
                y += height * np.exp(-0.5 * ((x - center) / width) ** 2)
            y += rng.normal(0, 0.025, size)
            if flat:
                y[:] = 2
            if zero:
                y[:] = 0
            weights = rng.uniform(0.2, 1, size)
            if descending:
                x, y, weights = x[::-1], y[::-1], weights[::-1]
            refs = {}
            for label, array in [("x", x), ("y", y.astype(dtype)), ("weights", weights)]:
                if label == "weights" and not weighted:
                    continue
                path = fixtures / f"{name}_{i}_{label}.npy"
                np.save(path, array)
                refs[label] = {"$npy": path.relative_to(project).as_posix()}
            kwargs = {"poly_order": 3, **options, "x_data": refs["x"]}
            if weighted:
                kwargs["weights"] = refs["weights"]
            calls.append({"args": [refs["y"]], "kwargs": kwargs})
        result.append({"id": name, "split": split, "calls": calls})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=1800)
    parser.add_argument("--final-reserve", type=float, default=600)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="medium")
    args = parser.parse_args()
    project, out = args.project.resolve(), args.output.resolve()
    if git(project, "rev-parse", "HEAD") != REVISION or git(project, "status", "--porcelain"):
        raise ValueError("expected clean pinned pybaselines v1.2.1 checkout")
    if out == project or out.is_relative_to(project) or project.is_relative_to(out):
        raise ValueError("study output must be separate from upstream")
    out.mkdir(parents=True, exist_ok=False)
    log(
        f"START expected=10-20min limit={args.budget}s result={out / 'run/report.json'} log={out.with_suffix('.log')}"
    )
    prepared = out / "prepared"
    copy_project(project, prepared, inventory(project))
    selected = make_cases(prepared)
    write(out / "cases.json", selected)
    initialize(
        prepared,
        out / "job",
        entry="pybaselines.polynomial:modpoly",
        cases=out / "cases.json",
        python=sys.executable,
    )
    plan = {
        "schema_version": 1,
        "source_scope": [{"path": "pybaselines/polynomial.py", "symbol": "_Polynomial.modpoly"}],
        "context_symbols": [
            {"path": "pybaselines/utils.py", "symbol": "relative_difference"},
            {"path": "pybaselines/_algorithm_setup.py", "symbol": "_Algorithm._setup_polynomial"},
            {"path": "pybaselines/_algorithm_setup.py", "symbol": "_PolyHelper.recalc_vandermonde"},
            {"path": "pybaselines/_algorithm_setup.py", "symbol": "_PolyHelper.pseudo_inverse"},
        ],
        "contract": "Pinned pybaselines 1.2.1 public polynomial.modpoly, ordinary single calls and declared batches within an already-running Python application. Original public callable is the frozen deployment comparator; no separately retained optimized recipe exists for this workload. Finite real float32/float64 1D spectra, finite x coordinates, valid orders/weights, normal/flat/zero inputs, use_original, mask_initial_peaks, return_coef branches. Preserve exact baseline, returned weights, coefficient and tolerance-history arrays, input mutation and previously returned results. Preserve original tol and max_iter. No global cache, changed lifecycle, changed entry or relaxed arithmetic. No new dependencies. No claims about untested invalid-input exceptions, array aliasing, concurrent calls, class-instance caching, views, larger inputs, other algorithms, or fresh-process-per-call latency. Equal case weighting is an experiment choice; actual deployment traffic is unknown.",
        "wall_time_seconds": args.budget,
        "final_reserve_seconds": args.final_reserve,
        "max_candidates": args.max_candidates,
        "provider": {
            "provider": "codex-chatgpt",
            "model": args.model,
            "effort": args.effort,
            "timeout_seconds": args.budget,
        },
    }
    write(out / "plan.json", plan)
    write(
        out / "study.json",
        {
            "study": "application-search-pybaselines",
            "upstream": {"url": URL, "revision": REVISION, "license": "BSD-3-Clause"},
            "study_sha256": sha(Path(__file__)),
            "cases_sha256": sha(out / "cases.json"),
            "plan_sha256": sha(out / "plan.json"),
        },
    )
    result = search(out / "job/lossless.json", plan=out / "plan.json", output=out / "run")
    log(
        f"DONE status={result.summary['status']} original_clean={not git(project, 'status', '--porcelain')} result={result.directory / 'report.json'} log={out.with_suffix('.log')}"
    )
    return 0 if result.summary["status"] in {"accepted_experiment", "reference_retained"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
