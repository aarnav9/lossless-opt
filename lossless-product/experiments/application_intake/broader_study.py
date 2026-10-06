"""One-hour, broader exact CPU application search with four frozen comparators.

Uses the subscription Codex provider through the public Lossless search API.
Synthetic spectrum-processing workload using pinned BSD-licensed pybaselines.
Preparation measures discovery only; one source winner is frozen before new
final cases open. All old study cases are development data in this experiment.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

from lossless.applications import initialize, search
from lossless.applications import patches
from lossless.applications.comparison import compare, eligible, run_case
from lossless.applications.config import resolve
from lossless.applications.discovery import inventory
from lossless.applications.replay import copy_project
from lossless.applications.runtime import clock

from search_study import make_cases
from study import REVISION, URL, git, log, sha, write

HERE = Path(__file__).resolve().parent
SCOPE = [
    {"path": "spectrum_workload.py", "symbol": "Processor.__init__"},
    {"path": "spectrum_workload.py", "symbol": "Processor.prepare"},
    {"path": "spectrum_workload.py", "symbol": "Processor.fit"},
    {"path": "spectrum_workload.py", "symbol": "Processor.__call__"},
    {"path": "pybaselines/polynomial.py", "symbol": "_Polynomial.modpoly"},
    {"path": "pybaselines/_algorithm_setup.py", "symbol": "_Algorithm._setup_polynomial"},
    {"path": "pybaselines/_algorithm_setup.py", "symbol": "_PolyHelper.recalc_vandermonde"},
    {"path": "pybaselines/_algorithm_setup.py", "symbol": "_PolyHelper.pseudo_inverse"},
]


def application_case(name, split, records, requests=3):
    # Fresh arrays are decoded for each request. Equal-valued but distinct arrays,
    # changing weights/orders and retained earlier outputs exercise cache guards.
    return {
        "id": name,
        "split": split,
        "calls": [
            {"args": [copy.deepcopy(records if i % 2 == 0 else records[::-1])]}
            for i in range(requests)
        ],
    }


def new_records(project, name, size, batch, seed, options=None, *, pattern="peaks"):
    options = dict(options or {})
    dtype = options.pop("dtype", "float64")
    ordering = options.pop("ordering", "ascending")
    weight_mode = options.pop("weight_mode", None)
    varying_order = options.pop("varying_order", False)
    irregular = options.pop("irregular", False)
    records = []
    for i in range(batch):
        rng = np.random.default_rng(seed + i)
        x = np.sort(rng.uniform(-2, 3, size)) if irregular else np.linspace(-2, 3, size)
        t = (x + 2) / 5
        y = 0.7 + 1.8 * np.exp(-2.4 * t) + 0.2 * t**2
        if pattern == "oscillatory":
            y += 0.3 * np.sin(19 * t) ** 2 + 0.5 * np.exp(-(((t - 0.62) / 0.055) ** 2))
        elif pattern == "steps":
            y += 0.6 * (t > 0.3) + 0.7 * (t > 0.71)
        elif pattern == "flat":
            y[:] = 0 if i % 2 else 1.5
        else:
            for _ in range(4):
                center, width, height = (
                    rng.uniform(0.05, 0.95),
                    rng.uniform(0.008, 0.08),
                    rng.uniform(0.5, 4),
                )
                y += height * np.exp(-0.5 * ((t - center) / width) ** 2)
        if pattern != "flat":
            y += rng.normal(0, 0.015, size)
        weight_rng = np.random.default_rng(seed if weight_mode == "shared" else seed + i)
        weights = weight_rng.uniform(0.15, 1, size)
        if weight_mode == "sparse":
            weights[::7] = 0
        order = np.arange(size)
        if ordering == "descending":
            order = order[::-1]
        elif ordering == "shuffled":
            order = np.random.default_rng(seed).permutation(size)
        arrays = {"x": x[order], "y": y[order].astype(dtype), "weights": weights[order]}
        refs = {}
        for label, array in arrays.items():
            if label == "weights" and weight_mode is None:
                continue
            path = project / "lossless_fixtures" / f"{name}_{i}_{label}.npy"
            np.save(path, array)
            refs[label] = {"$npy": path.relative_to(project).as_posix()}
        settings = {"poly_order": 3, "max_iter": 250, **options}
        if varying_order:
            settings["poly_order"] = [1, 5, 2, 4][i % 4]
        if weight_mode:
            settings["weights"] = refs["weights"]
        records.append({"x": refs["x"], "y": refs["y"], "options": settings})
    return records


def make_application_cases(project):
    old = make_cases(project)
    cases = []
    for row in old:
        records = []
        for i in range(16):
            call = copy.deepcopy(row["calls"][i % len(row["calls"])])
            x = call["kwargs"].pop("x_data")
            records.append({"x": x, "y": call["args"][0], "options": call["kwargs"]})
        cases.append(application_case("old_" + row["id"], "discovery", records, requests=2))
    development = [
        ("shared_weights", 1024, 24, 2001, {"weight_mode": "shared", "return_coef": True}),
        ("changing_orders", 777, 16, 2101, {"varying_order": True, "weight_mode": "changing"}),
        ("mixed_grids", 521, 12, 2201, {"irregular": True}),
    ]
    for name, size, batch, seed, options in development:
        records = new_records(project, name, size, batch, seed, options)
        cases.append(application_case(name, "discovery", records, requests=2))
    # Cases are declared before the model runs. They are never placed in provider
    # context; all these seeds/shapes/distributions differ from prior studies.
    final = [
        ("fresh_small", 193, 24, 30001, {}, "peaks"),
        ("fresh_large", 6143, 12, 31001, {"poly_order": 5, "return_coef": True}, "peaks"),
        ("fresh_f32", 897, 20, 32001, {"dtype": "float32", "use_original": True}, "oscillatory"),
        ("fresh_shared_weights", 1409, 24, 33001, {"weight_mode": "shared"}, "peaks"),
        (
            "fresh_changed_weights",
            997,
            16,
            34001,
            {"weight_mode": "changing", "return_coef": True},
            "steps",
        ),
        (
            "fresh_mask",
            1793,
            12,
            35001,
            {"mask_initial_peaks": True, "use_original": True},
            "peaks",
        ),
        (
            "fresh_descending",
            643,
            16,
            36001,
            {"ordering": "descending", "weight_mode": "sparse"},
            "peaks",
        ),
        ("fresh_orders", 1151, 20, 37001, {"varying_order": True, "return_coef": True}, "peaks"),
        (
            "fresh_irregular",
            863,
            16,
            38001,
            {"ordering": "shuffled", "irregular": True},
            "oscillatory",
        ),
        ("fresh_flat_zero", 383, 12, 39001, {"max_iter": 9}, "flat"),
        ("fresh_single", 271, 1, 40001, {"poly_order": 4}, "peaks"),
    ]
    for name, size, batch, seed, options, pattern in final:
        records = new_records(project, name, size, batch, seed, options, pattern=pattern)
        cases.append(application_case(name, "evaluation", records, requests=3))
    return cases


def compact(report):
    return {k: v for k, v in report.items() if k != "cases"}


def audit_variants(variants, snapshots, value, cases, out, deadline, pairs):
    reports = {}
    for name, candidate in variants.items():
        if name == "public_stock":
            continue
        log(f"BASELINE {name} discovery only")
        result = compare(
            variants["public_stock"],
            candidate,
            snapshots["public_stock"],
            snapshots[name],
            value,
            cases,
            out / name,
            pairs=pairs,
            deadline=deadline,
            progress=log,
            minimum_speedup=1.02,
        )
        reports[name] = compact(result)
        if result["status"] != "matched":
            raise ValueError(
                f"baseline {name} failed exact development comparison: {result.get('failed_case')}"
            )
    scores = {
        "public_stock": 1.0,
        **{k: v["statistics"]["case_balanced_geomean_speedup"] for k, v in reports.items()},
    }
    chosen = max(scores, key=scores.get)
    return {"reports": reports, "scores_vs_public_stock": scores, "chosen": chosen}


def memory_and_setup(
    reference, candidate, ref_snapshot, candidate_snapshot, value, cases, report, out, deadline
):
    memory = []
    for case, stats in zip(cases, report["statistics"]["cases"]):
        log(f"FINAL memory case={case['id']} comparator={out.name}")
        row = {"id": case["id"]}
        for name, project, snapshot in [
            ("reference", reference, ref_snapshot),
            ("candidate", candidate, candidate_snapshot),
        ]:
            row[name] = run_case(
                project,
                snapshot,
                value,
                case,
                out / (case["id"] + "_" + name),
                deadline,
                log,
                "memory",
            )
        expected = report["cases"][len(memory)]["pairs"][0]["reference"]["fingerprint"]
        row["matched"] = (
            row["candidate"]["fingerprint"] == row["reference"]["fingerprint"] == expected
        )
        row["eligible"] = (
            row["matched"]
            and row["candidate"]["peak_rss_bytes"] <= 1.2 * row["reference"]["peak_rss_bytes"]
            and row["candidate"]["traced_peak_bytes"]
            <= 1.2 * max(1, row["reference"]["traced_peak_bytes"])
        )
        memory.append(row)
    startup = all(
        c["medians"]["candidate"]["setup_seconds"]
        <= c["medians"]["reference"]["setup_seconds"] + 0.1
        for c in report["statistics"]["cases"]
    )
    return {
        "memory": memory,
        "startup_eligible": startup,
        "memory_eligible": all(r["eligible"] for r in memory),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-seconds", type=float, default=3600)
    parser.add_argument("--final-reserve", type=float, default=1200)
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    project, out = args.project.resolve(), args.output.resolve()
    if git(project, "rev-parse", "HEAD") != REVISION or git(project, "status", "--porcelain"):
        raise ValueError("expected clean pinned upstream checkout")
    if out == project or out.is_relative_to(project) or project.is_relative_to(out):
        raise ValueError("output must be separate from upstream")
    out.mkdir(parents=True, exist_ok=False)
    started = clock()
    log(
        f"START expected=75-100min search={args.search_seconds}s result={out / 'summary.json'} log={out.with_suffix('.log')}"
    )
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    base = out / "prepared"
    copy_project(project, base, inventory(project))
    shutil.copyfile(HERE / "spectrum_workload.py", base / "spectrum_workload.py")
    cases = make_application_cases(base)
    write(out / "cases.json", cases)
    variants = {}
    original = (base / "spectrum_workload.py").read_text()
    proposals = json.loads((HERE / "frozen-candidates.json").read_text())
    retained = proposals[2]
    for name in ("public_stock", "public_previous", "reuse_stock", "reuse_previous"):
        destination = out / "baselines" / name
        copy_project(base, destination, inventory(base))
        if name.startswith("public"):
            (destination / "spectrum_workload.py").write_text(
                original.replace('MODE = "reuse"', 'MODE = "public"')
            )
        if name.endswith("previous"):
            patches.apply(base, destination, retained, [SCOPE[4]])
        variants[name] = destination
    snapshots = {name: inventory(path) for name, path in variants.items()}
    initialize(
        base,
        out / "job",
        entry="spectrum_workload:make_processor",
        factory=True,
        cases=out / "cases.json",
        python=sys.executable,
    )
    config_path = out / "job/lossless.json"
    config = json.loads(config_path.read_text())
    config["environment"]["inherit_env"] = ["OPENBLAS_NUM_THREADS"]
    config["entry"]["observe"] = "spectrum_workload:observe_processor"
    config["replay"]["timeout_seconds"] = 60
    config["replay"]["max_output_bytes"] = 32000000
    write(config_path, config)
    value, cases = resolve(config_path)
    discovery = [c for c in cases if c["split"] == "discovery"]
    evaluation = [c for c in cases if c["split"] == "evaluation"]
    protocol = {
        "study": "broader-application-cpu-search",
        "upstream": {"url": URL, "revision": REVISION, "license": "BSD-3-Clause"},
        "source_hashes": {
            p.name: sha(p)
            for p in [
                Path(__file__),
                HERE / "spectrum_workload.py",
                HERE / "search_study.py",
                HERE / "frozen-candidates.json",
            ]
        },
        "cases_sha256": sha(out / "cases.json"),
        "comparators": {k: v["files"] for k, v in snapshots.items()},
        "search_seconds": args.search_seconds,
        "max_candidates": args.max_candidates,
        "selection": "Fastest global case-balanced discovery baseline among public/reused-object × stock/previous winner. Frozen before authoring; final candidate also compared with every baseline, no reselection.",
        "final_policy": "One frozen candidate. Exact all outputs/input mutation/retained outputs. Every comparator must pass >=1.02 overall, no case worse than 1/1.03, fixed 12-round sign p<=.05, setup increase<=.1s, separate RSS and traced peak<=1.2x. No tuning after final.",
        "measurement": "CPU, OpenBLAS one thread. Complete declared processor requests including input preparation, cache lookup/build, every fit, baseline subtraction and output construction. First call included. Import/factory setup and worker wall time separate; npy fixture decoding/hashing excluded. Synthetic application, not arbitrary user traffic.",
    }
    write(out / "protocol.json", protocol)
    if args.prepare_only:
        log(f"PREPARED result={out / 'protocol.json'} log={out.with_suffix('.log')}")
        return 0
    baseline_audit = audit_variants(
        variants, snapshots, value, discovery, out / "baseline-audit", clock() + 1200, 3
    )
    write(out / "baseline-audit.json", baseline_audit)
    chosen = baseline_audit["chosen"]
    log(f"FROZEN deployment comparator={chosen} scores={baseline_audit['scores_vs_public_stock']}")
    config["project"]["root"] = str(variants[chosen])
    write(config_path, config)
    plan = {
        "schema_version": 1,
        "source_scope": SCOPE,
        "context_symbols": [
            {"path": "pybaselines/_algorithm_setup.py", "symbol": "_Algorithm.__init__"},
            {"path": "pybaselines/_algorithm_setup.py", "symbol": "_Algorithm._register"},
            {"path": "pybaselines/utils.py", "symbol": "relative_difference"},
        ],
        "contract": "Broader exact CPU spectrum-processing application, pybaselines 1.2.1. Deployment comparator frozen before search: "
        + chosen
        + ". Existing Baseline object reuse and the previous convergence-norm winner are already available baselines: beat these, do not take credit for rediscovering them. Finite real float32/float64 1D spectra/x coordinates, orders 1..5, valid positive/zero weights, finite original/masked/flat/zero branches, original tol and max_iter. Full baseline/corrected spectra, weights, coefficients and every tol_history bit, unchanged inputs and retained earlier outputs are observed. Factory returns Processor once per process; all declared requests include first call, repeated and changed records. Author may change Processor lifecycle, preparation, fitting, batching, reusable buffers and listed polynomial/shared helpers; use bounded instance-local caches with content-based guards for shape/order/x/weights/layout as needed, no answer/result caches. At most four live grid/order entries and 16 MiB of persistent extra cache data. Never mutate caller inputs or returned arrays on subsequent requests. No global/unbounded caches, no hidden warmup or work moved to observer/evaluator/factory inputs, no changed entry/weights/precision/tolerances, no new dependencies/environment/files/threads/processes or evaluator introspection. Keep CPU, exact arithmetic/order or independently validated exact paths; fallback for unsupported branches. Other functions using modified shared helpers must preserve behavior. NumPy/SciPy/Numba already installed may be imported inside allowed bodies; JIT startup counts. Cases seen in prior studies are discovery now; final cases are new and never shown. Prior duplicate spectra are test fixtures, not permission to memoize answers. Mean case weights are experimental, production mix unknown. Timing includes every operation inside Processor.__call__ (input conversion, cache/build, fit, correction and construction), import/factory setup separately. Benchmark complete requests and first-call cost, not isolated helpers. Choose changes based on discovery evidence and try alternatives while time remains.",
        "wall_time_seconds": args.search_seconds + args.final_reserve,
        "final_reserve_seconds": args.final_reserve,
        "max_candidates": args.max_candidates,
        "honor_author_stop": False,
        "discovery_pairs": 3,
        "evaluation_pairs": 12,
        "provider": {
            "provider": "codex-chatgpt",
            "model": args.model,
            "effort": args.effort,
            "timeout_seconds": 600,
        },
    }
    write(out / "plan.json", plan)
    result = search(config_path, plan=plan, output=out / "run")
    summary = {
        "study": protocol["study"],
        "status": result.summary["status"],
        "comparator": chosen,
        "search_report": "run/report.json",
        "search_report_sha256": sha(out / "run/report.json"),
        "baseline_audit": baseline_audit,
        "final_comparisons": {},
    }
    # A failure still gets explanatory comparisons of the same frozen candidate.
    # It can never cause another author call or change which source is selected.
    frozen_path = out / "run/frozen_winner.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        selected = out / "run/candidates" / frozen["id"] / "project"
        selected_snapshot = inventory(selected)
        if selected_snapshot["files"] != frozen["source"]["files"]:
            raise ValueError("frozen selected source changed")
        comparison_deadline = clock() + 1800
        for name, reference in variants.items():
            if name == chosen:
                report = json.loads((out / "run/evaluation/comparison.json").read_text())
                extra = {
                    "memory": result.summary.get("memory", []),
                    "memory_eligible": bool(result.summary.get("memory"))
                    and all(r["eligible"] for r in result.summary.get("memory", [])),
                    "startup_eligible": result.summary.get("startup_eligible", False),
                }
            else:
                log(f"FINAL additional comparator={name}; authoring remains closed")
                report = compare(
                    reference,
                    selected,
                    snapshots[name],
                    selected_snapshot,
                    value,
                    evaluation,
                    out / "final" / name,
                    pairs=12,
                    deadline=comparison_deadline,
                    progress=log,
                    minimum_speedup=1.02,
                )
                extra = (
                    memory_and_setup(
                        reference,
                        selected,
                        snapshots[name],
                        selected_snapshot,
                        value,
                        evaluation,
                        report,
                        out / "final-memory" / name,
                        comparison_deadline,
                    )
                    if report["status"] == "matched"
                    else {}
                )
            passed = (
                eligible(
                    report,
                    {"minimum_speedup": 1.02, "max_case_regression": 0.03, "significance": 0.05},
                    final=True,
                )
                and extra.get("startup_eligible", False)
                and extra.get("memory_eligible", False)
            )
            summary["final_comparisons"][name] = {
                **compact(report),
                **extra,
                "all_gates_passed": passed,
            }
            write(out / "summary.json", summary)
        if not all(c["all_gates_passed"] for c in summary["final_comparisons"].values()):
            summary["status"] = "reference_retained"
        summary["frozen_candidate"] = frozen["id"]
    summary["upstream_unchanged"] = not git(project, "status", "--porcelain")
    summary["comparators_unchanged"] = all(
        inventory(variants[name])["files"] == snapshots[name]["files"] for name in variants
    )
    summary["protocol_sources_unchanged"] = all(
        sha(HERE / name) == digest for name, digest in protocol["source_hashes"].items()
    )
    if not all(
        summary[k]
        for k in ("upstream_unchanged", "comparators_unchanged", "protocol_sources_unchanged")
    ):
        summary["status"] = "failed"
    summary["elapsed_seconds"] = clock() - started
    write(out / "summary.json", summary)
    log(
        f"DONE status={summary['status']} elapsed={summary['elapsed_seconds']:.1f}s result={out / 'summary.json'} log={out.with_suffix('.log')}"
    )
    return 0 if summary["status"] in {"accepted_experiment", "reference_retained"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
