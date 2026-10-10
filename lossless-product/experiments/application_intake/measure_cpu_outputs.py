"""Measure fixed optimized programs against pristine pybaselines, without search.

The main objective is final application request time. Optimization time is not
part of the score. Saved proposals are fixed before any measurements begin.
"""

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys

from lossless.applications import initialize, patches
from lossless.applications.comparison import compare, eligible, statistics_for
from lossless.applications.config import resolve
from lossless.applications.discovery import inventory
from lossless.applications.replay import copy_project
from lossless.applications.runtime import clock

from broader_study import HERE, SCOPE, compact, memory_and_setup
from current_cpu_study import workload_cases
from study import REVISION, URL, git, log, sha, write

MAIN_REVISION = "8595c915e5826065bcd9a37a5b03b9f76b84d217"
VARIANTS = ("stock_reuse", "github_main_output", "local_best_output")
POLICY = {
    "minimum_speedup": 1.02,
    "max_case_regression": 0.03,
    "max_call_regression": 0.03,
    # Two independently reported comparisons against stock; Bonferroni family .05.
    "significance": 0.025,
    "max_setup_increase_seconds": 0.1,
    "max_peak_rss_ratio": 1.2,
    "max_traced_peak_ratio": 1.2,
}


def prepare(args):
    upstream, main, out = (
        args.project.resolve(),
        args.main_checkout.resolve(),
        args.output.resolve(),
    )
    for root, expected in ((upstream, REVISION), (main, MAIN_REVISION)):
        if git(root, "rev-parse", "HEAD") != expected or git(root, "status", "--porcelain"):
            raise ValueError(f"expected clean pinned checkout: {root}")
        if out == root or out.is_relative_to(root) or root.is_relative_to(out):
            raise ValueError("output must be separate from both source checkouts")
    out.mkdir(parents=True, exist_ok=False)
    upstream_snapshot = inventory(upstream)
    stock = out / VARIANTS[0]
    copy_project(upstream, stock, upstream_snapshot)
    shutil.copyfile(HERE / "spectrum_workload.py", stock / "spectrum_workload.py")
    # New values for the three final profiles. No candidate is authored/selected here.
    cases = [c for c in workload_cases(stock, args.seed) if c["split"] == "evaluation"]
    write(out / "cases.json", cases)
    published = main / "lossless-product/experiments/application_intake/continuation-candidate.json"
    local = HERE / "cpu-control-candidate.json"
    for name in VARIANTS[1:]:
        destination = out / name
        copy_project(stock, destination, inventory(stock))
        patches.apply(destination, destination, json.loads(published.read_text()), SCOPE)
        if name == "local_best_output":
            patches.apply(destination, destination, json.loads(local.read_text()), SCOPE)
    snapshots = {name: inventory(out / name) for name in VARIANTS}
    # Prevent a repeat of the mistaken comparison against a preoptimized reference.
    if any(
        snapshots["stock_reuse"]["files"].get(n) != v for n, v in upstream_snapshot["files"].items()
    ):
        raise ValueError("stock comparator differs from the unmodified input repository")
    initialize(
        stock,
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
    config["replay"].update(timeout_seconds=30, max_output_bytes=32000000)
    write(config_path, config)
    product = Path(patches.__file__).resolve().parents[1]
    protocol = {
        "schema_version": 1,
        "experiment": "fixed-optimized-programs-vs-unmodified-input",
        "upstream": {"url": URL, "revision": REVISION},
        "github_main_revision": MAIN_REVISION,
        "variants": {
            "stock_reuse": "Unmodified pybaselines with reusable public Baseline objects and the common study wrapper. No previous optimization patches.",
            "github_main_output": "Frozen continuation candidate recorded on GitHub main, applied to stock. Experimental saved output, not a new search or automatic packaged recipe.",
            "local_best_output": "Frozen continuation plus previously qualified CPU-control candidate. Chosen from prior evidence before this experiment, not selected using these results.",
        },
        "selection": "Both outputs frozen before new-seed fixtures are measured. No authoring, repair, reselection or automatic deployment.",
        "objective": "Final program execution speed; optimization/search duration excluded from the performance score.",
        "timing_scope": "calls",
        "measurement": "Four complete in-memory requests per profile including first use, preparation, allocations, cache/build, CPU dispatch, fitting, correction and result construction. Import/factory setup separately measured. Fixture decoding, correctness checks and worker/interpreter startup excluded. Timer and completion-check bookkeeping remain inside each measured call.",
        "scope": "One M2 CPU environment, one OpenBLAS thread, three simulated-spectra profiles using a real library. Equal-case weights are not production frequencies. No GPU or whole-process startup claim.",
        "solver_used": False,
        "llm_calls": 0,
        "seed": args.seed,
        "pairs": args.pairs,
        "policy": POLICY,
        "aa_pairs": 6,
        "budget_seconds": args.budget,
        "proposal_sha256": {
            "github_main_continuation": sha(published),
            "local_cpu_control": sha(local),
        },
        "snapshots": snapshots,
        "stock_matches_upstream_files": True,
        "frozen_files": {
            str(p.relative_to(out)): sha(p)
            for p in (out / "cases.json", config_path, out / "job/cases.json")
        },
        "driver_sources": {
            n: sha(HERE / n)
            for n in (
                "measure_cpu_outputs.py",
                "current_cpu_study.py",
                "broader_study.py",
                "study.py",
                "spectrum_workload.py",
            )
        },
        "product_python_sha256": {
            str(p.relative_to(product)): sha(p) for p in product.rglob("*.py")
        },
        "packages": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "numba")},
    }
    write(out / "protocol.json", protocol)
    shutil.copyfile(Path(__file__), out / "executed_measure_cpu_outputs.py")
    return protocol


def check_frozen(out, protocol):
    for n, h in protocol["frozen_files"].items():
        if sha(out / n) != h:
            raise ValueError(f"frozen input changed: {n}")
    for n, h in protocol["driver_sources"].items():
        if sha(HERE / n) != h:
            raise ValueError(f"driver changed: {n}")
    for n, snapshot in protocol["snapshots"].items():
        if inventory(out / n)["files"] != snapshot["files"]:
            raise ValueError(f"program or fixtures changed: {n}")
    product = Path(patches.__file__).resolve().parents[1]
    if {str(p.relative_to(product)): sha(p) for p in product.rglob("*.py")} != protocol[
        "product_python_sha256"
    ]:
        raise ValueError("measurement implementation changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--main-checkout", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026101017)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--budget", type=float, default=285)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        parser.error("set OPENBLAS_NUM_THREADS=1")
    if not 6 <= args.pairs <= 40 or not math.isfinite(args.budget) or args.budget <= 0:
        parser.error("require 6..40 pairs and a finite positive time allowance")
    out = args.output.resolve()
    log(
        f"START expected=3-4min result={out / 'summary.json'} log={out.with_suffix('.log')} model_calls=0 score=program_execution"
    )
    started = clock()
    protocol = json.loads((out / "protocol.json").read_text()) if args.resume else prepare(args)
    check_frozen(out, protocol)
    if args.prepare_only:
        log(f"PREPARED result={out / 'protocol.json'} log={out.with_suffix('.log')}")
        return 0
    if (out / "measurements").exists():
        raise ValueError("measurements already exist; do not silently repeat a run")
    deadline = started + protocol["budget_seconds"]
    value, cases = resolve(out / "job/lossless.json")
    policy = protocol["policy"]
    snapshots = protocol["snapshots"]
    stock = out / "stock_reuse"
    summary = {
        "status": "incomplete",
        "objective": protocol["objective"],
        "protocol_sha256": sha(out / "protocol.json"),
        "llm_calls": 0,
        "solver_used": False,
        "comparisons": {},
    }
    try:
        aa = compare(
            stock,
            stock,
            snapshots["stock_reuse"],
            snapshots["stock_reuse"],
            value,
            cases[:1],
            out / "measurements/aa",
            pairs=protocol["aa_pairs"],
            deadline=deadline,
            progress=log,
            minimum_speedup=policy["minimum_speedup"],
        )
        summary["aa"] = compact(aa)
        if aa["status"] != "matched" or eligible(aa, policy, final=True):
            raise ValueError("A/A failed or spuriously passed the speed gate")
        for name in VARIANTS[1:]:
            report = compare(
                stock,
                out / name,
                snapshots["stock_reuse"],
                snapshots[name],
                value,
                cases,
                out / "measurements" / name,
                pairs=protocol["pairs"],
                deadline=deadline,
                progress=log,
                minimum_speedup=policy["minimum_speedup"],
            )
            extra = {}
            if report["status"] == "matched":
                extra = memory_and_setup(
                    stock,
                    out / name,
                    snapshots["stock_reuse"],
                    snapshots[name],
                    value,
                    cases,
                    report,
                    out / "memory" / name,
                    deadline,
                )
                extra["import_plus_requests_diagnostic"] = statistics_for(
                    report["cases"], policy["minimum_speedup"], timing_scope="setup_and_calls"
                )
            summary["comparisons"][name] = {
                **compact(report),
                **extra,
                "request_speed_gates_passed": eligible(report, policy, final=True),
                "all_gates_passed": eligible(report, policy, final=True)
                and extra.get("startup_eligible", False)
                and extra.get("memory_eligible", False),
            }
            write(out / "summary.json", summary)
        check_frozen(out, protocol)
        summary["sources_unchanged"] = not git(args.project.resolve(), "status", "--porcelain")
        summary["status"] = (
            "completed"
            if all(r["status"] == "matched" for r in summary["comparisons"].values())
            and summary["sources_unchanged"]
            else "failed"
        )
    except (ValueError, OSError, TimeoutError) as error:
        summary.update(status="failed", reason=str(error))
    summary["measurement_run_seconds_not_scored"] = clock() - started
    write(out / "summary.json", summary)
    log(
        f"COMPLETE status={summary['status']} result={out / 'summary.json'} log={out.with_suffix('.log')}"
    )
    return 0 if summary["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
