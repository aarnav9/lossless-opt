"""Selected checkout's application search versus a retained strong CPU baseline.

No solver, new optimizer behavior, manual candidate, or changed acceptance gate.
Small bounded live pilot; longer replications are separate foreground runs.
"""

import argparse
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys

from lossless.applications import initialize, search
from lossless.applications import patches
from lossless.applications.comparison import compare, eligible
from lossless.applications.config import resolve
from lossless.applications.discovery import inventory
from lossless.applications.replay import copy_project
from lossless.applications.runtime import clock

from broader_study import HERE, SCOPE, new_records
from study import REVISION, URL, git, log, sha, write


def workload_cases(project, seed):
    """Fresh signal values on every request, with reused and changing fitters."""
    (project / "lossless_fixtures").mkdir(exist_ok=True)
    specifications = [
        ("dev_stream", "discovery", 1021, 12, {}, "peaks"),
        (
            "dev_weights",
            "discovery",
            2047,
            8,
            {"weight_mode": "changing", "return_coef": True},
            "oscillatory",
        ),
        (
            "dev_branches",
            "discovery",
            509,
            6,
            {"varying_order": True, "use_original": True},
            "steps",
        ),
        ("final_stream", "evaluation", 3079, 10, {"return_coef": True}, "oscillatory"),
        (
            "final_weights",
            "evaluation",
            1483,
            8,
            {"weight_mode": "changing", "varying_order": True, "return_coef": True},
            "peaks",
        ),
        (
            "final_small_mixed",
            "evaluation",
            347,
            1,
            {"dtype": "float32", "use_original": True},
            "steps",
        ),
    ]
    result = []
    for index, (name, split, size, batch, options, pattern) in enumerate(specifications):
        calls = []
        for request in range(4):
            selected = dict(options)
            selected_pattern = pattern
            if name in {"dev_branches", "final_small_mixed"}:
                if request == 1:
                    selected["mask_initial_peaks"] = True
                elif request == 2:
                    selected["ordering"] = "shuffled"
                    selected["irregular"] = True
                elif request == 3:
                    selected_pattern = "flat"
            records = new_records(
                project,
                f"{name}_request{request}",
                size,
                batch,
                seed + index * 10000 + request * 100,
                selected,
                pattern=selected_pattern,
            )
            calls.append({"args": [records]})
        result.append(dict(id=name, split=split, calls=calls))
    return result


def plan_for(args):
    return {
        "schema_version": 1,
        "source_scope": copy.deepcopy(SCOPE),
        "context_symbols": [{"path": "pybaselines/utils.py", "symbol": "relative_difference"}],
        "contract": (
            "CPU streaming spectrum correction with pinned pybaselines 1.2.1. The frozen comparator "
            "uses reusable Baseline objects, the continuation candidate, and the stronger previously "
            "qualified CPU-control patch. These are explicitly chosen experimental source baselines, "
            "not promoted deployment recipes. Ordinary complete Processor requests include preparation, "
            "cache lookup/build, fitting, correction and result construction. Each request supplies "
            "new signal values; four requests exercise first use and subsequent reuse. Domain: finite "
            "1D float32/float64 spectra, valid x/order/weights, changing weights/orders, optional mask, "
            "use_original, return_coef, shuffled/irregular grids and flat/zero signals. Preserve exact "
            "baseline, corrected, weights, coefficients and tolerance-history arrays, inputs and retained "
            "earlier outputs. Preserve stopping thresholds, iteration semantics and floating-point "
            "expression order. At most four cached fitters and 16 MiB counted retained state. Only the "
            "listed bodies may change; no new dependencies, observer changes, environment changes or "
            "benchmark detection. One OpenBLAS thread for both versions. Timing includes every request, "
            "including first use; module import/factory setup is measured separately. Fixture decoding "
            "and correctness inspection are excluded. No production frequency, fresh-process latency, "
            "GPU, arbitrary-input or universal-equivalence claim. These are simulated spectra around a "
            "real scientific library. Evaluation stays closed until one source winner is frozen."
        ),
        # Architecture, early-stop, timing and all acceptance thresholds retain
        # the normal product defaults. This study changes only workload/budgets.
        "wall_time_seconds": args.budget,
        "final_reserve_seconds": args.final_reserve,
        "max_candidates": args.max_candidates,
        "provider": {
            "provider": "codex-chatgpt",
            "model": args.model,
            "effort": args.effort,
            "timeout_seconds": args.provider_timeout,
        },
    }


def prepare(args):
    project, out = args.project.resolve(), args.output.resolve()
    if git(project, "rev-parse", "HEAD") != REVISION or git(project, "status", "--porcelain"):
        raise ValueError("expected clean pinned pybaselines v1.2.1 checkout")
    if project == out or project.is_relative_to(out) or out.is_relative_to(project):
        raise ValueError("output must be separate from upstream")
    out.mkdir(parents=True, exist_ok=False)
    prepared = out / "baseline"
    copy_project(project, prepared, inventory(project))
    shutil.copyfile(HERE / "spectrum_workload.py", prepared / "spectrum_workload.py")
    write(out / "cases.json", workload_cases(prepared, args.seed))
    copy_project(prepared, out / "stock_reuse", inventory(prepared))
    for name in ("continuation-candidate.json", "cpu-control-candidate.json"):
        proposal = json.loads((HERE / name).read_text())
        patches.apply(prepared, prepared, proposal, SCOPE)
    initialize(
        prepared,
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
    write(out / "plan.json", plan_for(args))
    product = Path(patches.__file__).resolve().parents[1]
    product_hashes = {str(p.relative_to(product)): sha(p) for p in product.rglob("*.py")}
    protocol = {
        "schema_version": 1,
        "experiment": "cpu-application-no-solver",
        "upstream": {"url": URL, "revision": REVISION, "license": "BSD-3-Clause"},
        "optimizer": "Normal lossless.applications.search from the selected source checkout; its default context, early stop, timing and acceptance gates. Not a published-wheel claim.",
        "optimizer_source": {
            "label": args.optimizer_label,
            "python_root": str(product),
            "checkout_commit": git(product, "rev-parse", "HEAD"),
            "python_tree_status": git(product, "status", "--porcelain", "--", "."),
        },
        "solver_used": False,
        "baseline": "Reusable public Baseline API plus the continuation and CPU-control frozen patches. Strongest previously qualified implementation in that comparison, not a claim of globally optimal CPU code.",
        "baseline_patches": {
            n: sha(HERE / n) for n in ("continuation-candidate.json", "cpu-control-candidate.json")
        },
        "case_design": "Three discovery and three held-out profiles, four complete requests each, new signals per request. Equal-case weights are declared experimental weights, not measured traffic frequencies.",
        "scope": "In-memory simulated spectral processing using a real library. No GPU, file decoding, production trace or provider-efficiency generalization. Short budget may miss later improvements.",
        "controls": "Before authoring, validate the fixed strong baseline against stock reusable API on discovery only, then run identical-source A/A. Neither control chooses or modifies baseline, candidates or thresholds.",
        "preflight_budget_seconds": args.preflight_budget,
        "preflight_aa_pairs": 6,
        "preparation_seconds": None,
        "product_python_sha256": product_hashes,
        "sources": {
            n: sha(HERE / n)
            for n in (
                "current_cpu_study.py",
                "broader_study.py",
                "study.py",
                "spectrum_workload.py",
            )
        },
        "frozen_files": {
            str(p.relative_to(out)): sha(p)
            for p in (out / "cases.json", out / "plan.json", config_path, out / "job/cases.json")
        },
        "baseline_snapshot": inventory(prepared)["files"],
        "stock_snapshot": inventory(out / "stock_reuse")["files"],
        "environment": {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "numba")
        },
        "OPENBLAS_NUM_THREADS": "1",
    }
    write(out / "protocol.json", protocol)
    shutil.copyfile(Path(__file__), out / "executed_current_cpu_study.py")
    return protocol


def check_frozen(out, protocol):
    for name, expected in protocol["frozen_files"].items():
        if sha(out / name) != expected:
            raise ValueError(f"frozen experiment file changed: {name}")
    for name, expected in protocol["sources"].items():
        if sha(HERE / name) != expected:
            raise ValueError(f"experiment source changed: {name}")
    product = Path(patches.__file__).resolve().parents[1]
    if {str(p.relative_to(product)): sha(p) for p in product.rglob("*.py")} != protocol[
        "product_python_sha256"
    ]:
        raise ValueError("optimizer source changed after preparation")
    for name, key in (("baseline", "baseline_snapshot"), ("stock_reuse", "stock_snapshot")):
        if inventory(out / name)["files"] != protocol[key]:
            raise ValueError(f"frozen {name} changed")


def preflight(out, protocol):
    value, records = resolve(out / "job/lossless.json")
    discovery = [r for r in records if r["split"] == "discovery"]
    deadline = clock() + protocol["preflight_budget_seconds"]
    started = clock()
    root = out / "baseline"
    snap = inventory(root)
    stock = out / "stock_reuse"
    audit = compare(
        stock,
        root,
        inventory(stock),
        snap,
        value,
        discovery,
        out / "preflight/stock_vs_strong",
        pairs=2,
        deadline=deadline,
        progress=log,
        minimum_speedup=1.02,
    )
    if audit["status"] != "matched":
        raise ValueError("fixed strong baseline failed discovery equality against stock reuse")
    aa = compare(
        root,
        root,
        snap,
        snap,
        value,
        discovery[:1],
        out / "preflight/aa",
        pairs=protocol["preflight_aa_pairs"],
        deadline=deadline,
        progress=log,
        minimum_speedup=1.02,
    )
    from lossless.applications.search import resolve_plan

    policy = resolve_plan(out / "plan.json", None)
    false_acceptance = eligible(aa, policy, final=True)
    result = {
        "elapsed_seconds": clock() - started,
        "stock_vs_strong": {k: v for k, v in audit.items() if k != "cases"},
        "aa": {k: v for k, v in aa.items() if k != "cases"},
        "aa_false_acceptance": false_acceptance,
        "status": "passed" if aa["status"] == "matched" and not false_acceptance else "failed",
    }
    write(out / "preflight.json", result)
    if result["status"] != "passed":
        raise ValueError(
            "A/A was incomplete, unequal or spuriously passed the speed gate; no author run"
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--budget", type=float, default=240)
    p.add_argument("--final-reserve", type=float, default=90)
    p.add_argument("--preflight-budget", type=float, default=45)
    p.add_argument("--max-candidates", type=int, default=2)
    p.add_argument("--provider-timeout", type=float, default=90)
    p.add_argument("--optimizer-label", default="local-checkout")
    p.add_argument("--model", default="gpt-6-astra")
    p.add_argument("--effort", default="medium")
    p.add_argument("--seed", type=int, default=20261010)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        p.error("set OPENBLAS_NUM_THREADS=1 for the frozen CPU environment")
    if (
        any(
            not math.isfinite(v) or v <= 0
            for v in (args.budget, args.final_reserve, args.preflight_budget, args.provider_timeout)
        )
        or args.budget <= args.final_reserve
    ):
        p.error("finite positive budgets required, with final reserve smaller than total")
    if not 1 <= args.max_candidates <= 256:
        p.error("require 1..256 candidates")
    out = args.output.resolve()
    started = clock()
    protocol = json.loads((out / "protocol.json").read_text()) if args.resume else prepare(args)
    if not args.resume:
        protocol["preparation_seconds"] = clock() - started
        write(out / "protocol.json", protocol)
    log(
        f"START expected=3-5min with default caps result={out / 'summary.json'} log={out.with_suffix('.log')} no_solver=True"
    )
    check_frozen(out, protocol)
    if args.prepare_only:
        log(
            f"PREPARED result={out / 'protocol.json'} log={out.with_suffix('.log')}; no model calls"
        )
        return 0
    controls = (
        json.loads((out / "preflight.json").read_text())
        if (out / "preflight.json").exists()
        else preflight(out, protocol)
    )
    if controls["status"] != "passed":
        raise ValueError("preflight did not pass")
    if args.preflight_only:
        log(
            f"PREFLIGHT_COMPLETE result={out / 'preflight.json'} log={out.with_suffix('.log')}; no model calls"
        )
        return 0
    if (out / "run").exists():
        raise ValueError("run already exists; do not silently repeat an incomplete experiment")
    check_frozen(out, protocol)
    result = search(out / "job/lossless.json", plan=out / "plan.json", output=out / "run")
    report = result.summary
    total = (
        protocol["preparation_seconds"] + controls["elapsed_seconds"] + report["elapsed_seconds"]
    )
    summary = {
        "status": report["status"],
        "solver_used": False,
        "protocol_sha256": sha(out / "protocol.json"),
        "report_sha256": sha(out / "run/report.json"),
        "selected": report["selected"],
        "llm_calls": report["llm_calls"],
        "search_and_validation_seconds": report["elapsed_seconds"],
        "total_experiment_seconds": total,
        "preflight": controls,
        "evaluation": report.get("evaluation"),
        "reason": report.get("reason"),
        "payback_search_only": report.get("payback"),
        "original_upstream_unchanged": not git(args.project.resolve(), "status", "--porcelain"),
        "scope": protocol["scope"],
    }
    write(out / "summary.json", summary)
    log(
        f"COMPLETE status={report['status']} result={out / 'summary.json'} log={out.with_suffix('.log')}"
    )
    return 0 if report["status"] in {"accepted_experiment", "reference_retained"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
