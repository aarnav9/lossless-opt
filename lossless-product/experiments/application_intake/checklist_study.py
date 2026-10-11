"""Matched CPU-checklist pilot on an already optimized external application.

Two sequential author arms receive the same total/proposal/evaluation caps and
frozen baseline. Additional diagnostics and structured reviews are part of the
treatment, including their time/token cost. No population-level efficacy claim
from one pair. Run more independent pairs with --replicates.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys

from lossless.applications import initialize, search
from lossless.applications.discovery import inventory
from lossless.applications import patches
from lossless.applications.replay import copy_project
from lossless.applications.runtime import clock

from broader_study import HERE, SCOPE, application_case, new_records
from study import REVISION, URL, git, log, sha, write


def make_cases(project, replicate):
    (project / "lossless_fixtures").mkdir(exist_ok=True)
    specifications = [
        ("dev_small", "discovery", 281, 12, {}, "peaks"),
        ("dev_large", "discovery", 4481, 8, {"poly_order": 4, "return_coef": True}, "steps"),
        ("dev_weights", "discovery", 1103, 10, {"weight_mode": "changing"}, "oscillatory"),
        ("dev_orders", "discovery", 809, 10, {"varying_order": True, "return_coef": True}, "peaks"),
        (
            "dev_mask",
            "discovery",
            1471,
            8,
            {"mask_initial_peaks": True, "use_original": True},
            "peaks",
        ),
        ("dev_flat", "discovery", 397, 10, {"max_iter": 13}, "flat"),
        ("closed_small", "evaluation", 313, 14, {"poly_order": 2}, "oscillatory"),
        ("closed_large", "evaluation", 5237, 8, {"poly_order": 5, "return_coef": True}, "peaks"),
        (
            "closed_weights",
            "evaluation",
            1423,
            10,
            {"weight_mode": "changing", "return_coef": True},
            "steps",
        ),
        ("closed_shared", "evaluation", 1877, 12, {"weight_mode": "shared"}, "oscillatory"),
        (
            "closed_order",
            "evaluation",
            947,
            12,
            {"varying_order": True, "return_coef": True},
            "steps",
        ),
        ("closed_mask", "evaluation", 2161, 8, {"mask_initial_peaks": True}, "peaks"),
        (
            "closed_layout",
            "evaluation",
            1171,
            10,
            {"ordering": "shuffled", "irregular": True, "dtype": "float32"},
            "peaks",
        ),
        ("closed_flat", "evaluation", 433, 12, {"max_iter": 11}, "flat"),
    ]
    return [
        application_case(
            name,
            split,
            new_records(
                project,
                name,
                size,
                batch,
                100000 + replicate * 10000 + i * 100,
                options,
                pattern=pattern,
            ),
            requests=2,
        )
        for i, (name, split, size, batch, options, pattern) in enumerate(specifications)
    ]


def plan_for(assisted, args):
    return {
        "schema_version": 1,
        "source_scope": copy.deepcopy(SCOPE),
        "context_symbols": [{"path": "pybaselines/utils.py", "symbol": "relative_difference"}],
        "contract": (
            "CPU spectrum-processing application using pinned pybaselines 1.2.1 reusable Baseline objects "
            "and the retained continuation candidate. This already-optimized experimental source is the "
            "frozen comparator for BOTH arms. Domain: finite real float32/float64 one-dimensional spectra, "
            "valid x/order/weights, optional masking, use_original and return_coef, changing grids, "
            "weights and orders, flat/zero signals and repeated requests. Preserve all exact baseline, "
            "corrected, weights, coefficient and tol_history arrays, input mutation and retained output "
            "ownership. Preserve tol/max_iter and exact rounded expression trees. At most four cached "
            "fitters and 16 MiB counted persistent state. Source changes within the listed bodies only; "
            "no new dependencies, changed observer or benchmark detection. OpenBLAS remains one thread; "
            "no environment changes. Ordinary complete Processor requests include setup/cache lookup, "
            "fitting, correction and result construction; imports are reported separately. No claims "
            "about invalid-input exceptions, concurrent use, other algorithms or unrestricted layouts. "
            "Equal case weights are an experiment choice. Finite validation is not universal proof."
        ),
        "architecture": assisted,
        "wall_time_seconds": args.budget,
        "final_reserve_seconds": args.final_reserve,
        "max_candidates": args.max_candidates,
        "honor_author_stop": False,
        "discovery_pairs": 3,
        "evaluation_pairs": 12,
        "minimum_speedup": 1.02,
        "max_case_regression": 0.03,
        "significance": 0.05,
        "max_setup_increase_seconds": 0.1,
        "max_peak_rss_ratio": 1.2,
        "max_traced_peak_ratio": 1.2,
        "provider": {
            "provider": "codex-chatgpt",
            "model": args.model,
            "effort": args.effort,
            "timeout_seconds": args.budget,
        },
    }


def prepare(args):
    project, out = args.project.resolve(), args.output.resolve()
    if git(project, "rev-parse", "HEAD") != REVISION or git(project, "status", "--porcelain"):
        raise ValueError("expected a clean pinned pybaselines v1.2.1 checkout")
    if project == out or project.is_relative_to(out) or out.is_relative_to(project):
        raise ValueError("output must be separate from upstream")
    out.mkdir(parents=True, exist_ok=False)
    schedule, files, baselines = [], {}, {}
    for replicate in range(args.replicates):
        folder = out / f"pair_{replicate + 1:02d}"
        folder.mkdir()
        baseline = folder / "baseline"
        copy_project(project, baseline, inventory(project))
        shutil.copyfile(HERE / "spectrum_workload.py", baseline / "spectrum_workload.py")
        # Apply the previous strong experimental implementation before either author.
        candidate = json.loads((HERE / "continuation-candidate.json").read_text())
        patches.apply(baseline, baseline, candidate, SCOPE)
        write(folder / "cases.json", make_cases(baseline, replicate))
        baselines[str(baseline.relative_to(out))] = inventory(baseline)["files"]
        initialize(
            baseline,
            folder / "job",
            entry="spectrum_workload:make_processor",
            factory=True,
            cases=folder / "cases.json",
            python=sys.executable,
        )
        config_path = folder / "job/lossless.json"
        config = json.loads(config_path.read_text())
        config["environment"]["inherit_env"] = ["OPENBLAS_NUM_THREADS"]
        config["entry"]["observe"] = "spectrum_workload:observe_processor"
        config["replay"].update(timeout_seconds=60, max_output_bytes=32000000)
        write(config_path, config)
        order = ["control", "checklist"]
        random.Random(args.order_seed + replicate).shuffle(order)
        for arm in order:
            plan_path = folder / (arm + "-plan.json")
            write(plan_path, plan_for(arm == "checklist", args))
            schedule.append(
                {
                    "pair": replicate + 1,
                    "arm": arm,
                    "config": str(config_path.relative_to(out)),
                    "plan": str(plan_path.relative_to(out)),
                    "output": str((folder / arm).relative_to(out)),
                }
            )
            files[str(plan_path.relative_to(out))] = sha(plan_path)
        for path in (config_path, folder / "cases.json", folder / "job/cases.json"):
            files[str(path.relative_to(out))] = sha(path)
    protocol = {
        "study": "cpu-checklist-application-pilot",
        "schema_version": 1,
        "upstream": {"url": URL, "revision": REVISION, "license": "BSD-3-Clause"},
        "seed_candidate_sha256": sha(HERE / "continuation-candidate.json"),
        "schedule": schedule,
        "baselines": baselines,
        "frozen_files": files,
        "source_hashes": {
            str(p.relative_to(HERE)): sha(p)
            for p in (Path(__file__), HERE / "broader_study.py", HERE / "spectrum_workload.py")
        },
        "runner_hashes": {p.name: sha(p) for p in Path(patches.__file__).parent.glob("*.py")},
        "arm_budget_seconds": args.budget,
        "final_reserve_seconds": args.final_reserve,
        "comparison": "Joint effect of hardware/library/layout diagnostics, sourced catalog and mandatory structured reviews versus prior author context; not an isolated retrieval ablation.",
        "selection": "Each arm independently freezes its highest discovery-eligible candidate before final evaluation. No cross-arm proposal or result sharing; no final-case reselection.",
        "outcomes": "Accepted final speedup, reference retention, per-case regressions, invalid/duplicate proposals, candidate/call counts, elapsed search/total/provider time, extra collection cost, tokens and unknown dollar spend.",
        "limitations": "Sequential fixed-cap pilot on one already-optimized application; stochastic author output and hardware drift remain. One pair cannot establish general efficacy. No manual-author substitution.",
    }
    write(out / "protocol.json", protocol)
    return protocol


def arm_summary(summary):
    candidates = summary.get("candidates", [])
    usage = [u for c in candidates for u in c["provider_receipt"].get("usage", [])]
    counts = {
        state: sum(c["status"] == state for c in candidates)
        for state in sorted({c["status"] for c in candidates})
    }
    evaluation = summary.get("evaluation", {})
    stats = evaluation.get("statistics", {})
    return {
        "status": summary["status"],
        "selected": summary["selected"],
        "search_stop_reason": summary.get("search_stop_reason"),
        "reason": summary.get("reason"),
        "candidate_status_counts": counts,
        "llm_calls": summary["llm_calls"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "search_seconds": summary.get("search_seconds"),
        "provider_seconds": summary["provider_seconds"],
        "architecture_collection_seconds": summary.get("architecture_collection_seconds", 0),
        "accepted_speedup": stats.get("case_balanced_geomean_speedup")
        if summary["status"] == "accepted_experiment"
        else 1.0,
        "final_evaluation": evaluation,
        "startup_eligible": summary.get("startup_eligible"),
        "memory": summary.get("memory"),
        "coverage": summary.get("checklist_coverage"),
        "tokens": {
            k: sum(u[k] for u in usage)
            if usage and all(isinstance(u.get(k), int) for u in usage)
            else None
            for k in (
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
            )
        },
        "cost_usd": None,
        "token_scope": "Reported completed turns only; missing timeout/provider usage remains unknown.",
        "provider_messages": [
            m for c in candidates for m in c["provider_receipt"].get("provider_messages", [])
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=1800)
    parser.add_argument("--final-reserve", type=float, default=600)
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--order-seed", type=int, default=20261008)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="execute already prepared protocol; never rerun a completed arm",
    )
    args = parser.parse_args()
    if (
        not 1 <= args.replicates <= 10
        or not math.isfinite(args.budget)
        or args.budget <= args.final_reserve
    ):
        raise ValueError(
            "positive bounded replicate count and budget greater than final reserve required"
        )
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise ValueError("set OPENBLAS_NUM_THREADS=1 for both arms")
    out = args.output.resolve()
    started = clock()
    protocol = json.loads((out / "protocol.json").read_text()) if args.resume else prepare(args)
    log(
        f"START expected=30-60min per pair max_author_and_validation={len(protocol['schedule']) * protocol['arm_budget_seconds']}s result={out / 'summary.json'} log={out.with_suffix('.log')}"
    )
    if args.prepare_only:
        log(
            f"PREPARED result={out / 'protocol.json'} log={out.with_suffix('.log')}; no author calls"
        )
        return 0
    summary = {"protocol_sha256": sha(out / "protocol.json"), "status": "running", "arms": []}
    for run in protocol["schedule"]:
        if any(sha(out / name) != expected for name, expected in protocol["frozen_files"].items()):
            raise ValueError("frozen study files changed")
        if any(
            sha(HERE / name) != expected for name, expected in protocol["source_hashes"].items()
        ):
            raise ValueError("study source changed after freezing")
        if any(
            inventory(out / name)["files"] != expected
            for name, expected in protocol["baselines"].items()
        ):
            raise ValueError("frozen application baseline changed")
        if {p.name: sha(p) for p in Path(patches.__file__).parent.glob("*.py")} != protocol[
            "runner_hashes"
        ]:
            raise ValueError("application runner changed after protocol was frozen")
        destination = out / run["output"]
        if destination.exists():
            report = json.loads((destination / "report.json").read_text())
            if report["status"] not in {"accepted_experiment", "reference_retained"}:
                raise ValueError("cannot silently rerun an incomplete/failed arm")
        else:
            log(f"ARM pair={run['pair']} treatment={run['arm']}")
            result = search(out / run["config"], plan=out / run["plan"], output=destination)
            report = result.summary
        summary["arms"].append({"pair": run["pair"], "arm": run["arm"], **arm_summary(report)})
        write(out / "summary.json", summary)
        if report["status"] not in {"accepted_experiment", "reference_retained"}:
            summary["status"] = "incomplete"
            break
    else:
        summary["status"] = "completed"
    summary["elapsed_seconds_this_invocation"] = clock() - started
    summary["scope"] = protocol["limitations"]
    write(out / "summary.json", summary)
    log(
        f"DONE status={summary['status']} result={out / 'summary.json'} log={out.with_suffix('.log')}"
    )
    return 0 if summary["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
