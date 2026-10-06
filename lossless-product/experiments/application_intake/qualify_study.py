"""Fresh qualification of frozen model proposals for declared repeated workloads.

No model calls, no hidden warmup and no reuse of discovery timing samples.
The first single-call/default-thread pilot remains a separate negative result.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import sys

from lossless.applications import initialize, search
from lossless.applications.comparison import run_case
from lossless.applications.discovery import inventory
from lossless.applications.replay import copy_project
from lossless.applications.runtime import clock

from search_study import make_cases
from study import REVISION, URL, git, log, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--authored-run", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=1800)
    args = parser.parse_args()
    if bool(args.authored_run) == bool(args.candidates):
        raise ValueError("supply an authored run or the published candidates JSON")
    project, out = args.project.resolve(), args.output.resolve()
    if git(project, "rev-parse", "HEAD") != REVISION or git(project, "status", "--porcelain"):
        raise ValueError("expected clean pinned upstream checkout")
    if out == project or out.is_relative_to(project) or project.is_relative_to(out):
        raise ValueError("output must be separate from upstream")
    out.mkdir(parents=True, exist_ok=False)
    started = clock()
    log(
        f"START qualification expected=6-9min result={out / 'summary.json'} log={out.with_suffix('.log')}"
    )
    if args.authored_run:
        proposals = [
            json.loads(p.read_text())
            for p in sorted(args.authored_run.glob("candidates/*/author/response.json"))
        ]
        for proposal in proposals:
            proposal.pop("usage", None)
    else:
        proposals = json.loads(args.candidates.read_text())
    if not 1 <= len(proposals) <= 4:
        raise ValueError("expected 1..4 frozen source proposals")
    write(out / "candidates.json", proposals)
    prepared = out / "prepared"
    copy_project(project, prepared, inventory(project))
    cases = make_cases(prepared)
    for case in cases:
        original = case["calls"]
        case["calls"] = [copy.deepcopy(original[i % len(original)]) for i in range(32)]
    write(out / "cases.json", cases)
    initialize(
        prepared,
        out / "job",
        entry="pybaselines.polynomial:modpoly",
        cases=out / "cases.json",
        python=sys.executable,
    )
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    config = out / "job/lossless.json"
    value = json.loads(config.read_text())
    value["environment"]["inherit_env"] = ["OPENBLAS_NUM_THREADS"]
    write(config, value)
    plan = {
        "schema_version": 1,
        "source_scope": [{"path": "pybaselines/polynomial.py", "symbol": "_Polynomial.modpoly"}],
        "contract": "pybaselines 1.2.1 public polynomial.modpoly, already-running application, explicit 32-call sequences. All 32 calls including the first are timed, and earlier results retained. Same finite real float32/64 domain/branches as search_study.py; exact baseline, weights, coefficients, tolerance history, input mutation and retained results. OPENBLAS_NUM_THREADS=1 for both the original callable comparator and candidates. This is a NEW qualification protocol after noisy single-call/default-thread pilot, not a successful rerun of its contract. Does not qualify single-call/other thread settings, arbitrary aliasing/exceptions/concurrency, or end-to-end process latency. No parameter, weights, precision, entry or evaluator changes. Original default-thread reference is checked for bitwise agreement after one winner is frozen; no final feedback is used for tuning.",
        "candidates": proposals,
        "max_candidates": len(proposals),
        "discovery_pairs": 6,
        "evaluation_pairs": 12,
        "wall_time_seconds": args.budget - 120,
        "final_reserve_seconds": 600,
    }
    write(out / "plan.json", plan)
    summary = {
        "study": "application-repeated-qualification-pybaselines",
        "upstream": {"url": URL, "revision": REVISION, "license": "BSD-3-Clause"},
        "protocol": {
            "calls_per_case": 32,
            "OPENBLAS_NUM_THREADS": "1",
            "default_thread_reference": "8 (separate final correctness check)",
            "discovery_pairs": 6,
            "evaluation_pairs": 12,
            "budget_seconds": args.budget,
            "no_new_authoring": True,
        },
        "study_sha256": sha(Path(__file__)),
        "case_generator_sha256": sha(Path(__file__).with_name("search_study.py")),
        "candidates_sha256": sha(out / "candidates.json"),
        "cases_sha256": sha(out / "cases.json"),
        "plan_sha256": sha(out / "plan.json"),
    }
    # Write the whole protocol, including the default-thread check, before search.
    write(out / "study.json", summary)
    result = search(config, plan=plan, output=out / "run")
    summary.update(
        status=result.summary["status"],
        selected=result.summary["selected"],
        search_report_sha256=sha(result.directory / "report.json"),
    )
    if result.summary["status"] == "accepted_experiment":
        frozen_value = json.loads((result.directory / "resolved.json").read_text())
        frozen_cases = json.loads((result.directory / "cases.json").read_text())
        source = json.loads((result.directory / "source.json").read_text())
        comparisons = [
            json.loads(
                (
                    result.directory
                    / "candidates"
                    / result.summary["selected"]
                    / "discovery/comparison.json"
                ).read_text()
            ),
            json.loads((result.directory / "evaluation/comparison.json").read_text()),
        ]
        expected = {
            c["id"]: c["pairs"][0]["reference"]["fingerprint"]
            for comparison in comparisons
            for c in comparison["cases"]
        }
        os.environ["OPENBLAS_NUM_THREADS"] = "8"
        controls = []
        for case in frozen_cases:
            log(f"DEFAULT-THREAD correctness case={case['id']}")
            observed = run_case(
                result.directory / "reference",
                source,
                frozen_value,
                case,
                out / "default-thread" / case["id"],
                started + args.budget,
                log,
            )
            controls.append(
                {
                    "id": case["id"],
                    "matched": observed["fingerprint"] == expected[case["id"]],
                    "observed": observed,
                }
            )
        summary["default_thread_correctness"] = controls
        if not all(c["matched"] for c in controls):
            summary["status"] = "default_thread_bits_differ"
    summary["upstream_unchanged"] = not git(project, "status", "--porcelain")
    summary["elapsed_seconds"] = clock() - started
    write(out / "summary.json", summary)
    log(
        f"DONE status={summary['status']} elapsed={summary['elapsed_seconds']:.1f}s result={out / 'summary.json'} log={out.with_suffix('.log')}"
    )
    return 0 if summary["status"] in {"accepted_experiment", "reference_retained"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
