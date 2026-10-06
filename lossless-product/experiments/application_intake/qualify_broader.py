"""Recheck one frozen broader-study proposal against all four baselines, no LLM.

Uses the published study case generator, so this is reproduction on known cases,
not a new independently unseen-workload claim. Never searches or selects source.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from lossless.applications import patches
from lossless.applications.comparison import compare, eligible
from lossless.applications.config import resolve
from lossless.applications.discovery import inventory
from lossless.applications.replay import copy_project
from lossless.applications.runtime import clock

from broader_study import SCOPE, compact, memory_and_setup
from study import git, log, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--comparator",
        choices=["public_stock", "public_previous", "reuse_stock", "reuse_previous"],
        default="reuse_previous",
    )
    parser.add_argument("--budget", type=float, default=1800)
    parser.add_argument("--pairs", type=int, default=12)
    args = parser.parse_args()
    if not 6 <= args.pairs <= 40 or not 0 < args.budget <= 14400:
        raise ValueError("require 6..40 final pairs and a positive budget <= four hours")
    candidate_file = args.candidate.resolve()
    proposal_hash = sha(candidate_file)
    proposal = json.loads(candidate_file.read_text())
    proposal.pop("usage", None)
    patches.validate(proposal, SCOPE)
    root = args.output.resolve()
    started = clock()
    deadline = started + args.budget
    log(
        f"START frozen qualification expected=15-25min no_model_calls=True result={root / 'qualification.json'} log={root.with_suffix('.log')}"
    )
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # Reuse exactly the published setup path; --prepare-only exits before any
    # baseline timing, model authentication, authoring or source selection.
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("broader_study.py")),
            "--project",
            str(args.project.resolve()),
            "--output",
            str(root),
            "--prepare-only",
        ],
        check=True,
        timeout=min(120, args.budget),
    )
    value, cases = resolve(root / "job/lossless.json")
    cases = [case for case in cases if case["split"] == "evaluation"]
    names = ["public_stock", "public_previous", "reuse_stock", "reuse_previous"]
    variants = {name: root / "baselines" / name for name in names}
    snapshots = {name: inventory(project) for name, project in variants.items()}
    reference = variants[args.comparator]
    selected = root / "selected"
    copy_project(reference, selected, snapshots[args.comparator])
    difference, candidate_identity = patches.apply(reference, selected, proposal, SCOPE)
    (root / "candidate.patch").write_text(difference)
    candidate_snapshot = inventory(selected)
    write(root / "candidate.json", proposal)
    write(root / "candidate-source.json", candidate_snapshot)
    frozen = {
        "candidate_sha256": proposal_hash,
        "candidate_identity": candidate_identity,
        "applied_to": args.comparator,
        "cases_sha256": sha(root / "cases.json"),
        "pairs": args.pairs,
        "minimum_speedup": 1.02,
        "max_case_regression": 0.03,
        "significance": 0.05,
        "llm_calls": 0,
        "scope": "Remeasure one frozen source proposal on published cases; no authoring, source selection or unseen-data claim. Full request, setup and memory gates against all four comparators.",
    }
    write(root / "qualification-protocol.json", frozen)
    summary = {"status": "incomplete", "protocol": frozen, "comparisons": {}}
    try:
        for name in names:
            report = compare(
                variants[name],
                selected,
                snapshots[name],
                candidate_snapshot,
                value,
                cases,
                root / "qualification" / name,
                pairs=args.pairs,
                deadline=deadline,
                progress=log,
                minimum_speedup=1.02,
            )
            extra = (
                memory_and_setup(
                    variants[name],
                    selected,
                    snapshots[name],
                    candidate_snapshot,
                    value,
                    cases,
                    report,
                    root / "qualification-memory" / name,
                    deadline,
                )
                if report["status"] == "matched"
                else {}
            )
            passed = (
                eligible(report, frozen, final=True)
                and extra.get("startup_eligible", False)
                and extra.get("memory_eligible", False)
            )
            summary["comparisons"][name] = {**compact(report), **extra, "all_gates_passed": passed}
            write(root / "qualification.json", summary)
        summary["status"] = (
            "accepted_experiment"
            if all(row["all_gates_passed"] for row in summary["comparisons"].values())
            else "reference_retained"
        )
    except (ValueError, OSError, TimeoutError) as error:
        summary.update(status="failed", reason=str(error))
    summary["sources_unchanged"] = (
        sha(candidate_file) == proposal_hash
        and inventory(selected)["files"] == candidate_snapshot["files"]
        and all(inventory(variants[name])["files"] == snapshots[name]["files"] for name in names)
        and sha(root / "cases.json") == frozen["cases_sha256"]
        and not git(args.project.resolve(), "status", "--porcelain")
    )
    summary["elapsed_seconds"] = clock() - started
    if not summary["sources_unchanged"] or clock() > deadline:
        summary.update(status="failed", reason="provenance changed or total deadline exhausted")
    write(root / "qualification.json", summary)
    log(
        f"DONE status={summary['status']} elapsed={summary['elapsed_seconds']:.1f}s result={root / 'qualification.json'} log={root.with_suffix('.log')}"
    )
    return 0 if summary["status"] in {"accepted_experiment", "reference_retained"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
