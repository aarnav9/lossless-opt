"""Remeasure a frozen continuation proposal, with no model or source selection.

Published final cases are known on reproduction. This is not new unseen evidence.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from lossless.applications import initialize, patches
from lossless.applications.comparison import compare, eligible
from lossless.applications.config import resolve
from lossless.applications.discovery import inventory
from lossless.applications.replay import copy_project
from lossless.applications.runtime import clock

from broader_study import HERE, SCOPE, compact, memory_and_setup
from continue_broader import BASELINES, POLICY, make_cases
from study import REVISION, git, log, sha, write


def prepare(project, out):
    if git(project, "rev-parse", "HEAD") != REVISION or git(project, "status", "--porcelain"):
        raise ValueError("expected clean pinned upstream checkout")
    if out == project or out.is_relative_to(project) or project.is_relative_to(out):
        raise ValueError("output must be separate from upstream")
    out.mkdir(parents=True, exist_ok=False)
    base = out / "prepared"
    copy_project(project, base, inventory(project))
    shutil.copyfile(HERE / "spectrum_workload.py", base / "spectrum_workload.py")
    write(out / "cases.json", make_cases(base))
    source = (base / "spectrum_workload.py").read_text()
    previous = json.loads((HERE / "frozen-candidates.json").read_text())[2]
    prior_broader = json.loads((HERE / "broader-candidate.json").read_text())
    variants = {}
    for name in (*BASELINES, "broader_previous"):
        path = out / "baselines" / name
        copy_project(base, path, inventory(base))
        if name.startswith("public"):
            (path / "spectrum_workload.py").write_text(
                source.replace('MODE = "reuse"', 'MODE = "public"')
            )
        if name.endswith("previous"):
            patches.apply(base, path, previous, [SCOPE[4]])
        if name == "broader_previous":
            patches.apply(path, path, prior_broader, SCOPE)
        variants[name] = path
    initialize(
        variants["reuse_previous"],
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
    config["replay"].update(timeout_seconds=60, max_output_bytes=32000000)
    write(config_path, config)
    value, cases = resolve(config_path)
    return variants, value, [c for c in cases if c["split"] == "evaluation"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=2400)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.budget <= 14400 or not 6 <= args.pairs <= 40:
        raise ValueError("require positive budget<=4h and 6..40 pairs")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    started = clock()
    deadline = started + args.budget
    out, project, candidate_file = (
        args.output.resolve(),
        args.project.resolve(),
        args.candidate.resolve(),
    )
    log(
        f"START expected=20-30min no_model_calls=True result={out / 'qualification.json'} log={out.with_suffix('.log')}"
    )
    proposal_hash = sha(candidate_file)
    proposal = json.loads(candidate_file.read_text())
    patches.validate(proposal, SCOPE)
    variants, value, cases = prepare(project, out)
    snapshots = {n: inventory(p) for n, p in variants.items()}
    selected = out / "selected"
    copy_project(variants["reuse_previous"], selected, snapshots["reuse_previous"])
    difference, candidate_identity = patches.apply(
        variants["reuse_previous"], selected, proposal, SCOPE
    )
    (out / "candidate.patch").write_text(difference)
    candidate_snapshot = inventory(selected)
    write(out / "candidate-source.json", candidate_snapshot)
    protocol = {
        "candidate_sha256": proposal_hash,
        "candidate_identity": candidate_identity,
        "cases_sha256": sha(out / "cases.json"),
        "llm_calls": 0,
        "pairs": args.pairs,
        "policy": POLICY,
        "source_hashes": {
            p.name: sha(p)
            for p in (
                Path(__file__),
                HERE / "continue_broader.py",
                HERE / "broader_study.py",
                HERE / "spectrum_workload.py",
            )
        },
        "scope": "Frozen source; published cases; no selection or new unseen-workload claim. Four deployment gates plus diagnostic previous broader candidate.",
    }
    write(out / "qualification-protocol.json", protocol)
    if args.prepare_only:
        log(f"PREPARED result={out / 'qualification-protocol.json'} log={out.with_suffix('.log')}")
        return 0
    summary = {"status": "incomplete", "protocol": protocol, "comparisons": {}}
    try:
        for name, reference in variants.items():
            report = compare(
                reference,
                selected,
                snapshots[name],
                candidate_snapshot,
                value,
                cases,
                out / "qualification" / name,
                pairs=args.pairs,
                deadline=deadline,
                progress=log,
                minimum_speedup=1.02,
            )
            extra = (
                memory_and_setup(
                    reference,
                    selected,
                    snapshots[name],
                    candidate_snapshot,
                    value,
                    cases,
                    report,
                    out / "qualification-memory" / name,
                    deadline,
                )
                if report["status"] == "matched"
                else {}
            )
            passed = (
                eligible(report, POLICY, final=True)
                and extra.get("startup_eligible", False)
                and extra.get("memory_eligible", False)
            )
            summary["comparisons"][name] = {**compact(report), **extra, "all_gates_passed": passed}
            write(out / "qualification.json", summary)
        summary["status"] = (
            "accepted_experiment"
            if all(summary["comparisons"][n]["all_gates_passed"] for n in BASELINES)
            else "reference_retained"
        )
    except (ValueError, OSError, TimeoutError) as error:
        summary.update(status="failed", reason=str(error))
    summary["sources_unchanged"] = (
        sha(candidate_file) == proposal_hash
        and sha(out / "cases.json") == protocol["cases_sha256"]
        and inventory(selected)["files"] == candidate_snapshot["files"]
        and all(inventory(p)["files"] == snapshots[n]["files"] for n, p in variants.items())
        and all(sha(HERE / n) == h for n, h in protocol["source_hashes"].items())
        and not git(project, "status", "--porcelain")
    )
    summary["elapsed_seconds"] = clock() - started
    if not summary["sources_unchanged"] or clock() > deadline:
        summary.update(status="failed", reason="provenance changed or deadline exhausted")
    write(out / "qualification.json", summary)
    log(
        f"DONE status={summary['status']} result={out / 'qualification.json'} log={out.with_suffix('.log')}"
    )
    return 0 if summary["status"] in {"accepted_experiment", "reference_retained"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
