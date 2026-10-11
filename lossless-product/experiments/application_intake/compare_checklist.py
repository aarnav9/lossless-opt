"""Directly time both frozen arm selections after authoring has ended.

This diagnostic never selects another candidate or feeds results to an author.
Declare its plan before the second arm finishes; it is additional measurement
cost outside each arm's equal search allowance, not another author replication.
"""

import argparse
import json
import os
from pathlib import Path

from lossless.applications.comparison import compare
from lossless.applications.config import resolve
from lossless.applications.discovery import inventory
from lossless.applications.runtime import clock

from study import log, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--declare-only", action="store_true")
    args = parser.parse_args()
    root = args.study.resolve()
    protocol_path = root / "final-comparison-plan.json"
    if args.declare_only:
        if protocol_path.exists():
            raise ValueError("comparison plan already exists")
        plan = {
            "schema_version": 1,
            "pairs": 12,
            "minimum_speedup": 1.02,
            "budget_seconds_per_pair": 600,
            "source_sha256": sha(Path(__file__)),
            "scope": "Additional paired timing of the independently selected implementations after both arms finish. Control is reference and checklist is candidate. A retained arm uses the original baseline. No reselection, further authors or deployment acceptance. One author pair remains one replicate.",
            "study_protocol_sha256": sha(root / "protocol.json"),
        }
        write(protocol_path, plan)
        log(f"DECLARED result={protocol_path}; no benchmarks or author calls")
        return 0
    plan = json.loads(protocol_path.read_text())
    if (
        sha(Path(__file__)) != plan["source_sha256"]
        or sha(root / "protocol.json") != plan["study_protocol_sha256"]
    ):
        raise ValueError("comparison source or study protocol changed")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise ValueError("set OPENBLAS_NUM_THREADS=1")
    protocol = json.loads((root / "protocol.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if summary["status"] != "completed":
        raise ValueError("both arms must finish before direct comparison")
    output = root / "direct-comparison"
    output.mkdir(exist_ok=False)
    started = clock()
    log(
        f"START expected=3min per pair result={output / 'summary.json'} log={root / 'direct-comparison.log'}"
    )
    results = []
    for pair in sorted({r["pair"] for r in protocol["schedule"]}):
        sources, snapshots, reports = {}, {}, {}
        runs = [r for r in protocol["schedule"] if r["pair"] == pair]
        for run in runs:
            folder = root / run["output"]
            report = json.loads((folder / "report.json").read_text())
            reports[run["arm"]] = sha(folder / "report.json")
            sources[run["arm"]] = (
                folder / "candidates" / report["selected"] / "project"
                if report["status"] == "accepted_experiment"
                else folder / "reference"
            )
            snapshots[run["arm"]] = inventory(sources[run["arm"]])
        value, cases = resolve(root / runs[0]["config"])
        if snapshots["control"]["files"] == snapshots["checklist"]["files"]:
            result = {"status": "identical_source", "speedup": 1.0}
        else:
            result = compare(
                sources["control"],
                sources["checklist"],
                snapshots["control"],
                snapshots["checklist"],
                value,
                [c for c in cases if c["split"] == "evaluation"],
                output / f"pair_{pair:02d}",
                pairs=plan["pairs"],
                deadline=clock() + plan["budget_seconds_per_pair"],
                progress=log,
                minimum_speedup=plan["minimum_speedup"],
            )
        results.append(
            {
                "pair": pair,
                "report_hashes": reports,
                "source_snapshots": snapshots,
                "comparison": result,
            }
        )
    record = {
        "plan_sha256": sha(protocol_path),
        "results": results,
        "elapsed_seconds": clock() - started,
        "scope": plan["scope"],
    }
    write(output / "summary.json", record)
    log(f"DONE result={output / 'summary.json'} log={root / 'direct-comparison.log'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
