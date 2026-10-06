"""Package completed broader-study evidence without running code or new trials."""

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    root = args.study.resolve()
    summary = read(root / "summary.json")
    report = read(root / "run/report.json")
    protocol = read(root / "protocol.json")
    records = []
    for row in report["candidates"]:
        path = root / "run/candidates" / row["id"]
        record = copy.deepcopy(row)
        for name in ("author/response.json", "author/invocation.json", "discovery/comparison.json"):
            if (path / name).exists():
                record[name] = read(path / name)
        events = path / "author/events.jsonl"
        if events.exists() and row["status"] == "provider_failed":
            record["provider_failures"] = [
                event
                for line in events.read_text().splitlines()
                if (event := json.loads(line)).get("type") in {"error", "turn.failed"}
            ]
        records.append(record)
    comparisons = {}
    for base in (root / "baseline-audit", root / "final"):
        if base.exists():
            for path in base.glob("*/comparison.json"):
                comparisons[str(path.relative_to(root))] = read(path)
    if (root / "run/evaluation/comparison.json").exists():
        comparisons["run/evaluation/comparison.json"] = read(
            root / "run/evaluation/comparison.json"
        )
    # A next-author marker conservatively bounds when the previous attempt was
    # fully evaluated. It is not the exact completion timestamp of each kernel.
    markers = re.findall(
        r"AUTHOR candidate=(\d+)/\d+ remaining=([\d.]+)s", (root / "run/run.log").read_text()
    )
    frontier, best = [], 1.0
    for i, (number, remaining) in enumerate(markers):
        index = int(number) - 1
        row = report["candidates"][index]
        stats = row.get("discovery", {}).get("statistics", {})
        score = stats.get("case_balanced_geomean_speedup")
        candidate_ok = (
            row.get("status") == "matched"
            and score is not None
            and score >= 1.02
            and all(c["speedup"] >= 1 / 1.03 for c in stats.get("cases", []))
        )
        if candidate_ok:
            best = max(best, score)
        upper = (
            protocol["search_seconds"] - float(markers[i + 1][1])
            if i + 1 < len(markers)
            else report.get("search_seconds")
        )
        frontier.append(
            {
                "candidate": row["id"],
                "started_search_seconds": protocol["search_seconds"] - float(remaining),
                "completed_by_search_seconds": upper,
                "discovery_speedup": score,
                "discovery_eligible": candidate_ok,
                "best_discovery_speedup": best,
            }
        )
    evidence = {
        "schema_version": 1,
        "summary": summary,
        "report": report,
        "protocol": protocol,
        "plan": read(root / "plan.json"),
        "comparisons": comparisons,
        "attempts": records,
        "discovery_frontier": frontier,
        "manifest": read(root / "run/manifest.json"),
        "provenance": {
            "summary_sha256": hashlib.sha256((root / "summary.json").read_bytes()).hexdigest(),
            "report_sha256": hashlib.sha256((root / "run/report.json").read_bytes()).hexdigest(),
            "frontier_scope": "Development timing only; completion bounded by next author start. Not independent validation of time-budget effects.",
        },
    }
    if (root / "validation.json").exists():
        evidence["validation"] = read(root / "validation.json")
    if args.qualification:
        qualification = args.qualification.resolve()
        evidence["recovery"] = read(root / "recovery.json")
        evidence["qualification"] = read(qualification / "qualification.json")
        evidence["qualification_protocol"] = read(qualification / "qualification-protocol.json")
        evidence["qualification_comparisons"] = {
            path.parent.name: read(path)
            for path in (qualification / "qualification").glob("*/comparison.json")
        }
        evidence["provenance"]["qualification_sha256"] = hashlib.sha256(
            (qualification / "qualification.json").read_bytes()
        ).hexdigest()
        evidence["outcome"] = {
            "authoring_complete": False,
            "authoring_status": report["status"],
            "qualification_status": evidence["qualification"]["status"],
        }
    # The public artifact retains identities and measurements, not this user's
    # absolute checkout/interpreter paths or Codex temporary directories.
    payload = json.dumps(evidence, indent=2, allow_nan=False)
    payload = payload.replace(str(root), "<study>")
    if args.qualification:
        payload = payload.replace(str(args.qualification.resolve()), "<qualification>")
    payload = re.sub(r"/Users/[^/\"\\]+", "<user>", payload)
    payload = re.sub(r"/private/var/folders/[^\"\\ ]+", "<temporary>", payload)
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    with args.evidence.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as archive:
            archive.write((payload + "\n").encode())
    frozen = root / "run/frozen_winner.json"
    if not frozen.exists() and (root / "recovery.json").exists():
        frozen = root / "recovery.json"
    if frozen.exists():
        selection = read(frozen)
        selected = selection.get("id", selection.get("selected"))
        proposal = read(root / "run/candidates" / selected / "author/response.json")
        proposal.pop("usage", None)
        args.candidate.write_text(json.dumps(proposal, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": evidence.get("qualification", summary)["status"],
                "evidence": str(args.evidence),
                "bytes": args.evidence.stat().st_size,
                "candidate": str(args.candidate) if frozen.exists() else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
