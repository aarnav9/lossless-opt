"""Package matched-checklist evidence without CLI logs or user-specific paths."""

import argparse
import gzip
import json
from pathlib import Path
import re

from study import sha


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    root = args.study.resolve()
    protocol, summary = read(root / "protocol.json"), read(root / "summary.json")
    if summary["status"] not in {"completed", "incomplete"}:
        raise ValueError("finish the pilot before publishing its evidence")
    arms = []
    for run in protocol["schedule"]:
        directory = root / run["output"]
        if not (directory / "report.json").exists():
            continue
        arm = {"pair": run["pair"], "arm": run["arm"], "artifacts": {}}
        for name in (
            "report.json",
            "manifest.json",
            "plan.json",
            "author_context.json",
            "frozen_winner.json",
            "checklist.json",
            "profile/hardware.json",
            "profile/cpu_catalog.json",
            "evaluation/comparison.json",
        ):
            path = directory / name
            if path.exists():
                arm["artifacts"][name] = read(path)
        arm["attempts"] = []
        for folder in sorted((directory / "candidates").glob("candidate_*")):
            record = {"id": folder.name}
            for name in (
                "author/receipt.json",
                "author/response.json",
                "discovery/comparison.json",
            ):
                path = folder / name
                if path.exists():
                    record[name] = read(path)
            if (folder / "author/prompt.json").exists():
                record["prompt_sha256"] = sha(folder / "author/prompt.json")
            arm["attempts"].append(record)
        arms.append(arm)
    evidence = {
        "schema_version": 1,
        "protocol": protocol,
        "summary": summary,
        "arms": arms,
        "provenance": {
            "protocol_sha256": sha(root / "protocol.json"),
            "summary_sha256": sha(root / "summary.json"),
        },
    }
    evidence["executed_context_sources"] = {
        p.name: {"sha256": sha(p), "source": p.read_text()}
        for p in (root / "executed-context").glob("*.py")
    }
    if (root / "direct-comparison/summary.json").exists():
        evidence["direct_comparison"] = read(root / "direct-comparison/summary.json")
        evidence["direct_comparison_plan"] = read(root / "final-comparison-plan.json")
    if (root / "context-compaction.json").exists():
        evidence["context_compaction"] = read(root / "context-compaction.json")
    payload = json.dumps(evidence, indent=2, allow_nan=False).replace(str(root), "<study>")
    payload = re.sub(r"/Users/[^/\"\\]+", "<user>", payload)
    payload = re.sub(r"/private/var/folders/[^\"\\ ]+", "<temporary>", payload)
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    with args.evidence.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as archive:
            archive.write((payload + "\n").encode())
    print(
        json.dumps(
            {
                "evidence": str(args.evidence),
                "bytes": args.evidence.stat().st_size,
                "status": summary["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
