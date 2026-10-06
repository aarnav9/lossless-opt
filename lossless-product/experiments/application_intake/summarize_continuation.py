"""Package continuation receipts, raw comparisons and the frozen source proposal."""

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
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    root = args.study.resolve()
    summary = read(root / "summary.json")
    winner = read(root / "frozen-winner.json")
    attempts = []
    for row in summary["attempts"]:
        folder = root / "candidates" / row["id"]
        record = dict(row)
        for name in ("proposal.json", "author/invocation.json", "discovery/comparison.json"):
            if (folder / name).exists():
                record[name] = read(folder / name)
        attempts.append(record)
    evidence = {
        "schema_version": 1,
        "summary": summary,
        "protocol": read(root / "protocol.json"),
        "search_clock": read(root / "search-clock.json"),
        "author_context": read(root / "author-context.json"),
        "frozen_winner": winner,
        "attempts": attempts,
        "final_comparisons": {
            path.parent.name: read(path) for path in (root / "final").glob("*/comparison.json")
        },
        "provenance": {
            "summary_sha256": sha(root / "summary.json"),
            "frozen_winner_sha256": sha(root / "frozen-winner.json"),
            "executed_controller_sha256": sha(root / "executed-controller.py"),
        },
        "executed_controller": (root / "executed-controller.py").read_text(),
    }
    if (root / "validation.json").exists():
        evidence["validation"] = read(root / "validation.json")
    candidate_path = root / "candidates" / winner["id"] / "proposal.json"
    args.candidate.write_bytes(candidate_path.read_bytes())
    evidence["provenance"]["candidate_sha256"] = sha(candidate_path)
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
                "status": summary["status"],
                "evidence": str(args.evidence),
                "bytes": args.evidence.stat().st_size,
                "candidate": str(args.candidate),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
