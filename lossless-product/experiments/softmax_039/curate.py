"""Create compact public measurements and an optional complete compressed evidence asset."""

import argparse
import gzip
import json
from pathlib import Path
from lossless._native.common import read, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--archive")
    a = p.parse_args()
    root = Path(a.results).resolve()
    summary = read(root / "summary.json")
    sources = {p.name: p.read_text() for p in (root / "sources").glob("*.c")}
    full, compact = [], []
    for process in range(3):
        path = root / f"process_{process}/results.json"
        records = read(path)
        environment = read(path.with_name("environment.json"))
        full.append({"environment": environment, "results": records})
        rows = []
        for record in records:
            checks = record["checks"]
            aggregate = []
            for implementation in sorted({c["implementation"] for c in checks}):
                selected = [c for c in checks if c["implementation"] == implementation]
                aggregate.append(
                    {
                        "implementation": implementation,
                        "checks": len(selected),
                        "all_passed": all(
                            c["passed"] and c["guards_intact"] and c["inputs_unchanged"]
                            for c in selected
                        ),
                        "distributions": [c["distribution"] for c in selected],
                        "max_error_over_tolerance": max(
                            c["max_error_over_tolerance"] for c in selected
                        ),
                        "max_row_sum_error": max(c["max_row_sum_error"] for c in selected),
                    }
                )
            rows.append(
                {**{k: v for k, v in record.items() if k != "checks"}, "check_summary": aggregate}
            )
        compact.append({"raw_sha256": sha(path), "environment": environment, "results": rows})
    evidence = {
        "summary": summary,
        "sources": sources,
        "processes": compact,
        "scope": "All measured samples retained. Per-distribution correctness records summarized per implementation; complete records are in the optional release evidence archive. Three processes on one M2, not independent hardware replication.",
    }
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(evidence, separators=(",", ":"), allow_nan=False) + "\n")
    if a.archive:
        archive = Path(a.archive)
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(
            gzip.compress(
                json.dumps(
                    {"summary": summary, "sources": sources, "processes": full},
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode(),
                mtime=0,
            )
        )
    print(f"Public evidence: {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
