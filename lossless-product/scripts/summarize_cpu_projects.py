"""Archive bounded CPU workflow evidence, including negative outcomes.

Pass --run LABEL=/path/to/workflow repeatedly. Source/fixture bytes are identified
by the frozen manifests; upstream code and full project copies are not republished.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {name: Path(path).resolve() for name, path in (item.split("=", 1) for item in args.run)}

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, str):
            for name, root in roots.items():
                value = value.replace(str(root), "<run:" + name + ">")
            return re.sub(r"/Users/[^/\s]+", "<home>", value)
        return value

    evidence = {}
    for name, root in roots.items():
        records = {}
        patterns = [
            "report.json",
            "job/lossless.json",
            "job/cases.json",
            "profile/report.json",
            "plan.json",
            "search/report.json",
            "search/source.json",
            "search/manifest.json",
            "search/frozen_winner.json",
            "search/evaluation/comparison.json",
            "search/candidates/*/proposal.json",
            "search/candidates/*/discovery/comparison.json",
            "search/memory/*/receipt.json",
            "search/memory/*/worker_result.json",
            "profile/attempts/*/worker_result.json",
            "delivery/checks.json",
            "delivery/*/*/receipt.json",
            "application/manifest.json",
            "application/prepared.json",
            "prepared/report.json",
            "diagnostic/report.json",
            "diagnostic/attempts/*/receipt.json",
        ]
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                records[path.relative_to(root).as_posix()] = dict(
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    data=clean(json.loads(path.read_text())),
                )
        for relative in ("run.log", "search/selected.patch", "prepared/run.log"):
            path = root / relative
            if path.is_file():
                records[relative] = dict(
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    text=clean(path.read_text()),
                )
        evidence[name] = records
    result = dict(
        schema_version=1,
        runs=evidence,
        scope="Local exact CPU project workflows. Include no-win and failed-reference results. SHA256 values describe original records before path redaction. Launcher timings are single descriptive pairs. Search evaluation qualifies callable speedups; a separate prepared report qualifies complete reusable-worker requests. No model calls.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        gzip.compress(
            json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode(), mtime=0
        )
    )
    print(f"Evidence: {args.output} ({args.output.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
