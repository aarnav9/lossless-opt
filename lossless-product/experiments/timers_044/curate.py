"""Archive auditable timer evidence without raw CLI events, auth metadata or binaries."""

import argparse
import gzip
import json
from pathlib import Path

from lossless._native import engine
from lossless._native.common import read, sha

import study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.study.resolve()
    plan = study.verify(root)
    study.require_all_sealed(root, plan)
    assert (root / "summary.json").exists(), "finish the study before curating"
    repository = Path(__file__).resolve().parents[3]

    def scrub(value):
        if isinstance(value, str):
            return value.replace(str(root), "<study>").replace(str(repository), "<repo>")
        if isinstance(value, dict):
            return {scrub(key): scrub(item) for key, item in value.items()}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    def record(path):
        return {"original_sha256": sha(path), "record": scrub(read(path))}

    frozen = {name: record(root / name) for name in plan["hashes"]}
    authors, runs = {}, {}
    for condition in plan["conditions"]:
        authors[condition] = []
        for index in range(3):
            folder = root / condition / f"author_{index}"
            receipt = read(folder / "receipt.json")
            response = folder / "response.json"
            assert (sha(response) if response.exists() else None) == receipt["response_sha256"]
            invocation = read(folder / "invocation.json")
            assert invocation["prompt_sha256"] == sha(root / condition / "prompt.json")
            argv = invocation["argv"]
            argv[argv.index("--cd") + 1] = "<temporary-empty-directory>"
            authors[condition].append(
                {
                    "receipt": record(folder / "receipt.json"),
                    "response": record(response) if response.exists() else None,
                    "invocation": scrub(invocation),
                }
            )
        for index, arm in read(root / condition / "plan.json")["order"]:
            folder = root / condition / f"{arm}_{index}"
            state = engine.verify(folder)
            jobs = {}
            for job_id, job in state["jobs"].items():
                if "result" in job:
                    result = folder / job["result"]
                    assert sha(result) == job["result_sha256"]
                    jobs[job_id] = record(result)
            names = [
                "seal.json",
                "failed_search.json",
                "summary.json",
                "submission_failures.json",
                "discovery_wall.json",
                "evaluation_wall.json",
                "manifest.json",
                "contract.json",
                "hardware_profile.json",
            ]
            runs[f"{condition}/{arm}_{index}"] = {
                "state": record(folder / "state.json"),
                "files": {
                    name: record(folder / name) for name in names if (folder / name).exists()
                },
                "job_results": jobs,
            }
    paired = {
        str(path.relative_to(root)): record(path)
        for path in sorted(root.glob("*/paired_winners/*.json"))
    }
    paired.update(
        {
            str(path.relative_to(root)): record(path)
            for path in sorted(root.glob("between_conditions/*.json"))
        }
    )
    sources = {
        str(path.relative_to(repository)): {"sha256": sha(path), "source": path.read_text()}
        for path in [*study.frozen_sources(), Path(__file__)]
    }
    value = {
        "schema_version": 1,
        "scope": "Original digests precede path redaction; paths use <study> and <repo>. "
        "Raw CLI event streams, temporary auth/runtime metadata and binaries are excluded.",
        "plan": record(root / "plan.json"),
        "frozen": frozen,
        "authors": authors,
        "runs": runs,
        "paired": paired,
        "sources": sources,
        "condition_summaries": {
            name: record(root / name / "summary.json") for name in plan["conditions"]
        },
        "summary": record(root / "summary.json"),
    }
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(encoded, mtime=0))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha(args.output),
                "bytes": args.output.stat().st_size,
                "uncompressed_bytes": len(encoded),
            }
        )
    )


if __name__ == "__main__":
    main()
