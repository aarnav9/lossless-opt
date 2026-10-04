"""Export campaign 045 evidence without raw CLI streams, auth metadata or binaries."""

import argparse
import gzip
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository / "lossless-product/experiments/iterate_045"))
    import study
    from lossless._native import engine
    from lossless._native.common import read, sha

    root = args.study.resolve()
    plan = study.verify(root)
    study.require_sealed(root)
    assert (root / "summary.json").exists(), "complete the study before archiving"
    for arm in ["iterative", "enumerated"]:
        engine.verify(root / arm)
    for folder in sorted((root / "rounds").iterdir()):
        receipt = read(folder / "receipt.json")
        response = folder / "response.json"
        assert (sha(response) if response.exists() else None) == receipt["response_sha256"]
        invocation = read(folder / "invocation.json")
        assert invocation["prompt_sha256"] == sha(folder / "prompt.json")

    def scrub(value):
        if isinstance(value, str):
            return value.replace(str(root), "<study>").replace(str(repository), "<repo>")
        if isinstance(value, list):
            return [scrub(v) for v in value]
        if isinstance(value, dict):
            return {scrub(k): scrub(v) for k, v in value.items()}
        return value

    records = {}
    for path in sorted(root.rglob("*.json")):
        if path.name == "raw_response.json":
            continue
        value = read(path)
        if path.name == "invocation.json":
            argv = value["argv"]
            argv[argv.index("--cd") + 1] = "<temporary-empty-directory>"
        records[str(path.relative_to(root))] = {
            "original_sha256": sha(path),
            "record": scrub(value),
        }
    sources = {
        str(Path(path).relative_to(repository)): {
            "sha256": digest,
            "source": Path(path).read_text(),
        }
        for path, digest in plan["source_hashes"].items()
    }
    for path in sorted(root.glob("*/proposals/*/kernel.c")):
        sources["<study>/" + str(path.relative_to(root))] = {
            "sha256": sha(path),
            "source": path.read_text(),
        }
    value = {
        "schema_version": 1,
        "campaign": "045",
        "scope": "Original digests precede path redaction. Raw CLI event streams, stderr, "
        "temporary auth/runtime metadata and binaries are excluded. Timing/validation JSON, "
        "prompts, responses and exact frozen source are retained. Missing timeout usage is unknown.",
        "records": records,
        "sources": sources,
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
