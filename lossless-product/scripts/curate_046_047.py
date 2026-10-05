"""Archive frozen qualification/model evidence without raw CLI streams or binaries."""

import argparse
import gzip
import importlib.util
import json
from pathlib import Path

from lossless._native import engine
from lossless._native.common import read, sha


def curate(campaign, root, output, extras=()):
    repository = Path(__file__).resolve().parents[2]
    product = repository / "lossless-product"
    path = (
        product
        / "experiments"
        / ("softmax_046/qualify.py" if campaign == "046" else "models_047/study.py")
    )
    loader = importlib.util.spec_from_file_location("curated_study", path)
    study = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(study)
    plan = study.verify(root)
    assert (root / "summary.json").exists(), "finish the study before curating"
    if campaign == "047":
        study.require_sealed(root)
        for name in [*plan["models"], "enumerated"]:
            engine.verify(root / name)
        for folder in sorted((root / "authors").glob("*/*")):
            receipt = read(folder / "receipt.json")
            response = folder / "response.json"
            assert (sha(response) if response.exists() else None) == receipt["response_sha256"]
            assert read(folder / "invocation.json")["prompt_sha256"] == sha(folder / "prompt.json")
    elif (root / "selection.json").exists():
        selection = read(root / "selection.json")
        assert selection["source_sha256"] == sha(root / "sources" / (selection["candidate"] + ".c"))
        assert selection["discovery_summary_sha256"] == sha(root / "discovery_summary.json")

    locations = [(root, "study"), *[(p, p.name) for p in extras]]

    def scrub(value):
        if isinstance(value, str):
            for directory, label in locations:
                value = value.replace(str(directory), f"<{label}>")
            return value.replace(str(repository), "<repo>").replace(str(Path.home()), "<home>")
        if isinstance(value, dict):
            return {scrub(k): scrub(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [scrub(v) for v in value]
        return value

    records = {}
    sources = {}
    for directory, label in locations:
        if (directory / "manifest.json").exists() and (directory / "state.json").exists():
            engine.verify(directory)
        for path in sorted(directory.rglob("*.json")):
            if path.name == "raw_response.json":
                continue
            value = read(path)
            if path.name == "invocation.json":
                argv = value["argv"]
                argv[argv.index("--cd") + 1] = "<temporary-empty-directory>"
            records[f"<{label}>/{path.relative_to(directory)}"] = {
                "original_sha256": sha(path),
                "record": scrub(value),
            }
        for path in sorted(directory.rglob("*.c")):
            sources[f"<{label}>/{path.relative_to(directory)}"] = {
                "sha256": sha(path),
                "source": path.read_text(),
            }
    frozen = plan["harness" if campaign == "046" else "source_hashes"]
    for path, digest in frozen.items():
        path = Path(path)
        assert sha(path) == digest
        sources[str(path.relative_to(repository))] = {"sha256": digest, "source": path.read_text()}
    companions = [Path(__file__)]
    if campaign == "046":
        companions += [
            product / "experiments/softmax_046/deploy.py",
            *sorted((product / "examples/softmax-fortran").glob("*.py")),
        ]
    for path in companions:
        sources[str(path.relative_to(repository))] = {
            "sha256": sha(path),
            "source": path.read_text(),
        }
    value = {
        "schema_version": 1,
        "campaign": campaign,
        "scope": "Original digests precede path redaction. Exact frozen harness and kernel sources, raw timing/validation JSON, final proposal responses and receipts retained. Raw CLI JSONL/stderr, auth/runtime databases, temporary directories and binaries excluded. Missing timeout token usage remains unknown.",
        "records": records,
        "sources": sources,
    }
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    assert str(Path.home()).encode() not in encoded
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(gzip.compress(encoded, mtime=0))
    return {
        "output": str(output),
        "sha256": sha(output),
        "bytes": output.stat().st_size,
        "uncompressed_bytes": len(encoded),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", choices=["046", "047"])
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra", type=Path, action="append", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            curate(
                args.campaign,
                args.study.resolve(),
                args.output.resolve(),
                [p.resolve() for p in args.extra],
            )
        )
    )


if __name__ == "__main__":
    main()
