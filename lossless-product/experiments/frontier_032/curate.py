"""Copy small public evidence; never include model weights, binaries or credentials."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path(args.results).resolve()
    repo = Path(__file__).resolve().parents[3]
    files = [
        "mlx_gpu/summary.json",
        "mlx_replay/summary.json",
        "mlx_deploy/summary.json",
        "mlx_wheel/summary.json",
        "search/summary.json",
        "search/authoring.json",
        "verified_v2/summary.json",
        "verified_ir/summary.json",
        "verified_ir/proofs.json",
        "scheduling_v2/summary.json",
        "projection_data/manifest.json",
        "certificate/summary.json",
        "racing/summary.json",
        "tensor/summary.json",
        "overhead/summary.json",
    ]
    files += [
        str(p.relative_to(root)) for p in sorted((root / "search").glob("*_[012]/dataset.jsonl"))
    ]
    records = {}
    for name in files:
        raw = (root / name).read_bytes()
        data = raw.decode().replace(str(repo), "<REPO>")
        records[name] = {
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "data": [json.loads(s) for s in data.splitlines()]
            if name.endswith(".jsonl")
            else json.loads(data),
        }
    sources = [
        p
        for folder in [
            repo / "lossless-product/experiments/frontier_032",
            repo / "lossless-product/src",
        ]
        for p in folder.rglob("*")
        if p.suffix in {".py", ".c"}
    ]
    evidence = {
        "schema_version": 1,
        "date": "2026-10-02",
        "source_sha256_at_curation": {
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources)
        },
        "notes": [
            "Absolute checkout paths replaced with <REPO>; raw hashes precede this replacement.",
            "Timing records were produced during development; each native run retains its own frozen harness locally.",
            "Initial MLX forced-history checks did not exercise capture; use mlx_replay's 24 graph checks.",
            "Scheduling cohort_active_kv_bytes is exported storage including spare capacity, not logical active bytes.",
            "Manual proposals share one authoring session; three workload repetitions are not independent model draws.",
            "See frontier-032-038.md for negative results, measurement scope and reproduction commands.",
        ],
        "records": records,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(evidence, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Evidence: {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
