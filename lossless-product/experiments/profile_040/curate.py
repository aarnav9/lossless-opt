"""Preserve profile samples and deployment checks without local absolute paths."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    product = Path(__file__).resolve().parents[2]
    repository = product.parent

    def scrub(value):
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        if isinstance(value, str):
            return value.replace(str(repository), "<repository>")
        return value

    records = {}
    for name, filename in [
        ("native_profile", "copy_release/report.json"),
        ("mlx_profile", "mlx_profile/report.json"),
        ("mlx_deployment", "mlx_deploy/summary.json"),
    ]:
        raw = (args.results / filename).read_bytes()
        value = json.loads(raw)
        if name.endswith("profile"):
            assert value["status"] == "profiled" and value["llm_calls"] == 0
            assert value["input_split"] == "discovery"
        records[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "record": scrub(value)}
    mlx = records["mlx_profile"]["record"]
    assert all(all(v.values()) for v in mlx["correctness"])
    for implementation in mlx["cases"][0]["implementations"].values():
        assert implementation["fallback_reason"] is None
        assert abs(sum(v["fraction"] for v in implementation["stages"]) - 1) < 1e-10
        for stages, total in zip(
            implementation["stage_samples_seconds"], implementation["warm"]["samples_seconds"]
        ):
            assert abs(sum(stages.values()) - total) < 1e-10
    deployment = records["mlx_deployment"]["record"]
    assert deployment["export_load_exact"] and deployment["continuation_checks"] == 8
    result = {
        "schema_version": 1,
        "scope": "One local Apple M2 validation. Discovery profiling is not independent final evaluation. Absolute repository paths replaced; original record digests retained.",
        "source_sha256": {
            name: hashlib.sha256((product / name).read_bytes()).hexdigest()
            for name in [
                "src/lossless/profiling.py",
                "src/lossless/_profile_worker.py",
                "src/lossless/adapters/mlx/runtime.py",
                "experiments/frontier_032/mlx_deploy.py",
            ]
        },
        **records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Curated profile evidence: {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
