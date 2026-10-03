"""Publish auditable 041/042 records without local paths, weights or credentials."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-results", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    product = root / "lossless-product"
    assets = args.assets
    assets.mkdir(parents=True, exist_ok=True)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def record(path):
        return {"original_sha256": digest(path), "record": json.loads(path.read_text())}

    def source(path):
        return {"sha256": digest(path), "source": path.read_text()}

    def save(name, value):
        text = json.dumps(value, indent=2, allow_nan=False).replace(str(root), "<repository>")
        (assets / name).write_text(text + "\n")
        print(assets / name)

    suites = {}
    checks = confirmed = stock_failures = profiles = 0
    stock_requests = 0
    stock_mismatches = dict(tokens=0, probabilities=0, kv=0)
    for name in ["mlx_041", "mlx_041_natural"]:
        folder = args.research_results / name
        plan = record(folder / "plan.json")
        harness = source(
            folder / "frozen_suite.py"
            if name == "mlx_041"
            else product / "experiments/mlx_041/suite.py"
        )
        assert plan["record"]["script_sha256"] == harness["sha256"]
        suites[name] = {
            "plan": plan,
            "selection": record(folder / "selection.json"),
            "harness": harness,
        }
        for phase in ["discovery", "evaluation"]:
            evidence = record(folder / phase / "summary.json")
            suites[name][phase] = evidence
            assert evidence["record"]["plan_sha256"] == plan["original_sha256"]
            for row in evidence["record"]["records"]:
                profiles += 1
                stock_failures += not row["exact"]["stock_batch"]
                for comparison in row["checks"]["stock_batch"]:
                    stock_requests += 1
                    for key in stock_mismatches:
                        stock_mismatches[key] += not comparison[key]
                for method in ["retained", "decoder_026", "fullhead_027"]:
                    assert row["exact"][method]
                    for comparison in row["checks"][method]:
                        assert all(comparison[k] for k in ["tokens", "probabilities", "kv"])
                        checks += 1
                if phase == "evaluation":
                    confirmed += bool(row["confirmed"])
    assert (profiles, checks, stock_failures, confirmed) == (32, 360, 24, 10)
    save(
        "mlx-041.json",
        {
            "scope": "Local Apple M2, pinned model/runtime. Seven interleaved warm repeats per method. Exact means sampled tokens/full probabilities/active KV; no universal proof. Curated natural prompts are not production traces. Selection precedes held-out evaluation. Original raw-file digests precede path redaction.",
            "counts": dict(
                profiles=profiles,
                product_request_checks=checks,
                stock_batch_nonexact_profiles=stock_failures,
                confirmed_evaluation_choices=confirmed,
                evaluation_profiles=16,
                stock_batch_request_checks=stock_requests,
                stock_batch_request_mismatches=stock_mismatches,
            ),
            "suites": suites,
        },
    )
    folder = args.research_results / "constraints_042"
    plan = record(folder / "plan.json")
    harness = source(product / "experiments/constraints_042/run.py")
    assert plan["record"]["script_sha256"] == harness["sha256"]
    summary = record(folder / "summary.json")
    assert summary["record"]["export_load_exact"]
    for row in summary["record"]["results"]:
        assert row["status"] == (
            "accepted" if row["name"] == "permissive" else "reference_retained"
        )
        assert row["report_sha256"] == digest(folder / row["name"] / "report.json")
    save(
        "constraints-042.json",
        {
            "scope": "Real MLX workers with five predeclared policies, including deliberately impossible limits. Original raw-file digests precede path redaction. Warm active MLX allocation is not total device/process memory; sample p95 is not a production guarantee.",
            "plan": plan,
            "harness": harness,
            "configurations": {
                name: record(folder / (name + ".json")) for name in plan["record"]["cases"]
            },
            "summary": summary,
            "exported_resolved": record(folder / "artifact/resolved.json"),
            "implementation": {
                str(path.relative_to(product)): source(path)
                for path in [
                    product / "src/lossless/constraints.py",
                    product / "src/lossless/adapters/mlx/worker.py",
                ]
            },
        },
    )


if __name__ == "__main__":
    main()
