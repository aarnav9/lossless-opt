"""Real-worker acceptance and export/load checks for frozen deployment limits."""

import argparse
import copy
from pathlib import Path
import time

import lossless
from lossless._native.common import read, write, stamp, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    product = Path(__file__).resolve().parents[2]
    cfg = read(product / "examples/mlx-fixed-count/lossless.json")
    cfg["workload"]["parameters"].update(
        model=str(Path(a.model).resolve()), repeats=20, graph_recipes=[]
    )
    cfg["workload"]["inputs"] = str(product / "examples/mlx-fixed-count/requests.json")
    cfg["proofs"] = {"check": False}
    cfg["budget"] = {"wall_time_seconds": 90, "max_candidates": 1, "max_llm_calls": 0}
    cases = {
        "permissive": dict(
            max_p95_ttft_seconds=1,
            max_p95_completion_seconds=2,
            max_peak_mlx_bytes=1073741824,
            max_break_even_calls=10000,
            min_tokens_per_second=1,
        ),
        "tight_ttft": dict(max_p95_ttft_seconds=0.000001),
        "tight_memory": dict(max_peak_mlx_bytes=1),
        "tight_payback": dict(max_break_even_calls=1),
        "tight_throughput": dict(min_tokens_per_second=1e9),
    }
    write(
        out / "plan.json",
        dict(
            cases=cases,
            script_sha256=sha(__file__),
            expected="permissive accepts; every tight case retains reference; constraints remain frozen through export",
        ),
    )
    stamp(f"START estimated=1-2min result={out} log={out.parent / (out.name + '.log')}")
    results = []
    started = time.monotonic()
    for name, limits in cases.items():
        config = copy.deepcopy(cfg)
        config["objective"]["constraints"] = limits
        if name not in {"permissive", "tight_ttft"}:
            config["workload"]["parameters"]["repeats"] = 3
        write(out / (name + ".json"), config)
        result = lossless.optimize(lossless.Workload.from_dict(config), output=out / name)
        expected = "accepted" if name == "permissive" else "reference_retained"
        assert result.summary["status"] == expected, (name, result.summary["status"])
        assert result.summary["discovery"]["retained"]["exact"]
        if name == "permissive":
            assert result.summary["deployment_constraints"]["passed"]
            artifact = result.export(out / "artifact")
            assert read(artifact / "resolved.json")["objective"]["constraints"] == limits
            import mlx.core as mx
            from lossless.adapters.mlx.runtime import generate
            from lossless.adapters.mlx.checks import diff, cache_diff

            op = lossless.load(artifact).open(model=a.model)
            texts = ["The river flows through the valley.", "A quiet forest in the morning."]
            baseline = generate(op._model, None, op._tokenizer, texts, [3, 5], capture=True)
            value = op.generate(texts, [3, 5], capture=True)
            assert baseline[0]["output_ids"] == value[0]["output_ids"]
            assert all(diff(x, y)["bitwise"] for x, y in zip(baseline[2], value[2]))
            assert all(cache_diff(x, y)["bitwise"] for x, y in zip(baseline[1], value[1]))
            assert op.fallback_reason is None
            del baseline, value, op
            mx.clear_cache()
        else:
            checks = result.summary["discovery"]["retained"]["constraints"]["checks"]
            assert any(not c["passed"] for c in checks.values())
        results.append(
            dict(
                name=name,
                status=result.summary["status"],
                summary=result.summary,
                report_sha256=sha(out / name / "report.json"),
            )
        )
        stamp(f"CONSTRAINT case={name} outcome={result.summary['status']}")
    write(
        out / "summary.json",
        dict(results=results, wall_seconds=time.monotonic() - started, export_load_exact=True),
    )
    stamp(f"COMPLETE result={out / 'summary.json'} log={out.parent / (out.name + '.log')}")


if __name__ == "__main__":
    main()
