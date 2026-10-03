"""Product optimize/export/load check for opt-in bounded graph recipes."""

import argparse
import json
from pathlib import Path
import lossless
from lossless._native.common import stamp, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    stamp(f"START estimated=1-3min result={out} log={out.parent / (out.name + '.log')}")
    examples = Path(__file__).resolve().parents[2] / "examples/mlx-fixed-count"
    config = json.loads((examples / "lossless.json").read_text())
    config["workload"]["parameters"].update(
        model=str(Path(a.model).resolve()), graph_recipes=["decoder_026", "fullhead_027"], repeats=7
    )
    config["workload"]["inputs"] = str(examples / "requests.json")
    config["contract"] = {"preset": "exact", "required_evidence": "validated"}
    config["proofs"] = {"check": False}
    config["budget"] = {"wall_time_seconds": 180, "max_candidates": 3, "max_llm_calls": 0}
    result = lossless.optimize(lossless.Workload.from_dict(config, base=out), output=out / "run")
    artifact = result.export(out / "artifact")
    operation = lossless.load(artifact).open()
    from lossless.adapters.mlx.runtime import generate
    from lossless.adapters.mlx.checks import diff, cache_diff

    rows = [
        r
        for r in json.loads((examples / "requests.json").read_text())
        if r["split"] == "evaluation"
    ]
    texts = [r["text"] for r in rows]
    counts = [r["max_tokens"] for r in rows]
    reference = generate(
        operation._model, operation._grouped, operation._tokenizer, texts, counts, capture=True
    )
    candidate = operation.generate(texts, counts, capture=True)
    assert reference[0]["output_ids"] == candidate[0]["output_ids"]
    assert all(diff(x, y)["bitwise"] for x, y in zip(reference[2], candidate[2]))
    assert all(cache_diff(x, y)["bitwise"] for x, y in zip(reference[1], candidate[1]))
    # Continuation observes the final state, beyond just matching generated tokens.
    import mlx.core as mx

    continuation = 0
    for ca, cb in zip(reference[1], candidate[1]):
        inp = mx.array([[17]])
        va = operation._model(inp, cache=ca)
        vb = operation._model(inp, cache=cb)
        mx.eval(va, vb)
        assert diff(va, vb)["bitwise"] and cache_diff(ca, cb)["bitwise"]
        continuation += 1
    stopped = operation.generate(texts[:2], counts[:2], cancel_after=[1, 2])
    assert [len(x) for x in stopped[0]["output_ids"]] == [1, 2]
    write(
        out / "summary.json",
        {
            "product_report": result.summary,
            "export_load_exact": True,
            "continuation_checks": continuation,
            "cancellation_fallback": stopped[0]["fallback_reason"],
            "graph_after_load": candidate[0]["graph"],
        },
    )
    stamp(
        f"COMPLETE selected={result.summary['selected']} result={out / 'summary.json'} log={out.parent / (out.name + '.log')}"
    )


if __name__ == "__main__":
    main()
