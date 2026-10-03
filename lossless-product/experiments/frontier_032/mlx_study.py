"""Fresh product extraction checks and whole-request timings; no research imports."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

from lossless._native.common import stamp, write, interval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    stamp(f"START estimated=2-4min result={out} log={out.parent / (out.name + '.log')}")
    import mlx.core as mx
    from mlx_lm import load
    from lossless.adapters.mlx.model import GroupedHead
    from lossless.adapters.mlx.graph import GraphRecipe
    from lossless.adapters.mlx.runtime import generate, admission
    from lossless.adapters.mlx.checks import diff, cache_diff
    from lossless.adapters.mlx.execution import teacher_checks, DeadHead
    from mlx_lm.models import llama
    from lossless.adapters.mlx.identity import fingerprint

    started = time.monotonic()
    t = time.perf_counter()
    model_id = fingerprint(a.model)
    hash_seconds = time.perf_counter() - t
    assert admission(a.model, mx.device_info()["device_name"], model_identity=model_id)[0]
    model, tok = load(a.model)
    mx.eval(model.parameters())
    loading = time.monotonic() - started
    grouped = GroupedHead(model, 4)
    graphs = {n: GraphRecipe(model, n, 64) for n in ["decoder_026", "fullhead_027"]}
    original = (llama.LlamaModel.__call__, DeadHead.__call__)
    source = Path(__file__).resolve().parents[2] / "examples/mlx-fixed-count/requests.json"
    requests = json.loads(source.read_text())
    batches = {s: [r for r in requests if r["split"] == s] for s in ["discovery", "evaluation"]}
    batches["ragged"] = [
        {"text": t, "max_tokens": n}
        for t, n in zip(
            ["The moon is", "A longer explanation of why rivers flow downhill:", "One"], [3, 5, 2]
        )
    ]
    records = []
    methods = ["serial", "retained", *graphs]

    def execute(name, rows, capture=False, cancel=None):
        options = {"allocator_cache_mib": 256}
        if name in graphs:
            options["execution_recipe"] = name
        return generate(
            model,
            grouped,
            tok,
            [r["text"] for r in rows],
            [r["max_tokens"] for r in rows],
            optimized=name != "serial",
            options=options,
            graph=graphs.get(name),
            capture=capture,
            cancel_after=cancel,
        )

    for split, rows in batches.items():
        ref = execute("serial", rows, True)
        cold, checks = {}, {}
        for name in methods:
            mx.reset_peak_memory()
            v = execute(name, rows, True)
            cold[name] = {
                "seconds": v[0]["seconds"],
                "peak_bytes": mx.get_peak_memory(),
                "graph": v[0]["graph"],
            }
            checks[name] = [
                dict(
                    tokens=ref[0]["output_ids"][i] == v[0]["output_ids"][i],
                    probabilities=diff(ref[2][i], v[2][i])["bitwise"],
                    kv=cache_diff(ref[1][i], v[1][i])["bitwise"],
                )
                for i in range(len(rows))
            ]
            assert all(all(c.values()) for c in checks[name]), (split, name, checks[name])
            del v
        del ref
        samples = {n: [] for n in methods}
        ttft = {n: [] for n in methods}
        completion = {n: [] for n in methods}
        orders = []
        for repeat in range(7):
            order = methods.copy()
            random.Random(3200 + repeat).shuffle(order)
            orders.append(order)
            for name in order:
                v = execute(name, rows)
                samples[name].append(v[0]["seconds"])
                ttft[name].append(v[0]["ttft_seconds"])
                completion[name].append(v[0]["completion_seconds"])
            stamp(f"MLX batch={split} repeat={repeat + 1}/7")
        record = {
            "split": split,
            "checks": checks,
            "cold_validation": cold,
            "samples_seconds": samples,
            "ttft_seconds": ttft,
            "completion_seconds": completion,
            "orders": orders,
            "median_seconds": {n: statistics.median(v) for n, v in samples.items()},
            "speedup_vs_retained": {
                n: statistics.median(samples["retained"]) / statistics.median(v)
                for n, v in samples.items()
            },
            "ci95_vs_retained": {
                n: interval(samples["retained"], v, 32) for n, v in samples.items()
            },
        }
        records.append(record)
        write(out / "timings.json", records)
    # Capacity exhaustion, changing batch shapes, exception restoration and fallback.
    cap = GraphRecipe(model, "fullhead_027", 1)
    for count in [3, 1, 3]:
        rows = batches["discovery"][:count]
        ref = execute("serial", rows, True)
        v = generate(
            model,
            grouped,
            tok,
            [r["text"] for r in rows],
            [r["max_tokens"] for r in rows],
            optimized=True,
            options={"allocator_cache_mib": 256, "execution_recipe": "fullhead_027"},
            graph=cap,
            capture=True,
        )
        assert ref[0]["output_ids"] == v[0]["output_ids"]
        assert all(diff(x, y)["bitwise"] for x, y in zip(ref[2], v[2]))
        assert all(cache_diff(x, y)["bitwise"] for x, y in zip(ref[1], v[1]))
    assert len(cap.functions) <= 1 and cap.stats["capacity_fallbacks"] > 0
    try:
        with cap.installed():
            raise RuntimeError("restore test")
    except RuntimeError:
        pass
    assert original == (llama.LlamaModel.__call__, DeadHead.__call__)
    stopped = execute("fullhead_027", batches["ragged"], cancel=[1, 2, 1])
    assert [len(x) for x in stopped[0]["output_ids"]] == [1, 2, 1]
    from lossless.adapters.mlx.cache import cache_policy
    from lossless.adapters.mlx.runtime import CACHE_RECIPE

    forced = []
    for name in graphs:
        verifier = GraphRecipe(model, name, 64)
        with cache_policy({**CACHE_RECIPE, "append_metal": False, "append_concat": True}):
            checked = teacher_checks(
                grouped,
                [tok.encode(r["text"], add_special_tokens=False) for r in batches["discovery"][:3]],
                [2, 4, 3],
                candidate_context=verifier.installed,
            )
        assert verifier.stats["builds"] > 0, "forced history must actually exercise capture"
        assert all(r["bitwise"] for r in checked)
        forced.extend(checked)
    # Actual projection shapes for the numerical decomposition track, outside timed runs.
    projection_shapes = []
    for i, layer in enumerate(model.layers[:1]):
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            mod = getattr(layer.self_attn, name)
            projection_shapes.append(
                {
                    "layer": i,
                    "module": name,
                    "packed_weight_shape": list(mod.weight.shape),
                    "bits": getattr(mod, "bits", None),
                }
            )
    summary = {
        "hardware": mx.device_info(),
        "model_identity": model_id,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_load_and_admission_seconds": loading,
        "one_model_hash_seconds": hash_seconds,
        "records": records,
        "capacity_test": cap.stats,
        "forced_history_checks": len(forced),
        "projection_shapes": projection_shapes,
        "elapsed_seconds": time.monotonic() - started,
        "scope": "Whole requests, synchronized only at boundaries; no additive per-layer timing attribution. Exact finite checks; no automatic graph promotion.",
    }
    write(out / "summary.json", summary)
    stamp(
        f"COMPLETE result={out / 'summary.json'} log={out.parent / (out.name + '.log')} elapsed={summary['elapsed_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
