"""Frozen MLX workload matrix with a contract-checked stock BatchGenerator control."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import time

from lossless._native.common import read, write, sha, stamp, interval

METHODS = ["serial", "stock_batch", "retained", "decoder_026", "fullhead_027"]


def freeze(out, natural=False):
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    for split in ["discovery", "evaluation"]:
        offset = int(split == "evaluation")
        for size in [1, 2, 4, 8]:
            for regime in ["short", "long", "mixed"]:
                lengths = (
                    [8 + offset * 2] * size
                    if regime == "short"
                    else [96 + offset * 16] * size
                    if regime == "long"
                    else [[8, 32, 96, 192][i % 4] + offset * 8 for i in range(size)]
                )
                counts = (
                    [4 + offset] * size
                    if regime == "short"
                    else [12 + offset * 2] * size
                    if regime == "long"
                    else [[2, 5, 9, 16][i % 4] + offset for i in range(size)]
                )
                # Synthetic deterministic token-length profiles, disclosed rather than claimed as production traces.
                texts = [
                    (" river" if split == "discovery" else " forest") * length for length in lengths
                ]
                cases.append(
                    dict(
                        id=f"{split}_{size}_{regime}",
                        split=split,
                        batch_size=size,
                        regime=regime,
                        texts=texts,
                        counts=counts,
                    )
                )
    if natural:
        prompts = {
            "discovery": [
                "Explain why rainbows form.",
                "Write a Python function that returns the sum of the even numbers in a list. Include one example call.",
                "Resume en español: los árboles absorben dióxido de carbono y proporcionan refugio a muchos animales.",
                "A shop has 18 red notebooks and 27 blue notebooks. It sells one third of each kind. How many notebooks remain? Show the arithmetic.",
                "Return a JSON object with keys name, units and available for a product called copper wire with twelve units in stock.",
                "Compare breadth-first search and depth-first search. Discuss their memory use, traversal order and the conditions under which each finds a shortest path in a graph.",
                "Translate into French: The meeting has moved to Thursday afternoon. Please bring the revised report and a list of questions.",
                "A gardener records temperature, rainfall and soil moisture every morning for a month. Explain how to check the records for missing values and unusual readings before drawing conclusions about plant growth. Give a careful sequence of practical steps.",
            ],
            "evaluation": [
                "Explain why the ocean has tides.",
                "Write a Python function that removes duplicate strings from a list while preserving their original order. Include a small example.",
                "Resume en español: las abejas transportan polen entre flores y contribuyen a la producción de frutas y semillas.",
                "A library has 24 history books and 35 science books on a cart. It shelves half the history books and one fifth of the science books. How many remain? Show the arithmetic.",
                "Return a JSON object with keys city, temperature and raining describing Oslo at seven degrees with no rain.",
                "Compare merge sort and insertion sort. Discuss time complexity, extra memory, stability and which input sizes or orderings make each useful.",
                "Translate into German: The train leaves at nine in the morning. Keep your ticket ready and check the platform before boarding.",
                "A researcher surveys households about commute time, transport mode and working hours. Explain how to inspect the responses for missing entries, inconsistent units and selection bias before estimating average travel time. Give a careful sequence of practical steps.",
            ],
        }
        cases = [
            dict(
                id=f"{split}_{size}_natural",
                split=split,
                batch_size=size,
                regime="natural",
                texts=prompts[split][:size],
                counts=[3, 6, 9, 12, 5, 8, 11, 16][:size],
            )
            for split in prompts
            for size in [1, 2, 4, 8]
        ]
    write(
        out / "plan.json",
        dict(
            schema_version=1,
            created_utc=datetime.now(timezone.utc).isoformat(),
            cases=cases,
            methods=METHODS,
            repeats=7,
            script_sha256=sha(__file__),
            rule="Freeze fastest exact library baseline and product choice from discovery per regime. Evaluation tests those choices unchanged. Product discovery: median >=1.02 vs comparator and lower paired bound>1; graph also requires lower bound>1.01 vs retained. Final rule same, with exact tokens/probabilities/KV. Raw stock batching remains diagnostic if its exact contract fails. No automatic default promotion.",
            scope="Two processes: discovery then fresh evaluation. Curated natural-language/code tasks"
            if natural
            else "Two processes: discovery then fresh evaluation. 24 synthetic workload profiles; one pinned device/model, no production arrival trace or new-device qualification.",
        ),
    )


def execute(out, model_path, split):
    plan = read(out / "plan.json")
    assert plan["script_sha256"] == sha(__file__), "study changed after freeze"
    folder = out / split
    folder.mkdir(exist_ok=False)
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generation_stream
    from lossless.adapters.mlx.model import GroupedHead
    from lossless.adapters.mlx.graph import GraphRecipe
    from lossless.adapters.mlx.runtime import generate, admission, runtime_identity
    from lossless.adapters.mlx.identity import fingerprint
    from lossless.adapters.mlx.batching import native
    from lossless.adapters.mlx.checks import diff, cache_diff
    from lossless.constraints import measurements

    started = time.monotonic()
    model_identity = fingerprint(model_path)
    assert admission(model_path, mx.device_info()["device_name"], model_identity=model_identity)[0]
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    grouped = GroupedHead(model, 4)
    loading = time.monotonic() - started
    selections = read(out / "selection.json") if split == "evaluation" else {}
    records = []
    for case_index, case in enumerate(c for c in plan["cases"] if c["split"] == split):
        graphs = {n: GraphRecipe(model, n, 64) for n in METHODS[3:]}

        def call(name, capture, graphs=graphs):
            if name != "stock_batch":
                return generate(
                    model,
                    grouped,
                    tokenizer,
                    case["texts"],
                    case["counts"],
                    optimized=name != "serial",
                    options={
                        "allocator_cache_mib": 256,
                        "execution_recipe": name if name in graphs else "retained",
                    },
                    graph=graphs.get(name),
                    capture=capture,
                )
            mx.synchronize(generation_stream)
            mx.synchronize()
            start = time.perf_counter()
            prompts = [tokenizer.encode(text, add_special_tokens=False) for text in case["texts"]]
            encoded = time.perf_counter() - start
            row, caches, probabilities = native(model, prompts, case["counts"], capture=capture)
            mx.eval([c.state for cs in caches for c in cs])
            mx.synchronize(generation_stream)
            mx.synchronize()
            row["output_texts"] = [tokenizer.decode(ids) for ids in row["output_ids"]]
            row["finish_reasons"] = ["length"] * len(prompts)
            row["response_bytes"] = len(
                json.dumps(
                    {
                        "text": row["output_texts"],
                        "tokens": row["output_ids"],
                        "finish_reasons": row["finish_reasons"],
                    },
                    ensure_ascii=False,
                ).encode()
            )
            row["ttft_seconds"] = [v + encoded for v in row["ttft_seconds"]]
            row["completion_seconds"] = [v + encoded for v in row["completion_seconds"]]
            row["seconds"] = time.perf_counter() - start
            row["tokens_per_second"] = sum(case["counts"]) / row["seconds"]
            return row, caches, probabilities

        reference = call("serial", True)
        expected_ids = reference[0]["output_ids"]
        checks, cold, errors = {}, {}, {}
        for name in METHODS:
            try:
                value = call(name, True)
                checks[name] = [
                    dict(
                        tokens=expected_ids[i] == value[0]["output_ids"][i],
                        probabilities=diff(reference[2][i], value[2][i])["bitwise"],
                        kv=cache_diff(reference[1][i], value[1][i])["bitwise"],
                    )
                    for i in range(case["batch_size"])
                ]
                cold[name] = value[0]["seconds"]
                del value
            except Exception as error:
                errors[name] = type(error).__name__ + ": " + str(error)
                checks[name] = []
        del reference
        exact = {n: bool(checks[n]) and all(all(c.values()) for c in checks[n]) for n in METHODS}
        assert exact["serial"], "reference is not repeatable"
        samples = {n: [] for n in METHODS if n not in errors}
        metrics = {n: [] for n in samples}
        peaks = {n: [] for n in samples}
        orders = []
        for repeat in range(plan["repeats"]):
            order = list(samples)
            random.Random(41000 + case_index * 100 + repeat).shuffle(order)
            orders.append(order)
            for name in order:
                mx.reset_peak_memory()
                value = call(name, False)
                row = value[0]
                if row["output_ids"] != expected_ids:
                    exact[name] = False
                samples[name].append(row["seconds"])
                peaks[name].append(mx.get_peak_memory())
                metrics[name].append(
                    {k: row[k] for k in ["tokens_per_second", "ttft_seconds", "completion_seconds"]}
                )
                del value
            stamp(f"MLX split={split} case={case_index + 1} repeat={repeat + 1}/7 id={case['id']}")
        medians = {n: statistics.median(v) for n, v in samples.items()}
        key = f"{case['batch_size']}_{case['regime']}"
        if split == "discovery":
            comparator = min(
                [n for n in ["serial", "stock_batch"] if exact[n]], key=lambda n: medians[n]
            )
            eligible = [
                n
                for n in METHODS[2:]
                if exact[n]
                and medians[comparator] / medians[n] >= 1.02
                and interval(samples[comparator], samples[n], 41)[0] > 1
                and (n == "retained" or interval(samples["retained"], samples[n], 41)[0] > 1.01)
            ]
            selected = min(eligible, key=lambda n: medians[n]) if eligible else comparator
            selections[key] = dict(comparator=comparator, selected=selected)
        else:
            comparator, selected = (selections[key][k] for k in ["comparator", "selected"])
        confirmed = (
            selected in METHODS[2:]
            and exact[selected]
            and exact[comparator]
            and medians[comparator] / medians[selected] >= 1.02
            and interval(samples[comparator], samples[selected], 41)[0] > 1
            and (
                selected == "retained"
                or interval(samples["retained"], samples[selected], 41)[0] > 1.01
            )
        )
        record = dict(
            case=case,
            exact=exact,
            checks=checks,
            errors=errors,
            first_validation_seconds=cold,
            samples_seconds=samples,
            orders=orders,
            deployment_metrics={n: measurements(metrics[n], peaks[n]) for n in samples},
            deployment_samples=metrics,
            peak_mlx_samples_bytes=peaks,
            median_seconds=medians,
            frozen_selection=selections[key],
            confirmed=confirmed,
            speedup_vs_frozen_comparator={n: medians[comparator] / v for n, v in medians.items()},
            ci95_vs_comparator={
                n: interval(samples[comparator], v, 41) for n, v in samples.items()
            },
            graph={
                n: dict(**g.stats, entries=len(g.functions), capacity=g.capacity)
                for n, g in graphs.items()
            },
            prompt_lengths=[
                len(tokenizer.encode(t, add_special_tokens=False)) for t in case["texts"]
            ],
        )
        records.append(record)
        write(folder / (case["id"] + ".json"), record)
        del graphs
        mx.clear_cache()
    if split == "discovery":
        write(out / "selection.json", selections)
    write(
        folder / "summary.json",
        dict(
            split=split,
            records=records,
            hardware=mx.device_info(),
            model_identity=model_identity,
            runtime_identity=runtime_identity(),
            model_loading_seconds=loading,
            wall_seconds=time.monotonic() - started,
            plan_sha256=sha(out / "plan.json"),
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["freeze", "discovery", "evaluation"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model")
    p.add_argument(
        "--natural",
        action="store_true",
        help="Freeze the supplementary natural-language/code workloads",
    )
    a = p.parse_args()
    out = a.output.resolve()
    stamp(
        f"START phase={a.phase} result={out} log={out.parent / (out.name + '_' + a.phase + '.log')}"
    )
    if a.phase == "freeze":
        freeze(out, a.natural)
    else:
        execute(out, a.model, a.phase)
    stamp(
        f"COMPLETE phase={a.phase} result={out} log={out.parent / (out.name + '_' + a.phase + '.log')}"
    )


if __name__ == "__main__":
    main()
