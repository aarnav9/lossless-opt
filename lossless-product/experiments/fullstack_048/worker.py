"""Fixed GPU worker: complete requests, independent byte oracles, bounded measurements."""

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import random
import statistics
import time
import traceback

from protocol import METHODS, read, write
from lossless._native.common import stamp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    job = read(a.job)
    import numpy as np
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.generate import generation_stream, BatchGenerator
    from mlx_lm.models import llama
    from mlx_lm.models.cache import BatchKVCache, KVCache
    from lossless.adapters.mlx import batching, cache, execution, schedule
    from lossless.adapters.mlx.model import GroupedHead
    from lossless.adapters.mlx.graph import GraphRecipe
    from lossless.adapters.mlx.runtime import admission, allocator, CACHE_RECIPE
    from lossless.adapters.mlx.identity import fingerprint

    started = time.monotonic()
    model_id = fingerprint(job["model"])
    assert admission(job["model"], mx.device_info()["device_name"], model_identity=model_id)[0]
    model, tokenizer = load(job["model"])
    mx.eval(model.parameters())
    loading = time.monotonic() - started

    def weights():
        h = hashlib.sha256()
        for name, value in tree_flatten(model.parameters()):
            h.update(name.encode())
            h.update(np.array(value).tobytes())
        return h.hexdigest()

    original_weights = weights()
    watch = [
        (mx, n)
        for n in [
            "eval",
            "async_eval",
            "synchronize",
            "clear_cache",
            "logsumexp",
            "get_peak_memory",
            "reset_peak_memory",
        ]
    ]
    watch += [
        (BatchKVCache, n)
        for n in ["merge", "filter", "extract", "update_and_fetch", "prepare", "extend"]
    ]
    watch += [
        (llama.LlamaModel, "__call__"),
        (execution.DeadHead, "__call__"),
        (execution, "scheduled"),
        (schedule, "cohorts"),
        (cache, "run"),
        (batching, "native"),
        (batching, "serial"),
    ]
    originals = {(id(o), n): inspect.getattr_static(o, n) for o, n in watch}

    def restored():
        assert all(inspect.getattr_static(o, n) is originals[id(o), n] for o, n in watch), (
            "global implementation override leaked outside candidate call"
        )

    class Baseline:
        def __init__(self, name):
            self.name = name
            self.grouped = GroupedHead(model, 4)
            self.graph = GraphRecipe(model, name, 64) if name in METHODS[3:] else None

        def run(self, prompts, counts, capture=True):
            if self.name == "serial":
                return batching.serial(model, prompts, counts, capture)
            if self.name == "stock_batch":
                return batching.native(model, prompts, counts, capture)
            recipe = CACHE_RECIPE
            from contextlib import nullcontext

            if self.graph:
                recipe = {**recipe, "append_metal": False, "append_concat": True}
            with (
                allocator({"allocator_cache_mib": 256}),
                self.graph.installed() if self.graph else nullcontext(),
            ):
                return cache.run(self.grouped, prompts, counts, recipe, capture)

    runners = {n: Baseline(n) for n in METHODS}
    setup_seconds = {}
    if job.get("candidate"):
        begin = time.perf_counter()
        spec = importlib.util.spec_from_file_location("candidate_048", job["candidate"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runners["candidate"] = module.Runner(model)
        setup_seconds["candidate"] = time.perf_counter() - begin
        restored()

    def sync():
        mx.synchronize(generation_stream)
        mx.synchronize()

    def array_sig(value):
        if "bfloat16" in str(value.dtype):
            raw = np.array(value.view(mx.uint16))
        else:
            raw = np.array(value)
        return dict(
            shape=list(value.shape),
            dtype=str(value.dtype),
            sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
        )

    def cache_sig(states):
        return [
            [
                dict(offset=int(c.offset), arrays=[array_sig(v) for v in c.keys_and_values()])
                for c in cs
            ]
            for cs in states
        ]

    def signature(value, continuations=False):
        row, states, probabilities = value
        result = dict(
            tokens=[list(ids) for ids in row["output_ids"]],
            probabilities=[array_sig(v) for v in probabilities],
            kv=cache_sig(states),
            texts=list(row["output_texts"]),
        )
        if continuations:
            checks = []
            for i, original in enumerate(states):
                copied = copy.deepcopy(original)
                logits = []
                for step in range(3):
                    v = model(
                        mx.array([[(17 + i * 31 + step * 47) % model.args.vocab_size]]),
                        cache=copied,
                    )
                    mx.eval(v, [c.state for c in copied])
                    logits.append(array_sig(v))
                checks.append(dict(logits=logits, kv=cache_sig([copied])))
            result["continuations"] = checks
        return result

    def call(runner, case):
        restored()
        sync()
        mx.reset_peak_memory()
        start = time.perf_counter()
        prompts = [tokenizer.encode(t, add_special_tokens=False) for t in case["texts"]]
        before = json.dumps(prompts)
        counts = list(case["counts"])
        encoded = time.perf_counter()
        value = runner.run(prompts, counts, capture=True)
        executed = time.perf_counter()
        row, states, probabilities = value
        assert len(states) == len(probabilities) == len(prompts)
        assert [len(ids) for ids in row["output_ids"]] == counts
        mx.eval([c.state for cs in states for c in cs], probabilities)
        sync()
        materialized = time.perf_counter()
        row["output_texts"] = [tokenizer.decode(ids) for ids in row["output_ids"]]
        json.dumps(
            {
                "text": row["output_texts"],
                "tokens": row["output_ids"],
                "finish_reasons": ["length"] * len(counts),
            }
        )
        seconds = time.perf_counter() - start
        peak = mx.get_peak_memory()
        restored()
        assert before == json.dumps(prompts) and counts == case["counts"], "input mutation"
        assert len(row["ttft_seconds"]) == len(counts)
        assert all(0 <= t <= seconds * 1.05 + 0.001 for t in row["ttft_seconds"]), (
            "invalid backend timing"
        )
        metrics = dict(
            seconds=seconds,
            peak_bytes=peak,
            max_reported_ttft_seconds=max(row["ttft_seconds"]) + encoded - start,
            stages_seconds=dict(
                tokenization=encoded - start,
                execution=executed - encoded,
                materialization=materialized - executed,
                response=seconds - (materialized - start),
            ),
        )
        return value, metrics

    oracle = read(job["oracle"])["oracles"] if job.get("oracle") else {}
    records = []
    generated = {}
    for index, case in enumerate(job["cases"]):
        stamp(
            f"048 worker phase={job['phase']} case={index + 1}/{len(job['cases'])} id={case['id']}"
        )
        if job["phase"] == "oracle":
            value, metrics = call(runners["serial"], case)
            generated[case["id"]] = signature(value, True)
            del value
            records.append(dict(case=case, metrics=metrics))
            continue
        names = METHODS if job["phase"] == "baseline" else ["candidate", "reference", "serial"]
        if "reference" in names:
            runners["reference"] = runners[job["references"][case["key"]]]
        record = dict(
            case=case,
            exact={},
            first_call={},
            errors={},
            samples={n: [] for n in names},
            peaks={n: [] for n in names},
            ttft={n: [] for n in names},
            orders=[],
            checks={},
        )
        held = {}
        held_hash = {}
        for name in names:
            try:
                value, metrics = call(runners[name], case)
                sig = signature(value, True)
                exact = sig == oracle[case["id"]]
                record["checks"][name] = {k: sig[k] == oracle[case["id"]][k] for k in sig}
                # Mutating one exported request must not change other requests.
                other_before = cache_sig(value[1][1:])
                if len(value[1]) > 1:
                    c = value[1][0][0]
                    c.keys[..., 0, :] = c.keys[..., 0, :] + 1
                    mx.eval(c.state)
                independent = other_before == cache_sig(value[1][1:])
                record["checks"][name]["independent_exports"] = independent
                record["exact"][name] = exact and independent
                record["first_call"][name] = metrics
                held[name] = value
                held_hash[name] = signature(value)
            except Exception:
                record["errors"][name] = traceback.format_exc(limit=6)
                record["exact"][name] = False
                restored()
        for repeat in range(job["repeats"]):
            order = [n for n in names if record["exact"].get(n)]
            random.Random(
                480000 + job.get("process_index", 0) * 1000 + index * 100 + repeat
            ).shuffle(order)
            record["orders"].append(order)
            for name in order:
                value, metrics = call(runners[name], case)
                expected = {k: v for k, v in oracle[case["id"]].items() if k != "continuations"}
                if signature(value) != expected or signature(held[name]) != held_hash[name]:
                    record["exact"][name] = False
                    record["errors"][name] = "warm output or retained previous-call state changed"
                record["samples"][name].append(metrics["seconds"])
                record["peaks"][name].append(metrics["peak_bytes"])
                record["ttft"][name].append(metrics["max_reported_ttft_seconds"])
                del value
            stamp(f"048 worker case={case['id']} batch={repeat + 1}/{job['repeats']}")
        record["median_seconds"] = {
            n: statistics.median(v) for n, v in record["samples"].items() if v
        }
        records.append(record)
        write(a.output, dict(status="running", records=records))
        held.clear()
        mx.clear_cache()
    assert weights() == original_weights, "model parameters mutated"
    restored()
    result = dict(
        status="complete",
        records=records,
        oracles=generated,
        hardware=mx.device_info(),
        model_identity=model_id,
        model_loading_seconds=loading,
        setup_seconds=setup_seconds,
        model_parameters_sha256=original_weights,
        weights_unchanged=True,
        wall_seconds=time.monotonic() - started,
    )
    if job["phase"] == "oracle" and job.get("include_source"):
        from mlx_lm.models import cache as upstream_cache

        result["upstream_source"] = {
            "BatchGenerator": inspect.getsource(BatchGenerator),
            "BatchKVCache": inspect.getsource(upstream_cache.BatchKVCache),
            "KVCache": inspect.getsource(KVCache),
        }
    write(a.output, result)
    stamp(f"048 worker COMPLETE result={a.output}")


if __name__ == "__main__":
    main()
