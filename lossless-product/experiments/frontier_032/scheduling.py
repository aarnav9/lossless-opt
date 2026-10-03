"""Exact subset-DP scheduling under measured cohort costs and explicit buffer budgets."""

import argparse
import json
from pathlib import Path
import random
import statistics
import time
from lossless._native.common import stamp, write


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
    from lossless.adapters.mlx.runtime import generate
    from lossless.adapters.mlx.schedule import cohorts

    model, tok = load(a.model)
    mx.eval(model.parameters())
    grouped = GroupedHead(model, 4)
    original = json.loads(
        (Path(__file__).resolve().parents[2] / "examples/mlx-fixed-count/requests.json").read_text()
    )
    rows = [r for r in original if r["split"] == "discovery"][:6]
    # Two length buckets, heterogeneous completion counts. Preserve tokenizer roundtrips.
    texts = [r["text"] for r in rows]
    counts = [2, 8, 3, 7, 4, 6]
    lengths = [len(tok.encode(t, add_special_tokens=False)) for t in texts]
    for i in [3, 4, 5]:
        texts[i] += " Further details."
    lengths = [len(tok.encode(t, add_special_tokens=False)) for t in texts]
    n = len(texts)
    full = (1 << n) - 1
    arrivals = [0, 0.015, 0.010, 0, 0.020, 0.005]

    def execute(ids, capture=False):
        return generate(
            model,
            grouped,
            tok,
            [texts[i] for i in ids],
            [counts[i] for i in ids],
            optimized=True,
            options={"allocator_cache_mib": 256},
            capture=capture,
        )

    costs = {}
    memory = {}
    raw = {}
    # Every legal cohort is calibrated on discovery only. Model costs include real work.
    for mask in range(1, full + 1):
        ids = [i for i in range(n) if mask >> i & 1]
        if len(set(lengths[i] for i in ids)) != 1:
            continue
        samples = []
        for _ in range(3):
            v = execute(ids)
            samples.append(v[0]["seconds"])
        costs[mask] = statistics.median(samples)
        # Exported KV array storage bytes (including singleton spare capacity), not driver or process RSS.
        memory[mask] = sum(
            int(x.nbytes) for cs in v[1] for c in cs for x in c.state if hasattr(x, "nbytes")
        )
        raw[mask] = samples
        stamp(f"CALIBRATE cohort={ids} seconds={costs[mask]:.4f} exported_storage={memory[mask]}")
    groups = cohorts(lengths, counts, "short_first")
    baseline = [sum(1 << i for i in g) for g in groups]
    # Buffer-lifetime choice: compact/hand off on cohort completion or retain until whole batch returns.
    # Costs for compaction measured separately using the actual exported arrays.
    copy_samples = []
    last = execute(list(range(3)))
    for _ in range(5):
        t = time.perf_counter()
        copied = [
            mx.array(x) for cs in last[1] for c in cs for x in c.state if hasattr(x, "nbytes")
        ]
        mx.eval(copied)
        mx.synchronize()
        copy_samples.append(time.perf_counter() - t)
    compact_cost = statistics.median(copy_samples)
    budget = max(memory[m] for m in baseline)
    # Enumerate small exact state space: subset, completion time, retained bytes.
    visited = 0

    def solve(done, t, held, score, path):
        nonlocal visited
        visited += 1
        if done == full:
            return score, path, t
        best = (float("inf"), [], 0)
        for mask, cost in costs.items():
            if mask & done:
                continue
            ids = [i for i in range(n) if mask >> i & 1]
            if held + memory[mask] > budget:
                continue
            start = max(t, max(arrivals[i] for i in ids))
            for release in [True, False]:
                end = start + cost + (compact_cost if release else 0)
                value = solve(
                    done | mask,
                    end,
                    0 if release else held + memory[mask],
                    score + sum(end - arrivals[i] for i in ids),
                    path + [(mask, release)],
                )
                if value[0] < best[0]:
                    best = value
        return best

    t = time.perf_counter()
    optimum, path, makespan = solve(0, 0, 0, 0, [])
    planning = time.perf_counter() - t
    assert path and sum(mask for mask, _ in path) == full

    def predicted(order):
        t = 0
        score = 0
        for mask in order:
            ids = [i for i in range(n) if mask >> i & 1]
            t = max(t, max(arrivals[i] for i in ids)) + costs[mask] + compact_cost
            score += sum(t - arrivals[i] for i in ids)
        return score, t

    # Replay both fixed schedules on actual requests; arrival waits included, no candidate adjustment.
    frozen = {"baseline": [(m, True) for m in baseline], "solver": path}
    samples = {k: [] for k in frozen}
    checks = 0
    refs = {i: execute([i], True) for i in range(n)}
    from lossless.adapters.mlx.checks import cache_diff, diff

    for rep in range(7):
        order = list(frozen)
        random.Random(320 + rep).shuffle(order)
        for name in order:
            t = time.perf_counter()
            latencies = []
            held_outputs = []
            for mask, release in frozen[name]:
                ids = [i for i in range(n) if mask >> i & 1]
                wait = max(arrivals[i] for i in ids) - (time.perf_counter() - t)
                if wait > 0:
                    time.sleep(wait)
                v = execute(ids, rep == 0)
                held_outputs.append(v[1])
                if release:
                    copied = [
                        mx.array(x)
                        for group in held_outputs
                        for cs in group
                        for c in cs
                        for x in c.state
                        if hasattr(x, "nbytes")
                    ]
                    mx.eval(copied)
                    mx.synchronize()
                    held_outputs.clear()
                    del copied
                elapsed = time.perf_counter() - t
                latencies.extend(elapsed - arrivals[i] for i in ids)
                if rep == 0:
                    for j, i in enumerate(ids):
                        assert refs[i][0]["output_ids"][0] == v[0]["output_ids"][j]
                        assert (
                            diff(refs[i][2][0], v[2][j])["bitwise"]
                            and cache_diff(refs[i][1][0], v[1][j])["bitwise"]
                        )
                        checks += 1
            samples[name].append(
                {
                    "makespan": time.perf_counter() - t,
                    "mean_completion_latency": statistics.mean(latencies),
                }
            )
        stamp(f"REPLAY repeat={rep + 1}/7")
    write(
        out / "summary.json",
        {
            "calibration": raw,
            "cohort_exported_storage_bytes": memory,
            "buffer_budget": budget,
            "compact_cost_proxy_seconds": compact_cost,
            "baseline": baseline,
            "solver_schedule": path,
            "predicted_baseline": predicted(baseline),
            "predicted_solver_sum_latency": optimum,
            "planning_seconds": planning,
            "states_visited": visited,
            "samples": samples,
            "exact_checks": checks,
            "scope": "Six-request exhaustive schedule/lifetime model, measured cohort costs. Optimality applies only to this enumerated model. mx.array is a copy/materialization proxy; replay returns one cohort at a time and does not implement a production streaming allocator. No global runtime optimality claim.",
        },
    )
    stamp(f"COMPLETE result={out / 'summary.json'} log={out.parent / (out.name + '.log')}")


if __name__ == "__main__":
    main()
