"""Actual timing race with a conservative anytime-valid bounded-score rule."""

import argparse
import math
from pathlib import Path
import random
import statistics
import time
from lossless._native import operators
from lossless._native.common import stamp, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--search", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    stamp(f"START estimated=under2min result={out} log={out.parent / (out.name + '.log')}")
    paths = {
        "retained": Path(a.search) / "retained_0/build/softmax_column_4.dylib",
        "manual": Path(a.search) / "manual_codex_0/build/layout_dispatch.dylib",
        "unroll32": Path(a.search) / "enumerated_0/build/unroll_32.dylib",
    }
    arrays = operators.input_arrays("softmax", {"rows": 96, "columns": 311, "layout": "f"}, 934)
    libraries = {n: operators.load_library(p) for n, p in paths.items()}
    funcs = {n: operators.bind_native(arrays, l) for n, l in libraries.items()}
    expected = operators.reference("softmax", arrays)
    assert all(operators.check("softmax", f(), expected)["passed"] for f in funcs.values())

    # A deliberately losing but correct control tests early-rejection power.
    def slow():
        value = funcs["unroll32"]()
        for _ in range(7):
            value = funcs["unroll32"]()
        return value

    funcs["slow_control"] = slow
    samples = {n: [] for n in funcs}
    timings = {n: [] for n in funcs}
    orders = []
    count = 768
    alpha = 0.05
    k = len(funcs)
    stops = {}
    for block in range(count):
        order = list(funcs)
        random.Random(932 + block).shuffle(order)
        orders.append(order)
        for name in order:
            # Randomized paired order, three invocations per sample.
            pair = ["base", "candidate"]
            random.Random(223 + block * 7 + order.index(name)).shuffle(pair)
            v = {}
            for which in pair:
                f = funcs["retained"] if which == "base" else funcs[name]
                t = time.perf_counter_ns()
                for _ in range(3):
                    f()
                v[which] = (time.perf_counter_ns() - t) / 3e9
            score = (v["base"] - v["candidate"]) / (v["base"] + v["candidate"])
            samples[name].append(score)
            timings[name].append(v)
            n = block + 1
            # Hoeffding on [-1,1], alpha spending across all arms and all sample counts.
            radius = math.sqrt(2 * math.log(2 * k * n * (n + 1) / alpha) / n)
            upper = statistics.mean(samples[name]) + radius
            if upper < 0 and name not in stops:
                stops[name] = n
        if (block + 1) % 128 == 0:
            stamp(f"RACE block={block + 1}/{count} rejected={stops}")
    fixed_cost = sum(sum(v["base"] + v["candidate"] for v in ts) for ts in timings.values())
    race_cost = sum(
        sum(v["base"] + v["candidate"] for v in ts[: stops.get(n, count)])
        for n, ts in timings.items()
    )
    survivors = [n for n in funcs if n not in stops]
    fixed = min(funcs, key=lambda n: statistics.median(v["candidate"] for v in timings[n]))
    raced = min(
        survivors,
        key=lambda n: statistics.median(v["candidate"] for v in timings[n][: stops.get(n, count)]),
    )
    # Separate later validation samples, no further selection.
    confirmation = {}
    for name in {fixed, raced}:
        a_samples = []
        b_samples = []
        for rep in range(31):
            t = time.perf_counter_ns()
            funcs["retained"]()
            a_samples.append((time.perf_counter_ns() - t) / 1e9)
            t = time.perf_counter_ns()
            funcs[name]()
            b_samples.append((time.perf_counter_ns() - t) / 1e9)
        confirmation[name] = {
            "speedup_vs_retained": statistics.median(a_samples) / statistics.median(b_samples),
            "reference": a_samples,
            "candidate": b_samples,
        }
    # Actual fixed-budget and live-racing runs, including decision overhead.
    live_results = {}
    for mode in ["fixed", "racing"]:
        live = {n: [] for n in funcs}
        sums = {n: 0.0 for n in funcs}
        active = set(funcs)
        rejected = {}
        start = time.perf_counter()
        for step in range(count):
            order = sorted(active)
            random.Random(3300 + step).shuffle(order)
            for name in order:
                pair = ["base", "candidate"]
                random.Random(550 + step).shuffle(pair)
                values = {}
                for which in pair:
                    f = funcs["retained"] if which == "base" else funcs[name]
                    t = time.perf_counter_ns()
                    for _ in range(3):
                        f()
                    values[which] = (time.perf_counter_ns() - t) / 3e9
                live[name].append(values)
                n = len(live[name])
                sums[name] += (values["base"] - values["candidate"]) / (
                    values["base"] + values["candidate"]
                )
                if (
                    mode == "racing"
                    and sums[name] / n + math.sqrt(2 * math.log(2 * k * n * (n + 1) / alpha) / n)
                    < 0
                ):
                    active.remove(name)
                    rejected[name] = n
        elapsed = time.perf_counter() - start
        winner = min(active, key=lambda name: statistics.median(v["candidate"] for v in live[name]))
        live_results[mode] = {
            "wall_seconds": elapsed,
            "winner": winner,
            "rejected": rejected,
            "pairs": sum(map(len, live.values())),
            "samples": live,
        }
        stamp(f"LIVE mode={mode} seconds={elapsed:.3f} winner={winner} rejected={rejected}")
    write(
        out / "summary.json",
        {
            "live_runs": live_results,
            "fixed_winner": fixed,
            "racing_winner": raced,
            "early_rejections": stops,
            "fixed_measurement_seconds": fixed_cost,
            "replayed_racing_measurement_seconds": race_cost,
            "saved_measurement_fraction": 1 - race_cost / fixed_cost,
            "confirmation": confirmation,
            "samples": timings,
            "rule": "Hoeffding bounded pair score, alpha/(K*n*(n+1)), stop only if upper confidence bound < 0",
            "scope": "Logged full randomized timing stream with an offline replay of stopping decisions. Sampling-cost savings are counterfactual, not wall-clock savings from a live stopped run. Validity requires independent observations with stable means; device drift is not certified away.",
        },
    )
    stamp(f"COMPLETE result={out / 'summary.json'} log={out.parent / (out.name + '.log')}")


if __name__ == "__main__":
    main()
