import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stamp(message):
    print(f"[{now()}] {message}", flush=True)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def geomean(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def interval(base, candidate, seed):
    import numpy as np

    a = np.asarray(base)
    b = np.asarray(candidate)
    ix = np.random.default_rng(seed).integers(0, len(a), (2000, len(a)))
    return [
        float(v)
        for v in np.quantile(np.median(a[ix], axis=1) / np.median(b[ix], axis=1), [0.025, 0.975])
    ]


def measure(funcs, seed, blocks, target_ns):
    import time
    import numpy as np

    repetitions = {}
    cost = {}
    samples = {name: [] for name in funcs}
    orders = []
    for name, fn in funcs.items():
        started = time.perf_counter_ns()
        fn()
        fn()
        single = []
        for _ in range(3):
            t = time.perf_counter_ns()
            fn()
            single.append(time.perf_counter_ns() - t)
        repetitions[name] = max(1, min(5000, int(target_ns / max(np.median(single), 1))))
        cost[name] = time.perf_counter_ns() - started
    rng = np.random.default_rng(seed)
    for _ in range(blocks):
        order = rng.permutation(list(funcs)).tolist()
        orders.append(order)
        for name in order:
            started = time.perf_counter_ns()
            for _ in range(repetitions[name]):
                funcs[name]()
            elapsed = time.perf_counter_ns() - started
            samples[name].append(elapsed / repetitions[name] / 1000)
            cost[name] += elapsed
    return {
        "samples_us": samples,
        "median_us": {n: float(np.median(v)) for n, v in samples.items()},
        "repetitions": repetitions,
        "orders": orders,
        "measurement_cost_ms": {n: v / 1e6 for n, v in cost.items()},
    }
