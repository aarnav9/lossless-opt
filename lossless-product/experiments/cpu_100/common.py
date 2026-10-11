"""Evaluator-owned protocol and persistence for sequential CPU experiments."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_RUN = ROOT / ".lossless/cpu-100-20261011"
POLICY = dict(
    minimum_speedup=1.02,
    max_regression=0.03,
    final_pairs=16,
    significance=0.01,
    numerical_atol=1e-10,
    numerical_rtol=1e-8,
)


def stamp(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def tree(root):
    return {
        str(p.relative_to(root)): sha(p)
        for p in sorted(Path(root).rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def score(rows):
    cases = []
    rounds = []
    for row in rows:
        ratios = [sum(p["reference"]["calls"]) / sum(p["candidate"]["calls"]) for p in row["pairs"]]
        call_ratios = [
            [p["reference"]["calls"][i] / p["candidate"]["calls"][i] for p in row["pairs"]]
            for i in range(len(row["pairs"][0]["reference"]["calls"]))
        ]
        setup = {
            v: statistics.median(p[v].get("setup_seconds", 0.0) for p in row["pairs"])
            for v in ("reference", "candidate")
        }
        cold = [
            (p["reference"].get("setup_seconds", 0.0) + sum(p["reference"]["calls"]))
            / (p["candidate"].get("setup_seconds", 0.0) + sum(p["candidate"]["calls"]))
            for p in row["pairs"]
        ]
        saved_per_request = statistics.median(
            (sum(p["reference"]["calls"]) - sum(p["candidate"]["calls"]))
            / len(p["reference"]["calls"])
            for p in row["pairs"]
        )
        cases.append(
            dict(
                id=row["id"],
                speedup=statistics.geometric_mean(ratios),
                ratios=ratios,
                setup_seconds=setup,
                setup_and_calls_speedup=statistics.geometric_mean(cold),
                setup_and_calls_ratios=cold,
                estimated_payback_requests=max(0, setup["candidate"] - setup["reference"])
                / saved_per_request
                if saved_per_request > 0
                else None,
                calls=[statistics.geometric_mean(v) for v in call_ratios],
                median_seconds={
                    v: statistics.median(sum(p[v]["calls"]) for p in row["pairs"])
                    for v in ("reference", "candidate")
                },
            )
        )
    for i in range(len(rows[0]["pairs"])):
        rounds.append(statistics.geometric_mean(c["ratios"][i] for c in cases))
    wins = sum(v > POLICY["minimum_speedup"] for v in rounds)
    n = len(rounds)
    return dict(
        speedup=statistics.geometric_mean(rounds),
        cases=cases,
        rounds=rounds,
        sign_p=sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n,
        no_regression=all(
            min(c["calls"] + [c["speedup"]]) >= 1 / (1 + POLICY["max_regression"]) for c in cases
        ),
    )


def verdict(correctness, timing, final=False):
    category = (
        "exact" if correctness["exact"] else "numerical" if correctness["numerical"] else "rejected"
    )
    speed = timing["speedup"] >= POLICY["minimum_speedup"] and timing["no_regression"]
    if final:
        speed = speed and timing["sign_p"] <= POLICY["significance"]
    return dict(
        category=category,
        speed_eligible=speed,
        eligible=category == "exact" and speed and correctness.get("resource_eligible", True),
        numerical_eligible=category == "numerical" and speed,
        quality="not_evaluated_no_task_labels",
    )
