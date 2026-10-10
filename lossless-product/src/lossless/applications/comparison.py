"""Paired, fresh-process measurements of two frozen callable implementations."""

import math
import os
from pathlib import Path
import random
import statistics
import tempfile

from .._native.common import write
from ..jobs import identity, read_json
from .discovery import digest
from .replay import copy_project
from .runtime import clock, execute


def run_case(root, snapshot, value, case, folder, deadline, progress, mode="timing"):
    folder.mkdir(parents=True)
    left = deadline - clock()
    if left <= 0:
        raise TimeoutError("application search deadline exhausted")
    env = {
        k: v
        for k, v in os.environ.items()
        if k in {"PATH", "LANG", "SYSTEMROOT", *value["environment"]["inherit_env"]}
    }
    missing = set(value["environment"]["inherit_env"]) - set(env)
    if missing:
        raise ValueError("missing requested environment variables: " + ", ".join(sorted(missing)))
    with tempfile.TemporaryDirectory(prefix="workspace-", dir=folder) as temporary:
        temporary = Path(temporary)
        workspace = temporary / "project"
        copy_project(root, workspace, snapshot)
        home = temporary / "home"
        home.mkdir()
        payload = {
            "workspace": str(workspace),
            "entry": value["entry"],
            "case": case,
            "seed": value["replay"]["seed"],
            "max_output_bytes": value["replay"]["max_output_bytes"],
            "profile_mode": mode,
            "retain_outputs": True,
            "result": str(folder / "worker_result.json"),
        }
        write(folder / "payload.json", payload)
        left = deadline - clock()
        if left <= 0:
            raise TimeoutError("deadline exhausted during workspace setup")
        receipt = execute(
            [
                value["environment"]["python"],
                "-s",
                str(Path(__file__).with_name("profile_worker.py")),
                str(folder / "payload.json"),
            ],
            cwd=workspace,
            env={
                **env,
                "HOME": str(home),
                "TMPDIR": str(temporary) + "/",
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": str(payload["seed"]),
            },
            stdin="",
            folder=folder,
            timeout=min(left, value["replay"]["timeout_seconds"]),
            max_output_bytes=value["replay"]["max_output_bytes"],
            progress=progress,
        )
        write(folder / "receipt.json", receipt)
        if receipt["status"] != "completed":
            error_path = folder / "worker_result.json"
            error = read_json(error_path) if error_path.exists() else {}
            raise ValueError(
                f"worker {receipt['status']}: {error.get('error', '')}: {error.get('message', '')}"
            )
        result = read_json(folder / "worker_result.json")
        if result.get("status") != "completed" or len(result.get("calls", [])) != len(
            case["calls"]
        ):
            raise ValueError("incomplete application worker result")
        if any(
            not (workspace / name).is_file() or digest(workspace / name) != expected["sha256"]
            for name, expected in snapshot["files"].items()
        ):
            raise ValueError("application modified frozen project files")
    observation = {
        "calls": [row["observation"] for row in result["calls"]],
        "retained_outputs": result.get("retained_outputs"),
    }
    times = [
        result["sequence_seconds"],
        result["setup_seconds"],
        *[r["measurement"]["call_seconds"] for r in result["calls"]],
    ]
    if any(not math.isfinite(t) or t < 0 for t in times) or result["sequence_seconds"] <= 0:
        raise ValueError("invalid measured times")
    return {
        "fingerprint": identity(observation),
        "call_fingerprints": [identity(row) for row in observation["calls"]],
        "retained_fingerprint": identity(observation["retained_outputs"]),
        "sequence_seconds": result["sequence_seconds"],
        "setup_and_sequence_seconds": result["setup_seconds"] + result["sequence_seconds"],
        "call_seconds": times[2:],
        "setup_seconds": result["setup_seconds"],
        "first_call_seconds": times[2],
        "process_seconds": receipt["elapsed_seconds"],
        "peak_rss_bytes": max(r["measurement"]["process_peak_rss_bytes"] for r in result["calls"]),
        "traced_peak_bytes": max(
            (
                r["measurement"].get("traced_memory", {}).get("peak_bytes_above_start", 0)
                for r in result["calls"]
            )
        ),
    }


def statistics_for(rows, minimum_speedup, timing_scope="calls"):
    if timing_scope not in {"calls", "setup_and_calls"}:
        raise ValueError("application timing_scope must be calls or setup_and_calls")
    metric = "sequence_seconds" if timing_scope == "calls" else "setup_and_sequence_seconds"
    cases, rounds = [], []
    count = len(rows[0]["pairs"])
    for row in rows:
        pairs = row["pairs"]
        if len(pairs) != count:
            raise ValueError("incomplete timing pairs")
        ratios = [p["reference"][metric] / p["candidate"][metric] for p in pairs]
        medians = {
            variant: {
                key: statistics.median(p[variant][key] for p in pairs)
                for key in (
                    "sequence_seconds",
                    "setup_seconds",
                    "first_call_seconds",
                    "process_seconds",
                    "peak_rss_bytes",
                )
            }
            for variant in ("reference", "candidate")
        }
        for variant in medians:
            medians[variant]["setup_and_sequence_seconds"] = statistics.median(
                p[variant].get(
                    "setup_and_sequence_seconds",
                    p[variant]["setup_seconds"] + p[variant]["sequence_seconds"],
                )
                for p in pairs
            )
        call_stats = None
        if all("call_seconds" in p[v] for p in pairs for v in ("reference", "candidate")):
            calls = len(pairs[0]["reference"]["call_seconds"])
            if not calls or any(
                len(p[v]["call_seconds"]) != calls
                for p in pairs
                for v in ("reference", "candidate")
            ):
                raise ValueError("incomplete per-call timings")
            if any(
                not math.isfinite(t) or t <= 0
                for p in pairs
                for v in ("reference", "candidate")
                for t in p[v]["call_seconds"]
            ):
                raise ValueError("invalid per-call timings")
            call_stats = []
            for position in range(calls):
                call_ratios = [
                    p["reference"]["call_seconds"][position]
                    / p["candidate"]["call_seconds"][position]
                    for p in pairs
                ]
                call_stats.append(
                    {
                        "position": position,
                        "speedup": math.exp(statistics.mean(map(math.log, call_ratios))),
                        "paired_ratios": call_ratios,
                        "medians_seconds": {
                            v: statistics.median(p[v]["call_seconds"][position] for p in pairs)
                            for v in ("reference", "candidate")
                        },
                    }
                )
        cases.append(
            {
                "id": row["id"],
                "speedup": math.exp(statistics.mean(map(math.log, ratios))),
                "paired_ratios": ratios,
                "medians": medians,
                "calls": call_stats,
            }
        )
    for i in range(count):
        rounds.append(math.exp(statistics.mean(math.log(c["paired_ratios"][i]) for c in cases)))
    wins = sum(r > minimum_speedup for r in rounds)
    # Fixed final sample size; no sequential peeking or reuse for another author.
    sign_p = sum(math.comb(count, k) for k in range(wins, count + 1)) / 2**count
    return {
        "timing_scope": timing_scope,
        "timing_metric": metric,
        "cases": cases,
        "case_balanced_geomean_speedup": math.exp(statistics.mean(map(math.log, rounds))),
        "round_speedups": rounds,
        "rounds_above_minimum": wins,
        "sign_test_p": sign_p,
        "statistical_scope": "One-sided fixed-sample sign test of case-balanced round speedup exceeding the frozen minimum. Interpretation assumes independent round signs; correlated thermal/system drift can violate that assumption. AB/BA ordering is balanced within each case. Equal case weights describe this suite, not a production traffic mix.",
    }


def compare(
    reference,
    candidate,
    reference_snapshot,
    candidate_snapshot,
    value,
    cases,
    folder,
    *,
    pairs,
    deadline,
    progress,
    minimum_speedup,
    timing_scope="calls",
):
    folder.mkdir(parents=True)
    rows = [{"id": c["id"], "pairs": []} for c in cases]
    orders = []
    for i in range(len(cases)):
        order = [j % 2 for j in range(pairs)]
        random.Random(value["replay"]["seed"] + i).shuffle(order)
        orders.append(order)
    fingerprints = {}
    report = {"status": "incomplete", "cases": rows}
    try:
        for repeat in range(pairs):
            for i, case in enumerate(cases):
                progress(f"COMPARE {folder.name} round={repeat + 1}/{pairs} case={case['id']}")
                pair = {
                    "order": ["reference", "candidate"]
                    if not orders[i][repeat]
                    else ["candidate", "reference"]
                }
                for variant in pair["order"]:
                    root, snapshot = (
                        (reference, reference_snapshot)
                        if variant == "reference"
                        else (candidate, candidate_snapshot)
                    )
                    attempt = folder / f"r{repeat + 1:02d}_{case['id']}_{variant}"
                    pair[variant] = run_case(
                        root, snapshot, value, case, attempt, deadline, progress
                    )
                rows[i]["pairs"].append(pair)
                expected = fingerprints.setdefault(case["id"], pair["reference"]["fingerprint"])
                if pair["reference"]["fingerprint"] != expected:
                    report.update(status="reference_unstable", failed_case=case["id"])
                    return report
                if pair["candidate"]["fingerprint"] != expected:
                    report.update(
                        status="mismatch",
                        failed_case=case["id"],
                        failed_calls=[
                            n
                            for n, (a, b) in enumerate(
                                zip(
                                    pair["reference"]["call_fingerprints"],
                                    pair["candidate"]["call_fingerprints"],
                                )
                            )
                            if a != b
                        ],
                        retained_outputs_match=pair["reference"]["retained_fingerprint"]
                        == pair["candidate"]["retained_fingerprint"],
                    )
                    return report
        report.update(
            status="matched", statistics=statistics_for(rows, minimum_speedup, timing_scope)
        )
    except (ValueError, OSError, TimeoutError) as error:
        report.update(status="failed", reason=str(error))
    finally:
        write(folder / "comparison.json", report)
    return report


def eligible(report, policy, *, final):
    if report["status"] != "matched":
        return False
    stats = report["statistics"]
    if stats["case_balanced_geomean_speedup"] < policy["minimum_speedup"]:
        return False
    if any(c["speedup"] < 1 / (1 + policy["max_case_regression"]) for c in stats["cases"]):
        return False
    if "max_call_regression" in policy:
        for case in stats["cases"]:
            if not case.get("calls") or any(
                call["speedup"] < 1 / (1 + policy["max_call_regression"]) for call in case["calls"]
            ):
                return False
    return not final or stats["sign_test_p"] <= policy["significance"]
