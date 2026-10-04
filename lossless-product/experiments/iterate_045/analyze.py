"""Held-out paired checks of frozen choices, with actual time and payback."""

import os
import subprocess
import sys

from lossless._native import engine
from lossless._native.common import read, write, stamp, geomean
from lossless.economics import payback
import study


def binary(out, arm, name):
    if name in {None, "deployment_reference", "native_baseline"}:
        return None
    run = out / arm
    state = engine.verify(run)
    if name not in state["proposals"] or state["proposals"][name]["rejected"]:
        return None
    return next(
        run / job["binary"]
        for job in state["jobs"].values()
        if job["kind"] == "compile" and job["candidate"] == name and job["status"] == "ok"
    )


def pair(out, label, candidate, reference, search_seconds):
    candidate_binary = binary(out, *candidate)
    reference_binary = binary(out, *reference)
    result = {"label": label, "candidate": candidate, "reference": reference}
    if not candidate_binary or not reference_binary:
        return {**result, "status": "unavailable: reference selected or candidate rejected"}
    folder = out / "paired"
    folder.mkdir(exist_ok=True)
    records = []
    cases = [c for c in read(out / "spec.json")["cases"] if c["split"] == "evaluation"]
    for index, case in enumerate(cases):
        stem = folder / f"{label}_{index:02d}"
        job = {
            "kind": "benchmark",
            "operator": "softmax",
            "case": case,
            "baseline": str(reference_binary),
            "candidate": str(candidate_binary),
            "comparator": "native_baseline",
            "seed": 450000 + index,
            "candidate_seed": 45,
            "screening_blocks": 5,
            "confirmation_blocks": 15,
            "target_ns": 500000,
            "timeout_seconds": 8,
        }
        job_path, result_path = stem.with_suffix(".job.json"), stem.with_suffix(".result.json")
        if result_path.exists():
            assert read(job_path) == job, "cannot reuse another comparison"
        else:
            write(job_path, job)
            env = {
                k: v
                for k, v in os.environ.items()
                if k in {"PATH", "TMPDIR", "LANG", "SYSTEMROOT", "PYTHONPATH"}
            }
            env.update(
                {
                    k: "1"
                    for k in [
                        "VECLIB_MAXIMUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    ]
                }
            )
            with stem.with_suffix(".log").open("w") as log:
                try:
                    worker = subprocess.run(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "lossless._native.worker",
                            str(job_path),
                            str(result_path),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=10,
                        env=env,
                    )
                    if worker.returncode or not result_path.exists():
                        write(
                            result_path,
                            {
                                "status": "worker_failed",
                                "case": case,
                                "returncode": worker.returncode,
                            },
                        )
                except subprocess.TimeoutExpired:
                    write(result_path, {"status": "worker_timeout", "case": case})
        records.append(read(result_path))
        stamp(f"PAIR {label} case={index + 1}/{len(cases)}")
    correct = all(r["status"] == "ok" for r in records)
    gain = geomean([r["speedup_vs_native"] for r in records]) if correct else None
    return {
        **result,
        "status": "measured",
        "checks_passed": correct,
        "geomean": gain,
        "incremental_passed": correct
        and gain >= 1.02
        and all(r["ci95_vs_native"][0] > 1 for r in records),
        "all_case_lower_intervals_above_one": correct
        and all(r["ci95_vs_native"][0] > 1 for r in records),
        "records": records,
        "payback": [
            {
                "case": r["case"],
                **payback(
                    search_seconds,
                    r["confirmation"]["median_us"]["native_baseline"] / 1e6,
                    r["confirmation"]["median_us"]["proposal"] / 1e6,
                ),
            }
            for r in records
            if r["status"] == "ok"
        ],
    }


def summarize(out):
    plan = study.verify(out)
    study.require_sealed(out)
    discovery = read(out / "discovery_complete.json")
    loop = discovery["loop"]
    reports = {arm: read(out / arm / "summary.json") for arm in ["iterative", "enumerated"]}
    winner = reports["iterative"]["frozen_global_choice"]
    control = reports["enumerated"]["frozen_global_choice"]
    total = loop["search_seconds"] + read(out / "iterative/evaluation_wall.json")["seconds"]
    first_cost = loop["rounds"][0]["elapsed_seconds"] if loop["rounds"] else 0
    pairs = []
    for label, reference in [
        ("final_over_first", ("iterative", loop.get("first_round_choice"))),
        ("final_over_prior", ("iterative", "prior_044")),
        ("final_over_retained", ("iterative", loop["retained_choice"])),
        ("final_over_enumerated", ("enumerated", control)),
    ]:
        pairs.append(
            pair(
                out,
                label,
                ("iterative", winner),
                reference,
                max(0, total - first_cost) if label == "final_over_first" else total,
            )
        )
    receipts = [read(p) for p in sorted((out / "rounds").glob("*/receipt.json"))]
    usage = [u for r in receipts for u in r["usage"] if u]
    accepted = {}
    for arm, report in reports.items():
        selected = report["frozen_global_choice"]
        state = engine.verify(out / arm)
        rows = [
            read(out / arm / j["result"])
            for j in state["jobs"].values()
            if j["kind"] == "benchmark"
            and j["stage"] == "evaluation"
            and j["candidate"] == selected
            and j["status"] == "ok"
        ]
        check = report["frozen_choice_evaluation"]
        accepted[arm] = (
            bool(rows)
            and check["complete"]
            and check["geomean_vs_comparator"] >= 1.02
            and all(r["ci95_vs_comparator"][0] > 1 for r in rows)
        )
    result = {
        "plan": plan,
        "discovery": discovery,
        "reports": reports,
        "accepted_vs_numpy": accepted,
        "pairs": pairs,
        "receipts": receipts,
        "total_loop_plus_final_seconds": total,
        "usage": {
            key: sum(u.get(key, 0) for u in usage)
            for key in ["input_tokens", "output_tokens", "reasoning_output_tokens"]
        },
        "usage_scope": "Completed-turn receipts only; missing timeout usage unknown, not zero.",
        "missing_usage_receipts": sum(not bool(r["usage"]) for r in receipts),
        "cost_usd": None,
        "extra_deployment_setup_measured": False,
    }
    write(out / "summary.json", result)
    lines = [
        "# Campaign 045: one-hour iterative pilot",
        "",
        plan["rules"],
        "",
        f"Search: {loop['search_seconds'] / 60:.2f} min; rounds: {len(loop['rounds'])}; attempted new slots: {loop['attempted_slots']}.",
        f"Stop reason: {loop['stop_reason']}. Frozen final choice: `{winner}`.",
        "",
        "| Comparison | Geomean | Incremental pass |",
        "| --- | --- | --- |",
    ]
    for p in pairs:
        value = f"{p['geomean']:.5f}×" if p.get("geomean") else p["status"]
        lines.append(f"| {p['label']} | {value} | {p.get('incremental_passed', False)} |")
    lines += [
        "",
        "One trajectory; timing-sample intervals do not establish across-author superiority. "
        "Numerical finite validation, not bitwise proof or full-model acceleration. "
        "Search-only payback omits unmeasured deployment setup. No default promotion.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    stamp(f"COMPLETE result={out / 'summary.json'} report={out / 'REPORT.md'}")
