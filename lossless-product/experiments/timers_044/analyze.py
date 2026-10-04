"""Summarize all timer outcomes and replay frozen winners directly against one another."""

import argparse
import os
from pathlib import Path
import statistics
import subprocess
import sys

from lossless._native import engine
from lossless._native.common import geomean, read, stamp, write
import study


def selected_binary(run):
    if (run / "failed_search.json").exists():
        return None
    chosen = read(run / "summary.json")["frozen_global_choice"]
    if chosen in {"deployment_reference", "native_baseline"}:
        return None
    state = engine.verify(run)
    return next(
        run / j["binary"]
        for j in state["jobs"].values()
        if j["kind"] == "compile" and j["candidate"] == chosen
    )


def compare(out, reference_condition, candidate_condition, index):
    reference = selected_binary(out / reference_condition / f"llm_{index}")
    candidate = selected_binary(out / candidate_condition / f"llm_{index}")
    result = dict(reference=reference_condition, candidate=candidate_condition, replicate=index)
    if reference is None or candidate is None:
        return {**result, "status": "unavailable: no native winner on both sides"}
    folder = out / "between_conditions"
    folder.mkdir(exist_ok=True)
    cases = [
        c
        for c in read(out / reference_condition / "spec.json")["cases"]
        if c["split"] == "evaluation"
    ]
    records = []
    for case_index, case in enumerate(cases):
        prefix = folder / f"{candidate_condition}_over_{reference_condition}_{index}_{case_index}"
        job = dict(
            kind="benchmark",
            operator="softmax",
            case=case,
            baseline=str(reference),
            candidate=str(candidate),
            comparator="native_baseline",
            seed=440000 + case_index,
            candidate_seed=index,
            screening_blocks=5,
            confirmation_blocks=15,
            target_ns=500000,
            timeout_seconds=8,
        )
        job_path = prefix.with_suffix(".job.json")
        result_path = prefix.with_suffix(".result.json")
        if result_path.exists():
            assert read(job_path) == job, "cannot reuse evidence from another pair"
        else:
            write(job_path, job)
            env = {
                k: v
                for k, v in os.environ.items()
                if k in {"PATH", "TMPDIR", "SYSTEMROOT", "LANG", "PYTHONPATH"}
            }
            env.update(
                {
                    name: "1"
                    for name in [
                        "VECLIB_MAXIMUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    ]
                }
            )
            with prefix.with_suffix(".log").open("w") as log:
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
                        env=env,
                        timeout=10,
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
        stamp(
            f"PAIR {candidate_condition}/{reference_condition} replicate={index + 1}/3 "
            f"case={case_index + 1}/{len(cases)}"
        )
    correct = all(r["status"] == "ok" for r in records)
    gain = geomean([r["speedup_vs_native"] for r in records]) if correct else None
    return {
        **result,
        "status": "measured",
        "checks_passed": correct,
        "geomean": gain,
        "every_case_lower_interval_above_one": correct
        and all(r["ci95_vs_native"][0] > 1 for r in records),
        "records": records,
    }


def summarize(out):
    plan = study.verify(out)
    study.require_all_sealed(out, plan)
    conditions = []
    for name, config in plan["conditions"].items():
        summary = read(out / name / "summary.json")
        authors = [read(out / name / f"author_{i}" / "receipt.json") for i in range(3)]
        arms = [a for a in summary["arms"] if a["arm"] == "llm"]
        paired = [a for a in summary["paired_winners"] if a["arm"] == "llm"]
        gains = [
            p["geomean_vs_retained"]
            for p in paired
            if p["status"] == "measured" and p["checks_passed"]
        ]
        proposals = []
        for index in range(3):
            run = out / name / f"llm_{index}"
            report = read(run / "summary.json") if (run / "summary.json").exists() else None
            state = engine.verify(run)
            proposals.append(
                {
                    "replicate": index,
                    "submitted": len(state["proposals"]),
                    "rejected": {
                        n: p["rejected"] for n, p in state["proposals"].items() if p["rejected"]
                    },
                    "report": report,
                }
            )
        conditions.append(
            {
                "condition": name,
                **config,
                "authors": authors,
                "proposals": proposals,
                "completed_authors": sum(a["eligible"] for a in authors),
                "timeouts": sum(a["timed_out"] for a in authors),
                "author_seconds": [a["elapsed_seconds"] for a in authors],
                "author_seconds_total": sum(a["elapsed_seconds"] for a in authors),
                "search_seconds_total": sum(a["total_observed_seconds"] for a in arms),
                "accepted_vs_numpy": sum(a["accepted"] for a in arms),
                "incremental_passed_vs_retained": sum(
                    p.get("incremental_passed", False) for p in paired
                ),
                "paired_gains_vs_retained": gains,
                "median_gain_completed": statistics.median(gains) if gains else None,
                "input_tokens": sum(u.get("input_tokens", 0) for a in authors for u in a["usage"]),
                "output_tokens": sum(
                    u.get("output_tokens", 0) for a in authors for u in a["usage"]
                ),
                "token_scope": "Only completed-turn receipts; timeout token usage may be missing.",
                "cost_usd": None,
                "payback": [p.get("payback_vs_retained", []) for p in paired],
            }
        )
    pairs = []
    for reference, candidate in [
        ("announced_30m", "announced_60m"),
        ("unannounced_60m", "announced_60m"),
        ("announced_5m", "announced_30m"),
    ]:
        for index in range(3):
            pairs.append(compare(out, reference, candidate, index))
    result = {
        "plan": plan,
        "conditions": conditions,
        "between_conditions": pairs,
        "scope": plan["rules"],
        "extra_deployment_setup_measured": False,
    }
    write(out / "summary.json", result)
    lines = [
        "# Campaign 044: author deadlines",
        "",
        plan["rules"],
        "",
        "| Condition | Completed | Timed out | Author minutes (each) | Total search min | "
        "Paired speedups over retained | Incremental acceptance |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in conditions:
        durations = ", ".join(f"{s / 60:.2f}" for s in c["author_seconds"])
        gains = ", ".join(f"{g:.4f}×" for g in c["paired_gains_vs_retained"]) or "unmeasured"
        lines.append(
            f"| {c['condition']} | {c['completed_authors']}/3 | {c['timeouts']}/3 | "
            f"{durations} | {c['search_seconds_total'] / 60:.2f} | {gains} | "
            f"{c['incremental_passed_vs_retained']}/3 |"
        )
    lines += [
        "",
        "Speedups are conditional on a validated frozen native winner. Failed authors "
        "are retained in the completion and cost denominators; they are not assigned a "
        "fictitious kernel speed. Each comparison uses all nine final cases.",
        "",
        "## Direct paired comparisons of frozen winners",
        "",
        "| Candidate / reference | Replicate | Geomean speedup | All case lower intervals >1 |",
        "| --- | --- | --- | --- |",
    ]
    for p in pairs:
        gain = f"{p['geomean']:.4f}×" if p.get("geomean") else p["status"]
        lines.append(
            f"| {p['candidate']} / {p['reference']} | {p['replicate'] + 1} | "
            f"{gain} | {p.get('every_case_lower_interval_above_one', 'n/a')} |"
        )
    lines += [
        "",
        "Intervals describe timing samples within a pair, not uncertainty across "
        "the population of possible authors. Three independent replicates cannot establish "
        "a universal time/quality relationship. Billing is unknown; token totals omit "
        "any timeout usage not returned by the CLI. Search-only break-even estimates "
        "exclude deployment setup and study engineering.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    stamp(
        f"COMPLETE result={out / 'summary.json'} report={out / 'REPORT.md'} "
        f"log={out.parent / (out.name + '.log')}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.output.resolve())


if __name__ == "__main__":
    main()
