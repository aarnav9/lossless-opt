"""Compare sealed winners in fresh paired workers; retain authoring failures and costs."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from lossless._native import engine
from lossless._native.common import read, write, geomean, stamp
from lossless.economics import payback


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    plan = read(out / "plan.json")
    stamp(
        f"START estimated=15-30s result={out / 'summary.json'} log={out.parent / (out.name + '_compare.log')}"
    )
    records = []
    for index, arm in plan["order"]:
        run = out / f"{arm}_{index}"
        state = engine.verify(run)
        author = read(out / f"author_{index}/receipt.json") if arm == "llm" else None
        timing = sum(
            read(run / f"{phase}_wall.json")["seconds"] for phase in ["discovery", "evaluation"]
        )
        if (run / "failed_search.json").exists():
            result = dict(
                arm=arm,
                replicate=index,
                accepted=False,
                status="no_validated_proposals",
                selected="reference",
                evaluation=None,
            )
        else:
            summary = read(run / "summary.json")
            selected = summary["frozen_global_choice"]
            chosen = summary["frozen_choice_evaluation"]
            rows = [
                read(run / j["result"])
                for j in state["jobs"].values()
                if j["kind"] == "benchmark"
                and j["stage"] == "evaluation"
                and j["candidate"] == selected
                and j["status"] == "ok"
            ]
            accepted = (
                bool(rows)
                and chosen["complete"]
                and chosen["geomean_vs_comparator"] >= 1.02
                and all(r["ci95_vs_comparator"][0] > 1 for r in rows)
            )
            result = dict(
                arm=arm,
                replicate=index,
                accepted=accepted,
                status="evaluated",
                selected=selected,
                evaluation=chosen,
                summary=summary,
            )
        total = timing + (author["elapsed_seconds"] if author else 0)
        result.update(
            controller_seconds=timing,
            author=author,
            total_observed_seconds=total,
            accepted_improvements_per_minute=60 / total if result["accepted"] else 0,
            cost_usd=None if arm == "llm" else 0,
            invalid_or_missing=read(run / "submission_failures.json"),
        )
        records.append(result)
    paired = []
    folder = out / "paired_winners"
    folder.mkdir(exist_ok=True)
    spec = read(out / "spec.json")
    for index in range(3):
        retained = out / f"retained_{index}"
        reference = read(retained / "summary.json")["frozen_global_choice"]
        for arm in ["enumerated", "llm"]:
            run = out / f"{arm}_{index}"
            if (run / "failed_search.json").exists():
                continue
            candidate = read(run / "summary.json")["frozen_global_choice"]
            if any(
                n in {"deployment_reference", "native_baseline"} for n in [reference, candidate]
            ):
                paired.append(
                    dict(replicate=index, arm=arm, status="reference retained; no native pair")
                )
                continue
            state_ref, state_opt = engine.verify(retained), engine.verify(run)

            def binary(state, name):
                return next(
                    j["binary"]
                    for j in state["jobs"].values()
                    if j["kind"] == "compile" and j["candidate"] == name
                )

            comparisons = []
            for case_index, case in enumerate(
                c for c in spec["cases"] if c["split"] == "evaluation"
            ):
                prefix = folder / f"{arm}_{index}_{case_index}"
                job = dict(
                    kind="benchmark",
                    operator="softmax",
                    case=case,
                    baseline=str(retained / binary(state_ref, reference)),
                    candidate=str(run / binary(state_opt, candidate)),
                    comparator="native_baseline",
                    seed=430000 + case_index,
                    candidate_seed=index,
                    screening_blocks=5,
                    confirmation_blocks=15,
                    target_ns=500000,
                    timeout_seconds=8,
                )
                if prefix.with_suffix(".result.json").exists():
                    assert read(prefix.with_suffix(".job.json")) == job
                    comparisons.append(read(prefix.with_suffix(".result.json")))
                    continue
                write(prefix.with_suffix(".job.json"), job)
                started = time.monotonic()
                with prefix.with_suffix(".log").open("w") as log:
                    env = {
                        key: value
                        for key, value in os.environ.items()
                        if key in ("PATH", "TMPDIR", "SYSTEMROOT", "LANG", "PYTHONPATH")
                    }
                    env.update(
                        VECLIB_MAXIMUM_THREADS="1",
                        OPENBLAS_NUM_THREADS="1",
                        OMP_NUM_THREADS="1",
                        MKL_NUM_THREADS="1",
                        NUMEXPR_NUM_THREADS="1",
                    )
                    process = subprocess.run(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "lossless._native.worker",
                            str(prefix.with_suffix(".job.json")),
                            str(prefix.with_suffix(".result.json")),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=env,
                        timeout=10,
                    )
                if process.returncode:
                    comparisons.append(
                        dict(case=case, status="worker_failed", returncode=process.returncode)
                    )
                else:
                    comparisons.append(read(prefix.with_suffix(".result.json")))
                stamp(
                    f"PAIRED winner arm={arm} run={index + 1}/3 case={case_index + 1}/9 seconds={time.monotonic() - started:.2f}"
                )
            correct = all(r["status"] == "ok" for r in comparisons)
            gain = geomean([r["speedup_vs_native"] for r in comparisons]) if correct else None
            cost = next(
                r["total_observed_seconds"]
                for r in records
                if r["arm"] == arm and r["replicate"] == index
            )
            amortization = [
                {
                    "case": r["case"],
                    **payback(
                        cost,
                        r["confirmation"]["median_us"]["native_baseline"] / 1e6,
                        r["confirmation"]["median_us"]["proposal"] / 1e6,
                    ),
                }
                for r in comparisons
                if r["status"] == "ok"
            ]
            paired.append(
                dict(
                    replicate=index,
                    arm=arm,
                    status="measured",
                    checks_passed=correct,
                    geomean_vs_retained=gain,
                    incremental_passed=correct
                    and gain >= 1.02
                    and all(r["ci95_vs_native"][0] > 1 for r in comparisons),
                    records=comparisons,
                    payback_vs_retained=amortization,
                )
            )
    write(
        out / "summary.json",
        dict(
            plan=plan,
            arms=records,
            paired_winners=paired,
            scope=f"Three independent one-shot CLI sessions under a fixed {plan['author_timeout_seconds']}-second authoring cap; failures count. Candidate counts/timing blocks matched, total wall time reported separately. Successful control generation is deterministic enumeration; cost_usd is provider billing only, unknown for CLI access. Study engineering/curation and the shared freeze step are excluded from arm timings. Kernel quality is unmeasured when no proposal is returned. All choices sealed before final cases; extra paired measurements compare frozen winners without reselection.",
        ),
    )
    stamp(f"COMPLETE result={out / 'summary.json'} log={out.parent / (out.name + '_compare.log')}")


if __name__ == "__main__":
    main()
