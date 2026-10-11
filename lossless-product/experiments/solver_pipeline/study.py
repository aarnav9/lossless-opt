"""Optional measured-cost planning pilot. No product defaults are changed."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import time

import numpy as np

from pipeline import (
    CONTROLS,
    OPERATIONS,
    REFERENCE,
    STAGES,
    Pipeline,
    cases,
    evaluate,
    measure_catalog,
    sha,
    verdict,
    write,
)
from planner import (
    dynamic_program,
    lean_witness,
    response_schema,
    solve,
    validate_response,
)

HERE = Path(__file__).resolve().parent


def stamp(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def request_for(catalog, inputs, history, suggestion, remaining):
    request = {
        "task": "Choose one complete execution plan for the fixed uint8 image pipeline.",
        "transport_instructions": "Return only hypothesis and segments JSON. No tools, execution or browsing. Choose supplied segment IDs; costs, correctness and timing gates are evaluator-owned.",
        "operations": OPERATIONS,
        "contract": "Apply all six stages exactly once in order. uint8 arithmetic wraps. Outputs must be fresh contiguous arrays; inputs and retained outputs must remain unchanged.",
        "workload": [dict(shape=list(x.shape), strides=list(x.strides)) for x in inputs],
        "objective": "Minimize measured latency for one sequence containing each declared input once; no individual call may regress more than 3% versus the reference. Require 2% sequence improvement and a bounded traced-memory increase.",
        "cost_model": "Sum measured median nanoseconds of selected region calls across the declared sequence. Includes region guard/packing/allocation/dispatch. Full sequence wrapper and interactions can cause prediction error. Do not add launch costs again. CPU only, serial execution, no inferred GPU costs.",
        "catalog": catalog,
        "previous_attempts": history,
        "remaining_seconds": remaining,
        "instructions": "Propose a new plan using measured evidence and previous prediction errors. The fixed catalog includes full native and lookup-table fusion; the solver cannot invent new kernels in this pilot.",
    }
    if suggestion is not None:
        request["solver_suggestion"] = suggestion
    return request


def fixed_response(index, suggestion):
    plans = [["native_0_3", "native_3_6"], ["native_0_6"], ["lut_0_6"]]
    return {
        "hypothesis": "Fixed mechanical smoke fixture; not an LLM proposal or an efficacy comparison.",
        "segments": suggestion["plan"]
        if suggestion and "plan" in suggestion
        else plans[index % len(plans)],
    }


def run_arm(args, arm, out, pipeline, catalog, inputs, shared_seconds):
    from lossless import codex_provider

    folder = out / arm
    folder.mkdir()
    started = time.perf_counter()
    deadline = started + max(0, args.budget_seconds - args.final_reserve_seconds - shared_seconds)
    history, seen, best = [], [], None
    for index in range(args.max_candidates):
        remaining = deadline - time.perf_counter()
        if remaining <= 0.1:
            break
        stamp(
            f"SEARCH arm={arm} candidate={index + 1}/{args.max_candidates} remaining={remaining:.1f}s"
        )
        candidate_dir = folder / f"candidate_{index + 1:02d}"
        candidate_dir.mkdir()
        suggestion = None
        if arm == "solver":
            suggestion = solve(
                catalog, STAGES, exclude=seen, timeout=min(args.solver_seconds, remaining)
            )
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        request = request_for(catalog, inputs, history, suggestion, remaining)
        write(candidate_dir / "request.json", request)
        if args.author == "codex":
            receipt = codex_provider.invoke(
                request,
                candidate_dir / "author",
                model=args.model,
                effort=args.effort,
                timeout=min(args.provider_timeout_seconds, remaining),
                schema_override=response_schema(catalog),
                validator=lambda response: validate_response(response, catalog, STAGES),
            )
            if not receipt["eligible"]:
                history.append(dict(status="provider_failed", receipt=receipt))
                stamp(
                    f"AUTHOR_FAILED arm={arm} candidate={index + 1}; inspect {candidate_dir / 'author'}"
                )
                continue
            response = json.loads((candidate_dir / "author/response.json").read_text())
            response.pop("usage", None)
        else:
            response = fixed_response(index, suggestion)
            receipt = {
                "provider": "fixed-smoke-fixture",
                "elapsed_seconds": 0.0,
                "usage": [],
                "cost_usd": None,
            }
        predicted = validate_response(response, catalog, STAGES)
        plan = response["segments"]
        record = dict(proposal=response, predicted_sequence_ns=predicted, provider_receipt=receipt)
        write(candidate_dir / "proposal.json", record)
        if plan in seen:
            record["status"] = "duplicate"
        elif time.perf_counter() >= deadline:
            record["status"] = "search_budget_exhausted"
        else:
            seen.append(plan)
            evidence = evaluate(
                pipeline,
                {"reference": REFERENCE, "candidate": plan},
                inputs,
                repeats=args.discovery_repeats,
                target_seconds=args.sample_seconds,
                seed=args.seed + index,
            )
            write(candidate_dir / "measurement.json", evidence)
            actual = evidence["sequence"]["median_ns"]["candidate"]
            record.update(
                status="evaluated",
                measured_sequence_ns=actual,
                prediction_error_fraction=(actual - predicted) / predicted,
                decision=verdict(evidence, "candidate"),
            )
            if time.perf_counter() > deadline:
                record["status"] = "search_budget_exceeded"
            elif record["decision"]["eligible"] and (best is None or actual < best[0]):
                best = actual, list(plan)
        history.append(record)
        write(candidate_dir / "result.json", record)
    frozen = best[1] if best else list(REFERENCE)
    record = dict(
        arm=arm,
        author=args.author,
        frozen_plan=frozen,
        history=history,
        search_seconds=time.perf_counter() - started,
        budget_seconds=args.budget_seconds,
        max_candidates=args.max_candidates,
        shared_seconds_charged=shared_seconds,
        valid_author_responses=sum("proposal" in row for row in history),
    )
    write(folder / "frozen.json", record)
    stamp(f"FROZEN arm={arm} plan={frozen}; final data remain closed")
    return record


def check_lean(out, catalog, plan):
    from lossless.proofs import TOOLCHAIN, lean_binary

    folder = out / "proof"
    folder.mkdir()
    shutil.copyfile(HERE / "PlanCost.lean", folder / "PlanCost.lean")
    (folder / "CostWitness.lean").write_text(lean_witness(catalog, STAGES, plan))
    lean = lean_binary()  # Pinned existing toolchain only; no implicit download.
    env = {**os.environ, "LEAN_PATH": str(folder)}
    started = time.perf_counter()
    outputs = []
    for name in ("PlanCost", "CostWitness"):
        result = subprocess.run(
            [lean, "-o", name + ".olean", name + ".lean"],
            cwd=folder,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        outputs.append(result.stdout + result.stderr)
        if result.returncode or "sorryAx" in outputs[-1]:
            raise ValueError(f"Lean certificate failed: {outputs[-1]}")
    (folder / "checker.log").write_text("\n".join(outputs))
    return dict(
        status="checked",
        toolchain=TOOLCHAIN,
        checker_sha256=sha(lean),
        seconds=time.perf_counter() - started,
        source_sha256={p.name: sha(p) for p in folder.glob("*.lean")},
        scope="Global minimum among contiguous-region plans in this frozen serial integer-cost model. No proof of measured time, generated C semantics, GPU behavior or performance outside this catalog.",
    )


def execute(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    log = args.log.resolve() if args.log else out.with_suffix(".log")
    started = time.perf_counter()
    stamp(
        f"START author={args.author} expected={'under 2 minutes' if args.author == 'fixed' else 'up to two arm budgets plus setup'} result={out / 'report.json'} log={log}"
    )
    if args.author == "codex":
        from lossless.codex_provider import require_chatgpt

        require_chatgpt()
    protocol = {
        "schema_version": 1,
        "author": args.author,
        "scope": "Synthetic CPU uint8 preprocessing-style pipeline; serial fixed-kernel-catalog plan selection. No CUDA, external application or general solver-efficacy claim.",
        "timing_scope": "Ordinary calls including input guards, strided packing, allocations, NumPy/native calls, ctypes dispatch, result tuple construction and completion. Timed outputs are discarded. Compilation/calibration/tuning excluded from steady-state samples and charged separately; interpreter/module-import startup is not charged. Warm repeated inputs; no cold-cache claim.",
        "costs": "Evaluator-owned rounded nanosecond region medians; sum across the declared sequence (one request per discovery shape). No separately added dispatch penalty. Estimates, not physical lower bounds.",
        "contract": "uint8 exact finite validation with all 256 values for each compiled region, plus held-out shapes/values, input mutation and retained output checks. Lean cost proof is separate from implementation correctness.",
        "gates": {
            "minimum_sequence_speedup": 1.02,
            "max_per_case_regression": 0.03,
            "max_traced_peak_ratio": 2.0,
            "traced_peak_slack_bytes": 32768,
            "uncertainty": "descriptive pilot medians only; not a statistical speedup certification",
        },
        "memory_scope": "Python/NumPy traced allocation peak in a separate diagnostic pass; not RSS or GPU memory.",
        "comparison": "Both arms get identical measured catalogs, budgets, proposal caps and timing gates. Assisted authors additionally see CP-SAT suggestions excluding previous plans. Each arm freezes before either receives final evidence. Solver and provider time count toward its search cap. Fixed mode is a mechanical smoke fixture, never an LLM ablation.",
        "limitations": "No kernel generation, concurrency scheduling, online cost fitting or serving p95. Prediction errors return to the author; immutable measured segment costs remain unchanged. A single live pair cannot establish efficacy.",
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "environment": dict(
            python=platform.python_version(),
            platform=platform.platform(),
            machine=platform.machine(),
            numpy=np.__version__,
            ortools=importlib.metadata.version("ortools"),
        ),
        "sources": {p.name: sha(p) for p in HERE.iterdir() if p.suffix in {".py", ".lean", ".txt"}},
    }
    write(out / "protocol.json", protocol)
    (out / "sources").mkdir()
    for name in protocol["sources"]:
        shutil.copyfile(HERE / name, out / "sources" / name)
    pipeline = Pipeline(out / "build")
    checks = pipeline.validate_regions()
    inputs = cases("discovery", args.seed)
    catalog, raw = measure_catalog(
        pipeline,
        inputs,
        repeats=args.discovery_repeats,
        target_seconds=args.sample_seconds,
        seed=args.seed,
        progress=stamp,
    )
    write(out / "catalog-measurements.json", raw)
    write(out / "catalog.json", catalog)
    model_solution = solve(catalog, STAGES, timeout=args.solver_seconds)
    dp = dynamic_program(catalog, STAGES)
    if "plan" not in model_solution or model_solution["cost_ns"] != dp["cost_ns"]:
        raise ValueError("CP-SAT did not match the independent dynamic-programming optimum")
    write(out / "model-control.json", dict(cp_sat=model_solution, dynamic_program=dp))
    proof = (
        check_lean(out, catalog, model_solution["plan"])
        if args.check_lean
        else {"status": "not_requested"}
    )
    shared = time.perf_counter() - started
    order = ["control", "solver"]
    random.Random(args.seed).shuffle(order)
    write(out / "arm-order.json", order)
    arms = {arm: run_arm(args, arm, out, pipeline, catalog, inputs, shared) for arm in order}
    # Final fixtures/timings are opened only after BOTH arms have frozen.
    final_started = time.perf_counter()
    final_inputs = cases("evaluation", args.seed)
    plans = {
        **CONTROLS,
        "model_optimum": model_solution["plan"],
        **{k: v["frozen_plan"] for k, v in arms.items()},
    }
    stamp("FINAL both plans frozen; opening held-out shapes and values")
    final = evaluate(
        pipeline,
        plans,
        final_inputs,
        repeats=args.evaluation_repeats,
        target_seconds=args.sample_seconds,
        seed=args.seed + 20000,
    )
    write(out / "evaluation.json", final)
    final_seconds = time.perf_counter() - final_started
    for arm, record in arms.items():
        decision = verdict(final, arm)
        total = shared + record["search_seconds"] + final_seconds
        record.update(
            final=decision,
            standalone_tuning_seconds=total,
            within_total_budget=total <= args.budget_seconds,
        )
        record["accepted"] = decision["eligible"] and record["within_total_budget"]
        record["retained_plan"] = record["frozen_plan"] if record["accepted"] else list(REFERENCE)
        saved_ns = final["sequence"]["median_ns"]["reference"] - final["sequence"]["median_ns"][arm]
        record["break_even_sequences"] = (
            math.ceil(total * 1e9 / saved_ns) if record["accepted"] and saved_ns > 0 else None
        )
    final_speed = {name: verdict(final, name) for name in plans if name != "reference"}
    effective = {name: name if record["accepted"] else "reference" for name, record in arms.items()}
    paired = final["sequence"]["samples_ns"]
    report = dict(
        status="fixed_smoke"
        if args.author == "fixed"
        else (
            "completed_live_pair"
            if all(r["valid_author_responses"] for r in arms.values())
            else "incomplete_live_pair"
        ),
        protocol=protocol,
        catalog_sha256=sha(out / "catalog.json"),
        correctness_region_checks=checks,
        compile_seconds=pipeline.compile_seconds,
        compiler=dict(
            command=pipeline.command,
            version=pipeline.compiler_version,
            source_sha256=sha(pipeline.source_path),
        ),
        shared_calibration_audit_seconds=shared,
        final_seconds=final_seconds,
        experiment_wall_seconds=time.perf_counter() - started,
        lean=proof,
        model_solution=model_solution,
        controls=final_speed,
        final_sequence_median_ns=final["sequence"]["median_ns"],
        arms=arms,
        solver_vs_control_speedup=statistics.median(
            a / b for a, b in zip(paired[effective["control"]], paired[effective["solver"]])
        ),
        identical_retained_plans=arms["control"]["retained_plan"]
        == arms["solver"]["retained_plan"],
        conclusion="Mechanical smoke only; no LLM efficacy inference."
        if args.author == "fixed"
        else "One matched live-author pair; inspect controls, per-case gates, model errors and tuning cost before drawing conclusions.",
    )
    write(out / "report.json", report)
    stamp(
        f"COMPLETE result={out / 'report.json'} log={log}; control={arms['control']['final']['sequence_speedup']:.3f}x solver={arms['solver']['final']['sequence_speedup']:.3f}x"
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--log", type=Path)
    p.add_argument("--author", choices=["fixed", "codex"], default="fixed")
    p.add_argument("--model")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default="medium")
    p.add_argument("--budget-seconds", type=float, default=90)
    p.add_argument("--final-reserve-seconds", type=float, default=10)
    p.add_argument("--provider-timeout-seconds", type=float, default=60)
    p.add_argument("--solver-seconds", type=float, default=1)
    p.add_argument("--max-candidates", type=int, default=3)
    p.add_argument("--discovery-repeats", type=int, default=5)
    p.add_argument("--evaluation-repeats", type=int, default=9)
    p.add_argument("--sample-seconds", type=float, default=0.003)
    p.add_argument("--seed", type=int, default=1701)
    p.add_argument("--check-lean", action="store_true")
    args = p.parse_args()
    for key in (
        "budget_seconds",
        "final_reserve_seconds",
        "provider_timeout_seconds",
        "solver_seconds",
        "sample_seconds",
    ):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            p.error(f"{key} must be finite and positive")
    if args.final_reserve_seconds >= args.budget_seconds:
        p.error("final reserve must be smaller than total budget")
    if min(args.max_candidates, args.discovery_repeats, args.evaluation_repeats) < 1:
        p.error("candidate and repeat counts must be positive")
    if args.author == "codex" and not args.model:
        p.error("--model is required for live authors")
    execute(args)


if __name__ == "__main__":
    main()
