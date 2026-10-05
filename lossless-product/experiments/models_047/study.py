"""Three subscription-backed models, equal 30-minute iterative search allowances."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

from lossless import codex_provider, recipes
from lossless._native import engine, providers
from lossless._native.common import read, sha, stamp, write

HERE = Path(__file__).resolve().parent
PRODUCT = HERE.parents[1]
LEGACY = HERE.parent / "iterate_045"
sys.path.insert(0, str(LEGACY))
loader = importlib.util.spec_from_file_location("iterate_045_reuse", LEGACY / "study.py")
legacy = importlib.util.module_from_spec(loader)
loader.loader.exec_module(legacy)
sys.path.pop(0)
# Reuse only the standalone paired measurement function; preserve its original source.
previous = sys.modules.get("study")
sys.modules["study"] = legacy
loader = importlib.util.spec_from_file_location("iterate_045_pairs", LEGACY / "analyze.py")
paired = importlib.util.module_from_spec(loader)
loader.loader.exec_module(paired)
if previous is None:
    del sys.modules["study"]
else:
    sys.modules["study"] = previous

MODELS = ["gpt-6.1-sol", "gpt-6-astra", "gpt-6-luna"]


def pair_label(candidate, reference):
    # Legacy paired filenames use Path.with_suffix; periods would collapse labels.
    return f"{candidate}_over_{reference}".replace(".", "_")


def cases():
    return [
        {"id": f"{split}_{i}_{layout}", "split": split, "rows": r, "columns": c, "layout": layout}
        for split, shapes in [
            ("discovery", [(11, 211), (45, 809), (99, 1601)]),
            ("evaluation", [(13, 223), (41, 823), (107, 1613), (23, 1039)]),
        ]
        for i, (r, c) in enumerate(shapes)
        for layout in ["c", "f", "slice2"]
    ]


def freeze(out, seconds=1800, calls=6):
    if not 120 <= seconds <= 3600 or not 1 <= calls <= 12:
        raise ValueError("120..3600 seconds and 1..12 calls required")
    out.mkdir(parents=True, exist_ok=False)
    evidence = PRODUCT / "docs/assets/iterate-045.json.gz"
    bundle = json.loads(gzip.decompress(evidence.read_bytes()))
    incumbent = bundle["sources"]["<study>/iterative/proposals/prior_044/kernel.c"]["source"]
    write(
        out / "seeds.json",
        [
            {
                "id": "prior_044",
                "source": incumbent,
                "hypothesis": "Frozen shared starting kernel from campaign 044; no historical timings supplied.",
            },
            *[{**p, "id": "retained_" + p["id"]} for p in recipes.native_candidates("softmax")],
        ],
    )
    write(out / "enumeration.json", legacy.enumeration(incumbent)[:calls])
    spec = {
        "operator": "softmax",
        "comparator": "numpy_buffered",
        "cases": cases(),
        "budget_seconds": seconds,
        "job_timeout_seconds": 8,
        "compile_timeout_seconds": 8,
        "target_ns": 500000,
        "screening_blocks": 5,
        "confirmation_blocks": 15,
        "max_proposals": 3 + calls,
        "min_speedup": 1.02,
        "cache": {"directory": None},
    }
    write(out / "spec.json", spec)
    order = MODELS.copy()
    random.Random(4701).shuffle(order)
    source_files = sorted({Path(__file__), Path(paired.__file__), *legacy.frozen_sources()})
    write(
        out / "plan.json",
        {
            "campaign": "047",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "models": MODELS,
            "order": order,
            "effort": "xhigh",
            "replicates": 1,
            "search_seconds": seconds,
            "call_seconds": seconds,
            "max_calls": calls,
            "proposals_per_call": 1,
            "reserve_seconds": 60,
            "minimum_call_seconds": 90,
            "hashes": {
                name: sha(out / name) for name in ["spec.json", "seeds.json", "enumeration.json"]
            },
            "source_hashes": {str(p.resolve()): sha(p) for p in source_files},
            "source_evidence_sha256": sha(evidence),
            "rules": "One trajectory per model, identical maximum search wall time, reasoning label xhigh and six one-proposal call slots. Same seeds, numerical contract, compiler and nine discovery cases. Timings explicitly use Gaussian input; six distributions validate correctness. Source/feedback and elapsed/remaining time supplied afresh each round, tools disabled. Invalid/duplicate/timeout responses consume a slot, no retries. Discovery time includes seed setup, model calls and measurements. All arms including the fixed six-slot enumeration control seal before twelve new final cases are opened. Fresh paired winner comparisons and prior/retained comparisons; incremental gate >=1.02 geomean and all lower intervals >1. No population-level ranking, matched token/dollar budget or automatic promotion. Saved ChatGPT CLI sign-in, actual usage receipts and unknown dollars. Inherited strong seed was originally authored by Astra; this tests further optimization from that shared starting point.",
        },
    )


def verify(out):
    plan = read(out / "plan.json")
    for name, digest in plan["hashes"].items():
        assert sha(out / name) == digest, f"frozen input changed: {name}"
    for name, digest in plan["source_hashes"].items():
        assert sha(name) == digest, f"frozen source changed: {name}"
    return plan


def context(run, plan, checkpoint, remaining, allowance):
    request = providers.request_context(run)
    request["task"] = (
        "Return one complete C softmax proposal using the measured discovery feedback. Preserve scalar expf, precision and the full declared input domain. No threads, external libraries, altered compiler flags, undefined behavior or precomputed answers. No tools or filesystem access."
    )
    request["max_proposals"] = 1
    request["timing_workload"] = {
        "distribution": "Gaussian finite float32 from the frozen generator",
        "interface": "Prebound buffers; allocation is outside timed calls",
        "validation_distributions_are_not_timing_distributions": True,
    }
    request["search_schedule"] = {
        "total_seconds": plan["search_seconds"],
        "elapsed_seconds": plan["search_seconds"] - remaining,
        "remaining_seconds": remaining,
        "this_response_seconds": allowance,
        "remaining_call_slots": plan["max_calls"] - len(checkpoint["rounds"]),
    }
    request["feedback"]["remaining_seconds"] = remaining
    incumbent = legacy.chosen(request["feedback"])
    last = checkpoint["rounds"][-1]["submitted"] if checkpoint["rounds"] else []
    wanted = {incumbent, "prior_044", *last}
    if not checkpoint["rounds"]:
        wanted.update(p["id"] for p in request["existing_proposals"])
    request["existing_proposals"] = [p for p in request["existing_proposals"] if p["id"] in wanted]
    request["incumbent_id"] = incumbent
    request["previous_rounds"] = checkpoint["rounds"]
    request["failure_diagnostics"] = {}
    state = engine.verify(run)
    for name, proposal in state["proposals"].items():
        if proposal["rejected"]:
            job = state["jobs"].get(proposal["rejected"].get("job"), {})
            if job.get("result"):
                request["failure_diagnostics"][name] = json.dumps(read(run / job["result"]))[:4000]
    return request


def arm(out, model, author=codex_provider.invoke):
    plan = verify(out)
    checkpoint_file = out / (model + ".json")
    if checkpoint_file.exists():
        checkpoint = read(checkpoint_file)
        if checkpoint["status"] == "sealed":
            return checkpoint
    else:
        checkpoint = {
            "model": model,
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "rounds": [],
        }
        write(checkpoint_file, checkpoint)
    start = datetime.fromisoformat(checkpoint["started_utc"]).timestamp()
    deadline = time.monotonic() + max(0, start + plan["search_seconds"] - time.time())
    run = out / model
    if not run.exists():
        legacy.initialize(out, model)
    if "seed_seconds" not in checkpoint:
        engine.execute(run, "discovery", deadline=deadline)
        checkpoint["seed_seconds"] = time.time() - start
        checkpoint["retained_choice"] = legacy.chosen(
            engine.feedback(run),
            {p["id"] for p in read(out / "seeds.json") if p["id"].startswith("retained_")},
        )
        write(checkpoint_file, checkpoint)
    while len(checkpoint["rounds"]) < plan["max_calls"]:
        remaining = max(0, deadline - time.monotonic())
        allowance = min(plan["call_seconds"], remaining - plan["reserve_seconds"])
        if allowance < plan["minimum_call_seconds"]:
            checkpoint["stop_reason"] = "insufficient_time_for_response_and_discovery"
            break
        index = len(checkpoint["rounds"])
        folder = out / "authors" / model / f"round_{index:02d}"
        if folder.exists():
            raise RuntimeError("Partial round exists: preserve it; no silent retries")
        request = context(run, plan, checkpoint, remaining, allowance)
        stamp(
            f"AUTHOR START model={model} round={index + 1} remaining={remaining:.0f}s response_cap={allowance:.0f}s result={folder}"
        )
        receipt = author(
            request, folder, model=model, effort=plan["effort"], timeout=allowance, maximum=1
        )
        record = {
            "round": index + 1,
            "eligible": receipt["eligible"],
            "timed_out": receipt["timed_out"],
            "author_seconds": receipt["elapsed_seconds"],
            "submitted": [],
            "failures": [],
        }
        if receipt["eligible"]:
            proposal = read(folder / "response.json")["proposals"][0]
            digest = hashlib.sha256(proposal["source"].encode()).hexdigest()
            seen = {p["source_sha256"] for p in engine.verify(run)["proposals"].values()}
            if digest in seen:
                record["failures"].append("duplicate source consumes slot")
            else:
                try:
                    name = f"round_{index + 1:02d}"
                    legacy.submit(run, proposal, name)
                    record["submitted"].append(name)
                except (ValueError, KeyError, TypeError) as error:
                    record["failures"].append(str(error))
        engine.execute(run, "discovery", deadline=deadline)
        feedback = engine.feedback(run)
        write(folder / "feedback.json", feedback)
        record.update(choice=legacy.chosen(feedback), elapsed_seconds=time.time() - start)
        checkpoint["rounds"].append(record)
        write(checkpoint_file, checkpoint)
        stamp(
            f"AUTHOR COMPLETE model={model} round={index + 1} eligible={receipt['eligible']} choice={record['choice']}"
        )
    else:
        checkpoint["stop_reason"] = "call_cap_reached"
    legacy.close_incomplete(run)
    engine.seal(run)
    checkpoint.update(status="sealed", search_seconds=time.time() - start)
    write(checkpoint_file, checkpoint)
    return checkpoint


def require_sealed(out):
    if not (out / "discovery_complete.json").exists():
        raise ValueError("all model and control discovery choices must be sealed first")
    for name, digest in read(out / "discovery_complete.json")["seals"].items():
        assert sha(out / name / "seal.json") == digest, "selection changed"
        assert engine.verify(out / name)["sealed"]


def run(out):
    plan = verify(out)
    if (out / "summary.json").exists():
        require_sealed(out)
        return read(out / "summary.json")
    codex_provider.require_chatgpt()
    names = [*plan["order"], "enumerated"]
    if not (out / "discovery_complete.json").exists():
        for model in plan["order"]:
            arm(out, model)
        control_clock = out / "enumerated_wall.json"
        if not control_clock.exists():
            started = time.monotonic()
            control = legacy.initialize(out, "enumerated")
            for proposal in read(out / "enumeration.json"):
                legacy.submit(control, proposal, proposal["id"])
            engine.execute(control, "discovery", deadline=started + plan["search_seconds"])
            legacy.close_incomplete(control)
            engine.seal(control)
            write(control_clock, {"discovery_seconds": time.monotonic() - started})
        write(
            out / "discovery_complete.json",
            {
                "seals": {name: sha(out / name / "seal.json") for name in names},
                "utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    require_sealed(out)
    reports = {}
    for name in names:
        evaluation_wall = out / name / "evaluation_wall.json"
        if not evaluation_wall.exists():
            evaluation_start = out / name / "evaluation_started.json"
            if not evaluation_start.exists():
                write(evaluation_start, {"unix_seconds": time.time()})
            started = read(evaluation_start)["unix_seconds"]
            engine.execute(out / name, "evaluation")
            engine.report(out / name)
            write(
                evaluation_wall,
                {"seconds": time.time() - started, "includes_resume_downtime": True},
            )
        reports[name] = read(out / name / "summary.json")
    comparisons, costs = [], {}
    for model in plan["models"]:
        checkpoint = read(out / (model + ".json"))
        costs[model] = (
            checkpoint["search_seconds"] + read(out / model / "evaluation_wall.json")["seconds"]
        )
        winner = reports[model]["frozen_global_choice"]
        for suffix, reference in [
            ("prior", (model, "prior_044")),
            ("retained", (model, checkpoint["retained_choice"])),
            ("enumerated", ("enumerated", reports["enumerated"]["frozen_global_choice"])),
        ]:
            comparisons.append(
                paired.pair(
                    out, pair_label(model, suffix), (model, winner), reference, costs[model]
                )
            )
    for i, model in enumerate(plan["models"]):
        for other in plan["models"][i + 1 :]:
            comparisons.append(
                paired.pair(
                    out,
                    pair_label(model, other),
                    (model, reports[model]["frozen_global_choice"]),
                    (other, reports[other]["frozen_global_choice"]),
                    costs[model],
                )
            )
    receipts = {
        model: [read(p) for p in sorted((out / "authors" / model).glob("*/receipt.json"))]
        for model in plan["models"]
    }
    result = {
        "plan": plan,
        "reports": reports,
        "loops": {m: read(out / (m + ".json")) for m in plan["models"]},
        "search_plus_evaluation_seconds": costs,
        "pairs": comparisons,
        "receipts": receipts,
        "cost_usd": None,
        "usage_scope": "Available completed-turn receipts; missing timeout usage is unknown, not zero. Equal xhigh labels do not imply equal inference compute.",
    }
    write(out / "summary.json", result)
    lines = [
        "# Campaign 047: model-choice pilot",
        "",
        plan["rules"],
        "",
        "| Model | Completed / attempts | Search + evaluation, min | Frozen choice |",
        "| --- | --- | --- | --- |",
    ]
    for model in plan["models"]:
        rounds = result["loops"][model]["rounds"]
        lines.append(
            f"| {model} | {sum(r['eligible'] for r in rounds)}/{len(rounds)} | {costs[model] / 60:.2f} | {reports[model]['frozen_global_choice']} |"
        )
    lines += ["", "| Paired comparison | Geomean | Incremental gate |", "| --- | --- | --- |"]
    for pair in comparisons:
        lines.append(
            f"| {pair['label']} | {pair.get('geomean')} | {pair.get('incremental_passed')} |"
        )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["freeze", "verify", "run"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--calls", type=int, default=6)
    args = parser.parse_args()
    out = args.output.resolve()
    stamp(
        f"START phase={args.phase} expected_run=90-110min result={out / 'summary.json'} log={out.parent / (out.name + '.log')}"
    )
    if args.phase == "freeze":
        freeze(out, args.seconds, args.calls)
    elif args.phase == "verify":
        verify(out)
    else:
        run(out)
    stamp(
        f"COMPLETE phase={args.phase} result={out / 'summary.json'} log={out.parent / (out.name + '.log')}"
    )


if __name__ == "__main__":
    main()
