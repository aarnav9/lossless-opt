"""One-hour proposal/measurement/revision pilot with globally withheld final cases."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time

from lossless import recipes
from lossless._native import engine, providers as protocol
from lossless._native.common import read, write, sha, stamp

import author

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TIMER_PATH = HERE.parent / "timers_044/study.py"
loader = importlib.util.spec_from_file_location("timer_044", TIMER_PATH)
timer = importlib.util.module_from_spec(loader)
loader.loader.exec_module(timer)
EVIDENCE = HERE.parents[1] / "docs/assets/timers-044.json.gz"


def cases():
    return [
        {
            "id": f"{split}_{i}_{layout}",
            "split": split,
            "rows": rows,
            "columns": cols,
            "layout": layout,
        }
        for split, shapes in [
            ("discovery", [(8, 193), (47, 773), (96, 1537)]),
            ("evaluation", [(5, 199), (39, 797), (103, 1567), (17, 1021)]),
        ]
        for i, (rows, cols) in enumerate(shapes)
        for layout in ["c", "f", "slice2"]
    ]


def enumeration(source):
    """Frozen factorial tuning of the same strong starting kernel given to the LLM."""
    start = source.index("static inline void write_exponentials(")
    end = source.index("static inline double sum_row(", start)
    scale_start = source.index("static inline void scale_row(")
    scale_end = source.index("static inline int two_values(", scale_start)
    result = []
    for unroll in [1, 2, 4, 8, 16, 32]:
        for shortcut in [True, False]:
            for scalar_scale in [False, True]:
                body = "".join(f"o[j+{k}] = expf(p[(j+{k})*s]-m);\n" for k in range(unroll))
                exp = f"""static inline void write_exponentials(const float *p, float *o, size_t n, size_t s, float m) {{
                    size_t j=0; for(; n-j>={unroll}; j+={unroll}) {{{body}}}
                    for(; j<n; ++j) o[j]=expf(p[j*s]-m);
                }}\n\n"""
                candidate = source[:start] + exp + source[end:]
                if not shortcut:
                    candidate = candidate.replace(
                        "if (two_values(p,cols,cs,lo,hi,&high_count))", "if (0)"
                    )
                if scalar_scale:
                    candidate = (
                        candidate[:scale_start]
                        + """static inline void scale_row(float *p, size_t n, float scale) {
                        for(size_t j=0;j<n;++j) p[j] *= scale;
                    }\n\n"""
                        + candidate[scale_end:]
                    )
                result.append(
                    {
                        "id": f"enum_u{unroll}_two{int(shortcut)}_scalar{int(scalar_scale)}",
                        "hypothesis": f"Frozen factorial: expf unroll {unroll}, two-value shortcut {shortcut}, scalar normalization {scalar_scale}",
                        "source": candidate,
                    }
                )
    random.Random(4501).shuffle(result)
    assert len({hashlib.sha256(p["source"].encode()).hexdigest() for p in result}) == 24
    return result


def frozen_sources():
    return [
        *sorted(HERE.glob("*.py")),
        *timer.frozen_sources(),
        *sorted((HERE.parents[1] / "src/lossless").rglob("*.py")),
        *sorted((HERE.parents[1] / "src/lossless/resources").glob("*.c")),
    ]


def freeze(out, seconds=3600, call_seconds=1800, rounds=8):
    if not 1 <= seconds <= 28800 or not 1 <= call_seconds <= seconds or not 1 <= rounds <= 8:
        raise ValueError("positive bounded total/call limits and 1..8 rounds required")
    out.mkdir(parents=True, exist_ok=False)
    archive = json.loads(gzip.decompress(EVIDENCE.read_bytes()))
    incumbent = archive["authors"]["announced_30m"][0]["response"]["record"]["proposals"][2]
    seeds = [
        {**incumbent, "id": "prior_044"},
        *[{**p, "id": "retained_" + p["id"]} for p in recipes.native_candidates("softmax")],
    ]
    write(out / "seeds.json", seeds)
    write(out / "enumeration.json", enumeration(incumbent["source"]))
    write(out / "schema.json", archive["frozen"]["announced_30m/schema.json"]["record"])
    spec = {
        "operator": "softmax",
        "comparator": "numpy_buffered",
        "cases": cases(),
        "budget_seconds": 3600,
        "job_timeout_seconds": 8,
        "compile_timeout_seconds": 8,
        "target_ns": 500000,
        "screening_blocks": 5,
        "confirmation_blocks": 15,
        "max_proposals": len(seeds) + 3 * rounds,
        "min_speedup": 1.02,
        "cache": {"directory": None},
    }
    write(out / "spec.json", spec)
    write(
        out / "plan.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "campaign": "045",
            "model": "gpt-6-astra",
            "effort": "xhigh",
            "replicates": 1,
            "search_seconds": seconds,
            "call_seconds": call_seconds,
            "max_rounds": rounds,
            "proposals_per_round": 3,
            "new_candidate_cap": 3 * rounds,
            "discovery_reserve_seconds": min(90, seconds / 10),
            "minimum_call_seconds": min(180, call_seconds / 2),
            "incumbent_origin": "044 announced_30m author_0 proposal_2, chosen before new measurements; no historical speed data sent to author",
            "incumbent_evidence_sha256": sha(EVIDENCE),
            "hashes": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
            "source_hashes": {str(p): sha(p) for p in frozen_sources()},
            "rules": "One exploratory trajectory, up to one hour including seed measurement, authors and discovery. "
            "Fresh isolated responses receive explicit elapsed/remaining time, incumbent source and discovery feedback. "
            "Three proposal slots per round; duplicates, invalid responses and timeouts consume slots. "
            "No tools or access to held-out cases. Final cases open only after loop and control selections are frozen. "
            "Enumeration uses the same seeds, identical maximum candidate/timing allowances and a frozen shuffled "
            "parameter grid, truncated to the LLM's attempted slots. Actual time/tokens reported separately. "
            "Fresh paired final comparisons against first-round choice, prior_044, retained and enumeration; "
            "incremental pass requires geomean>=1.02 and every case lower paired interval>1. "
            "Final validation is outside the one-hour discovery budget. No population-level or default-promotion claim.",
        },
    )


def verify(out):
    plan = read(out / "plan.json")
    for file, digest in plan["hashes"].items():
        assert sha(out / file) == digest, f"frozen input changed: {file}"
    for file, digest in plan["source_hashes"].items():
        assert sha(file) == digest, f"frozen source changed: {file}"
    return plan


def submit(run, proposal, name):
    folder = run / "submissions" / name
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "kernel.c").write_text(proposal["source"])
    write(
        folder / "proposal.json",
        {
            "schema_version": 1,
            "id": name,
            "operator": "softmax",
            "source_file": "kernel.c",
            "hypothesis": proposal["hypothesis"],
        },
    )
    engine.submit(run, folder / "proposal.json")


def chosen(feedback, names=None):
    valid = {
        name: record
        for name, record in feedback["candidates"].items()
        if record["complete"] and (names is None or name in names)
    }
    if not valid:
        return "deployment_reference"
    winner = max(valid, key=lambda n: (valid[n]["geomean_speedup_vs_comparator"], n))
    return (
        winner if valid[winner]["geomean_speedup_vs_comparator"] > 1.02 else "deployment_reference"
    )


def prompt(run, plan, checkpoint, remaining, allowance):
    context = protocol.request_context(run)
    context["task"] = (
        "Return exactly three distinct complete C softmax kernels as {proposals:[{id,hypothesis,source}]}. "
        "This is a proposal -> benchmark -> revision loop. Use the supplied measured discovery feedback "
        "to improve the incumbent. Later rounds can revise failures; each consumes another proposal slot. "
        "Return JSON only before this response deadline. No tools, browsing, filesystem or code execution. "
        "Preserve scalar expf; no approximate exponentials, reduced precision, threads, external libraries, "
        "compiler changes, undefined behavior or shape-specific answers. Handle the entire declared input domain."
    )
    context["response_format"] = {
        "proposals": [{"id": "name", "hypothesis": "reason", "source": "complete C source"}]
    }
    context["max_proposals"] = 3
    context["search_schedule"] = {
        "round": len(checkpoint["rounds"]) + 1,
        "maximum_rounds": plan["max_rounds"],
        "total_allowance_seconds": plan["search_seconds"],
        "elapsed_seconds": plan["search_seconds"] - remaining,
        "remaining_search_seconds": remaining,
        "this_response_timeout_seconds": allowance,
        "remaining_proposal_slots": plan["new_candidate_cap"] - checkpoint["attempted_slots"],
        "instructions": "Allowance includes reasoning and JSON generation; finish early when ready. "
        "A controller supplies a fresh clock update and benchmarks after each response. "
        "Reserve time to return three complete sources; incomplete responses are excluded.",
    }
    context["feedback"]["remaining_seconds"] = remaining
    best = chosen(context["feedback"])
    last = checkpoint["rounds"][-1]["submitted"] if checkpoint["rounds"] else []
    wanted = {best, "prior_044", *last}
    if not checkpoint["rounds"]:
        wanted.update(p["id"] for p in context["existing_proposals"])
    context["existing_proposals"] = [p for p in context["existing_proposals"] if p["id"] in wanted]
    context["incumbent_id"] = best
    context["previous_rounds"] = checkpoint["rounds"]
    # Include bounded compiler/validation diagnostics for failed discovery proposals.
    state = engine.verify(run)
    diagnostics = {}
    for name, proposal in state["proposals"].items():
        if proposal["rejected"]:
            job = state["jobs"].get(proposal["rejected"].get("job"))
            if job and job.get("result"):
                diagnostics[name] = json.dumps(read(run / job["result"]))[:6000]
    context["failure_diagnostics"] = diagnostics
    return context


def initialize(out, name):
    run = out / name
    engine.initialize(out / "spec.json", run)
    for proposal in read(out / "seeds.json"):
        submit(run, proposal, proposal["id"])
    return run


def close_incomplete(run):
    with engine.locked(run) as (run, state):
        data = engine.feedback_data(run, state)
        for name, entry in data["candidates"].items():
            if not entry["complete"] and not entry["rejected"]:
                state["proposals"][name]["rejected"] = {"reason": "discovery_budget_exhausted"}
                engine.event(run, state, "candidate_cutoff", candidate=name)


def discovery(out, author_fn=None):
    plan = verify(out)
    if (out / "discovery_complete.json").exists():
        return read(out / "discovery_complete.json")
    checkpoint_path = out / "loop.json"
    if checkpoint_path.exists():
        checkpoint = read(checkpoint_path)
    else:
        checkpoint = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "rounds": [],
            "attempted_slots": 0,
            "status": "running",
        }
        write(checkpoint_path, checkpoint)
    started_wall = datetime.fromisoformat(checkpoint["started_utc"]).timestamp()
    deadline = time.monotonic() + max(0, started_wall + plan["search_seconds"] - time.time())
    run = out / "iterative"
    if not run.exists():
        initialize(out, "iterative")
    if "seed_seconds" not in checkpoint:
        engine.execute(run, "discovery", deadline=deadline)
        checkpoint["seed_seconds"] = time.time() - started_wall
        seed_feedback = engine.feedback(run)
        checkpoint["retained_choice"] = chosen(
            seed_feedback,
            {p["id"] for p in read(out / "seeds.json") if p["id"].startswith("retained_")},
        )
        write(checkpoint_path, checkpoint)
    while len(checkpoint["rounds"]) < plan["max_rounds"]:
        remaining = max(0, deadline - time.monotonic())
        allowance = min(plan["call_seconds"], remaining - plan["discovery_reserve_seconds"])
        if allowance < plan["minimum_call_seconds"]:
            checkpoint["stop_reason"] = "insufficient_time_for_another_response_and_discovery"
            break
        index = len(checkpoint["rounds"])
        folder = out / "rounds" / f"round_{index:02d}"
        if folder.exists():
            raise RuntimeError(
                "Incomplete round checkpoint; preserve evidence instead of silently retrying"
            )
        folder.mkdir(parents=True)
        request = prompt(run, plan, checkpoint, remaining, allowance)
        write(folder / "prompt.json", request)
        write(folder / "schema.json", read(out / "schema.json"))
        stamp(
            f"ROUND {index + 1} remaining={remaining:.0f}s response_cap={allowance:.0f}s result={folder}"
        )
        receipt = (author_fn or author.run)(folder, allowance, timer)
        proposed = read(folder / "response.json")["proposals"] if receipt["eligible"] else []
        record = {
            "round": index + 1,
            "eligible": receipt["eligible"],
            "timed_out": receipt["timed_out"],
            "author_seconds": receipt["elapsed_seconds"],
            "submitted": [],
            "failures": [],
        }
        state = engine.verify(run)
        seen = {p["source_sha256"] for p in state["proposals"].values()}
        for slot, proposal in enumerate(proposed):
            digest = hashlib.sha256(proposal["source"].encode()).hexdigest()
            name = f"round{index + 1:02d}_candidate{slot}"
            if digest in seen:
                record["failures"].append(
                    {"slot": slot, "reason": "duplicate source consumes slot"}
                )
                continue
            seen.add(digest)
            try:
                submit(run, proposal, name)
                record["submitted"].append(name)
            except (ValueError, KeyError, TypeError) as error:
                record["failures"].append({"slot": slot, "reason": str(error)})
        checkpoint["attempted_slots"] += 3
        engine.execute(run, "discovery", deadline=deadline)
        feedback = engine.feedback(run)
        write(folder / "feedback.json", feedback)
        record["choice"] = chosen(feedback)
        record["elapsed_seconds"] = time.time() - started_wall
        if index == 0:
            checkpoint["first_round_choice"] = record["choice"]
        checkpoint["rounds"].append(record)
        write(checkpoint_path, checkpoint)
        stamp(
            f"ROUND {index + 1} complete choice={record['choice']} elapsed={record['elapsed_seconds']:.1f}s"
        )
    else:
        checkpoint["stop_reason"] = "round_cap_reached"
    close_incomplete(run)
    engine.seal(run)
    checkpoint.update(status="sealed", search_seconds=time.time() - started_wall)
    write(checkpoint_path, checkpoint)
    # The fixed control receives the same seeds and number of attempted new slots.
    control_started = time.monotonic()
    control = initialize(out, "enumerated")
    for proposal in read(out / "enumeration.json")[: checkpoint["attempted_slots"]]:
        submit(control, proposal, proposal["id"])
    engine.execute(control, "discovery", deadline=control_started + plan["search_seconds"])
    close_incomplete(control)
    engine.seal(control)
    completion = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "loop": checkpoint,
        "enumerated_seconds": time.monotonic() - control_started,
        "enumerated_new_slots": checkpoint["attempted_slots"],
        "selection_sha256": {
            name: sha(out / name / "seal.json") for name in ["iterative", "enumerated"]
        },
    }
    write(out / "discovery_complete.json", completion)
    return completion


def require_sealed(out):
    if not (out / "discovery_complete.json").exists():
        raise ValueError("Finish and seal both discovery arms before opening any final case")
    for name, digest in read(out / "discovery_complete.json")["selection_sha256"].items():
        assert sha(out / name / "seal.json") == digest, "frozen choice changed"
        assert engine.verify(out / name)["sealed"]


def evaluate(out):
    verify(out)
    require_sealed(out)
    for name in ["iterative", "enumerated"]:
        started = time.monotonic()
        engine.execute(out / name, "evaluation")
        engine.report(out / name)
        write(out / name / "evaluation_wall.json", {"seconds": time.monotonic() - started})
    import analyze

    analyze.summarize(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["freeze", "run", "verify"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--call-seconds", type=int, default=1800)
    parser.add_argument("--rounds", type=int, default=8)
    args = parser.parse_args()
    out = args.output.resolve()
    stamp(
        f"START phase={args.phase} estimated=65-75min result={out / 'summary.json'} log={out.parent / (out.name + '.log')}"
    )
    if args.phase == "freeze":
        freeze(out, args.seconds, args.call_seconds, args.rounds)
    elif args.phase == "run":
        discovery(out)
        evaluate(out)
    else:
        verify(out)
    stamp(
        f"COMPLETE phase={args.phase} result={out / 'summary.json'} log={out.parent / (out.name + '.log')}"
    )


if __name__ == "__main__":
    main()
