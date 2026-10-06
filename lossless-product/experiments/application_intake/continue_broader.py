"""Continue interrupted broader search with memory feedback and new final cases.

The earlier final cases become development data. A fresh protocol is frozen
before authoring. This is a follow-up, not a matched timer experiment.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

from lossless import codex_provider
from lossless.applications import initialize, patches
from lossless.applications.comparison import compare, eligible
from lossless.applications.config import resolve
from lossless.applications.discovery import inventory
from lossless.applications.replay import copy_project
from lossless.applications.runtime import clock

from broader_study import (
    HERE,
    SCOPE,
    application_case,
    compact,
    make_application_cases,
    memory_and_setup,
    new_records,
)
from study import REVISION, git, log, sha, write

BASELINES = ("public_stock", "public_previous", "reuse_stock", "reuse_previous")
POLICY = {"minimum_speedup": 1.02, "max_case_regression": 0.03, "significance": 0.05}


def make_cases(project):
    cases = make_application_cases(project)
    for case in cases:
        case["split"] = "discovery"
    final = [
        ("next_small", 229, 20, 61001, {}, "oscillatory"),
        ("next_large", 5003, 10, 62001, {"poly_order": 4, "return_coef": True}, "steps"),
        ("next_f32", 1057, 18, 63001, {"dtype": "float32", "use_original": True}, "peaks"),
        ("next_shared", 1601, 20, 64001, {"weight_mode": "shared", "return_coef": True}, "steps"),
        ("next_changing", 1201, 14, 65001, {"weight_mode": "changing"}, "oscillatory"),
        ("next_mask", 2081, 10, 66001, {"mask_initial_peaks": True, "use_original": True}, "peaks"),
        (
            "next_descending",
            733,
            18,
            67001,
            {"ordering": "descending", "weight_mode": "sparse"},
            "steps",
        ),
        (
            "next_orders",
            1327,
            16,
            68001,
            {"varying_order": True, "return_coef": True},
            "oscillatory",
        ),
        ("next_irregular", 1019, 14, 69001, {"ordering": "shuffled", "irregular": True}, "peaks"),
        ("next_flat_zero", 457, 14, 70001, {"max_iter": 13}, "flat"),
        ("next_single", 337, 1, 71001, {"poly_order": 5, "return_coef": True}, "peaks"),
    ]
    for name, size, batch, seed, options, pattern in final:
        records = new_records(project, name, size, batch, seed, options, pattern=pattern)
        cases.append(application_case(name, "evaluation", records, requests=3))
    return cases


def case_description(case, project):
    records = case["calls"][0]["args"][0]
    first = records[0]
    array = np.load(project / first["y"]["$npy"], allow_pickle=False)
    return {
        "id": case["id"],
        "requests": len(case["calls"]),
        "fits_per_request": len(records),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "options": [
            {k: ("array" if k == "weights" else v) for k, v in row["options"].items()}
            for row in records[:4]
        ],
    }


def feedback(report, extra):
    result = {k: report[k] for k in ("status", "reason", "failed_case") if k in report}
    stats = report.get("statistics")
    if stats:
        result["speedup"] = stats["case_balanced_geomean_speedup"]
        result["cases"] = [{"id": row["id"], "speedup": row["speedup"]} for row in stats["cases"]]
    if extra:
        result.update(
            startup_eligible=extra["startup_eligible"],
            memory_eligible=extra["memory_eligible"],
            memory=[
                {
                    "id": row["id"],
                    "matched": row["matched"],
                    "eligible": row["eligible"],
                    "reference_traced_peak": row["reference"]["traced_peak_bytes"],
                    "candidate_traced_peak": row["candidate"]["traced_peak_bytes"],
                    "rss_ratio": row["candidate"]["peak_rss_bytes"]
                    / row["reference"]["peak_rss_bytes"],
                }
                for row in extra["memory"]
            ],
        )
    return result


def prepare(args):
    project, out = args.project.resolve(), args.output.resolve()
    if git(project, "rev-parse", "HEAD") != REVISION or git(project, "status", "--porcelain"):
        raise ValueError("expected clean pinned upstream checkout")
    if out == project or out.is_relative_to(project) or project.is_relative_to(out):
        raise ValueError("separate output directory required")
    out.mkdir(parents=True, exist_ok=False)
    base = out / "prepared"
    copy_project(project, base, inventory(project))
    shutil.copyfile(HERE / "spectrum_workload.py", base / "spectrum_workload.py")
    write(out / "cases.json", make_cases(base))
    source = (base / "spectrum_workload.py").read_text()
    previous = json.loads((HERE / "frozen-candidates.json").read_text())[2]
    seed = json.loads((HERE / "broader-candidate.json").read_text())
    variants = {}
    for name in (*BASELINES, "broader_previous"):
        path = out / "baselines" / name
        copy_project(base, path, inventory(base))
        if name.startswith("public"):
            (path / "spectrum_workload.py").write_text(
                source.replace('MODE = "reuse"', 'MODE = "public"')
            )
        if name.endswith("previous"):
            patches.apply(base, path, previous, [SCOPE[4]])
        if name == "broader_previous":
            patches.apply(path, path, seed, SCOPE)
        variants[name] = path
    initialize(
        variants["reuse_previous"],
        out / "job",
        entry="spectrum_workload:make_processor",
        factory=True,
        cases=out / "cases.json",
        python=sys.executable,
    )
    config_path = out / "job/lossless.json"
    config = json.loads(config_path.read_text())
    config["environment"]["inherit_env"] = ["OPENBLAS_NUM_THREADS"]
    config["entry"]["observe"] = "spectrum_workload:observe_processor"
    config["replay"].update(timeout_seconds=60, max_output_bytes=32000000)
    write(config_path, config)
    value, cases = resolve(config_path)
    snapshots = {name: inventory(path) for name, path in variants.items()}
    old_prompt = json.loads(
        (args.previous_study / "run/candidates/candidate_11/author/prompt.json").read_text()
    )
    context = {
        k: old_prompt[k]
        for k in (
            "task",
            "contract",
            "environment",
            "measurement_scope",
            "readonly_context_symbols",
        )
    }
    context["sources"] = patches.scope_sources(variants["reuse_previous"], SCOPE)
    context["discovery_cases"] = [
        case_description(c, base) for c in cases if c["split"] == "discovery"
    ]
    context["prior_discovery_profile"] = {
        k: v
        for k, v in old_prompt["evidence"].items()
        if v["kind"] == "timing"
        or (v["kind"] == "host_self_time" and int(k.rsplit(":", 1)[-1]) < 5)
    }
    context["prior_frozen_candidate"] = seed
    context["continuation"] = (
        "The previous authoring session exhausted subscription quota at 1755.52s of a 3600s search. "
        "Its candidate_11 later passed exact/speed/startup checks on new cases at 1.16255x the strong baseline "
        "but failed the unchanged traced allocation cap: +54.02%/319.6KiB on changed weights, +21.27% on "
        "masking and +48.88% on descending sparse weights. Those cases are DEVELOPMENT now. New final "
        "cases were frozen before this continuation and will not be supplied. Continue from that candidate, "
        "returning complete bodies relative to the ORIGINAL reuse_previous comparator. Memory results now "
        "follow every exact candidate. Retain useful speed while satisfying peak traced allocation<=1.2x, "
        "RSS<=1.2x and setup increase<=100ms. No relaxed gate, precision, output, iteration or cache-ownership "
        "contract. Max four fitters and 16MiB persistent storage still apply. Use remaining allowance to "
        "propose/test/revise; empty edits are an abstention, not early stop. Highest development speed "
        "among candidates passing speed, memory and startup gates is frozen once. Final failure ends selection."
    )
    context["policy"] = {
        **POLICY,
        "max_traced_peak_ratio": 1.2,
        "max_peak_rss_ratio": 1.2,
        "max_setup_increase_seconds": 0.1,
        "discovery_pairs": 3,
        "final_pairs": 12,
    }
    write(out / "author-context.json", context)
    protocol = {
        "study": "broader-application-continuation",
        "parent_study": str(args.previous_study.resolve()),
        "parent_report_sha256": sha(args.previous_study / "run/report.json"),
        "seed_candidate_sha256": sha(HERE / "broader-candidate.json"),
        "source_hashes": {
            p.name: sha(p)
            for p in (Path(__file__), HERE / "broader_study.py", HERE / "spectrum_workload.py")
        },
        "runner_hashes": {p.name: sha(p) for p in Path(patches.__file__).parent.glob("*.py")},
        "cases_sha256": sha(out / "cases.json"),
        "config_sha256": sha(config_path),
        "context_sha256": sha(out / "author-context.json"),
        "comparators": {k: v["files"] for k, v in snapshots.items()},
        "deployment_comparator": "reuse_previous",
        "search_seconds": args.search_seconds,
        "max_candidates": args.max_candidates,
        "policy": context["policy"],
        "selection": "Highest three-pair development speed among exact/speed/memory/setup passing candidates; if none, freeze highest exact speed for explanation only and retain reference.",
        "final_policy": "One frozen source; original four baselines must all pass unchanged gates at twelve pairs. Prior broader candidate is an additional diagnostic comparison, not a qualified deployment baseline.",
        "scope": "Follow-up with earlier final data reclassified as development. No matched time-budget inference. CPU one OpenBLAS thread; exact complete processor requests; setup and memory separate.",
        "model": args.model,
        "effort": args.effort,
    }
    write(out / "protocol.json", protocol)
    return out, value, cases, variants, snapshots, context, seed, protocol


def evaluate(proposal, folder, reference, ref_snapshot, value, cases, deadline):
    patches.validate(proposal, SCOPE)
    candidate = folder / "project"
    copy_project(reference, candidate, ref_snapshot)
    difference, identity = patches.apply(reference, candidate, proposal, SCOPE)
    (folder / "candidate.patch").write_text(difference)
    snapshot = inventory(candidate)
    write(folder / "candidate-source.json", snapshot)
    report = compare(
        reference,
        candidate,
        ref_snapshot,
        snapshot,
        value,
        cases,
        folder / "discovery",
        pairs=3,
        deadline=deadline,
        progress=log,
        minimum_speedup=1.02,
    )
    extra = (
        memory_and_setup(
            reference,
            candidate,
            ref_snapshot,
            snapshot,
            value,
            cases,
            report,
            folder / "discovery-memory",
            deadline,
        )
        if report["status"] == "matched"
        else {}
    )
    passed = (
        eligible(report, POLICY, final=False)
        and extra.get("memory_eligible", False)
        and extra.get("startup_eligible", False)
    )
    result = {
        "candidate_identity": identity,
        **feedback(report, extra),
        "all_development_gates_passed": passed,
    }
    write(folder / "result.json", result)
    return result, snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--previous-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-seconds", type=float, default=1845)
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.search_seconds <= 3600 or not 1 <= args.max_candidates <= 256:
        raise ValueError("require search<=3600s and 1..256 candidates")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    started = clock()
    log(
        f"START expected=50-65min result={args.output}/summary.json log={args.output.with_suffix('.log')}"
    )
    out, value, cases, variants, snapshots, context, seed, protocol = prepare(args)
    log(
        f"FROZEN cases={protocol['cases_sha256']} comparator=reuse_previous; earlier final cases are development"
    )
    if args.prepare_only:
        log(f"PREPARED result={out / 'protocol.json'} log={out.with_suffix('.log')}")
        return 0
    codex_provider.require_chatgpt()
    search_started = clock()
    deadline = search_started + args.search_seconds
    write(out / "search-clock.json", {"started": search_started, "deadline": deadline})
    discovery = [c for c in cases if c["split"] == "discovery"]
    evaluation = [c for c in cases if c["split"] == "evaluation"]
    summary = {"status": "searching", "attempts": [], "comparisons": {}, "llm_calls": 0}
    history, completed = [], []
    failures = 0
    for number in range(args.max_candidates):
        remaining = deadline - clock()
        if remaining <= 0:
            summary["stop_reason"] = "discovery_deadline"
            break
        folder = out / "candidates" / f"candidate_{number:02d}"
        folder.mkdir(parents=True)
        log(f"AUTHOR candidate={number} remaining={remaining:.1f}s")
        if number == 0:
            proposal = seed
            receipt = {"provider": "retained-seed", "eligible": True, "elapsed_seconds": 0}
        else:
            passing = [r for r in completed if r["result"]["all_development_gates_passed"]]
            best = max(passing, key=lambda r: r["result"]["speedup"], default=None)
            kept = {row["id"] for row in history[-3:]}
            if best:
                kept.add(best["id"])
            request = {
                **context,
                "previous_attempts": [
                    row
                    if row["id"] in kept
                    else {
                        "id": row["id"],
                        "hypothesis": row.get("proposal", {}).get("hypothesis"),
                        "result": row["result"],
                    }
                    for row in history
                ],
                "best_eligible_candidate": best["id"] if best else None,
                "search_seconds_remaining": remaining,
                "remaining_candidate_calls": args.max_candidates - number,
            }
            receipt = codex_provider.invoke(
                request,
                folder / "author",
                model=args.model,
                effort=args.effort,
                timeout=min(600, remaining),
                schema_override=patches.schema(),
                validator=lambda p: patches.validate(p, SCOPE),
            )
            summary["llm_calls"] += 1
            proposal = (
                json.loads((folder / "author/response.json").read_text())
                if receipt["eligible"]
                else None
            )
        row = {"id": folder.name, "receipt": receipt}
        summary["attempts"].append(row)
        if proposal is None:
            row["result"] = {
                "status": "provider_failed",
                "messages": receipt.get("provider_messages", []),
            }
            history.append({"id": folder.name, "result": row["result"]})
            failures += 1
            write(out / "summary.json", summary)
            if failures >= 3:
                summary.update(
                    status="authoring_interrupted",
                    stop_reason="provider_failures",
                    remaining_seconds=max(0, deadline - clock()),
                )
                write(out / "summary.json", summary)
                log(
                    f"PAUSED provider unavailable; no final cases opened result={out / 'summary.json'}"
                )
                return 3
            continue
        failures = 0
        proposal.pop("usage", None)
        write(folder / "proposal.json", proposal)
        if not proposal["edits"]:
            row["result"] = {"status": "author_abstained"}
        else:
            try:
                row["result"], source = evaluate(
                    proposal,
                    folder,
                    variants["reuse_previous"],
                    snapshots["reuse_previous"],
                    value,
                    discovery,
                    deadline,
                )
                if row["result"]["status"] == "matched":
                    completed.append({"id": folder.name, "result": row["result"], "source": source})
            except (ValueError, SyntaxError, OSError, TimeoutError) as error:
                row["result"] = {"status": "failed", "reason": str(error)}
        history.append({"id": folder.name, "proposal": proposal, "result": row["result"]})
        log(
            f"RESULT candidate={number} status={row['result']['status']} speed={row['result'].get('speedup')} gates={row['result'].get('all_development_gates_passed')}"
        )
        write(out / "summary.json", summary)
    summary.setdefault("stop_reason", "candidate_cap")
    summary["search_elapsed_seconds"] = clock() - search_started
    passing = [r for r in completed if r["result"]["all_development_gates_passed"]]
    winner = max(passing or completed, key=lambda r: r["result"]["speedup"], default=None)
    if winner is None:
        summary.update(status="reference_retained", reason="no complete exact candidate")
    else:
        summary["frozen_candidate"] = winner["id"]
        write(out / "frozen-winner.json", winner)
        candidate = out / "candidates" / winner["id"] / "project"
        log(f"FINAL frozen={winner['id']} no further author calls")
        final_deadline = clock() + 2400
        for name, reference in variants.items():
            report = compare(
                reference,
                candidate,
                snapshots[name],
                winner["source"],
                value,
                evaluation,
                out / "final" / name,
                pairs=12,
                deadline=final_deadline,
                progress=log,
                minimum_speedup=1.02,
            )
            extra = (
                memory_and_setup(
                    reference,
                    candidate,
                    snapshots[name],
                    winner["source"],
                    value,
                    evaluation,
                    report,
                    out / "final-memory" / name,
                    final_deadline,
                )
                if report["status"] == "matched"
                else {}
            )
            passed = (
                eligible(report, POLICY, final=True)
                and extra.get("memory_eligible", False)
                and extra.get("startup_eligible", False)
            )
            summary["comparisons"][name] = {**compact(report), **extra, "all_gates_passed": passed}
            write(out / "summary.json", summary)
        summary["status"] = (
            "accepted_experiment"
            if passing and all(summary["comparisons"][n]["all_gates_passed"] for n in BASELINES)
            else "reference_retained"
        )
        summary["candidate_unchanged"] = inventory(candidate)["files"] == winner["source"]["files"]
    summary["provenance_unchanged"] = (
        not git(args.project.resolve(), "status", "--porcelain")
        and sha(out / "cases.json") == protocol["cases_sha256"]
        and sha(out / "job/lossless.json") == protocol["config_sha256"]
        and sha(out / "author-context.json") == protocol["context_sha256"]
        and all(inventory(variants[n])["files"] == snapshots[n]["files"] for n in variants)
        and all(sha(HERE / n) == h for n, h in protocol["source_hashes"].items())
        and all(
            sha(Path(patches.__file__).parent / n) == h
            for n, h in protocol["runner_hashes"].items()
        )
    )
    if not summary["provenance_unchanged"] or not summary.get("candidate_unchanged", True):
        summary.update(status="failed", reason="source/configuration changed")
    summary["elapsed_seconds"] = clock() - started
    write(out / "summary.json", summary)
    log(
        f"DONE status={summary['status']} elapsed={summary['elapsed_seconds']:.1f}s result={out / 'summary.json'} log={out.with_suffix('.log')}"
    )
    return 0 if summary["status"] in {"accepted_experiment", "reference_retained"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
