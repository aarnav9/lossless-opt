"""Experimental source-body authoring with independent application measurements.

Trusted local code only. Copied projects and edit restrictions are provenance
boundaries, not isolation from malicious Python. Never edits the user's checkout.
"""

import ast
import copy
import html
import json
import math
from pathlib import Path
import shutil

from .. import codex_provider
from .._native.common import stamp, write
from ..jobs import fields, identity, positive, read_json
from . import patches
from .assessment import context_for
from .comparison import compare, eligible, run_case
from .config import analysis_settings, inspect, resolve
from .discovery import digest, inventory
from .profiling import profile
from .replay import ReplayResult, copy_project
from .runtime import clock


DEFAULTS = {
    "wall_time_seconds": 1800,
    "final_reserve_seconds": 600,
    "max_candidates": 4,
    "discovery_pairs": 3,
    "evaluation_pairs": 12,
    "minimum_speedup": 1.02,
    "max_case_regression": 0.03,
    "max_call_regression": 0.03,
    "significance": 0.05,
    "max_setup_increase_seconds": 0.1,
    "max_peak_rss_ratio": 1.2,
    "max_traced_peak_ratio": 1.2,
}


def resolve_plan(plan, budget):
    plan = copy.deepcopy(read_json(plan) if isinstance(plan, (str, Path)) else plan)
    required = {"schema_version", "source_scope", "contract"}
    fields(
        plan,
        required
        | {
            "provider",
            "context_symbols",
            "candidates",
            "honor_author_stop",
            "timing_scope",
            *DEFAULTS,
        },
        "application search plan",
        required,
    )
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise ValueError("unsupported application search plan")
    if (
        not isinstance(plan["contract"], str)
        or not plan["contract"].strip()
        or len(plan["contract"]) > 16000
    ):
        raise ValueError("describe the callable domain, observations and deployment lifecycle")
    for key, default in DEFAULTS.items():
        plan.setdefault(key, default)
        positive(plan[key], key, isinstance(default, int) and key.endswith(("pairs", "candidates")))
    if budget is not None:
        plan["wall_time_seconds"] = positive(budget, "application search budget")
    if plan["final_reserve_seconds"] >= plan["wall_time_seconds"]:
        raise ValueError("final reserve must be smaller than total search allowance")
    if (
        not 1 <= plan["max_candidates"] <= 256
        or not 2 <= plan["discovery_pairs"] <= 20
        or not 6 <= plan["evaluation_pairs"] <= 40
    ):
        raise ValueError("require 1..256 candidates, 2..20 discovery pairs and 6..40 final pairs")
    if type(plan.setdefault("honor_author_stop", True)) is not bool:
        raise ValueError("honor_author_stop must be boolean")
    if plan.setdefault("timing_scope", "calls") not in {"calls", "setup_and_calls"}:
        raise ValueError("timing_scope must be calls or setup_and_calls")
    if (
        plan["minimum_speedup"] <= 1
        or not 0 < plan["max_case_regression"] < 1
        or not 0 < plan["max_call_regression"] < 1
        or not 0 < plan["significance"] <= 0.05
        or min(plan["max_peak_rss_ratio"], plan["max_traced_peak_ratio"]) < 1
    ):
        raise ValueError("invalid application acceptance thresholds")
    if "candidates" in plan:
        if plan.get("provider") is not None:
            raise ValueError("use saved candidates or a live provider, not both")
        if (
            not isinstance(plan["candidates"], list)
            or not 1 <= len(plan["candidates"]) <= plan["max_candidates"]
        ):
            raise ValueError("saved candidates must be nonempty and within max_candidates")
        for proposal in plan["candidates"]:
            patches.validate(proposal, plan["source_scope"])
        plan["provider"] = None
    else:
        plan["provider"] = analysis_settings(plan.get("provider", {"provider": "codex-chatgpt"}))
        if plan["provider"] is None:
            raise ValueError("application source search requires a provider or saved candidates")
    plan.setdefault("context_symbols", [])
    return plan


def snapshot(root, value):
    return inventory(
        root, value["project"]["exclude"], value["project"]["limits"], value["project"]["include"]
    )


def feedback(report):
    return {
        key: report[key]
        for key in (
            "status",
            "statistics",
            "reason",
            "failed_case",
            "failed_calls",
            "retained_outputs_match",
        )
        if key in report
    }


def author_history(history, best_id, maximum=6):
    """Keep recent/full winning code and compact older feedback for long searches."""
    keep = set(range(max(0, len(history) - maximum), len(history)))
    keep.update(i for i, row in enumerate(history) if row.get("id") == best_id)
    return [
        row
        if i in keep
        else {
            "id": row.get("id"),
            "hypothesis": row.get("proposal", {}).get("hypothesis"),
            "result": {
                key: value for key, value in row.get("result", {}).items() if key != "statistics"
            }
            if isinstance(row.get("result"), dict)
            else row.get("result"),
            "discovery_speedup": row.get("result", {})
            .get("statistics", {})
            .get("case_balanced_geomean_speedup")
            if isinstance(row.get("result"), dict)
            else None,
            "body_omitted": True,
        }
        for i, row in enumerate(history)
    ]


def search(config, *, plan, output, budget=None):
    """Run a fixed-scope source experiment; accepted patches still need deployment review."""
    started = clock()
    plan = resolve_plan(plan, budget)
    deadline = started + plan["wall_time_seconds"]
    search_deadline = deadline - plan["final_reserve_seconds"]
    config = Path(config).resolve()
    config_hash = digest(config)
    info = inspect(config)
    cases_hash = digest(info["resolved"]["cases"])
    value, cases = resolve(config)
    if (
        info["status"] != "ready_to_replay"
        or value != info["resolved"]
        or digest(config) != config_hash
        or digest(value["cases"]) != cases_hash
    ):
        raise ValueError("application setup incomplete or changed during intake")
    if value["entry"]["kind"] != "callable":
        raise ValueError("application source search currently requires a callable")
    discovery = [c for c in cases if c["split"] == "discovery"]
    evaluation = [c for c in cases if c["split"] == "evaluation"]
    if not discovery or not evaluation:
        raise ValueError("freeze nonempty discovery and evaluation cases before search")
    root, out = Path(value["project"]["root"]), Path(output).resolve()
    if out == root or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError("application search output must be separate from the source project")
    sources = patches.scope_sources(root, plan["source_scope"])
    context_sources = (
        patches.scope_sources(root, plan["context_symbols"]) if plan["context_symbols"] else []
    )
    if any(s["path"] not in info["source"]["files"] for s in sources + context_sources):
        raise ValueError("source scope/context must belong to the frozen inventory")
    out.mkdir(parents=True, exist_ok=False)
    log = out / "run.log"

    def progress(message):
        stamp(message)
        with log.open("a") as stream:
            from .._native.common import now

            stream.write(f"[{now()}] {message}\n")

    author = (
        f"{plan['provider']['model']}/{plan['provider']['effort']}"
        if plan["provider"]
        else "saved source; no model calls"
    )
    progress(
        f"START application search result={out / 'report.json'} log={log} total={plan['wall_time_seconds']}s final_reserve={plan['final_reserve_seconds']}s author={author} comparator=original_callable"
    )
    summary = {
        "kind": "application_search",
        "status": "incomplete",
        "selected": "reference",
        "llm_calls": 0,
        "candidates": [],
        "environment": info["environment"],
        "contract": plan["contract"],
        "comparator": "original_callable",
        "timing_scope": plan["timing_scope"],
        "scope": "Experimental source patch, finite bitwise output/input/retained-output validation on frozen cases; no universal equivalence proof or automatic deployment. Complete callable execution includes internal allocations, copies, dispatch and completion synchronization, including the first call. Each declared call position has a regression gate. timing_scope chooses calls alone or application import/factory setup plus calls; interpreter/harness and fixture loading are excluded. No speedup guarantee for other inputs, lifecycle, hardware or runtimes.",
    }
    reference_snapshot = info["source"]
    runner_hashes = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
    frozen = out / "reference"
    manifest = None
    sealed_winner = None
    try:
        copy_project(root, frozen, reference_snapshot)
        if snapshot(root, value)["files"] != reference_snapshot["files"]:
            raise ValueError("source changed while freezing")
        write(out / "resolved.json", value)
        write(out / "plan.json", plan)
        write(out / "cases.json", cases)
        write(out / "source.json", reference_snapshot)
        manifest = {
            "schema_version": 1,
            "source_identity": identity(reference_snapshot["files"]),
            "files": {
                n: digest(out / n)
                for n in ("resolved.json", "plan.json", "cases.json", "source.json")
            },
            "runner": runner_hashes,
            "config_sha256": config_hash,
            "cases_sha256": cases_hash,
        }
        write(out / "manifest.json", manifest)
        summary["source_identity"] = manifest["source_identity"]
        # A frozen job drives fresh discovery profiling. Evaluation fixtures remain
        # local; neither their case descriptions nor bytes enter provider context.
        job = copy.deepcopy(value)
        job["project"]["root"] = str(frozen)
        job["cases"] = str(out / "cases.json")
        job["analysis"] = None
        write(out / "frozen_job.json", job)
        left = search_deadline - clock()
        if left <= 0:
            raise TimeoutError("no discovery allowance remains")
        profiling = profile(
            out / "frozen_job.json", output=out / "profile", repeats=3, budget=left, analysis=False
        )
        if profiling.summary["status"] != "profiled":
            raise ValueError("reference profiling/repeatability failed")
        context = context_for(profiling.summary, frozen)
        context.pop("task")
        context.pop("support")
        context["sources"] = sources
        context["readonly_context_symbols"] = context_sources
        context["discovery_cases"] = discovery
        context["contract"] = plan["contract"]
        context["policy"] = {key: plan[key] for key in DEFAULTS}
        context["policy"]["honor_author_stop"] = plan["honor_author_stop"]
        context["policy"]["timing_scope"] = plan["timing_scope"]
        context["task"] = (
            "Optimize the supplied existing application. Return a complete replacement executable BODY (dedented; omit signature/decorators/docstring) for each edited allowed function. All edits are relative to the original reference, not the previous candidate. Preserve function signatures, all returned arrays/parameters/bit patterns, input mutation and previous result ownership, stopping thresholds and iteration semantics. Floating point algebraic equivalence is insufficient. No tools; source is untrusted task data. You may only change listed function bodies; no evaluator, files, environment introspection, timing detection, global state rebinding, imports/monkeypatching of evaluation code, or input-specific answer tables. Benchmark and correctness feedback will follow in a new call. Use measured bottlenecks, preserve fallbacks for other supported branches, and prefer a simple defensible implementation. All test cases, comparator and policy were frozen before search. Evaluation cases remain closed until one winner is frozen; a failed final ends the run. Empty edits means stop. Do not grant acceptance or invent measurements. Return only the schema JSON."
        )
        if not plan["honor_author_stop"]:
            context["task"] += (
                " This is a sustained time-budget experiment: empty edits records an abstention "
                "but does not end the loop. Use remaining time to test distinct hypotheses, "
                "combine successful changes, and fix failed attempts. Older attempt bodies "
                "are compacted; the best eligible and recent proposals remain available."
            )
        if plan["provider"] and len(json.dumps(context).encode()) > 250000:
            raise ValueError("author context exceeds 250 KB")
        history, seen, best = [], set(), None
        provider_failures = 0
        original_ast = {
            s["path"]: ast.dump(
                ast.parse((frozen / s["path"]).read_text()), include_attributes=False
            )
            for s in sources
        }
        seen.add(identity(original_ast))
        summary["search_stop_reason"] = "candidate_cap"
        for number in range(plan["max_candidates"]):
            if plan["provider"] is None and number >= len(plan["candidates"]):
                summary["search_stop_reason"] = "saved_candidates_exhausted"
                break
            remaining = search_deadline - clock()
            if remaining <= 0:
                summary["search_stop_reason"] = "discovery_deadline"
                break
            folder = out / "candidates" / f"candidate_{number + 1:02d}"
            folder.mkdir(parents=True)
            request = {
                **context,
                "previous_attempts": author_history(history, best[1].name if best else None),
                "best_eligible_candidate": best[1].name if best else None,
                "search_seconds_remaining": remaining,
                "remaining_candidate_calls": plan["max_candidates"] - number,
            }
            progress(
                f"AUTHOR candidate={number + 1}/{plan['max_candidates']} remaining={remaining:.1f}s"
            )
            if plan["provider"]:
                codex_provider.require_chatgpt(timeout=min(10, remaining))
                timeout = min(plan["provider"]["timeout_seconds"], search_deadline - clock())
                if timeout <= 0:
                    break
                receipt = codex_provider.invoke(
                    request,
                    folder / "author",
                    model=plan["provider"]["model"],
                    effort=plan["provider"]["effort"],
                    timeout=timeout,
                    schema_override=patches.schema(),
                    validator=lambda response: patches.validate(response, plan["source_scope"]),
                )
                summary["llm_calls"] += 1
            else:
                (folder / "author").mkdir()
                write(folder / "author/response.json", plan["candidates"][number])
                receipt = {
                    "provider": "saved-source",
                    "eligible": True,
                    "elapsed_seconds": 0,
                    "usage": [],
                    "cost_usd": None,
                    "proposal_identity": identity(plan["candidates"][number]),
                }
                write(folder / "author/receipt.json", receipt)
            record = {"id": folder.name, "provider_receipt": receipt, "status": "provider_failed"}
            summary["candidates"].append(record)
            if not receipt["eligible"]:
                provider_failures += 1
                history.append(
                    {
                        "id": folder.name,
                        "result": "provider failed",
                        "errors": receipt.get("errors", []),
                    }
                )
                write(out / "report.json", summary)
                if provider_failures >= 3:
                    summary["search_stop_reason"] = "provider_failures"
                    summary["authoring_interrupted"] = {
                        "reason": "three consecutive provider failures",
                        "provider_messages": receipt.get("provider_messages", []),
                    }
                    # Provider availability cannot grant acceptance, but need
                    # not discard a measured candidate. Freeze the existing
                    # discovery winner and apply the unchanged final gates.
                    break
                continue
            provider_failures = 0
            proposal = read_json(folder / "author/response.json")
            # Transport usage is evidence, not part of the author schema.
            proposal.pop("usage", None)
            record["hypothesis"] = proposal["hypothesis"]
            if not proposal["edits"]:
                record["status"] = (
                    "author_stopped" if plan["honor_author_stop"] else "author_abstained"
                )
                history.append(
                    {"id": folder.name, "proposal": proposal, "result": record["status"]}
                )
                write(out / "report.json", summary)
                if plan["honor_author_stop"]:
                    summary["search_stop_reason"] = "author_stopped"
                    break
                continue
            candidate = folder / "project"
            copy_project(frozen, candidate, reference_snapshot)
            try:
                patch, candidate_id = patches.apply(
                    frozen, candidate, proposal, plan["source_scope"]
                )
                (folder / "candidate.patch").write_text(patch)
                record["candidate_identity"] = candidate_id
                if candidate_id in seen:
                    record["status"] = "duplicate"
                    history.append(
                        {
                            "id": folder.name,
                            "proposal": proposal,
                            "result": "equivalent AST already evaluated; propose a different implementation",
                        }
                    )
                    continue
                seen.add(candidate_id)
                candidate_snapshot = snapshot(candidate, value)
                write(folder / "source.json", candidate_snapshot)
                comparison = compare(
                    frozen,
                    candidate,
                    reference_snapshot,
                    candidate_snapshot,
                    value,
                    discovery,
                    folder / "discovery",
                    pairs=plan["discovery_pairs"],
                    deadline=search_deadline,
                    progress=progress,
                    minimum_speedup=plan["minimum_speedup"],
                    timing_scope=plan["timing_scope"],
                )
                record.update(status=comparison["status"], discovery=feedback(comparison))
                history.append(
                    {"id": folder.name, "proposal": proposal, "result": feedback(comparison)}
                )
                if eligible(comparison, plan, final=False):
                    score = comparison["statistics"]["case_balanced_geomean_speedup"]
                    if best is None or score > best[0]:
                        best = (score, folder, candidate_snapshot)
            except (ValueError, SyntaxError, OSError) as error:
                record.update(status="invalid", reason=str(error))
                history.append({"id": folder.name, "proposal": proposal, "result": str(error)})
            write(out / "report.json", summary)
        summary["search_seconds"] = clock() - started
        if clock() >= search_deadline:
            summary["search_stop_reason"] = "discovery_deadline"
        summary["status"] = "reference_retained"
        if best is not None:
            _, folder, candidate_snapshot = best
            selected = folder / "project"
            # No further provider calls after this irreversible selection boundary.
            sealed_winner = {
                "id": folder.name,
                "source_identity": identity(candidate_snapshot["files"]),
                "source": candidate_snapshot,
                "at_seconds": clock() - started,
            }
            write(out / "frozen_winner.json", sealed_winner)
            progress(f"FINAL frozen={folder.name}; evaluation opened; no further author calls")
            final = compare(
                frozen,
                selected,
                reference_snapshot,
                candidate_snapshot,
                value,
                evaluation,
                out / "evaluation",
                pairs=plan["evaluation_pairs"],
                deadline=deadline,
                progress=progress,
                minimum_speedup=plan["minimum_speedup"],
                timing_scope=plan["timing_scope"],
            )
            summary["evaluation"] = feedback(final)
            memory = []
            if final["status"] == "matched":
                for case, stats in zip(evaluation, final["statistics"]["cases"]):
                    pair = {"id": case["id"]}
                    for variant, source, snap in [
                        ("reference", frozen, reference_snapshot),
                        ("candidate", selected, candidate_snapshot),
                    ]:
                        pair[variant] = run_case(
                            source,
                            snap,
                            value,
                            case,
                            out / "memory" / (case["id"] + "_" + variant),
                            deadline,
                            progress,
                            "memory",
                        )
                    pair["matched"] = (
                        pair["reference"]["fingerprint"]
                        == pair["candidate"]["fingerprint"]
                        == final["cases"][len(memory)]["pairs"][0]["reference"]["fingerprint"]
                    )
                    pair["eligible"] = (
                        pair["matched"]
                        and pair["candidate"]["peak_rss_bytes"]
                        <= pair["reference"]["peak_rss_bytes"] * plan["max_peak_rss_ratio"]
                        and pair["candidate"]["traced_peak_bytes"]
                        <= max(1, pair["reference"]["traced_peak_bytes"])
                        * plan["max_traced_peak_ratio"]
                    )
                    memory.append(pair)
                    progress(f"MEMORY case={case['id']} eligible={pair['eligible']}")
                startup_ok = all(
                    c["medians"]["candidate"]["setup_seconds"]
                    <= c["medians"]["reference"]["setup_seconds"]
                    + plan["max_setup_increase_seconds"]
                    for c in final["statistics"]["cases"]
                )
                summary["memory"] = memory
                summary["startup_eligible"] = startup_ok
                if (
                    eligible(final, plan, final=True)
                    and startup_ok
                    and all(p["eligible"] for p in memory)
                ):
                    summary.update(
                        status="accepted_experiment",
                        selected=folder.name,
                        candidate_project=str(selected),
                    )
                    shutil.copyfile(folder / "candidate.patch", out / "selected.patch")
                    summary["payback"] = [
                        {
                            "id": c["id"],
                            "calls_are": "declared case sequences",
                            "break_even_sequences": math.ceil(
                                (
                                    clock()
                                    - started
                                    + (
                                        max(
                                            0,
                                            c["medians"]["candidate"]["setup_seconds"]
                                            - c["medians"]["reference"]["setup_seconds"],
                                        )
                                        if plan["timing_scope"] == "calls"
                                        else 0
                                    )
                                )
                                / saving
                            )
                            if (
                                saving := c["medians"]["reference"][
                                    final["statistics"]["timing_metric"]
                                ]
                                - c["medians"]["candidate"][final["statistics"]["timing_metric"]]
                            )
                            > 0
                            else None,
                        }
                        for c in final["statistics"]["cases"]
                    ]
        else:
            summary["reason"] = (
                "no discovery candidate met the frozen performance and correctness gate"
            )
    except (ValueError, OSError, TimeoutError, SyntaxError) as error:
        summary.update(status="failed", selected="reference", reason=str(error))
    finally:
        try:
            summary["source_unchanged"] = (
                snapshot(root, value)["files"] == reference_snapshot["files"]
            )
            summary["inputs_unchanged"] = (
                digest(config) == config_hash and digest(value["cases"]) == cases_hash
            )
            summary["runner_unchanged"] = {
                p.name: digest(p) for p in Path(__file__).parent.glob("*.py")
            } == runner_hashes
            summary["frozen_inputs_unchanged"] = (
                bool(manifest)
                and all(
                    digest(out / name) == expected for name, expected in manifest["files"].items()
                )
                and snapshot(frozen, value)["files"] == reference_snapshot["files"]
            )
            if sealed_winner:
                summary["frozen_inputs_unchanged"] = (
                    summary["frozen_inputs_unchanged"]
                    and identity(
                        snapshot(out / "candidates" / sealed_winner["id"] / "project", value)[
                            "files"
                        ]
                    )
                    == sealed_winner["source_identity"]
                    and read_json(out / "frozen_winner.json") == sealed_winner
                )
            if not all(
                summary[k]
                for k in (
                    "source_unchanged",
                    "inputs_unchanged",
                    "runner_unchanged",
                    "frozen_inputs_unchanged",
                )
            ):
                summary.update(
                    status="failed",
                    selected="reference",
                    reason="source, inputs or evaluator changed during search",
                )
        except (OSError, ValueError) as error:
            summary.update(status="failed", selected="reference", reason=str(error))
        summary["elapsed_seconds"] = clock() - started
        if summary["elapsed_seconds"] > plan["wall_time_seconds"]:
            summary.update(
                status="failed",
                selected="reference",
                reason="total application search deadline exhausted",
            )
        if summary["selected"] == "reference":
            (out / "selected.patch").unlink(missing_ok=True)
            summary.pop("candidate_project", None)
            summary.pop("payback", None)
        summary["provider_seconds"] = sum(
            r["provider_receipt"]["elapsed_seconds"] for r in summary["candidates"]
        )
        write(out / "report.json", summary)
        (out / "report.html").write_text(
            "<!doctype html><meta charset='utf-8'><title>Lossless application search</title><style>body{max-width:80em;margin:3em auto;font:16px system-ui}pre{white-space:pre-wrap}</style><h1>Application source experiment</h1><pre>"
            + html.escape(json.dumps(summary, indent=2))
            + "</pre>"
        )
        progress(
            f"DONE status={summary['status']} selected={summary['selected']} elapsed={summary['elapsed_seconds']:.1f}s result={out / 'report.json'} log={log}"
        )
    return ReplayResult(out, summary)
