"""User workflow over the extracted controller, with a bounded total search."""

from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import time

from .jobs import Workload, ADAPTERS, identity, read_json
from ._native.common import write, stamp


@dataclass
class Result:
    directory: Path
    summary: dict

    def export(self, destination):
        from .artifacts import export

        return export(self.directory, destination)


def save_report(run, summary):
    write(run / "report.json", summary)
    lines = [
        "Lossless optimization report",
        f"Outcome: {summary['status']}",
        f"Selected: {summary['selected']}",
        "Correctness evidence: finite validation; scoped proofs, if present, are separate.",
        f"Elapsed search time: {summary['wall_time_seconds']:.3f} seconds",
    ]
    (run / "report.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>Lossless report</title><style>body{max-width:70em;margin:3em auto;font:16px system-ui;padding:0 1em}pre{white-space:pre-wrap}</style><h1>Lossless optimization report</h1><pre>"
        + html.escape("\n".join(lines) + "\n\n" + json.dumps(summary, indent=2, allow_nan=False))
        + "</pre></html>\n"
    )


def optimize(workload, *, budget=None, llm=None, output=None, resume=False):
    if not isinstance(workload, Workload):
        workload = Workload.from_config(workload)
    resolved = workload.resolve(budget)
    if resolved["workload"]["adapter"] == "mlx.fixed_count":
        from .adapters.mlx.job import optimize as mlx_optimize

        return mlx_optimize(workload, resolved, llm=llm, output=output, resume=resume)
    if __import__("sys").platform not in {"darwin", "linux"}:
        raise ValueError("native alpha supports macOS and Linux")
    from ._native import engine, providers as protocol, hardware
    from . import providers, recipes
    import shutil

    if not shutil.which("clang"):
        raise ValueError("clang is required for native candidates; run lossless doctor")
    started = time.monotonic()
    run = (
        Path(output).resolve()
        if output
        else Path(resolved["output"]["directory"])
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    limit = resolved["budget"]["wall_time_seconds"]
    job_timeout = min(resolved["budget"].get("job_timeout_seconds", 8), limit)
    operator = ADAPTERS[resolved["workload"]["adapter"]]
    previous_spent = 0
    if resume:
        if not output:
            raise ValueError("resume requires the existing run directory via output")
        frozen = read_json(run / "resolved.json")
        if frozen != resolved:
            raise ValueError("resume configuration differs from the frozen job")
        if (run / "report.json").exists():
            engine.verify(run)
            return Result(run, read_json(run / "report.json"))
        previous_spent = read_json(run / "progress.json")["wall_time_seconds"]
        # An interrupted call may have consumed its whole reserved remaining budget.
        if read_json(run / "progress.json").get("running"):
            previous_spent = limit
        state = engine.verify(run)
    else:
        if run.exists():
            raise ValueError(
                f"output directory already exists: {run}; use resume for the same frozen job"
            )
        spec = {
            "operator": operator,
            "cases": resolved["workload"]["cases"],
            "budget_seconds": limit,
            "job_timeout_seconds": job_timeout,
            "compile_timeout_seconds": min(8, limit),
            "target_ns": 300_000,
            "screening_blocks": 5,
            "confirmation_blocks": 9,
            "max_proposals": resolved["budget"]["max_candidates"],
            "min_speedup": resolved["objective"]["min_speedup"],
        }
        import tempfile

        with tempfile.TemporaryDirectory(prefix="lossless-job-") as folder:
            path = Path(folder) / "spec.json"
            write(path, spec)
            profile = Path(folder) / "hardware.json"
            write(profile, hardware.collect_bounded(timeout=min(5, limit * 0.1)))
            engine.initialize(path, run, hardware_report=profile)
        write(run / "resolved.json", resolved)
        manifest = read_json(run / "manifest.json")
        from ._native.common import sha

        manifest["hashes"]["resolved.json"] = sha(run / "resolved.json")
        write(run / "manifest.json", manifest)
        detected = read_json(run / "hardware_profile.json")
        supplied = resolved["target"].get("supplied_profile")
        if supplied:
            for key in ("os", "machine"):
                requested = supplied.get("host", {}).get(key)
                if requested and requested != detected["host"][key]:
                    raise ValueError(f"supplied hardware profile mismatches actual worker {key}")
        write(run / "hardware.json", {"detected": detected, "supplied_context": supplied})
        write(run / "progress.json", {"wall_time_seconds": 0, "running": False, "llm_calls": 0})
        state = engine.verify(run)
    deadline = started + max(0, limit - previous_spent)
    stamp(
        f"START run={run} log={run / 'run.log'} budget={limit}s; trusted local candidate execution"
    )
    proof_record = None
    if resolved["proofs"]["check"] and not (run / "proofs.json").exists():
        from .proofs import check

        proof_record = check(timeout=max(0, deadline - time.monotonic()))
        write(run / "proofs.json", proof_record)
    elif (run / "proofs.json").exists():
        proof_record = read_json(run / "proofs.json")
    progress = read_json(run / "progress.json")
    calls = progress.get("llm_calls", 0)
    errors = []

    def checkpoint(running=False):
        write(
            run / "progress.json",
            {
                "wall_time_seconds": previous_spent + time.monotonic() - started,
                "running": running,
                "llm_calls": calls,
            },
        )

    def submit(proposal):
        folder = run / "submissions" / proposal["id"]
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "kernel.c").write_text(proposal["source"])
        write(
            folder / "proposal.json",
            {
                "schema_version": 1,
                "id": proposal["id"],
                "operator": operator,
                "source_file": "kernel.c",
                "hypothesis": proposal["hypothesis"],
                "parameters": proposal.get("parameters", {}),
            },
        )
        engine.submit(run, folder / "proposal.json")

    if not state["sealed"]:
        if not state["proposals"]:
            for proposal in recipes.native_candidates(operator):
                if len(engine.verify(run)["proposals"]) >= resolved["budget"]["max_candidates"]:
                    break
                submit(proposal)
        # Reserve enough time to evaluate the one frozen choice on fresh cases.
        reserve = min(
            limit * 0.4,
            max(
                5,
                job_timeout
                * len([c for c in resolved["workload"]["cases"] if c["split"] == "evaluation"]),
            ),
        )
        checkpoint(True)
        engine.execute(run, "discovery", deadline=deadline - reserve)
        checkpoint()
        while (
            (llm is not None or resolved.get("llm"))
            and calls < resolved["budget"]["max_llm_calls"]
            and time.monotonic() + reserve < deadline
        ):
            current = engine.verify(run)
            slots = resolved["budget"]["max_candidates"] - len(current["proposals"])
            if slots <= 0:
                break
            context = protocol.request_context(run)
            context["max_proposals"] = slots
            context["supplied_hardware_context"] = resolved["target"].get("supplied_profile")
            from .proofs import search

            context["formal_context"] = search(
                "copy index layout" if operator == "copy" else operator, limit=5
            )
            calls += 1
            checkpoint(True)
            call_dir = run / "model_calls" / f"call_{calls:03d}"
            call_dir.mkdir(parents=True, exist_ok=False)
            write(call_dir / "request.json", context)
            try:
                response, seconds = providers.call(
                    context,
                    resolved.get("llm"),
                    callback=llm,
                    base=workload.base,
                    timeout=min(
                        (resolved.get("llm") or {}).get("timeout_seconds", 30),
                        max(0, deadline - reserve - time.monotonic()),
                    ),
                )
                proposals, usage = protocol.validate_response(
                    response, context, current["proposals"]
                )
                write(call_dir / "response.json", response)
                for proposal in proposals:
                    submit(proposal)
                write(
                    call_dir / "receipt.json",
                    {
                        "status": "accepted_for_evaluation",
                        "elapsed_seconds": seconds,
                        "usage": usage,
                    },
                )
            except (ValueError, TimeoutError, EOFError, OSError) as error:
                errors.append({"call": calls, "error_type": type(error).__name__})
                write(
                    call_dir / "receipt.json",
                    {"status": "failed", "error_type": type(error).__name__},
                )
            checkpoint()
            checkpoint(True)
            engine.execute(run, "discovery", deadline=deadline - reserve)
            checkpoint()
        try:
            engine.seal(run)
        except ValueError as error:
            errors.append({"stage": "selection", "reason": str(error)})
    state = engine.verify(run)
    detail = None
    if state["sealed"]:
        checkpoint(True)
        engine.execute(run, "evaluation", deadline=deadline)
        checkpoint()
        try:
            detail = engine.report(run)
        except ValueError as error:
            errors.append({"stage": "evaluation", "reason": str(error)})
    selected = "reference"
    if detail:
        choice = detail["frozen_global_choice"]
        chosen = detail["frozen_choice_evaluation"]
        if choice != "native_baseline" and chosen["complete"]:
            entries = [
                read_json(run / j["result"])
                for j in engine.verify(run)["jobs"].values()
                if j["stage"] == "evaluation"
                and j["kind"] == "benchmark"
                and j["candidate"] == choice
                and j["status"] == "ok"
            ]
            if (
                chosen["geomean_vs_native"] >= resolved["objective"]["min_speedup"]
                and entries
                and all(r["ci95_vs_native"][0] > 1 for r in entries)
            ):
                selected = choice
    checkpoint()
    status = "accepted" if selected != "reference" else "reference_retained"
    if detail is None:
        status = "incomplete"
    summary = {
        "schema_version": 1,
        "adapter": resolved["workload"]["adapter"],
        "status": status,
        "selected": selected,
        "contract": read_json(run / "contract.json"),
        "required_evidence": "validated",
        "job_identity": identity(resolved),
        "wall_time_seconds": previous_spent + time.monotonic() - started,
        "llm_calls": calls,
        "errors": errors,
        "proofs": proof_record,
        "measurements": detail,
        "scope": "Finite validation on generated cases; exact copy or explicit numerical contract. Native timings include ctypes dispatch with preallocated buffers; Python deployment allocation cost is separate. POSIX trusted-local execution, not a security sandbox.",
    }
    save_report(run, summary)
    from ._native.common import sha

    manifest = read_json(run / "manifest.json")
    for name in ("report.json", "hardware.json", "proofs.json"):
        if (run / name).exists():
            manifest["hashes"][name] = sha(run / name)
    write(run / "manifest.json", manifest)
    stamp(
        f"COMPLETE outcome={status} selected={selected} result={run / 'report.json'} log={run / 'run.log'}"
    )
    return Result(run, summary)
