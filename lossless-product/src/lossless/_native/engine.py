"""Persistent controller protocol: init -> submit/run -> feedback -> seal -> evaluate."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from .common import now, read, write, sha, stamp, geomean

ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
PACKAGE = Path(__file__).resolve().parent
TERMINAL = {"ok", "incorrect", "compile_failed", "crashed", "timed_out", "reference_failure"}


def initialize(spec_path, run, *, hardware_report=None, hardware_notes=None):
    from .operators import CONTRACTS, baseline_source, validate_comparator
    from . import hardware

    spec = read(spec_path)
    if spec.get("operator") not in CONTRACTS:
        raise ValueError("operator adapter is not registered")
    spec["comparator"] = validate_comparator(
        spec["operator"], spec.get("comparator", "native_baseline")
    )
    # Low-level research specs retain their historical bound-buffer default.
    # Public Workload jobs explicitly freeze ordinary-call timing instead.
    if spec.setdefault("timing_scope", "bound") not in {"call", "bound"}:
        raise ValueError("timing_scope must be call or bound")
    cases = spec["cases"]
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)) or not cases:
        raise ValueError("case IDs must be unique")
    for case in cases:
        if not ID.fullmatch(case["id"]) or case["split"] not in ("discovery", "evaluation"):
            raise ValueError("invalid case identifier/split")
        if not all(type(case[k]) is int and case[k] > 0 for k in ("rows", "columns")):
            raise ValueError("invalid dimensions")
        if case["layout"] not in ("c", "f", "slice2"):
            raise ValueError("unsupported layout")
    shapes = {
        split: {(c["rows"], c["columns"]) for c in cases if c["split"] == split}
        for split in ("discovery", "evaluation")
    }
    if not all(shapes.values()) or shapes["discovery"] & shapes["evaluation"]:
        raise ValueError("discovery/evaluation shapes must be disjoint")
    for field in (
        "budget_seconds",
        "job_timeout_seconds",
        "compile_timeout_seconds",
        "target_ns",
        "max_proposals",
    ):
        if (
            type(spec[field]) not in (int, float)
            or not __import__("math").isfinite(spec[field])
            or spec[field] <= 0
        ):
            raise ValueError(f"invalid {field}")
    for field in ("screening_blocks", "confirmation_blocks"):
        if type(spec[field]) is not int or spec[field] < 3:
            raise ValueError(f"invalid {field}")
    profile = (
        hardware.load_report(hardware_report) if hardware_report else hardware.collect(basic=True)
    )
    notes = (
        hardware.load_notes(hardware_notes)
        if hardware_notes
        else {"schema_version": 1, "entries": []}
    )
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=False)
    for sub in ("harness/lossless/_native", "proposals", "jobs", "build"):
        (run / sub).mkdir(parents=True, exist_ok=True)
    (run / "harness/lossless/__init__.py").write_text("")
    for source in (*PACKAGE.glob("*.py"), *PACKAGE.glob("*.m")):
        shutil.copy2(source, run / "harness/lossless/_native" / source.name)
    write(run / "spec.json", spec)
    write(run / "contract.json", CONTRACTS[spec["operator"]])
    write(run / "hardware_profile.json", profile)
    write(run / "hardware_notes.json", notes)
    (run / "baseline.c").write_text(baseline_source(spec["operator"]))
    pinned = [
        "harness/lossless/__init__.py",
        "spec.json",
        "contract.json",
        "baseline.c",
        "hardware_profile.json",
        "hardware_notes.json",
    ] + [
        str(p.relative_to(run))
        for p in (run / "harness/lossless/_native").iterdir()
        if p.suffix in (".py", ".m")
    ]
    import numpy, platform
    import importlib.metadata

    try:
        scipy_version = importlib.metadata.version("scipy")
    except importlib.metadata.PackageNotFoundError:
        scipy_version = None
    compiler = shutil.which("clang")
    manifest = {
        "created_utc": now(),
        "hashes": {name: sha(run / name) for name in pinned},
        "python": sys.version,
        "executable": sys.executable,
        "platform": sys.platform,
        "machine": platform.machine(),
        "numpy": numpy.__version__,
        "scipy": scipy_version,
        "compiler_version": subprocess.check_output([compiler, "--version"], text=True)
        if compiler
        else None,
        "isolation": "separate process and process group; not an OS security sandbox",
        "llm_controller": "external session; token/cost accounting is supplied per proposal when available",
    }
    write(run / "manifest.json", manifest)
    write(
        run / "state.json",
        {
            "version": 1,
            "created_utc": now(),
            "proposals": {},
            "jobs": {},
            "sealed": None,
            "budget_seconds": spec["budget_seconds"],
            "spent_seconds": 0.0,
            "events": [],
        },
    )
    stamp(
        f"Initialized {run}; discovery cases={len([c for c in cases if c['split'] == 'discovery'])}; evaluation remains closed"
    )
    return run


def doctor():
    import numpy, platform
    import importlib.metadata

    try:
        scipy_version = importlib.metadata.version("scipy")
    except importlib.metadata.PackageNotFoundError:
        scipy_version = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy_version,
        "clang": shutil.which("clang"),
        "supported_operators": ["copy", "softmax", "rmsnorm_residual"],
        "backend": "CPU native C",
        "model_graph_integration": "not implemented",
        "llm_controller": "provider-neutral JSON command adapter or proposal files",
    }


def verify(run):
    manifest = read(run / "manifest.json")
    for name, digest in manifest["hashes"].items():
        if sha(run / name) != digest:
            raise ValueError(f"frozen contract/harness changed: {name}")
    # Commands must use the same controller version that was snapshotted.
    for source in (*PACKAGE.glob("*.py"), *PACKAGE.glob("*.m")):
        relative = "harness/lossless/_native/" + source.name
        if manifest["hashes"].get(relative) != sha(source):
            raise ValueError("controller version changed; use the frozen harness for this run")
    state = read(run / "state.json")
    if state["sealed"] and read(run / "seal.json") != state["sealed"]:
        raise ValueError("sealed decision changed")
    for name, entry in state["proposals"].items():
        for relative, digest in entry["hashes"].items():
            if sha(run / relative) != digest:
                raise ValueError(f"proposal modified after submission: {name}")
    for entry in state["jobs"].values():
        if entry.get("result_sha256") and sha(run / entry["result"]) != entry["result_sha256"]:
            raise ValueError("completed result changed")
        if entry.get("binary_sha256") and sha(run / entry["binary"]) != entry["binary_sha256"]:
            raise ValueError("compiled binary changed")
    return state


@contextmanager
def locked(run):
    run = Path(run).resolve()
    with (run / "controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = verify(run)
        yield run, state


def event(run, state, kind, **values):
    state["events"].append({"utc": now(), "kind": kind, **values})
    write(run / "state.json", state)
    with (run / "run.log").open("a") as log:
        log.write(f"[{now()}] {kind} " + json.dumps(values, sort_keys=True) + "\n")


def submit(run, proposal_path):
    path = Path(proposal_path).resolve()
    proposal = read(path)
    with locked(run) as (run, state):
        spec = read(run / "spec.json")
        name = proposal.get("id", "")
        if state["sealed"]:
            raise ValueError("candidate pool is sealed")
        if (
            proposal.get("schema_version") != 1
            or not ID.fullmatch(name)
            or name in {"baseline", "native_baseline", "deployment_reference"}
        ):
            raise ValueError("invalid proposal schema/id")
        if proposal.get("operator") != spec["operator"] or not proposal.get("hypothesis"):
            raise ValueError("proposal needs matching operator and hypothesis")
        source = (path.parent / proposal["source_file"]).resolve()
        if not source.is_file() or source.stat().st_size > 1000000:
            raise ValueError("missing/oversized source")
        encoded = json.dumps(proposal, sort_keys=True).encode()
        fingerprint = hashlib.sha256(encoded + source.read_bytes()).hexdigest()
        if name in state["proposals"]:
            if state["proposals"][name]["fingerprint"] != fingerprint:
                raise ValueError("candidate ID is immutable")
            return
        if len(state["proposals"]) >= spec["max_proposals"]:
            raise ValueError("proposal budget exhausted")
        folder = run / "proposals" / name
        folder.mkdir()
        shutil.copy2(source, folder / "kernel.c")
        write(folder / "proposal.json", proposal)
        names = [str(p.relative_to(run)) for p in folder.iterdir()]
        source_hash = sha(source)
        duplicate = next(
            (n for n, v in state["proposals"].items() if v.get("source_sha256") == source_hash),
            None,
        )
        state["proposals"][name] = {
            "source_sha256": source_hash,
            "fingerprint": fingerprint,
            "submitted_utc": now(),
            "hashes": {p: sha(run / p) for p in names},
            "rejected": {"reason": "duplicate_source", "of": duplicate} if duplicate else None,
        }
        event(run, state, "proposal_submitted", candidate=name, fingerprint=fingerprint)


def job_specs(run, state, stage):
    spec = read(run / "spec.json")
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    baseline = str(run / "build" / ("baseline" + suffix))
    yield (
        "compile_baseline",
        {
            "kind": "compile",
            "source": str(run / "baseline.c"),
            "binary": baseline,
            "timeout_seconds": spec["compile_timeout_seconds"],
            "candidate_id": "baseline",
        },
    )
    names = (
        sorted(state["proposals"]) if stage == "discovery" else state["sealed"]["valid_candidates"]
    )
    for name in names:
        if state["proposals"][name]["rejected"]:
            continue
        yield (
            "compile_" + name,
            {
                "kind": "compile",
                "source": str(run / "proposals" / name / "kernel.c"),
                "binary": str(run / "build" / (name + suffix)),
                "timeout_seconds": spec["compile_timeout_seconds"],
                "candidate_id": name,
            },
        )
    pairs = [(name, c) for name in names for c in spec["cases"] if c["split"] == stage]
    # Deterministic shuffled order without importing numeric libraries in the controller.
    pairs.sort(
        key=lambda pair: hashlib.sha256(
            f"{stage}:{pair[0]}:{pair[1]['id']}:2026".encode()
        ).hexdigest()
    )
    for name, case in pairs:
        if state["proposals"][name]["rejected"]:
            continue
        compiled = state["jobs"].get("compile_" + name, {})
        if compiled.get("status") != "ok":
            continue
        index = next(i for i, c in enumerate(spec["cases"]) if c["id"] == case["id"])
        yield (
            stage + "_" + name + "_" + case["id"],
            {
                "kind": "benchmark",
                "operator": spec["operator"],
                "comparator": spec["comparator"],
                "timing_scope": spec["timing_scope"],
                "case": case,
                "baseline": baseline,
                "candidate": str(run / "build" / (name + suffix)),
                "candidate_id": name,
                "timeout_seconds": spec["job_timeout_seconds"],
                "seed": 510000 + index * 100,
                "candidate_seed": int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 10000,
                **{k: spec[k] for k in ("screening_blocks", "confirmation_blocks", "target_ns")},
            },
        )


def execute(run, stage, max_jobs=None, deadline=None):
    if stage not in ("discovery", "evaluation"):
        raise ValueError("invalid stage")
    with locked(run) as (run, state):
        if state.get("fatal_error"):
            raise RuntimeError(
                "campaign has a trusted-reference failure; start a corrected campaign"
            )
        if stage == "evaluation" and not state["sealed"]:
            raise ValueError("seal discovery before evaluating")
        if stage == "discovery" and state["sealed"]:
            raise ValueError("discovery is closed")
        if not state["proposals"]:
            raise ValueError("submit a proposal first")
        for entry in state["jobs"].values():
            if entry["status"] == "running":
                # Conservatively charge the reserved timeout after an unclean interruption.
                state["spent_seconds"] += entry.get("reserved_seconds", 0)
                entry["status"] = "pending"
        event(run, state, "run_started", stage=stage)
        count = 0
        for job_id, job in job_specs(run, state, stage):
            previous = state["jobs"].get(job_id, {})
            if previous.get("status") in TERMINAL:
                continue
            if max_jobs is not None and count >= max_jobs:
                break
            remaining = state["budget_seconds"] - state["spent_seconds"]
            if deadline is not None:
                remaining = min(remaining, max(0, deadline - time.monotonic()))
            if remaining < job["timeout_seconds"]:
                event(run, state, "budget_deferred", remaining_seconds=remaining)
                break
            folder = run / "jobs" / job_id
            folder.mkdir(exist_ok=True)
            attempts = previous.get("attempts", 0) + 1
            job_path = folder / f"job_{attempts}.json"
            result_path = folder / f"result_{attempts}.json"
            spec = read(run / "spec.json")
            manifest = read(run / "manifest.json")
            job["cache"] = spec.get("cache", {})
            job["reuse_validation"] = stage == "discovery" and job["cache"].get(
                "reuse_validation", False
            )
            job["cache_context"] = {
                "environment": {
                    k: manifest.get(k)
                    for k in ("python", "platform", "machine", "numpy", "scipy", "compiler_version")
                },
                "hardware": read(run / "hardware_profile.json")["host"],
                "harness": {
                    k: v for k, v in manifest["hashes"].items() if k.startswith("harness/")
                },
                "contract": manifest["hashes"]["contract.json"],
                "comparator": spec["comparator"],
                "timing_scope": spec["timing_scope"],
            }
            write(job_path, job)
            entry = {
                "status": "running",
                "attempts": attempts,
                "stage": stage,
                "kind": job["kind"],
                "candidate": job["candidate_id"],
                "case_id": job.get("case", {}).get("id"),
                "reserved_seconds": job["timeout_seconds"],
            }
            state["jobs"][job_id] = entry
            write(run / "state.json", state)
            env = {
                key: value
                for key, value in os.environ.items()
                if key in ("PATH", "TMPDIR", "SYSTEMROOT", "LANG")
            }
            env.update(
                PYTHONPATH=str(run / "harness"),
                PYTHONUNBUFFERED="1",
                PYTHONDONTWRITEBYTECODE="1",
                VECLIB_MAXIMUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1",
            )
            started = time.monotonic()
            timed_out = False
            interrupted = False
            with (folder / f"worker_{attempts}.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "lossless._native.worker",
                        str(job_path),
                        str(result_path),
                    ],
                    cwd=folder,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    process.wait(timeout=job["timeout_seconds"])
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                    timed_out = isinstance(error, subprocess.TimeoutExpired)
                    interrupted = not timed_out
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            elapsed = time.monotonic() - started
            state["spent_seconds"] += elapsed
            count += 1
            if interrupted:
                entry.update(status="pending", elapsed_seconds=elapsed)
                event(run, state, "interrupted", job=job_id)
                break
            if timed_out or process.returncode == -signal.SIGALRM:
                result = {"status": "timed_out", "returncode": process.returncode}
            elif process.returncode != 0 or not result_path.exists():
                result = {"status": "crashed", "returncode": process.returncode}
            else:
                result = read(result_path)
            if result.get("status") not in TERMINAL:
                result = {"status": "crashed", "reason": "invalid worker result"}
            if not result_path.exists() or result["status"] in ("timed_out", "crashed"):
                write(result_path, result)
            entry.update(
                status=result["status"],
                elapsed_seconds=elapsed,
                result=str(result_path.relative_to(run)),
                result_sha256=sha(result_path),
            )
            if job["kind"] == "compile" and result["status"] == "ok":
                entry.update(
                    binary=str(Path(job["binary"]).relative_to(run)),
                    binary_sha256=result["binary_sha256"],
                )
            if result["status"] != "ok":
                if job["candidate_id"] == "baseline" or result["status"] == "reference_failure":
                    state["fatal_error"] = {"job": job_id, "status": result["status"]}
                    event(run, state, "reference_failure", job=job_id)
                    raise RuntimeError("trusted reference failed; campaign stopped")
                state["proposals"][job["candidate_id"]]["rejected"] = {
                    "job": job_id,
                    "reason": result["status"],
                }
            event(
                run,
                state,
                "job_finished",
                job=job_id,
                status=result["status"],
                elapsed_seconds=elapsed,
            )
            stamp(
                f"{stage} {job_id}: {result['status']}; {elapsed:.2f}s; budget {state['spent_seconds']:.1f}/{state['budget_seconds']:.1f}s"
            )
        event(run, state, "run_stopped", stage=stage, jobs_executed=count)
        return {
            "jobs_executed": count,
            "spent_seconds": state["spent_seconds"],
            "remaining_seconds": state["budget_seconds"] - state["spent_seconds"],
        }


def feedback_data(run, state):
    spec = read(run / "spec.json")
    case_ids = {c["id"] for c in spec["cases"] if c["split"] == "discovery"}
    entries = {}
    for name, proposal in state["proposals"].items():
        jobs = [
            j
            for j in state["jobs"].values()
            if j["stage"] == "discovery" and j["kind"] == "benchmark" and j["candidate"] == name
        ]
        records = [read(run / j["result"]) for j in jobs if j["status"] == "ok"]
        complete = {j["case_id"] for j in jobs if j["status"] == "ok"} == case_ids and not proposal[
            "rejected"
        ]
        entries[name] = {
            "complete": complete,
            "rejected": proposal["rejected"],
            "completed_cases": len(records),
            "geomean_speedup_vs_comparator": geomean([r["speedup_vs_comparator"] for r in records])
            if complete
            else None,
            "geomean_speedup_vs_native": geomean([r["speedup_vs_native"] for r in records])
            if complete
            else None,
            "worst_speedup_vs_native": min([r["speedup_vs_native"] for r in records], default=None),
            "confirmed_regressions": sum(r["confirmed_regression"] for r in records),
            "cases": [
                {
                    k: r[k]
                    for k in (
                        "case",
                        "speedup_vs_native",
                        "speedup_vs_library",
                        "ci95_vs_native",
                        "comparator",
                        "speedup_vs_comparator",
                        "ci95_vs_comparator",
                    )
                }
                for r in records
            ],
        }
    return {
        "stage": "discovery",
        "candidates": entries,
        "spent_seconds": state["spent_seconds"],
        "remaining_seconds": state["budget_seconds"] - state["spent_seconds"],
        "next_action": "propose_more_or_seal",
        "evaluation_visible": False,
    }


def feedback(run):
    with locked(run) as (run, state):
        result = feedback_data(run, state)
        write(run / "feedback.json", result)
        return result


def extend_budget(run, seconds, reason):
    if seconds <= 0 or not reason.strip():
        raise ValueError("positive seconds and an explicit reason are required")
    with locked(run) as (run, state):
        state["budget_seconds"] += seconds
        event(run, state, "budget_extended", seconds=seconds, reason=reason)
        return {"budget_seconds": state["budget_seconds"], "spent_seconds": state["spent_seconds"]}


def seal(run):
    with locked(run) as (run, state):
        if state.get("fatal_error"):
            raise RuntimeError("cannot seal a campaign with a reference failure")
        if state["sealed"]:
            return state["sealed"]
        data = feedback_data(run, state)
        if any(not v["complete"] and not v["rejected"] for v in data["candidates"].values()):
            raise ValueError("discovery jobs remain incomplete")
        valid = sorted(n for n, v in data["candidates"].items() if v["complete"])
        if not valid:
            raise ValueError("no validated proposals")
        winner = max(
            valid, key=lambda n: (data["candidates"][n]["geomean_speedup_vs_comparator"], n)
        )
        # A frozen global choice is an evaluation target, not a learned runtime selector.
        if data["candidates"][winner]["geomean_speedup_vs_comparator"] <= read(
            run / "spec.json"
        ).get("min_speedup", 1.02):
            winner = (
                "native_baseline"
                if read(run / "spec.json")["comparator"] == "native_baseline"
                else "deployment_reference"
            )
        state["sealed"] = {
            "utc": now(),
            "valid_candidates": valid,
            "global_choice": winner,
            "proposal_fingerprints": {
                n: state["proposals"][n]["fingerprint"] for n in sorted(state["proposals"])
            },
            "comparator": read(run / "spec.json")["comparator"],
            "selection_rule": f"best discovery confirmation geometric mean against frozen comparator; retain reference unless >{read(run / 'spec.json').get('min_speedup', 1.02)}x",
        }
        write(run / "seal.json", state["sealed"])
        event(run, state, "pool_sealed", global_choice=winner)
        return state["sealed"]


def report(run):
    with locked(run) as (run, state):
        if state.get("fatal_error"):
            raise RuntimeError("cannot report a successful campaign with a reference failure")
        if not state["sealed"]:
            raise ValueError("evaluation is not sealed")
        spec = read(run / "spec.json")
        evaluation = [c for c in spec["cases"] if c["split"] == "evaluation"]
        valid = state["sealed"]["valid_candidates"]
        for name in valid:
            for case in evaluation:
                job = state["jobs"].get("evaluation_" + name + "_" + case["id"])
                if not job and not state["proposals"][name]["rejected"]:
                    raise ValueError("evaluation jobs remain incomplete")
                if job and job["status"] not in TERMINAL:
                    raise ValueError("evaluation jobs remain incomplete")
        rows = []
        by_candidate = {}
        checks = 0
        failures = 0
        for job_id, job in state["jobs"].items():
            if job["kind"] != "benchmark" or "result" not in job:
                continue
            r = read(run / job["result"])
            validation = r.get("validation", {})
            checks += sum(len(values) for values in validation.values())
            failures += sum(
                not value["passed"] for values in validation.values() for value in values.values()
            )
            rows.append(
                {
                    "schema_version": 1,
                    "operator": spec["operator"],
                    "stage": job["stage"],
                    "candidate": job["candidate"],
                    "case": next(c for c in spec["cases"] if c["id"] == job["case_id"]),
                    "data_role": "discovery_training"
                    if job["stage"] == "discovery"
                    else "sealed_evaluation",
                    "status": job["status"],
                    "raw_record_file": job["result"],
                    "proposal_fingerprint": state["proposals"][job["candidate"]]["fingerprint"],
                    **{
                        k: r[k]
                        for k in (
                            "comparator",
                            "speedup_vs_comparator",
                            "ci95_vs_comparator",
                            "speedup_vs_native",
                            "speedup_vs_library",
                            "screening",
                            "confirmation",
                            "validation",
                        )
                        if k in r
                    },
                }
            )
        for name in valid:
            records = [
                read(run / j["result"])
                for j in state["jobs"].values()
                if j["kind"] == "benchmark"
                and j["stage"] == "evaluation"
                and j["candidate"] == name
                and j["status"] == "ok"
            ]
            complete = len(records) == len(evaluation) and not state["proposals"][name]["rejected"]
            by_candidate[name] = {
                "complete": complete,
                "cases": len(records),
                "geomean_vs_comparator": geomean([r["speedup_vs_comparator"] for r in records])
                if complete
                else None,
                "geomean_vs_native": geomean([r["speedup_vs_native"] for r in records])
                if complete
                else None,
                "geomean_vs_library": geomean([r["speedup_vs_library"] for r in records])
                if complete
                else None,
                "confirmed_wins": sum(r["confirmed_win"] for r in records),
                "confirmed_regressions": sum(r["confirmed_regression"] for r in records),
            }
        choice = state["sealed"]["global_choice"]
        if choice in {"native_baseline", "deployment_reference"}:
            native_records = []
            for case in evaluation:
                matches = [
                    j
                    for j in state["jobs"].values()
                    if j["kind"] == "benchmark"
                    and j["stage"] == "evaluation"
                    and j["case_id"] == case["id"]
                    and j["status"] == "ok"
                ]
                if matches:
                    j = min(matches, key=lambda j: j["candidate"])
                    native_records.append(read(run / j["result"]))
            complete = len(native_records) == len(evaluation)
            chosen = {
                "complete": complete,
                "cases": len(native_records),
                "geomean_vs_comparator": 1.0 if complete else None,
                "geomean_vs_native": geomean(
                    [
                        r["confirmation"]["median_us"]["native_baseline"]
                        / r["confirmation"]["median_us"][spec["comparator"]]
                        for r in native_records
                    ]
                )
                if complete
                else None,
                "geomean_vs_library": geomean(
                    [
                        r["confirmation"]["median_us"][r["library_choice"]]
                        / r["confirmation"]["median_us"][spec["comparator"]]
                        for r in native_records
                    ]
                )
                if complete
                else None,
            }
        else:
            chosen = by_candidate[choice]
        summary = {
            "operator": spec["operator"],
            "workloads": len(spec["cases"]),
            "discovery_cases": len(spec["cases"]) - len(evaluation),
            "evaluation_cases": len(evaluation),
            "proposals": len(state["proposals"]),
            "validated_before_evaluation": len(valid),
            "comparator": spec["comparator"],
            "frozen_global_choice": choice,
            "timing_scope": spec["timing_scope"],
            "frozen_choice_evaluation": chosen,
            "evaluation_by_candidate": by_candidate,
            "numerical_checks_including_repeated_baselines": checks,
            "numerical_failures": failures,
            "candidate_case_records": len(rows),
            "rejected": {n: v["rejected"] for n, v in state["proposals"].items() if v["rejected"]},
            "worker_wall_seconds": state["spent_seconds"],
            "allocated_worker_budget_seconds": state["budget_seconds"],
            "llm_token_cost": "not available from this session; not included in worker budget",
        }
        write(run / "summary.json", summary)
        with (run / "dataset.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        lines = [
            "# Kernel Lab campaign report",
            "",
            f"Operator: **{spec['operator']}**. Workloads: {len(spec['cases'])}; sealed evaluation: {len(evaluation)}.",
            f"Frozen global choice: `{choice}`. Candidate pool and choice were frozen before evaluation measurements.",
            f"Frozen deployment comparator: `{spec['comparator']}`. Acceptance is measured against this implementation.",
            "",
            "| Candidate | Evaluation/comparator | Evaluation/native | Evaluation/library envelope | Confirmed wins | Confirmed regressions |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, item in by_candidate.items():
            lines.append(
                f"| {name} | {item['geomean_vs_comparator'] if item['complete'] else 'incomplete'} | {item['geomean_vs_native'] if item['complete'] else 'incomplete'} | {item['geomean_vs_library'] if item['complete'] else 'incomplete'} | {item['confirmed_wins']} | {item['confirmed_regressions']} |"
            )
        lines += [
            "",
            f"Numerical checks (including repeated baselines): {checks}; failures: {failures}. Worker wall time: {state['spent_seconds']:.2f}s.",
            "",
            "Each candidate was measured in its own process with interleaved native/library baselines. Comparison ratios use confirmation data; the library envelope uses screening choices. Only the precommitted global choice is the primary evaluation result; other candidates are diagnostic.",
            "",
            f"Timing scope: {spec['timing_scope']}. Ordinary call timing includes deployment guards, allocation, binding and dispatch; bound timing excludes binding and allocation. Compilation, artifact loading and fixture generation are excluded. Kernel-only diagnostics never override the frozen acceptance boundary.",
            "Numerical checks are finite, not proofs. Bootstrap intervals describe within-job noise and have no multiple-comparison correction. CPU only. Worker budget excludes LLM/controller development time and token cost. Process isolation contains crashes but is not a security sandbox.",
        ]
        (run / "REPORT.md").write_text("\n".join(lines) + "\n")
        event(run, state, "report_written")
        return summary
