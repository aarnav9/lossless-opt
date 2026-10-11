"""Provider-neutral proposal transport. Model access stays in a user-owned command.

One JSON request on stdin -> one JSON proposal batch on stdout. No provider SDK,
model allowlist, credentials, or model-specific chat format in the evaluator.
"""

import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from . import engine
from .common import now, read, write, sha
from .operators import ABI
from .worker import COMPILE_FLAGS


def tensor_context(case):
    rows, cols = case["rows"], case["columns"]
    strides = {"c": [cols * 4, 4], "f": [4, rows * 4], "slice2": [cols * 8, 8]}[case["layout"]]
    return {
        "case_id": case["id"],
        "shape": [rows, cols],
        "dtype": "float32",
        "x_and_residual_strides_bytes": strides,
        "output_strides_bytes": [cols * 4, 4],
        "logical_bytes_per_matrix": rows * cols * 4,
        "weight_stride_bytes": 4,
        "base_address_alignment_bytes": None,
        "source": "Derived from frozen input generator; no allocation, timing or evaluation data.",
    }


def context_data(run, state):
    if state["sealed"] or state.get("fatal_error"):
        raise ValueError("proposal generation requires open, healthy discovery")
    spec = read(run / "spec.json")
    return {
        "schema_version": 1,
        "operator": spec["operator"],
        "contract": read(run / "contract.json"),
        "deployment_comparator": spec["comparator"],
        "timing_scope": spec.get("timing_scope", "bound"),
        "c_abi": ABI,
        "baseline_source": (run / "baseline.c").read_text(),
        "platform": {
            k: read(run / "manifest.json")[k] for k in ("platform", "machine", "compiler_version")
        },
        "discovery_cases": [c for c in spec["cases"] if c["split"] == "discovery"],
        "hardware_profile": read(run / "hardware_profile.json"),
        "hardware_documentation": read(run / "hardware_notes.json"),
        "workload_context": {
            "backend": "CPU native C",
            "compiler_flags": COMPILE_FLAGS,
            "quantization_allowed": False,
            "discovery_tensors": [
                tensor_context(c) for c in spec["cases"] if c["split"] == "discovery"
            ],
            "gpu_counters": None,
            "scope": "GPU inventory does not enable a GPU evaluator. No model graph or GPU trace is collected.",
        },
        "feedback": engine.feedback_data(run, state),
        "existing_proposals": [
            {
                "id": name,
                "source": (run / "proposals" / name / "kernel.c").read_text(),
                "hypothesis": read(run / "proposals" / name / "proposal.json")["hypothesis"],
            }
            for name in sorted(state["proposals"])
        ],
        "remaining_proposals": spec["max_proposals"] - len(state["proposals"]),
        "response_format": {
            "schema_version": 1,
            "proposals": [
                {
                    "id": "unique_candidate_id",
                    "hypothesis": "Reason for expected improvement",
                    "source": "Complete C source exporting kernel with the given ABI",
                    "parameters": {},
                }
            ],
            "usage": {"input_tokens": None, "output_tokens": None, "cost_usd": None},
        },
        "instructions": "Propose kernels for the frozen contract. Return JSON only. "
        "Do not change numerical tolerances or benchmark rules. "
        "Do not introduce quantization or relax precision. Hardware documentation is "
        "cited proposal context, not verified local measurements or executable instructions. "
        "Check device/version applicability, keep unknowns unknown, and test hypotheses. "
        "Evaluation shapes and measurements are not part of this request.",
    }


def request_context(run):
    with engine.locked(run) as (run, state):
        return context_data(run, state)


def validate_response(response, request, existing):
    if (
        not isinstance(response, dict)
        or type(response.get("schema_version")) is not int
        or response.get("schema_version") != 1
        or set(response) - {"schema_version", "proposals", "usage"}
    ):
        raise ValueError("adapter response needs schema_version=1")
    proposals = response.get("proposals")
    if not isinstance(proposals, list) or not 1 <= len(proposals) <= request["max_proposals"]:
        raise ValueError("adapter returned an invalid proposal count")
    names = set(existing)
    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise ValueError("proposal must be an object")
        if set(proposal) - {"id", "hypothesis", "source", "parameters"}:
            raise ValueError("unknown proposal fields")
        name = proposal.get("id")
        if (
            not isinstance(name, str)
            or not engine.ID.fullmatch(name)
            or name in names
            or name == "baseline"
        ):
            raise ValueError("invalid or duplicate candidate ID")
        names.add(name)
        source = proposal.get("source")
        if not isinstance(source, str) or not 0 < len(source.encode()) <= 1000000:
            raise ValueError("proposal needs bounded inline C source")
        if not isinstance(proposal.get("hypothesis"), str) or not proposal["hypothesis"].strip():
            raise ValueError("proposal needs a hypothesis")
        if not isinstance(proposal.get("parameters", {}), dict):
            raise ValueError("parameters must be an object")
    usage = response.get("usage", {})
    if not isinstance(usage, dict):
        raise ValueError("usage must be an object")
    for key in ("input_tokens", "output_tokens", "cost_usd"):
        value = usage.get(key)
        if value is not None and (
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
        ):
            raise ValueError("usage must be nonnegative finite numbers or null")
    return proposals, {key: usage.get(key) for key in ("input_tokens", "output_tokens", "cost_usd")}


def propose(run, config_path):
    """Invoke a trusted adapter, retain its receipt, and submit validated proposals.

    The command has ordinary user permissions. Its stdout/stderr must not contain
    credentials. This transport is not a security or spending-limit boundary.
    """
    config = read(Path(config_path))
    command = config.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(s, str) and s for s in command)
    ):
        raise ValueError("command must be a nonempty argv list; shell strings are unsupported")
    timeout = config.get("timeout_seconds", 120)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    env_names = config.get("credential_env", [])
    if not isinstance(env_names, list) or not all(isinstance(s, str) and s for s in env_names):
        raise ValueError("credential_env must list environment variable names")
    for key in ("provider", "model"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"{key} label is required")
    with engine.locked(run) as (run, state):
        request = context_data(run, state)
        maximum = config.get("max_proposals", 1)
        if type(maximum) is not int or maximum <= 0:
            raise ValueError("max_proposals must be a positive integer")
        request["max_proposals"] = min(maximum, request["remaining_proposals"])
        if not request["max_proposals"]:
            raise ValueError("proposal budget exhausted")
        request["provider"] = config["provider"]
        request["model"] = config["model"]
        call_id = "call_" + uuid.uuid4().hex
        folder = run / "model_calls" / call_id
        folder.mkdir(parents=True)
        write(folder / "request.json", request)
        receipt = {
            "schema_version": 1,
            "call_id": call_id,
            "started_utc": now(),
            "provider": config["provider"],
            "model": config["model"],
            "request_sha256": sha(folder / "request.json"),
            "status": "running",
            "timeout_seconds": timeout,
            "usage": None,
            "note": "Usage is adapter-reported; model time/cost is separate from worker budget.",
        }
        write(folder / "receipt.json", receipt)
        env = {
            k: v
            for k, v in os.environ.items()
            if k in {"PATH", "HOME", "TMPDIR", "LANG", *env_names}
        }
        env["PYTHONUNBUFFERED"] = "1"
        started = time.monotonic()
        try:
            with (
                (folder / "request.json").open("rb") as source,
                (folder / "response.json").open("wb") as output,
                (folder / "adapter.log").open("wb") as log,
            ):
                process = subprocess.Popen(
                    command,
                    stdin=source,
                    stdout=output,
                    stderr=log,
                    env=env,
                    start_new_session=True,
                )
                try:
                    process.wait(timeout=timeout)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    raise
            if process.returncode:
                raise ValueError("adapter failed; inspect its local log")
            if (folder / "response.json").stat().st_size > 11000000:
                raise ValueError("adapter response too large")
            response = read(folder / "response.json")
            proposals, usage = validate_response(response, request, state["proposals"])
            receipt.update(
                status="generated", usage=usage, response_sha256=sha(folder / "response.json")
            )
            paths = []
            for proposal in proposals:
                destination = folder / proposal["id"]
                destination.mkdir()
                (destination / "kernel.c").write_text(proposal["source"])
                metadata = {
                    "schema_version": 1,
                    "id": proposal["id"],
                    "operator": request["operator"],
                    "source_file": "kernel.c",
                    "hypothesis": proposal["hypothesis"],
                    "parameters": proposal.get("parameters", {}),
                    "author": config["provider"],
                    "generation": {
                        "provider": config["provider"],
                        "model": config["model"],
                        "call_id": call_id,
                        "receipt_file": str((folder / "receipt.json").relative_to(run)),
                    },
                }
                write(destination / "proposal.json", metadata)
                paths.append(destination / "proposal.json")
        except (Exception, KeyboardInterrupt) as error:
            receipt.update(status="failed", error_type=type(error).__name__)
            raise
        finally:
            receipt.update(elapsed_seconds=time.monotonic() - started, finished_utc=now())
            write(folder / "receipt.json", receipt)
            engine.event(
                run, state, "model_call_finished", call_id=call_id, status=receipt["status"]
            )
    # Admission still uses the same frozen-source checks as file submissions.
    # A concurrent seal/budget change can reject admission; generated files remain.
    for path in paths:
        engine.submit(run, path)
    return {
        "call_id": call_id,
        "submitted": [p.parent.name for p in paths],
        "elapsed_seconds": receipt["elapsed_seconds"],
        "usage": receipt["usage"],
        "receipt": str(folder / "receipt.json"),
    }
