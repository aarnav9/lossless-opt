"""Codex CLI proposal transport using the user's existing ChatGPT sign-in.

Run as the command in a Lossless job. Standard output contains only the proposal
JSON. The model receives supplied discovery context, with tools disabled and an
empty working directory. Native code is still evaluated by Lossless separately.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from .jobs import read_json

DISABLED = [
    "shell_tool",
    "unified_exec",
    "apps",
    "plugins",
    "multi_agent",
    "browser_use",
    "computer_use",
    "memories",
    "hooks",
    "skill_search",
]


def environment():
    # Use saved CLI sign-in; do not accidentally switch to a paid API key.
    return {
        k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "TMPDIR", "LANG", "CODEX_HOME"}
    }


def require_chatgpt(timeout=10):
    status = subprocess.run(
        ["codex", "login", "status"],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment(),
    )
    if status.returncode or "Logged in using ChatGPT" not in status.stdout + status.stderr:
        raise ValueError(
            "Sign into Codex with ChatGPT first; this example does not use API-key billing"
        )


def effective_timeout(explicit=None):
    """Use the outer job's actual allowance; an optional CLI cap can shorten it."""
    inherited = os.environ.get("LOSSLESS_PROVIDER_TIMEOUT_SECONDS")
    values = [float(inherited)] if inherited is not None else []
    if explicit is not None:
        values.append(explicit)
    if not values:
        values = [1800.0]
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("positive finite response timeout required")
    return min(values)


def response_schema(maximum=1, mlx=False):
    fields = {"id": {"type": "string"}, "hypothesis": {"type": "string"}}
    if mlx:
        fields["options"] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["allocator_cache_mib"],
            "properties": {"allocator_cache_mib": {"type": "integer", "enum": [0, 64, 256]}},
        }
    else:
        fields["source"] = {"type": "string"}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "proposals"],
        "properties": {
            "schema_version": {"type": "integer", "enum": [1]},
            "proposals": {
                "type": "array",
                "minItems": 1,
                "maxItems": maximum,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(fields),
                    "properties": fields,
                },
            },
        },
    }


def command(folder, empty, model, effort):
    argv = [
        "codex",
        "exec",
        "--ignore-user-config",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--cd",
        str(empty),
        "--model",
        model,
        "-c",
        f'model_reasoning_effort="{effort}"',
        "-c",
        "project_doc_max_bytes=0",
        "-c",
        'web_search="disabled"',
        "-c",
        'approval_policy="never"',
    ]
    for feature in DISABLED:
        argv += ["--disable", feature]
    return argv + [
        "--json",
        "--output-schema",
        str(folder / "schema.json"),
        "--output-last-message",
        str(folder / "raw_response.json"),
        "-",
    ]


def stop_tree(process):
    """CLI children share the outer Lossless process group; also clean up on local timeout."""
    try:
        listing = subprocess.run(
            ["ps", "-eo", "pid=,ppid="], capture_output=True, text=True, check=False
        )
    except OSError:
        listing = None
    descendants = {process.pid}
    if listing is not None and listing.returncode == 0:
        pairs = [
            tuple(map(int, row.split()))
            for row in listing.stdout.splitlines()
            if len(row.split()) == 2
        ]
        while True:
            expanded = descendants | {pid for pid, parent in pairs if parent in descendants}
            if expanded == descendants:
                break
            descendants = expanded
    for pid in descendants - {process.pid}:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait()


def provider_messages(events):
    """Retain bounded CLI failure diagnostics independently of schema errors."""
    messages = []
    for event in events:
        if event.get("type") not in {"error", "turn.failed"}:
            continue
        error = event.get("error")
        message = event.get("message") or (
            error.get("message") if isinstance(error, dict) else error
        )
        if isinstance(message, str) and message[:2000] not in messages:
            messages.append(message[:2000])
        if len(messages) == 3:
            break
    return messages


def invoke(
    request,
    folder,
    *,
    model,
    effort,
    timeout,
    maximum=1,
    probe=False,
    schema_override=None,
    validator=None,
):
    """One response, no retries. Return a local receipt; never invent dollar cost."""
    from ._native.common import sha, write
    from .applications.runtime import clock

    if not math.isfinite(timeout) or timeout <= 0 or not 1 <= maximum <= 3:
        raise ValueError("positive finite timeout and 1..3 proposals required")
    if effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError("unsupported reasoning effort")
    if (schema_override is None) != (validator is None) or (probe and schema_override is not None):
        raise ValueError(
            "custom structured responses require both schema and validator, without probe"
        )
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    schema = (
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
        }
        if probe
        else schema_override
        or response_schema(maximum, request.get("adapter") == "mlx.fixed_count")
    )
    write(folder / "schema.json", schema)
    context = dict(request)
    if not probe and schema_override is None:
        context["max_proposals"] = maximum
        context["transport_instructions"] = (
            "Return only the requested proposal JSON. No tools, browsing, repository access or execution. Use only supplied discovery evidence. Propose changes; never change the evaluator, contract or acceptance thresholds. Return before the response allowance expires."
        )
        context["transport_timeout_upper_bound_seconds"] = timeout
        proposal = {"id": "unique_candidate_id", "hypothesis": "reason"}
        if request.get("adapter") == "mlx.fixed_count":
            proposal["options"] = {"allocator_cache_mib": 64}
        else:
            proposal["source"] = "complete C source exporting the supplied ABI"
        context["response_format"] = {"schema_version": 1, "proposals": [proposal]}
    elif schema_override is not None:
        context["transport_timeout_upper_bound_seconds"] = timeout
    write(folder / "prompt.json", context)
    started = clock()
    timed_out = False
    with tempfile.TemporaryDirectory(prefix="lossless-codex-empty-") as empty:
        argv = command(folder, empty, model, effort)
        write(
            folder / "invocation.json",
            {
                "model": model,
                "effort": effort,
                "argv": argv,
                "prompt_sha256": sha(folder / "prompt.json"),
                "timeout_seconds": timeout,
                "started_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        with (
            (folder / "events.jsonl").open("w") as events,
            (folder / "stderr.log").open("w") as errors,
        ):
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=events,
                stderr=errors,
                text=True,
                env=environment(),
            )
            pending = json.dumps(context, allow_nan=False)
            try:
                while True:
                    remaining = timeout - (clock() - started)
                    if remaining <= 0:
                        timed_out = True
                        stop_tree(process)
                        break
                    try:
                        process.communicate(pending, timeout=min(30, remaining))
                        timed_out = clock() - started > timeout
                        break
                    except subprocess.TimeoutExpired:
                        pending = None
                        print(
                            f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] Codex model={model} elapsed={clock() - started:.0f}s allowance={timeout:.0f}s",
                            file=sys.stderr,
                            flush=True,
                        )
            except BaseException:
                stop_tree(process)
                raise
    events, errors = [], []
    for line in (folder / "events.jsonl").read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            errors.append("invalid event record")
    forbidden = sorted(
        {e.get("item", {}).get("type") for e in events if "item" in e}
        - {None, "agent_message", "reasoning", "plan"}
    )
    completed = [e for e in events if e.get("type") == "turn.completed"]
    response = None
    try:
        response = read_json(folder / "raw_response.json")
        if probe:
            if response != {"ok": True}:
                errors.append("probe response was not ok")
        elif validator is not None:
            validator(response)
        else:
            fields = set(schema["properties"]["proposals"]["items"]["properties"])
            if (
                not isinstance(response, dict)
                or set(response) != {"schema_version", "proposals"}
                or type(response["schema_version"]) is not int
                or response["schema_version"] != 1
            ):
                raise ValueError("invalid response header")
            proposals = response["proposals"]
            if not isinstance(proposals, list) or not 1 <= len(proposals) <= maximum:
                raise ValueError("invalid proposal count")
            for proposal in proposals:
                if not isinstance(proposal, dict) or set(proposal) != fields:
                    raise ValueError("invalid proposal fields")
                if any(
                    not isinstance(proposal[k], str) or not proposal[k].strip()
                    for k in fields - {"options"}
                ):
                    raise ValueError("empty proposal field")
                if "options" in fields:
                    options = proposal["options"]
                    if (
                        not isinstance(options, dict)
                        or set(options) != {"allocator_cache_mib"}
                        or type(options["allocator_cache_mib"]) is not int
                        or options["allocator_cache_mib"] not in {0, 64, 256}
                    ):
                        raise ValueError("invalid allocator policy")
    except (OSError, ValueError, KeyError, TypeError):
        errors.append("no complete schema-valid response")
    eligible = not (timed_out or process.returncode or forbidden or errors) and bool(completed)
    usage = [e.get("usage", {}) for e in completed]
    if eligible and not probe:
        response["usage"] = {
            key: sum(v[key] for v in usage)
            if usage
            and all(
                isinstance(v.get(key), int) and not isinstance(v[key], bool) and v[key] >= 0
                for v in usage
            )
            else None
            for key in ["input_tokens", "output_tokens"]
        }
        response["usage"]["cost_usd"] = None
    if eligible:
        write(folder / "response.json", response)
    receipt = {
        "transport_source_sha256": sha(Path(__file__)),
        "model": model,
        "effort": effort,
        "elapsed_seconds": clock() - started,
        "eligible": eligible,
        "timed_out": timed_out,
        "returncode": process.returncode,
        "turn_completed": bool(completed),
        "tool_items": forbidden,
        "errors": errors,
        "provider_messages": provider_messages(events),
        "usage": usage,
        "cost_usd": None,
        "response_sha256": sha(folder / "response.json") if eligible else None,
    }
    write(folder / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--effort", choices=["low", "medium", "high", "xhigh", "max"], default="xhigh"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="optional extra response cap; otherwise uses the Lossless job allowance, or 1800 seconds standalone",
    )
    parser.add_argument("--proposals", type=int, default=1)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    try:
        timeout = effective_timeout(args.timeout)
        require_chatgpt()
        request = (
            {"task": 'Return exactly {"ok":true}. No tools.'}
            if args.probe
            else json.load(sys.stdin)
        )
        if (
            not args.probe
            and request.get("operator") not in {"copy", "softmax", "rmsnorm_residual"}
            and request.get("adapter") != "mlx.fixed_count"
        ):
            raise ValueError("unsupported Lossless proposal context")
        maximum = min(args.proposals, request.get("max_proposals", args.proposals))
        with tempfile.TemporaryDirectory(prefix="lossless-codex-response-") as temp:
            folder = Path(temp) / "call"
            receipt = invoke(
                request,
                folder,
                model=args.model,
                effort=args.effort,
                timeout=timeout,
                maximum=maximum,
                probe=args.probe,
            )
            if not receipt["eligible"]:
                raise ValueError(
                    "Codex returned no eligible response; check model access, CLI version and allowance"
                )
            print(json.dumps(read_json(folder / "response.json"), allow_nan=False), flush=True)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        message = str(error) if isinstance(error, ValueError) else type(error).__name__
        print(f"Codex proposal transport failed: {message}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
