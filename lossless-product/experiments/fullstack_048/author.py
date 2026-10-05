"""Research-only Python proposal transport; reuses subscription CLI restrictions."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile
import time

from lossless import codex_provider as transport
from lossless._native.common import stamp
from protocol import read, write


def invoke(request, folder, timeout):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "schema.json", transport.response_schema(1))
    request = dict(
        request,
        response_format={
            "schema_version": 1,
            "proposals": [
                {
                    "id": "candidate_name",
                    "hypothesis": "why complete requests should improve",
                    "source": "Complete Python module defining Runner(model).run(prompts, counts, capture=True)",
                }
            ],
        },
    )
    write(folder / "prompt.json", request)
    started = time.monotonic()
    timed_out = False
    with tempfile.TemporaryDirectory(prefix="lossless-048-author-") as empty:
        argv = transport.command(folder, empty, request["author_model"], "xhigh")
        write(
            folder / "invocation.json",
            dict(
                model=request["author_model"],
                effort="xhigh",
                timeout_seconds=timeout,
                started_utc=datetime.now(timezone.utc).isoformat(),
                argv=argv,
            ),
        )
        with (
            (folder / "events.jsonl").open("w") as output,
            (folder / "stderr.log").open("w") as error,
        ):
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=error,
                text=True,
                env=transport.environment(),
            )
            pending = json.dumps(request)
            try:
                while True:
                    left = timeout - (time.monotonic() - started)
                    if left <= 0:
                        timed_out = True
                        transport.stop_tree(proc)
                        break
                    try:
                        proc.communicate(pending, timeout=min(left, 30))
                        break
                    except subprocess.TimeoutExpired:
                        pending = None
                        stamp(
                            f"048 author elapsed={time.monotonic() - started:.0f}s allowance={timeout:.0f}s"
                        )
            except BaseException:
                transport.stop_tree(proc)
                raise
    events, errors = [], []
    for line in (folder / "events.jsonl").read_text().splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            errors.append("invalid CLI event")
    completed = [e for e in events if e.get("type") == "turn.completed"]
    forbidden = sorted(
        {e.get("item", {}).get("type") for e in events if "item" in e}
        - {None, "agent_message", "reasoning", "plan"}
    )
    response = None
    try:
        response = read(folder / "raw_response.json")
        assert set(response) == {"schema_version", "proposals"} and response["schema_version"] == 1
        assert isinstance(response["proposals"], list) and len(response["proposals"]) == 1
        proposal = response["proposals"][0]
        assert set(proposal) == {"id", "hypothesis", "source"}
        assert all(isinstance(v, str) and v.strip() for v in proposal.values())
    except (OSError, ValueError, AssertionError, KeyError, TypeError):
        errors.append("incomplete or invalid proposal JSON")
    usage = [e.get("usage", {}) for e in completed]
    recorded = {
        k: sum(v[k] for v in usage) if usage and all(type(v.get(k)) is int for v in usage) else None
        for k in ["input_tokens", "output_tokens"]
    }
    receipt = dict(
        eligible=bool(completed) and not (timed_out or proc.returncode or forbidden or errors),
        elapsed_seconds=time.monotonic() - started,
        timed_out=timed_out,
        returncode=proc.returncode,
        forbidden_events=forbidden,
        errors=errors,
        usage={**recorded, "cost_usd": None},
    )
    write(folder / "receipt.json", receipt)
    if receipt["eligible"]:
        write(folder / "response.json", response)
    return receipt, response
