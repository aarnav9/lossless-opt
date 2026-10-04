"""One isolated CLI response; all state and feedback are supplied in the prompt."""

from datetime import datetime, timezone
import json
import subprocess
import tempfile
import time

from lossless._native.common import read, write, sha, stamp


def run(folder, seconds, timer):
    receipt_path = folder / "receipt.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        response = folder / "response.json"
        assert (sha(response) if response.exists() else None) == receipt["response_sha256"]
        return receipt
    if (folder / "invocation.json").exists():
        raise RuntimeError("Started author has no receipt; do not silently retry it")
    started = time.monotonic()
    timed_out = False
    with tempfile.TemporaryDirectory(prefix="lossless-iterate-author-") as empty:
        command = timer.command_for(folder, folder, empty)
        write(
            folder / "invocation.json",
            {
                "argv": command,
                "prompt_sha256": sha(folder / "prompt.json"),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "timeout_seconds": seconds,
            },
        )
        with (
            (folder / "events.jsonl").open("w") as events,
            (folder / "stderr.log").open("w") as errors,
        ):
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=events,
                stderr=errors,
                text=True,
                start_new_session=True,
            )
            pending = (folder / "prompt.json").read_text()
            try:
                while True:
                    elapsed = time.monotonic() - started
                    stamp(
                        f"AUTHOR round={folder.name} elapsed={elapsed:.0f}s cap={seconds:.0f}s result={folder}"
                    )
                    remaining = seconds - elapsed
                    if remaining <= 0:
                        timed_out = True
                        timer.stop_process(process)
                        break
                    try:
                        process.communicate(input=pending, timeout=min(30, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        pending = None
            except BaseException:
                timer.stop_process(process)
                write(folder / "interrupted.json", {"elapsed_seconds": time.monotonic() - started})
                raise
    events, parse_errors = [], []
    for line in (folder / "events.jsonl").read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            parse_errors.append("incomplete or invalid JSONL record")
    types = {event.get("item", {}).get("type") for event in events if "item" in event}
    forbidden = sorted(types - {"agent_message", "reasoning", "plan", None})
    completed = any(event.get("type") == "turn.completed" for event in events)
    raw = folder / "raw_response.json"
    response, error = None, "no final response"
    if raw.exists():
        try:
            response = read(raw)
            error = timer.response_error(response)
        except (ValueError, UnicodeDecodeError) as exc:
            error = str(exc)
    eligible = not (
        timed_out or process.returncode or forbidden or parse_errors or error or not completed
    )
    if eligible:
        write(folder / "response.json", response)
    receipt = {
        "elapsed_seconds": time.monotonic() - started,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "eligible": eligible,
        "turn_completed": completed,
        "response_error": error,
        "event_errors": parse_errors,
        "tool_items": forbidden,
        "usage": [event.get("usage") for event in events if event.get("type") == "turn.completed"],
        "cost_usd": None,
        "response_sha256": sha(folder / "response.json") if eligible else None,
        "raw_response_sha256": sha(raw) if raw.exists() else None,
    }
    write(receipt_path, receipt)
    stamp(
        f"AUTHOR complete round={folder.name} eligible={eligible} timeout={timed_out} seconds={receipt['elapsed_seconds']:.1f}"
    )
    return receipt
