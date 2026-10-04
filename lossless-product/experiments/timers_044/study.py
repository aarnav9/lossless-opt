"""Frozen, sequential one-shot authors: announced deadlines versus a blind control."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time

from lossless._native import engine
from lossless._native.common import read, sha, stamp, write


HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "search_043" / "study.py"
module_spec = importlib.util.spec_from_file_location("search_043_study", LEGACY)
legacy = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(legacy)

CONDITIONS = {
    "announced_5m": (300, True),
    "announced_30m": (1800, True),
    "announced_60m": (3600, True),
    "unannounced_60m": (3600, False),
}


def frozen_sources():
    return [Path(__file__), HERE / "analyze.py", LEGACY, LEGACY.with_name("compare.py")]


def freeze(out):
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, (seconds, announced) in CONDITIONS.items():
        folder = out / name
        legacy.freeze(folder, seconds)
        # New final cases: no campaign-043 holdout is reused to select a winner.
        spec = read(folder / "spec.json")
        shapes = [(9, 197), (43, 787), (101, 1553)]
        for case in spec["cases"]:
            if case["split"] == "evaluation":
                index = int(case["id"].split("_")[1])
                case["rows"], case["columns"] = shapes[index]
        write(folder / "spec.json", spec)
        prompt = read(folder / "prompt.json")
        if announced:
            prompt["authoring_allowance"] = {
                "wall_time_seconds": seconds,
                "instructions": (
                    "This is a maximum wall-clock allowance for this one response, including "
                    "reasoning and final JSON generation. The external controller will stop "
                    "the process at the deadline and an incomplete response will fail. "
                    "Choose your depth of analysis to fit this allowance, leaving time to "
                    "return all three complete candidates. Use additional analysis when it "
                    "can improve correctness or performance; you may finish early. This "
                    "does not add proposals, tools, feedback, or follow-up attempts."
                ),
            }
        write(folder / "prompt.json", prompt)
        plan = read(folder / "plan.json")
        plan["timer_condition"] = name
        plan["deadline_announced"] = announced
        plan["hashes"] = {key: sha(folder / key) for key in plan["hashes"]}
        write(folder / "plan.json", plan)
        for path in folder.iterdir():
            if path.is_file():
                files[str(path.relative_to(out))] = sha(path)
    rng = random.Random(4403)
    order = []
    for replicate in range(3):
        names = list(CONDITIONS)
        rng.shuffle(names)
        order.extend([name, replicate] for name in names)
    source_hashes = {str(path): sha(path) for path in frozen_sources()}
    write(
        out / "plan.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": "gpt-6-astra",
            "effort": "xhigh",
            "conditions": {
                name: {"seconds": seconds, "announced": announced}
                for name, (seconds, announced) in CONDITIONS.items()
            },
            "order": order,
            "hashes": files,
            "source_hashes": source_hashes,
            "authoring_cap_total_seconds": 3 * sum(v[0] for v in CONDITIONS.values()),
            "rules": (
                "Three independent authors per condition, three proposals per author, no retries "
                "or repairs. Sequential authoring in seeded randomized blocks. Same model, effort, "
                "discovery context, controls and numerical contract. All authoring and discovery "
                "choices across all conditions finish before any final evaluation. Timeouts and "
                "invalid responses count as failures; unfinished kernel quality is unmeasured. "
                "Compare actual wall time, tokens, completed/valid proposals, paired frozen winners "
                "and search-only payback. Announced budgets are prompt instructions, not a guarantee "
                "that a model knows elapsed wall time or consumes the allowance. Provider latency "
                "and stopping behavior are not controlled. This is exploratory with n=3 per arm, "
                "not a general scaling law, equal-wall-time search, or product-default promotion."
            ),
        },
    )


def verify(out):
    plan = read(out / "plan.json")
    for name, digest in plan["hashes"].items():
        assert sha(out / name) == digest, f"frozen input changed: {name}"
    for path, digest in plan["source_hashes"].items():
        assert sha(path) == digest, f"frozen harness changed: {path}"
    for name in plan["conditions"]:
        legacy.verify(out / name)
    return plan


def command_for(folder, target, empty_directory):
    command = [
        "codex",
        "exec",
        "--ignore-user-config",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--cd",
        empty_directory,
        "--model",
        "gpt-6-astra",
        "-c",
        'model_reasoning_effort="xhigh"',
        "-c",
        "project_doc_max_bytes=0",
        "-c",
        'web_search="disabled"',
        "-c",
        'approval_policy="never"',
    ]
    for feature in [
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
    ]:
        command += ["--disable", feature]
    return command + [
        "--json",
        "--output-schema",
        str(folder / "schema.json"),
        "--output-last-message",
        str(target / "raw_response.json"),
        "-",
    ]


def response_error(response):
    if not isinstance(response, dict) or set(response) != {"proposals"}:
        return "response must contain only proposals"
    proposals = response["proposals"]
    if not isinstance(proposals, list) or len(proposals) != 3:
        return "response must contain exactly three proposals"
    for proposal in proposals:
        if not isinstance(proposal, dict) or set(proposal) != {"id", "hypothesis", "source"}:
            return "invalid proposal fields"
        if any(not isinstance(v, str) or not v.strip() for v in proposal.values()):
            return "proposal fields must be nonempty strings"
    return None


def stop_process(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def author_one(folder, index, seconds):
    target = folder / f"author_{index}"
    receipt_path = target / "receipt.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        response = target / "response.json"
        assert (sha(response) if response.exists() else None) == receipt["response_sha256"]
        return
    # A started author without a receipt is never silently retried.
    target.mkdir(exist_ok=False)
    started = time.monotonic()
    timed_out = interrupted = False
    with tempfile.TemporaryDirectory(prefix="lossless-timer-author-") as empty:
        command = command_for(folder, target, empty)
        write(
            target / "invocation.json",
            {
                "argv": command,
                "prompt_sha256": sha(folder / "prompt.json"),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "timeout_seconds": seconds,
            },
        )
        with (
            (target / "events.jsonl").open("w") as events,
            (target / "stderr.log").open("w") as errors,
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
                        f"AUTHOR condition={folder.name} run={index + 1}/3 "
                        f"elapsed={elapsed:.0f}s cap={seconds}s result={target}"
                    )
                    remaining = seconds - elapsed
                    if remaining <= 0:
                        timed_out = True
                        stop_process(process)
                        break
                    try:
                        process.communicate(input=pending, timeout=min(30, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        pending = None
            except BaseException:
                interrupted = True
                stop_process(process)
                raise
            finally:
                if interrupted:
                    write(
                        target / "interrupted.json",
                        {
                            "elapsed_seconds": time.monotonic() - started,
                            "returncode": process.returncode,
                        },
                    )
        records, event_errors = [], []
        for line in (target / "events.jsonl").read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                event_errors.append("incomplete or invalid JSONL record")
        item_types = {r.get("item", {}).get("type") for r in records if "item" in r}
        # Only final messages/reasoning/plan annotations are allowed; unknown tools fail closed.
        forbidden = sorted(item_types - {"agent_message", "reasoning", "plan", None})
        completed = any(r.get("type") == "turn.completed" for r in records)
        error = None
        response = None
        raw_path = target / "raw_response.json"
        if raw_path.exists():
            try:
                response = read(raw_path)
                error = response_error(response)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                error = str(exc)
        else:
            error = "no final response"
        eligible = not (timed_out or process.returncode or forbidden or error or not completed)
        if eligible:
            write(target / "response.json", response)
        write(
            receipt_path,
            {
                "elapsed_seconds": time.monotonic() - started,
                "returncode": process.returncode,
                "timed_out": timed_out,
                "turn_completed": completed,
                "eligible": eligible,
                "response_error": error,
                "event_errors": event_errors,
                "usage": [r.get("usage") for r in records if r.get("type") == "turn.completed"],
                "cost_usd": None,
                "tool_items": forbidden,
                "blinding_passed": not forbidden,
                "response_sha256": sha(target / "response.json") if eligible else None,
                "raw_response_sha256": sha(raw_path) if raw_path.exists() else None,
            },
        )
    stamp(
        f"AUTHOR complete condition={folder.name} run={index + 1}/3 eligible={eligible} "
        f"timeout={timed_out} seconds={time.monotonic() - started:.1f}"
    )


def prepare_failed_authors(out, plan):
    for name, index in plan["order"]:
        folder = out / name
        receipt = read(folder / f"author_{index}" / "receipt.json")
        if receipt["eligible"]:
            continue
        run = folder / f"llm_{index}"
        if (run / "discovery_wall.json").exists():
            continue
        started = time.monotonic()
        engine.initialize(folder / "spec.json", run)
        write(
            run / "submission_failures.json",
            {
                "missing_slots": 3,
                "failures": [{"reason": "ineligible author response", "receipt": receipt}],
            },
        )
        write(
            run / "failed_search.json",
            {
                "status": "no_validated_proposals",
                "accepted": False,
                "reason": "ineligible author response",
                "choice": "reference",
                "evaluation_opened": False,
            },
        )
        write(run / "discovery_wall.json", {"seconds": time.monotonic() - started})


def require_all_sealed(out, plan):
    for name in plan["conditions"]:
        for index, arm in read(out / name / "plan.json")["order"]:
            run = out / name / f"{arm}_{index}"
            assert (run / "seal.json").exists() or (run / "failed_search.json").exists(), (
                f"final evaluation requires every condition to be sealed: {run}"
            )


def execute(out):
    plan = verify(out)
    for name, index in plan["order"]:
        author_one(out / name, index, plan["conditions"][name]["seconds"])
    prepare_failed_authors(out, plan)
    for name in plan["conditions"]:
        legacy.search(out / name, "discovery")
    require_all_sealed(out, plan)
    for name in plan["conditions"]:
        legacy.search(out / name, "evaluation")
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(LEGACY.with_name("compare.py")),
                "--output",
                str(out / name),
            ],
            check=True,
        )
    subprocess.run(
        [sys.executable, "-u", str(HERE / "analyze.py"), "--output", str(out)], check=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["freeze", "run", "verify"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    log = out.parent / (out.name + ".log")
    stamp(
        f"START phase={args.phase} estimated=1-2h author_cap_total=7h45m "
        f"result={out / 'summary.json'} log={log}"
    )
    if args.phase == "freeze":
        freeze(out)
    elif args.phase == "verify":
        verify(out)
    else:
        execute(out)
    stamp(f"COMPLETE phase={args.phase} result={out / 'summary.json'} log={log}")


if __name__ == "__main__":
    main()
