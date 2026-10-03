"""Independent one-shot authors, equal three-proposal budgets, sealed final cases."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import tempfile
import time

from lossless import recipes
from lossless._native import engine, operators
from lossless._native.common import read, write, sha, stamp
from lossless._native.worker import COMPILE_FLAGS


def sources(arm):
    retained = {p["id"]: p["source"] for p in recipes.native_candidates("softmax")}
    if arm == "retained":
        return {"baseline_control": operators.baseline_source("softmax"), **retained}
    row = retained["softmax_row_16"].replace("void kernel(", "static void row(")
    column = retained["softmax_column_4"]
    return {
        f"dispatch_{n}": row
        + column.replace("void kernel(", "static void column(")
        .replace("[4]", f"[{n}]")
        .replace("start+=4", f"start+={n}")
        .replace("count>4", f"count>{n}")
        .replace("count=4", f"count={n}")
        + "\nvoid kernel("
        + operators.ABI
        + "){if(rs==1 && rows>="
        + str(n)
        + ")column(x,r,w,out,inv,px,pr,rows,cols,rs,cs);else row(x,r,w,out,inv,px,pr,rows,cols,rs,cs);}\n"
        for n in [2, 8, 16]
    }


def freeze(out, author_seconds=1800):
    if not 1 <= author_seconds <= 28800:
        raise ValueError("author_seconds must be in 1..28800")
    out.mkdir(parents=True, exist_ok=False)
    cases = [
        dict(id=f"{split}_{i}_{layout}", split=split, rows=r, columns=c, layout=layout)
        for split, shapes in [
            ("discovery", [(6, 191), (48, 769)]),
            ("evaluation", [(5, 193), (41, 773), (97, 1543)]),
        ]
        for i, (r, c) in enumerate(shapes)
        for layout in ["c", "f", "slice2"]
    ]
    spec = dict(
        operator="softmax",
        comparator="numpy_buffered",
        cases=cases,
        budget_seconds=180,
        job_timeout_seconds=8,
        compile_timeout_seconds=8,
        target_ns=500000,
        screening_blocks=5,
        confirmation_blocks=15,
        max_proposals=3,
        min_speedup=1.02,
        cache={"directory": None},
    )
    write(out / "spec.json", spec)
    controls = {a: sources(a) for a in ["retained", "enumerated"]}
    write(out / "controls.json", controls)
    prompt = {
        "task": "Return exactly three distinct complete C softmax kernel candidates. One-shot authoring: no tools, browsing, filesystem access, or execution. Work only from the supplied context. Do not read or infer any evaluation data. Your three proposals consume the entire evaluation allowance; no repairs or follow-up feedback are available.",
        "objective": "Improve geometric-mean latency across small and medium row counts and C/F/slice2 layouts, against the supplied retained implementations and buffered NumPy. Avoid shape overfitting and regressions. No approximate exp, reduced precision, threads, external libraries, filesystem/network, undefined behavior, or compiler flag changes.",
        "hardware": "Apple M2 arm64 CPU, Apple clang; single-threaded native C via ctypes",
        "compiler_flags": COMPILE_FLAGS,
        "contract": operators.CONTRACTS["softmax"],
        "abi": operators.ABI,
        "buffers": "x indexed in float elements by i*rs+j*cs. out/px/pr each have rows*cols floats; inv has rows floats. r/w unused for softmax. Inputs read-only; output independent C layout. Buffers uninitialized; do not read before writing. Arbitrary positive sizes and supported nonoverlapping positive strides. Export void kernel(...), include stddef.h/math.h as needed.",
        "discovery_cases": [c for c in cases if c["split"] == "discovery"],
        "retained_sources": controls["retained"],
    }
    write(out / "prompt.json", prompt)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["proposals"],
        "properties": {
            "proposals": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "hypothesis", "source"],
                    "properties": {k: {"type": "string"} for k in ["id", "hypothesis", "source"]},
                },
            }
        },
    }
    write(out / "schema.json", schema)
    order = [(i, a) for i in range(3) for a in ["retained", "enumerated", "llm"]]
    random.Random(4301).shuffle(order)
    write(
        out / "plan.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": "gpt-6-astra",
            "effort": "xhigh",
            "independent_sessions": 3,
            "author_timeout_seconds": author_seconds,
            "candidate_cap": 3,
            "order": order,
            "hashes": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
            "script_sha256": sha(__file__),
            "rules": "Freeze all proposals and all discovery choices before opening final cases. Equal candidate and timing-block caps, not equal total wall time. No repairs/retries. Missing/invalid proposals count as failures. No default promotion. Final acceptance: geomean>=1.02 over NumPy and every case lower paired interval>1; incremental value requires comparison against independently selected retained winner. Dollar cost unavailable unless CLI provides billing.",
        },
    )


def verify(out):
    plan = read(out / "plan.json")
    assert sha(__file__) == plan["script_sha256"], "study script changed after freezing"
    for name, digest in plan["hashes"].items():
        assert sha(out / name) == digest, name
    return plan


def author(out):
    plan = verify(out)
    for index in range(plan["independent_sessions"]):
        target = out / f"author_{index}"
        target.mkdir(exist_ok=False)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="lossless-blind-author-") as folder:
            command = [
                "codex",
                "exec",
                "--ignore-user-config",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--cd",
                folder,
                "--model",
                plan["model"],
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
            command += [
                "--json",
                "--output-schema",
                str(out / "schema.json"),
                "--output-last-message",
                str(target / "response.json"),
                "-",
            ]
            write(
                target / "invocation.json",
                {"argv": command, "prompt_sha256": sha(out / "prompt.json")},
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
                process.stdin.write((out / "prompt.json").read_text())
                process.stdin.close()
                timed_out = False
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    stamp(f"AUTHOR run={index + 1}/3 elapsed={elapsed:.0f}s result={target}")
                    try:
                        process.wait(
                            timeout=min(30, max(0.1, plan["author_timeout_seconds"] - elapsed))
                        )
                    except subprocess.TimeoutExpired:
                        if time.monotonic() - started >= plan["author_timeout_seconds"]:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                            timed_out = True
                records = [
                    json.loads(v)
                    for v in (target / "events.jsonl").read_text().splitlines()
                    if v.startswith("{")
                ]
                usage = [r.get("usage") for r in records if r.get("type") == "turn.completed"]
                tool_items = [
                    r
                    for r in records
                    if r.get("item", {}).get("type")
                    in {
                        "command_execution",
                        "mcp_tool_call",
                        "web_search",
                        "file_change",
                        "collab_tool_call",
                    }
                ]
                write(
                    target / "receipt.json",
                    {
                        "elapsed_seconds": time.monotonic() - started,
                        "returncode": process.returncode,
                        "timed_out": timed_out,
                        "usage": usage,
                        "cost_usd": None,
                        "tool_items": tool_items,
                        "blinding_passed": not tool_items,
                        "response_sha256": sha(target / "response.json")
                        if (target / "response.json").exists()
                        else None,
                    },
                )
        stamp(f"AUTHOR complete run={index + 1}/3 seconds={time.monotonic() - started:.1f}")


def search(out, phase):
    plan = verify(out)
    if phase == "evaluation":
        assert all(
            (out / f"{a}_{i}" / "seal.json").exists()
            or (out / f"{a}_{i}" / "failed_search.json").exists()
            for i, a in plan["order"]
        ), "all discovery decisions must be sealed first"
    for index, arm in plan["order"]:
        run = out / f"{arm}_{index}"
        if (run / f"{phase}_wall.json").exists():
            continue
        started = time.monotonic()
        if phase == "discovery":
            if not run.exists():
                engine.initialize(out / "spec.json", run)
            if arm == "llm":
                receipt = read(out / f"author_{index}/receipt.json")
                assert receipt["blinding_passed"], "author tool use invalidates blinding"
                response_path = out / f"author_{index}/response.json"
                proposals = (
                    read(response_path).get("proposals", []) if response_path.exists() else []
                )
            else:
                proposals = [
                    dict(id=n, source=s, hypothesis=f"Frozen {arm} control")
                    for n, s in read(out / "controls.json")[arm].items()
                ]
            failures = []
            seen = set()
            for slot, proposal in enumerate(proposals[:3]):
                try:
                    source = proposal["source"]
                    digest = __import__("hashlib").sha256(source.encode()).hexdigest()
                    if digest in seen:
                        raise ValueError("duplicate source consumes a proposal slot")
                    seen.add(digest)
                    name = f"candidate_{slot}"
                    folder = run / "submitted" / name
                    folder.mkdir(parents=True)
                    (folder / "kernel.c").write_text(source)
                    write(
                        folder / "proposal.json",
                        dict(
                            schema_version=1,
                            id=name,
                            operator="softmax",
                            source_file="kernel.c",
                            hypothesis=proposal["hypothesis"],
                        ),
                    )
                    engine.submit(run, folder / "proposal.json")
                except (ValueError, KeyError, TypeError) as error:
                    failures.append({"slot": slot, "error": str(error)})
            write(
                run / "submission_failures.json",
                {"failures": failures, "missing_slots": max(0, 3 - len(proposals))},
            )
            try:
                if not engine.verify(run)["proposals"]:
                    raise ValueError("no validated proposals")
                engine.execute(run, "discovery")
                engine.seal(run)
            except ValueError as error:
                if str(error) != "no validated proposals":
                    raise
                write(
                    run / "failed_search.json",
                    {
                        "status": "no_validated_proposals",
                        "accepted": False,
                        "reason": str(error),
                        "choice": "reference",
                        "evaluation_opened": False,
                    },
                )
        else:
            if not (run / "failed_search.json").exists():
                engine.execute(run, "evaluation")
                engine.report(run)
        write(run / f"{phase}_wall.json", {"seconds": time.monotonic() - started})
        stamp(
            f"SEARCH {phase} arm={arm} run={index + 1}/3 seconds={time.monotonic() - started:.1f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "author", "discovery", "evaluation"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--author-seconds",
        type=int,
        default=1800,
        help="Per independent author; frozen before execution",
    )
    args = parser.parse_args()
    out = args.output.resolve()
    stamp(
        f"START phase={args.phase} result={out} log={out.parent / (out.name + '_' + args.phase + '.log')}"
    )
    if args.phase == "freeze":
        freeze(out, args.author_seconds)
    elif args.phase == "author":
        author(out)
    else:
        search(out, args.phase)
    stamp(
        f"COMPLETE phase={args.phase} result={out} log={out.parent / (out.name + '_' + args.phase + '.log')}"
    )


if __name__ == "__main__":
    main()
