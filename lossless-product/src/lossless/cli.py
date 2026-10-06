"""The supported command-line interface; imports do not install dependencies."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import subprocess
import threading
import time

from . import __version__
from .jobs import Workload, template, read_json


@contextmanager
def search_progress(output, interval=30):
    """Keep long foreground searches visible, including quiet provider waits."""
    from ._native.common import stamp

    stopped = threading.Event()
    started = time.monotonic()

    def heartbeat():
        while not stopped.wait(interval):
            stamp(
                f"SEARCH running elapsed={time.monotonic() - started:.0f}s result={output / 'report.json'}"
            )

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def duration(value):
    try:
        scale = {"s": 1, "m": 60, "h": 3600}
        result = float(value[:-1]) * scale[value[-1]] if value[-1] in scale else float(value)
        if not __import__("math").isfinite(result) or result <= 0:
            raise ValueError()
        return result
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError("use a positive duration such as 60s, 30m or 8h")


def doctor():
    import platform

    versions = {}
    for name in ["numpy", "scipy", "mlx", "mlx-lm"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    from .proofs import lean_binary, TOOLCHAIN

    try:
        proof = {"available": True, "executable": lean_binary(), "toolchain": TOOLCHAIN}
    except (ValueError, OSError, TimeoutError, subprocess.SubprocessError):
        proof = {
            "available": False,
            "toolchain": TOOLCHAIN,
            "next_step": "lossless setup --proofs lean",
        }
    native = (
        sys.platform in {"darwin", "linux"}
        and shutil.which("clang") is not None
        and versions["numpy"] is not None
    )
    mlx = (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and versions["mlx"] == "0.32.3"
        and versions["mlx-lm"] == "0.32.0"
    )
    return {
        "version": __version__,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "dependencies": versions,
        "clang": shutil.which("clang"),
        "proofs": proof,
        "adapters": {
            "native.copy": {"available": native, "contract": "exact, finite validation"},
            "native.softmax": {
                "available": native,
                "contract": "numerical, fixed adapter tolerances",
            },
            "native.rmsnorm_residual": {
                "available": native,
                "contract": "numerical, fixed adapter tolerances",
            },
            "mlx.fixed_count": {
                "dependencies_available": mlx,
                "admission": "Apple M2, pinned model identity and runtime; checked on model load",
            },
        },
        "next_step": "lossless demo --budget 60s"
        if native
        else "Install clang and NumPy in this Python environment, then rerun lossless doctor",
        "application_intake": {
            "available": sys.platform in {"darwin", "linux"},
            "actions": [
                "init --project",
                "inspect",
                "replay",
                "profile",
                "profile --explain",
                "search-application --plan",
            ],
            "optimization_available": False,
            "experimental_source_search_available": True,
        },
        "scope": "macOS/Linux alpha. Existing-application intake, replay, callable profiling/assessment and scoped experimental source search are available. Guarded application deployment, general vision/CUDA execution and end-to-end proved contracts are not implemented. Execution is trusted-local.",
    }


def init(folder, name):
    folder = Path(folder).resolve()
    if name == "mlx-fixed-count":
        config = {
            "schema_version": 1,
            "name": "mlx-fixed-count",
            "workload": {
                "adapter": "mlx.fixed_count",
                "parameters": {
                    "model": "./model",
                    "sampler": "greedy",
                    "fixed_count": True,
                    "repeats": 5,
                },
                "inputs": "./requests.json",
            },
            "contract": {"preset": "exact", "required_evidence": "validated"},
            "objective": {"metric": "throughput", "min_speedup": 1.02},
            "budget": {"wall_time_seconds": 180, "max_candidates": 3, "max_llm_calls": 1},
            "proofs": {"check": True},
            "target": {"device": "metal"},
            "output": {"directory": "./lossless-runs"},
        }
    else:
        adapter = {
            "native-kernel": "native.copy",
            "copy": "native.copy",
            "softmax": "native.softmax",
            "rmsnorm": "native.rmsnorm_residual",
        }[name]
        config = template(adapter)
    folder.mkdir(parents=True, exist_ok=False)
    from ._native.common import write

    write(folder / "lossless.json", config)
    (folder / ".gitignore").write_text("lossless-runs/\n.env\n*.local.json\n")
    if name == "mlx-fixed-count":
        write(
            folder / "requests.json",
            [
                {"text": "Explain how a cache works.", "max_tokens": 8, "split": "discovery"},
                {"text": "Describe how a CPU works.", "max_tokens": 8, "split": "discovery"},
                {"text": "Explain how a forest grows.", "max_tokens": 8, "split": "evaluation"},
                {"text": "Describe how a garden grows.", "max_tokens": 8, "split": "evaluation"},
            ],
        )
    (folder / "README.md").write_text(
        '# Lossless job\n\nInspect with `lossless inspect lossless.json`, then run `lossless optimize lossless.json`.\n\nOptionally add `llm: {"callback": "my_llm.py:complete"}` as a JSON section. The function receives a JSON prompt string and returns a JSON proposal string. Callbacks and generated native code execute with local user permissions. Keep provider keys in environment variables.\n\nNative cases are generated by the adapter; the preset fixes its supported reference and contract. MLX requires local model weights and its pinned support matrix. Never replace evaluation cases using discovery feedback.\n'
        "\nSearch time is a configurable performance input. Set `budget.wall_time_seconds` for the whole search and `llm.timeout_seconds` for each author response (default 30 minutes). With an LLM configured, inspect and run using `--budget 8h --llm-timeout 30m`. Each response is capped by the remaining search time after reserving evaluation time, so a short total budget can cut it off earlier. More time can help useful work finish; it does not guarantee a gain or change correctness. Candidate/call caps still apply, and the run can finish early. Leave time for compilation and final evaluation.\n"
    )
    return {"directory": str(folder), "config": str(folder / "lossless.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="lossless",
        description="Optimize supported computations under an explicit correctness contract.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    hardware = commands.add_parser("hardware")
    hardware.add_argument("--output", type=Path)
    hardware.add_argument("--basic", action="store_true")
    initialize = commands.add_parser("init")
    input_kind = initialize.add_mutually_exclusive_group()
    input_kind.add_argument(
        "--template",
        choices=["native-kernel", "copy", "softmax", "rmsnorm", "mlx-fixed-count"],
    )
    input_kind.add_argument(
        "--project",
        type=Path,
        help="Inspect an existing local application without importing or running it",
    )
    initialize.add_argument("--entry", help="Existing file.py:function or package.module:function")
    initialize.add_argument(
        "--factory",
        action="store_true",
        help="Entry creates a callable model/object with no arguments",
    )
    initialize.add_argument("--cases", type=Path, help="Representative application cases JSON")
    initialize.add_argument(
        "--python",
        type=Path,
        help="Selected interpreter, defaulting to the active Lossless environment",
    )
    initialize.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Project-relative path/glob excluded from intake; repeatable",
    )
    initialize.add_argument(
        "--include",
        action="append",
        default=[],
        help="Explicit project-relative fixture file to include even if gitignored; repeatable",
    )
    initialize.add_argument("--output", type=Path, required=True)
    initialize.add_argument(
        "--run",
        dest="run_argv",
        nargs=argparse.REMAINDER,
        help="Existing command argv; put this last, after all Lossless options. No shell expansion.",
    )
    inspect = commands.add_parser("inspect")
    inspect.add_argument("config", type=Path)
    inspect.add_argument(
        "--json",
        action="store_true",
        help="Full application intake record instead of readable preflight (native jobs already return JSON)",
    )
    inspect.add_argument("--budget", type=duration, help="Total search allowance, e.g. 30m or 8h")
    inspect.add_argument(
        "--llm-timeout",
        type=duration,
        help="Override the configured LLM response allowance (default when omitted from job: 30m)",
    )
    replay = commands.add_parser(
        "replay",
        help="Replay an existing application reference in fresh copied workspaces; no optimization",
    )
    replay.add_argument("config", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--repeats", type=int)
    replay.add_argument("--budget", type=duration)
    replay.add_argument("--split", choices=["discovery", "evaluation", "all"], default="discovery")
    application_search = commands.add_parser(
        "search-application",
        help="Experimental source-body author/test/revise loop; original project unchanged",
    )
    application_search.add_argument("config", type=Path)
    application_search.add_argument("--plan", type=Path, required=True)
    application_search.add_argument("--output", type=Path, required=True)
    application_search.add_argument("--budget", type=duration)
    profile = commands.add_parser(
        "profile", help="Profile discovery inputs; optional application assessment via --explain"
    )
    profile.add_argument("config", type=Path)
    profile.add_argument(
        "--artifact",
        type=Path,
        help="Compare an exported implementation with the declared reference",
    )
    profile.add_argument("--repeats", type=int)
    profile.add_argument("--budget", type=duration)
    profile.add_argument("--output", type=Path)
    explanation = profile.add_mutually_exclusive_group()
    explanation.add_argument(
        "--explain",
        dest="explain",
        action="store_true",
        default=None,
        help="Share bounded discovery measurements/source excerpts with subscription-backed Codex for an assessment",
    )
    explanation.add_argument("--no-explain", dest="explain", action="store_false")
    profile.add_argument(
        "--model", help="Assessment model (default gpt-6-astra); enables assessment"
    )
    profile.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Assessment reasoning (default medium)",
    )
    profile.add_argument(
        "--llm-timeout",
        type=duration,
        help="Assessment response cap, default 30m, clipped to total profile budget",
    )
    for name in ["optimize", "demo"]:
        command = commands.add_parser(name)
        if name == "optimize":
            command.add_argument("config", type=Path)
            command.add_argument("--resume", action="store_true")
            command.add_argument(
                "--llm-timeout",
                type=duration,
                help="Override the configured LLM response allowance (default when omitted from job: 30m); bounded by remaining search time",
            )
        command.add_argument(
            "--budget", type=duration, help="Total search allowance, e.g. 60s, 30m or 8h"
        )
        command.add_argument("--output", type=Path)
    report = commands.add_parser("report")
    report.add_argument("run", type=Path)
    export = commands.add_parser("export")
    export.add_argument("run", type=Path)
    export.add_argument("--output", type=Path, required=True)
    setup = commands.add_parser("setup")
    setup.add_argument("--proofs", choices=["lean", "mathlib"], required=True)
    setup.add_argument("--directory", type=Path)
    proofs = commands.add_parser("proofs")
    ps = proofs.add_subparsers(dest="operation", required=True)
    check = ps.add_parser("check")
    check.add_argument("--timeout", type=duration, default=60)
    check.add_argument("--output", type=Path)
    search = ps.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor()
        elif args.command == "hardware":
            from ._native.hardware import emit_report

            emit_report(args.output, basic=args.basic)
            return
        elif args.command == "init":
            if args.project is not None:
                from .applications import initialize as initialize_application

                if args.run_argv == []:
                    raise ValueError("--run requires a command argv list")
                result = initialize_application(
                    args.project,
                    args.output,
                    entry=args.entry,
                    factory=args.factory,
                    command=args.run_argv,
                    cases=args.cases,
                    python=args.python,
                    exclude=args.exclude,
                    include=args.include,
                )
            else:
                if (
                    args.entry
                    or args.factory
                    or args.run_argv is not None
                    or args.cases
                    or args.python
                    or args.exclude
                    or args.include
                ):
                    raise ValueError("application intake options require --project")
                result = init(args.output, args.template or "native-kernel")
        elif args.command == "inspect":
            configuration = read_json(args.config)
            if isinstance(configuration, dict) and configuration.get("kind") == "application":
                from .applications import inspect as inspect_application

                if args.llm_timeout is not None:
                    raise ValueError("application intake/replay makes no LLM calls")
                result = inspect_application(args.config)
                if args.budget is not None:
                    result["resolved"]["replay"]["wall_time_seconds"] = args.budget
                    result["preflight"]["total_seconds"] = args.budget
                if not args.json:
                    from .applications.config import preflight_text

                    print(preflight_text(result))
                    return
            else:
                result = (
                    Workload.from_config(args.config)
                    .with_time_limits(seconds=args.budget, llm_timeout_seconds=args.llm_timeout)
                    .resolve()
                )
        elif args.command == "replay":
            from .applications import replay as replay_application

            outcome = replay_application(
                args.config,
                output=args.output,
                repeats=args.repeats,
                budget=args.budget,
                split=args.split,
            )
            result = {
                "run": str(outcome.directory),
                "status": outcome.summary["status"],
                "report": str(outcome.directory / "report.html"),
            }
            if outcome.summary["status"] != "replayed":
                print(json.dumps(result, indent=2))
                raise SystemExit(2)
        elif args.command == "search-application":
            from .applications import search

            outcome = search(args.config, plan=args.plan, output=args.output, budget=args.budget)
            result = {
                "run": str(outcome.directory),
                "status": outcome.summary["status"],
                "selected": outcome.summary["selected"],
                "report": str(outcome.directory / "report.html"),
            }
            if outcome.summary["status"] not in {"accepted_experiment", "reference_retained"}:
                print(json.dumps(result, indent=2))
                raise SystemExit(2)
        elif args.command == "profile":
            from .profiling import profile

            analysis = None
            if args.explain is False:
                if args.model or args.effort or args.llm_timeout:
                    raise ValueError("--no-explain conflicts with assessment options")
                analysis = False
            elif args.explain or args.model or args.effort or args.llm_timeout:
                configured = read_json(args.config).get("analysis") or {}
                analysis = {**configured, "provider": "codex-chatgpt"}
                for key, value in (
                    ("model", args.model),
                    ("effort", args.effort),
                    ("timeout_seconds", args.llm_timeout),
                ):
                    if value is not None:
                        analysis[key] = value
            outcome = profile(
                args.config,
                artifact=args.artifact,
                repeats=args.repeats,
                budget=args.budget,
                output=args.output,
                analysis=analysis,
            )
            result = {
                "run": str(outcome.directory),
                "status": outcome.summary["status"],
                "report": str(outcome.directory / "report.html"),
            }
            if "assessment" in outcome.summary:
                result["assessment"] = outcome.summary["assessment"]["status"]
                if outcome.summary["status"] != "profiled" or result["assessment"] not in {
                    "disabled",
                    "completed",
                }:
                    print(json.dumps(result, indent=2))
                    raise SystemExit(2)
        elif args.command in {"optimize", "demo"}:
            from .api import optimize

            workload = (
                Workload.from_config(args.config)
                if args.command == "optimize"
                else Workload.from_dict(template())
            )
            workload = workload.with_time_limits(
                seconds=args.budget, llm_timeout_seconds=getattr(args, "llm_timeout", None)
            )
            resolved = workload.resolve()
            output = (
                args.output.resolve()
                if args.output
                else Path(resolved["output"]["directory"])
                / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            )
            with search_progress(output):
                outcome = optimize(workload, output=output, resume=getattr(args, "resume", False))
            result = {
                "run": str(outcome.directory),
                "status": outcome.summary["status"],
                "selected": outcome.summary["selected"],
                "report": str(outcome.directory / "report.html"),
            }
        elif args.command == "report":
            result = read_json(args.run / "report.json")
        elif args.command == "export":
            from .artifacts import export

            result = {"artifact": str(export(args.run, args.output))}
        elif args.command == "setup":
            from .proofs import setup

            result = setup(args.proofs, directory=args.directory)
        elif args.command == "proofs":
            from .proofs import check, search

            result = (
                check(timeout=args.timeout, output=args.output)
                if args.operation == "check"
                else search(args.query, args.limit)
            )
        print(json.dumps(result, indent=2, allow_nan=False))
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        RuntimeError,
        TimeoutError,
        ImportError,
        subprocess.SubprocessError,
    ) as error:
        print(f"lossless: {error}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
