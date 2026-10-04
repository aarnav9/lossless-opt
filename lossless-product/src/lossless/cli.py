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
        "scope": "macOS/Linux alpha. Vision, CUDA execution, arbitrary input functions, and end-to-end proved contracts are not implemented. Candidate execution is trusted-local.",
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
    initialize.add_argument(
        "--template",
        choices=["native-kernel", "copy", "softmax", "rmsnorm", "mlx-fixed-count"],
        default="native-kernel",
    )
    initialize.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("config", type=Path)
    inspect.add_argument("--budget", type=duration, help="Total search allowance, e.g. 30m or 8h")
    inspect.add_argument(
        "--llm-timeout",
        type=duration,
        help="Override the configured LLM response allowance (default when omitted from job: 30m)",
    )
    profile = commands.add_parser(
        "profile", help="Profile discovery inputs without searching or calling an LLM"
    )
    profile.add_argument("config", type=Path)
    profile.add_argument(
        "--artifact",
        type=Path,
        help="Compare an exported implementation with the declared reference",
    )
    profile.add_argument("--repeats", type=int, default=7)
    profile.add_argument("--budget", type=duration)
    profile.add_argument("--output", type=Path)
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
            result = init(args.output, args.template)
        elif args.command == "inspect":
            result = (
                Workload.from_config(args.config)
                .with_time_limits(seconds=args.budget, llm_timeout_seconds=args.llm_timeout)
                .resolve()
            )
        elif args.command == "profile":
            from .profiling import profile

            outcome = profile(
                args.config,
                artifact=args.artifact,
                repeats=args.repeats,
                budget=args.budget,
                output=args.output,
            )
            result = {
                "run": str(outcome.directory),
                "status": outcome.summary["status"],
                "report": str(outcome.directory / "report.html"),
            }
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
