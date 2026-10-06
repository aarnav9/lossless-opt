"""Freeze and replay the user's reference; never select or export an optimization."""

from dataclasses import dataclass
import html
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile

from .._native.common import stamp, write
from ..jobs import identity, positive, read_json
from .config import inspect, resolve
from .discovery import digest, inventory
from .runtime import clock, execute


@dataclass
class ReplayResult:
    directory: Path
    summary: dict


def copy_project(root, destination, snapshot):
    destination.mkdir(parents=True, exist_ok=False)
    for name, expected in snapshot["files"].items():
        source, target = root / name, destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink() or not source.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"source escaped the project during snapshot: {name}")
        shutil.copy2(source, target)
        if (
            target.stat().st_size != expected["bytes"]
            or digest(target) != expected["sha256"]
            or stat.S_IMODE(target.stat().st_mode) != expected["mode"]
        ):
            raise ValueError(f"source changed during snapshot: {name}; start a new replay")


def replay(config, *, output, repeats=None, budget=None, split="discovery", _profile=None):
    if sys.platform not in {"darwin", "linux"}:
        raise ValueError("application replay currently requires macOS or Linux")
    if split not in {"discovery", "evaluation", "all"}:
        raise ValueError("replay split must be discovery, evaluation or all")
    started = clock()
    config = Path(config).resolve()
    original_config_digest = digest(config)
    info = inspect(config)
    if info["status"] != "ready_to_replay":
        raise ValueError("Application setup is incomplete: " + " ".join(info["missing"]))
    original_cases_digest = digest(info["resolved"]["cases"])
    value, cases = resolve(config)
    if (
        value != info["resolved"]
        or digest(config) != original_config_digest
        or digest(value["cases"]) != original_cases_digest
    ):
        raise ValueError("configuration changed during intake; retry")
    settings = value["replay"]
    if _profile is not None:
        if value["entry"]["kind"] != "callable" or split != "discovery":
            raise ValueError("application profiling requires a callable and discovery inputs")
        settings["repeats"] = len(_profile["modes"])
    elif repeats is not None:
        if type(repeats) is not int or not 2 <= repeats <= 20:
            raise ValueError("repeats must be 2..20")
        settings["repeats"] = repeats
    if budget is not None:
        settings["wall_time_seconds"] = positive(budget, "replay budget")
    cases = [case for case in cases if split == "all" or case["split"] == split]
    if not cases:
        raise ValueError(f"no {split} cases supplied")
    root = Path(value["project"]["root"])
    out = Path(output).resolve()
    if out == root or out in root.parents:
        raise ValueError("replay output must not contain the source project")
    if out.is_relative_to(root):
        prefix = out.relative_to(root).as_posix()
        if any(n == prefix or n.startswith(prefix + "/") for n in info["source"]["files"]):
            raise ValueError("replay output overlaps source files")
        value["project"]["exclude"].append(prefix)
    out.mkdir(parents=True, exist_ok=False)
    log = out / "run.log"

    def progress(message):
        stamp(message)
        with log.open("a") as stream:
            from .._native.common import now

            stream.write(f"[{now()}] {message}\n")

    progress(
        f"START reference {'profiling' if _profile else 'replay'} result={out / 'report.json'} log={log} budget={settings['wall_time_seconds']}s"
    )
    summary = {
        "kind": "application_profile" if _profile else "application_replay",
        "status": "incomplete",
        "llm_calls": 0,
        "optimization_performed": False,
        "comparison": "exact observed fingerprints",
        "scope": "Finite repeatability of the original application on declared cases. No optimization, acceptance, hidden-test claim, or universal correctness proof. Timings include process/setup/observation overhead and are not performance benchmarks.",
        "cases": [],
        "split": split,
        "environment": info["environment"],
    }
    snapshot = info["source"]
    if _profile:
        summary["measurement_modes"] = _profile["modes"]
    pending_attempt = None
    try:
        frozen = out / "reference"
        copy_project(root, frozen, snapshot)
        checked = inventory(
            root,
            value["project"]["exclude"],
            value["project"]["limits"],
            value["project"]["include"],
        )
        if checked["files"] != snapshot["files"]:
            raise ValueError("project files changed while freezing; start a new replay")
        write(out / "resolved.json", value)
        write(out / "cases.json", cases)
        write(out / "source.json", snapshot)
        manifest = {
            "schema_version": 1,
            "files": {
                name: digest(out / name) for name in ("resolved.json", "cases.json", "source.json")
            },
            "source_identity": identity(snapshot["files"]),
            "original_config_sha256": original_config_digest,
            "original_cases_sha256": original_cases_digest,
            "runner": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        }
        write(out / "manifest.json", manifest)
        env = {
            k: v
            for k, v in os.environ.items()
            if k in {"PATH", "LANG", "SYSTEMROOT", *value["environment"]["inherit_env"]}
        }
        missing = [n for n in value["environment"]["inherit_env"] if n not in os.environ]
        if missing:
            raise ValueError(
                "missing explicitly requested environment variables: " + ", ".join(missing)
            )
        summary["environment_names"] = sorted(env)
        failed, unstable = False, False
        for index, case in enumerate(cases):
            row = {"id": case["id"], "split": case["split"], "attempts": []}
            summary["cases"].append(row)
            for repeat in range(settings["repeats"]):
                left = settings["wall_time_seconds"] - (clock() - started)
                if left <= 0:
                    failed = True
                    summary["reason"] = "total replay deadline exhausted"
                    break
                folder = out / "attempts" / f"{case['id']}_{repeat + 1:02d}"
                folder.mkdir(parents=True)
                pending_attempt = folder
                progress(
                    f"REPLAY case={index + 1}/{len(cases)} id={case['id']} repeat={repeat + 1}/{settings['repeats']}"
                )
                with tempfile.TemporaryDirectory(prefix="workspace-", dir=folder) as temporary:
                    workspace = Path(temporary) / "project"
                    copy_project(frozen, workspace, snapshot)
                    home = Path(temporary) / "home"
                    home.mkdir()
                    worker_env = {
                        **env,
                        "HOME": str(home),
                        "TMPDIR": str(Path(temporary)) + "/",
                        "PYTHONUNBUFFERED": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONHASHSEED": str(settings["seed"]),
                    }
                    entry = value["entry"]
                    if entry["kind"] == "callable":
                        payload = {
                            "workspace": str(workspace),
                            "entry": entry,
                            "case": case,
                            "seed": settings["seed"],
                            "max_output_bytes": settings["max_output_bytes"],
                            "result": str(folder / "worker_result.json"),
                        }
                        worker_name = "worker.py"
                        if _profile:
                            payload["profile_mode"] = _profile["modes"][repeat]
                            worker_name = "profile_worker.py"
                        write(folder / "payload.json", payload)
                        argv = [
                            value["environment"]["python"],
                            "-s",
                            str(Path(__file__).with_name(worker_name)),
                            str(folder / "payload.json"),
                        ]
                    else:
                        argv = [
                            value["environment"]["python"] if arg == "{python}" else arg
                            for arg in entry["argv"]
                        ] + case["argv"]
                    left = settings["wall_time_seconds"] - (clock() - started)
                    if left <= 0:
                        result = {
                            "status": "timeout",
                            "reason": "budget exhausted during workspace setup",
                        }
                    else:
                        result = execute(
                            argv,
                            cwd=workspace,
                            env=worker_env,
                            stdin=case.get("stdin", ""),
                            folder=folder,
                            timeout=min(settings["timeout_seconds"], left),
                            max_output_bytes=settings["max_output_bytes"],
                            progress=progress,
                        )
                    if result["status"] == "completed":
                        if entry["kind"] == "callable":
                            details = read_json(folder / "worker_result.json")
                            if details.get("status") != "completed" or len(
                                details.get("calls", [])
                            ) != len(case["calls"]):
                                raise ValueError(
                                    "reference worker returned incomplete observations"
                                )
                            observed = [c["observation"] for c in details["calls"]]
                            if _profile:
                                # Full observations remain in worker_result.json. Avoid
                                # duplicating large input trees in every profile receipt/report.
                                details["calls"] = [
                                    {k: v for k, v in call.items() if k != "observation"}
                                    for call in details["calls"]
                                ]
                            result["details"] = details
                        else:
                            observed = {"returncode": result["returncode"], "files": {}}
                            if entry["observe_stdout"]:
                                observed["stdout_sha256"] = digest(folder / "stdout.log")
                            for name in entry["outputs"]:
                                path = workspace / name
                                if (
                                    not path.is_file()
                                    or path.is_symlink()
                                    or not path.resolve().is_relative_to(workspace)
                                ):
                                    raise ValueError(
                                        f"declared output missing or outside workspace: {name}"
                                    )
                                if path.stat().st_size > settings["max_output_bytes"]:
                                    raise ValueError(
                                        f"declared output exceeds observation size limit: {name}"
                                    )
                                observed["files"][name] = {
                                    "bytes": path.stat().st_size,
                                    "sha256": digest(path),
                                }
                            result["observation"] = observed
                        result["fingerprint"] = identity(observed)
                    if clock() - started > settings["wall_time_seconds"]:
                        result["status"] = "timeout"
                        result.pop("fingerprint", None)
                    result["directory"] = str(folder.relative_to(out))
                    if result["status"] != "completed" and (folder / "worker_result.json").exists():
                        try:
                            details = read_json(folder / "worker_result.json")
                            result["error"] = details.get("error")
                            result["message"] = details.get("message")
                        except (OSError, ValueError):
                            pass
                    row["attempts"].append(result)
                    write(folder / "receipt.json", result)
                    pending_attempt = None
                if result["status"] != "completed":
                    failed = True
                    summary["reason"] = (
                        f"reference {result['status']}: {case['id']}; see attempt logs"
                    )
                    break
            row["repeatable"] = (
                len(row["attempts"]) == settings["repeats"]
                and all(a["status"] == "completed" for a in row["attempts"])
                and len({a.get("fingerprint") for a in row["attempts"]}) == 1
            )
            unstable = unstable or not row["repeatable"]
            if failed:
                break
        summary["status"] = "failed" if failed else "reference_unstable" if unstable else "replayed"
        summary["source_identity"] = manifest["source_identity"]
        summary["source_unchanged"] = (
            inventory(
                root,
                value["project"]["exclude"],
                value["project"]["limits"],
                value["project"]["include"],
            )["files"]
            == snapshot["files"]
        )
        if not summary["source_unchanged"]:
            summary.update(
                status="failed",
                reason="original project changed during replay; no repeatability conclusion",
            )
        if (
            digest(config) != original_config_digest
            or digest(value["cases"]) != original_cases_digest
        ):
            summary.update(
                status="failed",
                reason="application job/cases changed during replay; start a new run",
            )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        summary.update(status="failed", reason=str(error))
        if pending_attempt is not None:
            receipt = {
                "status": "failed",
                "reason": str(error),
                "directory": str(pending_attempt.relative_to(out)),
            }
            row["attempts"].append(receipt)
            row["repeatable"] = False
            write(pending_attempt / "receipt.json", receipt)
    except KeyboardInterrupt:
        summary.update(status="incomplete", reason="interrupted")
    finally:
        summary["elapsed_seconds"] = clock() - started
        write(out / "report.json", summary)
        (out / "report.html").write_text(
            "<!doctype html><meta charset='utf-8'><title>Lossless reference replay</title><h1>Reference replay: "
            + html.escape(summary["status"])
            + "</h1><p>"
            + html.escape(summary["scope"])
            + "</p><pre>"
            + html.escape(json.dumps(summary, indent=2))
            + "</pre>\n"
        )
        progress(f"COMPLETE status={summary['status']} result={out / 'report.json'} log={log}")
    return ReplayResult(out, summary)
