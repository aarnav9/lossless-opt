"""Versioned application intake jobs; resolution never imports the project."""

import copy
import os
from pathlib import Path
import re
import sys

from ..jobs import fields, positive, read_json
from .._native.common import write
from .discovery import DEFAULT_LIMITS, discover, environment, inventory


def relative(value, label):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a nonempty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value == ".":
        raise ValueError(f"{label} must stay inside the project/workspace")
    return path.as_posix()


def target(value):
    if not isinstance(value, str) or value.count(":") != 1:
        raise ValueError("entry target must be file.py:function or package.module:function")
    module, symbol = value.split(":")
    if not all(part.isidentifier() for part in symbol.split(".")):
        raise ValueError("entry symbol must be a Python identifier or attribute path")
    if module.endswith(".py"):
        relative(module, "entry file")
    elif not all(p.isidentifier() for p in module.split(".")):
        raise ValueError("entry module must be a dotted Python module name")
    return value


def check_entry_file(value, snapshot):
    module = value.split(":")[0]
    names = (
        [module]
        if module.endswith(".py")
        else [
            prefix + module.replace(".", "/") + suffix
            for prefix in ("", "src/")
            for suffix in (".py", "/__init__.py")
        ]
    )
    if not any(name in snapshot["files"] for name in names):
        raise ValueError(f"entry file/module is absent or excluded from the snapshot: {module}")


def string_list(value, label, empty=True):
    if (
        not isinstance(value, list)
        or (not value and not empty)
        or not all(isinstance(s, str) and s and "\x00" not in s for s in value)
    ):
        raise ValueError(f"{label} must be a list of nonempty strings")
    return value


def fixture_names(value):
    if isinstance(value, dict):
        if "$npy" in value:
            if set(value) != {"$npy"}:
                raise ValueError("array fixture must be exactly a $npy path object")
            yield relative(value["$npy"], "array fixture")
        else:
            for child in value.values():
                yield from fixture_names(child)
    elif isinstance(value, list):
        for child in value:
            yield from fixture_names(child)


def validate_cases(value, entry):
    if not isinstance(value, list) or len(value) > 1000:
        raise ValueError("cases must be a list with at most 1000 cases")
    seen, result = set(), []
    for row in value:
        fields(
            row,
            {"id", "split", "args", "kwargs", "calls", "argv", "stdin"},
            "application case",
            {"id", "split"},
        )
        if (
            not isinstance(row["id"], str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", row["id"])
            or row["id"] in seen
        ):
            raise ValueError("case IDs must be unique lowercase identifiers")
        seen.add(row["id"])
        if row["split"] not in {"discovery", "evaluation"}:
            raise ValueError("each case must declare discovery or evaluation")
        row = copy.deepcopy(row)
        if entry and entry["kind"] == "command":
            if any(k in row for k in {"args", "kwargs", "calls"}):
                raise ValueError("command cases use argv/stdin, not callable args")
            string_list(row.setdefault("argv", []), "case.argv")
            if not isinstance(row.setdefault("stdin", ""), str):
                raise ValueError("command case.stdin must be a string")
        elif entry:
            if any(k in row for k in {"argv", "stdin"}):
                raise ValueError("callable cases use args/kwargs or calls")
            if "calls" in row and any(k in row for k in {"args", "kwargs"}):
                raise ValueError("use calls or args/kwargs, not both")
            calls = row.get(
                "calls", [{"args": row.pop("args", []), "kwargs": row.pop("kwargs", {})}]
            )
            if not isinstance(calls, list) or not 1 <= len(calls) <= 100:
                raise ValueError("case.calls must contain 1..100 invocations")
            for call in calls:
                fields(call, {"args", "kwargs"}, "call")
                if not isinstance(call.setdefault("args", []), list) or not isinstance(
                    call.setdefault("kwargs", {}), dict
                ):
                    raise ValueError("call args must be a list and kwargs must be an object")
            row["calls"] = calls
        result.append(row)
    return result


def analysis_settings(value):
    if value is None:
        return None
    fields(value, {"provider", "model", "effort", "timeout_seconds"}, "analysis", {"provider"})
    if value["provider"] != "codex-chatgpt":
        raise ValueError("application assessment currently supports provider codex-chatgpt")
    value = copy.deepcopy(value)
    value.setdefault("model", "gpt-6-astra")
    value.setdefault("effort", "medium")
    value.setdefault("timeout_seconds", 1800)
    if not isinstance(value["model"], str) or not value["model"].strip():
        raise ValueError("analysis.model must be a nonempty model name")
    if value["effort"] not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError("unsupported analysis.effort")
    positive(value["timeout_seconds"], "analysis.timeout_seconds")
    return value


def resolve(config):
    config = Path(config).resolve()
    value = read_json(config)
    fields(
        value,
        {
            "schema_version",
            "kind",
            "name",
            "project",
            "environment",
            "entry",
            "cases",
            "replay",
            "profile",
            "analysis",
        },
        "application job",
        {"schema_version", "kind", "project", "environment", "entry", "cases", "replay"},
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["kind"] != "application"
    ):
        raise ValueError("expected application job schema_version 1")
    project = value["project"]
    fields(project, {"root", "exclude", "include", "limits"}, "project", {"root"})
    if not isinstance(project["root"], str) or not project["root"]:
        raise ValueError("project.root must be a directory path")
    project["root"] = str((config.parent / project["root"]).resolve())
    string_list(project.setdefault("exclude", []), "project.exclude")
    string_list(project.setdefault("include", []), "project.include")
    limits = project.setdefault("limits", dict(DEFAULT_LIMITS))
    fields(limits, DEFAULT_LIMITS, "project.limits", DEFAULT_LIMITS)
    for name, number in limits.items():
        positive(number, name, True)
    env = value["environment"]
    fields(env, {"python", "inherit_env"}, "environment", {"python"})
    if not isinstance(env["python"], str) or not Path(env["python"]).is_absolute():
        raise ValueError("environment.python must be the absolute selected interpreter path")
    string_list(env.setdefault("inherit_env", []), "environment.inherit_env")
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n) for n in env["inherit_env"]):
        raise ValueError("inherit_env must contain environment variable names")
    if set(env["inherit_env"]) & {"PYTHONPATH", "PYTHONHOME"}:
        raise ValueError(
            "PYTHONPATH/PYTHONHOME overrides are incompatible with copied-project replay"
        )
    entry = value["entry"]
    if entry is not None:
        if not isinstance(entry, dict):
            raise ValueError("entry must be an object or null for an unfinished setup")
        if entry.get("kind") == "callable":
            fields(
                entry,
                {
                    "kind",
                    "target",
                    "factory",
                    "factory_args",
                    "factory_kwargs",
                    "observe",
                    "cleanup",
                    "synchronize",
                },
                "entry",
                {"kind", "target"},
            )
            target(entry["target"])
            if type(entry.setdefault("factory", False)) is not bool:
                raise ValueError("entry.factory must be boolean")
            if not isinstance(entry.setdefault("factory_args", []), list) or not isinstance(
                entry.setdefault("factory_kwargs", {}), dict
            ):
                raise ValueError("factory_args must be a list and factory_kwargs must be an object")
            if not entry["factory"] and (entry["factory_args"] or entry["factory_kwargs"]):
                raise ValueError("factory arguments require entry.factory=true")
            for hook in ("observe", "cleanup", "synchronize"):
                if entry.get(hook) is not None:
                    target(entry[hook])
        elif entry.get("kind") == "command":
            fields(entry, {"kind", "argv", "outputs", "observe_stdout"}, "entry", {"kind", "argv"})
            string_list(entry["argv"], "entry.argv", False)
            if entry["argv"][0] in {"python", "python3"}:
                entry["argv"][0] = "{python}"
            for name in string_list(entry.setdefault("outputs", []), "entry.outputs"):
                relative(name, "observed output file")
            if type(entry.setdefault("observe_stdout", True)) is not bool:
                raise ValueError("entry.observe_stdout must be boolean")
            if not entry["observe_stdout"] and not entry["outputs"]:
                raise ValueError(
                    "command replay requires stdout or declared output files to observe"
                )
        else:
            raise ValueError("entry.kind must be callable or command")
    settings = value["replay"]
    fields(
        settings,
        {"repeats", "timeout_seconds", "wall_time_seconds", "seed", "max_output_bytes"},
        "replay",
    )
    defaults = {
        "repeats": 2,
        "timeout_seconds": 30,
        "wall_time_seconds": 120,
        "seed": 0,
        "max_output_bytes": 8 * 2**20,
    }
    for key, default in defaults.items():
        settings.setdefault(key, default)
    for key in ("timeout_seconds", "wall_time_seconds", "max_output_bytes"):
        positive(settings[key], f"replay.{key}", key == "max_output_bytes")
    if type(settings["repeats"]) is not int or not 2 <= settings["repeats"] <= 20:
        raise ValueError("replay.repeats must be 2..20")
    if type(settings["seed"]) is not int or not 0 <= settings["seed"] < 2**32:
        raise ValueError("replay.seed must be a uint32 integer")
    profiling = value.setdefault("profile", {})
    fields(profiling, {"repeats", "wall_time_seconds"}, "profile")
    profiling.setdefault("repeats", 5)
    profiling.setdefault("wall_time_seconds", 1800)
    if type(profiling["repeats"]) is not int or not 3 <= profiling["repeats"] <= 50:
        raise ValueError("profile.repeats must be 3..50")
    positive(profiling["wall_time_seconds"], "profile.wall_time_seconds")
    value["analysis"] = analysis_settings(value.get("analysis"))
    if not isinstance(value["cases"], str) or not value["cases"]:
        raise ValueError("cases must name a JSON file relative to the job")
    cases_path = (config.parent / value["cases"]).resolve()
    cases = validate_cases(read_json(cases_path), entry)
    project["include"] = sorted(
        set(project["include"]) | set(fixture_names(cases)) | set(fixture_names(entry))
    )
    value["cases"] = str(cases_path)
    return value, cases


def inspect(config):
    value, cases = resolve(config)
    project = value["project"]
    snapshot = inventory(project["root"], project["exclude"], project["limits"], project["include"])
    discovery = discover(project["root"], snapshot)
    missing = []
    if value["entry"] is None:
        missing.append(
            "Choose entry.target from the discovered functions, or provide a command argv list."
        )
    if not any(c["split"] == "discovery" for c in cases):
        missing.append(
            "Add representative discovery inputs to cases.json; inputs are not invented."
        )
    entry = value["entry"]
    if entry and entry["kind"] == "callable":
        for key in ("target", "observe", "cleanup", "synchronize"):
            if entry.get(key):
                check_entry_file(entry[key], snapshot)
    return {
        "status": "needs_input" if missing else "ready_to_replay",
        "missing": missing,
        "resolved": value,
        "cases": {s: sum(c["split"] == s for c in cases) for s in ("discovery", "evaluation")},
        "environment": environment(value["environment"]["python"]),
        "source": snapshot,
        "discovery": discovery,
        "preflight": {
            "action": "reference replay; no optimization or LLM calls",
            "total_seconds": value["replay"]["wall_time_seconds"],
            "per_process_seconds": value["replay"]["timeout_seconds"],
            "repeats": value["replay"]["repeats"],
            "baseline": "the supplied application",
            "comparison": "exact observed replay fingerprints; repeatability only",
            "execution": "trusted local code in fresh copied workspaces; not an OS security sandbox",
        },
    }


def preflight_text(info):
    value, env = info["resolved"], info["environment"]
    entry = value["entry"]
    selected = (
        entry["target"] + (" (factory)" if entry.get("factory") else "")
        if entry and entry["kind"] == "callable"
        else repr(entry["argv"])
        if entry
        else "not selected"
    )
    settings = value["replay"]
    lines = [
        f"Application: {value.get('name', Path(value['project']['root']).name)}",
        f"Status: {info['status']}",
        f"Project: {value['project']['root']}",
        f"Invocation: {selected}",
        f"Python: {value['environment']['python']} ({env['python'].split()[0]})",
        f"Host: {env['platform']}",
        f"Source: {len(info['source']['files'])} files, {info['source']['bytes'] / 2**20:.2f} MiB",
        f"Cases: {info['cases']['discovery']} discovery; {info['cases']['evaluation']} evaluation (not replayed by default)",
        f"Limits: {settings['wall_time_seconds']:g}s total; {settings['timeout_seconds']:g}s per process; {settings['repeats']} repeats",
        "Reference: your existing application",
        "Observation: exact output/input fingerprints; repeatability only",
        "LLM calls: 0; no author response allowance applies",
        "Execution: fresh processes/copied workspaces; trusted local code",
        f"Profiling: {value['profile']['repeats']} fresh timing repeats + separate attribution/memory; {value['profile']['wall_time_seconds']:g}s total",
        "Assessment: "
        + (
            f"{value['analysis']['model']} / {value['analysis']['effort']}; up to 1 Codex call, clipped to profile budget"
            if value.get("analysis")
            else "disabled; enable explicitly to share bounded discovery evidence with Codex"
        ),
        "Experimental source search: lossless search-application JOB --plan PLAN --output RUN; guarded export unavailable",
    ]
    if info["missing"]:
        lines += ["", "Setup needed:", *["- " + item for item in info["missing"]]]
        lines += ["- Candidate: " + c["target"] for c in info["discovery"]["entry_candidates"][:10]]
    return "\n".join(lines)


def initialize(
    project,
    output,
    *,
    entry=None,
    factory=False,
    command=None,
    cases=None,
    python=None,
    exclude=(),
    include=(),
):
    project, output = Path(project).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("output directory already exists; choose a new application job directory")
    if entry and command:
        raise ValueError("choose an entry point or a command, not both")
    if factory and not entry:
        raise ValueError("--factory requires --entry")
    patterns = list(exclude)
    if output.is_relative_to(project):
        patterns.append(output.relative_to(project).as_posix())
    inputs = (
        read_json(Path(cases).resolve())
        if cases
        else (
            [{"id": "invocation", "split": "discovery", "argv": [], "stdin": ""}] if command else []
        )
    )
    included = sorted(set(include) | set(fixture_names(inputs)))
    snapshot = inventory(project, patterns, include=included)
    found = discover(project, snapshot)
    detected = environment(python)
    selected = None
    if entry:
        selected = {"kind": "callable", "target": target(entry), "factory": factory}
        check_entry_file(entry, snapshot)
    elif command:
        selected = {
            "kind": "command",
            "argv": string_list(command, "command", False),
            "outputs": [],
            "observe_stdout": True,
        }
    validate_cases(inputs, selected)
    value = {
        "schema_version": 1,
        "kind": "application",
        "name": project.name,
        "project": {
            "root": os.path.relpath(project, output),
            "exclude": patterns,
            "include": included,
            "limits": dict(DEFAULT_LIMITS),
        },
        "environment": {"python": os.path.abspath(python or sys.executable), "inherit_env": []},
        "entry": selected,
        "cases": "cases.json",
        "profile": {"repeats": 5, "wall_time_seconds": 1800},
        "analysis": None,
        "replay": {
            "repeats": 2,
            "timeout_seconds": 30,
            "wall_time_seconds": 120,
            "seed": 0,
            "max_output_bytes": 8 * 2**20,
        },
    }
    output.mkdir(parents=True)
    write(output / "lossless.json", value)
    write(output / "cases.json", inputs)
    write(output / "environment.json", detected)
    write(output / "discovery.json", {**found, "source": snapshot})
    (output / ".gitignore").write_text("runs/\n.env\n*.local.json\n")
    return {
        "directory": str(output),
        "config": str(output / "lossless.json"),
        "status": "ready_to_replay"
        if selected and any(c["split"] == "discovery" for c in inputs)
        else "needs_input",
        "next_step": f"lossless inspect {output / 'lossless.json'}",
        "entry_candidates": found["entry_candidates"][:12],
        "additional_candidates": max(0, len(found["entry_candidates"]) - 12),
        "discovery": str(output / "discovery.json"),
        "scope": "Project inspected without import or execution. Review the invocation and cases before explicit replay.",
    }
