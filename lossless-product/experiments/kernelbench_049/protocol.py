"""Frozen numerical contract, scope checks and statistics for campaign 049."""

import ast
import hashlib
import json
import math
from pathlib import Path
import statistics

ATOL = RTOL = 1e-4
METHODS = ["eager", "compiled", "shapeless"]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_check(source):
    """Prevents accidental scope escapes; not a security boundary for hostile code."""
    if not isinstance(source, str) or not 0 < len(source.encode()) <= 160000:
        raise ValueError("Expected a bounded complete Python module")
    tree = ast.parse(source)
    allowed = {"mlx", "math", "functools", "collections", "itertools", "typing"}
    banned = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "input"}
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [n.name for n in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("relative import outside scope")
            names = [node.module or ""]
        if any(n.split(".")[0] not in allowed for n in names):
            raise ValueError(f"import outside scope: {names}")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in banned
        ):
            raise ValueError(f"call outside scope: {node.func.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("runtime introspection outside scope")
    if not any(isinstance(n, ast.ClassDef) and n.name == "Runner" for n in tree.body):
        raise ValueError("Runner(task, config, parameters, reference).run(*inputs) is required")
    return hashlib.sha256(source.encode()).hexdigest()


def assess(records):
    from lossless._native.common import interval

    rows = []
    for r in records:
        valid = r.get("checks", {}).get("candidate", {}).get("pass", False)
        valid = valid and r.get("checks", {}).get("reference", {}).get("pass", False)
        a, b = r.get("samples", {}).get("reference", []), r.get("samples", {}).get("candidate", [])
        valid = valid and bool(a and b)
        ratio = statistics.median(a) / statistics.median(b) if valid else None
        ci = interval(a, b, 4901) if valid else None
        memory = (
            bool(valid)
            and max(r["peaks"]["candidate"]) <= max(r["peaks"]["reference"]) * 1.1 + 16 * 1024**2
        )
        rows.append(
            dict(
                id=r["case"]["id"],
                task=r["case"]["task"],
                correct=bool(valid),
                ratio=ratio,
                ci95=ci,
                memory_pass=memory,
            )
        )
    eligible = len(rows) == 40 and all(r["correct"] and r["memory_pass"] for r in rows)
    geom = statistics.geometric_mean(r["ratio"] for r in rows) if eligible else None
    per_task = []
    for task in dict.fromkeys(r["task"] for r in rows):
        rr = [r for r in rows if r["task"] == task]
        ok = len(rr) == 2 and all(r["correct"] and r["memory_pass"] for r in rr)
        mean = math.prod(r["ratio"] for r in rr) ** 0.5 if ok else None
        per_task.append(
            dict(
                task=task,
                correct=ok,
                geomean=mean,
                confirmed=ok and mean >= 1.02 and all(r["ci95"][0] > 1 for r in rr),
            )
        )
    return dict(
        eligible=eligible,
        geomean=geom,
        cases=rows,
        tasks=per_task,
        confirmed=eligible and geom >= 1.02 and all(r["ci95"][0] > 1 for r in rows),
    )
