"""Frozen, finite exactness and whole-request performance protocol for campaign 048."""

import ast
import hashlib
import json
import math
from pathlib import Path
import statistics

METHODS = ["serial", "stock_batch", "retained", "decoder_026", "fullhead_027"]


def cases(split):
    e = int(split == "evaluation")
    word = " forest" if e else " river"
    profiles = [
        ("single_short", [12 + 3 * e], [6 + e]),
        ("cohort_short", [16 + 3 * e] * 4, [1 + e, 3 + e, 5 + e, 8 + e]),
        ("cohort_long", [96 + 11 * e] * 4, [3 + e, 6 + e, 9 + e, 12 + e]),
        ("cohort_eight", [32 + 5 * e] * 8, [n + e for n in [1, 2, 4, 6, 3, 5, 7, 9]]),
        ("ragged", [n + 7 * e for n in [9, 31, 75, 143]], [n + e for n in [3, 5, 7, 9]]),
        (
            "two_cohorts",
            [16 + 3 * e] * 4 + [64 + 9 * e] * 4,
            [n + e for n in [2, 4, 6, 8, 3, 5, 7, 9]],
        ),
        ("long_single", [256 + 31 * e], [12 + 2 * e]),
    ]
    result = [
        dict(id=f"{split}_{key}", key=key, texts=[word * n for n in ns], counts=cs)
        for key, ns, cs in profiles
    ]
    natural = (
        [
            "Explain why the ocean has tides.",
            "Write Python code to remove duplicate strings while preserving their order.",
            "Translate to German: The train leaves tomorrow morning. Keep your ticket ready.",
            "A library shelves half of its 24 history books and one fifth of its 35 science books. How many remain?",
        ]
        if e
        else [
            "Explain why rainbows form.",
            "Write Python code to sum the even numbers in a list.",
            "Translate to French: The meeting moved to Thursday. Bring the revised report.",
            "A shop sells one third of its 18 red notebooks and one third of its 27 blue notebooks. How many remain?",
        ]
    )
    result.append(
        dict(
            id=f"{split}_natural",
            key="natural",
            texts=natural,
            counts=[n + e for n in [3, 5, 7, 9]],
        )
    )
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_check(source):
    """Accidental-scope guard, NOT a sandbox for hostile Python."""
    if not isinstance(source, str) or not 0 < len(source.encode()) <= 160000:
        raise ValueError("bounded complete Python source required")
    tree = ast.parse(source)
    roots = {
        "mlx",
        "mlx_lm",
        "lossless",
        "math",
        "time",
        "collections",
        "contextlib",
        "functools",
        "itertools",
        "typing",
        "copy",
    }
    prohibited = {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "breakpoint",
        "input",
        "globals",
        "locals",
        "vars",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [n.name for n in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("relative imports are outside candidate scope")
            names = [node.module or ""]
        else:
            names = []
        for name in names:
            if name.split(".")[0] not in roots:
                raise ValueError(f"import outside candidate scope: {name}")
            if name.startswith("lossless") and not name.startswith("lossless.adapters.mlx"):
                raise ValueError("only MLX implementation modules may be imported")
            if name.startswith("lossless.adapters.mlx") and any(
                part in name.split(".")
                for part in ["checks", "validation", "worker", "identity", "job"]
            ):
                raise ValueError("evaluator/identity modules are outside candidate scope")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in prohibited
        ):
            raise ValueError(f"call outside candidate scope: {node.func.id}")
        if isinstance(node, ast.Attribute) and node.attr in {
            "__dict__",
            "__globals__",
            "__subclasses__",
        }:
            raise ValueError("runtime introspection is outside candidate scope")
    if not any(isinstance(n, ast.ClassDef) and n.name == "Runner" for n in tree.body):
        raise ValueError("source must define Runner(model).run(prompts, counts, capture=True)")
    return hashlib.sha256(source.encode()).hexdigest()


def assess(records):
    from lossless._native.common import interval

    rows = []
    for record in records:
        samples = record.get("samples", {})
        valid = record.get("exact", {}).get("candidate", False)
        valid = valid and record.get("exact", {}).get("reference", False)
        if not valid or not samples.get("candidate") or not samples.get("reference"):
            return {
                "eligible": False,
                "confirmed": False,
                "geomean": None,
                "reason": "incorrect/incomplete candidate or reference",
                "cases": rows,
            }
        a, b = samples["reference"], samples["candidate"]
        ratio = statistics.median(a) / statistics.median(b)
        ci = interval(a, b, 4801)
        memory = (
            max(record["peaks"]["candidate"])
            <= max(record["peaks"]["reference"]) * 1.10 + 16 * 1024**2
        )
        # Latency is backend-reported, range checked by the fixed wrapper.
        latency = (
            max(record["ttft"]["candidate"]) <= max(record["ttft"]["reference"]) * 1.10 + 0.005
        )
        rows.append(
            dict(
                key=record["case"]["key"],
                ratio=ratio,
                ci95=ci,
                memory_pass=memory,
                reported_ttft_pass=latency,
            )
        )
    geom = math.exp(statistics.mean(math.log(r["ratio"]) for r in rows)) if rows else None
    eligible = bool(rows) and all(r["memory_pass"] and r["reported_ttft_pass"] for r in rows)
    return dict(
        eligible=eligible,
        confirmed=eligible and geom >= 1.02 and all(r["ci95"][0] > 1 for r in rows),
        geomean=geom,
        cases=rows,
    )


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())
