"""Freeze fresh fixtures while retaining the previously selected application code."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from lossless._native.common import write
from lossless.applications.bundle import admitted_signatures
from lossless.applications.discovery import inventory
from lossless.applications.config import initialize, inspect, resolve
from lossless.applications.workflow import make_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--financial", type=Path, required=True)
    p.add_argument("--pybaselines", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261014)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name, source in [("financial", a.financial), ("pybaselines", a.pybaselines)]:
        dest = out / name
        shutil.copytree(
            source, dest, ignore=shutil.ignore_patterns("prepared.json", "__pycache__", "*.pyc")
        )
        cases = json.loads((dest / "cases.json").read_text())
        if name == "financial":
            for i, case in enumerate(cases):
                for j, call in enumerate(case["calls"]):
                    shape = np.load(
                        dest / "reference" / call["args"][0]["$npy"], allow_pickle=False
                    ).shape
                    rng = np.random.default_rng(a.seed + i * 100 + j)
                    n = rng.integers(12, 360, shape).astype(float)
                    v = rng.uniform(1000, 100000, shape)
                    rate = rng.uniform(0.003, 0.03, shape)
                    payment = -v * rate / (1 - (1 + rate) ** -n)
                    for arg, value in zip(call["args"], [n, payment, v]):
                        for variant in ["reference", "optimized"]:
                            np.save(dest / variant / arg["$npy"], value, allow_pickle=False)
        else:
            for i, path in enumerate(
                sorted((dest / "reference/lossless_fixtures").glob("*_y.npy"))
            ):
                rng = np.random.default_rng(a.seed + i)
                values = np.load(path, allow_pickle=False)
                values = values + rng.normal(0, 0.0001, values.shape)
                relative = path.relative_to(dest / "reference")
                for variant in ["reference", "optimized"]:
                    np.save(dest / variant / relative, values, allow_pickle=False)
        manifest = json.loads((dest / "manifest.json").read_text())
        for variant in ["reference", "optimized"]:
            manifest[variant + "_files"] = inventory(dest / variant)["files"]
        manifest["input_signatures"] = admitted_signatures(cases, dest / "reference")
        manifest["qualification"] = "runtime_research_only_fresh_inputs"
        write(dest / "manifest.json", manifest)
    project = out / "array-project"
    project.mkdir()
    (project / "app.py").write_text("def run(x):\n    return x.copy(order='K')\n")
    cases = []
    for index, (name, shape, dtype, split) in enumerate(
        [
            ("small_float", (64,), np.float64, "discovery"),
            ("large_float", (512, 512), np.float64, "discovery"),
            ("large_int", (512, 512), np.int64, "discovery"),
            ("held_float", (513, 257), np.float64, "evaluation"),
            ("held_int", (513, 257), np.int32, "evaluation"),
        ]
    ):
        calls = []
        for i in range(2):
            rng = np.random.default_rng(a.seed + 1000 + index * 10 + i)
            v = (
                rng.uniform(-1, 1, shape).astype(dtype)
                if np.dtype(dtype).kind == "f"
                else rng.integers(-100, 100, shape, dtype=dtype)
            )
            path = project / f"{name}_{i}.npy"
            np.save(path, v, allow_pickle=False)
            calls.append(dict(args=[{"$npy": path.name}]))
        cases.append(dict(id=name, split=split, calls=calls))
    write(out / "array-cases.json", cases)
    initialize(project, out / "array-job", entry="app.py:run", cases=out / "array-cases.json")
    raw = json.loads((out / "array-job/lossless.json").read_text())
    raw["environment"]["inherit_env"] = [
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]
    write(out / "array-job/lossless.json", raw)
    value, cases = resolve(out / "array-job/lossless.json")
    source = inspect(out / "array-job/lossless.json")["source"]
    (out / "array-bundle").mkdir()
    bundle = make_bundle(
        out / "array-bundle",
        project,
        source,
        project,
        source,
        value,
        cases,
        {},
        "runtime_research_only_identical_source",
    )
    shutil.move(bundle, out / "arrays")
    write(
        out / "provenance.json",
        dict(
            seed=a.seed,
            source_bundles={"financial": str(a.financial), "pybaselines": str(a.pybaselines)},
            scope="Source implementations fixed; fresh Financial inputs and fresh noise in spectrum signals. Array copy is a mechanism control with identical original/selected source. No performance certificate.",
        ),
    )
    print(out, flush=True)


if __name__ == "__main__":
    main()
