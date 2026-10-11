"""Model-free integration test on the pinned external pybaselines project.

Fresh fixtures, full source composition, discovery-only selection and one final
evaluation through the public application search API. No upstream files change.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import lossless
from lossless.applications import initialize, retained_plan, search
from lossless._native.common import write


def stamp(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def prepare(project, output):
    output.mkdir(parents=True, exist_ok=False)
    prepared = output / "project"
    prepared.mkdir()
    shutil.copytree(
        project / "pybaselines",
        prepared / "pybaselines",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    resources = Path(lossless.__file__).parent / "retained"
    shutil.copyfile(resources / "spectrum_workload.py", prepared / "spectrum_workload.py")
    plan = retained_plan(prepared)
    fixtures = prepared / "lossless_fixtures"
    fixtures.mkdir()
    cases = []
    # Retired research seeds/shapes are not used here. Freeze both splits before
    # profiling or measuring any candidate. New y/weights in every call.
    specs = [
        ("dev_small", "discovery", 383, 2, False, False),
        ("dev_batch", "discovery", 1151, 16, False, False),
        ("new_small", "evaluation", 467, 1, False, False),
        ("new_weighted", "evaluation", 1879, 6, True, False),
        ("new_branches", "evaluation", 887, 4, False, True),
    ]
    for i, (name, split, size, batch, weighted, branch) in enumerate(specs):
        calls = []
        for call in range(3):
            records = []
            for j in range(batch):
                rng = np.random.default_rng(99107771 + i * 10000 + call * 100 + j)
                x = np.linspace(-2, 3, size)
                t = (x + 2) / 5
                y = 0.7 + 1.8 * np.exp(-2.4 * t) + 0.2 * t**2
                y += rng.uniform(0.5, 3) * np.exp(-(((t - rng.uniform(0.2, 0.8)) / 0.07) ** 2))
                y += rng.normal(0, 0.015, size)
                options = dict(
                    poly_order=([1, 4, 2, 3][j % 4] if branch else 3),
                    max_iter=150,
                    return_coef=True,
                )
                if branch:
                    options.update(use_original=True, mask_initial_peaks=True)
                    permutation = np.random.default_rng(55092).permutation(size)
                    x, y = x[permutation], y[permutation]
                record = {"options": options}
                arrays = dict(x=x, y=y)
                if weighted:
                    arrays["weights"] = rng.uniform(0.15, 1, size)
                for label, array in arrays.items():
                    path = fixtures / f"{name}_{call}_{j}_{label}.npy"
                    np.save(path, array, allow_pickle=False)
                    ref = {"$npy": path.relative_to(prepared).as_posix()}
                    (options if label == "weights" else record)[label] = ref
                records.append(record)
            calls.append(dict(args=[records]))
        cases.append(dict(id=name, split=split, calls=calls))
    write(output / "cases.json", cases)
    initialize(
        prepared,
        output / "job",
        entry="spectrum_workload:make_processor",
        factory=True,
        cases=output / "cases.json",
        python=sys.executable,
    )
    config_path = output / "job/lossless.json"
    config = json.loads(config_path.read_text())
    config["environment"]["inherit_env"] = [
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]
    config["entry"]["observe"] = "spectrum_workload:observe_processor"
    config["replay"].update(timeout_seconds=25, max_output_bytes=32000000)
    write(config_path, config)
    return config_path, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=280)
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    out = args.output.resolve()
    stamp(f"START estimated=under5min result={out}/search/report.json log={out}.log")
    config, plan = prepare(args.project.resolve(), out)
    plan.update(
        wall_time_seconds=args.budget,
        final_reserve_seconds=90,
        discovery_pairs=2,
        evaluation_pairs=6,
    )
    write(out / "plan.json", plan)
    result = search(config, plan=plan, output=out / "search")
    stamp(
        f"COMPLETE status={result.summary['status']} selected={result.summary['selected']} result={out}/search/report.json log={out}.log"
    )
    if result.summary["status"] not in {"accepted_experiment", "reference_retained"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
