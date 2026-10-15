"""Reproduce one bounded local CPU workflow on a supplied external checkout.

Run each project separately. The default total workflow budget is 280 seconds;
all stages print timestamped progress. No providers or model calls are enabled.
"""

import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys

import numpy as np

from lossless._native.common import stamp, write
from lossless.applications import optimize_project


def copy(source, destination):
    shutil.copytree(
        source, destination, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc")
    )


def prepare(kind, upstream, out, seed=12062026, matrix=False):
    project = out / "prepared/project"
    if kind == "pybaselines":
        from check_retained_application import prepare as prepare_spectra

        prepare_spectra(upstream, out / "prepared")
        return (
            project,
            out / "prepared/cases.json",
            dict(
                entry="spectrum_workload:make_processor",
                factory=True,
                observe="spectrum_workload:observe_processor",
            ),
        )
    project.mkdir(parents=True)
    if kind == "numpy-financial":
        copy(upstream / "numpy_financial", project / "numpy_financial")
        shutil.copy2(upstream / "LICENSE.txt", project / "LICENSE.txt")
        (project / "workload.py").write_text(
            "import numpy as np\nimport numpy_financial as npf\n"
            "def run(periods, payments, principals):\n"
            "    return np.asarray([npf.rate(n, p, v, 0) for n, p, v in zip(periods, payments, principals)])\n"
        )
    else:
        copy(upstream, project / "tmm")
        (project / "workload.py").write_text(
            "import numpy as np\nfrom tmm import coh_tmm\n"
            "def run(wavelengths):\n"
            "    result = []\n"
            "    for wavelength in wavelengths:\n"
            "        value = coh_tmm('s', [1, 2.2+0.01j, 1.5, 1], [np.inf, 100, 300, np.inf], 0.2, wavelength)\n"
            "        result.append([value['R'], value['T']])\n"
            "    return np.asarray(result)\n"
        )
    cases = []
    shapes = [("dev", "discovery", 80, 8), ("new", "evaluation", 97, 8)]
    if matrix and kind == "numpy-financial":
        shapes += [("narrow", "evaluation", 121, 1), ("wide", "evaluation", 61, 32)]
    for i, (name, split, size, width) in enumerate(shapes):
        calls = []
        for call in range(3):
            rng = np.random.default_rng(seed + i * 100 + call)
            if kind == "numpy-financial":
                n = rng.integers(12, 360, (size, width)).astype(float)
                v = rng.uniform(1000, 100000, (size, width))
                rate = rng.uniform(0.003, 0.03, (size, width))
                payment = -v * rate / (1 - (1 + rate) ** -n)
                arrays = [n, payment, v]
            else:
                arrays = [rng.uniform(400, 900, size)]
            args = []
            for j, array in enumerate(arrays):
                path = project / f"{name}_{call}_{j}.npy"
                np.save(path, array, allow_pickle=False)
                args.append({"$npy": path.name})
            calls.append(dict(args=args))
        cases.append(dict(id=name, split=split, calls=calls))
    path = out / "prepared/cases.json"
    write(path, cases)
    return project, path, dict(entry="workload.py:run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["pybaselines", "numpy-financial", "tmm"], required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=280)
    parser.add_argument("--seed", type=int, default=12062026)
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--prepared-inputs", choices=["fixtures", "values"], default="fixtures")
    parser.add_argument(
        "--matrix", action="store_true", help="Include held-out financial widths 1, 8 and 32"
    )
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stamp(f"START estimated=under5min result={out}/workflow/report.json log={out}.log")
    inherited = ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]
    for key in inherited:
        os.environ[key] = "1"
    upstream = args.project.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    write(
        out / "provenance.json",
        dict(
            kind=args.kind,
            upstream_revision=revision,
            budget=args.budget,
            seed=args.seed if args.kind != "pybaselines" else "retained_fixture_seed",
            prepared=args.prepared,
            matrix=args.matrix,
        ),
    )
    project, cases, options = prepare(args.kind, upstream, out, seed=args.seed, matrix=args.matrix)
    result = optimize_project(
        project,
        output=out / "workflow",
        cases=cases,
        python=sys.executable,
        inherit_env=inherited,
        budget=args.budget,
        prepared=args.prepared,
        prepared_inputs=args.prepared_inputs,
        **options,
    )
    stamp(f"DONE status={result.summary['status']} result={out}/workflow/report.json log={out}.log")
    if result.summary["status"] not in {"accepted", "reference_retained"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
