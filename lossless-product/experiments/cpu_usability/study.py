"""Frozen external project onboarding trial; one foreground job per project.

No hand-authored candidate plans. Inputs and wrappers are frozen before search.
The report separates source improvements from amortized worker reuse.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from lossless._native.common import stamp, write
from lossless.applications import CallableApplication, optimize_project
from lossless.applications.worker import inputs, Observer
from lossless.jobs import identity


WRAPPERS = {
    "transforms3d": """import numpy as np
from transforms3d.quaternions import quat2axangle

def run(quaternions):
    axes, angles = zip(*[quat2axangle(q) for q in quaternions])
    return np.asarray(axes), np.asarray(angles)
""",
    "filterpy": """from filterpy.discrete_bayes import update, predict
import numpy as np

def run(likelihood, prior):
    posterior = prior.copy()
    kernel = np.array([.1, .8, .1])
    for _ in range(16):
        posterior = update(likelihood, predict(posterior, 1, kernel))
    return posterior
""",
    "perlin-numpy": """from perlin_numpy import generate_fractal_noise_2d

def run(canvas):
    return canvas + generate_fractal_noise_2d(canvas.shape, (4, 4), octaves=3)
""",
    "numpy-financial": """import numpy as np
import numpy_financial as npf

def run(periods, payments, principals):
    return np.asarray([npf.rate(n, p, v, 0) for n, p, v in zip(periods, payments, principals)])
""",
}
LOADER = """import numpy as np

def cases():
    rng = np.random.default_rng(202610145)
    rows = []
    for name, split, size in [('dev', 'discovery', SIZES[0]),
                              ('small', 'evaluation', SIZES[1]),
                              ('large', 'evaluation', SIZES[2])]:
        KIND_INPUT
        rows.append(dict(id=name, split=split, args=args))
    return rows
"""
INPUTS = {
    "transforms3d": ([512, 768, 1024], "args = [rng.normal(size=(size, 4))]"),
    "filterpy": (
        [1024, 2048, 8192],
        "prior = rng.uniform(.1, 1., size)\n        prior /= prior.sum()\n        args = [rng.uniform(.8, 1.2, size), prior]",
    ),
    "perlin-numpy": ([128, 256, 512], "args = [rng.normal(size=(size, size))]"),
    "numpy-financial": (
        [80, 97, 121],
        "n = rng.integers(12, 360, (size, 8)).astype(float)\n        v = rng.uniform(1000, 100000, (size, 8))\n        rate = rng.uniform(.003, .03, (size, 8))\n        args = [n, -v * rate / (1 - (1 + rate) ** -n), v]",
    ),
}
PACKAGES = {
    "transforms3d": "transforms3d",
    "filterpy": "filterpy",
    "perlin-numpy": "perlin_numpy",
    "numpy-financial": "numpy_financial",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=WRAPPERS, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=360)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stamp(f"START estimate=2-6min result={out}/result.json log={out}.log")
    start = time.perf_counter()
    project = out / "project"
    project.mkdir()
    package = PACKAGES[args.kind]
    shutil.copytree(
        args.upstream / package,
        project / package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"),
    )
    for name in ("LICENSE", "LICENSE.txt", "LICENSE.rst", "LICENSE.md"):
        if (args.upstream / name).exists():
            shutil.copy2(args.upstream / name, project / name)
    (project / "workload.py").write_text(WRAPPERS[args.kind])
    sizes, body = INPUTS[args.kind]
    (project / "examples.py").write_text(
        LOADER.replace("SIZES", repr(sizes)).replace("KIND_INPUT", body)
    )
    revision = subprocess.run(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True, capture_output=True
    ).stdout.strip()
    inherited = ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]
    for key in inherited:
        os.environ[key] = "1"
    setup = time.perf_counter() - start
    write(
        out / "protocol.json",
        dict(
            kind=args.kind,
            upstream=str(args.upstream),
            revision=revision,
            seed=202610145,
            sizes=sizes,
            budget=args.budget,
            expected_requests=100,
            deployment="auto",
            source_edits=0,
            manual_integration_files=["workload.py", "examples.py"],
            setup_script_seconds=setup,
            setup_human_seconds=None,
            human_study=False,
            notes="Agent-operated integration trial; not independent-user usability evidence.",
        ),
    )
    result = optimize_project(
        project,
        entry="workload.py:run",
        input_loader="examples.py:cases",
        deployment="auto",
        expected_requests=100,
        inherit_env=inherited,
        budget=args.budget,
        output=out / "workflow",
    )
    summary = dict(
        workflow_status=result.summary["status"],
        elapsed_seconds=time.perf_counter() - start,
        source_selected=result.summary["selected"],
        setup_script_seconds=setup,
        manual_integration_files=2,
        source_edits=0,
        llm_calls=0,
        deployment=result.summary.get("deployment"),
        reason=result.summary.get("reason"),
    )
    if "application" in result.summary:
        import numpy as np

        bundle = Path(result.summary["application"])
        case = json.loads((bundle / "cases.json").read_text())[1]
        calls = inputs(case["calls"], bundle / "reference")
        # New finite values, identical metadata. Copy mutations remain local.
        for value in calls[0]["args"]:
            if type(value) is np.ndarray:
                value *= 1.00001
        with CallableApplication(bundle) as selected:
            actual = selected.run_calls(calls)
            summary["actual_variant"] = selected.info["variant"]
            summary["actual_mode"] = selected.mode
        with CallableApplication(bundle, variant="reference") as reference:
            expected = reference.run_calls(calls)
        summary["changing_inputs_exact"] = identity(
            Observer(64 * 2**20).encode(actual)
        ) == identity(Observer(64 * 2**20).encode(expected))
        if not summary["changing_inputs_exact"]:
            raise ValueError("new public adapter outputs differ from reference")
    write(out / "result.json", summary)
    stamp(f"DONE status={summary['workflow_status']} result={out}/result.json log={out}.log")
    if summary["workflow_status"] not in {"accepted", "reference_retained"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
