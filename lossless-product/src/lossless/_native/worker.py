"""Isolated compiler/benchmark worker. Output is an atomic JSON record."""

import argparse
import math
import signal
import resource
import subprocess
import sys
import time

from .common import read, write, sha, measure, interval

COMPILE_FLAGS = [
    "-O3",
    "-std=c11",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-dynamiclib" if sys.platform == "darwin" else "-shared",
    "-fPIC",
]


def compile_job(job):
    command = ["clang", *COMPILE_FLAGS, job["source"], "-o", job["binary"], "-lm"]
    result = subprocess.run(command, capture_output=True, text=True)
    return {
        "status": "ok" if result.returncode == 0 else "compile_failed",
        "command": command,
        "returncode": result.returncode,
        "compiler_output": (result.stdout + result.stderr)[-16000:],
        "source_sha256": sha(job["source"]),
        "binary_sha256": sha(job["binary"]) if result.returncode == 0 else None,
    }


def benchmark_job(job):
    from . import operators

    operator = job["operator"]
    case = job["case"]
    contract = operators.CONTRACTS[operator]
    baseline = operators.load_library(job["baseline"])
    candidate = operators.load_library(job["candidate"])
    validation = {}
    seed = job["seed"]
    for index, distribution in enumerate(contract["distributions"]):
        arrays = operators.input_arrays(operator, case, seed + index, distribution)
        before = operators.digest(arrays)
        expected = operators.reference(operator, arrays)
        funcs, guards, poison = operators.executors(operator, case, arrays, baseline, candidate)
        # Validate trusted implementations before the proposal; a baseline defect stops the campaign.
        for name in [n for n in funcs if n != "proposal"] + ["proposal"]:
            poison()
            actual = funcs[name]()
            value = operators.check(operator, actual, expected)
            value["guards_intact"] = guards()
            value["inputs_unchanged"] = operators.digest(arrays) == before
            value["passed"] = (
                value["passed"] and value["guards_intact"] and value["inputs_unchanged"]
            )
            validation.setdefault(name, {})[distribution] = value
            if not value["passed"]:
                return {
                    "status": "incorrect" if name == "proposal" else "reference_failure",
                    "validation": validation,
                    "failed_implementation": name,
                    "failed_distribution": distribution,
                    "case": case,
                }
    phases = {}
    for phase, offset, blocks in [
        ("screening", 10000, job["screening_blocks"]),
        ("confirmation", 20000, job["confirmation_blocks"]),
    ]:
        arrays = operators.input_arrays(operator, case, seed + offset)
        before = operators.digest(arrays)
        funcs, guards, _ = operators.executors(operator, case, arrays, baseline, candidate)
        phases[phase] = measure(
            funcs, seed + offset + job["candidate_seed"], blocks, job["target_ns"]
        )
        if not guards() or operators.digest(arrays) != before:
            return {
                "status": "incorrect",
                "reason": "timing guard or input mutation",
                "validation": validation,
                "case": case,
            }
    med = phases["confirmation"]["median_us"]
    samples = phases["confirmation"]["samples_us"]
    libraries = [n for n in med if n not in ("proposal", "native_baseline")]
    library = min(libraries, key=lambda n: (phases["screening"]["median_us"][n], n))
    ci = interval(samples["native_baseline"], samples["proposal"], seed + 30000)
    return {
        "status": "ok",
        "case": case,
        "validation": validation,
        **phases,
        "library_choice": library,
        "speedup_vs_native": med["native_baseline"] / med["proposal"],
        "speedup_vs_library": med[library] / med["proposal"],
        "ci95_vs_native": ci,
        "confirmed_win": ci[0] > 1,
        "confirmed_regression": ci[1] < 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job")
    parser.add_argument("output")
    args = parser.parse_args()
    job = read(args.job)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # OS-default signal terminates native loops even if Python cannot regain control.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(max(1, math.ceil(job["timeout_seconds"])))
    started = time.perf_counter()
    result = compile_job(job) if job["kind"] == "compile" else benchmark_job(job)
    result["elapsed_seconds"] = time.perf_counter() - started
    write(args.output, result)


if __name__ == "__main__":
    main()
