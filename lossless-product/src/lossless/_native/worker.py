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


def _compile_job(job):
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


def compile_job(job):
    from . import cache
    import shutil

    identity = {
        "source": sha(job["source"]),
        "flags": COMPILE_FLAGS,
        "context": job.get("cache_context"),
    }
    with cache.entry(job.get("cache", {}).get("directory"), "compiled", identity) as folder:
        record = cache.load(folder)
        if record and record["status"] == "ok":
            shutil.copy2(folder / "binary", job["binary"])
            return {
                **record,
                "cache_hit": True,
                "command": ["clang", *COMPILE_FLAGS, job["source"], "-o", job["binary"], "-lm"],
            }
        record = _compile_job(job)
        if record["status"] == "ok":
            cache.save(folder, record, job["binary"])
        return {**record, "cache_hit": False}


def benchmark_job(job):
    from . import operators
    from .deployment import NativeOperation

    operator = job["operator"]
    case = job["case"]
    comparator = operators.validate_comparator(operator, job.get("comparator", "native_baseline"))
    contract = operators.CONTRACTS[operator]
    baseline = operators.load_library(job["baseline"])
    candidate = operators.load_library(job["candidate"])
    timing_scope = job.get("timing_scope", "bound")
    if timing_scope not in {"call", "bound"}:
        raise ValueError("unknown native timing scope")

    def ordinary_calls(arrays, names):
        args = arrays if operator == "rmsnorm_residual" else (arrays[0],)
        calls = {}
        for name in names:
            operation = NativeOperation.from_loaded_library(
                operator,
                candidate if name == "proposal" else baseline,
                [case],
                comparator=None if name == "proposal" else name,
            )
            calls[name] = lambda op=operation: op(*args)
        return calls

    validation = {}
    seed = job["seed"]
    from . import cache

    validation_key = {
        "context": job.get("cache_context"),
        "case": case,
        "seed": seed,
        "baseline": sha(job["baseline"]),
        "candidate": sha(job["candidate"]),
        "timing_scope": timing_scope,
    }
    directory = job.get("cache", {}).get("directory") if job.get("reuse_validation") else None
    validation_hit = False
    with cache.entry(directory, "validation", validation_key) as folder:
        cached = cache.load(folder)
        if cached:
            validation = cached["validation"]
            validation_hit = True
        else:
            for index, distribution in enumerate(contract["distributions"]):
                arrays = operators.input_arrays(operator, case, seed + index, distribution)
                before = operators.digest(arrays)
                expected = operators.reference(operator, arrays)
                funcs, guards, poison = operators.executors(
                    operator, case, arrays, baseline, candidate
                )
                deployed = ordinary_calls(arrays, funcs) if timing_scope == "call" else {}
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
                    if name in deployed:
                        value["ordinary_call"] = operators.check(
                            operator, deployed[name](), expected
                        )
                        value["passed"] = (
                            value["passed"]
                            and value["ordinary_call"]["passed"]
                            and operators.digest(arrays) == before
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
            cache.save(folder, {"validation": validation})
    phases, kernel_only = {}, {}
    for phase, offset, blocks in [
        ("screening", 10000, job["screening_blocks"]),
        ("confirmation", 20000, job["confirmation_blocks"]),
    ]:
        arrays = operators.input_arrays(operator, case, seed + offset)
        before = operators.digest(arrays)
        funcs, guards, _ = operators.executors(operator, case, arrays, baseline, candidate)
        measured = ordinary_calls(arrays, funcs) if timing_scope == "call" else funcs
        phases[phase] = measure(
            measured, seed + offset + job["candidate_seed"], blocks, job["target_ns"]
        )
        if timing_scope == "call":
            kernel_only[phase] = measure(
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
        "timing_scope": timing_scope,
        "kernel_only": kernel_only,
        "measurement_scope": (
            "Ordinary deployment calls: guards, allocation, binding and dispatch included. Library loading and fixture generation excluded. kernel_only is diagnostic."
            if timing_scope == "call"
            else "Preallocated calls including dispatch; allocation and binding excluded."
        ),
        **phases,
        "library_choice": library,
        "validation_cache_hit": validation_hit,
        "comparator": comparator,
        "speedup_vs_comparator": med[comparator] / med["proposal"],
        "ci95_vs_comparator": interval(samples[comparator], samples["proposal"], seed + 30000),
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
