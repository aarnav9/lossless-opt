"""Isolated native/MLX profiling worker. Timings never enter optimizer acceptance."""

import cProfile
import importlib.metadata
from pathlib import Path
import platform
import pstats
import random
import resource
import statistics
import sys
import time

from ._native.common import measure, read, sha, stamp
from .profiling import save_report


def peak_rss():
    return int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024)
    )


def stats(samples, requests=1, units=None, unit="elements"):
    import numpy as np

    median = statistics.median(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": median,
        "p95_seconds": float(np.quantile(samples, 0.95)),
        "requests_per_second": requests / median,
        "units_per_second": units / median if units is not None else None,
        "unit": unit,
    }


def host_profile(call, calls=3):
    profiler = cProfile.Profile()
    for _ in range(calls):
        profiler.runcall(call)
    data = pstats.Stats(profiler)
    records = []
    for (filename, line, name), (_, nc, own, cumulative, _) in data.stats.items():
        records.append(
            {
                "function": f"{Path(filename).name}:{line}:{name}",
                "calls": nc,
                "self_seconds_per_request": own / calls,
                "self_fraction": own / data.total_tt if data.total_tt else 0,
                "inclusive_seconds_per_request": cumulative / calls,
            }
        )
    return {
        "calls": calls,
        "total_instrumented_seconds": data.total_tt,
        "top_self_time": sorted(records, key=lambda v: v["self_seconds_per_request"], reverse=True)[
            :15
        ],
        "scope": "Separate instrumented run. Host execution and waits only; not GPU kernel timing. Inclusive times overlap; do not add them. Warm samples exclude profiler overhead.",
    }


def native(directory, job, summary):
    import numpy as np
    from . import load
    from .artifacts import NativeOperation
    from ._native import operators
    from ._native.worker import compile_job
    from .economics import payback

    work = job["resolved"]["workload"]
    operator = work["adapter"].split(".")[1]
    cases = [c for c in work["cases"] if c["split"] == "discovery"]
    comparator = job["resolved"]["objective"]["comparator"]
    summary["comparator"] = comparator
    source = directory / "baseline.c"
    source.write_text(operators.baseline_source(operator))
    binary = directory / ("baseline.dylib" if sys.platform == "darwin" else "baseline.so")
    t = time.perf_counter()
    compiled = compile_job({"source": str(source), "binary": str(binary)})
    if compiled["status"] != "ok":
        raise RuntimeError("profile baseline compilation failed: " + compiled["compiler_output"])
    summary["reference_preparation_seconds"] = time.perf_counter() - t
    manifest = {
        "adapter": work["adapter"],
        "selected": "reference",
        "comparator": comparator,
        "platform": sys.platform,
        "machine": platform.machine(),
        "numpy": np.__version__,
        "cases": [{**c, "layout": layout} for c in cases for layout in ["c", "f", "slice2"]],
        "binary": binary.name,
        "hashes": {binary.name: sha(binary)},
    }
    t = time.perf_counter()
    reference = NativeOperation(directory, manifest)
    operations = {"reference": reference}
    loads = {"reference": time.perf_counter() - t}
    original_search = None
    if job["artifact"]:
        t = time.perf_counter()
        operations["artifact"] = load(job["artifact"])
        loads["artifact"] = time.perf_counter() - t
        original_search = read(Path(job["artifact"]) / "report.json").get("wall_time_seconds")
        summary["artifact_selected"] = operations["artifact"].manifest["selected"]
    initial_reasons = {n: op.fallback_reason for n, op in operations.items()}
    for index, case in enumerate(cases):
        t = time.perf_counter()
        arrays = operators.input_arrays(operator, case, 7100 + index)
        args = arrays if operator == "rmsnorm_residual" else (arrays[0],)
        preparation = time.perf_counter() - t
        before = operators.digest(arrays)
        expected = operators.reference(operator, arrays)
        calls, bound, results = {}, {}, {}
        for name, operation in operations.items():
            operation.fallback_reason = initial_reasons[name]
            calls[name] = lambda op=operation: op(*args)
            t = time.perf_counter()
            value = calls[name]()
            first = time.perf_counter() - t
            check = operators.check(operator, value, expected)
            if not check["passed"] or operators.digest(arrays) != before:
                raise ValueError(f"profile correctness failed for {name}: {check}")
            t = time.perf_counter()
            bound[name] = operation.bind(*args)
            binding = time.perf_counter() - t
            if not operators.check(operator, bound[name](), expected)["passed"]:
                raise ValueError("bound profile correctness failed")
            results[name] = {
                "first_call_seconds": first,
                "binding_seconds": binding,
                "load_seconds": loads[name],
                "correctness": check,
                "fallback_reason": operation.fallback_reason,
            }
        timed = measure(calls, 1200 + index, job["repeats"], 1000000)
        bound_timed = measure(bound, 1800 + index, job["repeats"], 1000000)
        for name in results:
            results[name]["warm"] = stats(
                [v / 1e6 for v in timed["samples_us"][name]], units=arrays[0].size
            )
            results[name]["bound_warm"] = stats(
                [v / 1e6 for v in bound_timed["samples_us"][name]], units=arrays[0].size
            )
            results[name]["host_attribution"] = host_profile(calls[name])
        item = {
            "case": case,
            "label": f"{case['rows']}x{case['columns']} {case['layout']}",
            "input_generation_seconds": preparation,
            "input_bytes": sum(a.nbytes for a in arrays),
            "implementations": results,
            "paired_orders": timed["orders"],
            "memory": {
                "process_peak_rss_bytes": peak_rss(),
                "scope": "Cumulative worker high-water mark, including inputs, validation and libraries; not per-call allocation.",
            },
        }
        if original_search is not None:
            ref, opt = results["reference"], results["artifact"]
            extra = max(
                0,
                (opt["load_seconds"] + opt["first_call_seconds"] - opt["warm"]["median_seconds"])
                - (ref["load_seconds"] + ref["first_call_seconds"] - ref["warm"]["median_seconds"]),
            )
            item["payback"] = payback(
                original_search, ref["warm"]["median_seconds"], opt["warm"]["median_seconds"], extra
            )
            item["payback"]["scope"] = (
                "Ordinary allocating API; setup is artifact load plus first-call excess over warm, relative to reference. Input generation and profiling-only reference compilation excluded. Uses the recorded full search cost."
            )
        summary["cases"].append(item)
        save_report(directory, summary)
        stamp(f"PROFILE native case={index + 1}/{len(cases)} result={directory / 'report.json'}")
    summary["measurement_scope"] = (
        "Discovery inputs only. Ordinary allocating API and bound-buffer calls measured separately. Native p95 is across repeat-block averages, not a production request-tail percentile. Each first call is the first invocation of that case in this worker, after library setup; it is not a system-wide cold-cache guarantee."
    )


def mlx(directory, job, summary):
    import mlx.core as mx
    from mlx_lm import load as load_model
    from . import load
    from .adapters.mlx.runtime import generate
    from .adapters.mlx.checks import diff, cache_diff
    from .economics import payback

    versions = {n: importlib.metadata.version(n) for n in ["mlx", "mlx-lm"]}
    if versions != {"mlx": "0.32.3", "mlx-lm": "0.32.0"}:
        raise ValueError("MLX profiling requires mlx==0.32.3 and mlx-lm==0.32.0")
    work = job["resolved"]["workload"]
    rows = [r for r in work["inputs"] if r["split"] == "discovery"]
    texts, counts = [r["text"] for r in rows], [r["max_tokens"] for r in rows]
    operation = None
    t = time.perf_counter()
    if job["artifact"]:
        operation = load(job["artifact"]).open(model=work["parameters"]["model"])
        model, tokenizer = operation._model, operation._tokenizer
        summary["artifact_selected"] = operation.manifest["selected"]
        summary["artifact_expected_model_identity"] = operation.manifest["model_identity"]
    else:
        from .adapters.mlx.identity import fingerprint

        summary["model_identity"] = fingerprint(work["parameters"]["model"])
        model, tokenizer = load_model(work["parameters"]["model"])
        mx.eval(model.parameters())
    summary["model_load_and_admission_seconds"] = time.perf_counter() - t
    summary["hardware"] = mx.device_info()
    summary["model_path"] = work["parameters"]["model"]
    calls = {
        "reference": lambda: generate(
            model, None, tokenizer, texts, counts, capture=True, profile=True
        )
    }
    if operation is not None:
        calls["artifact"] = lambda: generate(
            model,
            operation._grouped,
            tokenizer,
            texts,
            counts,
            optimized=operation._allowed,
            options=operation.manifest.get("options", {}),
            graph=operation._graph,
            fallback_reason=operation.fallback_reason,
            capture=True,
            profile=True,
        )
    results, samples, cold = {}, {n: [] for n in calls}, {}
    for name, call in calls.items():
        mx.reset_peak_memory()
        cold[name] = call()
        results[name] = {
            "first_call_seconds": cold[name][0]["seconds"],
            "cold_peak_mlx_allocated_bytes": mx.get_peak_memory(),
            "fallback_reason": cold[name][0]["fallback_reason"],
            "cold_peak_scope": "Validation retains reference outputs while checking an artifact; warm peaks describe repeated deployment calls.",
        }
    if operation is not None:
        a, b = cold["reference"], cold["artifact"]
        checks = [
            {
                "tokens": a[0]["output_ids"][i] == b[0]["output_ids"][i],
                "probabilities": diff(a[2][i], b[2][i])["bitwise"],
                "kv": cache_diff(a[1][i], b[1][i])["bitwise"],
            }
            for i in range(len(rows))
        ]
        if not all(all(c.values()) for c in checks):
            raise ValueError("profile artifact failed exact comparison against serial reference")
        summary["correctness"] = checks
        del a, b
    expected_ids = cold["reference"][0]["output_ids"]
    cold.clear()
    orders, latency, stages, peaks = (
        [],
        {n: [] for n in calls},
        {n: [] for n in calls},
        {n: [] for n in calls},
    )
    for repeat in range(job["repeats"]):
        order = list(calls)
        random.Random(7300 + repeat).shuffle(order)
        orders.append(order)
        for name in order:
            mx.reset_peak_memory()
            value = calls[name]()
            row = value[0]
            if row["output_ids"] != expected_ids:
                raise ValueError("profile changed token decisions during timing")
            samples[name].append(row["seconds"])
            latency[name].append(
                {
                    "ttft_seconds": row["ttft_seconds"],
                    "completion_seconds": row["completion_seconds"],
                }
            )
            stages[name].append(row["profile_stages_seconds"])
            peaks[name].append(mx.get_peak_memory())
            del value
        stamp(f"PROFILE MLX batch={repeat + 1}/{job['repeats']}")
    for name in calls:
        results[name].update(
            warm=stats(samples[name], requests=len(rows), units=sum(counts), unit="tokens"),
            request_latencies=latency[name],
            stage_samples_seconds=stages[name],
            peak_mlx_allocated_bytes=max(peaks[name]),
            host_attribution=host_profile(calls[name], calls=1),
            median_ttft_seconds_by_request=[
                statistics.median(v["ttft_seconds"][i] for v in latency[name])
                for i in range(len(rows))
            ],
            median_completion_seconds_by_request=[
                statistics.median(v["completion_seconds"][i] for v in latency[name])
                for i in range(len(rows))
            ],
            stages=[
                {
                    "name": stage,
                    "mean_seconds": statistics.mean(v[stage] for v in stages[name]),
                    "fraction": sum(v[stage] for v in stages[name]) / sum(samples[name]),
                }
                for stage in stages[name][0]
            ],
        )
    item = {
        "label": f"{len(rows)} requests / {sum(counts)} tokens",
        "implementations": results,
        "paired_orders": orders,
    }
    if operation is not None:
        search = read(Path(job["artifact"]) / "report.json")["wall_time_seconds"]
        ref, opt = results["reference"], results["artifact"]
        extra = max(
            0,
            (opt["first_call_seconds"] - opt["warm"]["median_seconds"])
            - (ref["first_call_seconds"] - ref["warm"]["median_seconds"]),
        )
        item["payback"] = payback(
            search, ref["warm"]["median_seconds"], opt["warm"]["median_seconds"], extra
        )
        item["payback"]["scope"] = (
            "Whole batches against stock serial, after shared model loading; extra setup is observed first-call excess over warm. Includes recorded search cost."
        )
    summary["cases"].append(item)
    summary["measurement_scope"] = (
        "Discovery requests only, complete fixed-count calls with probability capture and final KV materialization. TTFT/completion are per request from batch start, including serial queueing. Stage timing is coarse wall time, not per-kernel GPU attribution. First calls share a loaded model and are order-dependent. MLX peak allocation excludes untracked driver/process memory."
    )


def main(directory):
    job = read(directory / "profile_job.json")
    adapter = job["resolved"]["workload"]["adapter"]
    packages = {}
    for name in ["numpy", "scipy", "mlx", "mlx-lm"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    summary = {
        "schema_version": 1,
        "kind": "profile",
        "status": "profiling",
        "adapter": adapter,
        "input_split": "discovery",
        "llm_calls": 0,
        "cases": [],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
        },
        "scope": "Profiling does not search, export an optimization, or change acceptance. Detailed attribution is sampled separately from warm timing.",
    }
    started = time.monotonic()
    (mlx if adapter == "mlx.fixed_count" else native)(directory, job, summary)
    summary.update(
        status="profiled",
        worker_seconds=time.monotonic() - started,
        process_peak_rss_bytes=peak_rss(),
    )
    save_report(directory, summary)
    stamp(f"COMPLETE worker result={directory / 'report.json'} log={directory / 'run.log'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
