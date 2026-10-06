"""Fresh-process callable measurements; executed in the application's interpreter."""

import cProfile
import importlib.util
import json
from pathlib import Path
import pstats
import random
import resource
import sys
import time
import traceback
import tracemalloc

# Do not occupy sys.modules['worker']: many real applications have worker.py.
_spec = importlib.util.spec_from_file_location(
    "_lossless_reference_support", Path(__file__).with_name("worker.py")
)
_support = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_support)
Observer, inputs, load_target = _support.Observer, _support.inputs, _support.load_target


def location(filename, root):
    if filename.startswith(("<", "~")):
        return {"file": filename, "origin": "builtin"}
    path = Path(filename).resolve()
    if path.is_relative_to(root):
        return {"file": path.relative_to(root).as_posix(), "origin": "project"}
    for marker in ("site-packages/", "dist-packages/"):
        if marker in path.as_posix():
            return {"file": path.as_posix().split(marker, 1)[1], "origin": "dependency"}
    return {"file": path.name, "origin": "runtime"}


def rss():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def synchronize(value, hook, backends):
    if hook:
        hook()
        backends.add("explicit hook")
        return
    if any(name in sys.modules for name in ("jax", "cupy", "tensorflow")):
        raise ValueError("this runtime needs an explicit entry.synchronize hook for profiling")
    mlx = sys.modules.get("mlx.core")
    if mlx is not None:
        # Materialize result leaves as well as waiting for already-enqueued work.
        def leaves(item):
            if type(item).__module__.split(".")[0] == "mlx":
                yield item
            elif isinstance(item, dict):
                for child in item.values():
                    yield from leaves(child)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    yield from leaves(child)

        arrays = list(leaves(value))
        if arrays:
            mlx.eval(*arrays)
        mlx.synchronize()
        backends.add("mlx.eval + mlx.synchronize")
    torch = sys.modules.get("torch")
    if torch is not None:
        if torch.cuda.is_initialized():
            for device in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device)
            backends.add("torch.cuda.synchronize (all visible devices)")
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
            backends.add("torch.mps.synchronize")


def attribution(profiler, root):
    stats = pstats.Stats(profiler)
    rows = []
    for (filename, line, function), (primitive, calls, own, cumulative, _) in stats.stats.items():
        rows.append(
            {
                **location(filename, root),
                "line": line,
                "function": function,
                "calls": calls,
                "primitive_calls": primitive,
                "self_seconds": own,
                "cumulative_seconds": cumulative,
                "self_fraction": own / stats.total_tt if stats.total_tt else 0.0,
            }
        )
    return {
        "total_instrumented_seconds": stats.total_tt,
        "top_self": sorted(rows, key=lambda row: row["self_seconds"], reverse=True)[:30],
        "top_cumulative": sorted(rows, key=lambda row: row["cumulative_seconds"], reverse=True)[
            :30
        ],
        "scope": "cProfile host wall-time attribution, including native calls exposed by Python. Inclusive times overlap. Opaque native internals and GPU kernels are not decomposed.",
    }


def run(payload):
    root = Path(payload["workspace"]).resolve()
    mode = payload["profile_mode"]
    if mode not in {"timing", "attribution", "memory"}:
        raise ValueError("unknown profile mode")
    sys.path[:0] = [str(root), str(root / "src")]
    random.seed(payload["seed"])
    import numpy as np

    np.random.seed(payload["seed"])
    entry, loaded, backends = payload["entry"], {}, set()
    started = time.perf_counter()
    function = load_target(entry["target"], root, loaded)
    selected = (
        function(*inputs(entry["factory_args"], root), **inputs(entry["factory_kwargs"], root))
        if entry["factory"]
        else function
    )
    if not callable(selected):
        raise ValueError("factory must return a callable")
    observe = load_target(entry["observe"], root, loaded) if entry.get("observe") else None
    cleanup = load_target(entry["cleanup"], root, loaded) if entry.get("cleanup") else None
    sync = load_target(entry["synchronize"], root, loaded) if entry.get("synchronize") else None
    synchronize(None, sync, backends)
    setup_seconds = time.perf_counter() - started
    rows = []
    retained = []
    profiler = cProfile.Profile() if mode == "attribution" else None
    try:
        for call in payload["case"]["calls"]:
            args, kwargs = inputs(call["args"], root), inputs(call["kwargs"], root)
            before = Observer(payload["max_output_bytes"]).encode({"args": args, "kwargs": kwargs})
            synchronize(None, sync, backends)
            peak_before = rss()
            if mode == "memory":
                tracemalloc.start()
                base, _ = tracemalloc.get_traced_memory()
                tracemalloc.reset_peak()
            started = time.perf_counter()
            if profiler:
                profiler.enable()
            try:
                result = selected(*args, **kwargs)
                synchronize(result, sync, backends)
            finally:
                if profiler:
                    profiler.disable()
            elapsed = time.perf_counter() - started
            peak_after = rss()
            measured = {
                "call_seconds": elapsed,
                "process_peak_rss_bytes": peak_after,
                "process_peak_increase_bytes": max(0, peak_after - peak_before),
            }
            if mode == "memory":
                current, peak = tracemalloc.get_traced_memory()
                sites = tracemalloc.take_snapshot().statistics("lineno")[:15]
                measured["traced_memory"] = {
                    "peak_bytes_above_start": max(0, peak - base),
                    "retained_bytes_above_start": current - base,
                    "top_retained": [
                        {
                            **location(s.traceback[0].filename, root),
                            "line": s.traceback[0].lineno,
                            "bytes": s.size,
                            "count": s.count,
                        }
                        for s in sites
                    ],
                }
                tracemalloc.stop()
            # Correctness fingerprints and user observers are outside measured calls.
            observed = observe(selected, result, args, kwargs) if observe else result
            encoded = Observer(payload["max_output_bytes"]).encode(observed)
            if payload.get("retain_outputs"):
                retained.append(observed)
            after = Observer(payload["max_output_bytes"]).encode({"args": args, "kwargs": kwargs})
            rows.append(
                {
                    "observation": {
                        "output": encoded,
                        "inputs_before": before,
                        "inputs_after": after,
                    },
                    "measurement": measured,
                }
            )
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        if cleanup:
            cleanup(selected)
    record = {
        "status": "completed",
        "profile_mode": mode,
        "setup_seconds": setup_seconds,
        "calls": rows,
        "sequence_seconds": sum(r["measurement"]["call_seconds"] for r in rows),
        "synchronization": sorted(backends) or ["synchronous CPU"],
        "scope": "Each declared call is executed once per fresh process. Calls include completion synchronization; input loading, fingerprints and observer hooks are outside timers. No hidden warmup or repeated state transition.",
    }
    if profiler:
        record["attribution"] = attribution(profiler, root)
    if payload.get("retain_outputs"):
        record["retained_outputs"] = [
            Observer(payload["max_output_bytes"]).encode(item) for item in retained
        ]
    return record


def main():
    payload = json.loads(Path(sys.argv[1]).read_text())
    try:
        result = run(payload)
    except BaseException as error:
        traceback.print_exc()
        result = {"status": "failed", "error": type(error).__name__, "message": str(error)}
    Path(payload["result"]).write_text(json.dumps(result, allow_nan=False) + "\n")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
