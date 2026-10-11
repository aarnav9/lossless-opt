"""One fresh CPU experiment process. Correctness and clean timing are separate."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback
import tracemalloc
import cProfile
import pstats

import numpy as np


def load_arrays(value, root):
    if isinstance(value, dict):
        if set(value) == {"$npy"}:
            return np.load(root / value["$npy"], allow_pickle=False)
        return {k: load_arrays(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [load_arrays(v, root) for v in value]
    return value


def flatten(value, arrays, path="root"):
    if isinstance(value, np.ndarray):
        arrays[path] = value.copy()
        return {"array": path, "dtype": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, dict):
        return {k: flatten(v, arrays, path + "/" + k) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [flatten(v, arrays, path + "/" + str(i)) for i, v in enumerate(value)]
    if isinstance(value, np.generic):
        return value.item()
    return value


def fingerprint(value):
    arrays = {}
    skeleton = flatten(value, arrays)
    h = hashlib.sha256(json.dumps(skeleton, sort_keys=True).encode())
    for key, array in arrays.items():
        h.update(key.encode())
        h.update(array.tobytes())
    return h.hexdigest()


def run(payload):
    root = Path(payload["project"])
    sys.path[:0] = [str(root), str(Path(__file__).resolve().parent)]
    config = payload.get("candidate", {})
    start = time.perf_counter()
    if payload.get("kind", "application") == "native":
        from native_trials import make_operation, native_inputs

        operation = make_operation(config, payload)
        setup = time.perf_counter() - start
        calls = native_inputs(payload["case"])
    else:
        import spectrum_workload
        from runtime import make_processor

        operation = make_processor(spectrum_workload, config)
        setup = time.perf_counter() - start
        calls = [
            load_arrays(c["args"][0], Path(payload["fixtures"])) for c in payload["case"]["calls"]
        ]
        for records in calls:
            for record in records:
                transform = payload["case"].get("transform", {})
                if transform.get("scale") is not None:
                    record["y"] *= transform["scale"]
                if transform.get("zeros"):
                    record["y"][:] = 0.0
                    record["y"][::2] = -0.0
                if transform.get("zero_weights") and "weights" in record["options"]:
                    record["options"]["weights"][:] = 0.0
                if transform.get("strided"):
                    for key in ("x", "y"):
                        original = record[key]
                        buffer = np.empty(original.size * 2, dtype=original.dtype)
                        buffer[::2] = original
                        record[key] = buffer[::2]
                if transform.get("readonly"):
                    record["x"].flags.writeable = False
                    record["y"].flags.writeable = False
    mode = payload["mode"]
    before = fingerprint(calls) if mode == "correctness" else None
    if mode == "memory":
        tracemalloc.start()
    prepared_calls = None
    if config.get("family") == 4 and config["options"].get("explicit"):
        started = time.perf_counter()
        prepared_calls = operation.prepare_requests(calls)
        setup += time.perf_counter() - started
    saved = []
    times = []
    previous = []
    oracle_checks = []
    profiler = cProfile.Profile() if mode == "attribution" else None
    try:
        for position, call in enumerate(calls):
            started = time.perf_counter()
            if profiler:
                profiler.enable()
            result = (
                operation.run_prepared(prepared_calls[position])
                if prepared_calls is not None
                else operation(call)
            )
            if profiler:
                profiler.disable()
            times.append(time.perf_counter() - started)
            if mode == "correctness":
                previous.append(fingerprint(result))
                if payload.get("kind") == "native":
                    from lossless._native import operators

                    operator = payload["case"]["operator"]
                    arrays = (
                        call
                        if operator == "rmsnorm_residual"
                        else (call[0], call[0], np.ones(call[0].shape[1], dtype=np.float32))
                    )
                    oracle_checks.append(
                        operators.check(operator, result, operators.reference(operator, arrays))
                    )
            # Declared retained-output lifecycle, identical for reference/candidate.
            saved.append(result)
        traced = tracemalloc.get_traced_memory() if mode == "memory" else None
        if mode == "memory":
            tracemalloc.stop()
        record = dict(
            status="completed",
            calls=times,
            setup_seconds=setup,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            traced=traced,
        )
        if hasattr(operation, "cache_payload_bytes"):
            record["retained_cache_payload_bytes"] = operation.cache_payload_bytes()
            record["resource_eligible"] = record["retained_cache_payload_bytes"] <= 16 * 1024**2
        if profiler:
            stats = pstats.Stats(profiler)
            record["hotspots"] = [
                dict(
                    file=Path(k[0]).name,
                    line=k[1],
                    function=k[2],
                    calls=v[1],
                    self_seconds=v[2],
                    cumulative_seconds=v[3],
                )
                for k, v in sorted(stats.stats.items(), key=lambda item: item[1][2], reverse=True)[
                    :20
                ]
            ]
        if mode == "correctness":
            arrays = {}
            record.update(
                inputs_unchanged=before == fingerprint(calls),
                retained_unchanged=previous == [fingerprint(v) for v in saved],
                structure=flatten(saved, arrays),
            )
            np.savez(Path(payload["result"]).with_suffix(".npz"), **arrays)
            if oracle_checks:
                record["native_oracle_checks"] = oracle_checks
                record["native_oracle_passed"] = all(c["passed"] for c in oracle_checks)
        return record
    finally:
        close = getattr(operation, "close", None)
        if close:
            close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("payload", type=Path)
    args = p.parse_args()
    payload = json.loads(args.payload.read_text())
    try:
        result = run(payload)
    except BaseException as error:
        traceback.print_exc()
        result = dict(status="failed", error=type(error).__name__, message=str(error))
    Path(payload["result"]).write_text(json.dumps(result, allow_nan=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
