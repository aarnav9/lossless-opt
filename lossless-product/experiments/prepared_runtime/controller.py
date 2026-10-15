"""Isolated parent runtime for paired old/new request measurements; no provider calls."""

import argparse
import json
from pathlib import Path
import resource
import sys
import time

from lossless.applications import PreparedApplication
from lossless.applications.worker import inputs, Observer
from lossless.jobs import identity, read_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--features", default="[]")
    p.add_argument("--new", action="store_true")
    p.add_argument("--values", action="store_true")
    a = p.parse_args()
    bundle = a.bundle.resolve()
    options = dict(variant=a.variant, _qualifying=True, timeout=30)
    if a.new:
        options["_features"] = json.loads(a.features)
    value_cases = (
        {
            c["id"]: inputs(c["calls"], bundle / "reference")
            for c in read_json(bundle / "cases.json")
        }
        if a.values
        else {}
    )
    stage = Path.cwd() / "inputs"
    if a.values:
        stage.mkdir(exist_ok=True)
    held = []
    with PreparedApplication(bundle, **options) as app:
        print(
            json.dumps(dict(status="ready", info=app.info, startup_seconds=app.startup_seconds)),
            flush=True,
        )
        for line in sys.stdin:
            command = json.loads(line)
            if command.get("stop"):
                break
            try:
                tick = time.perf_counter()
                if not a.values:
                    r = app.run_case(command["case"])
                elif a.new:
                    r = app.run_values({"calls": value_cases[command["case"]]})
                else:
                    # Fair live-array control: materialize caller-owned values as the
                    # old API requires. Include staging and fixture I/O in latency.
                    import numpy as np

                    count = [0]

                    def stage_value(v):
                        if type(v) is np.ndarray:
                            name = f"input_{count[0]}.npy"
                            count[0] += 1
                            np.save(stage / name, v, allow_pickle=False)
                            return {"$npy": name}
                        if type(v) is list:
                            return [stage_value(x) for x in v]
                        if type(v) is dict:
                            return {k: stage_value(x) for k, x in v.items()}
                        return v

                    request = {"calls": stage_value(value_cases[command["case"]])}
                    r = app.run(request, input_root=stage)
                elapsed = time.perf_counter() - tick
                limit = 64 * 2**20
                output_hash = identity(Observer(limit).encode(r["outputs"]))
                for values, expected in held:
                    if identity(Observer(limit).encode(values)) != expected:
                        raise ValueError("a later request overwrote previously returned values")
                held = [(r["outputs"], output_hash)]
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                response = dict(
                    status="completed",
                    fingerprint=r["fingerprint"],
                    outputs=output_hash,
                    request_seconds=elapsed,
                    worker_request_seconds=r["request_seconds"],
                    call_seconds=sum(r["call_seconds"]),
                    worker_peak_rss=max(r["worker_peak_rss_bytes"], app.peak_tree_rss_bytes),
                    parent_peak_rss=int(rss if sys.platform == "darwin" else rss * 1024),
                    memory_available=app.memory_available,
                    variant=r["variant"],
                )
            except Exception as error:
                response = dict(status="failed", error=type(error).__name__, reason=str(error))
            print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
