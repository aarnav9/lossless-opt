"""One private CPU request worker; project imports are reused, factories are fresh."""

import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import random
import resource
import sys
import time

from ..jobs import identity, read_json
from .bundle import select, signature
from .config import validate_cases
from .replay import copy_project
from .request_guard import RequestGuard
from .value_request import normalize
from .wire import MAX_MESSAGE, pack, unpack
from .worker import Observer, inputs, load_target


def serve(fd, bundle, requested, qualifying, features=(), input_mode="fixtures"):
    import numpy as np
    from . import raw_wire

    staging = bytearray() if "buffers" in features else None
    encode = (lambda value: raw_wire.pack(value, staging)) if "raw" in features else pack

    connection = Connection(fd)
    bundle = Path(bundle).resolve()
    manifest = read_json(bundle / "manifest.json")
    selected, reason = select(bundle, manifest, None, requested, check_inputs=False)
    certificate = bundle / ("prepared-values.json" if input_mode == "values" else "prepared.json")
    if selected == "optimized" and not qualifying:
        policy = read_json(certificate) if certificate.exists() else {}
        if (
            policy.get("status") != "accepted"
            or policy.get("manifest_identity") != identity(manifest)
            or policy.get("runtime_features") != sorted(features)
            or policy.get("input_mode", "fixtures") != input_mode
        ):
            selected, reason = "reference", "prepared request lifecycle has not qualified"
    root = Path.cwd() / "project"
    copy_project(bundle / selected, root, {"files": manifest[selected + "_files"]})
    cases = read_json(bundle / "cases.json")
    guard = (
        RequestGuard(cases, root, manifest["input_signatures"])
        if selected == "optimized" and "guards" in features
        else None
    )
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(root), str(root / "src")]
    os.chdir(root)
    random.seed(manifest["seed"])
    np.random.seed(manifest["seed"])
    entry, loaded = manifest["entry"], {}
    function = load_target(entry["target"], root, loaded)
    observe = load_target(entry["observe"], root, loaded) if entry.get("observe") else None
    cleanup = load_target(entry["cleanup"], root, loaded) if entry.get("cleanup") else None
    connection.send_bytes(
        encode(dict(status="ready", variant=selected, fallback_reason=reason, pid=os.getpid()))
    )
    while True:
        frame = connection.recv_bytes(MAX_MESSAGE)
        if "raw" in features:
            command = raw_wire.unpack(frame)
        elif frame.startswith(b"LLV1"):
            if "values" not in features:
                raise ValueError("direct values not enabled")
            command = unpack(frame[4:])
        else:
            if len(frame) > 2**20:
                raise ValueError("request JSON exceeds 1 MiB")
            command = json.loads(frame)
        del frame
        if command["action"] == "close":
            return
        try:
            if command["action"] == "values":
                if "values" not in features:
                    raise ValueError("direct values not enabled")
                calls = normalize(command["request"])["calls"]
            elif "case" in command:
                found = [case for case in cases if case["id"] == command["case"]]
                if len(found) != 1:
                    raise ValueError("unknown saved case")
                row, fixture_root = found[0], root
            else:
                row = validate_cases(
                    [{"id": "request", "split": "evaluation", **command["request"]}], entry
                )[0]
                fixture_root = Path(command["input_root"]).resolve()
            if command["action"] != "values":
                calls = inputs(row["calls"], fixture_root)
            if selected == "optimized":
                try:
                    admitted = (
                        guard.admits(calls)
                        if guard is not None
                        else identity(signature(calls)) in manifest["input_signatures"]
                    )
                except ValueError:
                    admitted = False
                if not admitted:
                    connection.send_bytes(
                        encode(
                            dict(
                                status="outside_scope",
                                reason="request metadata outside qualification; use an explicit reference session",
                            )
                        )
                    )
                    continue
            # Each request keeps the established fresh-factory, seeded lifecycle.
            # Imports survive; calls within this request share the same object.
            random.seed(manifest["seed"])
            np.random.seed(manifest["seed"])
            selected_call = (
                function(
                    *inputs(entry["factory_args"], root), **inputs(entry["factory_kwargs"], root)
                )
                if entry["factory"]
                else function
            )
            results, observations, retained, times = [], [], [], []
            limit = manifest["max_output_bytes"]
            try:
                for call in calls:
                    before = Observer(limit).encode(call)
                    tick = time.perf_counter()
                    result = selected_call(*call["args"], **call["kwargs"])
                    times.append(time.perf_counter() - tick)
                    observed = (
                        observe(selected_call, result, call["args"], call["kwargs"])
                        if observe
                        else result
                    )
                    # Actual returns and declared observations must both be bounded values.
                    encoded_result = Observer(limit).encode(result)
                    results.append(result)
                    retained.append(observed)
                    observations.append(
                        dict(
                            output=(
                                encoded_result
                                if "observations" in features and observed is result
                                else Observer(limit).encode(observed)
                            ),
                            inputs_before=before,
                            inputs_after=Observer(limit).encode(call),
                        )
                    )
                fingerprint = identity(
                    dict(
                        calls=observations,
                        retained_outputs=[Observer(limit).encode(r) for r in retained],
                    )
                )
            finally:
                if cleanup:
                    cleanup(selected_call)
                elif (
                    type(selected_call).__module__ == "lossless.execution"
                    and type(selected_call).__name__ == "IndependentMap"
                ):
                    selected_call.close()
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            response = encode(
                dict(
                    status="completed",
                    variant=selected,
                    outputs=results,
                    fingerprint=fingerprint,
                    call_seconds=times,
                    worker_peak_rss_bytes=int(rss if sys.platform == "darwin" else rss * 1024),
                )
            )
            if len(response) > MAX_MESSAGE:
                raise ValueError("response too large")
            connection.send_bytes(response)
            del results, retained, observations, response, selected_call
        except Exception as error:
            # State after an application failure is unknown. Report once and exit.
            connection.send_bytes(
                encode(dict(status="failed", error=type(error).__name__, reason=str(error)))
            )
            return


if __name__ == "__main__":
    serve(
        int(sys.argv[1]),
        sys.argv[2],
        sys.argv[3],
        sys.argv[4] == "qualify",
        json.loads(sys.argv[5]),
        sys.argv[6],
    )
