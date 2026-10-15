"""Qualify complete prepared requests against equally persistent reference workers."""

from contextlib import ExitStack
import math
from pathlib import Path
import statistics
import time

from .._native.common import stamp, write
from ..jobs import identity, read_json
from .comparison import eligible, statistics_for
from .prepared import PreparedApplication, DEFAULT_FEATURES
from .worker import Observer, inputs


def qualify_prepared(
    bundle,
    *,
    output,
    expected,
    expected_outputs,
    python=None,
    budget=120,
    rounds=6,
    requests=3,
    input_mode="fixtures",
):
    if (
        input_mode not in {"fixtures", "values"}
        or type(rounds) is not int
        or type(requests) is not int
        or rounds < 6
        or requests < 2
        or budget <= 0
        or not math.isfinite(budget)
    ):
        raise ValueError(
            "prepared qualification requires >=6 rounds, >=2 requests and positive budget"
        )
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline = started + budget
    manifest = read_json(bundle / "manifest.json")
    certificate = bundle / ("prepared-values.json" if input_mode == "values" else "prepared.json")
    # Requalification invalidates any previous certificate before attempting work.
    write(certificate, dict(status="incomplete", manifest_identity=identity(manifest)))
    cases = [c for c in read_json(bundle / "cases.json") if c["split"] == "evaluation"]
    if (
        not cases
        or {c["id"] for c in cases} != set(expected)
        or set(expected_outputs) != set(expected)
    ):
        raise ValueError(
            "prepared qualification needs fresh-launch reference fingerprints for every evaluation case"
        )
    policy = dict(
        minimum_speedup=1.02, max_case_regression=0.03, max_call_regression=0.03, significance=0.05
    )
    rows = [dict(id=c["id"], pairs=[]) for c in cases]
    value_cases = (
        {c["id"]: {"calls": inputs(c["calls"], bundle / "reference")} for c in cases}
        if input_mode == "values"
        else {}
    )
    report = dict(
        status="incomplete",
        input_mode=input_mode,
        runtime_features=sorted(
            (*DEFAULT_FEATURES, *(["values"] if input_mode == "values" else []))
        ),
        manifest_identity=identity(manifest),
        policy=policy,
        rows=rows,
        lifecycle="Imports reused; a fresh seeded callable factory per request; no retries; copied output values.",
        timing_scope="Complete parent request: JSON transport, fixture loading, metadata checks, factory, calls, observations, cleanup, output serialization/transport/deserialization. Worker startup recorded separately.",
        rounds=rounds,
        requests_per_case=requests,
        llm_calls=0,
    )
    if input_mode == "values":
        report["timing_scope"] = (
            "Complete parent live-value request: validation, copied input serialization/transport/decoding, metadata checks, factory, calls, observations, cleanup, output serialization/transport/deserialization. Caller already owns inputs; startup recorded separately."
        )

    def progress(message):
        stamp(message)
        with (output / "run.log").open("a") as log:
            from .._native.common import now

            log.write(f"[{now()}] {message}\n")

    def remaining():
        left = deadline - time.perf_counter()
        if left <= 0:
            raise TimeoutError("prepared qualification budget exhausted")
        return min(30, left)

    progress(
        f"START prepared qualification budget={budget}s result={output}/report.json log={output}/run.log"
    )
    try:
        if manifest["selected"] != "optimized":
            report.update(
                status="reference_retained", reason="source search retained the reference"
            )
        else:
            startups = []
            for round_index in range(rounds):
                progress(
                    f"PREPARED round={round_index + 1}/{rounds} requests={len(cases) * requests * 2}"
                )
                with ExitStack() as stack:
                    sessions = {}
                    order = (
                        ["reference", "candidate"]
                        if round_index % 2 == 0
                        else ["candidate", "reference"]
                    )
                    for name in order:
                        session = stack.enter_context(
                            PreparedApplication(
                                bundle,
                                python=python,
                                variant="reference" if name == "reference" else "auto",
                                timeout=remaining(),
                                _qualifying=True,
                                input_mode=input_mode,
                            )
                        )
                        if name == "candidate" and session.info["variant"] != "optimized":
                            raise ValueError(
                                "candidate failed prepared environment admission: "
                                + str(session.info["fallback_reason"])
                            )
                        sessions[name] = session
                    startups.append({name: s.startup_seconds for name, s in sessions.items()})
                    # Rotate case order; keep exactly the same histories in both workers.
                    for index in range(len(cases)):
                        case_index = (index + round_index) % len(cases)
                        case = cases[case_index]
                        timings = {name: [] for name in sessions}
                        peaks = {name: 0 for name in sessions}
                        fingerprints = {name: [] for name in sessions}
                        for request_index in range(requests):
                            for name in order if request_index % 2 == 0 else order[::-1]:
                                session = sessions[name]
                                session.timeout = remaining()
                                result = (
                                    session.run_values(value_cases[case["id"]])
                                    if input_mode == "values"
                                    else session.run_case(case["id"])
                                )
                                timings[name].append(result["request_seconds"])
                                fingerprints[name].append(result["fingerprint"])
                                peaks[name] = max(
                                    peaks[name],
                                    result["worker_peak_rss_bytes"],
                                    session.peak_tree_rss_bytes,
                                )
                                if result["fingerprint"] != expected[case["id"]]:
                                    raise ValueError(
                                        f"prepared exactness differs from fresh reference: {case['id']} {name} request {request_index}"
                                    )
                                limit = manifest["max_output_bytes"] * len(result["outputs"])
                                if (
                                    identity(Observer(limit).encode(result["outputs"]))
                                    != expected_outputs[case["id"]]
                                ):
                                    raise ValueError(
                                        "prepared transported values differ from fresh reference: "
                                        + case["id"]
                                    )
                                del result
                        if any(not s.memory_available for s in sessions.values()):
                            raise ValueError("prepared process-tree memory monitoring unavailable")
                        pair = {}
                        for name in sessions:
                            times = timings[name]
                            pair[name] = dict(
                                sequence_seconds=sum(times),
                                call_seconds=times,
                                setup_seconds=sessions[name].startup_seconds,
                                first_call_seconds=times[0],
                                process_seconds=sum(times),
                                peak_rss_bytes=peaks[name],
                                fingerprints=fingerprints[name],
                            )
                        rows[case_index]["pairs"].append(pair)
                write(output / "report.json", report)
            report["statistics"] = statistics_for(rows, policy["minimum_speedup"])
            report["statistics"]["timing_scope"] = "prepared_requests"
            report["statistics"]["position_scope"] = (
                "Repeated complete requests, each containing the case's original callable sequence"
            )
            report["startup_seconds"] = startups
            report["memory_eligible"] = all(
                statistics.median(p["candidate"]["peak_rss_bytes"] for p in row["pairs"])
                <= 1.2 * statistics.median(p["reference"]["peak_rss_bytes"] for p in row["pairs"])
                for row in rows
            )
            report["memory_scope"] = (
                "Worker high-water RSS plus 20ms process-tree samples; per-process sampled peaks double-count shared pages and can miss short-lived descendants."
            )
            matched = dict(status="matched", statistics=report["statistics"])
            report["status"] = (
                "accepted"
                if eligible(matched, policy, final=True) and report["memory_eligible"]
                else "reference_retained"
            )
            report["exact"] = True
    except (ValueError, OSError, TimeoutError) as error:
        report.update(status="failed", reason=str(error))
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        write(output / "report.json", report)
        # Missing or failed evidence can never enable an optimized session.
        write(
            certificate,
            dict(
                status=report["status"],
                input_mode=input_mode,
                runtime_features=report["runtime_features"],
                manifest_identity=report["manifest_identity"],
                statistics=report.get("statistics"),
                lifecycle=report["lifecycle"],
                timing_scope=report["timing_scope"],
            ),
        )
        progress(
            f"DONE prepared={report['status']} result={output}/report.json log={output}/run.log"
        )
    return report
