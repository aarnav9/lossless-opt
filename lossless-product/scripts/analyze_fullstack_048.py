"""Audit completed 048 evidence, calculate payback, and produce a shareable archive."""

import argparse
import base64
from datetime import datetime
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clock_record(root):
    log = root.parent / (root.name + ".log")
    milestones, gaps = {}, []
    previous = None
    for line in log.read_text().splitlines():
        match = re.match(r"\[([^]]+)\] (.*)", line)
        if not match:
            continue
        at, message = datetime.fromisoformat(match[1]), match[2]
        for key, marker in [
            ("start", "048 START phase=run"),
            ("search_start", "048 AUTHOR LOOP START"),
            ("search_sealed", "048 SEALED"),
            ("complete", "048 COMPLETE gate="),
        ]:
            if message.startswith(marker):
                milestones[key] = at
        elapsed = re.search(r"048 author elapsed=(\d+)s", message)
        if elapsed:
            active = int(elapsed[1])
            if previous and active > previous[1]:
                wall_delta = (at - previous[0]).total_seconds()
                active_delta = active - previous[1]
                if wall_delta - active_delta > 5:
                    gaps.append(
                        dict(
                            from_utc=previous[0].isoformat(),
                            to_utc=at.isoformat(),
                            wall_seconds=wall_delta,
                            reported_timer_seconds=active_delta,
                            excess_seconds=wall_delta - active_delta,
                        )
                    )
            previous = at, active
        if message.startswith("048 round="):
            previous = None
    assert set(milestones) == {"start", "search_start", "search_sealed", "complete"}
    return dict(
        milestones={k: v.isoformat() for k, v in milestones.items()},
        search_wall_seconds=(
            milestones["search_sealed"] - milestones["search_start"]
        ).total_seconds(),
        total_wall_seconds=(milestones["complete"] - milestones["start"]).total_seconds(),
        author_clock_gaps=gaps,
        timestamp_resolution_seconds=1,
        note="Controller UTC progress timestamps versus frozen time.monotonic counters. On this Mac the counter pauses during system sleep. A lid-close sleep during author round three was independently observed in the local power log. This is not a strict uninterrupted one-hour wall-time treatment; remote author compute during suspension is unknown.",
    )


def analyze(root, output):
    plan = read(root / "plan.json")
    assert sha(root / "plan.json") == read(root / "plan_sha256.json")["sha256"]
    assert sha(root / "references.json") == read(root / "references_sha256.json")["sha256"]
    assert sha(root / "selection.json") == read(root / "selection_sha256.json")["sha256"]
    for name, digest in plan["source_hashes"].items():
        assert sha(root / "frozen" / name) == digest, name
    spec = importlib.util.spec_from_file_location(
        "frozen_protocol_048", root / "frozen/harness/protocol.py"
    )
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    summary, history = read(root / "summary.json"), read(root / "history.json")
    assert summary["status"] == "complete"
    clocks = clock_record(root)
    selection = summary["selected"]
    if selection:
        assert sha(selection["source"]) == selection["source_sha256"]
    for row in history:
        if "assessment" in row:
            report = read(root / "workers" / (row["round"] + ".json"))
            assert row["assessment"] == protocol.assess(report["records"])
        if row.get("source"):
            assert sha(row["source"]) == row["source_sha256"]
    finals = []
    for i, expected in enumerate(summary["final_assessments"], 1):
        path = root / "workers" / f"final_{i}.json"
        if path.exists() and "failure" not in expected:
            report = read(path)
            assert expected == protocol.assess(report["records"])
            finals.append(report)
    usage = {
        key: sum(row["receipt"]["usage"].get(key) or 0 for row in history)
        for key in ["input_tokens", "output_tokens"]
    }
    usage["calls_missing_usage"] = sum(
        any(row["receipt"]["usage"].get(k) is None for k in ["input_tokens", "output_tokens"])
        for row in history
    )
    usage["cost_usd"] = None
    by_case = []
    for index, case in enumerate(plan["evaluation"]):
        observations = []
        for report in finals:
            record = report["records"][index]
            assert record["case"] == case
            if not all(record["exact"].values()):
                continue
            med = {k: statistics.median(v) for k, v in record["samples"].items()}
            saving = med["reference"] - med["candidate"]
            extra_setup = max(
                0,
                report["setup_seconds"]["candidate"]
                + record["first_call"]["candidate"]["seconds"]
                - record["first_call"]["reference"]["seconds"],
            )
            observations.append(
                dict(
                    reference_seconds=med["reference"],
                    candidate_seconds=med["candidate"],
                    serial_seconds=med["serial"],
                    speedup=med["reference"] / med["candidate"],
                    serial_speedup=med["serial"] / med["candidate"],
                    extra_first_call_and_constructor_seconds=extra_setup,
                    search_setup_payback_calls=math.ceil(
                        (summary["search"]["search_seconds"] + extra_setup) / saving
                    )
                    if saving > 0
                    else None,
                    full_campaign_payback_calls=math.ceil(
                        (clocks["total_wall_seconds"] + extra_setup) / saving
                    )
                    if saving > 0
                    else None,
                )
            )
        by_case.append(
            dict(
                key=case["key"],
                observations=observations,
                geomean_speedup=statistics.geometric_mean(o["speedup"] for o in observations)
                if observations
                else None,
                geomean_serial_speedup=statistics.geometric_mean(
                    o["serial_speedup"] for o in observations
                )
                if observations
                else None,
            )
        )
    analysis = dict(
        campaign="048",
        summary=summary,
        clocks=clocks,
        usage=usage,
        cases=by_case,
        note="Payback uses per-process per-case point savings. Search-timer+extra setup and full UTC-wall-campaign+extra setup are separate numerators; the latter charges suspension time. Shared model loading is reported separately. No positive saving means no finite payback. Timestamp-derived TTFT is implementation-reported; full-call clocks are fixed-controller measurements.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    return analysis


def archive(root, output, analysis, extras):
    repository = Path(__file__).resolve().parents[2]
    sources, records = {}, {}
    locations = [(root, "study"), *[(p, p.name) for p in extras]]

    def scrub(value):
        if isinstance(value, str):
            for path, name in locations:
                value = value.replace(str(path), f"<{name}>")
            return value.replace(str(repository), "<repo>").replace(str(Path.home()), "<home>")
        if isinstance(value, dict):
            return {scrub(k): scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value

    for folder, label in locations:
        for path in sorted(folder.rglob("*.json")):
            if "frozen" in path.relative_to(folder).parts or path.name == "raw_response.json":
                continue
            value = read(path)
            if path.name == "invocation.json":
                argv = value["argv"]
                argv[argv.index("--cd") + 1] = "<temporary-empty-directory>"
            records[f"<{label}>/{path.relative_to(folder)}"] = dict(
                original_sha256=sha(path), record=scrub(value)
            )
        for path in sorted(folder.rglob("*.py")):
            if "frozen" not in path.relative_to(folder).parts:
                sources[f"<{label}>/{path.relative_to(folder)}"] = dict(
                    sha256=sha(path), source=path.read_text()
                )
    plan = read(root / "plan.json")
    for name, digest in plan["source_hashes"].items():
        path = root / "frozen" / name
        try:
            record = dict(source=path.read_text())
        except UnicodeDecodeError:
            record = dict(base64=base64.b64encode(path.read_bytes()).decode())
        sources[f"<study>/frozen/{name}"] = dict(sha256=digest, **record)
    for companion in [Path(__file__), Path(__file__).with_name("plot_fullstack_048.py")]:
        sources[str(companion.relative_to(repository))] = dict(
            sha256=sha(companion), source=companion.read_text()
        )
    log = root.parent / (root.name + ".log")
    records["<study>/controller.log"] = dict(original_sha256=sha(log), text=scrub(log.read_text()))
    value = dict(
        schema_version=1,
        campaign="048",
        analysis=scrub(analysis),
        records=records,
        sources=sources,
        scope="Original digests precede path redaction in JSON records; frozen source/binary resources preserve original bytes. Includes raw timing/check records, proposal source, prompts and CLI receipts. Excludes CLI event streams/stderr/auth databases/model weights. Byte digests commit full observed arrays but do not reproduce their contents. Recompute them using the pinned model and worker. One author trajectory on one device; no product promotion.",
    )
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    assert str(Path.home()).encode() not in encoded, "local home path needs review"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(gzip.compress(encoded, mtime=0))
    return dict(
        path=str(output),
        sha256=sha(output),
        bytes=output.stat().st_size,
        uncompressed_bytes=len(encoded),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--extra", type=Path, action="append", default=[])
    a = p.parse_args()
    result = analyze(a.study.resolve(), a.analysis.resolve())
    print(
        json.dumps(
            archive(a.study.resolve(), a.archive.resolve(), result, [p.resolve() for p in a.extra])
        )
    )


if __name__ == "__main__":
    main()
