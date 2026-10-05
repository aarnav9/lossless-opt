"""Audit the frozen KernelBench-derived study and publish compact, redacted evidence."""

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
import sys


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(root):
    sys.path.insert(0, str(root / "frozen/src"))
    for name in ["plan", "references", "selection"]:
        assert sha(root / (name + ".json")) == read(root / (name + "_sha256.json"))["sha256"]
    plan, selection = read(root / "plan.json"), read(root / "selection.json")
    for name, digest in plan["source_hashes"].items():
        assert sha(root / "frozen" / name) == digest, name
    for split in ["discovery", "evaluation"]:
        folder = root / ("oracle_" + split)
        for item in read(folder / "index.json")["records"]:
            assert sha(folder / item["file"]) == item["sha256"], item["file"]
    spec = importlib.util.spec_from_file_location(
        "protocol049", root / "frozen/harness/protocol.py"
    )
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    summary, history = read(root / "summary.json"), read(root / "history.json")
    assert summary["status"] == "complete"
    assert sha(selection["source"]) == selection["source_sha256"]
    for row in history:
        if row.get("source"):
            assert sha(row["source"]) == row["source_sha256"]
        if row.get("assessment"):
            assert row["assessment"] == protocol.assess(
                read(root / "workers" / (row["round"] + ".json"))["records"]
            )
    finals = [
        read(root / "workers" / f"final_{i + 1}.json") for i in range(plan["final_processes"])
    ]
    for report, expected in zip(finals, summary["final_assessments"]):
        assert protocol.assess(report["records"]) == expected
    baseline = read(root / "workers/baseline.json")
    references = read(root / "references.json")
    for task, chosen in references.items():
        rows = [r for r in baseline["records"] if r["case"]["task"] == task]
        eligible = [
            m for m in plan["baseline_methods"] if all(r["checks"][m]["pass"] for r in rows)
        ]
        assert (
            min(
                eligible,
                key=lambda m: statistics.geometric_mean(
                    statistics.median(r["samples"][m]) for r in rows
                ),
            )
            == chosen
        )
    milestones = {}
    for line in (root.parent / (root.name + ".log")).read_text().splitlines():
        match = re.match(r"\[([^]]+)\] 049 (.*)", line)
        if match:
            for key, prefix in [
                ("start", "START phase=run"),
                ("search_start", "AUTHOR LOOP START"),
                ("sealed", "SEALED"),
                ("complete", "COMPLETE gate="),
            ]:
                if match[2].startswith(prefix):
                    milestones[key] = datetime.fromisoformat(match[1])
    wall = (milestones["complete"] - milestones["start"]).total_seconds()
    usage = {
        k: sum(row["receipt"]["usage"].get(k) or 0 for row in history)
        for k in ["input_tokens", "output_tokens"]
    }
    usage["calls_missing_usage"] = sum(
        any(row["receipt"]["usage"].get(k) is None for k in ["input_tokens", "output_tokens"])
        for row in history
    )
    usage["cost_usd"] = None
    discovery = {r["task"]: r for r in (selection.get("assessment") or {}).get("tasks", [])}
    task_rows = []
    for task, label in plan["tasks"].items():
        observations = []
        confirmations = []
        for p, report in enumerate(finals):
            task_assessment = next(
                r for r in summary["final_assessments"][p]["tasks"] if r["task"] == task
            )
            confirmations.append(task_assessment["confirmed"])
            for record in report["records"]:
                if record["case"]["task"] != task:
                    continue
                valid = all(
                    record["checks"].get(n, {}).get("pass", False)
                    for n in ["reference", "candidate"]
                )
                samples = record["samples"]
                med = {n: statistics.median(v) for n, v in samples.items() if v}
                if valid:
                    saving = med["reference"] - med["candidate"]
                    extra = max(
                        0,
                        report["import_seconds"]
                        + record["setup_seconds"]["candidate"]
                        + record["first_call"]["candidate"]["seconds"]
                        - record["setup_seconds"]["reference"]
                        - record["first_call"]["reference"]["seconds"],
                    )
                    payback = (
                        math.ceil((summary["search_seconds"] + extra) / saving)
                        if saving > 0
                        else None
                    )
                    campaign_payback = math.ceil((wall + extra) / saving) if saving > 0 else None
                    setup_payback = math.ceil(extra / saving) if saving > 0 else None
                else:
                    payback, campaign_payback, setup_payback, extra = None, None, None, None
                observations.append(
                    dict(
                        process=p + 1,
                        case=record["case"]["id"],
                        correct=valid,
                        median_seconds=med,
                        speedup=med["reference"] / med["candidate"] if valid else None,
                        eager_speedup=med["eager"] / med["candidate"]
                        if valid and record["checks"]["eager"]["pass"]
                        else None,
                        extra_setup_seconds=extra,
                        search_setup_payback_calls=payback,
                        campaign_setup_payback_calls=campaign_payback,
                        setup_only_payback_calls=setup_payback,
                        constructor_seconds=record["setup_seconds"],
                        first_call=record["first_call"],
                        max_peak_bytes={n: max(v) for n, v in record["peaks"].items() if v},
                    )
                )
        ok = all(r["correct"] for r in observations) and len(observations) == 4
        task_rows.append(
            dict(
                task=task,
                label=label,
                reference=references[task],
                correct=ok,
                geomean=statistics.geometric_mean(r["speedup"] for r in observations)
                if ok
                else None,
                eager_geomean=statistics.geometric_mean(r["eager_speedup"] for r in observations)
                if ok and all(r["eager_speedup"] for r in observations)
                else None,
                discovery_confirmed=discovery.get(task, {}).get("confirmed", False),
                final_process_confirmed=confirmations,
                confirmed=ok
                and discovery.get(task, {}).get("confirmed", False)
                and all(confirmations),
                observations=observations,
            )
        )
    return dict(
        campaign="049",
        summary=summary,
        tasks=task_rows,
        usage=usage,
        clock=dict(
            milestones={k: v.isoformat() for k, v in milestones.items()},
            campaign_wall_seconds=wall,
            search_wall_seconds=(milestones["sealed"] - milestones["search_start"]).total_seconds(),
            sleep_inclusive_counter_seconds=summary["total_seconds"],
        ),
        confirmed_tasks=sum(r["confirmed"] for r in task_rows),
        correct_tasks=sum(r["correct"] for r in task_rows),
        task_denominator=20,
        note="One shared author trajectory; M2-sized ports, not official KernelBench scores. Numerical tolerance, no bitwise claim. All 20 tasks remain in denominator. Payback charges the whole shared search (or entire campaign) to one repeatedly called case, plus positive extra candidate module import/constructor/first-call setup relative to reference constructor/first-call. Point savings are not confidence guarantees. Cases with nonpositive savings have null payback.",
    )


def archive(root, output, analysis, extra):
    repo = Path(__file__).resolve().parents[2]
    sources, records = {}, {}
    locations = [(root, "study"), *[(p, p.name) for p in extra]]

    def scrub(value):
        if isinstance(value, str):
            for path, name in locations:
                value = value.replace(str(path), f"<{name}>")
            return value.replace(str(repo), "<repo>").replace(str(Path.home()), "<home>")
        if isinstance(value, dict):
            return {scrub(k): scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value

    for folder, label in locations:
        for path in sorted(folder.rglob("*.json")):
            if "frozen" in path.relative_to(folder).parts or path.name in {
                "raw_response.json",
                "analysis.json",
            }:
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
            content = dict(source=path.read_text())
        except UnicodeDecodeError:
            content = dict(base64=base64.b64encode(path.read_bytes()).decode())
        sources[f"<study>/frozen/{name}"] = dict(sha256=digest, **content)
    for path in [Path(__file__), Path(__file__).with_name("plot_kernelbench_049.py")]:
        sources[str(path.relative_to(repo))] = dict(sha256=sha(path), source=path.read_text())
    log = root.parent / (root.name + ".log")
    records["<study>/controller.log"] = dict(original_sha256=sha(log), text=scrub(log.read_text()))
    value = dict(
        schema_version=1,
        campaign="049",
        analysis=scrub(analysis),
        sources=sources,
        records=records,
        scope="Pinned MIT upstream sources/license, original frozen harness, author proposals/prompts/receipts, raw timings, check diagnostics and oracle fixture identities. JSON record digests precede local path redaction. Fixture arrays are excluded to bound archive size; regenerate using pinned oracle source, seeds, parameters and CPU PyTorch runtime. No raw CLI streams, stderr or auth state. Numerical MLX study on one M2, not official CUDA scores; no product promotion.",
    )
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    assert str(Path.home()).encode() not in encoded
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(gzip.compress(encoded, mtime=0))
    return dict(path=str(output), sha256=sha(output), bytes=output.stat().st_size)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--extra", type=Path, action="append", default=[])
    a = p.parse_args()
    value = analyze(a.study.resolve())
    a.analysis.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            archive(a.study.resolve(), a.archive.resolve(), value, [p.resolve() for p in a.extra])
        )
    )
