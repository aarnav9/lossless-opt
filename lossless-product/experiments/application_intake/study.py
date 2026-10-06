"""Replay unmodified, pinned pybaselines code through the public application API.

No provider, candidate authoring, application benchmark, or speedup claim.
The script does not download code, install dependencies, or edit the upstream tree.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from lossless.applications import initialize, inspect, replay
from lossless.applications.config import preflight_text


REVISION = "6f8d927e08577a9c5af5f34129b14d57c12725f6"
URL = "https://github.com/derb12/pybaselines"
ALGORITHMS = {
    "asls": ("pybaselines.whittaker:asls", {"lam": 1e5, "p": 0.02}),
    "modpoly": ("pybaselines.polynomial:modpoly", {"poly_order": 3}),
    "snip": (
        "pybaselines.smooth:snip",
        {"max_half_window": 40, "decreasing": True, "smooth_half_window": 3},
    ),
}


def log(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(project, *args):
    return subprocess.check_output(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(project),
            *args,
        ],
        text=True,
        timeout=10,
    ).strip()


def cases(parameters):
    """Own seeded synthetic spectra, including different sampling lengths and a flat case."""
    result = []
    specifications = [
        ("short", "discovery", 256, 7),
        ("medium", "discovery", 1000, 19),
        ("long", "discovery", 4096, 31),
        ("flat", "discovery", 256, 0),
        ("odd_length", "evaluation", 511, 43),
        ("new_noise", "evaluation", 2048, 59),
    ]
    for name, split, size, seed in specifications:
        x = np.linspace(0, 1, size)
        baseline = 1 + 2 * np.exp(-3 * x) + 0.3 * x
        peaks = sum(
            height * np.exp(-0.5 * ((x - center) / width) ** 2)
            for center, width, height in [(0.17, 0.012, 3), (0.46, 0.027, 5), (0.79, 0.018, 2)]
        )
        y = baseline + peaks + np.random.default_rng(seed).normal(0, 0.025, size)
        if name == "flat":
            y = np.full(size, 2.0)
        result.append(
            {
                "id": name,
                "split": split,
                "args": [y.tolist()],
                "kwargs": {**parameters, "x_data": x.tolist()},
            }
        )
    return result


def compact(result):
    report = result.summary
    return {
        "status": report["status"],
        "source_identity": report.get("source_identity"),
        "source_unchanged": report.get("source_unchanged"),
        "elapsed_seconds_diagnostic": report["elapsed_seconds"],
        "llm_calls": report["llm_calls"],
        "optimization_performed": report["optimization_performed"],
        "report_sha256": sha(result.directory / "report.json"),
        "manifest_sha256": sha(result.directory / "manifest.json"),
        "cases": [
            {
                "id": row["id"],
                "repeatable": row["repeatable"],
                "attempts": [
                    {
                        k: attempt[k]
                        for k in ("status", "fingerprint", "error", "message")
                        if k in attempt
                    }
                    for attempt in row["attempts"]
                ],
            }
            for row in report["cases"]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    if output == project or output.is_relative_to(project):
        raise ValueError("study output must be outside the upstream project")
    if git(project, "rev-parse", "HEAD") != REVISION or git(project, "status", "--porcelain"):
        raise ValueError("supply a clean checkout of the pinned pybaselines revision")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    log(f"START result={output / 'summary.json'} log={output.with_suffix('.log')} expected=1-3min")
    initialize(project, output / "draft", python=sys.executable)
    draft = inspect(output / "draft/lossless.json")
    write(output / "intake.json", draft)
    candidates = {row["target"] for row in draft["discovery"]["entry_candidates"]}
    baseline_source = draft["source"]["files"]
    summary = {
        "schema_version": 1,
        "study": "application-intake-pybaselines",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "upstream": {
            "url": URL,
            "revision": REVISION,
            "tag": "v1.2.1",
            "license": "BSD-3-Clause",
            "license_sha256": sha(project / "LICENSE.txt"),
            "bundled_notices_sha256": sha(project / "LICENSES_bundled.txt"),
        },
        "study_source_sha256": sha(Path(__file__)),
        "environment": {k: draft["environment"][k] for k in ("python", "platform", "machine")},
        "relevant_packages": {
            name: draft["environment"]["packages"].get(name)
            for name in ("numpy", "scipy", "numba", "llvmlite", "pentapy")
        },
        "intake_record_sha256": sha(output / "intake.json"),
        "source_files": len(baseline_source),
        "source_bytes": draft["source"]["bytes"],
        "discovered_functions": len(candidates),
        "first_suggestions": [row["target"] for row in draft["discovery"]["entry_candidates"][:12]],
        "discovery_truncated": draft["discovery"]["scan_truncated"],
        "scope": "Unmodified upstream algorithms with our own synthetic spectra. Exact reference repeatability only; no optimization, algorithm-accuracy assessment, clinical data, or speedup benchmark. Evaluation cases are explicitly opened after discovery; they are public and not secret holdouts.",
        "algorithms": {},
    }
    for name, (entry, parameters) in ALGORITHMS.items():
        case_path = output / f"{name}-cases.json"
        write(case_path, cases(parameters))
        initialize(project, output / name, entry=entry, cases=case_path, python=sys.executable)
        config = output / name / "lossless.json"
        info = inspect(config)
        print(preflight_text(info), flush=True)
        function_hint = entry.split(":")[0].replace(".", "/") + ".py:" + name
        record = {
            "entry": entry,
            "suggested_by_discovery": function_hint in candidates,
            "cases_sha256": sha(case_path),
        }
        for split in ("discovery", "evaluation"):
            log(f"ALGORITHM={name} SPLIT={split}")
            result = replay(config, output=output / f"{name}-{split}", split=split, budget=90)
            record[split] = compact(result)
        summary["algorithms"][name] = record
        write(output / "summary.json", summary)

    # An upstream validation error must produce a failed receipt, never a pass.
    invalid = cases(ALGORITHMS["asls"][1])[:1]
    invalid[0]["kwargs"]["p"] = 2.0
    write(output / "invalid-cases.json", invalid)
    initialize(
        project,
        output / "invalid",
        entry=ALGORITHMS["asls"][0],
        cases=output / "invalid-cases.json",
        python=sys.executable,
    )
    result = replay(output / "invalid/lossless.json", output=output / "invalid-replay", budget=30)
    summary["invalid_parameter_control"] = compact(result)
    summary["source_unchanged"] = (
        not git(project, "status", "--porcelain")
        and inspect(output / "draft/lossless.json")["source"]["files"] == baseline_source
    )
    good = all(
        record[split]["status"] == "replayed" and record[split]["source_unchanged"]
        for record in summary["algorithms"].values()
        for split in ("discovery", "evaluation")
    )
    rejected = result.summary["status"] == "failed" and any(
        attempt.get("error") == "ValueError"
        for row in result.summary["cases"]
        for attempt in row["attempts"]
    )
    summary["status"] = "passed" if good and rejected and summary["source_unchanged"] else "failed"
    summary["replayed_cases"] = sum(
        len(record[split]["cases"])
        for record in summary["algorithms"].values()
        for split in ("discovery", "evaluation")
    )
    summary["successful_reference_processes"] = sum(
        attempt["status"] == "completed"
        for record in summary["algorithms"].values()
        for split in ("discovery", "evaluation")
        for row in record[split]["cases"]
        for attempt in row["attempts"]
    )
    summary["elapsed_seconds_diagnostic"] = time.perf_counter() - started
    write(output / "summary.json", summary)
    log(
        f"COMPLETE status={summary['status']} result={output / 'summary.json'} log={output.with_suffix('.log')}"
    )
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
