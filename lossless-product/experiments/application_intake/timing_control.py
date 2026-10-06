"""Identical-source A/A timing diagnostic on discovery cases; never selects code."""

import argparse
import json
import os
from pathlib import Path

from lossless.applications.comparison import compare
from lossless.applications.replay import copy_project
from lossless.applications.runtime import clock

from study import log, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blas-threads", type=int, choices=[1, 8])
    args = parser.parse_args()
    original, out = args.reference_run.resolve(), args.output.resolve()
    if out == original or out.is_relative_to(original) or original.is_relative_to(out):
        raise ValueError("control output must be separate from reference run")
    out.mkdir(parents=True, exist_ok=False)
    value = json.loads((original / "resolved.json").read_text())
    snapshot = json.loads((original / "source.json").read_text())
    cases = [
        c
        for c in json.loads((original / "cases.json").read_text())
        if c["split"] == "discovery" and c["id"] in {"long", "weighted", "flat"}
    ]
    if len(cases) != 3:
        raise ValueError("requires the pybaselines discovery-case suite")
    if args.blas_threads:
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.blas_threads)
        value["environment"]["inherit_env"] = sorted(
            set(value["environment"]["inherit_env"]) | {"OPENBLAS_NUM_THREADS"}
        )
    write(
        out / "protocol.json",
        {
            "kind": "identical_source_timing_control",
            "cases": [c["id"] for c in cases],
            "pairs": 8,
            "budget_seconds": 180,
            "OPENBLAS_NUM_THREADS": args.blas_threads,
            "script_sha256": sha(Path(__file__)),
            "scope": "Diagnostic only; no candidate selection, LLM calls or final-case execution.",
        },
    )
    copy_project(original / "reference", out / "identical", snapshot)
    log(
        f"START expected=1min result={out / 'comparison/comparison.json'} log={out.with_suffix('.log')}"
    )
    result = compare(
        original / "reference",
        out / "identical",
        snapshot,
        snapshot,
        value,
        cases,
        out / "comparison",
        pairs=8,
        deadline=clock() + 180,
        progress=log,
        minimum_speedup=1.02,
    )
    log(
        f"DONE status={result['status']} result={out / 'comparison/comparison.json'} log={out.with_suffix('.log')}"
    )
    return 0 if result["status"] == "matched" else 2


if __name__ == "__main__":
    raise SystemExit(main())
