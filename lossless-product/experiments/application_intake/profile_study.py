"""Profile the pinned external repository and run one live Codex assessment."""

import argparse
import json
from pathlib import Path
import sys
import time

import lossless
from lossless.applications import initialize

from study import ALGORITHMS, REVISION, URL, cases, git, log, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument(
        "--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"]
    )
    parser.add_argument("--llm-timeout", type=float, default=1800)
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    if git(project, "rev-parse", "HEAD") != REVISION or git(project, "status", "--porcelain"):
        raise ValueError("expected a clean pinned upstream checkout")
    if output == project or output.is_relative_to(project):
        raise ValueError("output must be outside the upstream checkout")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    log(
        f"START result={output / 'summary.json'} log={output.with_suffix('.log')} expected=3-8min; one model call unless --no-model"
    )
    summary = {
        "study": "external-application-profile",
        "upstream": {"url": URL, "revision": REVISION},
        "study_source_sha256": sha(Path(__file__)),
        "case_generator_sha256": sha(Path(__file__).with_name("study.py")),
        "scope": "Unmodified library callable profiling plus one advisory model assessment. Synthetic spectra, not production traffic. No candidate optimization or speedup claim.",
        "profiles": {},
    }
    # Finish deterministic timing before the live provider so provider work cannot
    # contend with another measured algorithm in this study.
    for name in ("asls", "snip", "modpoly"):
        entry, parameters = ALGORITHMS[name]
        original_cases = cases(parameters)
        medium, large = original_cases[1], original_cases[2]
        selected = [
            {**large, "id": "single_spectrum"},
            {
                "id": "batch_eight",
                "split": "discovery",
                "calls": [{"args": medium["args"], "kwargs": medium["kwargs"]} for _ in range(8)],
            },
            original_cases[-1],  # Evaluation is deliberately kept closed.
        ]
        case_path = output / f"{name}-cases.json"
        write(case_path, selected)
        job = output / f"{name}-job"
        initialize(project, job, entry=entry, cases=case_path, python=sys.executable)
        analysis = None
        if name == "modpoly" and not args.no_model:
            analysis = {
                "provider": "codex-chatgpt",
                "model": args.model,
                "effort": args.effort,
                "timeout_seconds": args.llm_timeout,
            }
        log(f"PROFILE algorithm={name} assessment={bool(analysis)}")
        result = lossless.profile(
            job / "lossless.json",
            repeats=5,
            budget=args.llm_timeout + 90 if analysis else 90,
            output=output / f"{name}-profile",
            analysis=analysis,
        )
        report = result.summary
        summary["profiles"][name] = {
            "status": report["status"],
            "cases": report["profiles"],
            "source_identity": report.get("source_identity"),
            "source_unchanged": report.get("source_unchanged"),
            "llm_calls": report["llm_calls"],
            "assessment": report["assessment"],
            "elapsed_seconds": report["elapsed_seconds"],
            "report_sha256": sha(result.directory / "report.json"),
            "manifest_sha256": sha(result.directory / "manifest.json"),
            "runner_sha256": json.loads((result.directory / "manifest.json").read_text())["runner"],
            "environment": {
                key: report["environment"][key] for key in ("python", "platform", "machine")
            },
            "packages": {
                key: report["environment"]["packages"].get(key)
                for key in ("numpy", "scipy", "numba", "llvmlite", "pentapy")
            },
            "cases_sha256": sha(case_path),
        }
        if (result.directory / "assessment_context.json").exists():
            context = json.loads((result.directory / "assessment_context.json").read_text())
            summary["profiles"][name]["assessment_context_sha256"] = sha(
                result.directory / "assessment_context.json"
            )
            summary["profiles"][name]["assessment_evidence_ids"] = list(context["evidence"])
        write(output / "summary.json", summary)
    summary["source_unchanged"] = not git(project, "status", "--porcelain")
    good = all(
        r["status"] == "profiled" and r["source_unchanged"] for r in summary["profiles"].values()
    )
    advice = args.no_model or summary["profiles"]["modpoly"]["assessment"]["status"] == "completed"
    summary["status"] = "passed" if good and advice and summary["source_unchanged"] else "failed"
    summary["elapsed_seconds"] = time.perf_counter() - started
    write(output / "summary.json", summary)
    log(
        f"COMPLETE status={summary['status']} result={output / 'summary.json'} log={output.with_suffix('.log')}"
    )
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
