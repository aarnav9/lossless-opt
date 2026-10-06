"""Application profiling orchestration and optional evidence-linked assessment."""

from dataclasses import dataclass
import html
import json
from pathlib import Path
from statistics import median
from urllib.parse import quote

from .._native.common import now, stamp, write
from ..jobs import positive
from .config import analysis_settings, resolve
from .replay import replay
from .runtime import clock


@dataclass
class ApplicationProfileResult:
    directory: Path
    summary: dict


def summarize(row):
    details = [attempt["details"] for attempt in row["attempts"]]
    clean = [d for d in details if d["profile_mode"] == "timing"]
    traced = next(d for d in details if d["profile_mode"] == "attribution")
    memory = next(d for d in details if d["profile_mode"] == "memory")
    samples = [d["sequence_seconds"] for d in clean]
    call_medians = [
        median(d["calls"][i]["measurement"]["call_seconds"] for d in clean)
        for i in range(len(clean[0]["calls"]))
    ]
    return {
        "id": row["id"],
        "observations_repeatable": row["repeatable"],
        "timing": {
            "sequence_samples_seconds": samples,
            "median_sequence_seconds": median(samples),
            "first_call_median_seconds": call_medians[0],
            "per_call_median_seconds": call_medians,
            "calls_per_sequence": len(call_medians),
            "continuation_scope": "Continuation timing exists only for explicitly supplied multi-call cases; calls may do different work. No implicit warmup.",
        },
        "setup": {
            "samples_seconds": [d["setup_seconds"] for d in clean],
            "median_seconds": median(d["setup_seconds"] for d in clean),
            "scope": "Application import/factory/synchronization, excluding interpreter and harness imports. Process elapsed time is separately retained in attempt receipts.",
        },
        "synchronization": clean[0]["synchronization"],
        "attribution": {
            **traced["attribution"],
            "instrumented_sequence_seconds": traced["sequence_seconds"],
            "instrumented_over_clean_ratio": traced["sequence_seconds"] / median(samples)
            if median(samples)
            else None,
            "overhead_scope": "One separately instrumented process divided by clean median; noisy diagnostic, not an optimization speedup.",
        },
        "memory": {
            "process_peak_rss_bytes": max(
                c["measurement"]["process_peak_rss_bytes"] for d in clean for c in d["calls"]
            ),
            "traced_calls": [c["measurement"]["traced_memory"] for c in memory["calls"]],
            "scope": "RSS is a lifetime process high-water mark including imports, fixtures and prior observations. Tracemalloc runs separately: traced live peak/retained allocation, not total allocated bytes, copy traffic or GPU memory.",
        },
    }


def save_report(directory, summary):
    write(directory / "report.json", summary)
    escape = lambda x: html.escape(str(x), quote=True)

    def table(headers, rows):
        return (
            "<table><tr>"
            + "".join("<th>" + escape(c) + "</th>" for c in headers)
            + "</tr>"
            + "".join(
                "<tr>" + "".join("<td>" + escape(c) + "</td>" for c in row) + "</tr>"
                for row in rows
            )
            + "</table>"
        )

    body = [
        "<h1>Application profile</h1><p>Status: <strong>"
        + escape(summary["status"])
        + "</strong></p>",
        "<p>" + escape(summary["measurement_scope"]) + "</p>",
    ]
    for case in summary.get("profiles", []):
        body += [
            "<section><h2>" + escape(case["id"]) + "</h2>",
            table(
                [
                    "Setup median ms",
                    "Declared sequence median ms",
                    "Calls per sequence",
                    "Process peak RSS MiB",
                ],
                [
                    [
                        f"{case['setup']['median_seconds'] * 1000:.3f}",
                        f"{case['timing']['median_sequence_seconds'] * 1000:.3f}",
                        case["timing"]["calls_per_sequence"],
                        f"{case['memory']['process_peak_rss_bytes'] / 2**20:.1f}",
                    ]
                ],
            ),
            "<p>" + escape(case["memory"]["scope"]) + "</p>",
            "<p>Profiler/clean time ratio: "
            + escape(case["attribution"]["instrumented_over_clean_ratio"])
            + ". This is instrumentation overhead and process variation, not speedup.</p>",
            table(
                ["Function", "File", "Line", "Self ms", "Instrumented self share"],
                [
                    [
                        r["function"],
                        r["file"],
                        r["line"],
                        f"{r['self_seconds'] * 1000:.3f}",
                        f"{r['self_fraction'] * 100:.1f}%",
                    ]
                    for r in case["attribution"]["top_self"][:15]
                ],
            ),
        ]
        links = {
            (r["file"], r["line"])
            for r in case["attribution"]["top_self"]
            if r["origin"] == "project"
        }
        body += [
            "<p>Frozen source: "
            + ", ".join(
                '<a href="reference/'
                + quote(file, safe="/")
                + '">'
                + escape(f"{file}:{line}")
                + "</a>"
                for file, line in sorted(links)
            )
            + "</p></section>"
        ]
    assessment = summary.get("assessment", {"status": "disabled"})
    body += ["<h2>Model assessment: " + escape(assessment["status"]) + "</h2>"]
    if "response" in assessment:
        response = assessment["response"]
        body += [
            "<p>Advisory hypotheses; evidence references checked. No optimization has been applied.</p>",
            "<p>" + escape(response["summary"]) + "</p>",
        ]
        for item in response["opportunities"]:
            body += [
                "<section><h3>"
                + escape(item["hypothesis"])
                + "</h3><p>"
                + escape(item["proposed_change"])
                + "</p><p>Validation: "
                + escape(item["validation"])
                + "</p><p>Evidence: "
                + escape(", ".join(item["evidence_ids"]))
                + "</p></section>"
            ]
        body += [
            "<ul>"
            + "".join("<li>" + escape(s) + "</li>" for s in response["limitations"])
            + "</ul>"
        ]
    elif assessment.get("reason"):
        body += ["<p>" + escape(assessment["reason"]) + "</p>"]
    body += [
        "<details><summary>Full evidence and raw measurements</summary><pre>"
        + escape(json.dumps(summary, indent=2))
        + "</pre></details>"
    ]
    (directory / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Lossless application profile</title>"
        "<style>body{font:16px system-ui;max-width:85em;margin:2em auto;padding:0 1em;color:#18222d}table{border-collapse:collapse;display:block;overflow:auto}td,th{padding:.6em;border-bottom:1px solid #ddd;text-align:left}section{margin:2em 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}p{line-height:1.5}</style>"
        + "".join(body)
    )


def profile(config, *, repeats=None, budget=None, output=None, analysis=None):
    started = clock()
    resolved, _ = resolve(config)
    if not resolved["entry"] or resolved["entry"]["kind"] != "callable":
        raise ValueError(
            "application profiling currently requires a callable entry; command replay remains available"
        )
    repeats = resolved["profile"]["repeats"] if repeats is None else repeats
    if type(repeats) is not int or not 3 <= repeats <= 50:
        raise ValueError("profile repeats must be between 3 and 50")
    budget = positive(
        resolved["profile"]["wall_time_seconds"] if budget is None else budget, "profile budget"
    )
    settings = analysis_settings(
        None
        if analysis is False
        else analysis
        if analysis is not None
        else resolved.get("analysis")
    )
    directory = (
        Path(output).resolve()
        if output
        else Path(config).resolve().parent
        / "runs"
        / ("profile-" + now().replace(":", "").replace("+", "_"))
    )
    modes = ["timing"] * repeats + ["attribution", "memory"]
    result = replay(
        config,
        output=directory,
        budget=max(0.000001, budget - (clock() - started)),
        _profile={"modes": modes},
    )
    summary = result.summary
    summary["measurement_scope"] = (
        "Discovery cases only. Fresh-process clean timings, separate cProfile and tracemalloc passes, exact observed repeatability. Timed calls include completion synchronization; fixture loading, fingerprinting and observer hooks are outside timers. Observation between sequence calls can perturb caches. No candidate search, performance acceptance or speedup claim. Copy traffic and device kernel attribution are not measured."
    )
    summary["scope"] = summary["measurement_scope"]
    summary["profiles"] = []
    summary["assessment"] = {"status": "disabled", "calls": 0}
    summary["effective_settings"] = {
        "repeats": repeats,
        "total_seconds": budget,
        "analysis": settings,
    }

    def progress(message):
        stamp(message)
        with (directory / "run.log").open("a") as stream:
            stream.write(f"[{now()}] {message}\n")

    try:
        if summary["status"] == "replayed":
            summary["profiles"] = [summarize(row) for row in summary["cases"]]
            summary["status"] = "profiled"
            save_report(directory, summary)
            if settings:
                if clock() >= started + budget:
                    summary["assessment"] = {
                        "status": "not_run",
                        "reason": "total budget exhausted",
                        "calls": 0,
                    }
                else:
                    from .assessment import assess

                    progress(
                        f"ASSESS model={settings['model']} effort={settings['effort']} response_allowance={min(settings['timeout_seconds'], started + budget - clock()):.1f}s; bounded discovery evidence and source excerpts"
                    )
                    summary["assessment"] = assess(summary, directory, settings, started + budget)
        elif settings:
            summary["assessment"] = {
                "status": "not_run",
                "reason": "reference measurements failed or were unstable",
                "calls": 0,
            }
    except KeyboardInterrupt:
        summary["assessment"] = {
            "status": "incomplete",
            "reason": "interrupted",
            "calls": int((directory / "assessment/invocation.json").exists()),
        }
    except Exception as error:
        summary["assessment"] = {
            "status": "failed",
            "reason": str(error),
            "calls": int((directory / "assessment/invocation.json").exists()),
        }
        if summary["status"] != "profiled":
            summary.update(status="failed", reason=str(error))
    finally:
        summary["llm_calls"] = summary["assessment"]["calls"]
        summary["elapsed_seconds"] = clock() - started
        save_report(directory, summary)
        progress(
            f"COMPLETE profile={summary['status']} assessment={summary['assessment']['status']} result={directory / 'report.html'} log={directory / 'run.log'}"
        )
    return ApplicationProfileResult(directory, summary)
