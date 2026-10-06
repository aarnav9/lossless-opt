"""Bounded profiling of admitted workload interfaces, without searching or providers."""

from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from .jobs import Workload, positive, read_json
from ._native.common import stamp, write


@dataclass
class ProfileResult:
    directory: Path
    summary: dict


def save_report(directory, summary):
    write(directory / "report.json", summary)
    escape = lambda value: html.escape(str(value))

    def table(headers, rows):
        return (
            "<div class='scroll'><table><thead><tr>"
            + "".join("<th>" + escape(v) + "</th>" for v in headers)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join("<td>" + escape(v) + "</td>" for v in row) + "</tr>"
                for row in rows
            )
            + "</tbody></table></div>"
        )

    sections = []
    for case in summary.get("cases", []):
        sections.append("<section><h2>" + escape(case.get("label", "Request batch")) + "</h2>")
        overview = []
        for name, value in case.get("implementations", {}).items():
            warm = value["warm"]
            overview.append(
                [
                    name,
                    f"{value['first_call_seconds'] * 1000:.3f}",
                    f"{warm['median_seconds'] * 1000:.3f}",
                    f"{warm['p95_seconds'] * 1000:.3f}",
                    f"{warm['requests_per_second']:.2f}",
                    f"{warm['units_per_second']:.1f} {warm['unit']}/s"
                    if warm.get("units_per_second") is not None
                    else "—",
                ]
            )
        sections.append(
            table(
                [
                    "Implementation",
                    "First call ms",
                    "Warm median ms",
                    "Warm p95 ms",
                    "Requests/s",
                    "Work/s",
                ],
                overview,
            )
        )
        if "payback" in case:
            payback = case["payback"]
            count = payback["break_even_calls"]
            sections.append(
                "<p><strong>Estimated payback: "
                + (f"{count:,} repeated calls/batches" if count is not None else "not established")
                + ".</strong> "
                + escape(payback["reason"])
                + "</p><p>"
                + escape(payback.get("scope", ""))
                + "</p>"
            )
        if "memory" in case:
            sections.append(
                f"<p>Worker peak RSS: {case['memory']['process_peak_rss_bytes'] / 2**20:.1f} MiB. "
                + escape(case["memory"]["scope"])
                + "</p>"
            )
        for name, value in case.get("implementations", {}).items():
            sections.append("<h3>" + escape(name) + "</h3>")
            if value.get("fallback_reason"):
                sections.append("<p>Fallback: " + escape(value["fallback_reason"]) + "</p>")
            if "bound_warm" in value:
                sections.append(
                    f"<p>Bound-buffer median: {value['bound_warm']['median_seconds'] * 1000:.3f} ms. "
                    f"Measured binding setup: {value['binding_seconds'] * 1000:.3f} ms; load: {value['load_seconds'] * 1000:.3f} ms.</p>"
                )
            if "peak_mlx_allocated_bytes" in value:
                sections.append(
                    f"<p>Peak MLX allocation during warm calls: {value['peak_mlx_allocated_bytes'] / 2**20:.1f} MiB.</p>"
                )
            if "stages" in value:
                sections.append(
                    table(
                        ["Request stage", "Mean ms", "Share of measured wall time"],
                        [
                            [
                                v["name"],
                                f"{v['mean_seconds'] * 1000:.3f}",
                                f"{v['fraction'] * 100:.1f}%",
                            ]
                            for v in value["stages"]
                        ],
                    )
                )
                sections.append(
                    table(
                        ["Request", "Median first token ms", "Median completion ms"],
                        [
                            [i + 1, f"{a * 1000:.3f}", f"{b * 1000:.3f}"]
                            for i, (a, b) in enumerate(
                                zip(
                                    value["median_ttft_seconds_by_request"],
                                    value["median_completion_seconds_by_request"],
                                )
                            )
                        ],
                    )
                )
            attribution = value["host_attribution"]
            sections.append(
                "<details><summary>Host execution and waiting hotspots</summary><p>"
                + escape(attribution["scope"])
                + "</p>"
                + table(
                    ["Function", "Self ms per call", "Share of instrumented host time"],
                    [
                        [
                            v["function"],
                            f"{v['self_seconds_per_request'] * 1000:.3f}",
                            f"{v['self_fraction'] * 100:.1f}%",
                        ]
                        for v in attribution["top_self_time"]
                    ],
                )
                + "</details>"
            )
        sections.append("</section>")
    details = escape(json.dumps(summary, indent=2, allow_nan=False))
    (directory / "report.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>Lossless profile</title>"
        "<style>body{max-width:80em;margin:3em auto;font:16px system-ui;padding:0 1em;color:#18222d}"
        "table{border-collapse:collapse;width:100%}td,th{padding:.6em;text-align:left;border-bottom:1px solid #ddd}"
        ".scroll{overflow-x:auto}section{margin:2em 0;padding-top:1em;border-top:2px solid #ccd3da}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:1em 0}p{line-height:1.5}</style>"
        "<h1>Lossless profile</h1>" + f"<p>Status: <strong>{escape(summary['status'])}</strong>. "
        "Discovery inputs only. No search or acceptance decision is performed.</p>"
        + "<p>"
        + escape(summary.get("measurement_scope", summary.get("reason", "")))
        + "</p>"
        + "".join(sections)
        + "<details><summary>Full measurements and provenance</summary>"
        + f"<pre>{details}</pre></details></html>\n"
    )


def profile(workload, *, artifact=None, repeats=None, budget=None, output=None, analysis=None):
    config = workload.config if isinstance(workload, Workload) else read_json(workload)
    if isinstance(config, dict) and config.get("kind") == "application":
        if artifact is not None:
            raise ValueError("application profiling does not compare deployment artifacts yet")
        if isinstance(workload, Workload):
            raise ValueError("pass the application configuration filename to profile")
        from .applications.profiling import profile as profile_application

        return profile_application(
            workload, repeats=repeats, budget=budget, output=output, analysis=analysis
        )
    if analysis not in (None, False):
        raise ValueError("model assessment is available for application profiles only")
    repeats = 7 if repeats is None else repeats
    if sys.platform not in {"darwin", "linux"}:
        raise ValueError("profiling currently requires macOS or Linux")
    if type(repeats) is not int or not 3 <= repeats <= 50:
        raise ValueError("profile repeats must be between 3 and 50")
    if not isinstance(workload, Workload):
        workload = Workload.from_config(workload)
    resolved = workload.resolve()
    limit = positive(
        budget if budget is not None else resolved["budget"]["wall_time_seconds"], "profile budget"
    )
    # Providers and proof setup are not part of profiling, and provider configuration
    # does not belong in profiling artifacts.
    resolved.pop("llm", None)
    resolved["proofs"] = {"check": False}
    artifact = Path(artifact).resolve() if artifact is not None else None
    if artifact is not None:
        manifest = read_json(artifact / "manifest.json")
        if manifest.get("adapter") != resolved["workload"]["adapter"]:
            raise ValueError("profile workload and artifact adapters differ")
    directory = (
        Path(output).resolve()
        if output
        else Path(resolved["output"]["directory"])
        / ("profile_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    )
    directory.mkdir(parents=True, exist_ok=False)
    write(
        directory / "profile_job.json",
        {"resolved": resolved, "repeats": repeats, "artifact": str(artifact) if artifact else None},
    )
    started = time.monotonic()
    stamp(
        f"START profile result={directory / 'report.json'} log={directory / 'run.log'} budget={limit}s"
    )
    env = {
        k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "TMPDIR", "LANG", "SYSTEMROOT"}
    }
    env.update(
        PYTHONPATH=str(Path(__file__).resolve().parent.parent),
        PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        VECLIB_MAXIMUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    with (directory / "run.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "lossless._profile_worker", str(directory)],
            cwd=directory,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        def relay():
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)

        reader = threading.Thread(target=relay, daemon=True)
        reader.start()
        incomplete = False
        try:
            process.wait(timeout=limit)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            incomplete = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        finally:
            reader.join(timeout=2)
            process.stdout.close()
    if incomplete:
        summary = (
            read_json(directory / "report.json")
            if (directory / "report.json").exists()
            else {"kind": "profile", "cases": []}
        )
        summary.update(
            status="incomplete",
            reason="profile budget exhausted or interrupted",
            wall_time_seconds=time.monotonic() - started,
        )
        save_report(directory, summary)
    elif process.returncode:
        raise RuntimeError(f"profile worker failed; inspect {directory / 'run.log'}")
    else:
        summary = read_json(directory / "report.json")
        summary["wall_time_seconds"] = time.monotonic() - started
        save_report(directory, summary)
    stamp(
        f"COMPLETE profile status={summary['status']} result={directory / 'report.json'} log={directory / 'run.log'}"
    )
    return ProfileResult(directory, summary)
