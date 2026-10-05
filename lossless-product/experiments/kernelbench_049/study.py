"""Frozen 20-task KernelBench-derived MLX study with a sleep-inclusive search deadline."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import traceback

from clocking import elapsed
from protocol import ATOL, RTOL, METHODS, assess, digest, read, source_check, write
from tasks import TASKS, cases
from lossless._native.common import stamp

HERE = Path(__file__).resolve().parent
PRODUCT = HERE.parents[1]


def freeze(out, seconds):
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        HERE, out / "frozen/harness", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copytree(
        PRODUCT / "src/lossless",
        out / "frozen/src/lossless",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    plan = dict(
        campaign="049",
        created_utc=datetime.now(timezone.utc).isoformat(),
        upstream=read(HERE / "upstream/manifest.json"),
        tasks=TASKS,
        discovery=cases("discovery"),
        evaluation=cases("evaluation"),
        author_model="gpt-6-astra",
        effort="xhigh",
        search_seconds=seconds,
        response_cap_seconds=1800,
        max_calls=16,
        measurement_reserve_seconds=120,
        minimum_call_seconds=120,
        worker_timeout_seconds=240,
        oracle_timeout_seconds=240,
        worker_rss_limit_bytes=4 * 1024**3,
        repeats=7,
        final_repeats=11,
        final_processes=2,
        atol=ATOL,
        rtol=RTOL,
        baseline_methods=METHODS,
        clock="mach_continuous_time on macOS; CLOCK_BOOTTIME on Linux",
        contract="float32, exact shape/dtype, finite outputs, per-element abs(candidate-reference) <= 1e-4 + 1e-4*abs(reference) against BOTH original CPU PyTorch Model and fixed MLX port. Five distributions, unchanged inputs/weights/prior outputs. Numerical contract, NOT bitwise. Logical NCHW public ABI. Synchronous full call from ready MLX tensors to ready MLX output; no host/device conversion in timing. No output caching or input-dependent specialization. Frozen parameters may be prepacked with setup reported.",
        rules="20 tasks, two discovery and two unopened final shape/layout cases each. Freeze fastest correct of eager/compiled/shapeless per task by discovery geometric mean before authoring. One shared 60-minute maximum (configurable), including author + evaluation; excludes reference screening and final tests. No manual repairs. Every invalid/duplicate/timeout consumes a call. Select one complete module by highest eligible discovery geometric mean, with baseline fallback if none exceeds 1.0. Eligible requires all 40 cases numerically correct and warm MLX peak <= reference*1.10 +16MiB. Full confirmation requires discovery and both fresh final processes: overall geomean >=1.02 and every case lower paired 95% interval >1. Per-task confirmation requires the same for that task's two cases. No final retry, no automatic deployment. No claim of optimizer transfer: task families are public and visible, final input shapes/data are held out.",
        source_hashes={
            str(p.relative_to(out / "frozen")): digest(p)
            for p in sorted((out / "frozen").rglob("*"))
            if p.is_file()
        },
    )
    write(out / "plan.json", plan)
    write(out / "plan_sha256.json", {"sha256": digest(out / "plan.json")})
    return plan


def verify(out):
    assert digest(out / "plan.json") == read(out / "plan_sha256.json")["sha256"]
    plan = read(out / "plan.json")
    for path, value in plan["source_hashes"].items():
        assert digest(out / "frozen" / path) == value, path
    return plan


def process(out, name, argv, limit):
    plan = verify(out)
    log = out / "workers" / (name + ".log")
    log.parent.mkdir(exist_ok=True)
    env = dict(
        os.environ, PYTHONPATH=str(out / "frozen/src"), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1"
    )
    start = elapsed()
    status = "completed"
    with log.open("w") as stream:
        proc = subprocess.Popen(
            argv, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        try:
            while proc.poll() is None:
                used = elapsed() - start
                rss = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(proc.pid)], capture_output=True, text=True
                )
                if used >= limit or (
                    rss.stdout.strip()
                    and int(rss.stdout.strip()) * 1024 > plan["worker_rss_limit_bytes"]
                ):
                    status = "timeout" if used >= limit else "rss_limit"
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    break
                try:
                    proc.wait(timeout=min(20, limit - used))
                except subprocess.TimeoutExpired:
                    tail = log.read_text().splitlines()[-1:]
                    stamp(
                        f"049 worker={name} elapsed={elapsed() - start:.0f}s log={log} last={tail}"
                    )
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    used = elapsed() - start
    if used > limit:
        status = "timeout"
    receipt = dict(status=status, returncode=proc.returncode, seconds=used, log=str(log))
    write(out / "workers" / (name + ".receipt.json"), receipt)
    stamp(f"049 worker={name} COMPLETE status={status} seconds={used:.1f}")
    if proc.returncode or status != "completed":
        raise RuntimeError(json.dumps(receipt) + "\n" + log.read_text()[-5000:])
    return receipt


def oracle(out, split):
    plan = verify(out)
    dest = out / ("oracle_" + split)
    casepath = out / (split + "_cases.json")
    write(casepath, plan[split])
    process(
        out,
        "oracle_" + split,
        [
            sys.executable,
            "-u",
            str(out / "frozen/harness/oracle.py"),
            "--cases",
            str(casepath),
            "--output",
            str(dest),
        ],
        plan["oracle_timeout_seconds"],
    )
    return dest


def worker(out, name, job, timeout=None):
    plan = verify(out)
    jobpath, result = out / "workers" / (name + ".job.json"), out / "workers" / (name + ".json")
    assert not result.exists()
    write(jobpath, {"repeats": plan["repeats"], **job})
    process(
        out,
        name,
        [
            sys.executable,
            "-u",
            str(out / "frozen/harness/worker.py"),
            "--job",
            str(jobpath),
            "--output",
            str(result),
        ],
        min(timeout or plan["worker_timeout_seconds"], plan["worker_timeout_seconds"]),
    )
    data = read(result)
    assert data["status"] == "complete"
    return data


def screen(out):
    plan = verify(out)
    folder = oracle(out, "discovery")
    base = worker(
        out, "baseline", dict(phase="baseline", cases=plan["discovery"], oracle=str(folder))
    )
    refs, failures = {}, []
    for task in TASKS:
        rows = [r for r in base["records"] if r["case"]["task"] == task]
        eligible = [m for m in METHODS if all(r["checks"][m]["pass"] for r in rows)]
        if not eligible or not all(r["checks"]["eager"]["pass"] for r in rows):
            failures.append(task)
            continue
        refs[task] = min(
            eligible,
            key=lambda m: statistics.geometric_mean(
                statistics.median(r["samples"][m]) for r in rows
            ),
        )
    write(
        out / "port_validation.json",
        dict(passed=not failures, failed_tasks=failures, denominator=20),
    )
    if failures:
        raise RuntimeError(
            f"Upstream-to-MLX port validation failed; no author launched: {failures}"
        )
    write(out / "references.json", refs)
    write(out / "references_sha256.json", {"sha256": digest(out / "references.json")})
    return base, refs


def compact(report):
    return [
        dict(
            case=r["case"],
            correctness={n: c["pass"] for n, c in r["checks"].items()},
            errors={n: c for n, c in r["checks"].items() if not c["pass"]},
            median_seconds={n: statistics.median(v) for n, v in r["samples"].items() if v},
            first_call=r["first_call"],
            constructor_seconds=r["setup_seconds"],
            max_peak_bytes={n: max(v) for n, v in r["peaks"].items() if v},
        )
        for r in report["records"]
    ]


def request(out, plan, base, refs, history, incumbent, left):
    root = out / "frozen/harness"
    return dict(
        author_model=plan["author_model"],
        task="Optimize this 20-task KernelBench-derived float32 MLX suite on the supplied M2. Return one complete Python module defining Runner(task, config, parameters, reference).run(*inputs) -> one MLX array. Constructor receives one task, static config, a dict of fixed MLX parameter arrays in upstream PyTorch storage order, and the frozen strongest applicable reference callable. You may prepack weights, fuse computations, change graph execution, or write mx.fast.metal_kernel implementations. Handle every task, using the reference callable for unchanged tasks. The same implementation will receive new shapes, parameter values and input distributions in fresh final processes. Do not merely select allocator settings.",
        contract=plan["contract"],
        rules=plan["rules"],
        restrictions="No tools, filesystem, network, time/clock access, evaluator changes, global monkeypatching, concurrency, output lookup/caching, memorized input data, quantization or reduced precision. Reference callable fallback allowed. Cache only bounded compiled functions and prepacked fixed parameters; no retained outputs or input-derived data across calls. Inputs/parameters and earlier outputs must remain unchanged. No source repair outside this loop. Only import mlx, math, functools, collections, itertools, typing. Python/AST guard is not a proof of correctness.",
        hardware=base["hardware"],
        runtime={k: base[k] for k in ["mlx", "numpy"]},
        search_remaining_seconds=left,
        response_deadline_seconds=min(1800, left - 120),
        evaluation_expected_seconds="Approximately 15-90 seconds; reserve 120 seconds. You share one hour across all tasks. Prioritize changes supported by complete-call profiles. Return a full module with safe reference fallbacks.",
        baseline_references=refs,
        discovery=compact(base),
        mlx_reference_source=(root / "ports.py").read_text(),
        upstream_sources={
            r["task"]: (root / "upstream" / r["source"]).read_text()
            for r in plan["upstream"]["tasks"]
        },
        current_source=Path(incumbent).read_text(),
        history=[
            {k: v for k, v in row.items() if k not in {"feedback", "source"}} for row in history
        ],
        latest_feedback=history[-1].get("feedback") if history else None,
    )


def run(out, smoke=False):
    plan = verify(out)
    started = elapsed()
    stamp(
        f"049 START phase={'smoke' if smoke else 'run'} result={out} log={out.parent / (out.name + '.log')}"
    )
    base, refs = screen(out)
    seed = out / "frozen/harness/seed.py"
    common = dict(
        phase="candidate",
        references=refs,
        cases=plan["discovery"],
        oracle=str(out / "oracle_discovery"),
    )
    if smoke:
        positive = worker(out, "seed", dict(**common, candidate=str(seed), repeats=3))
        assert all(r["checks"]["candidate"]["pass"] for r in positive["records"])
        bad = out / "negative.py"
        bad.write_text(
            seed.read_text().replace(
                "return self.reference(*inputs)", "return self.reference(*inputs) + 1.0"
            )
        )
        negative = worker(
            out,
            "negative",
            dict(**{**common, "cases": plan["discovery"][:2]}, candidate=str(bad), repeats=1),
        )
        assert all(not r["checks"]["candidate"]["pass"] for r in negative["records"])
        write(
            out / "summary.json",
            dict(
                status="smoke_complete",
                port_cases=40,
                seed_cases=40,
                negative_rejected=True,
                elapsed_seconds=elapsed() - started,
            ),
        )
        stamp(f"049 SMOKE COMPLETE result={out} log={out.parent / (out.name + '.log')}")
        return
    from author import invoke
    from lossless.codex_provider import require_chatgpt

    require_chatgpt()
    search_start = elapsed()
    deadline = search_start + plan["search_seconds"]
    stamp(f"049 AUTHOR LOOP START max_seconds={plan['search_seconds']} clock=sleep-inclusive")
    history, seen = [], set()
    incumbent, best, selected = seed, 1.0, None
    for number in range(1, plan["max_calls"] + 1):
        left = deadline - elapsed()
        if left < plan["measurement_reserve_seconds"] + plan["minimum_call_seconds"]:
            break
        name = f"round{number:02d}"
        stamp(f"049 AUTHOR {name} remaining={left:.0f}s")
        receipt, response = invoke(
            request(out, plan, base, refs, history, incumbent, left),
            out / "authors" / name,
            min(plan["response_cap_seconds"], left - plan["measurement_reserve_seconds"]),
        )
        row = dict(round=name, receipt=receipt)
        try:
            if not receipt["eligible"]:
                raise ValueError("provider response ineligible")
            proposal = response["proposals"][0]
            sha = source_check(proposal["source"])
            if sha in seen:
                raise ValueError("duplicate source")
            seen.add(sha)
            path = out / "candidates" / (name + ".py")
            path.parent.mkdir(exist_ok=True)
            path.write_text(proposal["source"])
            row.update(source=str(path), source_sha256=sha, hypothesis=proposal["hypothesis"])
            report = worker(
                out,
                name,
                dict(**common, candidate=str(path)),
                timeout=max(0.1, deadline - elapsed()),
            )
            if elapsed() > deadline:
                raise TimeoutError("completion after shared search deadline")
            row["assessment"] = assess(report["records"])
            row["feedback"] = compact(report)
            if row["assessment"]["eligible"] and row["assessment"]["geomean"] > best:
                best, incumbent = row["assessment"]["geomean"], path
                selected = {k: row[k] for k in ["round", "source", "source_sha256", "assessment"]}
        except Exception:
            row["failure"] = traceback.format_exc(limit=6)
        history.append(row)
        write(out / "history.json", history)
        stamp(
            f"049 {name} COMPLETE eligible={row.get('assessment', {}).get('eligible', False)} mean={row.get('assessment', {}).get('geomean')} best={best:.5f}"
        )
    search_used = elapsed() - search_start
    selection = selected or dict(
        round="baseline_fallback", source=str(seed), source_sha256=digest(seed), assessment=None
    )
    write(out / "selection.json", selection)
    write(out / "selection_sha256.json", {"sha256": digest(out / "selection.json")})
    stamp(f"049 SEALED source={selection['round']} search_seconds={search_used:.1f}")
    oracle(out, "evaluation")
    finals = []
    for i in range(plan["final_processes"]):
        assert digest(out / "references.json") == read(out / "references_sha256.json")["sha256"]
        assert digest(selection["source"]) == selection["source_sha256"]
        report = worker(
            out,
            f"final_{i + 1}",
            dict(
                phase="candidate",
                references=refs,
                cases=plan["evaluation"],
                oracle=str(out / "oracle_evaluation"),
                candidate=selection["source"],
                repeats=plan["final_repeats"],
                seed=4910 + i,
            ),
        )
        finals.append(assess(report["records"]))
    gate = bool(
        selected and selected["assessment"]["confirmed"] and all(r["confirmed"] for r in finals)
    )
    write(
        out / "summary.json",
        dict(
            status="complete",
            selected=selection,
            search_seconds=search_used,
            total_seconds=elapsed() - started,
            final_assessments=finals,
            experimental_gate_passed=gate,
            calls=len(history),
            sleep_inclusive_clock=True,
        ),
    )
    stamp(f"049 COMPLETE gate={gate} result={out} log={out.parent / (out.name + '.log')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["freeze", "run", "smoke"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=int, default=3600)
    a = p.parse_args()
    out = a.output.resolve()
    if a.phase == "freeze":
        if not 300 <= a.seconds <= 8 * 3600:
            raise ValueError("search allowance must be 300..28800 seconds")
        freeze(out, a.seconds)
        stamp(f"049 FROZEN result={out}")
    elif a.phase == "smoke":
        freeze(out, a.seconds)
        run(out, smoke=True)
    else:
        verify(out)
        # Execute the controller itself from the immutable snapshot as well.
        frozen = out / "frozen/harness/study.py"
        if Path(__file__).resolve() != frozen:
            os.environ["PYTHONPATH"] = str(out / "frozen/src")
            os.execv(
                sys.executable, [sys.executable, "-u", str(frozen), "run", "--output", str(out)]
            )
        run(out)
