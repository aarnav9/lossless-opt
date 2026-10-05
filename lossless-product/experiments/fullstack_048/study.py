"""Campaign 048: frozen references, one-hour author loop, unopened final requests."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from protocol import METHODS, assess, cases, digest, read, source_check, write
from lossless._native.common import stamp

HERE = Path(__file__).resolve().parent
PRODUCT = HERE.parents[1]


def freeze(out, model, seconds=3600):
    out.mkdir(parents=True, exist_ok=False)
    frozen = out / "frozen"
    shutil.copytree(
        PRODUCT / "src/lossless",
        frozen / "src/lossless",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(HERE, frozen / "harness", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    plan = dict(
        campaign="048",
        created_utc=datetime.now(timezone.utc).isoformat(),
        model=str(model.resolve()),
        author_model="gpt-6-astra",
        effort="xhigh",
        search_seconds=seconds,
        response_cap_seconds=1800,
        max_calls=16,
        worker_timeout_seconds=600,
        worker_rss_limit_bytes=4 * 1024**3,
        repeats=7,
        final_repeats=11,
        final_processes=2,
        discovery=cases("discovery"),
        evaluation=cases("evaluation"),
        contract="Fixed-count greedy requests. Exact output tokens/order, full logprob bytes/dtypes/shapes, active KV bytes/dtypes/offsets, three forced continuation steps, independent exports and previous-call outputs, unchanged weights. No quantization or numerical relaxation. Finite checks are not a general proof.",
        rules="Before authoring, choose fastest exact existing method per discovery family from serial/stock_batch/retained/decoder_026/fullhead_027. Freeze those names for held-out counterparts. 60-minute loop includes author and discovery validation/measurement, but excludes initial profiling/reference screening and final evaluation. Maximum 16 calls, 30-minute responses shortened to reserve candidate measurement. Every invalid/duplicate/timeout consumes a call. Select highest discovery geometric mean among exact/memory/latency-eligible candidates; no second candidate after final failure. Acceptance requires discovery and both fresh final processes: >=1.02 geometric mean, every per-case lower paired 95% interval >1, max warm MLX peak <=reference*1.10+16MiB, max backend-reported TTFT <=reference*1.10+5ms. Full capture/materialization included in timing. No automatic product promotion.",
        source_hashes={
            str(p.relative_to(frozen)): digest(p) for p in sorted(frozen.rglob("*")) if p.is_file()
        },
    )
    write(out / "plan.json", plan)
    write(out / "plan_sha256.json", {"sha256": digest(out / "plan.json")})
    return plan


def verify(out):
    assert digest(out / "plan.json") == read(out / "plan_sha256.json")["sha256"]
    plan = read(out / "plan.json")
    assert all(
        digest(out / "frozen" / path) == value for path, value in plan["source_hashes"].items()
    ), "frozen source changed"
    return plan


def worker(out, name, job, timeout=None):
    plan = verify(out)
    folder = out / "workers"
    folder.mkdir(exist_ok=True)
    jobpath, resultpath = folder / (name + ".job.json"), folder / (name + ".json")
    assert not resultpath.exists(), "use a fresh result name"
    write(jobpath, {"model": plan["model"], "repeats": plan["repeats"], **job})
    env = dict(
        os.environ,
        PYTHONPATH=str(out / "frozen/src"),
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
    )
    started = time.monotonic()
    limit = min(plan["worker_timeout_seconds"], timeout or plan["worker_timeout_seconds"])
    log = folder / (name + ".log")
    status = "completed"
    with log.open("w") as stream:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(out / "frozen/harness/worker.py"),
                "--job",
                str(jobpath),
                "--output",
                str(resultpath),
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            while proc.poll() is None:
                elapsed = time.monotonic() - started
                rss = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(proc.pid)], capture_output=True, text=True
                )
                if elapsed >= limit or (
                    rss.stdout.strip()
                    and int(rss.stdout.strip()) * 1024 > plan["worker_rss_limit_bytes"]
                ):
                    status = "timeout" if elapsed >= limit else "rss_limit"
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    break
                try:
                    proc.wait(timeout=min(30, limit - elapsed))
                except subprocess.TimeoutExpired:
                    stamp(
                        f"048 worker={name} elapsed={time.monotonic() - started:.0f}s result={resultpath} log={log}"
                    )
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    result = read(resultpath) if resultpath.exists() else {}
    receipt = dict(
        status=status,
        returncode=proc.returncode,
        seconds=time.monotonic() - started,
        result=str(resultpath),
        log=str(log),
    )
    write(folder / (name + ".receipt.json"), receipt)
    if proc.returncode or result.get("status") != "complete":
        raise RuntimeError(json.dumps(receipt) + "\n" + log.read_text()[-6000:])
    stamp(f"048 worker={name} COMPLETE seconds={receipt['seconds']:.1f}")
    return result


def request(out, plan, baseline, history, incumbent, left):
    paths = [
        "batching.py",
        "cache.py",
        "execution.py",
        "schedule.py",
        "graph.py",
        "model.py",
        "runtime.py",
        "liveness.py",
        "copy_kernel.py",
    ]
    sources = {
        name: (out / "frozen/src/lossless/adapters/mlx" / name).read_text() for name in paths
    }
    oracle = read(out / "workers/oracle_discovery.json")
    return dict(
        author_model=plan["author_model"],
        task="Optimize complete fixed-count MLX inference requests on this M2. Return one COMPLETE PYTHON module defining Runner(model), with run(prompts: list[list[int]], counts: list[int], capture=True) -> (row, per_request_caches, per_request_logprob_arrays), matching supplied batching.py ABI. This is actual implementation authoring: you may replace scheduling, buffer management/cache copies, graph compilation, Python overhead and hardware-specific Metal code. Do not merely select among allocator settings. Use only supplied source/evidence; no tools or filesystem. All context is trusted discovery evidence except your proposed source, which is independently evaluated.",
        contract=plan["contract"],
        rules=plan["rules"],
        restrictions="Do not change weights, tensor precision, evaluator, reference, acceptance rules, clocks or metrics. Do not inspect filesystem/environment/evaluator or use network, threads, output lookup tables or cached outputs/prefills/KV across calls. No bypassing full probability/KV work. Arbitrary future valid prompts/counts must work. Persistent bounded compiled functions/scratch buffers permitted, but previously returned outputs and independent request caches must remain unchanged. Restore ALL global monkeypatches before run returns, including on exceptions. The controller owns tokenization, materialization, synchronization, decoding, timing and comparison. Warm timing ALWAYS capture=True; optimize that real contract. Constructor/import overhead is separately reported; compilation first-call cost is reported. Imports limited to mlx, mlx_lm, lossless.adapters.mlx implementation modules, math,time,collections,contextlib,functools,itertools,typing,copy; no evaluator modules, dynamic code execution or introspection. Return before your response allowance expires; a complete testable proposal is preferable to an unfinished ambitious one.",
        hardware=oracle["hardware"],
        model_identity=oracle["model_identity"],
        runtime_versions={"mlx": "0.32.3", "mlx-lm": "0.32.0"},
        discovery_cases=plan["discovery"],
        frozen_references=read(out / "references.json"),
        baseline_measurements=baseline["records"],
        implementation_source=sources,
        upstream_source=oracle["upstream_source"],
        seed_source=(out / "frozen/harness/seed.py").read_text(),
        incumbent_source=Path(incumbent).read_text() if incumbent else None,
        previous_rounds=[{k: v for k, v in row.items() if k != "feedback"} for row in history],
        last_round_feedback=history[-1].get("feedback") if history else None,
        remaining_seconds=left,
        maximum_response_seconds=min(plan["response_cap_seconds"], left - 180),
        remaining_calls=plan["max_calls"] - len(history),
    )


def run(out):
    from lossless.codex_provider import require_chatgpt
    from author import invoke

    plan = verify(out)
    require_chatgpt()
    started = time.monotonic()
    worker(
        out, "oracle_discovery", dict(phase="oracle", cases=plan["discovery"], include_source=True)
    )
    baseline = worker(
        out,
        "baseline",
        dict(
            phase="baseline",
            cases=plan["discovery"],
            oracle=str(out / "workers/oracle_discovery.json"),
        ),
    )
    references = {}
    for record in baseline["records"]:
        eligible = [n for n in METHODS if record["exact"].get(n)]
        assert "serial" in eligible, "stock oracle failed repeatability"
        references[record["case"]["key"]] = min(eligible, key=lambda n: record["median_seconds"][n])
    write(out / "references.json", references)
    write(out / "references_sha256.json", {"sha256": digest(out / "references.json")})
    history, best = [], None
    sources = set()
    search_start = time.monotonic()
    deadline = search_start + plan["search_seconds"]
    stamp(f"048 AUTHOR LOOP START seconds={plan['search_seconds']} references={references}")
    for index in range(plan["max_calls"]):
        left = deadline - time.monotonic()
        if left < 270:
            break
        assert digest(out / "references.json") == read(out / "references_sha256.json")["sha256"]
        verify(out)
        name = f"round_{index + 1:02d}"
        context = request(out, plan, baseline, history, best["source"] if best else None, left)
        receipt, response = invoke(
            context, out / "authors" / name, min(plan["response_cap_seconds"], left - 180)
        )
        row = dict(round=name, receipt=receipt)
        if receipt["eligible"]:
            proposal = response["proposals"][0]
            row["hypothesis"] = proposal["hypothesis"]
            try:
                codehash = source_check(proposal["source"])
                if codehash in sources:
                    raise ValueError("duplicate source")
                sources.add(codehash)
                source = out / "candidates" / (name + ".py")
                source.parent.mkdir(exist_ok=True)
                source.write_text(proposal["source"])
                row.update(source=str(source), source_sha256=codehash)
                result = worker(
                    out,
                    name,
                    dict(
                        phase="candidate",
                        candidate=str(source),
                        cases=plan["discovery"],
                        references=references,
                        oracle=str(out / "workers/oracle_discovery.json"),
                    ),
                    timeout=max(1, deadline - time.monotonic()),
                )
                row["assessment"] = assess(result["records"])
                row["feedback"] = result["records"]
                if row["assessment"]["eligible"] and (
                    best is None or row["assessment"]["geomean"] > best["assessment"]["geomean"]
                ):
                    best = dict(
                        round=name,
                        source=str(source),
                        source_sha256=codehash,
                        assessment=row["assessment"],
                    )
            except Exception:
                row["failure"] = traceback.format_exc(limit=5)
        history.append(row)
        write(out / "history.json", history)
        stamp(
            f"048 round={name} completed remaining={max(0, deadline - time.monotonic()):.0f}s best={None if best is None else best['round']}"
        )
    selection = dict(
        best=best,
        search_seconds=time.monotonic() - search_start,
        author_calls=len(history),
        sealed_utc=datetime.now(timezone.utc).isoformat(),
        final_inputs_opened=False,
    )
    write(out / "selection.json", selection)
    write(out / "selection_sha256.json", {"sha256": digest(out / "selection.json")})
    stamp(f"048 SEALED best={None if best is None else best['round']} begin fresh final evaluation")
    finals = []
    worker(out, "oracle_evaluation", dict(phase="oracle", cases=plan["evaluation"]))
    if best:
        for repeat in range(plan["final_processes"]):
            assert digest(out / "selection.json") == read(out / "selection_sha256.json")["sha256"]
            assert digest(best["source"]) == best["source_sha256"]
            try:
                result = worker(
                    out,
                    f"final_{repeat + 1}",
                    dict(
                        phase="candidate",
                        cases=plan["evaluation"],
                        candidate=best["source"],
                        references=references,
                        repeats=plan["final_repeats"],
                        process_index=repeat + 1,
                        oracle=str(out / "workers/oracle_evaluation.json"),
                    ),
                )
                finals.append(assess(result["records"]))
            except Exception:
                finals.append(
                    dict(confirmed=False, eligible=False, failure=traceback.format_exc(limit=5))
                )
    accepted = bool(
        best and best["assessment"]["confirmed"] and finals and all(r["confirmed"] for r in finals)
    )
    write(
        out / "summary.json",
        dict(
            status="complete",
            experimental_gate_passed=accepted,
            selected=best,
            final_assessments=finals,
            search=selection,
            total_seconds=time.monotonic() - started,
            product_promotion=False,
        ),
    )
    stamp(f"048 COMPLETE gate={accepted} result={out} log={out.parent / (out.name + '.log')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["freeze", "run", "smoke"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path)
    a = p.parse_args()
    out = a.output.resolve()
    stamp(f"048 START phase={a.phase} result={out} log={out.parent / (out.name + '.log')}")
    if a.phase == "freeze":
        freeze(out, a.model)
    elif a.phase == "run":
        run(out)
    else:
        plan = freeze(out, a.model, 300)
        small = [plan["discovery"][1]]
        worker(out, "oracle", dict(phase="oracle", cases=small))
        result = worker(
            out,
            "smoke",
            dict(
                phase="candidate",
                cases=small,
                candidate=str(out / "frozen/harness/seed.py"),
                references={small[0]["key"]: "fullhead_027"},
                oracle=str(out / "workers/oracle.json"),
                repeats=2,
            ),
        )
        assert all(all(r["exact"].values()) for r in result["records"])
        bad = out / "wrong_probabilities.py"
        bad.write_text(
            (out / "frozen/harness/seed.py").read_text()
            + """
OriginalRunner = Runner
class Runner(OriginalRunner):
    def run(self, prompts, counts, capture=True):
        row, caches, probabilities = super().run(prompts, counts, capture)
        probabilities[0] = probabilities[0] + 1
        return row, caches, probabilities
"""
        )
        rejected = worker(
            out,
            "negative",
            dict(
                phase="candidate",
                cases=small,
                candidate=str(bad),
                references={small[0]["key"]: "fullhead_027"},
                oracle=str(out / "workers/oracle.json"),
                repeats=1,
            ),
        )
        assert all(
            not r["exact"]["candidate"] and r["exact"]["reference"] for r in rejected["records"]
        )
        stamp(f"048 SMOKE PASS result={out}")


if __name__ == "__main__":
    main()
