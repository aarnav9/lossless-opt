"""Resumable sequential study. Each experiment consumes only earlier discovery results."""

import argparse
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import inspect

import numpy as np
import psutil

from common import DEFAULT_RUN, HERE, ROOT, POLICY, read, score, sha, stamp, tree, verdict, write

INTAKE = HERE.parent / "application_intake"
sys.path.insert(0, str(INTAKE))
from broader_study import SCOPE, new_records
from lossless.applications import patches


def freeze_cases(folder, final=False, family=0):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "lossless_fixtures").mkdir(exist_ok=True)
    specs = [
        ("small", 349 if final else 257, 1, {}, "steps"),
        (
            "weighted",
            5003 if final else 4093,
            6,
            {"weight_mode": "changing", "return_coef": True},
            "peaks",
        ),
        ("long", 24571 if final else 16381, 4, {"return_coef": True}, "oscillatory"),
        ("batch", 2053 if final else 1027, 48, {"return_coef": True}, "peaks"),
        (
            "branches",
            1493 if final else 1023,
            4,
            {
                "mask_initial_peaks": True,
                "use_original": True,
                "varying_order": True,
                "ordering": "shuffled",
                "irregular": True,
            },
            "steps",
        ),
    ]
    if final:
        specs += [
            ("float32", 809, 3, {"dtype": "float32", "return_coef": True}, "oscillatory"),
            ("flat", 521, 2, {"max_iter": 11}, "flat"),
            ("sparse", 3079, 5, {"weight_mode": "sparse", "poly_order": 5}, "peaks"),
        ]
    cases = []
    for i, (name, size, batch, options, pattern) in enumerate(specs):
        if final:
            size += family * 6
        calls = []
        for j in range(3):
            records = new_records(
                folder,
                f"{name}_{j}",
                size,
                batch,
                (87234311 + family * 1000000 if final else 63827103) + i * 10000 + j * 100,
                options,
                pattern=pattern,
            )
            calls.append({"args": [records]})
        cases.append(dict(id=name, calls=calls))
    return cases


def prepare(out):
    if (out / "protocol.json").exists():
        return
    from study import REVISION  # application_intake is first in sys.path

    upstream = ROOT / ".lossless/external/pybaselines-v1.2.1"
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError("unexpected upstream revision")
    out.mkdir(parents=True, exist_ok=True)
    baseline = out / "baseline"
    baseline.mkdir()
    shutil.copytree(
        upstream / "pybaselines",
        baseline / "pybaselines",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copyfile(INTAKE / "spectrum_workload.py", baseline / "spectrum_workload.py")
    for name in ("continuation-candidate.json", "cpu-control-candidate.json"):
        patches.apply(baseline, baseline, read(INTAKE / name), SCOPE)
    development = freeze_cases(out / "discovery")
    write(out / "discovery/cases.json", development)
    # Freeze final specification/generator identity now. Values are materialized only after selection.
    write(
        out / "protocol.json",
        dict(
            schema_version=1,
            created=time.time(),
            policy=POLICY,
            upstream_revision=revision,
            baseline=tree(baseline),
            classification="Exact bytes and ownership; separate fixed numerical envelope; no quality labels.",
            experiment_count=100,
            families=5,
            sequential=True,
            discovery_seed=63827103,
            final_seed=87234311,
            timing="Three complete requests with new signals; retained output lifecycle; no hashing inside or between timed calls. Fixture loading and interpreter/import setup excluded; setup recorded separately.",
            memory="Separate worker with Python traced peak and sampled sum of parent plus descendant RSS. RSS sum is conservative and double-counts shared pages.",
            selection="Discovery-only adaptive proposals. Exactly one family winner frozen before fresh final evaluation. Failed final selection cannot be replaced using exposed final data.",
            final_generator_sha256=sha(Path(__file__)),
            operating_conditions="One BLAS thread; no simultaneous experiment workers. No thermal/PMU claim.",
        ),
    )
    write(out / "ledger.json", [])


def execute(payload, folder, mode="timing"):
    folder.mkdir(parents=True, exist_ok=False)
    payload = {**payload, "mode": mode, "result": str(folder / "result.json")}
    write(folder / "payload.json", payload)
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    begin = time.perf_counter()
    memory_pids = folder / "memory-pids"
    if mode == "memory":
        memory_pids.mkdir()
        env["LOSSLESS_CPU_MEMORY_DIR"] = str(memory_pids)
    with (folder / "stdout.log").open("w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(payload.get("worker_source", HERE / "worker.py")),
                str(folder / "payload.json"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        peak = 0
        child_peaks = {}
        parent = psutil.Process(proc.pid)
        while proc.poll() is None:
            if mode == "memory":
                try:
                    rss = parent.memory_info().rss
                    for path in memory_pids.glob("*.json"):
                        info = read(path)
                        pid = info["pid"]
                        if pid == proc.pid:
                            continue
                        child_peaks[pid] = max(child_peaks.get(pid, 0), info["peak_rss_bytes"])
                        try:
                            child_rss = psutil.Process(pid).memory_info().rss
                            rss += child_rss
                            child_peaks[pid] = max(child_peaks[pid], child_rss)
                        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                            pass
                    peak = max(peak, rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                    pass
            if time.perf_counter() - begin > 90:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise TimeoutError(f"worker exceeded 90s: {folder}")
            time.sleep(0.02)
    if not (folder / "result.json").exists():
        raise RuntimeError(f"worker failed without result: {folder}")
    result = read(folder / "result.json")
    if result["status"] != "completed" or proc.returncode:
        raise RuntimeError(f"worker failed: {result}; {folder}")
    result["process_seconds"] = time.perf_counter() - begin
    if mode == "memory":
        for path in memory_pids.glob("*.json"):
            info = read(path)
            if info["pid"] != proc.pid:
                child_peaks[info["pid"]] = max(
                    child_peaks.get(info["pid"], 0), info["peak_rss_bytes"]
                )
        result["sampled_process_tree_rss"] = peak
        result["conservative_process_tree_peak_estimate"] = result["peak_rss_bytes"] + sum(
            child_peaks.values()
        )
        result["child_peak_rss_bytes"] = child_peaks
        write(folder / "result.json", result)
    return result


def check_pair(a, b, afile, bfile):
    if not a.get("native_oracle_passed", True):
        raise ValueError("native reference violates its independent operator contract")
    ownership = all(
        v.get("inputs_unchanged", False) and v.get("retained_unchanged", False) for v in (a, b)
    )
    structure = a["structure"] == b["structure"]
    exact = numerical = ownership and structure and b.get("native_oracle_passed", True)
    failures = []
    worst = 0.0
    with np.load(afile) as av, np.load(bfile) as bv:
        if set(av.files) != set(bv.files):
            return dict(exact=False, numerical=False, reason="output fields differ")
        for key in av.files:
            x, y = av[key], bv[key]
            equal = x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
            close = equal
            if x.shape == y.shape and x.dtype == y.dtype and x.dtype.kind in "fc":
                close = bool(
                    np.allclose(
                        x,
                        y,
                        atol=POLICY["numerical_atol"],
                        rtol=POLICY["numerical_rtol"],
                        equal_nan=False,
                    )
                )
                difference = float(np.max(np.abs(x - y))) if x.size else 0.0
                worst = max(worst, difference) if np.isfinite(difference) else float("inf")
            exact = exact and equal
            numerical = numerical and close
            if not equal and len(failures) < 12:
                failures.append(dict(field=key, numerical=close, same_shape=x.shape == y.shape))
    return dict(
        exact=exact,
        numerical=numerical,
        ownership=ownership,
        structure=structure,
        resource_eligible=b.get("resource_eligible", True),
        max_absolute_error=worst if np.isfinite(worst) else None,
        differences=failures,
    )


def measure(out, reference, candidate, cases, fixtures, pairs=3, kind="application", final=False):
    out.mkdir(parents=True, exist_ok=True)
    payloads = {
        name: dict(
            project=str(v["project"]),
            candidate=v.get("config", {}),
            fixtures=str(fixtures),
            kind=kind,
        )
        for name, v in [("reference", reference), ("candidate", candidate)]
    }
    frozen_harness = next(
        (
            p / "harness/worker.py"
            for p in [out, *out.parents]
            if (p / "harness/worker.py").is_file()
        ),
        None,
    )
    if frozen_harness:
        for payload in payloads.values():
            payload["worker_source"] = str(frozen_harness)
    checks = []
    for case in cases:
        stamp(f"CHECK {out.name} case={case['id']}")
        paths = {v: out / "correctness" / case["id"] / v for v in payloads}
        results = {
            v: execute({**p, "case": case}, paths[v], "correctness") for v, p in payloads.items()
        }
        checks.append(
            dict(
                id=case["id"],
                **check_pair(
                    results["reference"],
                    results["candidate"],
                    paths["reference"] / "result.npz",
                    paths["candidate"] / "result.npz",
                ),
            )
        )
    correctness = dict(
        exact=all(c["exact"] for c in checks),
        numerical=all(c["numerical"] for c in checks),
        resource_eligible=all(c.get("resource_eligible", True) for c in checks),
        cases=checks,
    )
    write(out / "correctness.json", correctness)
    rows = [dict(id=c["id"], pairs=[]) for c in cases]
    orders = []
    for i in range(len(cases)):
        order = [j % 2 for j in range(pairs)]
        random.Random(2241 + i).shuffle(order)
        orders.append(order)
    # Even incorrect variants get timings for diagnostic classification, never promotion.
    for repeat in range(pairs):
        for i, case in enumerate(cases):
            stamp(f"TIME {out.name} pair={repeat + 1}/{pairs} case={case['id']}")
            names = ["candidate", "reference"] if orders[i][repeat] else ["reference", "candidate"]
            pair = {"order": names}
            for name in names:
                pair[name] = execute(
                    {**payloads[name], "case": case},
                    out / "timing" / f"{repeat:02d}" / case["id"] / name,
                )
            rows[i]["pairs"].append(pair)
            write(out / "raw.json", rows)
    timing = score(rows)
    record = dict(
        correctness=correctness, timing=timing, decision=verdict(correctness, timing, final)
    )
    write(out / "report.json", record)
    return record


def run_one(args):
    out = args.output.resolve()
    prepare(out)
    ledger = read(out / "ledger.json")
    if args.family == 5:
        from authoring import run

        return run(args, ledger, measure, execute)
    if not (out / "controls/passed.json").exists():
        raise ValueError("run the identical-source controls first")
    if args.family == 1:
        from compiled import proposal, build
    elif args.family == 2:
        from parallel import proposal, build
    elif args.family == 3:
        from native_trials import proposal, build
    elif args.family == 4:
        from prepared import proposal, build
    else:
        raise ValueError("authoring family runner not implemented yet")
    history = [r for r in ledger if r["family"] == args.family]
    number = len(history) + 1
    if number > 20:
        raise ValueError("all 20 experiments in this family already completed")
    folder = out / f"f{args.family}_{number:02d}"
    folder.mkdir(exist_ok=False)
    choice = proposal(number, history)
    write(folder / "plan.json", choice)
    stamp(
        f"EXPERIMENT family={args.family} number={number}/20 {choice['hypothesis']}; result={folder}/report.json"
    )
    started = time.perf_counter()
    baseline = out / "baseline"
    kind = "application"
    case_list = read(out / "discovery/cases.json")
    if args.family == 3:
        from native_trials import freeze_baseline, cases

        if number > 1 and not (out / "controls/native-passed.json").exists():
            raise ValueError("native identical-source control required before continuing")

        baseline = freeze_baseline(out)
        case_list = cases()
        kind = "native"
    candidate = build(baseline, folder, choice)
    write(folder / "source.json", tree(candidate["project"]))
    shutil.copytree(HERE, folder / "harness", ignore=shutil.ignore_patterns("__pycache__"))
    report = measure(
        folder,
        {"project": baseline},
        candidate,
        case_list,
        out / "discovery",
        pairs=args.pairs,
        kind=kind,
    )
    if read(folder / "source.json") != tree(candidate["project"]):
        raise RuntimeError("candidate source or binary changed during evaluation")
    row = dict(
        family=args.family,
        number=number,
        folder=str(folder),
        proposal=choice,
        candidate={"project": str(candidate["project"]), "config": candidate.get("config", {})},
        decision=report["decision"],
        speedup=report["timing"]["speedup"],
        case_speedups={c["id"]: c["speedup"] for c in report["timing"]["cases"]},
        elapsed_seconds=time.perf_counter() - started,
    )
    ledger.append(row)
    write(out / "ledger.json", ledger)
    stamp(
        f"COMPLETE {len(ledger)}/100 family={args.family} experiment={number} speedup={row['speedup']:.4f} category={row['decision']['category']} eligible={row['decision']['eligible']} result={folder}/report.json log={out}/campaign.log"
    )


def qualify(args):
    out = args.output.resolve()
    history = [r for r in read(out / "ledger.json") if r["family"] == args.family]
    if len(history) != 20:
        raise ValueError("finish all 20 family experiments before final selection")
    folder = out / f"final_f{args.family}"
    if not args.resume_memory:
        folder.mkdir(exist_ok=False)
        shutil.copytree(HERE, folder / "harness", ignore=shutil.ignore_patterns("__pycache__"))
    if args.family == 5:
        exact = []
        for trial in history:
            for arm, row in trial["arms"].items():
                if row.get("decision", {}).get("category") == "exact" and not row.get(
                    "unchanged_source"
                ):
                    exact.append(
                        dict(
                            number=f"{trial['number']}:{arm}",
                            decision=row["decision"],
                            speedup=row["root_relative_discovery_estimate"],
                            target=trial["target"],
                            source_sha256=row["source_sha256"],
                            candidate=dict(
                                project=str(out / f"f5_{trial['number']:02d}" / arm / "project"),
                                config={},
                            ),
                        )
                    )
    else:
        exact = [r for r in history if r["decision"]["category"] == "exact"]
    if args.family == 1:
        exact = [r for r in exact if r["proposal"]["options"].get("guard_source")]
    # Select the strongest exact discovery candidate even if exploratory gates failed,
    # so final evidence can establish a no-win. Never select again after opening final.
    passing = [r for r in exact if r["decision"]["eligible"]]
    best = max(passing or exact, key=lambda r: r["speedup"]) if exact else None
    frozen = dict(
        selected=best,
        selection="highest aggregate exact discovery score among eligible candidates; if none, strongest exact diagnostic only; unchanged final gates",
        case_generator=inspect.getsource(freeze_cases),
        policy=POLICY,
        rss_limit_bytes=512 * 1024**2 if args.family == 2 else None,
        memory_ratio_limit=6.0 if args.family == 2 else 1.2,
        max_setup_increase_seconds=2.0 if args.family == 2 else 0.1,
        lifecycle="persistent processor; startup reported and gated separately; setup-inclusive speedups also reported",
        memory_rationale="A process pool explicitly trades interpreter memory for throughput; native/application serial paths retain the existing 20% limit.",
    )
    if args.family == 5:
        frozen["selection"] = (
            "Highest eligible root-relative discovery estimate across authored sources, then fresh global validation against the ordinary baseline. Target-specific cumulative discovery ratios are only a selection heuristic. Matched context comparisons remain exploratory; this final does not establish a general causal benefit of compiler feedback."
        )
    if args.resume_memory:
        frozen = read(folder / "frozen.json")
        best = frozen["selected"]
    else:
        write(folder / "frozen.json", frozen)
    if best is None:
        write(
            folder / "report.json", dict(status="reference_retained", reason="no exact candidate")
        )
        return
    baseline = out / "baseline"
    kind = "application"
    fixtures = folder / "fixtures"
    if args.resume_memory:
        case_list = read(folder / "cases.json")
        if args.family == 3:
            baseline = out / "native_baseline"
            kind = "native"
    elif args.family == 3:
        from native_trials import cases

        case_list = cases(True)
        baseline = out / "native_baseline"
        kind = "native"
    else:
        case_list = freeze_cases(fixtures, True, args.family)
    if not args.resume_memory:
        write(folder / "cases.json", case_list)
    candidate = best["candidate"]
    if args.resume_memory:
        report = read(folder / "report.json")
        if len(report["timing"]["rounds"]) != POLICY["final_pairs"]:
            raise ValueError("timing confirmation incomplete")
    else:
        report = measure(
            folder,
            {"project": baseline},
            candidate,
            case_list,
            fixtures,
            pairs=POLICY["final_pairs"],
            kind=kind,
            final=True,
        )
    memory = []
    for case in case_list:
        pair = {"id": case["id"]}
        for name, v in [("reference", {"project": str(baseline)}), ("candidate", candidate)]:
            payload = dict(
                project=str(v["project"]),
                candidate=v.get("config", {}),
                fixtures=str(fixtures),
                case=case,
                kind=kind,
                worker_source=str(folder / "harness/worker.py"),
            )
            destination = folder / "memory" / case["id"] / name
            if (
                destination / "result.json"
            ).exists() and "conservative_process_tree_peak_estimate" in read(
                destination / "result.json"
            ):
                pair[name] = read(destination / "result.json")
            else:
                if destination.exists():
                    archived = (
                        out
                        / "harness_failures"
                        / f"final_f{args.family}_memory_{case['id']}_{name}_{time.time_ns()}"
                    )
                    archived.parent.mkdir(exist_ok=True)
                    destination.rename(archived)
                pair[name] = execute(payload, destination, "memory")
        a, b = pair["reference"], pair["candidate"]
        pair["rss_ratio"] = b["conservative_process_tree_peak_estimate"] / max(
            a["conservative_process_tree_peak_estimate"], 1
        )
        pair["traced_ratio"] = b["traced"][1] / max(a["traced"][1], 1)
        pair["eligible"] = pair["rss_ratio"] <= frozen["memory_ratio_limit"] and (
            args.family == 2 or pair["traced_ratio"] <= 1.2
        )
        if args.family == 2:
            pair["eligible"] &= (
                b["conservative_process_tree_peak_estimate"] <= frozen["rss_limit_bytes"]
            )
        memory.append(pair)
        write(folder / "memory_checkpoint.json", memory)
        stamp(
            f"MEMORY family={args.family} case={case['id']} rss_ratio={pair['rss_ratio']:.3f} eligible={pair['eligible']}"
        )
    report["memory"] = memory
    report["startup_eligible"] = all(
        c["setup_seconds"]["candidate"]
        <= c["setup_seconds"]["reference"] + frozen["max_setup_increase_seconds"]
        for c in report["timing"]["cases"]
    )
    report["accepted"] = (
        report["decision"]["eligible"]
        and report["startup_eligible"]
        and all(r["eligible"] for r in memory)
    )
    report["status"] = "qualified_exact" if report["accepted"] else "reference_retained"
    write(folder / "report.json", report)
    stamp(
        f"FINAL family={args.family} selected={best['number']} speedup={report['timing']['speedup']:.4f} status={report['status']} result={folder}/report.json"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=DEFAULT_RUN)
    p.add_argument("--family", type=int, default=1)
    p.add_argument("--pairs", type=int, default=3)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--preview-author", action="store_true")
    p.add_argument("--controls", action="store_true")
    p.add_argument("--qualify", action="store_true")
    p.add_argument("--resume-memory", action="store_true")
    args = p.parse_args()
    if args.preview_author and (
        args.family != 5 or args.count != 1 or args.qualify or args.controls
    ):
        p.error("--preview-author requires --family 5 --count 1 without evaluation flags")
    stamp(
        f"START CPU campaign result={args.output}/ledger.json log={args.output}/campaign.log expected=30-120s per non-author experiment; total campaign hours"
    )
    if args.qualify:
        qualify(args)
    elif args.controls:
        prepare(args.output.resolve())
        reference = {"project": args.output.resolve() / "baseline"}
        case_list = read(args.output / "discovery/cases.json")
        kind = "application"
        label = "aa"
        marker = "passed.json"
        if args.family == 3:
            from native_trials import freeze_baseline, cases

            reference = {"project": freeze_baseline(args.output.resolve())}
            case_list = cases()
            kind = "native"
            label = "native-aa"
            marker = "native-passed.json"
        control_folder = args.output / "controls" / label
        shutil.copytree(
            HERE, control_folder / "harness", ignore=shutil.ignore_patterns("__pycache__")
        )
        result = measure(
            control_folder,
            reference,
            reference,
            case_list,
            args.output / "discovery",
            pairs=6,
            kind=kind,
        )
        if not result["correctness"]["exact"] or result["decision"]["eligible"]:
            raise RuntimeError("identical-source control failed; stop campaign")
        write(
            args.output / "controls" / marker,
            dict(status="passed", aa_speedup=result["timing"]["speedup"]),
        )
    elif args.prepare_only:
        prepare(args.output.resolve())
    else:
        for _ in range(args.count):
            run_one(args)


if __name__ == "__main__":
    main()
