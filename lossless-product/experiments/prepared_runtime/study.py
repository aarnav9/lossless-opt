"""Sequential paired tests of request overhead with a frozen previous runtime.

Every source candidate is unchanged. Runtime variants compare against both the
previous optimized runtime and the original source under the same new runtime.
Use one invocation per family, inspect its outcome before choosing the next.
"""

import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import statistics
import subprocess
import sys
import time

from lossless._native.common import stamp, write
from lossless.jobs import identity, read_json


def adapt(source, destination, engine):
    shutil.copytree(
        source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "prepared.json")
    )
    manifest = read_json(destination / "manifest.json")
    # Research admission only: preserve exact source/fixture identities; qualify
    # new runtime outputs independently in this study. Never issue a certificate.
    code = 'import json;from lossless.applications.workflow import bundle_environment;import sys;print(json.dumps(bundle_environment(sys.executable,["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS"])))'
    env = {**os.environ, "PYTHONPATH": str(engine)}
    manifest["environment"] = json.loads(
        subprocess.check_output([sys.executable, "-c", code], env=env, text=True, timeout=30)
    )
    manifest["qualification"] = "runtime_research_only"
    manifest["environment_values_hash"] = identity(
        {name: os.environ.get(name) for name in manifest["inherit_env"]}
    )
    write(destination / "manifest.json", manifest)
    return destination


class Controller:
    def __init__(self, engine, bundle, folder, *, variant, features=None, values=False):
        folder.mkdir(parents=True)
        self.log = (folder / "stderr.log").open("wb")
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("controller.py")),
            "--bundle",
            str(bundle),
            "--variant",
            variant,
        ]
        if features is not None:
            command += ["--new", "--features", json.dumps(features)]
        if values:
            command += ["--values"]
        env = {
            **os.environ,
            "PYTHONPATH": str(engine),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            env=env,
            cwd=folder,
            start_new_session=True,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            self.info = self.read()
        except BaseException:
            self.close()
            raise
        if self.info.get("status") != "ready":
            self.close()
            raise ValueError(str(self.info))

    def read(self):
        if not self.selector.select(timeout=35):
            raise TimeoutError("controller timed out")
        line = self.process.stdout.readline()
        if not line:
            raise ValueError("controller exited; inspect stderr.log")
        return json.loads(line)

    def run(self, case):
        self.process.stdin.write(json.dumps({"case": case}) + "\n")
        self.process.stdin.flush()
        r = self.read()
        if r.get("status") != "completed":
            raise ValueError(str(r))
        return r

    def close(self):
        if self.process.poll() is None:
            # Workers own separate process groups. Snapshot only our descendants
            # before requesting shutdown, so a hung controller cannot orphan them.
            import psutil

            try:
                descendants = psutil.Process(self.process.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                descendants = []
            try:
                self.process.stdin.write('{"stop":true}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=3)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=3)
            finally:
                for process in reversed(descendants):
                    try:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass
        for stream in (self.process.stdin, self.process.stdout):
            stream.close()
        self.selector.close()
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def geometric(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def summary(rows, rounds):
    result = {}
    for label, left, right in [
        ("runtime_gain", "old_optimized", "new_optimized"),
        ("source_edge", "new_reference", "new_optimized"),
        ("original_source_edge", "old_reference", "old_optimized"),
        ("reference_runtime_gain", "old_reference", "new_reference"),
    ]:
        cases = []
        for row in rows:
            pairs = row["pairs"]
            ratios = [
                sum(p[left][i]["request_seconds"] for i in range(len(p[left])))
                / sum(p[right][i]["request_seconds"] for i in range(len(p[right])))
                for p in pairs
            ]
            cases.append(
                dict(
                    id=row["id"],
                    speedup=geometric(ratios),
                    round_ratios=ratios,
                    request_position_speedups=[
                        geometric(
                            p[left][i]["request_seconds"] / p[right][i]["request_seconds"]
                            for p in pairs
                        )
                        for i in range(len(pairs[0][left]))
                    ],
                    median_seconds={
                        k: statistics.median(r["request_seconds"] for p in pairs for r in p[k])
                        for k in (left, right)
                    },
                    memory_ratio=statistics.median(
                        max(r["worker_peak_rss"] + r["parent_peak_rss"] for r in p[right])
                        for p in pairs
                    )
                    / statistics.median(
                        max(r["worker_peak_rss"] + r["parent_peak_rss"] for r in p[left])
                        for p in pairs
                    ),
                )
            )
        round_ratios = [geometric(c["round_ratios"][i] for c in cases) for i in range(rounds)]
        wins = sum(x > 1.02 for x in round_ratios)
        p = sum(math.comb(rounds, k) for k in range(wins, rounds + 1)) / 2**rounds
        result[label] = dict(
            cases=cases,
            speedup=geometric(c["speedup"] for c in cases),
            round_speedups=round_ratios,
            sign_test_p=p,
        )
    return result


def eligible(statistics):
    return (
        statistics["speedup"] >= 1.02
        and statistics["sign_test_p"] <= 0.05
        and all(
            c["memory_ratio"] <= 1.2
            and c["speedup"] >= 1 / 1.03
            and min(c["request_position_speedups"]) >= 1 / 1.03
            for c in statistics["cases"]
        )
    )


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    deadline = time.perf_counter() + args.budget
    report = dict(
        status="incomplete",
        family=args.family,
        split=args.split,
        features=args.features,
        baseline_features=args.baseline_features,
        values=args.values,
        rounds=args.rounds,
        requests=args.requests,
        llm_calls=0,
        rows=[],
        policy=dict(
            minimum_speedup=1.02, max_case_regression=0.03, max_memory_ratio=1.2, significance=0.05
        ),
        scope="Exact observations and transported outputs. Full parent request time, including old-API file staging for the live-value boundary. No source edits, no numerical relaxation, no promotion certificate. Worker/parent RSS peaks are summed; shared pages can be double-counted.",
    )
    stamp(
        f"START family={args.family} estimated=1-3min budget={args.budget}s result={out}/report.json log={out}.log"
    )
    try:
        sources = {
            name: Path(path).resolve() for name, path in (s.split("=", 1) for s in args.bundle)
        }
        for name, source in sources.items():
            # Validation reference is a separate fresh original process for each case.
            cases = [c for c in read_json(source / "cases.json") if c["split"] == args.split]
            old = adapt(source, out / "bundles" / name / "old", args.baseline.resolve())
            new = adapt(source, out / "bundles" / name / "new", args.engine.resolve())
            report.setdefault("manifests", {})[name] = {
                k: read_json(v / "manifest.json") for k, v in [("old", old), ("new", new)]
            }
            expected = {}
            for case in cases:
                for repeat in range(2):
                    with Controller(
                        args.baseline,
                        old,
                        out / "fresh" / name / f"{case['id']}_{repeat}",
                        variant="reference",
                        features=args.baseline_features,
                    ) as c:
                        r = c.run(case["id"])
                    exact = (r["fingerprint"], r["outputs"])
                    if case["id"] in expected and expected[case["id"]] != exact:
                        raise ValueError(
                            "fresh reference is not repeatable: " + name + "/" + case["id"]
                        )
                    expected[case["id"]] = exact
            rows = [dict(id=name + "/" + c["id"], pairs=[]) for c in cases]
            report["rows"] += rows
            for round_index in range(args.rounds):
                if time.perf_counter() >= deadline:
                    raise TimeoutError("study budget exhausted")
                stamp(
                    f"EXPERIMENT family={args.family} project={name} round={round_index + 1}/{args.rounds}"
                )
                specifications = {
                    "old_reference": (args.baseline, old, "reference", args.baseline_features),
                    "old_optimized": (args.baseline, old, "auto", args.baseline_features),
                    "new_reference": (args.engine, new, "reference", args.features),
                    "new_optimized": (args.engine, new, "auto", args.features),
                }
                names = list(specifications)
                shift = round_index % len(names)
                names = names[shift:] + names[:shift]
                with ExitStack() as stack:
                    sessions = {}
                    for key in names:
                        engine, bundle, variant, features = specifications[key]
                        sessions[key] = stack.enter_context(
                            Controller(
                                engine,
                                bundle,
                                out / "runs" / name / f"{round_index:02d}" / key,
                                variant=variant,
                                features=features,
                                values=args.values,
                            )
                        )
                        if (
                            "optimized" in key
                            and sessions[key].info["info"]["variant"] != "optimized"
                        ):
                            raise ValueError(
                                "optimized admission failed: " + str(sessions[key].info)
                            )
                    report.setdefault("startups", []).append(
                        dict(
                            project=name,
                            round=round_index,
                            seconds={k: v.info["startup_seconds"] for k, v in sessions.items()},
                        )
                    )
                    for offset in range(len(cases)):
                        index = (offset + round_index) % len(cases)
                        case = cases[index]
                        pair = {k: [] for k in names}
                        for request_index in range(args.requests):
                            for key in names if request_index % 2 == 0 else names[::-1]:
                                r = sessions[key].run(case["id"])
                                if (r["fingerprint"], r["outputs"]) != expected[case["id"]]:
                                    raise ValueError(
                                        "exactness mismatch: " + name + "/" + case["id"] + "/" + key
                                    )
                                if not r["memory_available"]:
                                    raise ValueError("memory monitoring unavailable")
                                pair[key].append(r)
                        rows[index]["pairs"].append(pair)
                write(out / "report.json", report)
        report["statistics"] = summary(report["rows"], args.rounds)
        r = report["statistics"]["runtime_gain"]
        report["exact"] = True
        report["memory_eligible"] = all(c["memory_ratio"] <= 1.2 for c in r["cases"])
        report["runtime_nonregression"] = all(c["speedup"] >= 1 / 1.03 for c in r["cases"])
        report["status"] = (
            "qualified_runtime_gain"
            if (
                r["speedup"] >= 1.02
                and r["sign_test_p"] <= 0.05
                and report["runtime_nonregression"]
                and report["memory_eligible"]
            )
            else "no_broad_runtime_gain"
        )
        report["per_project"] = {
            name: summary(
                [r for r in report["rows"] if r["id"].startswith(name + "/")], args.rounds
            )
            for name in sources
        }
        report["per_project_decisions"] = {
            name: dict(
                runtime_gain_eligible=eligible(s["runtime_gain"]),
                source_edge_eligible=eligible(s["source_edge"]),
                runtime_and_source_eligible=eligible(s["runtime_gain"])
                and eligible(s["source_edge"]),
            )
            for name, s in report["per_project"].items()
        }
    except Exception as error:
        report.update(status="failed", reason=f"{type(error).__name__}: {error}")
    finally:
        report["elapsed_seconds"] = args.budget - (deadline - time.perf_counter())
        write(out / "report.json", report)
        stamp(
            f"DONE family={args.family} status={report['status']} result={out}/report.json log={out}.log"
        )
    if report["status"] == "failed":
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--bundle", action="append", required=True, help="NAME=APPLICATION_PATH")
    p.add_argument("--family", required=True)
    p.add_argument("--features", nargs="*", default=[])
    p.add_argument("--baseline-features", nargs="*")
    p.add_argument("--values", action="store_true")
    p.add_argument("--split", choices=["discovery", "evaluation"], default="discovery")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--requests", type=int, default=3)
    p.add_argument("--budget", type=float, default=240)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.rounds < 6 or a.requests < 2 or not math.isfinite(a.budget) or a.budget <= 0:
        raise ValueError("use >=6 rounds, >=2 requests, positive budget")
    run(a)


if __name__ == "__main__":
    main()
