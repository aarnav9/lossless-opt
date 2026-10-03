"""Measure compilation/evidence reuse, buffer binding, and paired softmax replay."""

import argparse
from pathlib import Path
import time
import lossless
from lossless import jobs
from lossless._native import operators
from lossless._native.common import stamp, write, read, measure, interval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--search", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    stamp(f"START estimated=under2min result={out} log={out.parent / (out.name + '.log')}")
    source = (Path(a.search) / "manual_proposals/layout_dispatch.c").read_text()

    def callback(prompt):
        return {
            "schema_version": 1,
            "proposals": [
                {
                    "id": "manual_dispatch",
                    "source": source,
                    "hypothesis": "frozen manual candidate from the discovery-guided search",
                }
            ],
        }

    config = jobs.template("native.softmax", 60)
    config["budget"]["max_candidates"] = 4
    for c in config["workload"]["cases"]:
        if c["split"] == "evaluation":
            c["rows"] += 12
            c["columns"] += 38
    work = lossless.Workload.from_dict(config, base=out)
    records = []
    for repeat in range(2):
        t = time.perf_counter()
        result = lossless.optimize(work, llm=callback, output=out / f"run{repeat}")
        elapsed = time.perf_counter() - t
        state = read(result.directory / "state.json")
        raw = [
            read(result.directory / r["result"]) for r in state["jobs"].values() if "result" in r
        ]
        records.append(
            {
                "repeat": repeat,
                "wall_seconds": elapsed,
                "compile_cache_hits": sum(r.get("cache_hit", False) for r in raw),
                "validation_cache_hits": sum(r.get("validation_cache_hit", False) for r in raw),
                "status": result.summary["status"],
                "selected": result.summary["selected"],
                "payback": result.summary["payback_by_evaluation_case"],
            }
        )
    artifact = result.export(out / "artifact")
    operation = lossless.load(artifact)
    paired = []
    # Fresh shapes and generated values; candidates were frozen in the earlier search.
    retained = operators.load_library(Path(a.search) / "retained_0/build/softmax_column_4.dylib")
    manual = operators.load_library(Path(a.search) / "manual_codex_0/build/layout_dispatch.dylib")
    for layout in ["c", "f", "slice2"]:
        case = {"rows": 91, "columns": 337, "layout": layout}
        arrays = operators.input_arrays("softmax", case, 9902)
        funcs = {
            "retained": operators.bind_native(arrays, retained),
            "manual": operators.bind_native(arrays, manual),
        }
        oracle = operators.reference("softmax", arrays)
        assert all(operators.check("softmax", f(), oracle)["passed"] for f in funcs.values())
        timing = measure(funcs, 33, 31, 2000000)
        paired.append(
            {
                "layout": layout,
                "samples_us": timing["samples_us"],
                "speedup": timing["median_us"]["retained"] / timing["median_us"]["manual"],
                "ci95": interval(
                    timing["samples_us"]["retained"], timing["samples_us"]["manual"], 32
                ),
            }
        )
    case = {"rows": 64, "columns": 257, "layout": "c"}
    arrays = operators.input_arrays("softmax", case, 329)
    legacy = lambda: operators.executors("softmax", case, arrays, manual, manual)[0]["proposal"]
    lean = lambda: operators.bind_native(arrays, manual)
    setup = measure({"legacy": legacy, "deployment": lean}, 31, 21, 1000000)
    bound = operation.bind(arrays[0])
    bound_result = bound()
    assert operators.check("softmax", bound_result, operators.reference("softmax", arrays))[
        "passed"
    ]
    api = measure(
        {"call_allocating": lambda: operation(arrays[0]), "bound": bound}, 32, 21, 1000000
    )
    write(
        out / "summary.json",
        {
            "search_reuse": records,
            "paired_softmax_replay": paired,
            "binding_setup": setup,
            "deployment_api": api,
            "scope": "Fresh native trials, real export/load and independent oracle. Cache reuses compilation and discovery validation only; evaluation correctness and all timings remain fresh. Binding initialization removed; full generated-kernel scratch ABI retained.",
        },
    )
    stamp(f"COMPLETE result={out / 'summary.json'} log={out.parent / (out.name + '.log')}")


if __name__ == "__main__":
    main()
