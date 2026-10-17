"""Fresh-process confirmation of one frozen pilot finalist; no reselection."""

import argparse
import ctypes
from pathlib import Path
import time

import pilot


def load_variant(run, metadata):
    folder = run / metadata["name"]
    binary = folder / "helper.dylib"
    if (
        pilot.sha(folder / "helper.c") != metadata["source_sha256"]
        or pilot.sha(binary) != metadata["binary_sha256"]
    ):
        raise ValueError("frozen candidate source or binary changed")
    library = ctypes.CDLL(str(binary))
    function = library.iterate
    function.argtypes = (
        [ctypes.c_void_p] * 12
        + [ctypes.c_longlong] * 4
        + [ctypes.c_int] * 4
        + [ctypes.c_double] * 2
    )
    function.restype = ctypes.c_int
    return {**metadata, "library": library, "function": function}


def passed(result):
    return all(
        result["metrics"][metric]["speedup"] >= 1.02
        and result["metrics"][metric]["no_case_regression"]
        and result["metrics"][metric]["sign_p"] <= 0.01
        for metric in ("warm_ns", "complete_ns")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--project", type=Path, default=pilot.ROOT / ".lossless/external/pybaselines-v1.2.1"
    )
    parser.add_argument("--seed", type=int, default=170712777)
    parser.add_argument("--size-offset", type=int, default=74)
    parser.add_argument("--pairs", type=int, default=24)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    pilot.stamp(f"START confirmation estimated=under1min results={out}/report.json log={out}.log")
    read = lambda path: pilot.json.loads(path.read_text())
    original = read(run / "report.json")
    protocol = read(run / "protocol.json")
    if pilot.sha(Path(pilot.__file__)) != protocol["runner_sha256"]:
        raise ValueError("pilot evaluator changed; use the frozen evaluator")
    for name, expected in protocol["resource_hashes"].items():
        if pilot.sha(pilot.RESOURCE / name) != expected:
            raise ValueError(f"retained source changed: {name}")
    baseline = load_variant(run, original["baseline"])
    winner = load_variant(run, original["selected"])
    control_meta = {
        **original["baseline"],
        "name": "identical_control",
        "binary_sha256": pilot.sha(run / "identical_control/helper.dylib"),
    }
    identical = load_variant(run, control_meta)
    source = dict(name="retained_python", mode="python")
    specs = pilot.specifications(True)
    for spec in specs:
        spec["size"] += args.size_offset
    pilot.write(
        out / "protocol.json",
        dict(
            selected=original["selected"],
            seed=args.seed,
            pairs=args.pairs,
            specs=specs,
            evaluator_sha256=pilot.sha(Path(pilot.__file__)),
            confirmation_sha256=pilot.sha(Path(__file__)),
            selection="Existing frozen finalist only; no new search or reselection.",
        ),
    )
    factories, binding = pilot.setup_application(args.project.resolve())
    control = pilot.benchmark(baseline, identical, specs, factories, binding, args.seed, args.pairs)
    checks = pilot.qualify(baseline, winner, specs, factories, binding, args.seed)
    source_checks = pilot.qualify(source, winner, specs, factories, binding, args.seed)
    pilot.write(out / "correctness.json", dict(native=checks, retained_python=source_checks))
    native = pilot.benchmark(baseline, winner, specs, factories, binding, args.seed, args.pairs)
    python = pilot.benchmark(source, winner, specs, factories, binding, args.seed, args.pairs)
    control_ok = all(
        abs(control["metrics"][m]["speedup"] - 1) <= 0.02 for m in ("warm_ns", "complete_ns")
    )
    exact = checks["exact"] and source_checks["exact"]
    report = dict(
        status="completed",
        selected=original["selected"],
        exact=exact,
        control=control,
        identical_control_passed=control_ok,
        versus_native=native,
        versus_retained_python=python,
        native_improvement_gate=exact and control_ok and passed(native),
        both_comparators_gate=exact and control_ok and passed(native) and passed(python),
        production_promoted=False,
        elapsed_seconds=time.perf_counter() - begin,
    )
    pilot.write(out / "report.json", report)
    pilot.stamp(
        f"COMPLETE exact={exact} native_gate={report['native_improvement_gate']} both_comparators_gate={report['both_comparators_gate']} warm={native['metrics']['warm_ns']['speedup']:.4f}x complete={native['metrics']['complete_ns']['speedup']:.4f}x results={out}/report.json log={out}.log"
    )


if __name__ == "__main__":
    main()
