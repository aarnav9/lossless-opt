"""Bounded TMM reference probe. Instrumentation is diagnostic, never acceptance evidence.

Use the prepared project from check_cpu_project.py --kind tmm (workload.py and
three dev_*.npy fixtures). Each subprocess starts fresh; run sequentially.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from lossless._native.common import stamp, write
from lossless.applications.diagnostics import differences


def worker(root, prefix, mode):
    sys.path.insert(0, str(root))
    import tmm.tmm_core as core
    import workload

    arrays, events = {}, []

    def save(value):
        array = np.asarray(value)
        if array.dtype.kind in "buifc":
            key = f"v{len(arrays)}"
            arrays[key] = array.copy()
            return key
        return None

    if mode == "locals":

        def trace(frame, event, arg):
            if frame.f_code.co_name != "coh_tmm":
                return None
            if event == "return":
                for name in [
                    "th_list",
                    "kz_list",
                    "delta",
                    "t_list",
                    "r_list",
                    "M_list",
                    "Mtilde",
                    "r",
                    "t",
                    "R",
                    "T",
                ]:
                    events.append(dict(operation=name, output=save(frame.f_locals[name])))
            return trace

        sys.settrace(trace)
    elif mode in {"functions", "dot"}:

        def wrap(owner, name):
            original = getattr(owner, name)

            def traced(*args, **kwargs):
                event = dict(operation=name, inputs=[save(v) for v in args])
                result = original(*args, **kwargs)
                event["output"] = save(result)
                events.append(event)
                return result

            setattr(owner, name, traced)

        for name in [
            "sin",
            "cos",
            "arcsin",
            "exp",
            "list_snell",
            "interface_r",
            "interface_t",
            "make_2x2_array",
            "R_from_r",
            "T_from_t",
        ]:
            wrap(core, name)
        if mode == "dot":
            wrap(core.np, "dot")
    for i in range(3):
        arrays[f"result{i}"] = workload.run(np.load(root / f"dev_{i}_0.npy", allow_pickle=False))
    np.savez(prefix.with_suffix(".npz"), **arrays)
    write(prefix.with_suffix(".json"), events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["plain", "functions", "locals", "dot"], default="locals")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root, out = args.project.resolve(), args.output.resolve()
    if args.worker:
        return worker(root, out, args.mode)
    if not 2 <= args.repeats <= 20:
        raise ValueError("repeats must be between 2 and 20")
    out.mkdir(parents=True, exist_ok=False)
    stamp(f"START estimated=under1min result={out}/report.json log={out}.log")
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "SYSTEMROOT", "PYTHONPATH"}}
    env.update(
        PYTHONHASHSEED="0",
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    report = dict(
        mode=args.mode,
        repeats=args.repeats,
        comparisons=[],
        scope="Reference diagnostic only; tracing changes execution and can suppress differences. No speedup or numerical acceptance.",
    )
    for repeat in range(args.repeats):
        prefix = out / f"{args.mode}_{repeat}"
        stamp(f"PROBE mode={args.mode} repeat={repeat + 1}/{args.repeats}")
        subprocess.run(
            [
                sys.executable,
                "-s",
                __file__,
                "--project",
                str(root),
                "--output",
                str(prefix),
                "--mode",
                args.mode,
                "--worker",
            ],
            env=env,
            check=True,
            timeout=30,
        )
    with np.load(out / f"{args.mode}_0.npz", allow_pickle=False) as base:
        events = json.loads((out / f"{args.mode}_0.json").read_text())
        for repeat in range(1, args.repeats):
            with np.load(out / f"{args.mode}_{repeat}.npz", allow_pickle=False) as other:
                row = dict(
                    repeat=repeat,
                    returned=[
                        differences(base[f"result{i}"], other[f"result{i}"]) for i in range(3)
                    ],
                )
                for index, event in enumerate(events):
                    key = event.get("output")
                    if key and key in other and base[key].tobytes() != other[key].tobytes():
                        row["first_saved_difference"] = dict(
                            event=event,
                            preceding_saved_outputs_match=True,
                            event_index=index,
                            differences=differences(base[key], other[key]),
                        )
                        if event.get("inputs"):
                            row["first_saved_difference"]["captured_inputs_equal"] = all(
                                k and k in other and base[k].tobytes() == other[k].tobytes()
                                for k in event["inputs"]
                            )
                        break
                report["comparisons"].append(row)
    write(out / "report.json", report)
    stamp(f"DONE result={out}/report.json log={out}.log")


if __name__ == "__main__":
    main()
