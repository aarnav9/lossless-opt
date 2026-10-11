"""Mutation, eviction and ownership controls for pool/prepared candidates."""

import argparse
from pathlib import Path
import sys


def main():
    from common import DEFAULT_RUN, read, write, stamp

    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=DEFAULT_RUN)
    p.add_argument("--family", type=int, choices=[2, 4], required=True)
    p.add_argument("--candidate", type=int, required=True)
    a = p.parse_args()
    row = next(
        r
        for r in read(a.output / "ledger.json")
        if r["family"] == a.family and r["number"] == a.candidate
    )
    project = Path(row["candidate"]["project"])
    harness = project.parent / "harness"
    sys.path[:0] = [str(harness), str(project)]
    import numpy as np
    import spectrum_workload
    from runtime import make_processor
    from worker import fingerprint

    config = row["candidate"]["config"]
    options = config["options"]
    candidate = make_processor(spectrum_workload, config)
    reference = spectrum_workload.make_processor()
    folder = a.output / f"conformance_f{a.family}_{a.candidate:02d}"
    folder.mkdir()
    observations = []
    saved = []
    expected = []
    rng = np.random.default_rng(836442)
    grids = [np.linspace(-2, 3, 137 + i * 2) for i in range(6)]
    try:
        for step in range(18):
            x = grids[step % 6]
            if step >= 6:
                x[2] += 0.00001
            records = []
            for i in range(4):
                y = rng.normal(size=x.size) + np.sin(x)
                opts = dict(
                    poly_order=(step % 5) + 1,
                    return_coef=True,
                    use_original=bool(step % 2),
                    mask_initial_peaks=bool(step % 3 == 0),
                    max_iter=17,
                )
                if step % 3 == 1:
                    opts["weights"] = rng.uniform(0.2, 1, size=x.size)
                if step % 4 == 0:
                    buffer = np.empty(x.size * 2)
                    buffer[::2] = y
                    y = buffer[::2]
                if step % 5 == 0:
                    y.flags.writeable = False
                records.append(dict(x=x, y=y, options=opts))
            before = fingerprint(records)
            if options.get("explicit"):
                # The new API owns metadata: later mutation of caller x must
                # not silently change the already prepared plan's grid.
                ref_records = [
                    dict(r, x=r["x"].copy(), options=dict(r["options"])) for r in records
                ]
                requests = candidate.prepare_requests([records])
                x[3] += 0.00001
                before = fingerprint(records)
                actual = candidate.run_prepared(requests[0])
                wanted = reference(ref_records)
                # Reuse the same handle with new y/weights, retaining old output.
                previous = fingerprint(actual)
                for i, entry in enumerate(requests[0]):
                    token, grid, y, opts = entry
                    new_y = rng.normal(size=y.size)
                    requests[0][i] = (token, grid, new_y, opts)
                    ref_records[i]["y"] = new_y
                second = candidate.run_prepared(requests[0])
                wanted_second = reference(ref_records)
                assert fingerprint(second) == fingerprint(wanted_second), (
                    "reused explicit handle differs"
                )
                assert fingerprint(actual) == previous, "explicit reuse overwrote an output"
            else:
                actual = candidate(records)
                wanted = reference(records)
            exact = fingerprint(actual) == fingerprint(wanted)
            assert fingerprint(records) == before, "input mutation"
            assert exact, f"changed metadata output differs at step {step}"
            assert [fingerprint(v) for v in saved] == expected, (
                "earlier returned output overwritten"
            )
            saved.append(actual)
            expected.append(fingerprint(actual))
            if a.family == 4:
                assert len(candidate.plans) <= options["capacity"], "entry budget exceeded"
                assert candidate.cache_payload_bytes() <= 16 * 1024**2, (
                    "retained payload budget exceeded"
                )
            observations.append(dict(step=step, exact=exact))
            stamp(
                f"RUNTIME CONFORMANCE family={a.family} candidate={a.candidate} step={step + 1}/18 exact={exact}"
            )
        write(
            folder / "report.json",
            dict(
                exact=True,
                observations=observations,
                contract="explicit owned grid/order handles"
                if options.get("explicit")
                else "ordinary mutable-input calls",
            ),
        )
    finally:
        close = getattr(candidate, "close", None)
        if close:
            close()


if __name__ == "__main__":
    main()
