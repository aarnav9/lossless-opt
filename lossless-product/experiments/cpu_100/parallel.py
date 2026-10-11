"""Discovery-adaptive independent-work execution experiments."""

import copy
import shutil


def proposal(number, history):
    exact = [r for r in history if r["decision"]["category"] == "exact"]
    best = max(exact, key=lambda r: r["speedup"]) if exact else None
    base = dict(
        workers=2,
        warm=True,
        threads=False,
        chunk=1,
        reuse=True,
        minimum_batch=0,
        minimum_work=0,
        transport_chunk=1,
        submit=False,
    )
    if number >= 4 and best:
        base["workers"] = best["proposal"]["options"]["workers"]
    if number >= 10 and best:
        for key in ("minimum_batch", "minimum_work"):
            base[key] = best["proposal"]["options"].get(key, 0)
    if number >= 16 and best:
        base["prime_serial"] = best["proposal"]["options"].get("prime_serial", False)
    experiments = [
        (
            "Measure two spawned persistent workers including first-request startup.",
            dict(warm=False),
        ),
        (
            "Move worker creation into explicit factory setup and isolate its request-time effect.",
            {},
        ),
        ("Use four ready workers and measure scaling beyond two.", dict(workers=4)),
        ("Amortize task dispatch by sending four spectra per task.", dict(chunk=4)),
        ("Send sixteen spectra per task to reduce IPC overhead.", dict(chunk=16)),
        ("Balance each request into one chunk per worker.", dict(chunk="balanced")),
        (
            "Balanced partitions reached 2x on batches: combine them with serial execution below two spectra.",
            dict(chunk="balanced", minimum_batch=2),
        ),
        (
            "Raise the serial crossover to four spectra while retaining balanced worker partitions.",
            dict(chunk="balanced", minimum_batch=4),
        ),
        (
            "Dispatch by sample count, requiring at least 32768 input elements.",
            dict(chunk="balanced", minimum_work=32768),
        ),
        (
            "Test two threads with per-thread fitters, avoiding IPC but retaining the GIL.",
            dict(threads=True, workers=2, chunk=4),
        ),
        (
            "Test four threads to expose GIL/contention saturation.",
            dict(threads=True, workers=4, chunk=4),
        ),
        ("Disable per-worker fitter reuse as a mechanism ablation.", dict(chunk=4, reuse=False)),
        (
            "Use explicit submit/result aggregation instead of executor.map.",
            dict(chunk=4, submit=True),
        ),
        ("Batch four task messages per transport submission.", dict(chunk=4, transport_chunk=4)),
        (
            "The guarded serial path loses mainly on first use: explicitly initialize its numerical libraries during setup with a fixed 33-point synthetic spectrum, then clear its fitter cache.",
            dict(adaptive=True, prime_serial=True),
        ),
        (
            "Test two spectra per task and an early serial batch check, retaining explicit initialization from experiment 15.",
            dict(workers=4, chunk=2, early_batch=True),
        ),
        (
            "Use three workers to test the intermediate scheduling/memory tradeoff.",
            dict(workers=3, chunk="balanced"),
        ),
        (
            "Use the best observed executor with an 8192-element serial fallback.",
            dict(adaptive=True, minimum_work=8192),
        ),
        (
            "Combine experiment 8's four-spectrum crossover with experiment 15's explicit serial initialization and balanced partitions.",
            dict(adaptive=True, minimum_batch=4, chunk="balanced"),
        ),
        (
            "Reduce the best executor to two workers with balanced partitions and conservative crossovers; measure the memory-versus-throughput tradeoff.",
            dict(adaptive=True, workers=2, chunk="balanced", minimum_batch=4, minimum_work=8192),
        ),
    ]
    hypothesis, changes = experiments[number - 1]
    changes = dict(changes)
    if changes.pop("adaptive", False) and best:
        base.update(copy.deepcopy(best["proposal"]["options"]))
    base.update(changes)
    return dict(
        hypothesis=hypothesis,
        options=base,
        parent_experiment=best["number"] if best else None,
        informed_by=[
            dict(experiment=r["number"], speedup=r["speedup"], cases=r["case_speedups"])
            for r in history[-2:]
        ],
        rationale="Earlier complete-request timings determine the incumbent; all scheduling alternatives preserve per-spectrum arithmetic.",
    )


def build(baseline, folder, plan):
    project = folder / "project"
    shutil.copytree(baseline, project, ignore=shutil.ignore_patterns("__pycache__"))
    return dict(project=project, config=dict(family=2, options=plan["options"]))
