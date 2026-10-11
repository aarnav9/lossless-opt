"""Prepared metadata candidates with explicit ownership and cache costs."""

import copy
import shutil


def proposal(number, history):
    eligible = [r for r in history if r["decision"]["category"] == "exact"]
    best = max(eligible, key=lambda r: r["speedup"]) if eligible else None
    base = dict(
        key="bytes",
        capacity=4,
        lru=False,
        last=False,
        prepare_batch=False,
        own_grid=False,
        prewarm=False,
        specialize_order=False,
        byte_limit=16 * 1024**2,
        trim_each_fit=True,
    )
    if number >= 3 and best:
        base["trim_each_fit"] = best["proposal"]["options"].get("trim_each_fit", True)
    experiments = [
        ("Replace dictionary byte keys with a bounded exact bitwise grid scan.", dict(key="scan")),
        (
            "Experiment 1 lost on small/batched calls: account for lazily created cache matrices once per complete request, preserving the retained budget at return.",
            dict(key="scan", trim_each_fit=False),
        ),
        (
            "Use SHA256 keys plus exact collision verification; all hashing stays timed.",
            dict(key="digest"),
        ),
        (
            "Copy and freeze cached grids to enforce ownership under caller mutation.",
            dict(key="scan", own_grid=True),
        ),
        (
            "Use a one-grid cache for locality and measure churn regressions.",
            dict(key="scan", capacity=1),
        ),
        ("Use a two-grid cache to reduce lookup overhead.", dict(key="scan", capacity=2)),
        (
            "Use least-recently-used rather than FIFO eviction for four grids.",
            dict(key="scan", lru=True),
        ),
        (
            "Prepare bounded owned grid/order handles explicitly; charge all metadata preparation to setup, change signal values normally.",
            dict(key="scan", explicit=True, own_grid=True),
        ),
        (
            "Precompute polynomial setup during explicit metadata preparation; compare setup-inclusive and persistent-call time.",
            dict(key="scan", explicit=True, own_grid=True, prewarm=True),
        ),
        (
            "Use exact byte-key plans with lazy polynomial setup as the preparation control.",
            dict(key="bytes", own_grid=True),
        ),
        (
            "Cache only unit-weight polynomial metadata; changing weighted operands remain dynamic.",
            dict(key="scan", specialize_order=True),
        ),
        (
            "Use read-only owned grids with a last-plan shortcut and bounded LRU.",
            dict(key="scan", own_grid=True, last=True, lru=True),
        ),
        (
            "Reduce retained plan budget to 1 MiB and measure miss/rebuild cost.",
            dict(key="scan", byte_limit=1024**2),
        ),
        (
            "Reduce retained plan budget to 256 KiB for memory-constrained calls.",
            dict(key="scan", byte_limit=256 * 1024),
        ),
        (
            "Prepare request plans using exact bitwise equality but do not retain fitters between calls.",
            dict(key="scan", prepare_batch=True, clear_each_call=True),
        ),
        (
            "Hoist return-coefficient setup into explicit plan preparation; keep changing weights dynamic.",
            dict(key="scan", explicit=True, own_grid=True, coefficient_prewarm=True),
        ),
        (
            "Use a one-entry exact front cache plus the ordinary byte-key backing cache.",
            dict(key="bytes", last=True),
        ),
        (
            "Expand the fastest exact plan to eight grid/order descriptors to cover mixed metadata without rebuilding; measure preparation and retained memory.",
            dict(adaptive=True, capacity=8),
        ),
        (
            "Apply a large-grid threshold to the best exact preparation strategy.",
            dict(adaptive=True, minimum_size=2048),
        ),
        (
            "Restrict the best exact preparation strategy to two descriptors to test whether smaller preparation and retained state repay additional fallback calls.",
            dict(adaptive=True, capacity=2),
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
        rationale="Ordinary variants charge metadata work to calls. Explicit-handle variants charge all metadata preparation to setup; setup-inclusive and repeated-call costs are both reported.",
    )


def build(baseline, folder, plan):
    project = folder / "project"
    shutil.copytree(baseline, project, ignore=shutil.ignore_patterns("__pycache__"))
    return dict(project=project, config=dict(family=4, options=plan["options"]))
