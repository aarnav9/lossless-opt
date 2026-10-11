"""Twenty sequential matched author trials with or without compiler diagnostics."""

import copy
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import platform
import random
import shutil
import subprocess

from common import HERE, read, write, sha, stamp, tree

SCHEMA = dict(
    type="object",
    additionalProperties=False,
    required=["hypothesis", "source"],
    properties={k: dict(type="string") for k in ["hypothesis", "source"]},
)


def validate(response):
    if set(response) != {"hypothesis", "source"} or not all(
        isinstance(v, str) and v.strip() for v in response.values()
    ):
        raise ValueError("hypothesis and complete source required")
    if len(response["source"].encode()) > 64000:
        raise ValueError("C source exceeds 64 KiB")


def run(args, ledger, measure, execute):
    from lossless import codex_provider
    import numpy as np

    out = args.output.resolve()
    previous = [r for r in ledger if r["family"] == 5]
    number = len(previous) + 1
    if number > 20:
        raise ValueError("all twenty matched trials completed")
    preview = getattr(args, "preview_author", False)
    if not preview:
        codex_provider.require_chatgpt()
    folder = (out / "author-preview" if preview else out) / f"f5_{number:02d}"
    folder.mkdir(parents=preview)
    shutil.copytree(HERE, folder / "harness", ignore=shutil.ignore_patterns("__pycache__"))
    target = ["small", "weighted", "long", "batch", "branches"][(number - 1) % 5]
    family1 = [
        r
        for r in ledger
        if r["family"] == 1
        and r["decision"]["category"] == "exact"
        and r["proposal"]["options"].get("full_loop")
        and r["proposal"]["options"].get("guard_source")
    ]
    eligible = [r for r in family1 if r["decision"]["eligible"]]
    seed = max(eligible or family1, key=lambda r: r["speedup"])
    project = folder / "seed"
    shutil.copytree(seed["candidate"]["project"], project)
    # Within each task, only earlier matched discovery results may advance its source.
    related = [r for r in previous if r["target"] == target and r.get("winning_arm")]
    seed_root_speedup = 1.0
    if related:
        last = related[-1]
        source_path = (
            out / f"f5_{last['number']:02d}" / last["winning_arm"] / "project/cpu_region.c"
        )
        if source_path.exists():
            shutil.copyfile(source_path, project / "cpu_region.c")
            seed_root_speedup = last["arms"][last["winning_arm"]][
                "root_relative_discovery_estimate"
            ]
    c = project / "cpu_region.c"
    build_receipt = read(out / f"f1_{seed['number']:02d}" / "build.json")
    flags = seed["proposal"]["options"]["flags"]
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"

    def compile_source(root, destination):
        command = [
            "clang",
            "-O3",
            "-std=c11",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-fPIC",
            "-dynamiclib" if platform.system() == "Darwin" else "-shared",
            "-Rpass=loop-vectorize",
            "-Rpass-missed=loop-vectorize",
            "-Rpass-analysis=loop-vectorize",
            *flags,
            str(root / "cpu_region.c"),
            "-o",
            str(root / ("cpu_region" + suffix)),
            "-lm",
        ]
        r = subprocess.run(command, capture_output=True, text=True, timeout=30)
        receipt = dict(
            command=command,
            returncode=r.returncode,
            diagnostics=r.stderr,
            source_sha256=sha(root / "cpu_region.c"),
            binary_sha256=sha(root / ("cpu_region" + suffix)) if r.returncode == 0 else None,
        )
        write(destination, receipt)
        return receipt

    compile_receipt = compile_source(project, folder / "seed-build.json")
    if compile_receipt["returncode"]:
        raise ValueError("adaptive author seed failed to recompile; do not evaluate a stale binary")
    cases = [c for c in read(out / "discovery/cases.json") if c["id"] == target]
    diagnostics = execute(
        dict(
            project=str(project),
            fixtures=str(out / "discovery"),
            case=cases[0],
            worker_source=str(folder / "harness/worker.py"),
        ),
        folder / "profile",
        "attribution",
    )
    old_failures = sum(
        any(not a.get("provider", {}).get("eligible", False) for a in r.get("arms", {}).values())
        for r in previous[-2:]
    )
    timeout = 600 if old_failures else 240
    feedback = {r["number"]: r for r in previous[-2:]}
    feedback.update({r["number"]: r for r in previous if r["target"] == target})
    feedback = [feedback[k] for k in sorted(feedback)[-4:]]
    metadata = []
    for record in cases[0]["calls"][0]["args"][0]:
        signal = np.load(out / "discovery" / record["y"]["$npy"], mmap_mode="r")
        description = dict(
            shape=list(signal.shape),
            dtype=signal.dtype.str,
            options={
                k: "dynamic array" if isinstance(v, dict) else v
                for k, v in record["options"].items()
            },
        )
        if description not in metadata:
            metadata.append(description)
    context = dict(
        task="Optimize this complete CPU application helper. Return complete replacement C source with all existing exported function signatures unchanged.",
        contract="Preserve exact double-precision outputs and complete convergence histories for finite pybaselines spectra. Preserve input and output ownership, loop stopping, BLAS call arithmetic, iteration order, and tails. No approximate math, changing tolerances, result caching, files, network, timers, environment inspection, extra dependencies, or benchmark detection. Numerical differences are independently classified, never accepted as exact.",
        objective="Lower complete request time including allocations, dispatch and first use. Discovery compares the shared frozen incumbent; one final selected source is later compared with the ordinary application baseline on fresh inputs.",
        response_allowance_seconds=timeout,
        source=c.read_text(),
        python_binding=(project / "cpu_helper.py").read_text(),
        compiler_flags=build_receipt["command"][1:-4],
        hardware=dict(
            machine=platform.machine(),
            processor="Apple M2",
            blas="NumPy OpenBLAS 0.3.23.dev ILP64 one thread",
            pmu="unavailable",
        ),
        discovery_case=dict(
            id=target,
            requests=3,
            records_per_request=len(cases[0]["calls"][0]["args"][0]),
            distinct_metadata=metadata,
        ),
        prior_feedback=[
            dict(
                experiment=r["number"],
                target=r["target"],
                arms={
                    k: dict(
                        speedup=v.get("speedup"),
                        decision=v.get("decision"),
                        hypothesis=v.get("proposal", {}).get("hypothesis"),
                        status=v.get("status", "provider_failed"),
                        failure=v.get("reason") or v.get("provider", {}).get("errors"),
                    )
                    for k, v in r["arms"].items()
                },
            )
            for r in feedback
        ],
        transport_instructions="Return only the requested JSON. No tools, browsing or execution. Use only supplied discovery evidence. Do not alter the ABI or evaluator.",
    )
    if number >= 11:
        context["shared_discovery_attribution_review"] = (
            "Review of trial 6 control found its iterate function unchanged apart from formatting, "
            "and the polynomial call site does not use bind_gemv. This workload calls iterate, "
            "which uses supplied OpenBLAS function pointers; it does not call the standalone "
            "exported gemv. Do not attribute that trial's measured gain to its standalone GEMV "
            "rewrite. Timing variation or code placement are possible explanations, not established "
            "causes. Trial 8 control also edited the unused helper. Focus new hypotheses on executed "
            "work. This review is shared equally with both arms and changes no measured result or gate."
        )
    # Freeze equal budgets and contexts before either author response is requested.
    requests = {
        "control": context,
        "diagnostics": {
            **copy.deepcopy(context),
            "measured_host_hotspots": diagnostics["hotspots"],
            "compiler_vectorization_remarks": compile_receipt["diagnostics"],
        },
    }
    order = ["control", "diagnostics"]
    random.Random(4411 + number).shuffle(order)
    plan = dict(
        family=5,
        number=number,
        target=target,
        timeout_per_arm=timeout,
        order=order,
        seed_experiment=seed["number"],
        seed_source_sha256=sha(c),
        interpretation="One matched pair, one proposal per arm; 20 pairs across five workloads. Extra evidence is the intervention. Current generic kernel-author context is adapted to this application helper, not the full 22-card application checklist.",
        comparison="Both arms propose before either is measured. Same seed, compiler, discovery case, model, response cap. Identical sources share measurements. Only discovery enters subsequent trials.",
    )
    write(folder / "plan.json", plan)
    if preview:
        for arm, request in requests.items():
            write(folder / (arm + "-request.json"), request)
        write(
            folder / "payload-manifest.json",
            dict(
                destination="Existing signed-in Codex provider, OpenAI gpt-6-astra medium",
                requests={
                    arm: dict(
                        path=str(folder / (arm + "-request.json")),
                        sha256=sha(folder / (arm + "-request.json")),
                        bytes=(folder / (arm + "-request.json")).stat().st_size,
                    )
                    for arm in requests
                },
                scope="20 sequential matched trials / 40 tools-disabled author requests; subsequent requests contain updated candidate source and discovery feedback of the same kinds",
                contains=[
                    "complete experimental C helper",
                    "Python binding",
                    "workload shapes/options, not signal arrays",
                    "compiler flags and remarks",
                    "CPU/BLAS facts",
                    "measured profile function names, paths, call counts and times",
                    "earlier discovery hypotheses and measurements",
                ],
                sent=False,
            ),
        )
        stamp(f"PREVIEW ONLY: no provider request sent; payload={folder}/payload-manifest.json")
        return
    arms = {}
    for arm in order:
        stamp(f"AUTHOR experiment={number}/20 arm={arm} timeout={timeout}s target={target}")
        write(folder / (arm + "-request.json"), requests[arm])
        # One provider request at a time. A waiting controller heartbeat keeps
        # long author turns observable without running any timing concurrently.
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                codex_provider.invoke,
                requests[arm],
                folder / (arm + "-author"),
                model="gpt-6-astra",
                effort="medium",
                timeout=timeout,
                schema_override=SCHEMA,
                validator=validate,
            )
            while True:
                try:
                    receipt = pending.result(timeout=30)
                    break
                except FutureTimeout:
                    if pending.done():
                        raise
                    stamp(
                        f"AUTHOR waiting experiment={number}/20 arm={arm} result={folder}/{arm}-author"
                    )
        arms[arm] = {"provider": receipt}
        if receipt["eligible"]:
            response = read(folder / (arm + "-author/response.json"))
            response.pop("usage", None)
            arms[arm]["proposal"] = response
    seen = {}
    if all(
        not row["provider"]["eligible"]
        and not row["provider"]["timed_out"]
        and row["provider"]["elapsed_seconds"] < 2
        for row in arms.values()
    ):
        raise RuntimeError(
            "transport unavailable before authoring; preserve attempt but do not count it as an experiment"
        )
    for arm in order:
        row = arms[arm]
        if "proposal" not in row:
            continue
        root = folder / arm / "project"
        shutil.copytree(project, root)
        (root / "cpu_region.c").write_text(row["proposal"]["source"])
        compiled = compile_source(root, folder / arm / "build.json")
        row["build"] = compiled
        if compiled["returncode"]:
            row["status"] = "compile_failed"
            continue
        digest = sha(root / "cpu_region.c")
        frozen_source = tree(root)
        write(folder / arm / "source.json", frozen_source)
        if digest in seen:
            row.update(copy.deepcopy(seen[digest]))
            row["shared_identical_source_samples"] = True
        else:
            try:
                result = measure(
                    folder / arm,
                    dict(project=str(project)),
                    dict(project=str(root)),
                    cases,
                    out / "discovery",
                    pairs=6,
                )
                if tree(root) != frozen_source:
                    raise RuntimeError(
                        "authored candidate source or binary changed during evaluation"
                    )
                row.update(
                    status="evaluated",
                    speedup=result["timing"]["speedup"],
                    decision=result["decision"],
                )
                seen[digest] = {k: copy.deepcopy(row[k]) for k in ["status", "speedup", "decision"]}
            except (RuntimeError, TimeoutError, ValueError) as error:
                row.update(status="evaluation_failed", reason=str(error))
        if row.get("status") == "evaluated":
            row["root_relative_discovery_estimate"] = seed_root_speedup * row["speedup"]
            row["source_sha256"] = digest
            row["unchanged_source"] = digest == sha(c)
    passing = [
        arm
        for arm, r in arms.items()
        if r.get("decision", {}).get("eligible") and not r.get("unchanged_source")
    ]
    winner = max(passing, key=lambda arm: arms[arm]["speedup"]) if passing else None
    row = dict(
        family=5,
        number=number,
        folder=str(folder),
        target=target,
        seed_root_speedup=seed_root_speedup,
        arms=arms,
        winning_arm=winner,
        decision=dict(category="matched_author_trial", eligible=bool(winner)),
        speedup=arms[winner]["speedup"] if winner else 1.0,
    )
    write(folder / "report.json", row)
    ledger.append(row)
    write(out / "ledger.json", ledger)
    stamp(
        f"COMPLETE {len(ledger)}/100 family=5 experiment={number} winner={winner} result={folder}/report.json log={out}/campaign.log"
    )
