"""Select source and lifecycle using discovery-only complete callable costs.

Freeze one choice and its incumbent before opening evaluation. A failed final
keeps fresh reference execution; evaluation never starts another search.
"""

import copy
from contextlib import ExitStack
import math
import statistics
import time

from .._native.common import write
from ..jobs import identity, read_json
from .callable import CallableApplication
from .comparison import eligible, statistics_for
from .prepared import DEFAULT_FEATURES
from .search import snapshot
from .worker import Observer, inputs


def fingerprint(outputs, limit):
    return identity(Observer(limit * max(1, len(outputs))).encode(outputs))


def paired(rows, reference, candidate, minimum=1.02):
    result = []
    for case_id, rounds in rows.items():
        result.append(
            dict(
                id=case_id,
                pairs=[dict(reference=r[reference], candidate=r[candidate]) for r in rounds],
            )
        )
    stats = statistics_for(result, minimum)
    raw = copy.deepcopy(result)
    for row in raw:
        for pair in row["pairs"]:
            for item in pair.values():
                if "request_seconds" in item:
                    item["call_seconds"] = item["request_seconds"]
                    item["sequence_seconds"] = sum(item["request_seconds"])
                    item["first_call_seconds"] = item["request_seconds"][0]
    request_stats = statistics_for(raw, 1 / 1.03)
    memory = all(
        statistics.median(p["candidate"]["peak_rss_bytes"] for p in c["pairs"])
        <= 1.2 * statistics.median(p["reference"]["peak_rss_bytes"] for p in c["pairs"])
        for c in result
    )
    return dict(
        status="matched", statistics=stats, request_statistics=request_stats, memory_eligible=memory
    )


def passes(comparison, *, final, minimum=1.02, significance=0.05):
    policy = dict(
        minimum_speedup=minimum,
        significance=significance,
        max_case_regression=0.03,
        max_call_regression=0.03,
    )
    raw_policy = dict(policy, minimum_speedup=1 / 1.03)
    requests_ok = eligible(
        dict(status="matched", statistics=comparison["request_statistics"]), raw_policy, final=False
    )
    return (
        comparison["memory_eligible"] and requests_ok and eligible(comparison, policy, final=final)
    )


def measure(
    options,
    cases,
    expected,
    *,
    rounds,
    requests,
    expected_requests,
    python,
    deadline,
    progress,
    output,
    failures=None,
):
    """Time the public adapter, including result extraction and cold cleanup."""
    import psutil

    values = {c["id"]: inputs(c["calls"], options[0]["bundle"] / "reference") for c in cases}
    rows = {c["id"]: [] for c in cases}
    manifest = read_json(options[0]["bundle"] / "manifest.json")
    limit = manifest["max_output_bytes"]
    parent = psutil.Process()
    for round_index in range(rounds):
        progress(
            f"DEPLOYMENT round={round_index + 1}/{rounds} options={len(options)} cases={len(cases)}"
        )
        round_rows = {c["id"]: {} for c in cases}
        order = options if round_index % 2 == 0 else options[::-1]
        for option in order:
            if failures is not None and option["id"] in failures:
                continue
            try:
                left = deadline - time.perf_counter()
                if left <= 0:
                    raise TimeoutError("deployment measurement budget exhausted")
                tick = time.perf_counter()
                with ExitStack() as stack:
                    app = stack.enter_context(
                        CallableApplication(
                            option["bundle"],
                            python=python,
                            mode=option["mode"],
                            variant="reference" if option["source"] == "reference" else "auto",
                            timeout=min(30, left),
                            _qualifying=True,
                        )
                    )
                    setup = time.perf_counter() - tick
                    retained = []
                    for index in range(len(cases)):
                        case = cases[(index + round_index) % len(cases)]
                        times, peak = [], 0
                        for request in range(requests):
                            left = deadline - time.perf_counter()
                            if left <= 0:
                                raise TimeoutError("deployment measurement budget exhausted")
                            app.timeout = min(30, left)
                            calls = values[case["id"]]
                            tick = time.perf_counter()
                            if len(calls) == 1:
                                result = [app(*calls[0]["args"], **calls[0]["kwargs"])]
                            else:
                                result = app.run_calls(calls)
                            times.append(time.perf_counter() - tick)
                            receipt = app.last_receipt
                            if (
                                option["source"] != "reference"
                                and receipt["variant"] != "optimized"
                            ):
                                raise ValueError("candidate failed environment admission")
                            observed = (receipt["fingerprint"], fingerprint(result, limit))
                            if observed != expected[case["id"]]:
                                raise ValueError(
                                    f"deployment exactness failed: {option['id']} {case['id']}"
                                )
                            if not receipt["memory_available"]:
                                raise ValueError(
                                    "deployment process-tree memory monitoring unavailable"
                                )
                            peak = max(
                                peak,
                                parent.memory_info().rss
                                + max(
                                    receipt["peak_tree_rss_bytes"], receipt["worker_peak_rss_bytes"]
                                ),
                            )
                            # Hold previous results through later requests to detect alias reuse.
                            retained.append((result, observed[1]))
                        round_rows[case["id"]][option["id"]] = dict(
                            sequence_seconds=sum(times),
                            call_seconds=times,
                            setup_seconds=setup,
                            first_call_seconds=times[0],
                            process_seconds=sum(times),
                            peak_rss_bytes=peak,
                            request_seconds=times,
                        )
                    if any(fingerprint(result, limit) != saved for result, saved in retained):
                        raise ValueError("later callable requests modified retained outputs")
                    close_tick = time.perf_counter()
                cleanup = time.perf_counter() - close_tick
                # Startup/teardown paid once per declared number of requests. Cold
                # request timings already include the worker's full owned lifecycle.
                lifecycle = (setup + cleanup) / expected_requests
                for row in round_rows.values():
                    item = row[option["id"]]
                    item["lifecycle_seconds"] = setup + cleanup
                    item["call_seconds"] = [t + lifecycle for t in item["request_seconds"]]
                    item["sequence_seconds"] = sum(item["call_seconds"])
                    item["first_call_seconds"] = item["call_seconds"][0]
            except (ValueError, OSError) as error:
                if failures is None or option["id"] == "reference:oneshot":
                    raise
                failures[option["id"]] = str(error)
                progress(f"DEPLOYMENT rejected={option['id']} reason={error}")
        for case_id in rows:
            rows[case_id].append(round_rows[case_id])
        write(output, rows)
    if failures:
        for rounds in rows.values():
            for row in rounds:
                for failed in failures:
                    row.pop(failed, None)
        write(output, rows)
    return rows


def fresh_expected(bundle, cases, python, deadline):
    manifest = read_json(bundle / "manifest.json")
    result = {}
    for case in cases:
        left = deadline - time.perf_counter()
        if left <= 0:
            raise TimeoutError("deployment reference budget exhausted")
        with CallableApplication(
            bundle, python=python, variant="reference", mode="oneshot", timeout=min(30, left)
        ) as app:
            values = app.run_calls(inputs(case["calls"], bundle / "reference"))
            result[case["id"]] = (
                app.last_receipt["fingerprint"],
                fingerprint(values, manifest["max_output_bytes"]),
            )
    return result


def select_deployment(
    *,
    output,
    reference,
    reference_source,
    value,
    cases,
    env,
    search_folder,
    mode,
    expected_requests,
    deadline,
    progress,
):
    from .workflow import make_bundle

    output.mkdir(parents=True)
    report = dict(
        status="incomplete",
        requested_mode=mode,
        expected_requests=expected_requests,
        timing_scope="Complete CallableApplication call plus adapter/worker startup and teardown amortized over expected_requests. Includes copies, checks, observations and returned values.",
        memory_scope="Parent RSS plus sampled worker/process-tree peaks; shared pages may be counted twice and very short descendants may be missed.",
        selection_split="discovery",
        evaluation_retries=0,
        llm_calls=0,
    )
    bundles, sources = {}, {"reference": (None, None)}
    incumbent_id = "reference"
    if search_folder and (search_folder / "report.json").exists():
        history = read_json(search_folder / "report.json")
        frozen = search_folder / "frozen_winner.json"
        if frozen.exists():
            incumbent_id = read_json(frozen)["id"]
        for record in history["candidates"]:
            if record.get("discovery", {}).get("status") == "matched" and record.get(
                "discovery_resources", {}
            ).get("eligible", True):
                source = search_folder / "candidates" / record["id"] / "project"
                sources[record["id"]] = (source, snapshot(source, value))
    # The incumbent identity was frozen by callable discovery, not selected from
    # held-out performance. Its failed evaluation cannot make another choice win.
    incumbent_id = incumbent_id if incumbent_id in sources else "reference"
    for name, (source, snap) in sources.items():
        folder = output / name
        folder.mkdir()
        bundles[name] = make_bundle(
            folder,
            reference,
            reference_source,
            source,
            snap,
            value,
            cases,
            env,
            "deployment_pending",
        )
    modes = ["oneshot", "prepared"] if mode == "auto" else [mode]
    options = [
        dict(id=name + ":" + lifecycle, source=name, mode=lifecycle, bundle=bundles[name])
        for name in sources
        for lifecycle in modes
    ]
    # Always keep a fresh reference as an independent lifecycle baseline.
    if not any(o["id"] == "reference:oneshot" for o in options):
        options.insert(
            0,
            dict(
                id="reference:oneshot",
                source="reference",
                mode="oneshot",
                bundle=bundles["reference"],
            ),
        )
    option_map = {o["id"]: o for o in options}
    discovery = [c for c in cases if c["split"] == "discovery"]
    evaluation = [c for c in cases if c["split"] == "evaluation"]
    winner = option_map["reference:oneshot"]
    selected = (None, None)
    try:
        expected = fresh_expected(
            bundles["reference"], discovery, value["environment"]["python"], deadline
        )
        failures = {}
        measured = measure(
            options,
            discovery,
            expected,
            rounds=2,
            requests=2,
            expected_requests=expected_requests,
            python=value["environment"]["python"],
            deadline=deadline,
            progress=progress,
            output=output / "discovery.json",
            failures=failures,
        )
        report["discovery_failures"] = failures
        options = [
            o
            for o in options
            if o["id"] not in failures and "reference:" + o["mode"] not in failures
        ]
        option_map = {o["id"]: o for o in options}
        # Rank using only discovery. An optimized source must beat the original
        # in the SAME mode; otherwise any speedup is only worker reuse.
        eligible_options = []
        all_scores = []
        for option in options:
            original = "reference:" + option["mode"]
            score = math.exp(
                statistics.mean(
                    math.log(statistics.median(r[option["id"]]["sequence_seconds"] for r in rounds))
                    for rounds in measured.values()
                )
            )
            all_scores.append((score, option["id"]))
            if option["source"] != "reference" and not passes(
                paired(measured, original, option["id"]), final=False
            ):
                continue
            eligible_options.append((score, option["id"]))
        winner = option_map[min(eligible_options)[1]]
        incumbent_options = [
            (score, name)
            for score, name in all_scores
            if option_map[name]["source"] == incumbent_id
        ]
        if not incumbent_options:
            # An incumbent that failed discovery lifecycle/exactness cannot be
            # deployed; its rejection remains in discovery_failures.
            incumbent_options = [
                (score, name)
                for score, name in all_scores
                if option_map[name]["source"] == "reference"
            ]
        incumbent = option_map[min(incumbent_options)[1]]
        guards = []
        if winner["source"] != "reference":
            guards.append(("same_mode_reference", "reference:" + winner["mode"], 1.02))
        if winner["mode"] == "prepared":
            guards.append(("fresh_reference", "reference:oneshot", 1.02))
        if incumbent["id"] != winner["id"]:
            guards.append(("incumbent", incumbent["id"], 1 / 1.03))
        # Avoid testing an identical comparison twice.
        unique_guards = {}
        for label, baseline, minimum in guards:
            if baseline not in unique_guards or minimum > unique_guards[baseline][2]:
                unique_guards[baseline] = (label, baseline, minimum)
        guards = list(unique_guards.values())
        report["frozen_selection"] = dict(
            winner=winner["id"],
            incumbent=incumbent["id"],
            callable_incumbent=incumbent_id,
            guards=guards,
            discovery_scores=all_scores,
        )
        report["source_identities"] = {
            name: identity(read_json(bundle / "manifest.json")) for name, bundle in bundles.items()
        }
        write(output / "frozen-selection.json", report["frozen_selection"])
        progress(f"DEPLOYMENT frozen={winner['id']} incumbent={incumbent['id']}; evaluation opened")
        final_options = [winner] + [option_map[g[1]] for g in guards]
        expected = fresh_expected(
            bundles["reference"], evaluation, value["environment"]["python"], deadline
        )
        final = measure(
            final_options,
            evaluation,
            expected,
            rounds=6,
            requests=3,
            expected_requests=expected_requests,
            python=value["environment"]["python"],
            deadline=deadline,
            progress=progress,
            output=output / "evaluation.json",
        )
        report["comparisons"] = {}
        accepted = True
        for label, baseline, minimum in guards:
            comparison = paired(final, baseline, winner["id"], minimum)
            comparison["accepted"] = passes(
                comparison, final=True, minimum=minimum, significance=0.05 / max(1, len(guards))
            )
            accepted &= comparison["accepted"]
            report["comparisons"][label] = comparison
        report["lifecycle_exact"] = True
        if accepted:
            report.update(
                status="accepted" if winner["source"] != "reference" else "reference_retained",
                mode=winner["mode"],
                source=winner["source"],
            )
            selected = sources[winner["source"]]
        else:
            report.update(
                status="reference_retained",
                mode="oneshot",
                source="reference",
                reason="frozen deployment failed performance or memory gates; no evaluation retry",
            )
    except (ValueError, OSError, TimeoutError, ImportError) as error:
        report.update(
            status="failed",
            mode="oneshot",
            source="reference",
            lifecycle_exact=False,
            reason=str(error),
        )
    report.setdefault("mode", "oneshot")
    report.setdefault("source", "reference")
    write(output / "report.json", report)
    return (*selected, report)


def certify(bundle, report):
    manifest = read_json(bundle / "manifest.json")
    write(
        bundle / "callable.json",
        dict(
            status=report["status"],
            manifest_identity=identity(manifest),
            input_mode="values",
            runtime_features=sorted((*DEFAULT_FEATURES, "values")),
            mode=report["mode"],
            lifecycle_exact=report.get("lifecycle_exact", False),
            expected_requests=report["expected_requests"],
            timing_scope=report["timing_scope"],
            comparisons=report.get("comparisons", {}),
        ),
    )
