"""One bounded MLX search in a dedicated process, with independent exact gates."""

from pathlib import Path
import random
import sys
import time

from ...jobs import read_json, identity
from ..._native.common import write, sha, stamp, interval
from ...api import save_report
from .runtime import admission, generate, validate_options, runtime_identity


def main(run):
    started = time.monotonic()
    config = read_json(run / "resolved.json")
    write(run / "runtime_identity.json", runtime_identity())
    deadline = started + config["budget"]["wall_time_seconds"]
    import mlx.core as mx
    from mlx_lm import load
    from .model import GroupedHead
    from .identity import fingerprint
    from .checks import diff, cache_diff

    model_path = config["workload"]["parameters"]["model"]
    model_identity = fingerprint(model_path)
    write(run / "model_identity.json", model_identity)
    hardware = mx.device_info()
    supplied = config["target"].get("supplied_profile")
    if supplied:
        import platform

        for key, actual in {"os": platform.system(), "machine": platform.machine()}.items():
            if supplied.get("host", {}).get(key, actual) != actual:
                raise ValueError(f"supplied hardware profile mismatches worker {key}")
    write(run / "hardware.json", hardware)
    allowed, reason = admission(model_path, hardware["device_name"])
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    grouped = GroupedHead(model, 4) if allowed else None
    loading_seconds = time.monotonic() - started
    requests = config["workload"]["inputs"]
    splits = {
        split: [r for r in requests if r["split"] == split] for split in ("discovery", "evaluation")
    }
    proof = None
    if config["proofs"]["check"]:
        from ...proofs import check

        proof = check(timeout=max(0, deadline - time.monotonic()))
        write(run / "proofs.json", proof)
    evaluations = {}

    def execute(rows, options, capture=False):
        return generate(
            model,
            grouped,
            tokenizer,
            [r["text"] for r in rows],
            [r["max_tokens"] for r in rows],
            optimized=options is not None,
            options=options,
            capture=capture,
        )

    def evaluate(name, options, split):
        rows = splits[split]
        reference = execute(rows, None, True)
        candidate = execute(rows, options, True)
        checks = []
        for i in range(len(rows)):
            checks.append(
                {
                    "tokens_equal": reference[0]["output_ids"][i] == candidate[0]["output_ids"][i],
                    "probabilities": diff(reference[2][i], candidate[2][i]),
                    "kv": cache_diff(reference[1][i], candidate[1][i]),
                }
            )
        exact = all(
            c["tokens_equal"]
            and c["probabilities"]["bitwise"]
            and c["probabilities"]["finite"]
            and c["kv"]["bitwise"]
            and c["kv"]["finite"]
            for c in checks
        )
        del reference, candidate
        samples = {"reference": [], "candidate": []}
        orders = []
        if exact:
            for repeat in range(config["workload"]["parameters"]["repeats"]):
                order = ["reference", "candidate"]
                random.Random(4100 + repeat).shuffle(order)
                orders.append(order)
                token_ids = {}
                for key in order:
                    row, _, _ = execute(rows, None if key == "reference" else options)
                    samples[key].append(row["seconds"])
                    token_ids[key] = row["output_ids"]
                if token_ids["reference"] != token_ids["candidate"]:
                    exact = False
                    break
                stamp(
                    f"MLX {split} candidate={name} batch={repeat + 1}/{config['workload']['parameters']['repeats']}"
                )
        import statistics

        speedup = (
            statistics.median(samples["reference"]) / statistics.median(samples["candidate"])
            if exact and samples["candidate"]
            else None
        )
        result = {
            "exact": exact,
            "checks": checks,
            "samples_seconds": samples,
            "orders": orders,
            "speedup": speedup,
            "ci95": interval(samples["reference"], samples["candidate"], 2026) if speedup else None,
        }
        write(run / (split + "_" + name + ".json"), result)
        return result

    selected = "reference"
    options = {}
    calls = 0
    confirmation = None
    if allowed:
        from .validation import cache_checks
        from .runtime import CACHE_RECIPE

        structural = cache_checks(CACHE_RECIPE)
        write(run / "cache_checks.json", structural)
        if not all(c["bitwise"] for c in structural):
            raise ValueError("retained cache recipe failed the independent cache gate")
        candidates = [("retained", {"allocator_cache_mib": 256})]
        discovery = evaluate("retained", candidates[0][1], "discovery")
        evaluations["retained"] = discovery
        reserve = (time.monotonic() - started) * 1.3
        if (
            config.get("llm")
            and config["budget"]["max_llm_calls"]
            and config["budget"]["max_candidates"] > 1
            and time.monotonic() + reserve < deadline
        ):
            from ...providers import call, bounded_timeout
            from ...proofs import search

            context = {
                "schema_version": 1,
                "adapter": "mlx.fixed_count",
                "contract": config["contract"],
                "hardware": hardware,
                "discovery_requests": splits["discovery"],
                "feedback": evaluations,
                "supplied_hardware_context": supplied,
                "formal_context": search("matrix index scheduling"),
                "response_format": {
                    "schema_version": 1,
                    "proposals": [
                        {
                            "id": "candidate_name",
                            "options": {"allocator_cache_mib": 64},
                            "hypothesis": "reason",
                        }
                    ],
                },
                "instructions": "Return JSON only. Select allocator_cache_mib 0, 64 or 256; every candidate preserves the retained exact computation recipe and undergoes independent equality checks. Do not change contract or evaluator.",
            }
            write(run / "llm_request.json", context)
            calls = 1
            try:
                response, seconds = call(
                    context,
                    config["llm"],
                    base=Path(read_json(run / "controller.json")["base"]),
                    timeout=bounded_timeout(config["llm"], deadline - reserve - time.monotonic()),
                )
                if (
                    not isinstance(response, dict)
                    or type(response.get("schema_version")) is not int
                    or response["schema_version"] != 1
                ):
                    raise ValueError("provider response needs schema_version=1")
                proposed = response.get("proposals", [])
                if (
                    not isinstance(proposed, list)
                    or not 1 <= len(proposed) <= config["budget"]["max_candidates"] - 1
                ):
                    raise ValueError("invalid proposal count")
                write(run / "llm_response.json", response)
                for proposal in proposed:
                    from ...jobs import fields

                    fields(
                        proposal,
                        {"id", "options", "hypothesis"},
                        "MLX proposal",
                        {"id", "options", "hypothesis"},
                    )
                    import re

                    name = proposal["id"]
                    if (
                        not isinstance(name, str)
                        or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name)
                        or name in evaluations
                        or name == "reference"
                    ):
                        raise ValueError("invalid/duplicate candidate identifier")
                    validate_options(proposal["options"])
                    if time.monotonic() + reserve >= deadline:
                        break
                    candidates.append((name, proposal["options"]))
                    evaluations[name] = evaluate(name, proposal["options"], "discovery")
                write(run / "llm_receipt.json", {"status": "completed", "seconds": seconds})
            except (ValueError, TimeoutError, EOFError, OSError) as error:
                write(
                    run / "llm_receipt.json",
                    {"status": "failed", "error_type": type(error).__name__},
                )
        eligible = [
            (name, opts)
            for name, opts in candidates
            if evaluations[name]["exact"]
            and evaluations[name]["speedup"] >= config["objective"]["min_speedup"]
        ]
        if eligible:
            finalist, chosen = max(eligible, key=lambda p: evaluations[p[0]]["speedup"])
            write(
                run / "selection.json",
                {
                    "candidate": finalist,
                    "options": chosen,
                    "rule": "best eligible discovery median; fresh evaluation must independently pass",
                },
            )
            confirmation = evaluate(finalist, chosen, "evaluation")
            if (
                confirmation["exact"]
                and confirmation["speedup"] >= config["objective"]["min_speedup"]
                and confirmation["ci95"][0] > 1
            ):
                selected, options = finalist, chosen
        else:
            write(
                run / "selection.json",
                {"candidate": "reference", "rule": "no discovery candidate qualified"},
            )
    summary = {
        "schema_version": 1,
        "adapter": "mlx.fixed_count",
        "status": "accepted" if selected != "reference" else "reference_retained",
        "selected": selected,
        "options": options,
        "admitted": allowed,
        "admission_reason": reason,
        "contract": config["contract"],
        "job_identity": identity(config),
        "wall_time_seconds": time.monotonic() - started,
        "model_loading_seconds": loading_seconds,
        "llm_calls": calls,
        "proofs": proof,
        "discovery": evaluations,
        "evaluation": confirmation,
        "scope": "Fixed-count greedy bulk inference, exclusive process, pinned model/runtime/device. Finite token/probability/active-KV checks. Timings include recipe setup, tokenization, execution and required state materialization; model loading is separate. Throughput gains do not imply faster first-token latency.",
    }
    save_report(run, summary)
    write(
        run / "sealed.json",
        {"hashes": {p.name: sha(p) for p in run.iterdir() if p.is_file() and p.suffix == ".json"}},
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
