"""The retained exact recipe in an explicitly exclusive MLX execution scope."""

from contextlib import contextmanager, nullcontext
import importlib.metadata
import json
from pathlib import Path
import threading
import time

LOCK = threading.Lock()
CACHE_RECIPE = {
    "zero_filter": True,
    "zero_extract": True,
    "prefix_slice": True,
    "append_metal": True,
    "threadgroup": 256,
    "append_limit": 64,
}


def runtime_identity():
    from .identity import sha

    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def admission(model_path, device_name, versions=None, *, model_identity=None):
    from .identity import fingerprint

    versions = versions or {n: importlib.metadata.version(n) for n in ["mlx", "mlx-lm"]}
    if versions != {"mlx": "0.32.3", "mlx-lm": "0.32.0"}:
        return False, "runtime outside validated MLX 0.32.3 / MLX-LM 0.32.0"
    if device_name != "Apple M2":
        return False, "hardware outside validated Apple M2"
    expected = json.loads(Path(__file__).with_name("model_identity.json").read_text())
    if (fingerprint(model_path) if model_identity is None else model_identity) != expected:
        return False, "model identity outside the validated 8-bit SmolLM2 artifact"
    return True, None


def validate_options(options):
    if not isinstance(options, dict) or set(options) - {
        "allocator_cache_mib",
        "execution_recipe",
        "graph_cache_entries",
    }:
        raise ValueError("unsupported MLX recipe options")
    if options.get("execution_recipe", "retained") not in {
        "retained",
        "decoder_026",
        "fullhead_027",
    }:
        raise ValueError("unsupported execution recipe")
    capacity = options.get("graph_cache_entries", 64)
    if type(capacity) is not int or not 1 <= capacity <= 128:
        raise ValueError("graph_cache_entries must be in 1..128")
    if "allocator_cache_mib" in options and (
        type(options["allocator_cache_mib"]) is not int
        or options["allocator_cache_mib"] not in {0, 64, 256}
    ):
        raise ValueError("allocator_cache_mib must be 0, 64 or 256")


@contextmanager
def allocator(options):
    import mlx.core as mx

    validate_options(options)
    original = mx.clear_cache
    limit = options.get("allocator_cache_mib", 0) * 1024**2
    if limit:

        def clear():
            if mx.get_cache_memory() > limit:
                return original()

        mx.clear_cache = clear
    try:
        yield
    finally:
        mx.clear_cache = original


def generate(
    stock,
    grouped,
    tokenizer,
    texts,
    counts,
    *,
    optimized=False,
    options=None,
    capture=False,
    stop_ids=None,
    cancel_after=None,
    fallback_reason=None,
    graph=None,
    profile=False,
):
    import mlx.core as mx
    from mlx_lm.generate import generation_stream
    from .batching import output_counts, serial
    from .cache import run
    from .text import stopped_serial

    if not LOCK.acquire(blocking=False):
        raise ValueError(
            "MLX adapter requires exclusive single-worker execution; overlapping calls are unsupported"
        )
    try:
        mx.synchronize(generation_stream)
        mx.synchronize()
        started = time.perf_counter()
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("nonempty text requests required")
        counts = output_counts(counts, len(texts))
        if optimized and (len(texts) > 64 or max(counts) > 256):
            optimized = False
            fallback_reason = "request counts outside the alpha optimization domain"
        prompts = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
        if any(not p or len(p) + n > 8192 for p, n in zip(prompts, counts)):
            raise ValueError("request outside the supported 8192-token context")
        if cancel_after is not None and (
            len(cancel_after) != len(texts)
            or any(type(n) is not int or n < 1 for n in cancel_after)
        ):
            raise ValueError("cancellation limits must be positive integers")
        encoded = time.perf_counter() - started
        if stop_ids or cancel_after is not None:
            value = stopped_serial(
                stock, prompts, counts, capture, set(stop_ids or []), cancel_after
            )
            fallback_reason = "stop/cancellation uses stock serial"
        elif optimized and grouped is not None:
            with allocator(options or {"allocator_cache_mib": 256}):
                recipe = CACHE_RECIPE
                if (options or {}).get("execution_recipe", "retained") != "retained":
                    if graph is None:
                        from .graph import GraphRecipe

                        graph = GraphRecipe(
                            stock,
                            options["execution_recipe"],
                            options.get("graph_cache_entries", 64),
                        )
                    if graph.model is not stock or graph.recipe != options["execution_recipe"]:
                        raise ValueError("graph belongs to a different model/recipe")
                    recipe = {**CACHE_RECIPE, "append_metal": False, "append_concat": True}
                with graph.installed() if graph is not None else nullcontext():
                    value = run(grouped, prompts, counts, recipe, capture)
        else:
            value = serial(stock, prompts, counts, capture)
        executed = time.perf_counter() if profile else 0
        row, caches, probs = value
        mx.eval([c.state for cs in caches for c in cs])
        mx.synchronize(generation_stream)
        mx.synchronize()
        materialized = time.perf_counter() if profile else 0
        row["output_texts"] = [tokenizer.decode(ids) for ids in row["output_ids"]]
        row["finish_reasons"] = row.get("finish_reasons", ["length"] * len(texts))
        row["response_bytes"] = len(
            json.dumps(
                {
                    "text": row["output_texts"],
                    "tokens": row["output_ids"],
                    "finish_reasons": row["finish_reasons"],
                },
                ensure_ascii=False,
            ).encode()
        )
        row["ttft_seconds"] = [v + encoded for v in row["ttft_seconds"]]
        row["completion_seconds"] = [v + encoded for v in row["completion_seconds"]]
        row["seconds"] = time.perf_counter() - started
        if profile:
            row["profile_stages_seconds"] = {
                "input_validation_and_tokenization": encoded,
                "execution": executed - started - encoded,
                "state_materialization": materialized - executed,
                "decode_and_response": row["seconds"] - (materialized - started),
            }
        row["tokens_per_second"] = sum(map(len, row["output_ids"])) / row["seconds"]
        row["fallback_reason"] = fallback_reason
        row["prompt_lengths"] = list(map(len, prompts))
        row["graph"] = (
            {**graph.stats, "entries": len(graph.functions), "capacity": graph.capacity}
            if graph is not None
            else None
        )
        return row, caches, probs
    finally:
        LOCK.release()
