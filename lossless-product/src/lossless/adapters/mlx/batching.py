import time
import mlx.core as mx
from mlx_lm.generate import BatchGenerator, generate_step, generation_stream
from mlx_lm.models.cache import make_prompt_cache


def output_counts(count, n):
    counts = [count] * n if type(count) is int else list(count)
    if len(counts) != n or any(type(x) is not int or x < 1 for x in counts):
        raise ValueError("one positive output count per request required")
    return counts


def serial(model, prompts, count, capture=False, options=None):
    counts = output_counts(count, len(prompts))
    mx.synchronize(generation_stream)
    mx.synchronize()
    start = time.perf_counter()
    outputs = []
    caches = []
    probs = []
    ttfts = []
    latencies = []
    for ids, limit in zip(prompts, counts):
        cache = make_prompt_cache(model)
        tokens = []
        values = []
        first = None
        for token, lp in generate_step(mx.array(ids), model, max_tokens=limit, prompt_cache=cache):
            if first is None:
                first = time.perf_counter()
            tokens.append(token)
            if capture:
                values.append(lp)
        mx.synchronize(generation_stream)
        mx.synchronize()
        outputs.append(tokens)
        caches.append(cache)
        ttfts.append(first - start)
        latencies.append(time.perf_counter() - start)
        if capture:
            probs.append(mx.stack(values))
    mx.synchronize()
    duration = time.perf_counter() - start
    return (
        {
            "seconds": duration,
            "output_ids": outputs,
            "ttft_seconds": ttfts,
            "completion_seconds": latencies,
            "tokens_per_second": sum(counts) / duration,
        },
        caches,
        probs,
    )


def native(model, prompts, count, capture=False, options=None):
    options = options or {}
    counts = output_counts(count, len(prompts))
    mx.synchronize()
    start = time.perf_counter()
    initial_caches = None
    all_tokens = None
    if options.get("serial_prefill"):
        initial_caches = []
        all_tokens = []
        for ids in prompts:
            cache = make_prompt_cache(model)
            for offset in range(0, len(ids) - 1, 2048):
                model(mx.array([ids[offset : min(offset + 2048, len(ids) - 1)]]), cache=cache)
                mx.eval([c.state for c in cache])
                mx.clear_cache()
            initial_caches.append(cache)
            all_tokens.append(ids[:-1])
        prompts = [ids[-1:] for ids in prompts]
    gen = BatchGenerator(
        model,
        max_tokens=max(counts),
        stop_tokens=[],
        prefill_batch_size=options.get("prefill_batch_size", len(prompts)),
        completion_batch_size=options.get("completion_batch_size", len(prompts)),
        prefill_step_size=options.get("prefill_step_size", 2048),
    )
    original_logsumexp = mx.logsumexp
    normalization_calls = 0

    def row_logsumexp(a, axis=None, keepdims=False, *, stream=None):
        nonlocal normalization_calls
        if a.ndim == 2 and a.shape[0] > 1 and axis == -1 and keepdims:
            normalization_calls += 1
            return mx.concatenate(
                [
                    original_logsumexp(a[i : i + 1], keepdims=True, stream=stream)
                    for i in range(a.shape[0])
                ],
                axis=0,
            )
        return original_logsumexp(a, axis=axis, keepdims=keepdims, stream=stream)

    # Narrow research-only process-global override, restored before returning.
    # Runs are single-threaded, no external requests or logits processors.
    if options.get("row_logsumexp"):
        mx.logsumexp = row_logsumexp
    try:
        uids = gen.insert(prompts, max_tokens=counts, caches=initial_caches, all_tokens=all_tokens)
        tokens = {u: [] for u in uids}
        values = {u: [] for u in uids}
        caches = {}
        first = {}
        done = {}
        while responses := gen.next_generated():
            for r in responses:
                first.setdefault(r.uid, time.perf_counter() - start)
                tokens[r.uid].append(r.token)
                if capture:
                    values[r.uid].append(r.logprobs)
                if r.finish_reason is not None:
                    caches[r.uid] = r.prompt_cache
                    done[r.uid] = time.perf_counter() - start
    finally:
        mx.logsumexp = original_logsumexp
        gen.close()
    mx.synchronize()
    duration = time.perf_counter() - start
    if any(len(tokens[u]) != limit for u, limit in zip(uids, counts)):
        raise ValueError("unequal generated work")
    return (
        {
            "seconds": duration,
            "output_ids": [tokens[u] for u in uids],
            "ttft_seconds": [first[u] for u in uids],
            "completion_seconds": [done[u] for u in uids],
            "tokens_per_second": sum(counts) / duration,
            "row_normalization_calls": normalization_calls,
        },
        [caches[u] for u in uids],
        ([mx.stack(values[u]) for u in uids] if capture else []),
    )
