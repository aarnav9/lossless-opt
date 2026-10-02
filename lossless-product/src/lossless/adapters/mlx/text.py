import time


def stopped_serial(model, prompts, counts, capture, stop_ids, cancel_after):
    import mlx.core as mx
    from mlx_lm.generate import generate_step, generation_stream
    from mlx_lm.models.cache import make_prompt_cache

    start = time.perf_counter()
    outputs, caches, probabilities, first, done, reasons = [], [], [], [], [], []
    for i, (ids, limit) in enumerate(zip(prompts, counts)):
        cache = make_prompt_cache(model)
        tokens, values = [], []
        t_first = None
        reason = "length"
        iterator = generate_step(mx.array(ids), model, max_tokens=limit, prompt_cache=cache)
        try:
            for token, lp in iterator:
                if t_first is None:
                    t_first = time.perf_counter() - start
                tokens.append(token)
                if capture:
                    values.append(lp)
                if token in stop_ids:
                    reason = "stop"
                    break
                if cancel_after is not None and len(tokens) >= cancel_after[i]:
                    reason = "cancelled"
                    break
        finally:
            iterator.close()
        mx.eval([c.state for c in cache])
        mx.synchronize(generation_stream)
        mx.synchronize()
        outputs.append(tokens)
        caches.append(cache)
        first.append(t_first)
        done.append(time.perf_counter() - start)
        reasons.append(reason)
        if capture:
            probabilities.append(mx.stack(values))
    return (
        {
            "output_ids": outputs,
            "ttft_seconds": first,
            "completion_seconds": done,
            "finish_reasons": reasons,
        },
        caches,
        probabilities,
    )
