from collections import Counter
import time
from .liveness import masks
from .schedule import cohorts

POLICY = {
    "serial_prefill": True,
    "group_size": 8,
    "bucket_by_length": True,
    "singleton_serial": True,
}


class DeadHead:
    def __init__(self, grouped_head, counts, enabled=True, trace=None):
        if grouped_head.model_type != "llama" or grouped_head.width != 4:
            raise ValueError("requires the pinned Llama head groups of four")
        masks(counts, 0)
        self.model, self.counts = grouped_head, list(counts)
        self.enabled, self.trace, self.step = enabled, trace, 0

    def __getattr__(self, key):
        return getattr(self.model, key)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        import mlx.core as mx
        from mlx_lm.models.cache import BatchKVCache, KVCache

        if input_embeddings is not None or not cache:
            raise ValueError("fixed token-input/cache contract required")
        if all(type(c) is KVCache for c in cache):
            if inputs.shape[0] != 1 or self.step:
                raise ValueError("unexpected serial prefix after generation started")
            return self.model(inputs, cache=cache)
        if not all(type(c) is BatchKVCache for c in cache) or inputs.shape[1] != 1:
            raise ValueError("requires one-position BatchKVCache generation")
        groups = masks(self.counts, self.step, self.model.width)
        if inputs.shape[0] != sum(len(ns) for ns, _ in groups):
            raise ValueError("active-row/count contract violated")
        inner = self.model.inner
        h = inner.model(inputs, cache)
        project = (
            inner.model.embed_tokens.as_linear if inner.args.tie_word_embeddings else inner.lm_head
        )
        parts, offset, skipped = [], 0, False
        for ns, live in groups:
            drop = self.enabled and not live
            if drop:
                parts.append(mx.zeros((len(ns), 1, inner.args.vocab_size), dtype=h.dtype))
                skipped = True
            else:
                # Match GroupedHead exactly: do not even introduce a full slice
                # when the original call fits in one head group.
                x = h if h.shape[0] <= self.model.width else h[offset : offset + len(ns)]
                parts.append(project(x))
            if self.trace is not None:
                self.trace["skipped_head_calls" if drop else "head_calls"] += 1
            offset += len(ns)
        if skipped:
            # The head normally forces these dependencies. Explicitly evaluate
            # the retained state even when every head output is dead.
            mx.async_eval([c.state for c in cache])
        if self.trace is not None:
            self.trace["body_calls"] += 1
        self.step += 1
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)


def scheduled(model, prompts, counts, capture=False, enabled=True, trace=False):
    import mlx.core as mx
    from mlx_lm.generate import generation_stream
    from lossless.adapters.mlx.batching import serial, native

    mx.synchronize(generation_stream)
    mx.synchronize()
    start = time.perf_counter()
    groups = cohorts([len(p) for p in prompts], counts, "short_first")
    outputs, caches, probabilities, first, done = [[None] * len(prompts) for _ in range(5)]
    stats = Counter(body_calls=0, head_calls=0, skipped_head_calls=0) if trace else None
    for group in groups:
        delay = time.perf_counter() - start
        ds = [counts[i] for i in group]
        wrapper = model if len(group) == 1 else DeadHead(model, ds, enabled, stats)
        fn = serial if len(group) == 1 else native
        row, states, values = fn(wrapper, [prompts[i] for i in group], ds, capture, POLICY)
        if len(group) > 1 and wrapper.step != max(ds) + 1:
            raise ValueError("generation pipeline call count changed")
        for j, i in enumerate(group):
            outputs[i], caches[i] = row["output_ids"][j], states[j]
            first[i], done[i] = delay + row["ttft_seconds"][j], delay + row["completion_seconds"][j]
            if capture:
                probabilities[i] = values[j]
    mx.synchronize(generation_stream)
    mx.synchronize()
    duration = time.perf_counter() - start
    return (
        {
            "seconds": duration,
            "output_ids": outputs,
            "ttft_seconds": first,
            "completion_seconds": done,
            "tokens_per_second": sum(counts) / duration,
            "trace": dict(stats) if trace else None,
        },
        caches,
        probabilities if capture else [],
    )


def teacher_checks(model, prompts, counts):
    """Identical forced histories through every stop, including mixed head groups."""
    import mlx.core as mx
    from mlx_lm.models.cache import BatchKVCache
    from lossless.adapters.mlx.checks import prepare_prefix
    from lossless.adapters.mlx.checks import diff, cache_diff

    rows = []
    for group in cohorts([len(p) for p in prompts], counts, "short_first"):
        if len(group) == 1:
            continue
        original = [prepare_prefix(model, prompts[i][:-1]) for i in group]
        copied = [prepare_prefix(model, prompts[i][:-1]) for i in group]
        a = [BatchKVCache.merge([c[j] for c in original]) for j in range(len(original[0]))]
        b = [BatchKVCache.merge([c[j] for c in copied]) for j in range(len(copied[0]))]
        ds = [counts[i] for i in group]
        candidate = DeadHead(model, ds)
        active = list(group)
        tokens = [prompts[i][-1] for i in active]
        for t in range(max(ds) + 1):
            inp = mx.array(tokens)[:, None]
            la, lb = model(inp, cache=a), candidate(inp, cache=b)
            mx.eval(la, lb, [c.state for c in a], [c.state for c in b])
            for j, request in enumerate(active):
                kv = cache_diff([c.extract(j) for c in a], [c.extract(j) for c in b])
                logits = diff(la[j], lb[j]) if t < counts[request] else None
                rows.append(
                    {
                        "step": t,
                        "request": request,
                        "live": logits is not None,
                        "logits": logits,
                        "kv": kv,
                        "bitwise": kv["bitwise"] and (logits is None or logits["bitwise"]),
                    }
                )
            keep = [j for j, i in enumerate(active) if t < counts[i]]
            if keep:
                # Fixed, independent histories exercise outputs off the greedy path.
                tokens = [
                    (prompts[active[j]][-1] + 31 * (t + 1) + active[j]) % model.args.vocab_size
                    for j in keep
                ]
                for c in a + b:
                    c.filter(keep)
                active = [active[j] for j in keep]
    mx.synchronize()
    return rows
