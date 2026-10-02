"""Independent deterministic cache values, unequal-length fallback and growth."""


def cache_checks(recipe):
    import copy
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache, BatchKVCache
    from lossless.adapters.mlx.checks import diff, cache_diff, arrays
    from lossless.adapters.mlx.cache import cache_policy

    rows = []
    for lengths in [[0] * 5, [7] * 5, [9] * 5, [255] * 5, [257] * 5, [3, 5, 7, 9, 11]]:
        initial = []
        for i, length in enumerate(lengths):
            cache = KVCache()
            if length:
                x = (mx.arange(3 * length * 64).reshape(1, 3, length, 64) % 113).astype(
                    mx.float32
                ) / 128 + i
                cache.update_and_fetch(x, -x)
                mx.eval(cache.state)
            initial.append(cache)
        reference = BatchKVCache.merge(copy.deepcopy(initial))
        with cache_policy(recipe):
            candidate = BatchKVCache.merge(copy.deepcopy(initial))
        for step in range(4):
            B = 5 if step < 2 else 3
            x = (mx.arange(B * 3 * 64).reshape(B, 3, 1, 64) % 97).astype(mx.float32) / 64 + step
            a = reference.update_and_fetch(x, -x)
            with cache_policy(recipe):
                b = candidate.update_and_fetch(x, -x)
            mx.eval(a, b)
            kv = [diff(u, v) for u, v in zip(a, b)]
            rows.append(
                {
                    "lengths": lengths,
                    "step": step,
                    "kind": "append",
                    "bitwise": all(d["bitwise"] for d in kv),
                    "values": kv,
                }
            )
            if step == 1:
                keep = [0, 1, 2] if len(set(lengths)) == 1 and lengths[0] != 9 else [0, 2, 4]
                reference.filter(keep)
                with cache_policy(recipe):
                    candidate.filter(keep)
        for i in range(3):
            a = reference.extract(i)
            with cache_policy(recipe):
                b = candidate.extract(i)
            kd = cache_diff([a], [b])
            rows.append(
                {
                    "lengths": lengths,
                    "row": i,
                    "kind": "extract",
                    "bitwise": kd["bitwise"],
                    "kv": kd,
                }
            )
        with cache_policy(recipe):
            exported = [candidate.extract(i) for i in range(3)]
        before = [[arrays(c.keys)[1], arrays(c.values)[1]] for c in exported[1:]]
        exported[0].keys[..., 0, :] = 0
        exported[0].values[..., 0, :] = 0
        mx.eval(exported[0].state)
        after = [[arrays(c.keys)[1], arrays(c.values)[1]] for c in exported[1:]]
        rows.append(
            {"lengths": lengths, "kind": "export_mutation_isolated", "bitwise": before == after}
        )
    # Copy semantics must preserve bit patterns, including signed zero and NaN
    # payloads. Compare uint32 views directly, without floating-point arithmetic.
    bits = mx.array(
        [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC01234, 0x7F800001, 0x3F800000, 0xBF800000],
        dtype=mx.uint32,
    )
    x = mx.tile(bits, 64).reshape(1, 1, 8, 64).view(mx.float32)
    initial = []
    for _ in range(3):
        c = KVCache()
        c.update_and_fetch(x, x)
        initial.append(c)
    reference = BatchKVCache.merge(copy.deepcopy(initial))
    with cache_policy(recipe):
        candidate = BatchKVCache.merge(copy.deepcopy(initial))
    for _ in range(2):
        inp = mx.concatenate([x[:, :, :1, :]] * 3, axis=0)
        reference.update_and_fetch(inp, inp)
        with cache_policy(recipe):
            candidate.update_and_fetch(inp, inp)
    reference.filter([0, 2])
    with cache_policy(recipe):
        candidate.filter([0, 2])
    for i in range(2):
        a = reference.extract(i)
        with cache_policy(recipe):
            b = candidate.extract(i)
        same = all(
            arrays(u.view(mx.uint32))[1] == arrays(v.view(mx.uint32))[1]
            for u, v in zip(a.keys_and_values(), b.keys_and_values())
        )
        rows.append({"kind": "special_bit_patterns", "row": i, "bitwise": same})
    # A caller introducing padding after an equal-length empty merge must
    # invalidate the cached zero-padding invariant and use original semantics.
    reference = BatchKVCache.merge([KVCache() for _ in range(3)])
    with cache_policy(recipe):
        candidate = BatchKVCache.merge([KVCache() for _ in range(3)])
    reference.prepare(left_padding=[2, 0, 1])
    with cache_policy(recipe):
        candidate.prepare(left_padding=[2, 0, 1])
    inp = mx.arange(3 * 4 * 64).astype(mx.float32).reshape(3, 1, 4, 64)
    reference.update_and_fetch(inp, -inp)
    reference.filter([0, 2])
    with cache_policy(recipe):
        candidate.update_and_fetch(inp, -inp)
        candidate.filter([0, 2])
    for i in range(2):
        a = reference.extract(i)
        with cache_policy(recipe):
            b = candidate.extract(i)
        kd = cache_diff([a], [b])
        rows.append(
            {
                "kind": "padding_invalidates_guard",
                "row": i,
                "kv": kd,
                "bitwise": kd["bitwise"] and not getattr(candidate, "_layout_zero", False),
            }
        )
    return rows


def teacher_checks(model, prompts, counts, recipe, steps):
    import mlx.core as mx
    from mlx_lm.models.cache import BatchKVCache
    from lossless.adapters.mlx.checks import prepare_prefix
    from lossless.adapters.mlx.checks import diff, cache_diff
    from lossless.adapters.mlx.schedule import cohorts
    from lossless.adapters.mlx.execution import DeadHead
    from lossless.adapters.mlx.cache import cache_policy

    rows = []
    for group in cohorts([len(p) for p in prompts], counts, "short_first"):
        if len(group) == 1:
            continue
        initial_a = [prepare_prefix(model, prompts[i][:-1]) for i in group]
        initial_b = [prepare_prefix(model, prompts[i][:-1]) for i in group]
        a = [BatchKVCache.merge([c[j] for c in initial_a]) for j in range(len(initial_a[0]))]
        with cache_policy(recipe):
            b = [BatchKVCache.merge([c[j] for c in initial_b]) for j in range(len(initial_b[0]))]
        ds = [counts[i] for i in group]
        reference, candidate = DeadHead(model, ds), DeadHead(model, ds)
        active = list(group)
        tokens = [prompts[i][-1] for i in active]
        for t in range(min(steps, max(ds) + 1)):
            inp = mx.array(tokens)[:, None]
            la = reference(inp, cache=a)
            with cache_policy(recipe):
                lb = candidate(inp, cache=b)
            mx.eval(la, lb, [c.state for c in a + b])
            for j, request in enumerate(active):
                ea = [c.extract(j) for c in a]
                with cache_policy(recipe):
                    eb = [c.extract(j) for c in b]
                kd = cache_diff(ea, eb)
                ld = diff(la[j], lb[j]) if t < counts[request] else None
                rows.append(
                    {
                        "step": t,
                        "request": request,
                        "live": ld is not None,
                        "output": ld,
                        "kv": kd,
                        "bitwise": kd["bitwise"] and (ld is None or ld["bitwise"]),
                    }
                )
            keep = [j for j, i in enumerate(active) if t < counts[i]]
            if keep:
                tokens = [
                    (prompts[active[j]][-1] + 37 * (t + 1) + active[j]) % model.args.vocab_size
                    for j in keep
                ]
                for c in a:
                    c.filter(keep)
                with cache_policy(recipe):
                    for c in b:
                        c.filter(keep)
                active = [active[j] for j in keep]
    return rows
