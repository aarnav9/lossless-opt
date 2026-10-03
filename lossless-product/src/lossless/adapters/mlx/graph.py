"""Opt-in extraction of 026 decoder and 027 full-head capture.

Exclusive immutable-model scope only. Every shape/dtype/index specialization has
its own key. Once capacity is reached, unseen keys execute the retained recipe;
we never accumulate an unbounded internal specialization cache or evict live work.
Finite equality validation remains necessary for the pinned MLX compiler.
"""

from contextlib import contextmanager


class GraphRecipe:
    def __init__(self, model, recipe, capacity=64):
        if (
            recipe not in {"decoder_026", "fullhead_027"}
            or type(capacity) is not int
            or not 1 <= capacity <= 128
        ):
            raise ValueError("unsupported graph recipe/capacity")
        self.model, self.recipe, self.capacity = model, recipe, capacity
        self.functions = {}
        self.stats = {"builds": 0, "hits": 0, "capacity_fallbacks": 0, "admission_fallbacks": 0}

    @contextmanager
    def installed(self):
        import mlx.core as mx
        from mlx_lm.models import llama
        from mlx_lm.models.cache import BatchKVCache
        from .execution import DeadHead
        from .liveness import masks

        owner = self.model.model
        original_body, original_head = llama.LlamaModel.__call__, DeadHead.__call__

        def admitted(inputs, caches):
            return (
                inputs.ndim == 2
                and inputs.shape[1] == 1
                and 1 <= inputs.shape[0] <= 8
                and caches
                and len(caches) == len(owner.layers)
                and owner.swa_idx is None
                and all(
                    type(c) is BatchKVCache
                    and c.keys is not None
                    and c._idx == caches[0]._idx
                    and 0 <= c._idx <= 8192
                    and getattr(c, "_layout_zero", False)
                    and c._right_padding is None
                    for c in caches
                )
            )

        def states(caches):
            return [(c.keys, c.values, c.offset, c.left_padding) for c in caches]

        def signature(inputs, caches, groups):
            return (
                tuple(inputs.shape),
                str(inputs.dtype),
                caches[0]._idx,
                groups,
                tuple(
                    tuple((tuple(a.shape), str(a.dtype)) for a in state) for state in states(caches)
                ),
            )

        def function(key, groups=None, width=4):
            if key in self.functions:
                self.stats["hits"] += 1
                return self.functions[key]
            if len(self.functions) >= self.capacity:
                self.stats["capacity_fallbacks"] += 1
                return None
            index = key[2]

            def execute(tokens, values):
                caches = []
                for state in values:
                    cache = object.__new__(BatchKVCache)
                    cache.keys, cache.values, cache.offset, cache.left_padding = state
                    cache._idx, cache._right_padding, cache._layout_zero = index, None, True
                    caches.append(cache)
                h = owner.embed_tokens(tokens)
                mask = (
                    None
                    if self.recipe == "fullhead_027"
                    else llama.create_attention_mask(h, caches[owner.fa_idx])
                )
                for layer, cache in zip(owner.layers, caches):
                    h = layer(h, mask, cache)
                h = owner.norm(h)
                if groups is not None:
                    project = (
                        owner.embed_tokens.as_linear
                        if self.model.args.tie_word_embeddings
                        else self.model.lm_head
                    )
                    parts, offset = [], 0
                    for count, live in groups:
                        if live:
                            part = h if h.shape[0] <= width else h[offset : offset + count]
                            parts.append(project(part))
                        else:
                            parts.append(
                                mx.zeros((count, 1, self.model.args.vocab_size), dtype=h.dtype)
                            )
                        offset += count
                    h = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
                return h, states(caches)

            fn = (
                mx.compile(execute)
                if self.recipe == "fullhead_027"
                else mx.compile(execute, inputs=owner.state)
            )
            self.functions[key] = fn
            self.stats["builds"] += 1
            return fn

        def commit(caches, values):
            for cache, state in zip(caches, values):
                cache.keys, cache.values, cache.offset, cache.left_padding = state
                cache._idx += 1

        def body(module, inputs, cache=None, input_embeddings=None):
            if module is not owner or input_embeddings is not None or not admitted(inputs, cache):
                self.stats["admission_fallbacks"] += 1
                return original_body(module, inputs, cache, input_embeddings)
            fn = function(signature(inputs, cache, None))
            if fn is None:
                return original_body(module, inputs, cache, input_embeddings)
            result, values = fn(inputs, states(cache))
            commit(cache, values)
            return result

        def head(wrapper, inputs, cache=None, input_embeddings=None):
            if (
                wrapper.model.inner is not self.model
                or input_embeddings is not None
                or not admitted(inputs, cache)
            ):
                self.stats["admission_fallbacks"] += 1
                return original_head(wrapper, inputs, cache, input_embeddings)
            groups = tuple(
                (len(ns), live or not wrapper.enabled)
                for ns, live in masks(wrapper.counts, wrapper.step, wrapper.model.width)
            )
            if sum(n for n, _ in groups) != inputs.shape[0]:
                raise ValueError("active row count drift")
            fn = function(signature(inputs, cache, groups), groups, wrapper.model.width)
            if fn is None:
                return original_head(wrapper, inputs, cache, input_embeddings)
            result, values = fn(inputs, states(cache))
            commit(cache, values)
            if any(not live for _, live in groups):
                mx.async_eval([c.state for c in cache])
            if wrapper.trace is not None:
                wrapper.trace["body_calls"] += 1
                for _, live in groups:
                    wrapper.trace["head_calls" if live else "skipped_head_calls"] += 1
            wrapper.step += 1
            return result

        try:
            if self.recipe == "fullhead_027":
                DeadHead.__call__ = head
            else:
                llama.LlamaModel.__call__ = body
            yield
        finally:
            llama.LlamaModel.__call__, DeadHead.__call__ = original_body, original_head
