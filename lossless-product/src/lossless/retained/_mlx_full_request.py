"""Fixed-count inference with overlapped export materialization.

Persistent state contains immutable model references and bounded compiled
functions. Every call creates fresh request state. Finished probability stacks
and cache exports join subsequent decode evaluations; the final evaluation
materializes the last KV update, all exports, and all captured probabilities.
"""

import time
import mlx.core as mx
from mlx_lm.generate import generation_stream
from mlx_lm.models import llama
from mlx_lm.models.cache import BatchKVCache, KVCache, make_prompt_cache
from lossless.adapters.mlx.batching import output_counts
from lossless.adapters.mlx.copy_kernel import kernel as copy_kernel
from lossless.adapters.mlx.schedule import cohorts


_MIB = 1024**2
_COMPACT_LIMIT = 1024


def _append(oldk, oldv, newk, newv, previous):
    if oldk is None:
        return newk, newv

    batch, heads, added, dimension = newk.shape
    total = previous + added
    capacity = oldk.shape[2]
    size = batch * heads * total * dimension
    old_size = batch * heads * capacity * dimension

    if (
        added == 1
        and total <= _COMPACT_LIMIT
        and min(batch, heads, dimension) > 0
        and newk.shape == newv.shape
        and newk.dtype == newv.dtype
        and newk.dtype in (mx.float16, mx.float32)
        and oldk.shape == oldv.shape
        and oldk.shape[:2] == (batch, heads)
        and oldk.shape[3] == dimension
        and oldk.dtype == newk.dtype
        and oldv.dtype == newv.dtype
        and 0 <= previous <= capacity
        and max(size, old_size) < 2**32
        and max(size, old_size) * newk.itemsize <= 4294967296
    ):
        shape = (batch, heads, total, dimension)
        metadata = mx.array([capacity, previous, added, dimension, size], dtype=mx.uint32)
        result = copy_kernel("flat")(
            inputs=[oldk, oldv, newk, newv, metadata],
            template=[("T", newk.dtype)],
            grid=(size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[shape, shape],
            output_dtypes=[newk.dtype, newv.dtype],
        )
        return result[0], result[1]

    return (
        mx.concatenate([oldk[..., :previous, :], newk], axis=2),
        mx.concatenate([oldv[..., :previous, :], newv], axis=2),
    )


class _HybridKV(KVCache):
    def update_and_fetch(self, keys, values):
        previous = self.offset
        added = keys.shape[2]
        if previous + added > _COMPACT_LIMIT:
            return KVCache.update_and_fetch(self, keys, values)
        self.keys, self.values = _append(self.keys, self.values, keys, values, previous)
        self.offset = previous + added
        return self.keys, self.values


class _HybridBatchKV(BatchKVCache):
    def update_and_fetch(self, keys, values):
        previous = self._idx
        added = keys.shape[2]
        if previous + added > _COMPACT_LIMIT:
            return BatchKVCache.update_and_fetch(self, keys, values)
        self.keys, self.values = _append(self.keys, self.values, keys, values, previous)
        self._idx = previous + added
        self.offset = self.offset + added
        return self.keys, self.values


class Runner:
    def __init__(self, model):
        if model.model_type != "llama" or model.model.swa_idx is not None:
            raise ValueError("requires the supplied full-attention Llama model")
        self.model = model
        self.owner = model.model
        self._functions = {}
        self._capacity = 256

    def _build_prefill(self, index, layer_count, populated):
        owner = self.owner

        def execute(tokens, states):
            caches = []
            for layer in range(layer_count):
                cache = KVCache()
                cache.offset = index
                if populated:
                    cache.keys, cache.values = states[layer]
                caches.append(cache)
            owner(tokens, cache=caches)
            return [(cache.keys, cache.values) for cache in caches]

        return mx.compile(execute)

    def _prefill_chunk(self, tokens, caches):
        index = caches[0].offset
        populated = caches[0].keys is not None
        signature = tuple(
            None
            if cache.keys is None
            else (
                tuple(cache.keys.shape),
                str(cache.keys.dtype),
                tuple(cache.values.shape),
                str(cache.values.dtype),
            )
            for cache in caches
        )
        key = ("prefill", tuple(tokens.shape), str(tokens.dtype), index, signature)
        function = self._functions.get(key)
        if function is None:
            if len(self._functions) >= self._capacity:
                self.owner(tokens, cache=caches)
                return
            function = self._build_prefill(index, len(caches), populated)
            self._functions[key] = function
        states = [(c.keys, c.values) for c in caches] if populated else []
        updated = function(tokens, states)
        end = index + tokens.shape[1]
        for cache, (keys, values) in zip(caches, updated):
            cache.keys, cache.values = keys, values
            cache.offset = end

    def _prefill(self, ids):
        original = make_prompt_cache(self.model)
        if any(type(c) is not KVCache for c in original):
            raise ValueError("requires ordinary, unquantized KV caches")
        end = len(ids) - 1
        for offset in range(0, end, 2048):
            stop = min(offset + 2048, end)
            self._prefill_chunk(mx.array([ids[offset:stop]]), original)
            if stop < end:
                mx.eval([c.state for c in original])
        result = []
        for source in original:
            cache = _HybridKV()
            cache.keys, cache.values = source.keys, source.values
            cache.offset = source.offset
            result.append(cache)
        return result

    @staticmethod
    def _merge(requests):
        result = []
        batch = len(requests)
        index = requests[0][0].offset
        offsets = mx.full((batch,), index, dtype=mx.int32)
        for layer in range(len(requests[0])):
            sources = [request[layer] for request in requests]
            cache = _HybridBatchKV([0] * batch)
            if index:
                cache.keys = mx.concatenate([c.keys[..., :index, :] for c in sources], axis=0)
                cache.values = mx.concatenate([c.values[..., :index, :] for c in sources], axis=0)
            cache._idx = index
            cache.offset = offsets
            result.append(cache)
        return result

    @staticmethod
    def _layout(caches):
        if caches[0].keys is None:
            return None, 0
        layout = tuple(
            (
                c.keys.shape[1],
                c.keys.shape[3],
                str(c.keys.dtype),
                c.values.shape[1],
                c.values.shape[3],
                str(c.values.dtype),
            )
            for c in caches
        )
        row_bytes = sum(
            c.keys.shape[1] * c.keys.shape[3] * c.keys.itemsize
            + c.values.shape[1] * c.values.shape[3] * c.values.itemsize
            for c in caches
        )
        return layout, row_bytes

    def _forward(self, tokens, caches, head_groups):
        owner = self.owner
        h = owner.embed_tokens(tokens)
        mask = llama.create_attention_mask(h, caches[owner.fa_idx])
        for layer, cache in zip(owner.layers, caches):
            h = layer(h, mask, cache)
        h = owner.norm(h)
        if not any(live for _, live in head_groups):
            # Every final KV dependency is retained through returned states.
            return mx.zeros((0,), dtype=h.dtype)
        project = (
            owner.embed_tokens.as_linear
            if self.model.args.tie_word_embeddings
            else self.model.lm_head
        )
        parts = []
        offset = 0
        for count, live in head_groups:
            if live:
                x = h if h.shape[0] <= 4 else h[offset : offset + count]
                parts.append(project(x))
            else:
                parts.append(mx.zeros((count, 1, self.model.args.vocab_size), dtype=h.dtype))
            offset += count
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)

    def _build_decode(self, index, batch, batched, head_groups):
        def execute(tokens, states):
            caches = []
            if batched:
                offsets = mx.full((batch,), index, dtype=mx.int32)
                padding = mx.zeros((batch,), dtype=mx.int32)
            for keys, values in states:
                if batched:
                    cache = object.__new__(_HybridBatchKV)
                    cache._idx = index
                    cache.offset = offsets
                    cache.left_padding = padding
                    cache._right_padding = None
                else:
                    cache = object.__new__(_HybridKV)
                    cache.offset = index
                cache.keys, cache.values = keys, values
                caches.append(cache)
            logits = self._forward(tokens, caches, head_groups)
            return logits, [(c.keys, c.values) for c in caches]

        return mx.compile(execute)

    def _decode(self, tokens, caches, batched, head_groups, layout):
        if layout is None:
            return self._forward(tokens, caches, head_groups)
        index = caches[0]._idx if batched else caches[0].offset
        key = (
            "decode",
            batched,
            tuple(tokens.shape),
            str(tokens.dtype),
            index,
            caches[0].keys.shape[2],
            head_groups,
            layout,
        )
        function = self._functions.get(key)
        if function is None:
            if len(self._functions) >= self._capacity:
                return self._forward(tokens, caches, head_groups)
            function = self._build_decode(index, tokens.shape[0], batched, head_groups)
            self._functions[key] = function
        states = [(c.keys, c.values) for c in caches]
        logits, updated = function(tokens, states)
        if batched:
            offsets = mx.full((tokens.shape[0],), index + 1, dtype=mx.int32)
        for cache, (keys, values) in zip(caches, updated):
            cache.keys, cache.values = keys, values
            if batched:
                cache._idx = index + 1
                cache.offset = offsets
            else:
                cache.offset = index + 1
        return logits

    @staticmethod
    def _extract(caches, row, batched):
        result = []
        for source in caches:
            cache = KVCache()
            if batched:
                index = source._idx
                cache.keys = mx.contiguous(source.keys[row : row + 1, :, :index, :])
                cache.values = mx.contiguous(source.values[row : row + 1, :, :index, :])
            else:
                index = source.offset
                cache.keys, cache.values = source.keys, source.values
            cache.offset = index
            result.append(cache)
        return result

    @staticmethod
    def _keep_prefix(caches, count):
        for cache in caches:
            cache.keys = cache.keys[:count]
            cache.values = cache.values[:count]
            cache.offset = cache.offset[:count]
            cache.left_padding = cache.left_padding[:count]

    def run(self, prompts: list[list[int]], counts: list[int], capture=True):
        counts = output_counts(counts, len(prompts))
        mx.synchronize(generation_stream)
        mx.synchronize()
        started = time.perf_counter()
        groups = cohorts([len(p) for p in prompts], counts, "short_first")
        n = len(prompts)
        outputs = [[] for _ in range(n)]
        probability_rows = [[] for _ in range(n)]
        exported = [None] * n
        probabilities = [None] * n
        first = [None] * n
        completed = [None] * n

        with mx.stream(generation_stream):
            for group in groups:
                initial = [self._prefill(prompts[i]) for i in group]
                batched = len(group) > 1
                caches = self._merge(initial) if batched else initial[0]
                del initial
                layout, row_bytes = self._layout(caches)
                active = list(group)
                tokens = mx.array([[prompts[i][-1]] for i in active])
                token_blocks = []
                pending_materializations = []
                queued_steps = 0
                horizon = max(counts[i] for i in group)
                submission_budget = (48 if batched else 96) * _MIB
                maximum_window = 12 if batched else 16

                # Iteration count includes consumption of the last output token.
                for step in range(horizon + 1):
                    batch = len(active)
                    live = sum(counts[i] > step for i in active)
                    head_groups = tuple((min(4, batch - j), j < live) for j in range(0, batch, 4))
                    logits = self._decode(tokens, caches, batched, head_groups, layout)
                    if layout is None:
                        layout, row_bytes = self._layout(caches)

                    if live:
                        logits = logits[:, -1, :]
                        # Preserve the incumbent's uncompiled normalization
                        # operations and their reduction dimensions exactly.
                        if batched:
                            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                        else:
                            logprobs = logits - mx.logsumexp(logits, keepdims=True)
                        next_tokens = mx.argmax(logprobs, axis=-1)
                        if capture:
                            for j in range(live):
                                probability_rows[active[j]].append(logprobs[j])

                        if step == 0:
                            mx.eval(next_tokens, logprobs)
                            ids = next_tokens.tolist()
                            observed = time.perf_counter() - started
                            for j, request in enumerate(active):
                                outputs[request].append(ids[j])
                                first[request] = observed
                        else:
                            # Submit decoding promptly. Exports constructed
                            # after the previous submission join this one.
                            mx.async_eval(next_tokens, logprobs, pending_materializations)
                            pending_materializations = []
                            block = next_tokens if live == batch else next_tokens[:live]
                            token_blocks.append((active[:live], block))

                    for j in range(live, batch):
                        request = active[j]
                        result = self._extract(caches, j, batched)
                        exported[request] = result
                        pending_materializations.extend(c.state for c in result)
                        if capture:
                            array = mx.stack(probability_rows[request])
                            probabilities[request] = array
                            probability_rows[request].clear()
                            pending_materializations.append(array)

                    # The final step joins the complete group evaluation below.
                    # Earlier windows still bound asynchronous submission for
                    # arbitrary request lengths and output counts.
                    if step and live:
                        queued_steps += 1
                        allocated_bytes = batch * caches[0].keys.shape[2] * row_bytes
                        if allocated_bytes > 16 * _MIB:
                            window = 2
                        else:
                            window = min(
                                maximum_window,
                                max(3, submission_budget // max(allocated_bytes, 1)),
                            )
                        if queued_steps >= window:
                            mx.synchronize(generation_stream)
                            queued_steps = 0

                    if live:
                        if live < batch:
                            self._keep_prefix(caches, live)
                        active = active[:live]
                        next_input = next_tokens if live == batch else next_tokens[:live]
                        tokens = next_input[:, None]

                if token_blocks:
                    blocks = [block for _, block in token_blocks]
                    packed_tokens = (
                        blocks[0] if len(blocks) == 1 else mx.concatenate(blocks, axis=0)
                    )
                else:
                    packed_tokens = mx.array([], dtype=mx.uint32)

                # These roots include all final KV dependencies, independently
                # exported caches, complete distributions, and generated IDs.
                mx.eval(
                    packed_tokens,
                    [probabilities[i] for i in group] if capture else [],
                    [c.state for i in group for c in exported[i]],
                )
                pending_materializations.clear()
                flat_tokens = packed_tokens.tolist()
                position = 0
                for requests, _ in token_blocks:
                    for request in requests:
                        outputs[request].append(flat_tokens[position])
                        position += 1
                token_blocks.clear()
                observed = time.perf_counter() - started
                for request in group:
                    completed[request] = observed

        mx.synchronize(generation_stream)
        mx.synchronize()
        duration = time.perf_counter() - started
        if any(len(ids) != count for ids, count in zip(outputs, counts)):
            raise ValueError("unequal generated work")
        return (
            {
                "seconds": duration,
                "output_ids": outputs,
                "ttft_seconds": first,
                "completion_seconds": completed,
                "tokens_per_second": sum(counts) / duration,
            },
            exported,
            probabilities if capture else [],
        )
