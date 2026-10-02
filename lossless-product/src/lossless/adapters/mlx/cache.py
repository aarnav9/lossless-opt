"""Isolated scoped cache rewrites; numerical model operations stay unchanged."""

from contextlib import contextmanager
from collections import Counter


@contextmanager
def cache_policy(recipe, trace=None):
    import mlx.core as mx
    from mlx_lm.models.cache import BatchKVCache, KVCache

    allowed = {
        "zero_filter",
        "zero_extract",
        "prefix_slice",
        "concat_merge",
        "extract_view",
        "append_concat",
        "append_metal",
        "threadgroup",
        "append_limit",
        "vector_width",
        "copy_variant",
    }
    if set(recipe) - allowed:
        raise ValueError(f"unimplemented recipe: {recipe}")
    names = ["merge", "filter", "extract", "update_and_fetch", "prepare", "finalize", "extend"]
    originals = {n: BatchKVCache.__dict__[n] for n in names}
    if recipe:
        merge_original = BatchKVCache.merge
        filter_original = BatchKVCache.filter
        extract_original = BatchKVCache.extract
        update_original = BatchKVCache.update_and_fetch
        prepare_original = BatchKVCache.prepare
        extend_original = BatchKVCache.extend

        def merge(cls, caches):
            lengths = [c.size() for c in caches]
            equal = len(set(lengths)) == 1
            if (
                recipe.get("concat_merge")
                and equal
                and lengths[0] > 0
                and all(type(c) is KVCache and c.keys is not None for c in caches)
                and len(
                    {
                        (
                            c.keys.shape[1],
                            c.keys.shape[3],
                            c.values.shape[3],
                            str(c.keys.dtype),
                            str(c.values.dtype),
                        )
                        for c in caches
                    }
                )
                == 1
            ):
                # Layout.concat_assembly proves equivalence to row-by-row
                # scatter into the zero buffer for each valid tensor element.
                result = cls([0] * len(caches))
                result.keys = mx.concatenate([c.keys[..., : lengths[0], :] for c in caches], axis=0)
                result.values = mx.concatenate(
                    [c.values[..., : lengths[0], :] for c in caches], axis=0
                )
                result.offset += lengths[0]
                result._idx = lengths[0]
                if trace is not None:
                    trace["concat_merge_hits"] += 1
            else:
                result = merge_original(caches)
            result._layout_zero = equal
            return result

        def prepare(self, **kwargs):
            if any(kwargs.get("left_padding") or []) or any(kwargs.get("right_padding") or []):
                self._layout_zero = False
            return prepare_original(self, **kwargs)

        def extend(self, other):
            self._layout_zero = False
            return extend_original(self, other)

        def filtered(self, indices):
            if not (recipe.get("zero_filter") and getattr(self, "_layout_zero", False)):
                return filter_original(self, indices)
            # Layout.zero_padding_preserved: gathering zeros yields zeros, so
            # min padding is zero. Keep all original tensor gathers unchanged.
            selector = indices
            if recipe.get("prefix_slice") and indices == list(range(len(indices))):
                # Layout.prefix_gather. The runtime guard is precisely its
                # premise keep[b] = b; only row indexing changes to a view.
                selector = slice(0, len(indices))
                if trace is not None:
                    trace["prefix_slice_hits"] += 1
            if self.keys is not None:
                self.keys = self.keys[selector]
                self.values = self.values[selector]
            self.offset = self.offset[selector]
            self.left_padding = self.left_padding[selector]
            if trace is not None:
                trace["zero_filter_hits"] += 1

        def extracted(self, index):
            if not (recipe.get("zero_extract") and getattr(self, "_layout_zero", False)):
                return extract_original(self, index)
            # Layout.zero_padding_extract: same original slices/contiguity;
            # the known zero replaces a GPU metadata read, not any tensor math.
            cache = KVCache()
            k = self.keys[index : index + 1, :, : self._idx]
            v = self.values[index : index + 1, :, : self._idx]
            cache.keys = k if recipe.get("extract_view") else mx.contiguous(k)
            cache.values = v if recipe.get("extract_view") else mx.contiguous(v)
            cache.offset = self._idx
            if trace is not None:
                trace["zero_extract_hits"] += 1
            return cache

        def updated(self, keys, values):
            if not (
                (recipe.get("append_concat") or recipe.get("append_metal"))
                and getattr(self, "_layout_zero", False)
            ):
                return update_original(self, keys, values)
            if self._idx + keys.shape[2] > recipe.get("append_limit", 2**32):
                return update_original(self, keys, values)
            # Layout.append_implementation: retain [0, offset), then copy each
            # new element to offset+j. Capacity is unobserved; active bits are not.
            prev = self._idx
            if recipe.get("append_metal"):
                from lossless.adapters.mlx.copy_kernel import append

                self.keys, self.values = append(
                    self.keys,
                    self.values,
                    keys,
                    values,
                    prev,
                    recipe.get("threadgroup", 128),
                    recipe.get("vector_width", 1),
                    recipe.get("copy_variant", "flat"),
                )
            else:
                self.keys = (
                    keys
                    if self.keys is None
                    else mx.concatenate([self.keys[..., :prev, :], keys], axis=2)
                )
                self.values = (
                    values
                    if self.values is None
                    else mx.concatenate([self.values[..., :prev, :], values], axis=2)
                )
            if trace is not None and recipe.get("append_metal"):
                elements = self.keys.size + self.values.size
                trace["logical_copy_reads"] += elements
                trace["logical_copy_writes"] += elements
                trace["fresh_copy_lower_bound"] += 2 * elements
            self._idx += keys.shape[2]
            self.offset += keys.shape[2]
            if trace is not None:
                trace[
                    "append_metal_hits" if recipe.get("append_metal") else "append_concat_hits"
                ] += 1
            return self.keys_and_values()

        BatchKVCache.merge = classmethod(merge)
        BatchKVCache.prepare = prepare
        BatchKVCache.extend = extend
        BatchKVCache.filter = filtered
        BatchKVCache.extract = extracted
        BatchKVCache.update_and_fetch = updated
    if trace is not None:
        # Counts/sizes describe calls, not physical GPU memory traffic.
        for name in ["merge", "filter", "extract", "update_and_fetch"]:
            original = getattr(BatchKVCache, name)
            if name == "merge":

                def merge(cls, caches, original=original):
                    trace["merge_calls"] += 1
                    trace["merge_rows"] += len(caches)
                    return original(caches)

                BatchKVCache.merge = classmethod(merge)
            else:

                def wrap(self, *args, original=original, name=name, **kwargs):
                    trace[name + "_calls"] += 1
                    if name == "update_and_fetch":
                        trace["appended_elements"] += args[0].size + args[1].size
                        if self.keys is not None:
                            trace["active_elements_before_append"] += (
                                2
                                * self.keys.shape[0]
                                * self.keys.shape[1]
                                * self._idx
                                * self.keys.shape[3]
                            )
                    return original(self, *args, **kwargs)

                setattr(BatchKVCache, name, wrap)
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(BatchKVCache, name, value)


def run(model, prompts, counts, recipe, capture=False, trace=False):
    import time
    import mlx.core as mx
    from mlx_lm.generate import generation_stream
    from lossless.adapters.mlx.execution import scheduled

    mx.synchronize(generation_stream)
    mx.synchronize()
    start = time.perf_counter()
    stats = Counter() if trace else None
    with cache_policy(recipe, stats):
        row, caches, probs = scheduled(model, prompts, counts, capture, enabled=True)
    mx.synchronize(generation_stream)
    mx.synchronize()
    row["seconds"] = time.perf_counter() - start
    row["tokens_per_second"] = sum(counts) / row["seconds"]
    row["cache_trace"] = dict(stats) if trace else None
    return row, caches, probs
