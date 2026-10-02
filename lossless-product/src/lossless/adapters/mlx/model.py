"""Exact recipe model wrapper extracted from the validated research path."""

import mlx.core as mx
import mlx.nn as nn


class GroupedHead(nn.Module):
    """Llama forward with smaller decode-only output-projection groups."""

    def __init__(self, original, width):
        super().__init__()
        if type(width) is not int or width < 1:
            raise ValueError("positive integer head group size required")
        if original.model_type != "llama":
            raise ValueError("grouped head currently supports Llama only")
        self.inner, self.width = original, width
        self.args, self.model_type = original.args, original.model_type

    @property
    def layers(self):
        return self.inner.layers

    def make_cache(self):
        return self.inner.make_cache()

    def __call__(self, inputs, cache=None, input_embeddings=None):
        # Mirrors mlx_lm.models.llama.Model; its MIT notice is bundled.
        h = self.inner.model(inputs, cache, input_embeddings)
        project = (
            self.inner.model.embed_tokens.as_linear
            if self.args.tie_word_embeddings
            else self.inner.lm_head
        )
        if h.shape[1] == 1 and h.shape[0] > self.width:
            return mx.concatenate(
                [project(h[i : i + self.width]) for i in range(0, h.shape[0], self.width)], axis=0
            )
        return project(h)
