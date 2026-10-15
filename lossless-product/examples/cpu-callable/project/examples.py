"""Use real representative values here; evaluation examples stay out of search."""

import numpy as np


def cases():
    rng = np.random.default_rng(20261014)
    return [
        dict(id="development", split="discovery", args=[rng.normal(size=(32, 64))]),
        dict(id="held_out", split="evaluation", args=[rng.normal(size=(64, 64))]),
    ]
