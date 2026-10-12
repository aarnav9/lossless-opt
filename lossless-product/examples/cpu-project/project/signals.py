"""Summaries for many short signal windows; public NumPy is the reference."""

import numpy as np


def summarize(windows):
    totals, means = [], []
    for window in windows:
        totals.append(np.sum(window))
        means.append(np.mean(window))
    return {"totals": np.asarray(totals), "means": np.asarray(means)}
