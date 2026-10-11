"""Experimental search-cost filter from 037, never a correctness/acceptance gate."""

import math


class BoundedRace:
    """Reject clearly slower arms using paired, randomized timing observations.

    The Hoeffding bound assumes independent bounded scores with stable means;
    device drift can violate these assumptions. Use only during discovery, keep
    the reference available, and independently confirm the frozen survivor.
    Alpha is spent across the declared arms and every sample count. This API
    does not execute candidates, select a winner or bypass correctness checks.
    """

    def __init__(self, arms, *, alpha=0.05):
        if not isinstance(arms, (list, tuple)) or not 1 <= len(arms) <= 256:
            raise ValueError("declare 1..256 arm names before measurement")
        if any(not isinstance(a, str) or not a for a in arms) or len(set(arms)) != len(arms):
            raise ValueError("unique nonempty arm names required")
        if not math.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        self.alpha = alpha
        self._scores = {a: (0, 0.0) for a in arms}
        self.rejected = {}

    def observe(self, arm, reference_seconds, candidate_seconds):
        if arm not in self._scores or arm in self.rejected:
            raise ValueError("unknown or already rejected arm")
        if any(not math.isfinite(t) or t <= 0 for t in (reference_seconds, candidate_seconds)):
            raise ValueError("positive finite paired durations required")
        scale = max(reference_seconds, candidate_seconds)
        a, b = reference_seconds / scale, candidate_seconds / scale
        count, total = self._scores[arm]
        count += 1
        total += (a - b) / (a + b)
        self._scores[arm] = count, total
        radius = math.sqrt(
            2 * math.log(2 * len(self._scores) * count * (count + 1) / self.alpha) / count
        )
        upper = total / count + radius
        reject = upper < 0
        if reject:
            self.rejected[arm] = count
        return dict(samples=count, upper_score=upper, reject=reject)
