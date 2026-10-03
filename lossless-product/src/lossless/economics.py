"""Amortization estimates for a declared workload and measured interface."""

import math


def payback(search_seconds, reference_seconds, candidate_seconds, extra_setup_seconds=None):
    saved = reference_seconds - candidate_seconds
    return {
        "search_seconds": search_seconds,
        "reference_seconds_per_call": reference_seconds,
        "candidate_seconds_per_call": candidate_seconds,
        "seconds_saved_per_call": saved,
        "extra_setup_seconds": extra_setup_seconds,
        "search_only_break_even_calls": math.ceil(search_seconds / saved) if saved > 0 else None,
        "break_even_calls": math.ceil((search_seconds + max(0, extra_setup_seconds)) / saved)
        if saved > 0 and extra_setup_seconds is not None
        else None,
        "reason": "no positive steady-state saving"
        if saved <= 0
        else "deployment setup unmeasured; search-only count is a lower bound"
        if extra_setup_seconds is None
        else "estimate assumes the measured per-call saving persists",
    }
