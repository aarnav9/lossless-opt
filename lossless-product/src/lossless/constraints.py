"""Frozen MLX deployment limits, applied independently of correctness and speed gates."""

import math
import statistics

from .economics import payback

LIMITS = {
    "max_p95_ttft_seconds": ("worst_request_p95_ttft_seconds", "max"),
    "max_p95_completion_seconds": ("worst_request_p95_completion_seconds", "max"),
    "max_peak_mlx_bytes": ("peak_mlx_allocated_bytes", "max"),
    "max_break_even_calls": ("break_even_calls", "max"),
    "min_tokens_per_second": ("tokens_per_second", "min"),
}


def validate(constraints, adapter):
    from .jobs import fields, positive

    fields(constraints, LIMITS, "objective.constraints")
    if constraints and adapter != "mlx.fixed_count":
        raise ValueError("deployment constraints currently require mlx.fixed_count")
    for name, value in constraints.items():
        positive(
            value,
            f"objective.constraints.{name}",
            name in {"max_peak_mlx_bytes", "max_break_even_calls"},
        )


def percentile95(values):
    values = sorted(values)
    position = (len(values) - 1) * 0.95
    lo = int(position)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (position - lo)


def measurements(rows, peaks):
    if not rows:
        return {}
    return {
        "repeats": len(rows),
        "tokens_per_second": statistics.median(r["tokens_per_second"] for r in rows),
        "worst_request_p95_ttft_seconds": max(
            percentile95([r["ttft_seconds"][i] for r in rows])
            for i in range(len(rows[0]["ttft_seconds"]))
        ),
        "worst_request_p95_completion_seconds": max(
            percentile95([r["completion_seconds"][i] for r in rows])
            for i in range(len(rows[0]["completion_seconds"]))
        ),
        "peak_mlx_allocated_bytes": max(peaks),
        "scope": "Warm complete calls; p95 estimated separately for each request position, then worst position selected. Descriptive finite-sample limits, not a production SLO guarantee. MLX allocation includes live model/cache/output arrays but excludes allocator cache and untracked driver/process memory.",
    }


def assess(
    limits, metrics, *, search_seconds, reference_seconds, candidate_seconds, extra_setup_seconds
):
    economics = payback(search_seconds, reference_seconds, candidate_seconds, extra_setup_seconds)
    values = {**metrics, "break_even_calls": economics["break_even_calls"]}
    checks = {}
    for name, limit in limits.items():
        metric, direction = LIMITS[name]
        value = values.get(metric)
        measured = type(value) in (int, float) and math.isfinite(value)
        if name.startswith("max_p95_") and metrics.get("repeats", 0) < 20:
            measured = False
        passed = measured and (value <= limit if direction == "max" else value >= limit)
        checks[name] = {
            "metric": metric,
            "limit": limit,
            "observed": value,
            "passed": passed,
            "reason": "within limit"
            if passed
            else "limit exceeded"
            if measured
            else "insufficient or unavailable measurement",
        }
    return {
        "passed": all(c["passed"] for c in checks.values()),
        "checks": checks,
        "payback": economics,
    }
