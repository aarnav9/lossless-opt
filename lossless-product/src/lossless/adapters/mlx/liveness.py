"""Pure liveness model for fixed-count, fixed-cohort generation."""

from lossless.adapters.mlx.schedule import cohorts


def masks(counts, step, width=4):
    if (
        not counts
        or any(type(n) is not int or n < 1 for n in counts)
        or type(step) is not int
        or not 0 <= step <= max(counts)
        or type(width) is not int
        or width < 1
    ):
        raise ValueError("positive counts/width and an in-range step required")
    active = [n for n in counts if step <= n]
    return [
        (active[i : i + width], any(step < n for n in active[i : i + width]))
        for i in range(0, len(active), width)
    ]


def certificate(lengths, counts):
    groups = cohorts(lengths, counts, "short_first")
    steps = []
    for group in groups:
        if len(group) == 1:
            continue  # unchanged serial fallback
        ds = [counts[i] for i in group]
        for t in range(max(ds) + 1):
            steps.append(
                {
                    "request_ids": group,
                    "step": t,
                    "groups": [{"counts": ns, "live": keep} for ns, keep in masks(ds, t)],
                }
            )
    old = sum(len(s["groups"]) for s in steps)
    kept = sum(g["live"] for s in steps for g in s["groups"])
    return {
        "cohorts": groups,
        "steps": steps,
        "batch_body_calls": len(steps),
        "old_batch_head_calls": old,
        "kept_batch_head_calls": kept,
        "skipped_batch_head_calls": old - kept,
        "scope": "batched single-position calls only; prefix and singleton paths unchanged",
    }


def witness(cert):
    rows = [(s["step"], g["counts"], g["live"]) for s in cert["steps"] for g in s["groups"]]
    data = ",\n  ".join(f"({t}, {ns}, {str(live).lower()})" for t, ns, live in rows)
    return f"""import Eliminate
open DeadProjection
def entries : List (Nat × List Nat × Bool) := [
  {data}]
example : entries.all (fun e => decide (live e.2.1 e.1 = e.2.2)) = true := by decide
example : entries.all (fun e => e.2.1.all (fun n => decide (e.1 ≤ n))) = true := by decide
example : calls (entries.map (fun e => e.2.2)) = {cert["kept_batch_head_calls"]} := by decide
example : entries.length = {cert["old_batch_head_calls"]} := by decide
"""
