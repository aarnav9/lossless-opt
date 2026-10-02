"""Pure scheduling model. No MLX imports; independent of hardware timing."""

from collections import Counter


def cohorts(lengths, counts, order="arrival", width=8):
    if width != 8:
        raise ValueError("this runtime policy is validated only with width eight")
    if len(lengths) != len(counts) or not counts:
        raise ValueError("nonempty matching lengths/counts required")
    if any(type(n) is not int or n < 1 for n in [*lengths, *counts]):
        raise ValueError("positive integer lengths and counts required")
    if order not in {"arrival", "long_first", "short_first"}:
        raise ValueError("unknown cohort order")
    buckets = {}
    for i, length in enumerate(lengths):
        buckets.setdefault(length, []).append(i)
    result = []
    for ids in buckets.values():
        if order != "arrival":
            ids = sorted(ids, key=lambda i: (-counts[i], i))
        groups = [ids[j : j + width] for j in range(0, len(ids), width)]
        if order == "short_first":
            groups.reverse()
        result.extend(groups)
    return result


def certificate(lengths, counts, groups):
    if sorted(i for g in groups for i in g) != list(range(len(counts))):
        raise ValueError("groups must partition request IDs exactly once")
    if any(not g or len(g) > 8 or len({lengths[i] for i in g}) != 1 for g in groups):
        raise ValueError("invalid capacity or mixed prompt lengths")
    buckets = {}
    for i, n in enumerate(lengths):
        buckets.setdefault(n, []).append(i)
    by_bucket = []
    histogram = Counter()
    for length, ids in buckets.items():
        gs = [g for g in groups if lengths[g[0]] == length]
        horizon = max(counts[i] for i in ids)
        active = [sum(counts[i] >= t for i in ids) for t in range(horizon + 1)]
        body_lb = sum((a + 7) // 8 for a in active)
        head_lb = sum((a + 3) // 4 for a in active)
        body = head = 0
        for g in gs:
            for t in range(max(counts[i] for i in g) + 1):
                a = sum(counts[i] >= t for i in g)
                body += 1
                head += (a + 3) // 4
                histogram[f"{a}x1"] += 1
        by_bucket.append(
            {
                "length": length,
                "ids": ids,
                "groups": gs,
                "counts": [counts[i] for i in ids],
                "horizon": horizon,
                "body_lower_bound": body_lb,
                "body_calls": body,
                "head_lower_bound": head_lb,
                "head_calls": head,
                "body_gap": body - body_lb,
                "head_gap": head - head_lb,
            }
        )
    prefill = Counter()
    for length in lengths:
        for offset in range(0, length - 1, 2048):
            prefill[f"1x{min(2048, length - 1 - offset)}"] += 1
    total = histogram + prefill
    b = sum(x["body_calls"] for x in by_bucket)
    bl = sum(x["body_lower_bound"] for x in by_bucket)
    h = sum(x["head_calls"] for x in by_bucket)
    hl = sum(x["head_lower_bound"] for x in by_bucket)
    return {
        "buckets": by_bucket,
        "body_calls": b,
        "body_lower_bound": bl,
        "body_factor_above_lower_bound": b / bl,
        "head_calls": h,
        "head_lower_bound": hl,
        "head_factor_above_lower_bound": h / hl,
        "attains_both": b == bl and h == hl,
        "decode_shapes": dict(sorted(histogram.items())),
        "prefill_shapes": dict(sorted(prefill.items())),
        "expected_model_shapes": dict(sorted(total.items())),
        "scope": "single-position Python model and head invocations in the pinned pipeline; not GPU launches, bytes or seconds",
    }


def lean_witness(name, lengths, counts, groups):
    """Produce closed, kernel-reduced witnesses; generic optimality is in Bounds."""
    cert = certificate(lengths, counts, groups)
    lines = [
        "import Bounds",
        "open KernelBounds",
        "set_option maxRecDepth 100000",
        "set_option maxHeartbeats 0",
        f"namespace {name}",
    ]
    for j, bucket in enumerate(cert["buckets"]):
        local = {rid: k for k, rid in enumerate(bucket["ids"])}
        ids = [[local[i] for i in g] for g in bucket["groups"]]
        cs = bucket["counts"]
        # Check ID permutation as well as value multiset; duplicates cannot hide
        # a request being duplicated while another equal-length request is lost.
        lines += [
            f"def counts{j} : List Nat := {cs}",
            f"def ids{j} : List (List Nat) := {ids}",
            f"def groups{j} := ids{j}.map (List.map (fun i => counts{j}[i]!))",
            f"theorem ids_partition{j} : ids{j}.flatten.Perm (List.range {len(cs)}) := by decide",
            f"theorem partition{j} : groups{j}.flatten.Perm counts{j} := by decide",
            f"theorem capacity{j} : ∀ g ∈ groups{j}, g.length ≤ 8 := by decide",
            f"theorem recorded_cost{j} : callCost groups{j} {bucket['horizon']} = {bucket['body_calls']} := by decide",
            f"theorem recorded_bound{j} : lowerBound counts{j} 8 {bucket['horizon']} = {bucket['body_lower_bound']} := by decide",
        ]
        if bucket["body_gap"] == 0:
            lines += [
                f"theorem attains{j} : callCost groups{j} {bucket['horizon']} = lowerBound counts{j} 8 {bucket['horizon']} := by decide",
                f"theorem optimal{j} (other : List (List Nat)) (hp : other.flatten.Perm counts{j}) (hc : ∀ g ∈ other, g.length ≤ 8) : callCost groups{j} {bucket['horizon']} ≤ callCost other {bucket['horizon']} := optimal_of_attains counts{j} groups{j} other 8 {bucket['horizon']} (by decide) hp hc attains{j}",
                f"#print axioms optimal{j}",
            ]
        # Head subgroups are formed independently within each width-eight cohort.
        head_ids = [g[k : k + 4] for g in ids for k in range(0, len(g), 4)]
        lines += [
            f"def headIds{j} : List (List Nat) := {head_ids}",
            f"def headGroups{j} := headIds{j}.map (List.map (fun i => counts{j}[i]!))",
            f"theorem head_partition{j} : headGroups{j}.flatten.Perm counts{j} := by decide",
            f"theorem head_capacity{j} : ∀ g ∈ headGroups{j}, g.length ≤ 4 := by decide",
        ]
        # Static head subgroup calculation agrees with dynamic compaction for
        # sorted survivors (arrival order may redistribute rows across subgroups).
        if bucket["body_gap"] == bucket["head_gap"] == 0:
            lines += [
                f"theorem head_attains{j} : callCost headGroups{j} {bucket['horizon']} = lowerBound counts{j} 4 {bucket['horizon']} := by decide",
                f"theorem head_optimal{j} (other : List (List Nat)) (hp : other.flatten.Perm counts{j}) (hc : ∀ g ∈ other, g.length ≤ 4) : callCost headGroups{j} {bucket['horizon']} ≤ callCost other {bucket['horizon']} := optimal_of_attains counts{j} headGroups{j} other 4 {bucket['horizon']} (by decide) hp hc head_attains{j}",
                f"#print axioms head_optimal{j}",
            ]
    return "\n".join(lines + [f"end {name}", ""])
