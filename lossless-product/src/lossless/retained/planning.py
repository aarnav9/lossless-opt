"""Finite additive serial-cost planner; hardware timing still decides acceptance."""


def validate_catalog(catalog, stages):
    if type(stages) is not int or stages < 1 or not catalog:
        raise ValueError("nonempty catalog and positive stage count required")
    seen = set()
    for row in catalog:
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in seen:
            raise ValueError("unique nonempty segment IDs required")
        seen.add(row["id"])
        if not all(type(row[k]) is int for k in ("start", "stop", "cost_ns")):
            raise ValueError("segment bounds and measured costs must be integers")
        if not 0 <= row["start"] < row["stop"] <= stages or row["cost_ns"] <= 0:
            raise ValueError("invalid segment bounds or nonpositive cost")


def validate_plan(plan, catalog, stages):
    validate_catalog(catalog, stages)
    if not isinstance(plan, list) or not plan or any(not isinstance(x, str) for x in plan):
        raise ValueError("plan must be a nonempty list of segment IDs")
    by_id = {r["id"]: r for r in catalog}
    end = 0
    for name in plan:
        if name not in by_id or by_id[name]["start"] != end:
            raise ValueError("plan must cover each stage exactly once in order")
        end = by_id[name]["stop"]
    if end != stages:
        raise ValueError("plan does not finish the pipeline")
    return sum(by_id[name]["cost_ns"] for name in plan)


def dynamic_program(catalog, stages):
    validate_catalog(catalog, stages)
    distance, paths = [0] + [None] * stages, [[]] + [None] * stages
    for end in range(1, stages + 1):
        options = [
            (distance[r["start"]] + r["cost_ns"], paths[r["start"]] + [r["id"]])
            for r in catalog
            if r["stop"] == end and distance[r["start"]] is not None
        ]
        if options:
            distance[end], paths[end] = min(options)
    if distance[-1] is None:
        raise ValueError("catalog cannot cover pipeline")
    return {"plan": paths[-1], "cost_ns": distance[-1], "potentials_ns": distance}
