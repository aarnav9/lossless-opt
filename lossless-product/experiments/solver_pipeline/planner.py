"""Discrete serial plans with evaluator-owned, integer-nanosecond costs.

This bounded chain also has an exact dynamic-programming solution. It is an
independent control for CP-SAT, not an example where a general solver is needed.
"""

import math
import time


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


def solve(catalog, stages, *, exclude=(), timeout=1.0):
    from ortools.sat.python import cp_model

    validate_catalog(catalog, stages)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("positive finite solver timeout required")
    # Leave headroom for integer arithmetic inside the solver.
    if sum(r["cost_ns"] for r in catalog) >= 2**60:
        raise ValueError("costs exceed safe integer range")
    started = time.perf_counter()
    model = cp_model.CpModel()
    chosen = {r["id"]: model.new_bool_var(r["id"]) for r in catalog}
    for stage in range(stages):
        model.add_exactly_one(chosen[r["id"]] for r in catalog if r["start"] <= stage < r["stop"])
    for plan in exclude:
        validate_plan(plan, catalog, stages)
        model.add(sum(chosen[name] for name in plan) <= len(plan) - 1)
    model.minimize(sum(r["cost_ns"] * chosen[r["id"]] for r in catalog))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.solve(model)
    record = {
        "status": solver.status_name(status),
        "elapsed_seconds": time.perf_counter() - started,
        "best_bound_ns": solver.best_objective_bound,
        "excluded_plans": len(exclude),
        "scope": "optimum/feasibility of frozen serial additive cost model only",
    }
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        rows = sorted(
            (r for r in catalog if solver.boolean_value(chosen[r["id"]])),
            key=lambda r: r["start"],
        )
        record["plan"] = [r["id"] for r in rows]
        record["cost_ns"] = validate_plan(record["plan"], catalog, stages)
    return record


def response_schema(catalog):
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["hypothesis", "segments"],
        "properties": {
            "hypothesis": {"type": "string"},
            "segments": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "enum": [r["id"] for r in catalog]},
            },
        },
    }


def validate_response(response, catalog, stages):
    if not isinstance(response, dict) or set(response) != {"hypothesis", "segments"}:
        raise ValueError("only hypothesis and segment IDs are author-controlled")
    if not isinstance(response["hypothesis"], str) or not response["hypothesis"].strip():
        raise ValueError("nonempty hypothesis required")
    return validate_plan(response["segments"], catalog, stages)


def lean_witness(catalog, stages, plan):
    """A Bellman-potential certificate checked independently of OR-Tools."""
    cost = validate_plan(plan, catalog, stages)
    dp = dynamic_program(catalog, stages)
    if cost != dp["cost_ns"] or any(x is None for x in dp["potentials_ns"]):
        raise ValueError("witness requires an optimal plan and reachable prefixes")
    rows = {r["id"]: r for r in catalog}

    def edges(items):
        return (
            "["
            + ", ".join(
                "⟨" + ", ".join(str(r[k]) for k in ("start", "stop", "cost_ns")) + "⟩"
                for r in items
            )
            + "]"
        )

    return f"""import PlanCost
open PlanCost
def catalog : List Edge := {edges(catalog)}
def chosen : List Edge := {edges(rows[name] for name in plan)}
def potential (i : Nat) : Nat := ({dp["potentials_ns"]} : List Nat)[i]!
theorem chosen_valid : Valid chosen 0 {stages} := by decide
theorem chosen_available : ∀ e ∈ chosen, e ∈ catalog := by decide
theorem bounds : ∀ e ∈ catalog, potential e.stop ≤ potential e.start + e.cost := by decide
theorem attained : potential {stages} = potential 0 + cost chosen := by decide
theorem modeled_optimal (other : List Edge) (hv : Valid other 0 {stages})
    (ha : ∀ e ∈ other, e ∈ catalog) : cost chosen ≤ cost other := by
  exact optimal_of_attains potential chosen other 0 {stages} hv
    (fun e he => bounds e (ha e he)) attained
#print axioms modeled_optimal
"""
