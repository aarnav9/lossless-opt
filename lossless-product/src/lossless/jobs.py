"""Strict, non-executing resolution of a versioned optimization job."""

from dataclasses import dataclass
import copy
import hashlib
import json
import math
from pathlib import Path

ADAPTERS = {
    "native.copy": "copy",
    "native.softmax": "softmax",
    "native.rmsnorm_residual": "rmsnorm_residual",
    "mlx.fixed_count": "mlx",
}


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 4_000_000:
        raise ValueError(f"JSON input exceeds 4 MB: {path.name}")
    return json.loads(
        path.read_text(),
        object_pairs_hook=object_pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite JSON value: {x}")),
    )


def fields(value, allowed, name, required=()):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    unknown = set(value) - set(allowed)
    missing = set(required) - set(value)
    if unknown or missing:
        raise ValueError(
            f"{name}: unknown fields {sorted(unknown)}, missing fields {sorted(missing)}"
        )


def positive(value, name, integer=False):
    if (
        type(value) not in ((int,) if integer else (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive {'integer' if integer else 'finite number'}")
    return value


def identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Budget:
    seconds: float = 60
    max_candidates: int = 4
    max_llm_calls: int = 1


@dataclass(frozen=True)
class Workload:
    config: dict
    base: Path

    @classmethod
    def from_config(cls, path):
        path = Path(path).resolve()
        return cls(read_json(path), path.parent)

    @classmethod
    def from_dict(cls, value, *, base="."):
        return cls(copy.deepcopy(value), Path(base).resolve())

    def resolve(self, budget=None):
        value = copy.deepcopy(self.config)
        fields(
            value,
            {
                "schema_version",
                "name",
                "workload",
                "contract",
                "objective",
                "llm",
                "budget",
                "target",
                "output",
                "proofs",
            },
            "job",
            {"schema_version", "workload", "contract", "budget"},
        )
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError(
                "supported schema_version is 1; use 'lossless init' for runnable examples"
            )
        work = value["workload"]
        fields(work, {"adapter", "cases", "parameters", "inputs"}, "workload", {"adapter"})
        if work["adapter"] not in ADAPTERS:
            raise ValueError(
                f"unsupported adapter: {work['adapter']}; available: {', '.join(ADAPTERS)}"
            )
        contract = value["contract"]
        if isinstance(contract, dict) and "file" in contract:
            fields(contract, {"file"}, "contract reference", {"file"})
            contract = read_json(self.base / contract["file"])
            value["contract"] = contract
        fields(
            contract,
            {"preset", "required_evidence", "observables", "weight_changes", "precision_changes"},
            "contract",
            {"preset"},
        )
        expected = "exact" if work["adapter"] in ("native.copy", "mlx.fixed_count") else "numerical"
        if contract["preset"] != expected:
            raise ValueError(
                f"{work['adapter']} implements only the {expected} contract; no silent relaxation"
            )
        if contract.get("required_evidence", "validated") != "validated":
            raise ValueError(
                "end-to-end proved contracts are not implemented; scoped Lean proofs are available separately"
            )
        contract["required_evidence"] = "validated"
        for flag in ("weight_changes", "precision_changes"):
            if contract.get(flag, False) is not False:
                raise ValueError(f"{flag} must be false")
            contract[flag] = False
        observers = (
            ["output", "input_immutability"]
            if expected == "numerical" or work["adapter"] == "native.copy"
            else ["token_ids", "emitted_log_probabilities", "active_kv_state", "request_order"]
        )
        if contract.get("observables", observers) != observers:
            raise ValueError(f"this adapter requires observables {observers}")
        contract["observables"] = observers
        limits = value["budget"]
        fields(
            limits,
            {"wall_time_seconds", "max_candidates", "max_llm_calls", "job_timeout_seconds"},
            "budget",
            {"wall_time_seconds"},
        )
        if budget:
            if not isinstance(budget, Budget):
                raise TypeError("budget must be lossless.Budget")
            limits.update(
                wall_time_seconds=budget.seconds,
                max_candidates=budget.max_candidates,
                max_llm_calls=budget.max_llm_calls,
            )
        positive(limits["wall_time_seconds"], "budget.wall_time_seconds")
        limits.setdefault("max_candidates", 4)
        limits.setdefault("max_llm_calls", 1)
        positive(limits["max_candidates"], "budget.max_candidates", True)
        if type(limits["max_llm_calls"]) is not int or limits["max_llm_calls"] < 0:
            raise ValueError("budget.max_llm_calls must be a nonnegative integer")
        if "job_timeout_seconds" in limits:
            positive(limits["job_timeout_seconds"], "budget.job_timeout_seconds")
        if limits["max_candidates"] > 100:
            raise ValueError("max_candidates exceeds alpha limit of 100")
        objective = value.setdefault(
            "objective",
            {
                "metric": "throughput"
                if expected == "exact" and work["adapter"] == "mlx.fixed_count"
                else "latency"
            },
        )
        fields(objective, {"metric", "min_speedup", "comparator"}, "objective", {"metric"})
        if work["adapter"] == "mlx.fixed_count":
            if objective.setdefault("comparator", "stock_serial") != "stock_serial":
                raise ValueError("MLX comparator must be stock_serial")
        else:
            from ._native.operators import COMPARATORS, validate_comparator

            operator = ADAPTERS[work["adapter"]]
            validate_comparator(
                operator, objective.setdefault("comparator", COMPARATORS[operator][0])
            )
        if objective["metric"] != (
            "throughput" if work["adapter"] == "mlx.fixed_count" else "latency"
        ):
            raise ValueError("unsupported metric for this adapter")
        objective.setdefault("min_speedup", 1.02)
        if positive(objective["min_speedup"], "objective.min_speedup") < 1:
            raise ValueError("min_speedup must be at least 1")
        target = value.setdefault("target", {"device": "auto"})
        fields(target, {"device", "profile_file"}, "target")
        device = target.setdefault("device", "auto")
        if device not in (
            {"auto", "metal"} if work["adapter"] == "mlx.fixed_count" else {"auto", "cpu"}
        ):
            raise ValueError("target device does not match adapter")
        if "profile_file" in target:
            target["supplied_profile"] = read_json(self.base / target.pop("profile_file"))
        llm = value.get("llm")
        if llm is not None:
            fields(
                llm,
                {"callback", "command", "provider", "model", "credential_env", "timeout_seconds"},
                "llm",
            )
            if ("callback" in llm) == ("command" in llm):
                raise ValueError("llm needs exactly one callback or command")
            if "callback" in llm and (
                not isinstance(llm["callback"], str) or ":" not in llm["callback"]
            ):
                raise ValueError("llm.callback must be module:function or file.py:function")
            if "command" in llm and (
                not isinstance(llm["command"], list)
                or not llm["command"]
                or not all(isinstance(x, str) and x for x in llm["command"])
            ):
                raise ValueError("llm.command must be an argv list")
            if "credential_env" in llm and (
                not isinstance(llm["credential_env"], list)
                or not all(isinstance(x, str) and x for x in llm["credential_env"])
            ):
                raise ValueError("credential_env must contain variable names, not secret values")
            positive(llm.get("timeout_seconds", 30), "llm.timeout_seconds")
        proof = value.setdefault("proofs", {"check": False})
        fields(proof, {"check"}, "proofs")
        if type(proof.setdefault("check", False)) is not bool:
            raise ValueError("proofs.check must be boolean")
        output = value.setdefault("output", {"directory": "./lossless-runs"})
        fields(output, {"directory"}, "output", {"directory"})
        output["directory"] = str((self.base / output["directory"]).resolve())
        if work["adapter"] == "mlx.fixed_count":
            from .adapters.mlx.job import resolve_workload

            resolve_workload(work, self.base)
        else:
            if "parameters" in work or "inputs" in work:
                raise ValueError(
                    "native adapters take generated cases; arbitrary callables/input files are not yet supported"
                )
            cases = work.get("cases")
            if not isinstance(cases, list) or not cases:
                raise ValueError(
                    "workload.cases must contain discovery and evaluation tensor cases"
                )
            ids = set()
            shapes = {"discovery": set(), "evaluation": set()}
            for case in cases:
                fields(
                    case,
                    {"id", "split", "rows", "columns", "layout"},
                    "case",
                    {"id", "split", "rows", "columns", "layout"},
                )
                import re

                if (
                    not isinstance(case["id"], str)
                    or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", case["id"])
                    or case["id"] in ids
                ):
                    raise ValueError("case IDs must be unique lowercase identifiers")
                ids.add(case["id"])
                if case["split"] not in shapes or case["layout"] not in {"c", "f", "slice2"}:
                    raise ValueError("case split/layout is unsupported")
                for axis in ("rows", "columns"):
                    positive(case[axis], f"case.{axis}", True)
                if case["rows"] * case["columns"] > 16_777_216:
                    raise ValueError("case exceeds alpha limit of 16 million elements")
                shapes[case["split"]].add((case["rows"], case["columns"]))
            if not all(shapes.values()) or shapes["discovery"] & shapes["evaluation"]:
                raise ValueError("discovery and evaluation require nonempty, disjoint shapes")
        return value


def template(adapter="native.copy", seconds=60):
    if adapter not in ADAPTERS or adapter == "mlx.fixed_count":
        raise ValueError("use the mlx-fixed-count template for the MLX adapter")
    return {
        "schema_version": 1,
        "name": adapter.replace(".", "-"),
        "workload": {
            "adapter": adapter,
            "cases": [
                {
                    "id": "discovery_c",
                    "split": "discovery",
                    "rows": 64,
                    "columns": 257,
                    "layout": "c",
                },
                {
                    "id": "discovery_f",
                    "split": "discovery",
                    "rows": 65,
                    "columns": 128,
                    "layout": "f",
                },
                {
                    "id": "evaluation_c",
                    "split": "evaluation",
                    "rows": 73,
                    "columns": 259,
                    "layout": "c",
                },
                {
                    "id": "evaluation_f",
                    "split": "evaluation",
                    "rows": 67,
                    "columns": 132,
                    "layout": "f",
                },
            ],
        },
        "contract": {
            "preset": "exact" if adapter == "native.copy" else "numerical",
            "required_evidence": "validated",
        },
        "objective": {
            "metric": "latency",
            "min_speedup": 1.02,
            "comparator": "numpy_copy" if adapter == "native.copy" else "numpy_buffered",
        },
        "budget": {"wall_time_seconds": seconds, "max_candidates": 3, "max_llm_calls": 1},
        "target": {"device": "auto"},
        "output": {"directory": "./lossless-runs"},
    }
