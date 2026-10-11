"""Hardware-aware investigation context and auditable coverage, not acceptance.

No network or microbenchmarks. Unavailable PMU/compiler evidence stays unknown.
Only trusted evaluator results can mark an experiment evaluated; model reviews
cannot assert proof, measured bottlenecks, acceptance or exhaustive coverage.
"""

import copy
import html
from collections import Counter
from pathlib import Path

from .._native import hardware
from ..jobs import fields, identity
from . import cpu_catalog
from .runtime import clock

UNKNOWN_MEASUREMENTS = (
    "cache_misses",
    "memory_traffic_bytes",
    "sustained_bandwidth",
    "translation_misses",
    "branch_mispredictions",
    "frontend_stalls",
    "execution_resource_pressure",
    "register_spills",
    "thermal_throttling",
    "core_placement",
)
PROOF_TEMPLATES = {
    "fusion": ("Fusion.lean", ["remove_intermediate", "preserve_rounding_boundary"]),
    "arithmetic": ("Fusion.lean", ["retain_mul_add", "fma_requires_equivalence"]),
    "memory": ("Movement.lean", ["fresh_copy_lower_bound"]),
    "layout": ("Movement.lean", ["vector_partition"]),
}


def catalog():
    cards = copy.deepcopy(cpu_catalog.CARDS)
    proofs = Path(__file__).resolve().parents[1] / "resources/proofs"
    for card in cards:
        card["proof_support"] = {"status": "obligations_only; no implementation proof checked"}
        if card["id"] in PROOF_TEMPLATES:
            name, statements = PROOF_TEMPLATES[card["id"]]
            card["proof_support"].update(
                file=name,
                statements=statements,
                source_identity=identity((proofs / name).read_text()),
                boundary="Abstract template only. Check its assumptions, machine ranges, memory/rounding semantics and correspondence to emitted code; compiler/runtime remain separate boundaries.",
            )
    content = {
        "schema_version": 1,
        "catalog_version": cpu_catalog.VERSION,
        "sources": copy.deepcopy(cpu_catalog.SOURCES),
        "cards": cards,
    }
    return {**content, "identity": identity(content)}


def collect(python, deadline):
    started = clock()
    report = hardware.collect_bounded(
        timeout=max(0, min(5, deadline - started)), cpu_only=True, python=python
    )
    return {
        "hardware": report,
        "collection_seconds": clock() - started,
        "scope": "Bounded CPU-only OS facts and selected-interpreter package inventory; no accelerator dispatch, network, PMU collection or benchmark. Installed libraries are not proof they were used.",
        "unavailable_measurements": {key: None for key in UNKNOWN_MEASUREMENTS},
    }


def applicability(card, hardware_report):
    host = hardware_report.get("host", {})
    machine = host.get("machine", "").lower()
    known = machine in {"x86_64", "amd64", "i386", "i686", "arm64", "aarch64"}
    if card["gate"] in {"x86", "arm"}:
        matches = machine in (
            {"x86_64", "amd64", "i386", "i686"} if card["gate"] == "x86" else {"arm64", "aarch64"}
        )
        return (
            ("applicable", "Detected instruction-set family; exact CPU model still required")
            if matches
            else (
                ("not_applicable", "Detected instruction-set family differs")
                if known
                else ("unknown", "Instruction-set family not established")
            )
        )
    if card["gate"] == "numa":
        nodes = host.get("cpu_details", {}).get("NUMA node(s)")
        try:
            nodes = int(nodes)
        except (TypeError, ValueError):
            return "unknown", "NUMA topology not reported; do not infer from CPU count"
        if nodes == 1:
            return "not_applicable", "One NUMA node reported; cross-node placement does not apply"
        if nodes > 1:
            return "applicable", "Multiple NUMA nodes reported; remote traffic still unmeasured"
        return "unknown", "NUMA node count unavailable"
    return (
        "applicable",
        "General investigation category; benefit and editable scope remain unestablished",
    )


def initial_coverage(cat, hardware_report):
    entries = []
    for card in cat["cards"]:
        state, reason = applicability(card, hardware_report)
        entries.append(
            {
                "id": card["id"],
                "area": card["area"],
                "applicability": state,
                "screen_reason": reason,
                "status": "not_applicable" if state == "not_applicable" else "unknown",
                "author_reviews": [],
                "experiments": [],
                "proof_status": "not_checked",
            }
        )
    return {
        "catalog_identity": cat["identity"],
        "entries": entries,
        "scope": "Coverage of this finite catalog, not all optimizations. Model reviews are advisory; experiment outcomes apply to the whole candidate, not a causal attribution to each tagged idea. No universal equivalence or hardware optimality proof.",
    }


def augment(context, summary):
    """Only discovery-derived profiles enter this context; never inspect case fixtures."""
    if "architecture" not in summary:
        return context
    architecture = summary["architecture"]
    context["evidence"]["hardware:cpu"] = {
        "kind": "os_reported_hardware",
        **architecture["hardware"],
    }
    context["evidence"]["hardware:unknown"] = {
        "kind": "unavailable_measurements",
        **architecture["unavailable_measurements"],
    }
    for case in sorted(
        summary["profiles"], key=lambda c: c["timing"]["median_sequence_seconds"], reverse=True
    )[:8]:
        if "diagnostics" in case:
            context["evidence"][case["id"] + ":diagnostics"] = {
                "kind": "separate_instrumented_diagnostics",
                **case["diagnostics"],
            }
    context["cpu_investigation"] = {
        "catalog": summary["cpu_catalog"],
        "coverage": copy.deepcopy(summary["checklist_coverage"]),
        "instructions": "Traverse the catalog cheaply, then prioritize measured application costs. Sources are references, not local measurements. Unknown counters stay unknown. CPU family is not a microarchitecture model. Cite supplied evidence IDs; do not invent hardware costs or conclude an idea was tested from source review. Review interactions after each change. Proof templates specify narrow abstract statements, not verified emitted code. Stay within the editable scope and frozen contract.",
    }
    return context


def notes_schema(ids, evidence_ids):
    return {
        "type": "array",
        "maxItems": len(ids),
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "assessment", "reason", "evidence_ids", "experiment"],
            "properties": {
                "id": {"type": "string", "enum": list(ids)},
                "assessment": {
                    "type": "string",
                    "enum": ["hypothesis", "ruled_out", "unknown", "deferred"],
                },
                "reason": {"type": "string"},
                "evidence_ids": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {"type": "string", "enum": list(evidence_ids)},
                },
                "experiment": {"type": "boolean"},
            },
        },
    }


def validate_notes(notes, *, ids=None, evidence_ids=None, required=()):
    if not isinstance(notes, list) or len(notes) > len(cpu_catalog.CARDS):
        raise ValueError("checklist must be a bounded list")
    ids = set(ids if ids is not None else [c["id"] for c in cpu_catalog.CARDS])
    seen = set()
    for note in notes:
        fields(
            note,
            {"id", "assessment", "reason", "evidence_ids", "experiment"},
            "checklist review",
            {"id", "assessment", "reason", "evidence_ids", "experiment"},
        )
        if not isinstance(note["id"], str) or note["id"] not in ids or note["id"] in seen:
            raise ValueError("unknown or duplicate checklist ID")
        seen.add(note["id"])
        if note["assessment"] not in {"hypothesis", "ruled_out", "unknown", "deferred"}:
            raise ValueError("checklist review cannot assert proof, measurement or acceptance")
        if not isinstance(note["reason"], str) or not 1 <= len(note["reason"]) <= 1200:
            raise ValueError("checklist review needs a bounded reason")
        refs = note["evidence_ids"]
        if not isinstance(refs, list) or len(refs) > 8 or not all(isinstance(r, str) for r in refs):
            raise ValueError("invalid checklist evidence references")
        if evidence_ids is not None and not set(refs) <= set(evidence_ids):
            raise ValueError("unknown checklist evidence reference")
        if note["assessment"] in {"hypothesis", "ruled_out"} and not refs:
            raise ValueError("checklist hypotheses and exclusions require evidence references")
        if type(note["experiment"]) is not bool or (
            note["experiment"] and note["assessment"] != "hypothesis"
        ):
            raise ValueError("only a hypothesis may tag a candidate experiment")
    if not set(required) <= seen:
        raise ValueError("first assisted response must account for every catalog entry")


def record_review(coverage, notes, candidate_id):
    for note in notes:
        entry = next(e for e in coverage["entries"] if e["id"] == note["id"])
        entry["author_reviews"].append({"candidate": candidate_id, **note})
        if entry["status"] != "not_applicable" and not entry["experiments"]:
            entry["status"] = "reviewed_unverified"


def record_result(coverage, notes, record):
    status = record["status"]
    evaluated = status in {"matched", "mismatch", "reference_unstable", "candidate_unstable"}
    for note in notes:
        if not note["experiment"]:
            continue
        entry = next(e for e in coverage["entries"] if e["id"] == note["id"])
        entry["experiments"].append(
            {
                "candidate": record["id"],
                "status": status,
                "evaluated": evaluated,
                "discovery": record.get("discovery"),
                "reason": record.get("reason"),
            }
        )
        if entry["applicability"] != "not_applicable":
            entry["status"] = (
                "evaluated"
                if any(e["evaluated"] for e in entry["experiments"])
                else "attempted_not_evaluated"
            )


def finish(coverage, reason):
    coverage["stop_reason"] = reason
    for entry in coverage["entries"]:
        entry["closure"] = (
            "not_applicable"
            if entry["applicability"] == "not_applicable"
            else "has_evaluated_candidate"
            if any(e["evaluated"] for e in entry["experiments"])
            else "deferred_at_stop"
        )
    coverage["counts"] = dict(Counter(e["status"] for e in coverage["entries"]))
    coverage["unreviewed_ids"] = [
        e["id"]
        for e in coverage["entries"]
        if not e["author_reviews"] and e["applicability"] != "not_applicable"
    ]
    coverage["untested_ids"] = [
        e["id"] for e in coverage["entries"] if e["closure"] == "deferred_at_stop"
    ]


def context_coverage(coverage, best_id=None):
    """Bound history per card for the author; retain complete evidence on disk."""

    def retained(rows):
        indices = set(range(max(0, len(rows) - 2), len(rows)))
        if best_id is not None:
            indices.update(i for i, row in enumerate(rows) if row["candidate"] == best_id)
        return [rows[i] for i in sorted(indices)]

    entries = []
    for original in coverage["entries"]:
        entry = {k: v for k, v in original.items() if k not in {"author_reviews", "experiments"}}
        reviews = retained(original["author_reviews"])
        experiments = retained(original["experiments"])
        entry["author_reviews"] = copy.deepcopy(reviews)
        entry["experiments"] = []
        for row in experiments:
            stats = (row.get("discovery") or {}).get("statistics", {})
            entry["experiments"].append(
                {
                    "candidate": row["candidate"],
                    "status": row["status"],
                    "evaluated": row["evaluated"],
                    "reason": str(row["reason"])[:800] if row.get("reason") is not None else None,
                    "case_balanced_speedup": stats.get("case_balanced_geomean_speedup"),
                    "minimum_case_speedup": min(
                        (c["speedup"] for c in stats.get("cases", [])), default=None
                    ),
                }
            )
        entry["total_reviews"] = len(original["author_reviews"])
        entry["total_attempts"] = len(original["experiments"])
        entry["omitted_reviews"] = entry["total_reviews"] - len(reviews)
        entry["omitted_attempts"] = entry["total_attempts"] - len(experiments)
        entries.append(entry)
    return {
        "catalog_identity": coverage["catalog_identity"],
        "scope": coverage["scope"],
        "entries": entries,
        "history_scope": "Latest two reviews/attempts per card plus the current best eligible candidate. Compact measured summaries only; full history and timing samples remain in local reports. Recent and winning source bodies are provided separately in previous_attempts.",
    }


def render(coverage):
    escape = lambda value: html.escape(str(value), quote=True)
    return (
        "<h2>CPU investigation coverage</h2><p>" + escape(coverage["scope"]) + "</p>"
        "<table><tr><th>Area</th><th>Applicability</th><th>Evidence status</th><th>Reviews / attempts</th></tr>"
        + "".join(
            "<tr>"
            + "".join(
                "<td>" + escape(v) + "</td>"
                for v in (
                    e["area"],
                    e["applicability"],
                    e["status"],
                    f"{len(e['author_reviews'])} / {len(e['experiments'])}",
                )
            )
            + "</tr>"
            for e in coverage["entries"]
        )
        + "</table><p>Unknown PMU measurements are not inferred from call stacks or array sizes. Proof obligations have not been checked for the candidate.</p>"
    )
