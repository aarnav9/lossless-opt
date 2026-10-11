"""Validate completion and export compact, portable evidence for all 100 trials."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import statistics

from common import DEFAULT_RUN, ROOT, read, sha, stamp, write


def summarize(run):
    ledger = read(run / "ledger.json")
    expected = {(family, number) for family in range(1, 6) for number in range(1, 21)}
    identities = [(row["family"], row["number"]) for row in ledger]
    if len(identities) != 100 or set(identities) != expected:
        raise ValueError("require exactly twenty distinct completed trials in each family")
    trials = []
    for row in ledger:
        folder = run / f"f{row['family']}_{row['number']:02d}"
        report = read(folder / "report.json")
        entry = {key: row[key] for key in ("family", "number", "decision", "speedup")}
        entry["artifact"] = folder.name
        entry["report_sha256"] = sha(folder / "report.json")
        if row["family"] != 5:
            if (
                report["decision"] != row["decision"]
                or report["timing"]["speedup"] != row["speedup"]
            ):
                raise ValueError(f"ledger/report mismatch: {folder.name}")
            entry.update(proposal=row["proposal"], result=report)
            entry["source"] = read(folder / "source.json")
        else:
            if report != row:
                raise ValueError(f"ledger/report mismatch: {folder.name}")
            entry.update(
                target=row["target"],
                plan=read(folder / "plan.json"),
                seed_root_speedup=row["seed_root_speedup"],
                winning_arm=row["winning_arm"],
            )
            arms = {}
            for name, arm in row["arms"].items():
                compact = {
                    key: value for key, value in arm.items() if key not in {"proposal", "provider"}
                }
                compact["hypothesis"] = arm.get("proposal", {}).get("hypothesis")
                compact["source_code"] = arm.get("proposal", {}).get("source")
                compact["provider"] = {
                    key: value
                    for key, value in arm["provider"].items()
                    if key not in {"response", "stdout", "stderr", "prompt"}
                }
                if (folder / name / "report.json").exists():
                    compact["result"] = read(folder / name / "report.json")
                arms[name] = compact
            entry["arms"] = arms
        trials.append(entry)
    finals = {}
    for family in range(1, 6):
        folder = run / f"final_f{family}"
        report = read(folder / "report.json")
        if report.get("status") not in {"qualified_exact", "reference_retained"}:
            raise ValueError(f"family {family} final incomplete")
        frozen = read(folder / "frozen.json")
        if frozen["selected"] is not None:
            if len(report["timing"]["rounds"]) != frozen["policy"]["final_pairs"]:
                raise ValueError(f"family {family} final timing incomplete")
            expected_cases = {case["id"] for case in read(folder / "cases.json")}
            if {case["id"] for case in report["memory"]} != expected_cases:
                raise ValueError(f"family {family} final memory incomplete")
            accepted = (
                report["decision"]["eligible"]
                and report["startup_eligible"]
                and all(case["eligible"] for case in report["memory"])
            )
            if accepted != report["accepted"] or accepted != (
                report["status"] == "qualified_exact"
            ):
                raise ValueError(f"family {family} final decision inconsistent")
        finals[str(family)] = dict(frozen=frozen, result=report)
    author_rows = [row for row in trials if row["family"] == 5]
    paired = []
    for row in author_rows:
        a, b = row["arms"]["control"], row["arms"]["diagnostics"]
        both_exact = all(arm.get("decision", {}).get("category") == "exact" for arm in [a, b])
        paired.append(
            dict(
                number=row["number"],
                target=row["target"],
                both_exact=both_exact,
                diagnostics_over_control=b["speedup"] / a["speedup"] if both_exact else None,
                identical_sources=a.get("source_sha256") is not None
                and a.get("source_sha256") == b.get("source_sha256"),
            )
        )
    conformance = {
        str(path.parent.relative_to(run)): read(path)
        for path in sorted(run.glob("conformance_*/report.json"))
    }
    author_summary = {
        "pairs": len(author_rows),
        "arms": {
            name: {
                "statuses": dict(
                    Counter(
                        row["arms"][name].get("status", "provider_failed") for row in author_rows
                    )
                ),
                "categories": dict(
                    Counter(
                        row["arms"][name].get("decision", {}).get("category", "not_evaluated")
                        for row in author_rows
                    )
                ),
                "advances": sum(row["winning_arm"] == name for row in author_rows),
                "discovery_eligible": sum(
                    row["arms"][name].get("decision", {}).get("eligible", False)
                    for row in author_rows
                ),
                "unchanged_sources": sum(
                    row["arms"][name].get("unchanged_source", False) for row in author_rows
                ),
                "provider_seconds": sum(
                    row["arms"][name]["provider"]["elapsed_seconds"] for row in author_rows
                ),
                "reported_tokens": {
                    key: sum(
                        usage.get(key, 0)
                        for row in author_rows
                        for usage in row["arms"][name]["provider"].get("usage", [])
                    )
                    for key in (
                        "input_tokens",
                        "cached_input_tokens",
                        "output_tokens",
                        "reasoning_output_tokens",
                    )
                },
            }
            for name in ("control", "diagnostics")
        },
        "both_exact_pairs": sum(p["both_exact"] for p in paired),
        "diagnostics_over_control_geomean": statistics.geometric_mean(
            p["diagnostics_over_control"] for p in paired if p["both_exact"]
        )
        if any(p["both_exact"] for p in paired)
        else None,
        "by_target": {
            target: {
                "pairs": sum(p["target"] == target for p in paired),
                "diagnostics_over_control_geomean": statistics.geometric_mean(
                    p["diagnostics_over_control"]
                    for p in paired
                    if p["both_exact"] and p["target"] == target
                )
                if any(p["both_exact"] and p["target"] == target for p in paired)
                else None,
            }
            for target in sorted({p["target"] for p in paired})
        },
        "interpretation": "Descriptive ratios from separate shared-incumbent measurements in adaptive trials; not independent replications or a general causal-effect estimate.",
    }
    evidence = dict(
        schema_version=1,
        completed=100,
        counts=dict(Counter(str(r["family"]) for r in ledger)),
        protocol=read(run / "protocol.json"),
        amendments={path.name: read(path) for path in sorted(run.glob("*amendment*.json"))},
        environment=read(run / "environment.json"),
        trials=trials,
        finals=finals,
        conformance=conformance,
        author_pairs=paired,
        author_summary=author_summary,
        author_audit=read(run / "author-audit.json"),
        author_attribution_review=read(run / "author-attribution-review.json")
        if (run / "author-attribution-review.json").exists()
        else None,
        limitations=[
            "One Apple M2 and one pinned software stack; finite evidence, not an equivalence proof.",
            "Exact native-wrapper comparisons use unchanged kernels; numerical oracle contracts remain unchanged.",
            "Author pairs are adaptive exploration, not independent replications of a general context effect.",
            "No task-quality labels: numerical agreement is not a task-quality result.",
            "Pool memory conservatively sums process peaks and double-counts shared pages.",
            "Final per-call gates were retained despite noise observed in identical-source controls.",
        ],
    )

    # Normalize artifact locations while preserving exact authored source bytes.
    def portable(value, key=None):
        if isinstance(value, dict):
            return {k: portable(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [portable(v) for v in value]
        if isinstance(value, str) and key != "source_code":
            return value.replace(str(ROOT) + "/", "<repo>/")
        return value

    return portable(evidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--evidence", type=Path, default=ROOT / "lossless-product/docs/evidence/cpu-100.json.gz"
    )
    parser.add_argument(
        "--appendix", type=Path, default=ROOT / "lossless-product/docs/cpu-100-experiments.md"
    )
    args = parser.parse_args()
    evidence = summarize(args.output.resolve())
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_bytes(
        gzip.compress((json.dumps(evidence, indent=2, allow_nan=False) + "\n").encode(), mtime=0)
    )
    lines = [
        "# All 100 sequential CPU experiments",
        "",
        "Generated by `experiments/cpu_100/summarize.py`. See [interpretation and final results](cpu-100-study.md).",
        "",
        "These are discovery observations, not accepted deployment speedups. Ratios above 1 are faster. Families 1–4 compare with their frozen baseline; each family-5 arm compares with its shared adaptive incumbent on one target workload. Exact means observed equality, not a universal proof. Family 1's unguarded early helpers later failed the strided-input control; trial 14 introduced the guard.",
        "",
    ]
    titles = {
        1: "Compiled application regions",
        2: "Independent spectra",
        3: "Ordinary native calls",
        4: "Prepared metadata",
        5: "Compiler feedback",
    }
    for family, title in titles.items():
        lines.extend([f"## {family}. {title}", ""])
        if family < 5:
            lines.extend(
                [
                    "| Trial | Hypothesis | Speedup | Correctness | Discovery eligible |",
                    "| --- | --- | ---: | --- | --- |",
                ]
            )
        else:
            lines.extend(
                [
                    "| Trial | Target | Control | Diagnostics | Advancing arm |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
        for row in evidence["trials"]:
            if row["family"] != family:
                continue
            if family < 5:
                hypothesis = row["proposal"]["hypothesis"].replace("|", "/").replace("\n", " ")
                lines.append(
                    f"| {row['number']} | {hypothesis} | {row['speedup']:.4f}× | {row['decision']['category']} | {row['decision']['eligible']} |"
                )
            else:

                def arm_cell(arm):
                    category = arm.get("decision", {}).get(
                        "category", arm.get("status", "provider_failed")
                    )
                    speed = f"{arm['speedup']:.4f}×" if "speedup" in arm else "—"
                    return (
                        speed
                        + " "
                        + category
                        + (" (unchanged source)" if arm.get("unchanged_source") else "")
                    )

                lines.append(
                    f"| {row['number']} | {row['target']} | {arm_cell(row['arms']['control'])} | {arm_cell(row['arms']['diagnostics'])} | {row['winning_arm'] or 'none'} |"
                )
        lines.append("")
    args.appendix.write_text("\n".join(lines))
    write(
        args.output / "summary.json",
        dict(
            completed=100,
            counts=evidence["counts"],
            finals={
                k: dict(
                    status=v["result"]["status"],
                    speedup=v["result"].get("timing", {}).get("speedup"),
                )
                for k, v in evidence["finals"].items()
            },
            evidence=str(args.evidence.resolve()),
            evidence_sha256=sha(args.evidence),
        ),
    )
    stamp(f"COMPLETE 100/100 evidence={args.evidence.resolve()} summary={args.output}/summary.json")


if __name__ == "__main__":
    main()
