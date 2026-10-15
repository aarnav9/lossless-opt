"""Keep protocols, raw measurements, patches and licenses for all ten trials."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign = json.loads((args.runs / "campaign.json").read_text())
    root = Path(__file__).resolve().parents[2]
    for name, expected in campaign["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen campaign source changed: {name}")
    evidence = dict(
        schema_version=1,
        campaign=campaign,
        trials={},
        licenses={},
        frozen_sources={name: (root / name).read_text() for name in campaign["files"]},
    )
    summaries = []
    for kind in campaign["kinds"]:
        for runtime in ("baseline", "expanded"):
            name = kind + "-" + runtime
            trial = args.runs / name
            result = json.loads((trial / "result.json").read_text())
            if not result.get("changing_inputs_exact"):
                raise ValueError(f"public adapter did not match: {name}")
            records = {}
            # Source trees and binary fixtures are reproducible from the pinned
            # upstream, wrapper, loader and seed. Preserve their hashes and all
            # reports/paired measurements, not redundant copied source trees.
            for path in sorted(trial.rglob("*")):
                relative = path.relative_to(trial)
                if any(
                    part in {"project", "reference", "optimized", "application"}
                    for part in relative.parts
                ):
                    continue
                if path.is_file() and path.suffix in {".json", ".patch"}:
                    records[str(relative)] = (
                        json.loads(path.read_text()) if path.suffix == ".json" else path.read_text()
                    )
            evidence["trials"][name] = records
            for path in (trial / "project").glob("LICENSE*"):
                evidence["licenses"][kind + "/" + path.name] = path.read_text()
            workflow = json.loads((trial / "workflow/report.json").read_text())
            plan = trial / "workflow/plan.json"
            deployment = result.get("deployment") or {}
            comparison = deployment.get("comparisons", {}).get("same_mode_reference", {})
            summaries.append(
                dict(
                    name=name,
                    status=result["workflow_status"],
                    source=result["source_selected"],
                    mode=result.get("actual_mode"),
                    seconds=result["elapsed_seconds"],
                    candidate_count=len(json.loads(plan.read_text()).get("candidates", []))
                    if plan.exists()
                    else 0,
                    source_scope=json.loads(plan.read_text()).get("source_scope", [])
                    if plan.exists()
                    else [],
                    source_search_status=(workflow.get("search") or {}).get("status"),
                    warm_speedup=comparison.get("request_statistics", {}).get(
                        "case_balanced_geomean_speedup"
                    ),
                    amortized_speedup=comparison.get("statistics", {}).get(
                        "case_balanced_geomean_speedup"
                    ),
                    comparison_accepted=comparison.get("accepted"),
                    reason=deployment.get("reason", result.get("reason")),
                    changing_inputs_exact=result["changing_inputs_exact"],
                )
            )
    evidence["summary"] = summaries
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(json.dumps(evidence, sort_keys=True).encode(), mtime=0))
    (args.runs / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    print(f"Evidence: {args.output}; bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
