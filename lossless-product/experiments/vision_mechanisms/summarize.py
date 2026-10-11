"""Curate the complete pilot observations without selecting a winning scope."""

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("vision_study", HERE / "study.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def curate(root):
    runs = {}
    digest = hashlib.sha256((HERE / "study.py").read_bytes()).hexdigest()
    for name in ["discovery", "final-1", "final-2"]:
        folder = root / name
        runs[name] = {
            k: json.loads((folder / f"{k}.json").read_text())
            for k in ["plan", "results", "completion"]
        }
        assert runs[name]["plan"]["source_sha256"] == digest
        assert runs[name]["plan"]["cases"] == [r["case"] for r in runs[name]["results"]]
        assert len(runs[name]["results"]) == runs[name]["completion"]["cases"]
    grouped = {}
    for name, run in runs.items():
        grouped[name] = {}
        for label in ["bicubic", "siglip_forward", "siglip_forward_backward"]:
            rows = [
                r
                for r in run["results"]
                if (label == "bicubic" and r["case"]["family"] == "bicubic")
                or (
                    label.startswith("siglip")
                    and r["case"]["family"] == "siglip_loss"
                    and r["case"]["backward"] == label.endswith("backward")
                )
            ]
            methods = list(rows[0]["vs_stock"])
            grouped[name][label] = {}
            for method in methods:
                checks = [c for r in rows for c in r["checks"][method]]
                direct = {}
                baselines = (
                    ["stock"] if label == "bicubic" else ["stock", "cached_sign", "bce_library"]
                )
                for baseline in baselines:
                    direct[baseline] = math.exp(
                        statistics.mean(
                            math.log(
                                study.summary(
                                    list(
                                        zip(
                                            r["samples_seconds"][baseline],
                                            r["samples_seconds"][method],
                                        )
                                    )
                                )["speedup"]
                            )
                            for r in rows
                        )
                    )
                grouped[name][label][method] = dict(
                    cases=len(rows),
                    exact_cases=sum(study.eligibility(r, method, "exact") for r in rows),
                    numerical_cases=sum(study.eligibility(r, method, "numerical") for r in rows),
                    max_abs=max(v["max_abs"] for c in checks for v in c["outputs"]),
                    timing_geomean_vs=direct,
                    interpretation="Timings include all cases, even correctness failures; not an acceptance gate or selected recipe.",
                )
    return dict(
        study="VISION-MECHANISMS",
        recorded_date="2026-10-09",
        unresolved="The supplied name 'lachmanick gca' could not be identified; no GCA algorithm was substituted or tested.",
        runs=runs,
        summary=grouped,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = curate(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        gzip.compress(json.dumps(data, indent=2, allow_nan=False).encode(), mtime=0)
    )
    print(json.dumps(data["summary"], indent=2))


if __name__ == "__main__":
    main()
