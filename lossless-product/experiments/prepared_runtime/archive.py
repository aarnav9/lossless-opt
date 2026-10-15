"""Curate complete runtime study evidence, including negative/failed runs."""

import argparse
import difflib
import gzip
import hashlib
import json
from pathlib import Path


def sources(folder):
    return {p.relative_to(folder).as_posix(): p.read_text() for p in sorted(folder.rglob("*.py"))}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    baseline = sources(root / "baseline-engine")
    evidence = dict(
        provenance=json.loads((root / "provenance.json").read_text()),
        fixtures_provenance=json.loads((root / "fixtures/provenance.json").read_text()),
        selection=json.loads((root / "selection.json").read_text()),
        reports={},
        logs={},
        engines={},
        fixtures={},
        harness=sources(Path(__file__).parent),
        baseline_sha256={name: digest(value) for name, value in baseline.items()},
        generators={
            name: (Path(__file__).parents[2] / "scripts" / name).read_text()
            for name in ("check_cpu_project.py", "check_retained_application.py")
        },
        upstream_licenses={
            p.name: p.read_text()
            for p in (Path(__file__).parents[2] / "src/lossless/retained").rglob("*LICENSE*")
            if p.is_file()
        },
    )
    for path in sorted(root.glob("[0-9][0-9]-*/report.json")):
        evidence["reports"][path.parent.name] = json.loads(path.read_text())
    for path in sorted(root.glob("05b-*/report.json")):
        evidence["reports"][path.parent.name] = json.loads(path.read_text())
    for name in evidence["reports"]:
        evidence["logs"][name] = (root / (name + ".log")).read_text()
    evidence["product_workflows"] = {
        path.parent.parent.name: json.loads(path.read_text())
        for path in sorted(root.glob("product-*/workflow/report.json"))
    }
    evidence["upstream_licenses"]["PYBASELINES_LICENSES_bundled.txt"] = (
        Path(__file__).parents[2] / "docs/evidence/PYBASELINES_LICENSES_bundled.txt"
    ).read_text()
    for folder in sorted(root.glob("engine-*")):
        current = sources(folder)
        patch = []
        for name in sorted(baseline.keys() | current.keys()):
            if baseline.get(name) != current.get(name):
                patch.extend(
                    difflib.unified_diff(
                        baseline.get(name, "").splitlines(keepends=True),
                        current.get(name, "").splitlines(keepends=True),
                        fromfile="a/" + name if name in baseline else "/dev/null",
                        tofile="b/" + name if name in current else "/dev/null",
                    )
                )
        evidence["engines"][folder.name] = dict(
            patch="".join(patch), sha256={name: digest(value) for name, value in current.items()}
        )
    for name in ("financial", "pybaselines", "arrays"):
        folder = root / "fixtures" / name
        evidence["fixtures"][name] = dict(
            manifest=json.loads((folder / "manifest.json").read_text()),
            cases=json.loads((folder / "cases.json").read_text()),
            sources={variant: sources(folder / variant) for variant in ("reference", "optimized")},
            licenses={
                p.relative_to(folder).as_posix(): p.read_text()
                for p in folder.rglob("*LICENSE*")
                if p.is_file()
            },
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n").encode()
    args.output.write_bytes(gzip.compress(data, mtime=0))
    print(
        f"reports={len(evidence['reports'])} bytes={args.output.stat().st_size} output={args.output}"
    )


if __name__ == "__main__":
    main()
