"""Rebuild frozen runtime sources and deterministic fixtures from published evidence."""

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def write_tree(root, files):
    for name, value in files.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("archive path escaped output")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--evidence",
        type=Path,
        default=Path(__file__).parents[2] / "docs/evidence/cpu-request-runtime.json.gz",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    evidence = json.loads(gzip.decompress(a.evidence.read_bytes()))
    commit = evidence["selection"]["baseline_commit"]
    repo = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    data = subprocess.check_output(["git", "archive", commit, "lossless-product/src"], cwd=repo)
    baseline = out / "baseline-engine"
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive:
            if not member.isfile():
                continue
            relative = Path(member.name).relative_to("lossless-product/src")
            path = baseline / relative
            if not path.resolve().is_relative_to(baseline):
                raise ValueError("git archive escaped output")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())
    for name, engine in evidence["engines"].items():
        root = out / name
        shutil.copytree(baseline, root)
        subprocess.run(
            ["git", "apply", "-"], input=engine["patch"], text=True, cwd=root, check=True
        )
        hashes = {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*.py")
        }
        if hashes != engine["sha256"]:
            raise ValueError("runtime hash mismatch: " + name)
    helpers = out / "generators"
    write_tree(helpers, evidence["generators"])
    sys.path.insert(0, str(helpers))
    spec = importlib.util.spec_from_file_location(
        "request_fixture_generator", helpers / "check_cpu_project.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, kind in [("financial", "numpy-financial"), ("pybaselines", "pybaselines")]:
        saved = evidence["fixtures"][name]
        upstream = out / "upstream" / name
        write_tree(upstream, saved["sources"]["reference"])
        # Financial's generator expects its upstream license at the package root.
        if name == "financial":
            license_text = next(
                v for k, v in saved["licenses"].items() if k.endswith("LICENSE.txt")
            )
            (upstream / "LICENSE.txt").write_text(license_text)
        project, _, _ = module.prepare(kind, upstream, out / "generated" / name, matrix=True)
        bundle = out / "source-bundles" / name
        for variant in ("reference", "optimized"):
            root = bundle / variant
            write_tree(root, saved["sources"][variant])
            for fixture in project.rglob("*.npy"):
                target = root / fixture.relative_to(project)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(fixture, target)
        write_tree(bundle, saved["licenses"])
        (bundle / "manifest.json").write_text(json.dumps(saved["manifest"]))
        (bundle / "cases.json").write_text(json.dumps(saved["cases"]))
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("prepare.py")),
            "--financial",
            str(out / "source-bundles/financial"),
            "--pybaselines",
            str(out / "source-bundles/pybaselines"),
            "--seed",
            str(evidence["fixtures_provenance"]["seed"]),
            "--output",
            str(out / "fixtures"),
        ],
        check=True,
    )
    for name, saved in evidence["fixtures"].items():
        for variant in ("reference", "optimized"):
            root = out / "fixtures" / name / variant
            actual = {
                p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*")
                if p.is_file()
            }
            expected = {k: v["sha256"] for k, v in saved["manifest"][variant + "_files"].items()}
            if actual != expected:
                raise ValueError("fixture/source hash mismatch: " + name + "/" + variant)
    print(f"Exact runtime and fixture hashes verified. Rebuilt research inputs: {out}", flush=True)


if __name__ == "__main__":
    main()
