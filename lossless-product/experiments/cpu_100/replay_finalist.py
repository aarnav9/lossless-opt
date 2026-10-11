"""Rebuild one published CPU-100 finalist without author calls or private artifacts.

This replays already-public cases. It is not a new holdout or deployment approval.
"""

import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

from common import HERE, ROOT, read, sha, stamp, tree, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.pairs <= 40:
        parser.error("--pairs must be between 2 and 40")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stamp(f"START replay family={args.family} result={out}/replay.json log=caller tee output")
    evidence_path = ROOT / "lossless-product/docs/evidence/cpu-100.json.gz"
    evidence = json.loads(gzip.decompress(evidence_path.read_bytes()))
    frozen = evidence["finals"][str(args.family)]["frozen"]
    write(out / "selection.json", frozen)
    spec = importlib.util.spec_from_file_location("campaign_driver", HERE / "study.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.prepare(out)
    if tree(out / "baseline") != evidence["protocol"]["baseline"]:
        raise ValueError("application reference differs from the published reference")
    baseline = out / "baseline"
    kind = "application"
    folder = out / "candidate"
    folder.mkdir()
    if args.family == 3:
        from native_trials import build, cases, freeze_baseline

        baseline = freeze_baseline(out)
        case_list = cases(True)
        kind = "native"
    else:
        case_list = driver.freeze_cases(out / "fixtures", True, args.family)
        if args.family in (1, 5):
            from compiled import build
        elif args.family == 2:
            from parallel import build
        else:
            from prepared import build
    selected = frozen["selected"]
    if args.family == 5:
        # The authored source uses the guarded family-1 finalist's Python binding.
        candidate = build(
            baseline, folder, evidence["finals"]["1"]["frozen"]["selected"]["proposal"]
        )
        number, arm = selected["number"].split(":")
        trial = next(
            r for r in evidence["trials"] if r["family"] == 5 and r["number"] == int(number)
        )
        source = candidate["project"] / "cpu_region.c"
        source.write_text(trial["arms"][arm]["source_code"])
        if sha(source) != selected["source_sha256"]:
            raise ValueError("published author source identity differs")
        command = read(folder / "build.json")["command"]
        compiled = subprocess.run(command, capture_output=True, text=True, timeout=30)
        write(
            folder / "author-build.json",
            dict(command=command, returncode=compiled.returncode, diagnostics=compiled.stderr),
        )
        compiled.check_returncode()
    else:
        candidate = build(baseline, folder, selected["proposal"])
    if args.smoke:
        case_list = case_list[:1]
    write(out / "cases.json", case_list)
    write(out / "candidate-source.json", tree(candidate["project"]))
    shutil.copytree(HERE, out / "harness", ignore=shutil.ignore_patterns("__pycache__"))
    result = driver.measure(
        out / "measurement",
        dict(project=baseline),
        candidate,
        case_list,
        out / "fixtures",
        pairs=2 if args.smoke else args.pairs,
        kind=kind,
    )
    write(
        out / "replay.json",
        dict(
            scope="Public-case replay; no new qualification or promotion",
            evidence_sha256=sha(evidence_path),
            family=args.family,
            smoke=args.smoke,
            result=result,
        ),
    )
    stamp(
        f"COMPLETE replay exact={result['correctness']['exact']} speedup={result['timing']['speedup']:.4f} result={out}/replay.json log=caller tee output"
    )


if __name__ == "__main__":
    main()
