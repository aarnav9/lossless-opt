"""Create fresh representative fixtures in an example project copy."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    shutil.copytree(
        Path(__file__).parent / "project",
        output / "project",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    cases = []
    for i, (name, split, shape) in enumerate(
        [
            ("short", "discovery", (128, 32)),
            ("wide", "discovery", (96, 128)),
            ("new_short", "evaluation", (160, 32)),
            ("new_wide", "evaluation", (112, 128)),
        ]
    ):
        values = np.random.default_rng(772901 + i).normal(size=shape)
        np.save(output / "project" / (name + ".npy"), values)
        cases.append(dict(id=name, split=split, args=[{"$npy": name + ".npy"}]))
    (output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Project: {output / 'project'}\nCases: {output / 'cases.json'}", flush=True)


if __name__ == "__main__":
    main()
