"""Freeze explicitly supplied Python examples in a copied project, before search."""

import copy
import json
import math
import os
from pathlib import Path
import random
import sys

from .._native.common import write
from ..jobs import read_json
from .config import check_entry_file, target, validate_cases
from .discovery import inventory
from .replay import copy_project
from .runtime import execute
from .worker import load_target


class CaseRecorder:
    """Opt-in test helper: record inputs before invoking the original callable.

    Use inside an input loader. Existing tests may call this helper explicitly;
    no test discovery, monkeypatching or inference of evaluation splits occurs.
    """

    def __init__(self):
        self.cases = []

    def call(self, case_id, split, function, /, *args, **kwargs):
        self.cases.append(
            dict(
                id=case_id,
                split=split,
                args=copy.deepcopy(list(args)),
                kwargs=copy.deepcopy(kwargs),
            )
        )
        return function(*args, **kwargs)


def freeze(rows, root):
    import numpy as np

    if type(rows) is not list or not 2 <= len(rows) <= 1000:
        raise ValueError("input loader must return a list of 2..1000 cases")
    directory = root / "_lossless_inputs"
    directory.mkdir(exist_ok=False)
    total, count = 0, 0

    def encode(value, depth=0):
        nonlocal total, count
        count += 1
        if depth > 40 or count > 100000:
            raise ValueError("loader input tree is too large or deep")
        if type(value) is np.ndarray:
            if (
                value.dtype.kind not in "buifc"
                or not value.flags.aligned
                or not value.flags.writeable
                or not (value.flags.c_contiguous or value.flags.f_contiguous)
            ):
                raise ValueError(
                    "loader arrays must be aligned, writeable, contiguous numeric arrays"
                )
            total += value.nbytes
            if total > 128 * 2**20:
                raise ValueError("loader arrays exceed the 128 MiB total input limit")
            name = f"_lossless_inputs/array_{count:06d}.npy"
            np.save(root / name, value, allow_pickle=False)
            return {"$npy": name}
        if value is None or type(value) in {bool, int, str}:
            return value
        if type(value) is float and math.isfinite(value):
            return value
        if type(value) is list:
            return [encode(v, depth + 1) for v in value]
        if type(value) is dict and all(type(k) is str for k in value) and "$npy" not in value:
            return {k: encode(v, depth + 1) for k, v in value.items()}
        raise ValueError(
            "loader inputs support numeric arrays and finite JSON values; $npy is reserved"
        )

    result = encode(rows)
    # Splits are intentionally mandatory for executed Python loaders.
    if any("split" not in row for row in result if isinstance(row, dict)):
        raise ValueError("input loaders must explicitly label discovery and evaluation cases")
    result = validate_cases(result, {"kind": "callable"})
    if {r["split"] for r in result} != {"discovery", "evaluation"}:
        raise ValueError("loader requires independent discovery and evaluation cases")
    data = json.dumps(result, allow_nan=False)
    if len(data.encode()) > 2**20:
        raise ValueError("loader JSON exceeds 1 MiB")
    return result


def collect(project, output, loader, *, python, include, exclude, inherit_env, timeout, progress):
    """Run trusted loader code once in selected Python; never touch original files."""
    loader = target(loader)
    before = inventory(project, exclude, include=include)
    check_entry_file(loader, before)
    staged = output / "input-project"
    copy_project(project, staged, before)
    folder = output / "input-loader"
    folder.mkdir()
    home = folder / "home"
    home.mkdir()
    payload = folder / "payload.json"
    write(payload, dict(root=str(staged), loader=loader, result=str(folder / "cases.json")))
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "SYSTEMROOT", *inherit_env}}
    env.update(
        HOME=str(home),
        TMPDIR=str(home) + "/",
        PYTHONHASHSEED="0",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
    )
    progress(f"INPUT LOADER {loader}; executing once in copied project")
    receipt = execute(
        [
            str(python or sys.executable),
            "-s",
            "-c",
            "import sys; from lossless.applications.input_loader import main; main(sys.argv[1])",
            str(payload),
        ],
        cwd=staged,
        env=env,
        stdin="",
        folder=folder,
        timeout=timeout,
        max_output_bytes=2**20,
        progress=progress,
    )
    write(folder / "receipt.json", receipt)
    if receipt["status"] != "completed":
        raise ValueError(f"input loader {receipt['status']}; inspect {folder / 'stderr.log'}")
    after = inventory(staged, exclude, include=include)
    if any(after["files"].get(k) != v for k, v in before["files"].items()):
        raise ValueError("input loader changed copied source files")
    if any(
        k not in before["files"] and not k.startswith("_lossless_inputs/") for k in after["files"]
    ):
        raise ValueError("input loader created unexpected project files")
    return staged, folder / "cases.json", before, receipt


def main(payload):
    import numpy as np

    config = read_json(payload)
    root = Path(config["root"]).resolve()
    sys.path[:0] = [str(root), str(root / "src")]
    random.seed(0)
    np.random.seed(0)
    rows = load_target(config["loader"], root, {})()
    write(config["result"], freeze(rows, root))


if __name__ == "__main__":
    main(sys.argv[1])
