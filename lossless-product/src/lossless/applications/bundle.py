"""Runnable, relocatable CPU project copies with pre-execution admission checks."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import struct
import subprocess
import sys
import tempfile
import time

from .._native.common import write
from ..jobs import identity, read_json
from .discovery import digest
from .worker import Observer, inputs, load_target


def environment():
    from .search import runner_identity

    packages = sorted(
        (d.metadata["Name"].lower(), d.version)
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    )
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        host=identity(dict(node=platform.node(), processors=os.cpu_count())),
        packages=packages,
        runner=runner_identity(),
    )


def signature(value):
    """Shape/layout and fixed controls; signal values can change within finite arrays."""
    import numpy as np

    if isinstance(value, np.ndarray):
        if (
            type(value) is not np.ndarray
            or value.dtype.kind not in "buifc"
            or not np.isfinite(value).all()
        ):
            raise ValueError("optimized execution requires plain finite numeric arrays")
        return dict(
            array=True,
            shape=list(value.shape),
            dtype=value.dtype.str,
            strides=list(value.strides),
            aligned=bool(value.flags.aligned),
            writeable=bool(value.flags.writeable),
        )
    if isinstance(value, list):
        return [signature(item) for item in value]
    if isinstance(value, dict):
        return {key: signature(item) for key, item in value.items()}
    return Observer(1024 * 1024).encode(value)


def admitted_signatures(cases, root):
    result = set()
    for case in cases:
        try:
            result.add(identity(signature(inputs(case["calls"], root))))
        except ValueError:
            pass
    return sorted(result)


def verify_tree(root, files):
    root = Path(root).resolve()
    actual = set()
    for folder, directories, names in os.walk(root):
        if any((Path(folder) / name).is_symlink() for name in directories):
            return False
        # Bytecode can shadow source and is deliberately excluded at creation.
        actual.update((Path(folder) / name).relative_to(root).as_posix() for name in names)
    return actual == set(files) and all(
        (root / name).is_file()
        and not (root / name).is_symlink()
        and (root / name).resolve().is_relative_to(root)
        and digest(root / name) == info["sha256"]
        for name, info in files.items()
    )


def select(bundle, manifest, calls, requested="auto"):
    bundle = Path(bundle).resolve()
    reference = bundle / "reference"
    if not verify_tree(reference, manifest["reference_files"]):
        raise ValueError("reference copy changed; rebuild the application bundle")
    if requested == "reference" or manifest["selected"] == "reference":
        return "reference", "reference requested or retained"
    if identity(environment()) != identity(manifest["environment"]):
        return "reference", "Python, packages, host or Lossless runtime changed"
    if (
        identity({name: os.environ.get(name) for name in manifest["inherit_env"]})
        != manifest["environment_values_hash"]
    ):
        return "reference", "declared environment settings changed"
    if not verify_tree(bundle / "optimized", manifest["optimized_files"]):
        return "reference", "optimized copy changed"
    try:
        supported = identity(signature(calls)) in manifest["input_signatures"]
    except ValueError:
        supported = False
    if not supported:
        return "reference", "input metadata or scalar controls outside measured cases"
    return "optimized", None


def save_value(value, output, counter, depth=0):
    import numpy as np

    if depth > 40:
        raise ValueError("output tree too deep")
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError("object arrays cannot be saved without pickle")
        name = f"array_{counter[0]:05d}.npy"
        counter[0] += 1
        np.save(output / name, array, allow_pickle=False)
        return {
            "type": "numpy_scalar" if isinstance(value, np.generic) else "ndarray",
            "file": name,
        }
    if value is None or type(value) in {bool, int, str}:
        return {"type": type(value).__name__, "value": value}
    if type(value) is float:
        return {"type": "float", "bits": struct.pack(">d", value).hex()}
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "items": [save_value(v, output, counter, depth + 1) for v in value],
        }
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {
            "type": "dict",
            "items": [[k, save_value(v, output, counter, depth + 1)] for k, v in value.items()],
        }
    raise ValueError("deliverable outputs must be Python scalars/containers or NumPy arrays")


def read_outputs(output):
    """Read saved return values; array loading never enables pickle."""
    import numpy as np

    output = Path(output).resolve()

    def decode(row):
        kind = row["type"]
        if kind in {"ndarray", "numpy_scalar"}:
            path = (output / row["file"]).resolve()
            if not path.is_relative_to(output):
                raise ValueError("array output escaped result directory")
            array = np.load(path, allow_pickle=False)
            return array[()] if kind == "numpy_scalar" else array
        if kind in {"NoneType", "bool", "int", "str"}:
            return row["value"]
        if kind == "float":
            return struct.unpack(">d", bytes.fromhex(row["bits"]))[0]
        if kind == "bytes":
            return bytes.fromhex(row["hex"])
        if kind == "dict":
            return {k: decode(v) for k, v in row["items"]}
        if kind in {"list", "tuple"}:
            result = [decode(v) for v in row["items"]]
            return tuple(result) if kind == "tuple" else result
        raise ValueError("unknown saved output type")

    return [decode(row) for row in read_json(output / "result.json")["outputs"]]


def run(bundle, *, case=None, request=None, output, variant="auto"):
    """Called in a fresh process by run.py; never switches variants mid-sequence."""
    import numpy as np
    from .config import validate_cases

    started = time.perf_counter()
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    if output == bundle or output.is_relative_to(bundle) or bundle.is_relative_to(output):
        raise ValueError("result directory must be separate from the bundle")
    if output.exists():
        raise ValueError("result directory exists; choose a fresh output path")
    manifest = read_json(bundle / "manifest.json")
    if (case is None) == (request is None):
        raise ValueError("choose one saved --case or new --request JSON")
    if request is not None:
        request = Path(request).resolve()
        row = {"id": "request", "split": "evaluation", **read_json(request)}
        row = validate_cases([row], manifest["entry"])[0]
        fixture_root = request.parent
    else:
        matches = [row for row in read_json(bundle / "cases.json") if row["id"] == case]
        if len(matches) != 1:
            raise ValueError("unknown saved case; inspect cases.json")
        row, fixture_root = matches[0], bundle / "reference"
    calls = inputs(row["calls"], fixture_root)
    selected, reason = select(bundle, manifest, calls, variant)
    root = bundle / selected
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(root), str(root / "src")]
    os.chdir(root)
    random.seed(manifest["seed"])
    np.random.seed(manifest["seed"])
    entry, loaded = manifest["entry"], {}
    setup_started = time.perf_counter()
    function = load_target(entry["target"], root, loaded)
    selected_call = (
        function(*inputs(entry["factory_args"], root), **inputs(entry["factory_kwargs"], root))
        if entry["factory"]
        else function
    )
    if not callable(selected_call):
        raise ValueError("factory must return a callable")
    observe = load_target(entry["observe"], root, loaded) if entry.get("observe") else None
    cleanup = load_target(entry["cleanup"], root, loaded) if entry.get("cleanup") else None
    setup_seconds = time.perf_counter() - setup_started
    results, retained, observations, times = [], [], [], []
    limit = manifest["max_output_bytes"]
    try:
        for call in calls:
            args, kwargs = call["args"], call["kwargs"]
            before = Observer(limit).encode(call)
            tick = time.perf_counter()
            result = selected_call(*args, **kwargs)
            times.append(time.perf_counter() - tick)
            observed = observe(selected_call, result, args, kwargs) if observe else result
            results.append(result)
            retained.append(observed)
            observations.append(
                dict(
                    output=Observer(limit).encode(observed),
                    inputs_before=before,
                    inputs_after=Observer(limit).encode(call),
                )
            )
        fingerprint = identity(
            dict(calls=observations, retained_outputs=[Observer(limit).encode(r) for r in retained])
        )
    finally:
        if cleanup:
            cleanup(selected_call)
        elif (
            type(selected_call).__module__ == "lossless.execution"
            and type(selected_call).__name__ == "IndependentMap"
        ):
            selected_call.close()
    output.mkdir(parents=True)
    counter = [0]
    saved = [save_value(result, output, counter) for result in results]
    record = dict(
        status="completed",
        variant=selected,
        fallback_reason=reason,
        outputs=saved,
        fingerprint=fingerprint,
        call_seconds=times,
        setup_seconds=setup_seconds,
        elapsed_seconds=time.perf_counter() - started,
        scope="Fresh-process launch. Serialized returns preserve values, not cross-result object aliases. No retry after a callable failure.",
    )
    write(output / "result.json", record)
    return record


def main(bundle):
    # Match evaluation's clean environment before loading the application or
    # NumPy. A parent process owns the temporary HOME and cleans it on failure.
    if "--_lossless_worker" not in sys.argv:
        manifest = read_json(bundle / "manifest.json")
        env = {
            k: v
            for k, v in os.environ.items()
            if k in {"PATH", "LANG", "SYSTEMROOT", *manifest["inherit_env"]}
        }
        with tempfile.TemporaryDirectory(prefix="lossless-launch-") as temporary:
            home = Path(temporary) / "home"
            home.mkdir()
            env.update(
                HOME=str(home),
                TMPDIR=temporary + "/",
                PYTHONHASHSEED=str(manifest["seed"]),
                PYTHONUNBUFFERED="1",
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONPATH=str(Path(__file__).resolve().parents[2]),
            )
            result = subprocess.run(
                [sys.executable, str(bundle / "run.py"), "--_lossless_worker", *sys.argv[1:]],
                env=env,
            )
        raise SystemExit(result.returncode)
    sys.argv.remove("--_lossless_worker")
    parser = argparse.ArgumentParser(
        description="Run a frozen CPU application copy; --reference disables optimization."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--case")
    source.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    result = run(
        bundle,
        case=args.case,
        request=args.request,
        output=args.output,
        variant="reference" if args.reference else "auto",
    )
    print(
        json.dumps(
            {k: result[k] for k in ("status", "variant", "fallback_reason", "elapsed_seconds")}
        )
    )
