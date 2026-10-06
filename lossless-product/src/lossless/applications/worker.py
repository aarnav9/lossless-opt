"""Standalone reference worker, runnable in the user's selected Python environment.

This executes trusted user code. Fresh processes/workspaces are not a security
sandbox. No candidate search, profiling or acceptance happens here.
"""

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import random
import struct
import sys
import time
import traceback


def load_target(name, root, loaded):
    module, symbol = name.split(":")
    if module not in loaded:
        if module.endswith(".py"):
            path = (root / module).resolve()
            if not path.is_relative_to(root):
                raise ValueError("entry file escaped copied project")
            base = root / "src" if path.is_relative_to(root / "src") else root
            parts = list(path.relative_to(base).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            if parts and all(part.isidentifier() for part in parts):
                obj = importlib.import_module(".".join(parts))
                if Path(obj.__file__).resolve() != path:
                    raise ValueError(
                        "entry resolved to a different module; use its dotted package name"
                    )
            else:
                spec = importlib.util.spec_from_file_location(
                    "_lossless_application_" + hashlib.sha256(module.encode()).hexdigest()[:12],
                    path,
                )
                obj = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = obj
                spec.loader.exec_module(obj)
        else:
            obj = importlib.import_module(module)
            origin = getattr(obj, "__file__", None)
            if not origin or not Path(origin).resolve().is_relative_to(root):
                raise ValueError(
                    "entry module must resolve inside the copied project, not an installed original"
                )
        loaded[module] = obj
    result = loaded[module]
    for part in symbol.split("."):
        result = getattr(result, part)
    if not callable(result):
        raise ValueError(f"{name} is not callable")
    return result


def inputs(value, root):
    if isinstance(value, dict):
        if "$npy" in value:
            if set(value) != {"$npy"} or not isinstance(value["$npy"], str):
                raise ValueError('array input must be exactly {"$npy": "relative/file.npy"}')
            path = (root / value["$npy"]).resolve()
            if not path.is_relative_to(root) or path.suffix != ".npy":
                raise ValueError("array fixture must be a .npy file in the copied project")
            import numpy as np

            return np.load(path, allow_pickle=False)
        return {k: inputs(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [inputs(v, root) for v in value]
    return value


class Observer:
    def __init__(self, limit):
        self.remaining = limit
        self.nodes = 100000

    def bytes(self, value):
        self.remaining -= len(value)
        if self.remaining < 0:
            raise ValueError("observation exceeds replay.max_output_bytes")
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}

    def encode(self, value, depth=0):
        self.nodes -= 1
        if depth > 40 or self.nodes < 0:
            raise ValueError("observation tree is too deep or large (possibly cyclic)")
        if value is None or type(value) in {bool, int}:
            return {"type": type(value).__name__, "value": value}
        if type(value) is float:
            return {"type": "float64", "bits": struct.pack(">d", value).hex()}
        if isinstance(value, str):
            return {"type": "str", **self.bytes(value.encode())}
        if isinstance(value, bytes):
            return {"type": "bytes", **self.bytes(value)}
        if isinstance(value, (list, tuple)):
            return {
                "type": type(value).__name__,
                "items": [self.encode(v, depth + 1) for v in value],
            }
        if isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                raise ValueError(
                    "observed dictionaries require string keys; supply entry.observe for custom objects"
                )
            return {
                "type": "dict",
                "items": [[k, self.encode(v, depth + 1)] for k, v in value.items()],
            }
        module = type(value).__module__.split(".")[0]
        if module in {"numpy", "mlx", "torch"}:
            import numpy as np

            if module == "torch":
                import torch

                tensor = value.detach()
                if tensor.layout != torch.strided or tensor.is_quantized:
                    raise ValueError("sparse/quantized tensors require entry.observe")
                data = (
                    tensor.cpu()
                    .resolve_conj()
                    .resolve_neg()
                    .contiguous()
                    .reshape(-1)
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )
                shape, dtype = list(tensor.shape), str(tensor.dtype)
            else:
                array = np.asarray(value)  # Materializes lazy MLX results.
                if array.dtype.hasobject:
                    raise ValueError("object arrays require entry.observe")
                shape, dtype, data = list(array.shape), str(array.dtype), array.tobytes(order="C")
            return {"type": module + ".array", "shape": shape, "dtype": dtype, **self.bytes(data)}
        raise ValueError(f"unsupported observed type {type(value).__name__}; provide entry.observe")


def run(payload):
    root = Path(payload["workspace"]).resolve()
    sys.path[:0] = [str(root), str(root / "src")]
    random.seed(payload["seed"])
    if importlib.util.find_spec("numpy") is not None:
        import numpy as np

        np.random.seed(payload["seed"])
    entry, loaded = payload["entry"], {}
    started = time.perf_counter()
    function = load_target(entry["target"], root, loaded)
    selected = (
        function(*inputs(entry["factory_args"], root), **inputs(entry["factory_kwargs"], root))
        if entry["factory"]
        else function
    )
    if not callable(selected):
        raise ValueError("factory must return a callable function/model object")
    observe = load_target(entry["observe"], root, loaded) if entry.get("observe") else None
    cleanup = load_target(entry["cleanup"], root, loaded) if entry.get("cleanup") else None
    setup = time.perf_counter() - started
    records = []
    try:
        for call in payload["case"]["calls"]:
            args, kwargs = inputs(call["args"], root), inputs(call["kwargs"], root)
            before = Observer(payload["max_output_bytes"]).encode({"args": args, "kwargs": kwargs})
            started = time.perf_counter()
            result = selected(*args, **kwargs)
            observed = observe(selected, result, args, kwargs) if observe else result
            outputs = Observer(payload["max_output_bytes"]).encode(observed)
            after = Observer(payload["max_output_bytes"]).encode({"args": args, "kwargs": kwargs})
            records.append(
                {
                    "observation": {
                        "output": outputs,
                        "inputs_before": before,
                        "inputs_after": after,
                    },
                    "call_and_observation_seconds": time.perf_counter() - started,
                }
            )
    finally:
        if cleanup:
            cleanup(selected)
    return {
        "status": "completed",
        "setup_seconds": setup,
        "calls": records,
        "scope": "Output and post-call input fingerprints, optionally augmented by entry.observe. Includes observation/materialization overhead; not benchmark timing or a correctness proof.",
    }


def main():
    payload = json.loads(Path(sys.argv[1]).read_text())
    try:
        result = run(payload)
    except BaseException as error:
        traceback.print_exc()
        result = {"status": "failed", "error": type(error).__name__, "message": str(error)}
    Path(payload["result"]).write_text(json.dumps(result, allow_nan=False) + "\n")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
