"""Portable manifests and guarded native implementations; no pickle loading."""

from pathlib import Path
import platform
import shutil
import sys

from .jobs import read_json
from ._native.common import sha, write
from ._native.deployment import NativeOperation as NativeOperation


def export(run, destination):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    summary = read_json(run / "report.json")
    if isinstance(summary, dict) and summary.get("kind") in {
        "application_replay",
        "application_profile",
        "application_search",
    }:
        raise ValueError(
            "reference replay, profiling and source experiments do not produce a guarded deployment artifact"
        )
    if not isinstance(summary, dict) or "adapter" not in summary:
        raise ValueError("invalid or changed optimization report")
    if summary["adapter"] == "mlx.fixed_count":
        from .adapters.mlx.job import export as export_mlx

        return export_mlx(run, destination)
    from ._native import engine

    state = engine.verify(run)
    if summary["status"] == "incomplete":
        raise ValueError(
            "an incomplete search has no deployable optimization; keep using the reference"
        )
    chosen = summary["selected"]
    if chosen != "reference" and (
        not state["sealed"]
        or state["sealed"]["global_choice"] != chosen
        or state["proposals"][chosen]["rejected"]
    ):
        raise ValueError("report does not match the sealed candidate")
    resolved = read_json(run / "resolved.json")
    destination.mkdir(parents=True, exist_ok=False)
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    files = ["baseline.c", "contract.json", "report.json", "resolved.json"]
    for name in files:
        shutil.copy2(run / name, destination / name)
    source = run / ("baseline.c" if chosen == "reference" else f"proposals/{chosen}/kernel.c")
    shutil.copy2(source, destination / "implementation.c")
    binary = run / "build" / (("baseline" if chosen == "reference" else chosen) + suffix)
    if binary.exists():
        shutil.copy2(binary, destination / ("implementation" + suffix))
    manifest = {
        "schema_version": 1,
        "adapter": summary["adapter"],
        "selected": chosen,
        "comparator": summary.get("comparator", "native_baseline"),
        "platform": sys.platform,
        "machine": platform.machine(),
        "python_abi": list(sys.version_info[:2]),
        "numpy": __import__("numpy").__version__,
        "binary": "implementation" + suffix if binary.exists() else None,
        "cases": resolved["workload"]["cases"],
        "contract": summary["contract"],
        "evidence": "validated",
        "timing_scope": summary.get("timing_scope", "bound"),
        "scope": "Finite case evidence. "
        + (
            "Acceptance measured ordinary operation(...) calls, including guards, allocation, binding and dispatch."
            if summary.get("timing_scope", "bound") == "call"
            else "Acceptance measured repeated bound() calls; input guards, binding and allocation are excluded."
        ),
        "hashes": {p.name: sha(p) for p in destination.iterdir() if p.is_file()},
    }
    write(destination / "manifest.json", manifest)
    (destination / "USAGE.txt").write_text(
        "import lossless\noperation = lossless.load('PATH_TO_THIS_DIRECTORY')\n# x must be float32; optional residual/weight apply to RMSNorm.\nresult = operation(x)\n# manifest.json timing_scope identifies the measured interface.\n# Explicit reusable-buffer interface:\n# bound = operation.bind(x)\n# result = bound()\n# Bound inputs must satisfy the contract for every invocation.\n# No LLM or proof checker is called during execution.\n"
    )
    return destination


def verify(path):
    path = Path(path).resolve()
    manifest = read_json(path / "manifest.json")
    if type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1:
        raise ValueError("unsupported artifact version")
    hashes = manifest.get("hashes")
    required = {"report.json", "resolved.json"}
    if manifest.get("adapter") == "mlx.fixed_count":
        required.update(
            {"hardware.json", "model_identity.json", "runtime_identity.json", "sealed.json"}
        )
    else:
        required.update({"contract.json", "implementation.c", "baseline.c"})
        if manifest.get("binary"):
            required.add(manifest["binary"])
    if not isinstance(hashes, dict) or not required <= set(hashes):
        raise ValueError("artifact manifest is missing required content hashes")
    for name, digest in hashes.items():
        candidate = (path / name).resolve()
        if candidate.parent != path or not candidate.is_file() or sha(candidate) != digest:
            raise ValueError(f"artifact content changed or missing: {name}")
    return path, manifest


def load(path):
    path, manifest = verify(path)
    if manifest["adapter"] == "mlx.fixed_count":
        from .adapters.mlx.job import load as load_mlx

        return load_mlx(path, manifest)
    if manifest["adapter"] not in {"native.copy", "native.softmax", "native.rmsnorm_residual"}:
        raise ValueError("unsupported artifact adapter")
    return NativeOperation(path, manifest)
