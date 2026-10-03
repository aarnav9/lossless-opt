"""Portable manifests and guarded native implementations; no pickle loading."""

from pathlib import Path
import platform
import shutil
import sys

from .jobs import read_json
from ._native.common import sha, write


def export(run, destination):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    summary = read_json(run / "report.json")
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
        "scope": "Finite case evidence. Timed deployment interface is bind(...), then repeated bound calls; bind validates and allocates outside steady-state timing.",
        "hashes": {p.name: sha(p) for p in destination.iterdir() if p.is_file()},
    }
    write(destination / "manifest.json", manifest)
    (destination / "USAGE.txt").write_text(
        "import lossless\noperation = lossless.load('PATH_TO_THIS_DIRECTORY')\n# x must be float32; optional residual/weight apply to RMSNorm.\nresult = operation(x)\n# For the measured preallocated interface:\n# bound = operation.bind(x)\n# result = bound()\n# Bound inputs must satisfy the contract for every invocation.\n# No LLM or proof checker is called during execution.\n"
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


class NativeOperation:
    def __init__(self, path, manifest):
        import numpy as np
        from ._native.operators import load_library

        self.manifest = manifest
        self.operator = manifest["adapter"].split(".")[1]
        self.library = None
        self.fallback_reason = None
        self.cases = {(c["rows"], c["columns"], c["layout"]) for c in manifest["cases"]}
        compatible = (
            manifest["platform"] == sys.platform
            and manifest["machine"] == platform.machine()
            and manifest["numpy"] == np.__version__
        )
        binary = manifest.get("binary")
        if compatible and binary and binary in manifest["hashes"]:
            self.library = load_library(path / binary)
        else:
            self.fallback_reason = "artifact environment differs or compiled binary is absent"

    def bind(self, x, residual=None, weight=None):
        """Bind buffers once. Caller preserves their shape, strides and contract.

        Output/scratch ownership belongs to this bound callable. Its return value
        is overwritten by its next call; separate bindings have separate buffers.
        Input mutations must remain within the admitted domain. No concurrent calls.
        """
        import numpy as np
        from ._native import operators

        if not isinstance(x, np.ndarray) or x.dtype != np.float32 or x.ndim != 2 or not x.size:
            raise ValueError("x must be a nonempty rank-two float32 array")
        if any(s <= 0 or s % 4 for s in x.strides):
            raise ValueError("only positive aligned strides are supported")
        rows, cols = x.shape
        if x.strides == (cols * 4, 4):
            layout = "c"
        elif x.strides == (4, rows * 4):
            layout = "f"
        elif x.strides == (cols * 8, 8):
            layout = "slice2"
        else:
            layout = None
        if self.operator != "copy":
            bound = 10000 if self.operator == "softmax" else 1000
            if not np.isfinite(x).all() or np.max(np.abs(x)) > bound:
                raise ValueError("input lies outside the numerical contract")
        if self.operator == "rmsnorm_residual":
            if (
                not isinstance(residual, np.ndarray)
                or residual.dtype != np.float32
                or residual.shape != x.shape
                or residual.strides != x.strides
            ):
                raise ValueError("residual must match x shape, dtype and strides")
            if (
                not isinstance(weight, np.ndarray)
                or weight.dtype != np.float32
                or weight.shape != (cols,)
                or not weight.flags.c_contiguous
            ):
                raise ValueError("weight must be a contiguous float32 vector matching columns")
            if (
                not np.isfinite(residual).all()
                or np.max(np.abs(residual)) > 1000
                or not np.isfinite(weight).all()
                or np.max(np.abs(weight)) > 2
                or np.shares_memory(x, residual)
                or np.shares_memory(x, weight)
                or np.shares_memory(residual, weight)
            ):
                raise ValueError(
                    "RMSNorm buffers must be finite and nonoverlapping, with abs(residual)<=1000 and abs(weight)<=2"
                )
        else:
            residual = x  # unused by copy and softmax implementations
            weight = np.ones(cols, dtype=np.float32)
        arrays = (x, residual, weight)
        case = {"rows": rows, "columns": cols, "layout": layout}
        compatible = (
            layout is not None and (rows, cols, layout) in self.cases and self.library is not None
        )
        if not compatible:
            self.fallback_reason = (
                self.fallback_reason or "shape or layout outside validated artifact cases"
            )

            def reference():
                return np.asarray(
                    operators.reference(self.operator, arrays), dtype=np.float32, order="C"
                )

            return reference
        if self.manifest["selected"] == "reference":
            funcs, _, _ = operators.executors(
                self.operator, case, arrays, self.library, self.library
            )
            comparator = self.manifest.get("comparator", "native_baseline")
            if comparator not in funcs:
                self.fallback_reason = "declared comparator unavailable; independent reference used"
                return lambda: np.asarray(
                    operators.reference(self.operator, arrays), dtype=np.float32, order="C"
                )
            return funcs[comparator]
        funcs, _, _ = operators.executors(self.operator, case, arrays, self.library, self.library)
        return funcs["proposal"]

    def __call__(self, x, residual=None, weight=None):
        return self.bind(x, residual, weight)()
