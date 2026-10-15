"""Save real CPU reference outputs and explain bit differences without relaxing gates."""

import os
from pathlib import Path
import time

from .._native.common import stamp, write
from ..jobs import identity, read_json
from ..jobs import positive
from .bundle import read_outputs
from .config import inspect, resolve
from .replay import ReplayResult
from .runtime import execute
from .worker import Observer


def differences(left, right, *, limit=20):
    import numpy as np

    rows = []

    def visit(a, b, path):
        if len(rows) >= limit:
            return
        if isinstance(a, (np.ndarray, np.generic)) and isinstance(b, (np.ndarray, np.generic)):
            if isinstance(a, np.generic) != isinstance(b, np.generic):
                rows.append(
                    dict(
                        path=path,
                        kind="value_type",
                        left_type=type(a).__name__,
                        right_type=type(b).__name__,
                    )
                )
                return
            a, b = np.asarray(a), np.asarray(b)
            if a.shape != b.shape or a.dtype != b.dtype:
                rows.append(
                    dict(
                        path=path,
                        kind="array_metadata",
                        left_shape=list(a.shape),
                        right_shape=list(b.shape),
                        left_dtype=str(a.dtype),
                        right_dtype=str(b.dtype),
                    )
                )
                return
            if a.tobytes() == b.tobytes():
                return
            raw_a = (
                np.ascontiguousarray(a).reshape(-1).view(np.uint8).reshape(a.size, a.dtype.itemsize)
            )
            raw_b = (
                np.ascontiguousarray(b).reshape(-1).view(np.uint8).reshape(b.size, b.dtype.itemsize)
            )
            mask = np.any(raw_a != raw_b, axis=1)
            first = int(np.flatnonzero(mask)[0])
            row = dict(
                path=path,
                kind="array_bits",
                dtype=str(a.dtype),
                shape=list(a.shape),
                differing_elements=int(mask.sum()),
                first_index=list(map(int, np.unravel_index(first, a.shape))),
                left=repr(a.reshape(-1)[first].item()),
                right=repr(b.reshape(-1)[first].item()),
            )
            if a.dtype.kind in "fc":
                # Compare real components for complex numbers; preserve nonfinite distinctions.
                component = (
                    np.dtype(f"f{a.dtype.itemsize // 2}").newbyteorder(a.dtype.byteorder)
                    if a.dtype.kind == "c"
                    else a.dtype
                )
                x = np.ascontiguousarray(a).reshape(-1).view(component)
                y = np.ascontiguousarray(b).reshape(-1).view(component)
                finite = np.isfinite(x) & np.isfinite(y)
                if np.any(finite):
                    dx = np.abs(x[finite].astype(np.longdouble) - y[finite].astype(np.longdouble))
                    scale = np.maximum(np.abs(x[finite]), np.abs(y[finite])).astype(np.longdouble)
                    row["max_absolute_component_difference"] = str(np.max(dx))
                    row["max_relative_component_difference"] = str(
                        np.max(dx / np.maximum(scale, np.finfo(component).tiny))
                    )
                    if component.itemsize in (4, 8) and component.isnative:
                        unsigned = np.dtype(f"u{component.itemsize}")
                        sign = np.array(1 << (component.itemsize * 8 - 1), dtype=unsigned)

                        def ordered(v):
                            bits = v[finite].view(unsigned)
                            return np.where(bits & sign, ~bits, bits ^ sign)

                        u, v = ordered(x), ordered(y)
                        row["max_ulp_component_distance"] = int(
                            np.max(np.maximum(u, v) - np.minimum(u, v))
                        )
                        row["ulp_scope"] = (
                            "Adjacent ordered IEEE representations; signed zeros are distinct. Finite components only."
                        )
                row["nonfinite_components"] = int((~finite).sum())
            rows.append(row)
        elif type(a) is type(b) and isinstance(a, (list, tuple)) and len(a) == len(b):
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f"{path}[{index}]")
        elif type(a) is dict and type(b) is dict and list(a) == list(b):
            for key in a:
                visit(a[key], b[key], f"{path}.{key}")
        elif identity(Observer(64 * 2**20).encode(a)) != identity(Observer(64 * 2**20).encode(b)):
            if type(a) is float and type(b) is float:
                visit(np.asarray(a), np.asarray(b), path)
            else:
                rows.append(
                    dict(
                        path=path,
                        kind="value_or_structure",
                        left=repr(a)[:200],
                        right=repr(b)[:200],
                    )
                )

    visit(left, right, "outputs")
    return rows


def diagnose_reference(config, *, output, repeats=5, budget=60):
    from .workflow import make_bundle

    if type(repeats) is not int or not 2 <= repeats <= 20 or budget <= 0:
        raise ValueError("diagnostics require 2..20 repeats and a positive budget")
    positive(budget, "diagnostic budget")
    started = time.perf_counter()
    out = Path(output).resolve()
    value, cases = resolve(config)
    root = Path(value["project"]["root"]).resolve()
    if out == root or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError("diagnostic output must be separate from the source project")
    out.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="incomplete",
        kind="reference_diagnostics",
        llm_calls=0,
        cases=[],
        scope="Discovery-only diagnostic snapshots of returned values. No tolerance, numerical acceptance or optimization claim. Identical return values do not establish identical hidden state.",
    )
    try:
        info = inspect(config)
        if not value.get("entry") or value["entry"]["kind"] != "callable":
            raise ValueError("diagnostics require a callable")
        selected = [c for c in cases if c["split"] == "discovery"][:2]
        if not selected:
            raise ValueError("diagnostics require discovery cases")
        bundle = make_bundle(
            out,
            Path(value["project"]["root"]),
            info["source"],
            None,
            None,
            value,
            selected,
            {},
            "diagnostic_only",
        )
        for case in selected:
            row = dict(id=case["id"], attempts=[])
            report["cases"].append(row)
            base = None
            for repeat in range(repeats):
                left = budget - (time.perf_counter() - started)
                if left <= 0:
                    raise TimeoutError("reference diagnosis budget exhausted")
                folder = out / "attempts" / f"{case['id']}_{repeat:02d}"
                folder.mkdir(parents=True)
                stamp(
                    f"DIAGNOSE case={case['id']} repeat={repeat + 1}/{repeats} result={out}/report.json"
                )
                receipt = execute(
                    [
                        value["environment"]["python"],
                        str(bundle / "run.py"),
                        "--case",
                        case["id"],
                        "--reference",
                        "--output",
                        str(folder / "values"),
                    ],
                    cwd=bundle,
                    env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
                    stdin="",
                    folder=folder,
                    timeout=min(left, value["replay"]["timeout_seconds"]),
                    max_output_bytes=value["replay"]["max_output_bytes"],
                    progress=stamp,
                )
                write(folder / "receipt.json", receipt)
                if receipt["status"] != "completed":
                    raise ValueError(f"reference diagnostic failed; inspect {folder}/stderr.log")
                result = read_json(folder / "values/result.json")
                values = read_outputs(folder / "values")
                if base is None:
                    base = values
                row["attempts"].append(
                    dict(
                        fingerprint=result["fingerprint"],
                        differences=differences(base, values),
                        values=str((folder / "values").relative_to(out)),
                    )
                )
                write(out / "report.json", report)
        report["status"] = (
            "reference_unstable"
            if any(len({a["fingerprint"] for a in row["attempts"]}) > 1 for row in report["cases"])
            else "repeatable_in_diagnostic"
        )
    except (ValueError, OSError, TimeoutError) as error:
        report.update(status="failed", reason=str(error))
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        write(out / "report.json", report)
        stamp(f"DONE diagnostic={report['status']} result={out}/report.json")
    return ReplayResult(out, report)
