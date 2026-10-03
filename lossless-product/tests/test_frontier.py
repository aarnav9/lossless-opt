"""Regression gates for frozen comparators and configured provider timeouts."""

import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from lossless import jobs, providers
from lossless.artifacts import NativeOperation
from lossless._native import engine, operators
from lossless._native.common import write


class FrontierTests(unittest.TestCase):
    def test_configuration_freezes_supported_comparator(self):
        cfg = jobs.template("native.copy")
        self.assertEqual(
            jobs.Workload.from_dict(cfg).resolve()["objective"]["comparator"], "numpy_copy"
        )
        cfg["objective"]["comparator"] = "numpy_buffered"
        with self.assertRaises(ValueError):
            jobs.Workload.from_dict(cfg).resolve()
        cfg["objective"]["comparator"] = "native_baseline"
        self.assertEqual(
            jobs.Workload.from_dict(cfg).resolve()["objective"]["comparator"], "native_baseline"
        )

    def test_native_win_that_loses_to_declared_library_is_not_selected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            spec = {
                "operator": "copy",
                "comparator": "numpy_copy",
                "cases": [
                    {"id": "d", "rows": 3, "columns": 7, "layout": "c", "split": "discovery"},
                    {"id": "e", "rows": 4, "columns": 9, "layout": "c", "split": "evaluation"},
                ],
                "budget_seconds": 30,
                "job_timeout_seconds": 2,
                "compile_timeout_seconds": 2,
                "target_ns": 10000,
                "screening_blocks": 3,
                "confirmation_blocks": 3,
                "max_proposals": 3,
            }
            write(root / "spec.json", spec)
            run = engine.initialize(root / "spec.json", root / "run")
            (root / "kernel.c").write_text(operators.baseline_source("copy"))
            write(
                root / "p.json",
                {
                    "schema_version": 1,
                    "id": "candidate",
                    "operator": "copy",
                    "source_file": "kernel.c",
                    "hypothesis": "test",
                },
            )
            engine.submit(run, root / "p.json")
            data = {
                "candidates": {
                    "candidate": {
                        "complete": True,
                        "rejected": None,
                        "geomean_speedup_vs_native": 2.0,
                        "geomean_speedup_vs_comparator": 0.9,
                    }
                }
            }
            with patch.object(engine, "feedback_data", return_value=data):
                self.assertEqual(engine.seal(run)["global_choice"], "deployment_reference")
            self.assertEqual(
                json.loads((run / "seal.json").read_text())["comparator"], "numpy_copy"
            )
            changed = json.loads((run / "spec.json").read_text())
            changed["comparator"] = "native_baseline"
            write(run / "spec.json", changed)
            with self.assertRaises(ValueError):
                engine.verify(run)

    def test_retained_library_is_the_exported_callable(self):
        operation = NativeOperation.__new__(NativeOperation)
        operation.manifest = {"selected": "reference", "comparator": "numpy_copy"}
        operation.operator = "copy"
        operation.fallback_reason = None
        operation.cases = {(3, 4, "c")}

        def forbidden(*args):
            raise AssertionError("must execute the declared NumPy comparator")

        operation.library = types.SimpleNamespace(kernel=forbidden)
        x = np.arange(12, dtype=np.uint32).reshape(3, 4).view(np.float32)
        bound = operation.bind(x)
        self.assertEqual(bound().tobytes(), x.tobytes())
        self.assertFalse(np.shares_memory(bound(), x))

    def test_timeout_boundaries(self):
        self.assertEqual(providers.bounded_timeout({"timeout_seconds": 120}, 200), 120)
        self.assertEqual(providers.bounded_timeout({"timeout_seconds": 0.2}, 200), 0.2)
        self.assertEqual(providers.bounded_timeout({"timeout_seconds": 120}, 2), 2)
        self.assertEqual(providers.bounded_timeout({}, -1), 0)


if __name__ == "__main__":
    unittest.main()
