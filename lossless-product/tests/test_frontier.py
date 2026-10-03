"""Regression gates for explicit comparators, evidence reuse and deployment overhead."""

import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from lossless import jobs, providers
from lossless.artifacts import NativeOperation
from lossless._native import engine, operators, worker
from lossless._native.common import write
from lossless.economics import payback


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
            duplicate = json.loads((root / "p.json").read_text())
            duplicate["id"] = "same_source"
            write(root / "p.json", duplicate)
            engine.submit(run, root / "p.json")
            state = engine.verify(run)
            self.assertEqual(
                state["proposals"]["same_source"]["rejected"],
                {"reason": "duplicate_source", "of": "candidate"},
            )
            self.assertFalse(
                any("same_source" in name for name, _ in engine.job_specs(run, state, "discovery"))
            )
            duplicate["id"] = "deployment_reference"
            write(root / "p.json", duplicate)
            with self.assertRaises(ValueError):
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

    def test_compiled_cache_hits_invalidates_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "source.c"
            source.write_text(operators.baseline_source("copy"))
            job = {
                "source": str(source),
                "binary": str(root / "a.dylib"),
                "cache": {"directory": str(root / "cache")},
                "cache_context": {"compiler": "test1"},
            }
            self.assertFalse(worker.compile_job(job)["cache_hit"])
            job["binary"] = str(root / "b.dylib")
            self.assertTrue(worker.compile_job(job)["cache_hit"])
            next((root / "cache/compiled").glob("*/binary")).write_bytes(b"broken")
            self.assertFalse(worker.compile_job(job)["cache_hit"])
            job["cache_context"]["compiler"] = "test2"
            self.assertFalse(worker.compile_job(job)["cache_hit"])

    def test_validation_cache_never_reuses_timings_or_skips_fresh_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "source.c"
            source.write_text(operators.baseline_source("copy"))
            binary = root / ("a.dylib" if __import__("sys").platform == "darwin" else "a.so")
            worker.compile_job({"source": str(source), "binary": str(binary)})
            job = {
                "operator": "copy",
                "comparator": "numpy_copy",
                "case": {"rows": 3, "columns": 7, "layout": "c"},
                "baseline": str(binary),
                "candidate": str(binary),
                "seed": 32,
                "candidate_seed": 0,
                "screening_blocks": 3,
                "confirmation_blocks": 3,
                "target_ns": 10000,
                "reuse_validation": True,
                "cache": {"directory": str(root / "cache")},
                "cache_context": {"contract": "copy1"},
            }
            first = worker.benchmark_job(job)
            second = worker.benchmark_job(job)
            self.assertFalse(first["validation_cache_hit"])
            self.assertTrue(second["validation_cache_hit"])
            self.assertNotEqual(
                first["confirmation"]["samples_us"], second["confirmation"]["samples_us"]
            )
            job["reuse_validation"] = False
            self.assertFalse(worker.benchmark_job(job)["validation_cache_hit"])
            job["reuse_validation"] = True
            job["seed"] += 1
            self.assertFalse(worker.benchmark_job(job)["validation_cache_hit"])

    def test_timeout_and_payback_boundaries(self):
        self.assertEqual(providers.bounded_timeout({"timeout_seconds": 120}, 200), 120)
        self.assertEqual(providers.bounded_timeout({"timeout_seconds": 0.2}, 200), 0.2)
        self.assertEqual(providers.bounded_timeout({"timeout_seconds": 120}, 2), 2)
        self.assertEqual(providers.bounded_timeout({}, -1), 0)
        self.assertEqual(payback(10, 2, 1, 3)["break_even_calls"], 13)
        self.assertIsNone(payback(10, 1, 2, 0)["break_even_calls"])
        self.assertIsNone(payback(10, 2, 1)["break_even_calls"])

    def test_admission_reuses_identity_only_within_explicit_scope(self):
        from lossless.adapters.mlx import runtime

        expected = json.loads(Path(runtime.__file__).with_name("model_identity.json").read_text())
        with patch(
            "lossless.adapters.mlx.identity.fingerprint", side_effect=AssertionError("second read")
        ):
            self.assertTrue(
                runtime.admission(
                    "unused",
                    "Apple M2",
                    {"mlx": "0.32.3", "mlx-lm": "0.32.0"},
                    model_identity=expected,
                )[0]
            )
            self.assertFalse(
                runtime.admission(
                    "unused", "Apple M2", {"mlx": "0.32.3", "mlx-lm": "0.32.0"}, model_identity={}
                )[0]
            )


if __name__ == "__main__":
    unittest.main()
