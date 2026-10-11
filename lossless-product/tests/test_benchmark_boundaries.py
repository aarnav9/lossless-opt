"""Catch kernel wins that disappear at deployment and hidden request regressions."""

import copy
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from lossless.artifacts import NativeOperation
from lossless._native import operators, worker
from lossless.applications.comparison import eligible, statistics_for
from lossless.jobs import Workload, template


class BenchmarkBoundaryTests(unittest.TestCase):
    def test_native_public_default_and_explicit_reuse_are_frozen(self):
        config = template("native.softmax")
        self.assertEqual(Workload.from_dict(config).resolve()["objective"]["timing_scope"], "call")
        self.assertNotIn("timing_scope", config["objective"])
        config["objective"]["timing_scope"] = "bound"
        self.assertEqual(Workload.from_dict(config).resolve()["objective"]["timing_scope"], "bound")
        config["objective"]["timing_scope"] = "kernel_claimed_as_application"
        with self.assertRaises(ValueError):
            Workload.from_dict(config).resolve()

    @unittest.skipUnless(shutil.which("clang"), "clang required")
    def test_ordinary_timing_contains_binding_cost_and_keeps_kernel_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "copy.c"
            source.write_text(operators.baseline_source("copy"))
            binary = root / ("copy.dylib" if sys.platform == "darwin" else "copy.so")
            self.assertEqual(
                worker.compile_job({"source": source, "binary": binary})["status"], "ok"
            )
            job = {
                "operator": "copy",
                "comparator": "numpy_copy",
                "timing_scope": "call",
                "case": {"rows": 3, "columns": 7, "layout": "c"},
                "baseline": binary,
                "candidate": binary,
                "seed": 32,
                "candidate_seed": 0,
                "screening_blocks": 3,
                "confirmation_blocks": 3,
                "target_ns": 10000,
            }
            bind = NativeOperation.bind

            def costly_bind(operation, *args):
                time.sleep(0.002)
                return bind(operation, *args)

            with patch.object(NativeOperation, "bind", costly_bind):
                record = worker.benchmark_job(job)
            self.assertEqual(record["status"], "ok")
            self.assertGreater(record["confirmation"]["median_us"]["proposal"], 1500)
            self.assertGreater(
                record["confirmation"]["median_us"]["proposal"],
                record["kernel_only"]["confirmation"]["median_us"]["proposal"],
            )
            self.assertTrue(
                record["validation"]["proposal"]["special_bits"]["ordinary_call"]["passed"]
            )

    def test_library_deployment_does_not_allocate_benchmark_red_zones(self):
        operation = NativeOperation.from_loaded_library(
            "copy", object(), [{"rows": 3, "columns": 4, "layout": "c"}], comparator="numpy_copy"
        )
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        with patch.object(operators, "executors", side_effect=AssertionError("benchmark buffers")):
            a, b = operation(x), operation(x)
        self.assertTrue(np.array_equal(a, x))
        self.assertFalse(np.shares_memory(a, b))
        self.assertFalse(np.shares_memory(a, x))

    def pairs(self, reference, candidate, reference_setup=0.1, candidate_setup=0.1):
        def row(calls, setup):
            return {
                "sequence_seconds": sum(calls),
                "setup_seconds": setup,
                "setup_and_sequence_seconds": setup + sum(calls),
                "call_seconds": calls,
                "first_call_seconds": calls[0],
                "process_seconds": 1,
                "peak_rss_bytes": 100,
            }

        return [
            {
                "reference": row(reference, reference_setup),
                "candidate": row(candidate, candidate_setup),
            }
            for _ in range(6)
        ]

    def test_mixed_request_regression_cannot_hide_behind_twofold_sequence_win(self):
        # Full-batch work improves, but a different request pays many small-copy costs.
        stats = statistics_for(
            [{"id": "mixed", "pairs": self.pairs([0.01, 0.00004], [0.005, 0.00029])}], 1.02
        )
        policy = {
            "minimum_speedup": 1.02,
            "max_case_regression": 0.03,
            "max_call_regression": 0.03,
            "significance": 0.05,
        }
        self.assertGreater(stats["case_balanced_geomean_speedup"], 1.8)
        self.assertLess(stats["cases"][0]["calls"][1]["speedup"], 0.14)
        self.assertFalse(eligible({"status": "matched", "statistics": stats}, policy, final=True))
        missing = copy.deepcopy(stats)
        missing["cases"][0]["calls"] = None
        self.assertFalse(eligible({"status": "matched", "statistics": missing}, policy, final=True))

    def test_startup_can_reverse_a_one_shot_gain(self):
        rows = [{"id": "startup", "pairs": self.pairs([1.0], [0.5], 0.1, 1.0)}]
        self.assertEqual(statistics_for(rows, 1.02)["case_balanced_geomean_speedup"], 2)
        stats = statistics_for(rows, 1.02, "setup_and_calls")
        self.assertAlmostEqual(stats["case_balanced_geomean_speedup"], 1.1 / 1.5)
        self.assertEqual(stats["timing_metric"], "setup_and_sequence_seconds")

    def test_mismatched_call_counts_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "per-call"):
            statistics_for([{"id": "missing", "pairs": self.pairs([1, 2], [1])}], 1.02)


if __name__ == "__main__":
    unittest.main()
