"""Protect scoped qualification decisions from averages hiding repeat regressions."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

from lossless._native.common import write

SOURCE = Path(__file__).resolve().parents[1] / "experiments/softmax_046/qualify.py"
spec = importlib.util.spec_from_file_location("qualify_046", SOURCE)
qualify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualify)


class SoftmaxQualificationTests(unittest.TestCase):
    def test_fresh_shapes_and_guard_boundaries(self):
        self.assertFalse(set(qualify.FINAL_SHAPES) & set(qualify.old.SHAPES))
        self.assertFalse(qualify.in_scope({"rows": 7, "columns": 255, "layout": "f"}, "large"))
        self.assertTrue(qualify.in_scope({"rows": 32, "columns": 512, "layout": "f"}, "large_f"))
        self.assertFalse(qualify.in_scope({"rows": 32, "columns": 512, "layout": "c"}, "large_f"))

    def test_repeated_regression_rejects_even_with_large_geomean(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for process in range(3):
                folder = out / f"discovery_{process}"
                folder.mkdir()
                timings = [
                    {
                        "interface": interface,
                        "distribution": distribution,
                        "comparisons": {"candidate": {"speedup": speed, "ci95": interval}},
                    }
                    for interface in ["bound", "ordinary"]
                    for distribution, speed, interval in [
                        ("gaussian", 0.90, [0.88, 0.92]),
                        ("constant", 4, [3, 5]),
                    ]
                ]
                write(
                    folder / "case_000.json",
                    {
                        "case": {"rows": 64, "columns": 512, "layout": "c"},
                        "correct": True,
                        "timings": timings,
                    },
                )
            result = qualify.assess(out, "discovery", "candidate", "large")
            self.assertTrue(all(v > 1.02 for row in result["geomeans"] for v in row.values()))
            self.assertFalse(result["passed"])
            self.assertEqual(len(result["repeated_regressions"]), 2)

    def test_source_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "study"
            qualify.freeze(out)
            qualify.verify(out)
            (out / "sources/final_045.c").write_text("changed")
            with self.assertRaisesRegex(AssertionError, "source changed"):
                qualify.verify(out)


if __name__ == "__main__":
    unittest.main()
