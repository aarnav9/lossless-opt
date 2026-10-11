"""Evaluator controls for the sequential CPU campaign."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "experiments/cpu_100"


class CpuCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(HERE))
        spec = importlib.util.spec_from_file_location("cpu_campaign_study", HERE / "study.py")
        cls.study = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.study)

    def compare(self, x, y, ownership=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "a.npz", result=x)
            np.savez(root / "b.npz", result=y)
            a = dict(structure={}, inputs_unchanged=True, retained_unchanged=True)
            b = {**a, "inputs_unchanged": ownership}
            return self.study.check_pair(a, b, root / "a.npz", root / "b.npz")

    def test_exact_numerical_wrong_and_mutation_are_distinct(self):
        x = np.array([1.0, 2.0])
        self.assertTrue(self.compare(x, x.copy())["exact"])
        small = self.compare(x, x + 1e-11)
        self.assertFalse(small["exact"])
        self.assertTrue(small["numerical"])
        self.assertFalse(self.compare(x, x + 0.1)["numerical"])
        self.assertFalse(self.compare(x, x, False)["numerical"])

    def test_signed_zero_and_changed_history_shape_cannot_be_exact(self):
        self.assertFalse(self.compare(np.array([0.0]), np.array([-0.0]))["exact"])
        self.assertFalse(self.compare(np.zeros(3), np.zeros(4))["numerical"])

    def test_total_win_cannot_hide_slow_request(self):
        from common import score, verdict

        pairs = [
            {"reference": {"calls": [1.0, 0.01]}, "candidate": {"calls": [0.5, 0.02]}}
            for _ in range(16)
        ]
        result = score([{"id": "x", "pairs": pairs}])
        self.assertGreater(result["speedup"], 1.9)
        self.assertFalse(verdict({"exact": True}, result, True)["eligible"])

    def test_identical_timings_cannot_qualify(self):
        from common import score, verdict

        pairs = [
            {"reference": {"calls": [1.0, 2.0]}, "candidate": {"calls": [1.0, 2.0]}}
            for _ in range(16)
        ]
        result = score([{"id": "x", "pairs": pairs}])
        self.assertFalse(verdict({"exact": True}, result, True)["eligible"])


if __name__ == "__main__":
    unittest.main()
