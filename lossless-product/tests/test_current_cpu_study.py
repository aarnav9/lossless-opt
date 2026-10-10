"""The application pilot keeps current optimizer defaults and fresh requests."""

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "experiments/application_intake"


class CurrentCpuStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(HERE))
        try:
            spec = importlib.util.spec_from_file_location(
                "current_cpu_study_test", HERE / "current_cpu_study.py"
            )
            cls.study = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.study)
        finally:
            sys.path.remove(str(HERE))

    def test_keeps_product_acceptance_defaults(self):
        args = SimpleNamespace(
            budget=240,
            final_reserve=90,
            max_candidates=2,
            model="test-model",
            effort="medium",
            provider_timeout=45,
        )
        proposed = self.study.plan_for(args)
        engine = importlib.import_module("lossless.applications.search")
        resolved = engine.resolve_plan(proposed, None)
        for name, value in engine.DEFAULTS.items():
            if name not in {"wall_time_seconds", "final_reserve_seconds", "max_candidates"}:
                self.assertEqual(resolved[name], value)
        self.assertTrue(resolved["honor_author_stop"])
        self.assertEqual(resolved["timing_scope"], "calls")
        self.assertNotIn("candidates", proposed)

    def test_fresh_signals_per_request_and_separate_heldout_fixtures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = self.study.workload_cases(root, 10203)
            self.assertEqual([c["split"] for c in cases].count("discovery"), 3)
            self.assertEqual([c["split"] for c in cases].count("evaluation"), 3)
            names = set()
            for case in cases:
                self.assertEqual(len(case["calls"]), 4)
                arrays = []
                for call in case["calls"]:
                    first = call["args"][0][0]
                    path = first["y"]["$npy"]
                    self.assertNotIn(path, names)
                    names.add(path)
                    arrays.append(np.load(root / path))
                self.assertFalse(np.array_equal(arrays[0], arrays[1]))
                self.assertFalse(np.array_equal(arrays[1], arrays[2]))


if __name__ == "__main__":
    unittest.main()
