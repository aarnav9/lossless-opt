"""Matched arms differ in the declared treatment, not resources or acceptance."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

HERE = Path(__file__).resolve().parents[1] / "experiments/application_intake"


class ChecklistStudyTests(unittest.TestCase):
    def test_only_treatment_differs_between_arm_plans(self):
        sys.path.insert(0, str(HERE))
        try:
            spec = importlib.util.spec_from_file_location(
                "checklist_study_test", HERE / "checklist_study.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            args = SimpleNamespace(
                budget=1800,
                final_reserve=600,
                max_candidates=128,
                model="test-model",
                effort="medium",
            )
            control, assisted = module.plan_for(False, args), module.plan_for(True, args)
            self.assertFalse(control.pop("architecture"))
            self.assertTrue(assisted.pop("architecture"))
            self.assertEqual(control, assisted)
            self.assertEqual(control["wall_time_seconds"], 1800)
            self.assertEqual(control["final_reserve_seconds"], 600)
            self.assertFalse(control["honor_author_stop"])
        finally:
            sys.path.remove(str(HERE))


if __name__ == "__main__":
    unittest.main()
