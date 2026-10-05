"""Protect the common budget and final-evaluation barrier of the model pilot."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from lossless._native.common import read, write

PATH = Path(__file__).resolve().parents[1] / "experiments/models_047/study.py"
loader = importlib.util.spec_from_file_location("model_study", PATH)
study = importlib.util.module_from_spec(loader)
loader.loader.exec_module(study)


class ModelStudyTests(unittest.TestCase):
    def test_comparison_filenames_do_not_collide_on_dotted_model_versions(self):
        labels = [
            study.pair_label(m, ref)
            for m in study.MODELS
            for ref in ["prior", "retained", "enumerated"]
        ]
        paths = [Path(label + "_00").with_suffix(".result.json") for label in labels]
        self.assertEqual(len(paths), len(set(paths)))

    def test_budget_barrier_and_input_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "study"
            study.freeze(out)
            plan = study.verify(out)
            self.assertEqual(plan["search_seconds"], 1800)
            self.assertEqual(plan["proposals_per_call"], 1)
            self.assertEqual(set(plan["order"]), set(study.MODELS))
            self.assertEqual(len(read(out / "enumeration.json")), plan["max_calls"])
            with self.assertRaisesRegex(ValueError, "must be sealed"):
                study.require_sealed(out)
            write(out / "seeds.json", [])
            with self.assertRaisesRegex(AssertionError, "frozen input changed"):
                study.verify(out)

    def test_prompt_withholds_final_cases_and_declares_timing_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "study"
            study.freeze(out)
            run = study.legacy.initialize(out, study.MODELS[0])
            context = study.context(run, study.verify(out), {"rounds": []}, 1000, 940)
            self.assertEqual(context["search_schedule"]["elapsed_seconds"], 800)
            self.assertEqual(context["search_schedule"]["this_response_seconds"], 940)
            self.assertIn("Gaussian", context["timing_workload"]["distribution"])
            self.assertEqual(context["max_proposals"], 1)
            serialized = json.dumps(context)
            for case in read(out / "spec.json")["cases"]:
                if case["split"] == "evaluation":
                    self.assertNotIn(case["id"], serialized)
            self.assertTrue(all(c["split"] == "discovery" for c in context["discovery_cases"]))


if __name__ == "__main__":
    unittest.main()
