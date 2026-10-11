"""Independent optimum, untrusted proposal boundaries, and ordinary-call ABI."""

import importlib.util
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "experiments/solver_pipeline"
sys.path.insert(0, str(HERE))
try:
    import pipeline
    import planner

    spec = importlib.util.spec_from_file_location("solver_pipeline_study", HERE / "study.py")
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
finally:
    sys.path.remove(str(HERE))

HAS_SOLVER = importlib.util.find_spec("ortools") is not None


def catalog(seed=1, stages=4):
    rng = random.Random(seed)
    return [
        dict(id=f"{i}_{j}_{backend}", start=i, stop=j, cost_ns=rng.randrange(1, 1000))
        for i in range(stages)
        for j in range(i + 1, stages + 1)
        for backend in range(2)
    ]


def exhaustive(rows, stages, start=0):
    if start == stages:
        return [(0, [])]
    return [
        (row["cost_ns"] + cost, [row["id"]] + plan)
        for row in rows
        if row["start"] == start
        for cost, plan in exhaustive(rows, stages, row["stop"])
    ]


class PlannerTests(unittest.TestCase):
    def test_dynamic_program_matches_exhaustive_enumeration(self):
        for seed in range(12):
            rows = catalog(seed)
            result = planner.dynamic_program(rows, 4)
            self.assertEqual((result["cost_ns"], result["plan"]), min(exhaustive(rows, 4)))

    @unittest.skipUnless(HAS_SOLVER, "optional OR-Tools pilot dependency")
    def test_solver_matches_exhaustive_and_excludes_exact_previous_plans(self):
        for seed in range(4):
            rows = catalog(seed)
            remaining = exhaustive(rows, 4)
            excluded = []
            for _ in range(4):
                result = planner.solve(rows, 4, exclude=excluded)
                self.assertEqual(result["status"], "OPTIMAL")
                self.assertEqual(result["cost_ns"], min(c for c, p in remaining))
                self.assertNotIn(result["plan"], excluded)
                excluded.append(result["plan"])
                remaining = [(c, p) for c, p in remaining if p != result["plan"]]

    def test_rejects_overlap_gaps_reordering_and_forged_costs(self):
        rows = catalog()
        for plan in ([], ["0_2_0", "1_4_0"], ["0_1_0", "2_4_0"], ["2_4_0", "0_2_0"], ["missing"]):
            with self.assertRaises(ValueError):
                planner.validate_plan(plan, rows, 4)
        good = dict(hypothesis="reuse", segments=["0_4_0"])
        self.assertEqual(planner.validate_response(good, rows, 4), rows[6]["cost_ns"])
        with self.assertRaises(ValueError):
            planner.validate_response({**good, "cost_ns": 1}, rows, 4)
        for bad in (0, -1, True, float("nan"), 0.5):
            with self.assertRaises(ValueError):
                planner.validate_catalog([{**rows[0], "cost_ns": bad}], 4)

    def test_certificate_requires_actual_model_optimum(self):
        rows = catalog()
        optimal = planner.dynamic_program(rows, 4)
        source = planner.lean_witness(rows, 4, optimal["plan"])
        self.assertIn("theorem modeled_optimal", source)
        self.assertNotIn("sorry", source)
        worse = max(exhaustive(rows, 4))[1]
        with self.assertRaises(ValueError):
            planner.lean_witness(rows, 4, worse)

    def test_author_context_only_adds_solver_treatment(self):
        rows = catalog()
        inputs = [np.zeros((4, 4), dtype=np.uint8)]
        control = study.request_for(rows, inputs, [], None, 60)
        assisted = study.request_for(rows, inputs, [], {"plan": ["0_4_0"]}, 60)
        assisted.pop("solver_suggestion")
        self.assertEqual(control, assisted)
        self.assertNotIn("evaluation", control)

    def test_large_single_call_regression_vetoes_aggregate_gain(self):
        evidence = dict(
            sequence={"samples_ns": {"reference": [100, 100, 100], "candidate": [60, 60, 60]}},
            per_case=[
                {"median_ns": {"reference": 90, "candidate": 40}},
                {"median_ns": {"reference": 10, "candidate": 20}},
            ],
            traced_peak_bytes={"reference": 100, "candidate": 100},
        )
        result = pipeline.verdict(evidence, "candidate")
        self.assertGreater(result["sequence_speedup"], 1.5)
        self.assertFalse(result["eligible"])


class CompiledPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.operation = pipeline.Pipeline(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_every_region_all_uint8_values_and_ownership(self):
        self.assertEqual(self.operation.validate_regions(), len(pipeline.specs()) * 2)

    def test_identical_plans_share_final_samples(self):
        plan = ["native_0_6"]
        result = pipeline.evaluate(
            self.operation,
            {"reference": pipeline.REFERENCE, "control": plan, "solver": plan},
            [np.arange(256, dtype=np.uint8)],
            repeats=2,
            target_seconds=0.0001,
            seed=1,
        )
        self.assertEqual(result["plan_aliases"]["solver"], "control")
        self.assertEqual(
            result["sequence"]["samples_ns"]["control"], result["sequence"]["samples_ns"]["solver"]
        )

    def test_ordinary_calls_pack_strides_allocate_and_preserve_inputs(self):
        x = np.arange(240, dtype=np.uint8).reshape(10, 24)[:, ::2]
        before = x.copy()
        expected = pipeline.numpy_region(x, 0, pipeline.STAGES)
        for plan in (["native_0_3", "lut_3_6"], ["numpy_0_2", "native_2_6"], ["lut_0_6"]):
            result = self.operation.run(plan, x)
            next_result = self.operation.run(plan, x)
            np.testing.assert_array_equal(result, expected)
            np.testing.assert_array_equal(x, before)
            self.assertTrue(result.flags.c_contiguous)
            self.assertFalse(np.shares_memory(result, x))
            self.assertFalse(np.shares_memory(result, next_result))


if __name__ == "__main__":
    unittest.main()
