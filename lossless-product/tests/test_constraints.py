import copy
import json
from pathlib import Path
import tempfile
import unittest

from lossless.constraints import assess, measurements
from lossless.jobs import Workload, template, identity


class DeploymentConstraintTests(unittest.TestCase):
    def metrics(self):
        rows = [
            dict(tokens_per_second=100, ttft_seconds=[0.01, 0.2], completion_seconds=[0.03, 0.4])
            for _ in range(20)
        ]
        return measurements(rows, [1000] * 20)

    def test_worst_request_not_pooled_percentile(self):
        rows = [
            dict(
                tokens_per_second=100,
                ttft_seconds=[0.01] * 63 + [0.5],
                completion_seconds=[0.1] * 63 + [1.0],
            )
            for _ in range(20)
        ]
        metrics = measurements(rows, [1000] * 20)
        self.assertEqual(metrics["worst_request_p95_ttft_seconds"], 0.5)
        self.assertEqual(metrics["worst_request_p95_completion_seconds"], 1.0)

    def test_fast_candidate_can_fail_each_deployment_limit(self):
        limits = dict(
            max_p95_ttft_seconds=0.1,
            max_p95_completion_seconds=0.3,
            max_peak_mlx_bytes=999,
            max_break_even_calls=2,
            min_tokens_per_second=101,
        )
        outcome = assess(
            limits,
            self.metrics(),
            search_seconds=10,
            reference_seconds=2,
            candidate_seconds=1,
            extra_setup_seconds=1,
        )
        self.assertFalse(outcome["passed"])
        self.assertTrue(all(not c["passed"] for c in outcome["checks"].values()))
        self.assertEqual(outcome["payback"]["break_even_calls"], 11)

    def test_equal_limit_passes_and_final_search_cost_can_reject(self):
        limits = dict(
            max_p95_ttft_seconds=0.2,
            max_p95_completion_seconds=0.4,
            max_peak_mlx_bytes=1000,
            max_break_even_calls=11,
            min_tokens_per_second=100,
        )
        kwargs = dict(reference_seconds=2, candidate_seconds=1, extra_setup_seconds=1)
        self.assertTrue(assess(limits, self.metrics(), search_seconds=10, **kwargs)["passed"])
        self.assertFalse(assess(limits, self.metrics(), search_seconds=11, **kwargs)["passed"])

    def test_missing_measurements_and_nonpositive_savings_fail_closed(self):
        limits = dict(max_break_even_calls=100, max_p95_ttft_seconds=1)
        outcome = assess(
            limits,
            {},
            search_seconds=1,
            reference_seconds=1,
            candidate_seconds=2,
            extra_setup_seconds=0,
        )
        self.assertFalse(outcome["passed"])
        self.assertTrue(all(not c["passed"] for c in outcome["checks"].values()))

    def test_resolution_freezes_limits_and_rejects_unsupported_or_under_sampled(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model").mkdir()
            (root / "requests.json").write_text(
                json.dumps(
                    [
                        dict(text="discovery", split="discovery", max_tokens=2),
                        dict(text="evaluation", split="evaluation", max_tokens=2),
                    ]
                )
            )
            cfg = dict(
                schema_version=1,
                workload=dict(
                    adapter="mlx.fixed_count",
                    parameters=dict(model="model", fixed_count=True, repeats=20),
                    inputs="requests.json",
                ),
                contract=dict(preset="exact"),
                budget=dict(wall_time_seconds=60),
                objective=dict(metric="throughput", constraints=dict(max_p95_ttft_seconds=0.2)),
            )
            original = Workload.from_dict(cfg, base=root).resolve()
            changed = copy.deepcopy(cfg)
            changed["objective"]["constraints"]["max_p95_ttft_seconds"] = 0.3
            self.assertNotEqual(
                identity(original), identity(Workload.from_dict(changed, base=root).resolve())
            )
            changed["workload"]["parameters"]["repeats"] = 7
            with self.assertRaisesRegex(ValueError, "repeats >= 20"):
                Workload.from_dict(changed, base=root).resolve()
            for value in [True, 0, -1, float("nan")]:
                changed = copy.deepcopy(cfg)
                changed["objective"]["constraints"]["max_p95_ttft_seconds"] = value
                with self.assertRaises(ValueError):
                    Workload.from_dict(changed, base=root).resolve()
            cfg = template()
            cfg["objective"]["constraints"] = {"max_peak_mlx_bytes": 1000}
            with self.assertRaisesRegex(ValueError, "require mlx"):
                Workload.from_dict(cfg, base=root).resolve()


if __name__ == "__main__":
    unittest.main()
