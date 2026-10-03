"""Profile complete interfaces without providers, holdout access or acceptance."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import lossless
from lossless.jobs import template


class ProfileTests(unittest.TestCase):
    def workload(self, root, adapter="native.copy"):
        cfg = template(adapter, 35)
        cfg["workload"]["cases"] = [
            {"id": "d", "split": "discovery", "rows": 3, "columns": 17, "layout": "f"},
            {"id": "e", "split": "evaluation", "rows": 9, "columns": 23, "layout": "slice2"},
        ]
        cfg["proofs"] = {"check": False}
        return lossless.Workload.from_dict(cfg, base=root)

    def test_profiles_only_discovery_without_provider_or_proof_setup(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            workload = self.workload(root)
            (root / "never.py").write_text(
                "def complete(prompt):\n    raise RuntimeError('provider must not run')\n"
            )
            workload.config["llm"] = {"callback": "never.py:complete", "timeout_seconds": 1}
            workload.config["proofs"] = {"check": True}
            with contextlib.redirect_stdout(io.StringIO()):
                outcome = lossless.profile(workload, repeats=3, output=root / "profile")
            self.assertEqual(outcome.summary["status"], "profiled")
            self.assertEqual(outcome.summary["llm_calls"], 0)
            self.assertEqual([r["case"]["id"] for r in outcome.summary["cases"]], ["d"])
            record = outcome.summary["cases"][0]["implementations"]["reference"]
            self.assertTrue(record["correctness"]["passed"])
            self.assertEqual(len(record["warm"]["samples_seconds"]), 3)
            self.assertGreater(record["bound_warm"]["median_seconds"], 0)
            self.assertTrue(record["host_attribution"]["top_self_time"])
            self.assertNotIn(
                "llm", json.loads((root / "profile/profile_job.json").read_text())["resolved"]
            )
            self.assertFalse((root / "profile/manifest.json").exists())
            self.assertIn("Lossless profile", (root / "profile/report.html").read_text())
            with self.assertRaises(FileExistsError):
                lossless.profile(workload, output=root / "profile")

    def test_numerical_adapters_profile_declared_comparator(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for adapter in ["native.softmax", "native.rmsnorm_residual"]:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = lossless.profile(
                        self.workload(root, adapter), repeats=3, output=root / adapter
                    )
                self.assertEqual(result.summary["comparator"], "numpy_buffered")
                self.assertTrue(
                    result.summary["cases"][0]["implementations"]["reference"]["correctness"][
                        "passed"
                    ]
                )

    def test_exported_comparison_checks_integrity_and_adapter(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            workload = self.workload(root)
            workload.config["budget"]["max_llm_calls"] = 0
            with contextlib.redirect_stdout(io.StringIO()):
                result = lossless.optimize(workload, output=root / "search")
                artifact = result.export(root / "artifact")
                report = lossless.profile(
                    workload, artifact=artifact, repeats=3, output=root / "profile"
                ).summary
            self.assertIn("artifact", report["cases"][0]["implementations"])
            self.assertIn("payback", report["cases"][0])
            with self.assertRaisesRegex(ValueError, "adapters differ"):
                lossless.profile(
                    self.workload(root, "native.softmax"), artifact=artifact, output=root / "bad"
                )
            with (artifact / "implementation.c").open("a") as stream:
                stream.write("\n/* changed */\n")
            with (
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "worker failed"),
            ):
                lossless.profile(workload, artifact=artifact, output=root / "tampered")

    def test_budget_kills_worker_without_success_or_export(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with contextlib.redirect_stdout(io.StringIO()):
                outcome = lossless.profile(self.workload(root), budget=0.001, output=root / "short")
            self.assertEqual(outcome.summary["status"], "incomplete")
            self.assertFalse((root / "short/manifest.json").exists())
            self.assertLess(outcome.summary["wall_time_seconds"], 5)


if __name__ == "__main__":
    unittest.main()
