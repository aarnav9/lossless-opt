"""Candidate authors cannot bypass the frozen application evaluator."""

import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from lossless import codex_provider
from lossless.applications import initialize, search
from lossless.applications import patches
from lossless.applications.comparison import eligible, statistics_for
from lossless.applications.search import author_history, resolve_plan
from lossless.artifacts import export


class ApplicationSearchTests(unittest.TestCase):
    def test_public_search_remains_callable_after_loading_its_module(self):
        import importlib
        import lossless.applications as applications

        module = importlib.import_module("lossless.applications.search")
        self.assertIs(applications.search, module.search)
        self.assertTrue(callable(applications.search))

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.source = 'import time\ndef run(n):\n    """Keep this docstring."""\n    time.sleep(.02)\n    return n + 1\n\ndef untouched():\n    return 42\n'
        (self.project / "app.py").write_text(self.source)
        cases = self.root / "cases.json"
        cases.write_text(
            json.dumps(
                [
                    {"id": "visible", "split": "discovery", "args": [3]},
                    {"id": "secret_evaluation", "split": "evaluation", "args": [92837465]},
                ]
            )
        )
        initialize(self.project, self.root / "job", entry="app.py:run", cases=cases)
        self.config = self.root / "job/lossless.json"
        self.scope = [{"path": "app.py", "symbol": "run"}]
        self.plan = {
            "schema_version": 1,
            "source_scope": self.scope,
            "contract": "Pure integer n -> n + 1. Delay is not observable; full callable latency is.",
            "wall_time_seconds": 90,
            "final_reserve_seconds": 45,
            "max_candidates": 1,
            "discovery_pairs": 2,
            "evaluation_pairs": 6,
            "max_setup_increase_seconds": 1,
            "max_peak_rss_ratio": 2,
            "max_traced_peak_ratio": 10,
        }

    def proposal(self, body="return n + 1\n"):
        return {
            "schema_version": 1,
            "hypothesis": "Remove unobserved delay",
            "edits": [{**self.scope[0], "body": body}],
        }

    def fake(self, proposal, requests):
        def invoke(request, folder, **kwargs):
            requests.append(copy.deepcopy(request))
            kwargs["validator"](proposal)
            folder.mkdir(parents=True)
            (folder / "response.json").write_text(json.dumps(proposal))
            return {"eligible": True, "elapsed_seconds": 0.01, "timed_out": False}

        return invoke

    def run_search(self, proposal):
        requests = []
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(codex_provider, "require_chatgpt"),
            patch.object(codex_provider, "invoke", side_effect=self.fake(proposal, requests)),
        ):
            result = search(self.config, plan=self.plan, output=self.root / "run")
        return result, requests

    def test_end_to_end_accepts_only_after_closed_evaluation(self):
        result, requests = self.run_search(self.proposal())
        self.assertEqual(result.summary["status"], "accepted_experiment", result.summary)
        self.assertEqual(result.summary["evaluation"]["status"], "matched")
        self.assertEqual(len(requests), 1)
        self.assertNotIn("92837465", json.dumps(requests))
        self.assertNotIn("secret_evaluation", json.dumps(requests))
        self.assertTrue((result.directory / "selected.patch").exists())
        self.assertEqual((self.project / "app.py").read_text(), self.source)
        self.assertTrue(result.summary["runner_unchanged"])
        with self.assertRaisesRegex(ValueError, "deployment artifact"):
            export(result.directory, self.root / "deployment")

    def test_discovery_mismatch_keeps_evaluation_closed(self):
        result, _ = self.run_search(self.proposal("return n + 2\n"))
        self.assertEqual(result.summary["status"], "reference_retained", result.summary)
        self.assertEqual(result.summary["candidates"][0]["status"], "mismatch")
        self.assertFalse((result.directory / "evaluation").exists())
        self.assertFalse((result.directory / "selected.patch").exists())

    def test_saved_candidates_do_not_call_a_provider(self):
        self.plan["candidates"] = [self.proposal()]
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(codex_provider, "invoke", side_effect=AssertionError("no provider")),
            patch(
                "lossless.applications.search.context_for",
                return_value={"task": "", "support": "", "unused_large_evidence": "x" * 300000},
            ),
            patch.object(codex_provider, "require_chatgpt", side_effect=AssertionError("no auth")),
        ):
            result = search(self.config, plan=self.plan, output=self.root / "saved")
        self.assertEqual(result.summary["status"], "accepted_experiment", result.summary)
        self.assertEqual(result.summary["llm_calls"], 0)
        self.assertEqual(result.summary["provider_seconds"], 0)

    def test_reusing_previous_output_buffer_is_rejected(self):
        (self.project / "app.py").write_text(
            "import numpy as np\ndef run(n):\n    return np.array([n])\n"
        )
        (self.root / "job/cases.json").write_text(
            json.dumps(
                [
                    {
                        "id": "visible",
                        "split": "discovery",
                        "calls": [{"args": [3]}, {"args": [4]}],
                    },
                    {"id": "closed", "split": "evaluation", "args": [9]},
                ]
            )
        )
        result, _ = self.run_search(
            self.proposal(
                "if not hasattr(run, 'buffer'):\n    run.buffer = np.empty(1, dtype=np.int64)\nrun.buffer[0] = n\nreturn run.buffer\n"
            )
        )
        self.assertEqual(result.summary["status"], "reference_retained", result.summary)
        discovery = result.summary["candidates"][0]["discovery"]
        self.assertEqual(discovery["status"], "mismatch")
        self.assertEqual(discovery["failed_calls"], [])
        self.assertFalse(discovery["retained_outputs_match"])

    def test_overfit_candidate_fails_final_without_another_author(self):
        result, requests = self.run_search(self.proposal("return 4\n"))
        self.assertEqual(result.summary["status"], "reference_retained", result.summary)
        self.assertEqual(result.summary["evaluation"]["status"], "mismatch")
        self.assertEqual(len(requests), 1)
        self.assertFalse((result.directory / "selected.patch").exists())

    def test_body_edits_keep_signature_docstrings_and_other_functions(self):
        destination = self.root / "candidate"
        shutil.copytree(self.project, destination)
        diff, identity = patches.apply(self.project, destination, self.proposal(), self.scope)
        self.assertIn("-    time.sleep", diff)
        changed = (destination / "app.py").read_text()
        self.assertIn('"""Keep this docstring."""', changed)
        self.assertIn("def untouched():\n    return 42", changed)
        proposal = self.proposal("# Equivalent comment\nreturn n+1\n")
        self.assertEqual(
            patches.apply(self.project, destination, proposal, self.scope)[1], identity
        )
        for body in ("global x\nx = 2\n", "return (\n"):
            with self.assertRaises(ValueError):
                patches.validate(self.proposal(body), self.scope)
        bad = self.proposal()
        bad["edits"][0]["path"] = "../app.py"
        with self.assertRaises(ValueError):
            patches.validate(bad, self.scope)
        bad = self.proposal()
        bad["accepted"] = True
        with self.assertRaises(ValueError):
            patches.validate(bad, self.scope)

    def test_final_sign_gate_and_regression_veto(self):
        def samples(ref, candidate):
            return [
                {
                    "reference": dict(
                        sequence_seconds=ref,
                        setup_seconds=1,
                        first_call_seconds=ref,
                        process_seconds=1,
                        peak_rss_bytes=100,
                    ),
                    "candidate": dict(
                        sequence_seconds=candidate,
                        setup_seconds=1,
                        first_call_seconds=candidate,
                        process_seconds=1,
                        peak_rss_bytes=100,
                    ),
                }
                for _ in range(6)
            ]

        policy = resolve_plan(self.plan, None)
        good = statistics_for([{"id": "fast", "pairs": samples(2, 1)}], policy["minimum_speedup"])
        self.assertEqual(good["sign_test_p"], 1 / 64)
        self.assertTrue(eligible({"status": "matched", "statistics": good}, policy, final=True))
        regression = statistics_for(
            [{"id": "fast", "pairs": samples(2, 1)}, {"id": "slow", "pairs": samples(1, 1.5)}],
            policy["minimum_speedup"],
        )
        self.assertFalse(
            eligible({"status": "matched", "statistics": regression}, policy, final=True)
        )

    def test_budget_and_missing_evaluation_fail_before_provider(self):
        with self.assertRaises(ValueError):
            resolve_plan(self.plan, 1)
        (self.root / "job/cases.json").write_text(
            '[{"id":"sample","split":"discovery","args":[1]}]'
        )
        with patch.object(codex_provider, "invoke", side_effect=AssertionError("not authorized")):
            with self.assertRaisesRegex(ValueError, "evaluation"):
                search(self.config, plan=self.plan, output=self.root / "run")

    def test_long_budget_history_retains_best_and_recent_code(self):
        history = [
            {
                "id": f"candidate_{i}",
                "proposal": self.proposal(f"return n + {i}"),
                "result": {
                    "status": "matched",
                    "statistics": {"case_balanced_geomean_speedup": 1 + i / 100},
                },
            }
            for i in range(25)
        ]
        bounded = author_history(history, "candidate_3")
        self.assertIn("proposal", bounded[3])
        self.assertTrue(all("proposal" in row for row in bounded[-6:]))
        self.assertNotIn("proposal", bounded[4])
        self.assertEqual(bounded[4]["discovery_speedup"], 1.04)
        self.assertIn("proposal", history[4])
        self.plan.update(max_candidates=128, honor_author_stop=False)
        self.assertFalse(resolve_plan(self.plan, None)["honor_author_stop"])
        self.plan["honor_author_stop"] = 1
        with self.assertRaisesRegex(ValueError, "boolean"):
            resolve_plan(self.plan, None)

    def test_sustained_search_continues_after_author_abstains(self):
        requests = []
        count = 0
        abstention = {"schema_version": 1, "hypothesis": "No proposal yet", "edits": []}

        def invoke(request, folder, **kwargs):
            nonlocal count
            count += 1
            return self.fake(abstention if count == 1 else self.proposal(), requests)(
                request, folder, **kwargs
            )

        self.plan.update(max_candidates=2, honor_author_stop=False)
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(codex_provider, "require_chatgpt"),
            patch.object(codex_provider, "invoke", side_effect=invoke),
        ):
            result = search(self.config, plan=self.plan, output=self.root / "sustained")
        self.assertEqual(result.summary["llm_calls"], 2)
        self.assertEqual(result.summary["candidates"][0]["status"], "author_abstained")
        self.assertEqual(result.summary["status"], "accepted_experiment", result.summary)
        self.assertEqual(result.summary["search_stop_reason"], "candidate_cap")

    def test_provider_outage_keeps_final_validation_independent(self):
        requests = []
        count = 0

        def invoke(request, folder, **kwargs):
            nonlocal count
            count += 1
            if count == 1:
                return self.fake(self.proposal(), requests)(request, folder, **kwargs)
            requests.append(copy.deepcopy(request))
            folder.mkdir(parents=True)
            return {
                "eligible": False,
                "elapsed_seconds": 0.01,
                "errors": ["no response"],
                "provider_messages": ["usage limit"],
            }

        self.plan.update(max_candidates=128, honor_author_stop=False)
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(codex_provider, "require_chatgpt"),
            patch.object(codex_provider, "invoke", side_effect=invoke),
        ):
            result = search(self.config, plan=self.plan, output=self.root / "interrupted")
        self.assertEqual(result.summary["llm_calls"], 4)
        self.assertEqual(result.summary["search_stop_reason"], "provider_failures")
        self.assertEqual(
            result.summary["authoring_interrupted"]["provider_messages"], ["usage limit"]
        )
        self.assertEqual(result.summary["status"], "accepted_experiment", result.summary)
        self.assertNotIn("92837465", json.dumps(requests))


if __name__ == "__main__":
    unittest.main()
