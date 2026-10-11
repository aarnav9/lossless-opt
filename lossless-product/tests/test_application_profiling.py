"""Measure declared applications; model advice cannot invent evidence or grant acceptance."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import lossless
from lossless import codex_provider
from lossless.applications import initialize
from lossless.applications.assessment import context_for, validate
from lossless.artifacts import export
from lossless.cli import main


class ApplicationProfileTests(unittest.TestCase):
    def test_memory_context_stays_bounded_for_long_sequences(self):
        summary = {
            "measurement_scope": "test",
            "environment": {"python": "3", "platform": "test", "machine": "test"},
            "profiles": [
                {
                    "id": "sequence",
                    "timing": {"median_sequence_seconds": 1},
                    "setup": {},
                    "memory": {
                        "process_peak_rss_bytes": 100,
                        "scope": "traced",
                        "traced_calls": [
                            {"peak_bytes_above_start": i, "top_retained": []} for i in range(100)
                        ],
                    },
                    "attribution": {"top_self": []},
                }
            ],
        }
        context = context_for(summary, Path("/unused"))
        memory = context["evidence"]["sequence:memory"]
        self.assertLessEqual(len(memory["traced_calls"]), 2)
        self.assertEqual(memory["traced_call_positions"], [0, 99])
        self.assertEqual(memory["traced_peak_range_bytes"], [0, 99])
        self.assertEqual(memory["omitted_traced_calls"], 98)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.cases = self.root / "cases.json"
        self.cases.write_text(
            json.dumps(
                [
                    {"id": "sample", "split": "discovery", "args": [1000]},
                    {"id": "secret_evaluation", "split": "evaluation", "args": [987654321]},
                ]
            )
        )

    def make(
        self,
        source="import numpy as np\ndef run(n): return np.arange(n, dtype=np.float64)\n",
        factory=False,
    ):
        (self.project / "app.py").write_text(source)
        initialize(
            self.project, self.root / "job", entry="app.py:run", cases=self.cases, factory=factory
        )
        self.config = self.root / "job/lossless.json"

    def profile(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return lossless.profile(self.config, repeats=3, output=self.root / "profile", **kwargs)

    def response(self, evidence_id="sample:timing"):
        return {
            "schema_version": 1,
            "summary": "Inspect allocation costs.",
            "opportunities": [
                {
                    "evidence_ids": [evidence_id],
                    "hypothesis": "Repeated allocation may matter.",
                    "proposed_change": "Measure a caller-owned reusable output buffer.",
                    "validation": "Preserve ownership and compare full calls on fresh inputs.",
                    "confidence": "low",
                }
            ],
            "limitations": ["No optimization has been benchmarked."],
        }

    def fake_command(self, response):
        def command(folder, empty, model, effort):
            script = "import sys,pathlib;sys.stdin.read();pathlib.Path(sys.argv[1]).write_text(sys.argv[2]);print(sys.argv[3],flush=True)"
            return [
                sys.executable,
                "-u",
                "-c",
                script,
                str(folder / "raw_response.json"),
                json.dumps(response),
                json.dumps(
                    {"type": "turn.completed", "usage": {"input_tokens": 20, "output_tokens": 40}}
                ),
            ]

        return command

    def test_clean_and_instrumented_passes_close_evaluation_and_cannot_export(self):
        self.make()
        with patch.object(codex_provider, "invoke", side_effect=AssertionError("no provider")):
            result = self.profile()
        summary = result.summary
        self.assertEqual(summary["status"], "profiled", summary)
        self.assertEqual(summary["llm_calls"], 0)
        self.assertEqual([c["id"] for c in summary["profiles"]], ["sample"])
        case = summary["profiles"][0]
        self.assertEqual(len(case["timing"]["sequence_samples_seconds"]), 3)
        self.assertEqual(len(summary["cases"][0]["attempts"]), 5)
        self.assertGreater(case["memory"]["process_peak_rss_bytes"], 0)
        self.assertGreater(case["memory"]["traced_calls"][0]["peak_bytes_above_start"], 7000)
        self.assertTrue(case["attribution"]["top_self"])
        self.assertTrue(summary["source_unchanged"])
        context = context_for(summary, result.directory / "reference")
        self.assertNotIn("secret_evaluation", json.dumps(context))
        self.assertNotIn("987654321", json.dumps(context))
        with self.assertRaisesRegex(ValueError, "deployment artifact"):
            export(result.directory, self.root / "artifact")

    def test_declared_stateful_sequence_has_no_hidden_warmups(self):
        self.cases.write_text(
            '[{"id":"sample","split":"discovery","calls":[{"args":[1]},{"args":[2]}]}]'
        )
        self.make(
            "class Counter:\n def __init__(self): self.n=0\n def __call__(self,x):\n  self.n+=1\n  return self.n+x\ndef run(): return Counter()\n",
            factory=True,
        )
        result = self.profile()
        self.assertEqual(result.summary["status"], "profiled", result.summary)
        for attempt in result.summary["cases"][0]["attempts"]:
            calls = json.loads(
                (result.directory / attempt["directory"] / "worker_result.json").read_text()
            )["calls"]
            self.assertEqual([c["observation"]["output"]["value"] for c in calls], [2, 4])
        self.assertEqual(result.summary["profiles"][0]["timing"]["calls_per_sequence"], 2)

    def test_observer_cost_is_outside_timing(self):
        self.make(
            "import time\ndef run(n): return n\ndef observe(target,result,args,kwargs):\n time.sleep(.15)\n return result\n"
        )
        value = json.loads(self.config.read_text())
        value["entry"]["observe"] = "app.py:observe"
        self.config.write_text(json.dumps(value))
        result = self.profile()
        self.assertEqual(result.summary["status"], "profiled")
        self.assertLess(result.summary["profiles"][0]["timing"]["median_sequence_seconds"], 0.05)

    def test_completion_wait_is_inside_call_timing(self):
        self.make(
            "import time\npending=False\ndef run(n):\n global pending\n pending=True\n return n\ndef synchronize():\n global pending\n if pending:\n  time.sleep(.03)\n  pending=False\n"
        )
        value = json.loads(self.config.read_text())
        value["entry"]["synchronize"] = "app.py:synchronize"
        self.config.write_text(json.dumps(value))
        result = self.profile()
        self.assertEqual(result.summary["status"], "profiled")
        case = result.summary["profiles"][0]
        self.assertGreaterEqual(case["timing"]["first_call_median_seconds"], 0.025)
        self.assertEqual(case["synchronization"], ["explicit hook"])

    def test_bad_reference_prevents_assessment_and_retains_failure(self):
        self.make("import os\ndef run(n): return os.urandom(16)\n")
        with patch.object(
            codex_provider, "require_chatgpt", side_effect=AssertionError("must not call")
        ):
            result = self.profile(analysis={"provider": "codex-chatgpt"})
        self.assertEqual(result.summary["status"], "reference_unstable")
        self.assertEqual(result.summary["assessment"]["status"], "not_run")
        self.assertEqual(result.summary["llm_calls"], 0)

    def test_budget_stops_before_application_execution(self):
        marker = self.root / "executed"
        self.make(
            f"from pathlib import Path\ndef run(n):\n Path({str(marker)!r}).touch()\n return n\n"
        )
        result = self.profile(budget=0.001, analysis={"provider": "codex-chatgpt"})
        self.assertFalse(marker.exists())
        self.assertEqual(result.summary["status"], "failed")
        self.assertEqual(result.summary["llm_calls"], 0)

    def test_configured_subscription_assessment_and_fabricated_citation_rejection(self):
        for fake_id, expected in [("sample:timing", "completed"), ("invented:benchmark", "failed")]:
            with self.subTest(evidence=fake_id), tempfile.TemporaryDirectory() as folder:
                self.root = Path(folder)
                self.project = self.root / "project"
                self.project.mkdir()
                self.cases = self.root / "cases.json"
                self.cases.write_text('[{"id":"sample","split":"discovery","args":[100]}]')
                self.make()
                value = json.loads(self.config.read_text())
                value["analysis"] = {"provider": "codex-chatgpt"}
                self.config.write_text(json.dumps(value))
                with (
                    patch.object(codex_provider, "require_chatgpt"),
                    patch.object(
                        codex_provider,
                        "command",
                        side_effect=self.fake_command(self.response(fake_id)),
                    ),
                ):
                    result = self.profile()
                self.assertEqual(result.summary["status"], "profiled")
                self.assertEqual(result.summary["assessment"]["status"], expected)
                self.assertEqual(result.summary["llm_calls"], 1)
                self.assertEqual(result.summary["assessment"]["model"], "gpt-6-astra")
                self.assertEqual(result.summary["assessment"]["effort"], "medium")
                self.assertFalse(result.summary["optimization_performed"])

    def test_assessment_schema_rejects_acceptance_fields(self):
        response = self.response()
        response["accepted"] = True
        with self.assertRaises(ValueError):
            validate(response, {"sample:timing": {}})

    def test_project_worker_module_does_not_collide_with_profiler_support(self):
        self.make()
        (self.project / "app.py").rename(self.project / "worker.py")
        value = json.loads(self.config.read_text())
        value["entry"]["target"] = "worker.py:run"
        self.config.write_text(json.dumps(value))
        self.assertEqual(self.profile().summary["status"], "profiled")

    def test_cli_can_disable_configured_assessment(self):
        self.make()
        value = json.loads(self.config.read_text())
        value["analysis"] = {"provider": "codex-chatgpt"}
        self.config.write_text(json.dumps(value))
        with (
            contextlib.redirect_stdout(io.StringIO()) as output,
            patch.object(
                codex_provider, "require_chatgpt", side_effect=AssertionError("must not call")
            ),
        ):
            main(
                [
                    "profile",
                    str(self.config),
                    "--no-explain",
                    "--repeats",
                    "3",
                    "--output",
                    str(self.root / "cli"),
                ]
            )
        self.assertIn('"assessment": "disabled"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
