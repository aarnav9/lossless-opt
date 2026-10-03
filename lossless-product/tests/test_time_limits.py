"""Time overrides must reach execution without weakening or mutating a job."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from lossless.api import Result
from lossless.cli import main, search_progress
from lossless.jobs import Workload, identity, template
from lossless.providers import bounded_timeout


class TimeLimitTests(unittest.TestCase):
    def test_quiet_search_emits_progress_and_stops_on_error(self):
        observed = threading.Event()
        messages = []

        def record(message):
            messages.append(message)
            observed.set()

        with patch("lossless._native.common.stamp", side_effect=record):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with search_progress(Path("run"), interval=0.01):
                    self.assertTrue(observed.wait(2))
                    raise RuntimeError("interrupted")
        self.assertTrue(messages)
        self.assertTrue(all("result=run/report.json" in message for message in messages))
        self.assertTrue(all("elapsed=" in message for message in messages))

    def test_python_overrides_are_frozen_and_leave_the_original_unchanged(self):
        cfg = template()
        cfg["llm"] = {"callback": "provider.py:complete", "timeout_seconds": 30}
        original = Workload.from_dict(cfg)
        updated = original.with_time_limits(seconds=28800, llm_timeout_seconds=1800)
        before, after = original.resolve(), updated.resolve()
        self.assertEqual(original.config, cfg)
        self.assertEqual(after["contract"], before["contract"])
        self.assertEqual(after["objective"], before["objective"])
        self.assertEqual(after["workload"], before["workload"])
        self.assertEqual(after["budget"]["max_candidates"], before["budget"]["max_candidates"])
        self.assertEqual(after["budget"]["max_llm_calls"], before["budget"]["max_llm_calls"])
        self.assertNotEqual(identity(before), identity(after))
        self.assertEqual(after["budget"]["wall_time_seconds"], 28800)
        self.assertEqual(bounded_timeout(after["llm"], 5000), 1800)
        self.assertEqual(bounded_timeout(after["llm"], 12), 12)

    def test_inspect_and_optimize_use_the_same_overrides_without_editing_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = template()
            cfg["llm"] = {"command": ["unused-provider"]}
            config = root / "job.json"
            config.write_text(json.dumps(cfg))
            saved = config.read_bytes()
            options = [str(config), "--budget", "8h", "--llm-timeout", "30m"]
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                main(["inspect", *options])
            inspected = json.loads(stdout.getvalue())
            self.assertEqual(inspected["budget"]["wall_time_seconds"], 28800)
            self.assertEqual(inspected["llm"]["timeout_seconds"], 1800)

            def execute(workload, *, output, resume):
                self.assertEqual(workload.resolve(), inspected)
                self.assertFalse(resume)
                return Result(output, {"status": "reference_retained", "selected": "reference"})

            with (
                patch("lossless.api.optimize", side_effect=execute) as optimize,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main(["optimize", *options, "--output", str(root / "run")])
            optimize.assert_called_once()
            self.assertEqual(config.read_bytes(), saved)

    def test_invalid_or_unused_authoring_limits_are_rejected(self):
        job = Workload.from_dict(template())
        with self.assertRaisesRegex(ValueError, "configured callback or command"):
            job.with_time_limits(llm_timeout_seconds=1800)
        for value in [0, -1, True, float("inf"), float("nan")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                job.with_time_limits(seconds=value)
        cfg = template()
        cfg["llm"] = {"callback": "provider.py:complete"}
        for value in [0, -1, True, float("inf"), float("nan")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                Workload.from_dict(cfg).with_time_limits(llm_timeout_seconds=value)


if __name__ == "__main__":
    unittest.main()
