"""Protect costly timer studies from contamination, retries and incomplete responses."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from lossless._native.common import read, write


SOURCE = Path(__file__).resolve().parents[1] / "experiments/timers_044/study.py"
spec = importlib.util.spec_from_file_location("timer_study", SOURCE)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class TimerStudyTests(unittest.TestCase):
    def test_only_announced_allowance_differs_and_inputs_are_frozen(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "study"
            study.freeze(root)
            plan = study.verify(root)
            self.assertEqual(plan["authoring_cap_total_seconds"], 27900)
            self.assertEqual(len(set(tuple(v) for v in plan["order"])), 12)
            blind = read(root / "unannounced_60m/prompt.json")
            for name, (seconds, announced) in study.CONDITIONS.items():
                prompt = read(root / name / "prompt.json")
                if announced:
                    allowance = prompt.pop("authoring_allowance")
                    self.assertEqual(allowance["wall_time_seconds"], seconds)
                self.assertEqual(prompt, blind)
            with self.assertRaisesRegex(AssertionError, "every condition"):
                study.require_all_sealed(root, plan)
            write(root / "announced_30m/prompt.json", {"tampered": True})
            with self.assertRaisesRegex(AssertionError, "frozen input changed"):
                study.verify(root)

    def test_deadline_kills_process_and_excludes_even_a_written_response(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write(root / "prompt.json", {"test": True})

            def command(folder, target, empty):
                script = (
                    "import json, pathlib, sys, time; sys.stdin.read(); "
                    "pathlib.Path(sys.argv[1]).write_text(json.dumps({'proposals': "
                    "[{'id':'p','hypothesis':'h','source':'s'}]*3})); "
                    "print(json.dumps({'type':'turn.completed','usage':{}}), flush=True); "
                    "time.sleep(20)"
                )
                return [sys.executable, "-u", "-c", script, str(target / "raw_response.json")]

            with patch.object(study, "command_for", side_effect=command):
                study.author_one(root, 0, 0.5)
            receipt = read(root / "author_0/receipt.json")
            self.assertTrue(receipt["timed_out"])
            self.assertFalse(receipt["eligible"])
            self.assertFalse((root / "author_0/response.json").exists())
            self.assertLess(receipt["elapsed_seconds"], 5)
            with patch.object(study, "command_for", side_effect=AssertionError("no retry")):
                study.author_one(root, 0, 60)

    def test_completed_response_is_retained_but_tool_use_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write(root / "prompt.json", {"test": True})
            for index, tool in enumerate([False, True]):

                def command(folder, target, empty):
                    events = [{"type": "turn.completed", "usage": {"output_tokens": 7}}]
                    if tool:
                        events.insert(
                            0, {"type": "item.completed", "item": {"type": "command_execution"}}
                        )
                    script = (
                        "import json, pathlib, sys; sys.stdin.read(); "
                        "pathlib.Path(sys.argv[1]).write_text(sys.argv[2]); "
                        "print(sys.argv[3], flush=True)"
                    )
                    response = {"proposals": [{"id": "p", "hypothesis": "h", "source": "s"}] * 3}
                    return [
                        sys.executable,
                        "-u",
                        "-c",
                        script,
                        str(target / "raw_response.json"),
                        json.dumps(response),
                        "\n".join(json.dumps(event) for event in events),
                    ]

                with patch.object(study, "command_for", side_effect=command):
                    study.author_one(root, index, 5)
                receipt = read(root / f"author_{index}/receipt.json")
                self.assertEqual(receipt["eligible"], not tool)
                self.assertEqual(receipt["blinding_passed"], not tool)
                self.assertEqual((root / f"author_{index}/response.json").exists(), not tool)


if __name__ == "__main__":
    unittest.main()
