"""Guard the iterative experiment's held-out barrier and explicit time feedback."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from lossless._native.common import read, write

HERE = Path(__file__).resolve().parents[1] / "experiments/iterate_045"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("iterate_study", HERE / "study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
sys.path.pop(0)


class IterativeStudyTests(unittest.TestCase):
    def test_author_deadline_excludes_partial_work_and_never_retries(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write(folder / "prompt.json", {"task": "synthetic transport test"})

            def command(source, target, empty):
                script = (
                    "import json,pathlib,sys,time; sys.stdin.read(); "
                    "pathlib.Path(sys.argv[1]).write_text(json.dumps({'proposals': "
                    "[{'id':'p','hypothesis':'h','source':'s'}]*3})); "
                    "print(json.dumps({'type':'turn.completed','usage':{}}),flush=True); "
                    "time.sleep(20)"
                )
                return [sys.executable, "-u", "-c", script, str(target / "raw_response.json")]

            with patch.object(study.timer, "command_for", side_effect=command):
                receipt = study.author.run(folder, 0.3, study.timer)
            self.assertTrue(receipt["timed_out"])
            self.assertFalse(receipt["eligible"])
            self.assertFalse((folder / "response.json").exists())
            self.assertLess(receipt["elapsed_seconds"], 5)
            with patch.object(study.timer, "command_for", side_effect=AssertionError("no retry")):
                self.assertEqual(study.author.run(folder, 60, study.timer), receipt)

    def test_frozen_plan_and_final_barrier(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "study"
            study.freeze(out)
            plan = study.verify(out)
            self.assertEqual(plan["search_seconds"], 3600)
            self.assertEqual(plan["call_seconds"], 1800)
            self.assertEqual(plan["new_candidate_cap"], 24)
            self.assertEqual(len(read(out / "enumeration.json")), 24)
            with self.assertRaisesRegex(ValueError, "both discovery arms"):
                study.require_sealed(out)
            write(out / "seeds.json", [])
            with self.assertRaisesRegex(AssertionError, "frozen input changed"):
                study.verify(out)

    def test_prompt_contains_clock_and_discovery_but_no_final_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "study"
            study.freeze(out)
            run = study.initialize(out, "iterative")
            checkpoint = {"rounds": [], "attempted_slots": 0}
            context = study.prompt(run, study.verify(out), checkpoint, 1200, 1110)
            self.assertEqual(context["search_schedule"]["elapsed_seconds"], 2400)
            self.assertEqual(context["search_schedule"]["remaining_search_seconds"], 1200)
            self.assertEqual(context["search_schedule"]["this_response_timeout_seconds"], 1110)
            self.assertEqual(context["feedback"]["remaining_seconds"], 1200)
            serialized = json.dumps(context)
            for case in read(out / "spec.json")["cases"]:
                if case["split"] == "evaluation":
                    self.assertNotIn(case["id"], serialized)
            self.assertTrue(all(c["split"] == "discovery" for c in context["discovery_cases"]))
            self.assertEqual(len(context["existing_proposals"]), 3)


if __name__ == "__main__":
    unittest.main()
