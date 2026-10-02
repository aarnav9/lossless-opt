from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

from lossless._native import engine, operators, providers
from lossless._native.common import read, write


class KernelLabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def campaign(self, operator="softmax", budget=60):
        spec = {
            "operator": operator,
            "cases": [
                {"id": "discover", "split": "discovery", "rows": 3, "columns": 7, "layout": "c"},
                {"id": "evaluate", "split": "evaluation", "rows": 5, "columns": 9, "layout": "f"},
            ],
            "budget_seconds": budget,
            "job_timeout_seconds": 2,
            "compile_timeout_seconds": 3,
            "target_ns": 100000,
            "screening_blocks": 3,
            "confirmation_blocks": 3,
            "max_proposals": 8,
        }
        path = self.root / "spec.json"
        write(path, spec)
        return engine.initialize(path, self.root / "run")

    def proposal(self, run, name, source, operator="softmax"):
        c = self.root / (name + ".c")
        c.write_text(source)
        p = self.root / (name + ".json")
        write(
            p,
            {
                "schema_version": 1,
                "id": name,
                "operator": operator,
                "source_file": c.name,
                "hypothesis": "test",
            },
        )
        engine.submit(run, p)
        return p

    def test_resume_idempotence_seal_and_fixed_evaluation(self):
        run = self.campaign()
        proposal = self.proposal(run, "valid", operators.baseline_source("softmax"))
        with self.assertRaises(ValueError):
            engine.execute(run, "evaluation")
        self.assertEqual(engine.execute(run, "discovery", max_jobs=1)["jobs_executed"], 1)
        with self.assertRaises(ValueError):
            engine.seal(run)
        engine.execute(run, "discovery")
        state = read(run / "state.json")
        spent = state["spent_seconds"]
        self.assertEqual(engine.execute(run, "discovery")["jobs_executed"], 0)
        self.assertEqual(read(run / "state.json")["spent_seconds"], spent)
        engine.submit(run, proposal)  # identical submission is idempotent before sealing
        self.assertTrue(engine.feedback(run)["candidates"]["valid"]["complete"])
        seal = engine.seal(run)
        with self.assertRaises(ValueError):
            engine.submit(run, proposal)
        engine.execute(run, "evaluation")
        summary = engine.report(run)
        self.assertTrue(summary["frozen_choice_evaluation"]["complete"])
        self.assertEqual(summary["numerical_failures"], 0)
        self.assertEqual(summary["candidate_case_records"], 2)
        self.assertEqual(seal, read(run / "seal.json"))

    def test_compile_wrong_crash_and_timeout_are_contained(self):
        run = self.campaign()
        bodies = {
            "bad_compile": "this is not C",
            "wrong": "#include <stddef.h>\nvoid kernel(" + operators.ABI + "){}",
            "crash": "#include <stddef.h>\n#include <stdlib.h>\nvoid kernel("
            + operators.ABI
            + "){abort();}",
            "hang": "#include <stddef.h>\nvoid kernel("
            + operators.ABI
            + "){volatile unsigned spin=0;while(1){spin=spin+1;}}",
        }
        for name, source in bodies.items():
            self.proposal(run, name, source)
        engine.execute(run, "discovery")
        state = read(run / "state.json")
        self.assertEqual(
            {n: v["rejected"]["reason"] for n, v in state["proposals"].items()},
            {
                "bad_compile": "compile_failed",
                "wrong": "incorrect",
                "crash": "crashed",
                "hang": "timed_out",
            },
        )
        self.assertEqual(state["jobs"]["compile_baseline"]["status"], "ok")

    def test_frozen_contract_and_proposal_mutation_rejected(self):
        run = self.campaign()
        self.proposal(run, "valid", operators.baseline_source("softmax"))
        p = run / "contract.json"
        original = p.read_bytes()
        p.write_text("{}")
        with self.assertRaises(ValueError):
            engine.verify(run)
        p.write_bytes(original)
        engine.verify(run)
        (run / "proposals/valid/kernel.c").write_text("changed")
        with self.assertRaises(ValueError):
            engine.verify(run)

    def test_budget_defers_without_launch_and_can_be_explicitly_extended(self):
        run = self.campaign(budget=1)
        self.proposal(run, "valid", operators.baseline_source("softmax"))
        self.assertEqual(engine.execute(run, "discovery")["jobs_executed"], 0)
        self.assertEqual(read(run / "state.json")["spent_seconds"], 0)
        engine.extend_budget(run, 10, "unit test explicit extension")
        self.assertGreater(engine.execute(run, "discovery")["jobs_executed"], 0)

    def test_reference_failure_stops_instead_of_rejecting_proposal(self):
        with patch.object(
            operators,
            "baseline_source",
            return_value="#include <stddef.h>\nvoid kernel(" + operators.ABI + "){}",
        ):
            run = self.campaign()
        self.proposal(run, "valid", operators.baseline_source("softmax"))
        with self.assertRaises(RuntimeError):
            engine.execute(run, "discovery")
        self.assertIsNone(read(run / "state.json")["proposals"]["valid"]["rejected"])

    def test_rmsnorm_adapter_uses_same_controller(self):
        run = self.campaign("rmsnorm_residual")
        self.proposal(
            run, "valid", operators.baseline_source("rmsnorm_residual"), "rmsnorm_residual"
        )
        engine.execute(run, "discovery")
        engine.seal(run)
        engine.execute(run, "evaluation")
        self.assertEqual(engine.report(run)["numerical_failures"], 0)

    def test_evaluation_failure_cannot_be_reported_as_a_complete_success(self):
        run = self.campaign()
        source = operators.baseline_source("softmax").replace("){", "){if(rows==5)return;", 1)
        self.proposal(run, "shape_limited", source)
        engine.execute(run, "discovery")
        engine.seal(run)
        engine.execute(run, "evaluation")
        summary = engine.report(run)
        self.assertFalse(summary["frozen_choice_evaluation"]["complete"])
        self.assertEqual(summary["rejected"]["shape_limited"]["reason"], "incorrect")

    def test_scipy_baseline_meets_strict_row_sum_contract_on_fortran_inputs(self):
        case = {"rows": 64, "columns": 2048, "layout": "f"}
        for index, distribution in enumerate(operators.CONTRACTS["softmax"]["distributions"]):
            arrays = operators.input_arrays("softmax", case, 510700 + index, distribution)
            funcs, guards, _ = operators.executors("softmax", case, arrays, None, None)
            actual = funcs["scipy_softmax_float64"]()
            self.assertTrue(
                operators.check("softmax", actual, operators.reference("softmax", arrays))["passed"]
            )
            self.assertTrue(guards())

    def test_provider_neutral_command_submits_without_evaluation_context(self):
        run = self.campaign()
        adapter = self.root / "adapter.py"
        adapter.write_text(
            "import sys,json\nr=json.load(sys.stdin)\n"
            'assert all(c["split"]=="discovery" for c in r["discovery_cases"])\n'
            'assert "evaluation_cases" not in r and r["model"]=="arbitrary-user-model"\n'
            'print(json.dumps({"schema_version":1,"proposals":[{"id":"from_adapter",'
            '"hypothesis":"test transport", "source":r["baseline_source"]}],'
            '"usage":{"input_tokens":123,"cost_usd":0}}))\n'
        )
        config = self.root / "adapter.json"
        write(
            config,
            {
                "provider": "any-vendor-or-local",
                "model": "arbitrary-user-model",
                "command": [sys.executable, str(adapter)],
                "timeout_seconds": 2,
            },
        )
        result = providers.propose(run, config)
        self.assertEqual(result["submitted"], ["from_adapter"])
        self.assertEqual(result["usage"]["input_tokens"], 123)
        self.assertIn("from_adapter", engine.verify(run)["proposals"])
        engine.execute(run, "discovery")
        engine.seal(run)
        with self.assertRaises(ValueError):
            providers.request_context(run)

    def test_provider_timeout_and_invalid_response_do_not_admit_candidates(self):
        run = self.campaign()
        config = self.root / "adapter.json"
        for program in ("import time; time.sleep(5)", 'print("not JSON")'):
            write(
                config,
                {
                    "provider": "test",
                    "model": "fixture",
                    "command": [sys.executable, "-c", program],
                    "timeout_seconds": 0.1,
                },
            )
            with self.assertRaises(Exception):
                providers.propose(run, config)
        self.assertFalse(engine.verify(run)["proposals"])
        self.assertTrue(
            all(read(p)["status"] == "failed" for p in (run / "model_calls").glob("*/receipt.json"))
        )

    def test_provider_rejects_path_ids_and_batch_exceeding_limit(self):
        request = {"max_proposals": 1}
        for proposals in (
            [{"id": "../escape", "source": "C", "hypothesis": "h"}],
            [{"id": "ok", "source": "C", "hypothesis": "h"}] * 2,
        ):
            with self.assertRaises(ValueError):
                providers.validate_response(
                    {"schema_version": 1, "proposals": proposals}, request, []
                )


if __name__ == "__main__":
    unittest.main()
