"""Public workflow regressions: contracts, model boundary, export and fallback."""

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

import lossless
from lossless import providers
from lossless.jobs import Workload, read_json, template
from lossless._native.common import write


def propose_reference(prompt):
    request = json.loads(prompt)
    assert "evaluation_c" not in prompt and "evaluation_f" not in prompt
    assert request["formal_context"]["results"]
    return {
        "schema_version": 1,
        "proposals": [
            {
                "id": "from_callback",
                "hypothesis": "Control candidate exercises the provider boundary",
                "source": request["baseline_source"],
            }
        ],
    }


def slow_callback(prompt):
    time.sleep(20)


class ContractTests(unittest.TestCase):
    def test_exact_is_not_silently_relaxed_or_claimed_proved(self):
        for key, value in [
            ("preset", "numerical"),
            ("required_evidence", "proved"),
            ("weight_changes", True),
            ("precision_changes", True),
        ]:
            config = template()
            config["contract"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                Workload.from_dict(config).resolve()

    def test_separate_contract_is_resolved_without_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root / "contract.json", {"preset": "exact"})
            config = template()
            config["contract"] = {"file": "contract.json"}
            self.assertEqual(
                Workload.from_dict(config, base=root).resolve()["contract"]["preset"], "exact"
            )

    def test_rejects_duplicate_nonfinite_unknown_and_overlapping_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "job.json"
            for data in ['{"a":1,"a":2}', '{"a":NaN}']:
                path.write_text(data)
                with self.assertRaises(ValueError):
                    read_json(path)
        configs = []
        config = template()
        config["budegt"] = {}
        configs.append(config)
        config = template()
        config["budget"]["wall_time_seconds"] = True
        configs.append(config)
        config = template()
        config["budget"]["max_candidates"] = float("inf")
        configs.append(config)
        config = template()
        config["workload"]["cases"][2].update(rows=64, columns=257)
        configs.append(config)
        config = template()
        config["target"]["device"] = "cuda"
        configs.append(config)
        for config in configs:
            with self.assertRaises(ValueError):
                Workload.from_dict(config).resolve()

    def test_model_searches_never_receive_evaluation_text(self):
        # MLX worker constructs the model context explicitly; splitting is a
        # validated input invariant even on machines with no GPU dependencies.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model").mkdir()
            write(
                root / "requests.json",
                [
                    {"text": "same", "max_tokens": 2, "split": s}
                    for s in ["discovery", "evaluation"]
                ],
            )
            config = {
                "schema_version": 1,
                "workload": {
                    "adapter": "mlx.fixed_count",
                    "parameters": {"model": "model", "fixed_count": True},
                    "inputs": "requests.json",
                },
                "contract": {"preset": "exact"},
                "budget": {"wall_time_seconds": 60},
            }
            with self.assertRaisesRegex(ValueError, "distinct"):
                Workload.from_dict(config, base=root).resolve()


class ProviderTests(unittest.TestCase):
    def test_named_callback_runs_in_a_fresh_process(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "provider.py").write_text(
                'import json,os\ndef complete(prompt):\n return {"pid":os.getpid(),"input":json.loads(prompt)}\n'
            )
            value, seconds = providers.call(
                {"hello": "world"}, {"callback": "provider.py:complete"}, base=root, timeout=5
            )
            self.assertNotEqual(value["pid"], os.getpid())
            self.assertEqual(value["input"], {"hello": "world"})

    def test_timeout_contains_a_hung_callback(self):
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            providers.call({}, {}, callback=slow_callback, timeout=0.1)
        self.assertLess(time.monotonic() - started, 3)

    def test_command_receives_only_named_credentials_and_rejects_echo(self):
        code = 'import json,os,sys; data=json.load(sys.stdin); print(json.dumps({"key":os.environ["TEST_LOSSLESS_KEY"],"other":os.environ.get("TEST_LOSSLESS_OTHER")}))'
        connection = {
            "command": [sys.executable, "-c", code],
            "credential_env": ["TEST_LOSSLESS_KEY"],
        }
        with patch.dict(
            os.environ, TEST_LOSSLESS_KEY="private-test-value", TEST_LOSSLESS_OTHER="not-forwarded"
        ):
            with self.assertRaisesRegex(ValueError, "credential"):
                providers.call({}, connection, timeout=5)
        code = 'import json,os,sys; json.load(sys.stdin); print(json.dumps({"other":os.environ.get("TEST_LOSSLESS_OTHER")}))'
        with patch.dict(os.environ, TEST_LOSSLESS_OTHER="not-forwarded"):
            value, _ = providers.call({}, {"command": [sys.executable, "-c", code]}, timeout=5)
        self.assertIsNone(value["other"])


@unittest.skipUnless(shutil.which("clang"), "clang required")
class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.workload = Workload.from_dict(template(seconds=45), base=cls.root)
        cls.result = lossless.optimize(cls.workload, llm=propose_reference, output=cls.root / "run")
        cls.artifact = cls.result.export(cls.root / "artifact")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_full_search_has_real_provider_validation_and_separate_evaluation(self):
        self.assertIn(self.result.summary["status"], {"accepted", "reference_retained"})
        self.assertEqual(self.result.summary["llm_calls"], 1)
        self.assertIn("from_callback", read_json(self.root / "run/state.json")["proposals"])
        self.assertTrue((self.root / "run/report.html").is_file())
        self.assertEqual(read_json(self.root / "run/contract.json")["relation"], "bitwise_equal")

    def test_exported_native_operation_preserves_all_bits_and_independence(self):
        operation = lossless.load(self.artifact)
        for rows, columns, order in [(64, 257, "C"), (65, 128, "F"), (9, 11, "C")]:
            bits = np.resize(
                np.array([0, 0x80000000, 0x7F800001, 0x7FC01234, 0xFF800000], dtype=np.uint32),
                (rows, columns),
            )
            x = np.array(bits, order=order).view(np.float32)
            before = x.tobytes()
            bound = operation.bind(x)
            y = bound()
            self.assertEqual(y.tobytes(), before)
            self.assertFalse(np.shares_memory(x, y))
            y.flat[0] = 1
            self.assertEqual(x.tobytes(), before)
            self.assertEqual(bound().tobytes(), before)
        self.assertIn("outside validated", operation.fallback_reason)

    def test_completed_resume_does_not_repeat_model_calls(self):
        again = lossless.optimize(self.workload, output=self.root / "run", resume=True)
        self.assertEqual(again.summary, self.result.summary)

    def test_tampered_artifact_or_report_is_rejected(self):
        folder = self.root / "tampered"
        shutil.copytree(self.artifact, folder)
        (folder / "implementation.c").write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            lossless.load(folder)
        manifest = read_json(folder / "manifest.json")
        manifest["hashes"] = {}
        write(folder / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "missing"):
            lossless.load(folder)
        report = self.root / "run/report.json"
        original = report.read_bytes()
        try:
            report.write_text("{}")
            with self.assertRaises(ValueError):
                self.result.export(self.root / "bad-export")
        finally:
            report.write_bytes(original)

    def test_changed_machine_uses_reference(self):
        folder = self.root / "other-machine"
        shutil.copytree(self.artifact, folder)
        manifest = read_json(folder / "manifest.json")
        manifest["machine"] = "different"
        write(folder / "manifest.json", manifest)
        operation = lossless.load(folder)
        self.assertIsNone(operation.library)
        x = np.ones((64, 257), np.float32)
        np.testing.assert_array_equal(operation(x), x)

    def test_insufficient_budget_never_exports_an_optimization(self):
        workload = Workload.from_dict(template(seconds=0.1), base=self.root)
        result = lossless.optimize(workload, output=self.root / "short")
        self.assertEqual(result.summary["status"], "incomplete")
        with self.assertRaises(ValueError):
            result.export(self.root / "short-export")


class ProofTests(unittest.TestCase):
    def test_theorem_index_matches_its_extraction_receipt(self):
        import gzip, hashlib
        from lossless.proofs import RESOURCES

        receipt = read_json(RESOURCES / "theorems/receipt.json")
        data = gzip.decompress((RESOURCES / "theorems/index.jsonl.gz").read_bytes())
        self.assertEqual(hashlib.sha256(data).hexdigest(), receipt["index_sha256"])
        self.assertEqual(len(data.splitlines()), receipt["indexed_theorems"])

    def test_scoped_library_is_available_without_lean(self):
        from lossless.proofs import search

        result = search("matrix transpose", limit=3)
        self.assertEqual(len(result["results"]), 3)
        self.assertIn("retrieval_only", result["status"])

    def test_mathlib_setup_uses_pinned_manifest_without_update(self):
        from lossless import proofs

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(proofs, "tool", return_value="/fake/elan"),
            patch.object(proofs, "lean_binary", return_value="/fake/bin/lean"),
            patch.object(proofs, "check", return_value={"status": "checked"}),
            patch.object(proofs.subprocess, "run") as run,
        ):
            result = proofs.setup("mathlib", directory=folder)
            commands = [args.args[0] for args in run.call_args_list]
            self.assertEqual(
                commands,
                [
                    ["/fake/bin/lake", "exe", "cache", "get"],
                    ["/fake/bin/lake", "build", "KernelTheorems"],
                ],
            )
            self.assertEqual(
                (Path(folder) / "lake-manifest.json").read_bytes(),
                (proofs.RESOURCES / "theorems/lake-manifest.json").read_bytes(),
            )
            self.assertEqual(result["status"], "checked")


if __name__ == "__main__":
    unittest.main()
