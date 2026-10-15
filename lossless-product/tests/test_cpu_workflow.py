"""CPU project intake, conservative admission, and real runnable delivery."""

import ast
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lossless.applications import PreparedApplication, optimize_project, read_outputs
from lossless.applications.automatic import numpy_candidates
from lossless.applications.bundle import environment, select, signature, verify_tree
from lossless.applications.discovery import inventory
from lossless.applications.workflow import prepare_cases
from lossless.jobs import identity


class CPUWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.cases = self.root / "cases.json"
        self.cases.write_text(json.dumps([{"args": [2]}, {"args": [9]}]))

    def optimize(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return optimize_project(
                self.project,
                output=self.root / "workflow",
                budget=100,
                python=sys.executable,
                **kwargs,
            ).summary

    def launch(self, application, output, *args):
        return subprocess.run(
            [sys.executable, str(application / "run.py"), "--output", str(output), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def plan(self, body):
        return dict(
            schema_version=1,
            source_scope=[dict(path="app.py", symbol="run")],
            contract="Exact values and mutation; delay is unobserved.",
            architecture=False,
            discovery_resources=True,
            max_candidates=1,
            discovery_pairs=2,
            evaluation_pairs=6,
            max_setup_increase_seconds=1,
            max_peak_rss_ratio=2,
            max_traced_peak_ratio=10,
            candidates=[
                dict(
                    schema_version=1,
                    hypothesis="Test candidate",
                    edits=[dict(path="app.py", symbol="run", body=body)],
                )
            ],
        )

    def test_missing_inputs_does_not_import_project_or_call_provider(self):
        (self.project / "app.py").write_text(
            "raise RuntimeError('must not import')\ndef run(x):\n    return x\n"
        )
        result = self.optimize()
        self.assertEqual(result["status"], "needs_input")
        self.assertEqual(result["llm_calls"], 0)
        self.assertFalse((self.root / "workflow/profile").exists())
        self.assertTrue(result["entry_suggestions"])

    def test_case_split_requires_independent_cases(self):
        rows = prepare_cases(self.cases)
        self.assertEqual([r["split"] for r in rows], ["discovery", "evaluation"])
        for invalid in ([{}], [1, 2], [{"split": "discovery"}, {}], [{"split": "discovery"}] * 2):
            self.cases.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                prepare_cases(self.cases)

    def test_recipe_coverage_cli_forwarding_and_validation(self):
        (self.project / "app.py").write_text("def run(x):\n    return x\n")
        for coverage in ("stable", "expanded"):
            out = self.root / coverage
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lossless",
                    "run-project",
                    str(self.project),
                    "--recipe-coverage",
                    coverage,
                    "--output",
                    str(out),
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(completed.returncode, 2, completed.stderr)
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["status"], "needs_input")
            self.assertEqual(report["recipe_coverage"], coverage)
        with self.assertRaisesRegex(ValueError, "recipe_coverage"):
            self.optimize(recipe_coverage="unknown")

    def test_unstable_reference_cannot_produce_a_qualified_bundle(self):
        (self.project / "app.py").write_text("import os\ndef run(x):\n    return os.urandom(32)\n")
        result = self.optimize(entry="app.py:run", cases=self.cases)
        self.assertEqual(result["status"], "failed")
        self.assertIn("repeatability", result["reason"])
        self.assertNotIn("application", result)

    def test_factory_requires_an_explicit_state_observer(self):
        (self.project / "app.py").write_text("def make():\n    return lambda x: x\n")
        result = self.optimize(entry="app.py:make", cases=self.cases, factory=True)
        self.assertEqual(result["status"], "failed")
        self.assertIn("--observe", result["reason"])
        self.assertFalse((self.root / "workflow/profile").exists())

    def test_no_applicable_candidate_still_delivers_values_and_relocates(self):
        (self.project / "app.py").write_text("def run(x):\n    return {'value': x * 3}\n")
        result = self.optimize(entry="app.py:run", cases=self.cases)
        self.assertEqual(result["status"], "reference_retained", result.get("reason"))
        self.assertNotIn("search", result)
        self.assertTrue(result["source_unchanged"])
        moved = self.root / "moved"
        shutil.move(result["application"], moved)
        output = self.root / "saved"
        receipt = self.launch(moved, output, "--case", "case_001")
        self.assertEqual(receipt.returncode, 0, receipt.stderr)
        self.assertEqual(read_outputs(output), [{"value": 27}])

    def test_incorrect_candidate_is_rejected_and_reference_is_runnable(self):
        (self.project / "app.py").write_text("def run(x):\n    return x + 1\n")
        result = self.optimize(
            entry="app.py:run", cases=self.cases, plan=self.plan("return x + 2\n")
        )
        self.assertEqual(result["status"], "reference_retained", result.get("reason"))
        self.assertEqual(result["search"]["candidates"][0]["status"], "mismatch")
        self.assertTrue(all(row["exact"] for row in result["delivery"]))
        self.assertFalse((self.root / "workflow/application/optimized").exists())

    def test_equal_bits_with_changed_numpy_return_type_are_rejected(self):
        (self.project / "app.py").write_text(
            "import numpy as np\ndef run(x):\n    return np.float64(x)\n"
        )
        result = self.optimize(
            entry="app.py:run", cases=self.cases, plan=self.plan("return np.array(float(x))\n")
        )
        self.assertEqual(result["status"], "reference_retained", result)
        self.assertEqual(result["search"]["candidates"][0]["status"], "mismatch")
        self.assertTrue(all(row["exact"] for row in result["delivery"]))

    def test_accepted_delivery_new_values_and_changed_metadata_fallback(self):
        # Shared resident application data keeps this admission/routing test from
        # depending on tiny-process import RSS ratios under platform load.
        source = "import time\nimport numpy as np\nresident = np.ones(4_000_000)\ndef run(x):\n    time.sleep(.025)\n    return x * 3\n"
        (self.project / "app.py").write_text(source)
        for name, values in [("a", np.arange(16.0)), ("b", np.linspace(-1, 1, 16))]:
            np.save(self.project / (name + ".npy"), values)
        self.cases.write_text(
            json.dumps([{"args": [{"$npy": "a.npy"}]}, {"args": [{"$npy": "b.npy"}]}])
        )
        result = self.optimize(
            entry="app.py:run", cases=self.cases, plan=self.plan("return x * 3\n"), prepared=True
        )
        self.assertEqual(result["status"], "accepted", result.get("reason", result.get("search")))
        self.assertTrue(
            all(
                row["exact"] and row["auto"]["variant"] == "optimized" for row in result["delivery"]
            )
        )
        application = Path(result["application"])
        self.assertEqual(result["prepared"]["status"], "accepted", result["prepared"])
        with PreparedApplication(application) as prepared:
            self.assertEqual(prepared.info["variant"], "optimized")
            np.testing.assert_array_equal(
                prepared.run_case("case_001")["outputs"][0], np.linspace(-1, 1, 16) * 3
            )
        from lossless.applications.prepared_check import qualify_prepared
        from lossless.applications.worker import Observer

        original_certificate = (application / "prepared.json").read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            values_report = qualify_prepared(
                application,
                output=self.root / "value-qualification",
                input_mode="values",
                expected={row["id"]: row["reference"]["fingerprint"] for row in result["delivery"]},
                expected_outputs={
                    row["id"]: identity(
                        Observer(2**20).encode(
                            read_outputs(
                                self.root / "workflow/delivery" / row["id"] / "reference/outputs"
                            )
                        )
                    )
                    for row in result["delivery"]
                },
            )
        self.assertEqual(values_report["status"], "accepted", values_report)
        self.assertEqual((application / "prepared.json").read_bytes(), original_certificate)
        with PreparedApplication(application, input_mode="values") as prepared:
            self.assertEqual(prepared.info["variant"], "optimized")
            values = np.linspace(7, 9, 16)
            np.testing.assert_array_equal(
                prepared.run_values({"calls": [{"args": [values]}]})["outputs"][0], values * 3
            )
        request = self.root / "request.json"
        request.write_text(json.dumps({"args": [{"$npy": "new.npy"}]}))
        for label, values, expected in [
            ("new", np.linspace(3, 5, 16), "optimized"),
            ("shape", np.arange(17.0), "reference"),
            ("dtype", np.arange(16, dtype=np.float32), "reference"),
            ("nonfinite", np.full(16, np.nan), "reference"),
        ]:
            np.save(self.root / "new.npy", values)
            output = self.root / label
            launched = self.launch(application, output, "--request", str(request))
            self.assertEqual(launched.returncode, 0, launched.stderr)
            self.assertEqual(json.loads((output / "result.json").read_text())["variant"], expected)
            np.testing.assert_array_equal(read_outputs(output)[0], values * 3)
        self.assertEqual((self.project / "app.py").read_text(), source)
        # Added modules can change import resolution even if existing hashes match.
        (application / "optimized/extra.py").write_text("x = 1\n")
        output = self.root / "tampered"
        launched = self.launch(application, output, "--case", "case_001")
        self.assertEqual(launched.returncode, 0, launched.stderr)
        self.assertEqual(json.loads((output / "result.json").read_text())["variant"], "reference")
        (application / "reference/app.py").write_text("def run(x): return 0\n")
        self.assertNotEqual(
            self.launch(application, self.root / "bad", "--case", "case_001").returncode, 0
        )

    def test_failure_does_not_retry_callable(self):
        (self.project / "app.py").write_text(
            "import os\ndef run(x):\n    if x < 0:\n        with open(os.environ['CALL_MARKER'], 'a') as f: f.write('called\\n')\n        raise ValueError('runtime failure')\n    return x\n"
        )
        marker = self.root / "marker"
        with patch.dict(os.environ, CALL_MARKER=str(marker)):
            result = self.optimize(
                entry="app.py:run", cases=self.cases, inherit_env=["CALL_MARKER"]
            )
            request = self.root / "request.json"
            request.write_text('{"args": [-1]}')
            launched = self.launch(
                Path(result["application"]), self.root / "failure", "--request", str(request)
            )
        self.assertNotEqual(launched.returncode, 0)
        self.assertEqual(marker.read_text(), "called\n")

    def test_signature_and_environment_checks(self):
        values = np.arange(6.0)
        self.assertEqual(signature(values), signature(values + 8))
        self.assertNotEqual(signature(values), signature(values[::-1]))
        self.assertNotEqual(signature(1), signature(2))
        with self.assertRaises(ValueError):
            signature(np.array([np.inf]))
        (self.project / "app.py").write_text("x = 1\n")
        files = inventory(self.project)["files"]
        self.assertTrue(verify_tree(self.project, files))
        app = self.root / "app"
        shutil.copytree(self.project, app / "reference")
        shutil.copytree(self.project, app / "optimized")
        manifest = dict(
            reference_files=files,
            optimized_files=files,
            selected="optimized",
            environment=environment(),
            inherit_env=[],
            environment_values_hash=identity({}),
            input_signatures=[identity(signature([1]))],
        )
        self.assertEqual(select(app, manifest, [1])[0], "optimized")
        with patch("lossless.applications.bundle.environment", return_value={"changed": True}):
            self.assertEqual(select(app, manifest, [1])[0], "reference")

    def test_guarded_candidates_preserve_subclass_dispatch_and_copy_layout(self):
        code = "import numpy as np\ndef run(x):\n    return np.sum(x), np.copy(x)\n"
        (self.project / "app.py").write_text(code)
        profile = {
            "profiles": [
                {
                    "attribution": {
                        "top_cumulative": [
                            dict(
                                file="app.py",
                                function="run",
                                origin="project",
                                cumulative_seconds=1,
                            )
                        ]
                    }
                }
            ]
        }
        scope, candidates = numpy_candidates(self.project, inventory(self.project), profile)
        self.assertEqual(len(scope), 1)
        self.assertEqual(len(candidates), 3)
        tree = ast.parse(code)
        tree.body[1].body = ast.parse(candidates[-1]["edits"][0]["body"]).body
        namespace = {}
        exec(compile(ast.fix_missing_locations(tree), "<test>", "exec"), namespace)

        class Special(np.ndarray):
            def sum(self, *args, **kwargs):
                return 123

        for value in [np.arange(12.0).reshape(3, 4).T, np.array([1]).view(Special), [1, 2]]:
            actual = namespace["run"](value)
            self.assertEqual(actual[0], np.sum(value))
            np.testing.assert_array_equal(actual[1], np.copy(value))
            self.assertEqual(actual[1].strides, np.copy(value).strides)
        for extra in [
            "type = lambda x: np.ndarray\n",
            "from somewhere import type\n",
            "np = None\n",
        ]:
            (self.project / "app.py").write_text(code + extra)
            self.assertEqual(
                numpy_candidates(self.project, inventory(self.project), profile), ([], [])
            )


if __name__ == "__main__":
    unittest.main()
