"""Real loader, callable delivery, qualification boundaries and lifecycle failures."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lossless.applications import CallableApplication, CaseRecorder, optimize_project
from lossless.applications.input_loader import freeze
from lossless.applications.deployment import certify, paired, passes
from lossless.applications.discovery import inventory
from lossless.applications.workflow import make_bundle, bundle_environment
from lossless.applications.config import initialize, inspect, resolve
from lossless.jobs import identity


class OnboardingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.project = self.root / "project"
        self.project.mkdir()

    def bundle(self, code, candidate=None, inherit=()):
        (self.project / "app.py").write_text(code)
        np.save(self.project / "x.npy", np.arange(8.0))
        cases = self.root / "cases.json"
        cases.write_text(
            json.dumps(
                [
                    dict(id="dev", split="discovery", args=[{"$npy": "x.npy"}]),
                    dict(id="new", split="evaluation", args=[{"$npy": "x.npy"}]),
                ]
            )
        )
        initialize(self.project, self.root / "job", entry="app.py:run", cases=cases)
        config = self.root / "job/lossless.json"
        raw = json.loads(config.read_text())
        raw["environment"]["inherit_env"] = list(inherit)
        config.write_text(json.dumps(raw))
        value, cases = resolve(config)
        source = inspect(config)["source"]
        selected, selected_source = None, None
        if candidate:
            import shutil

            selected = self.root / "candidate"
            shutil.copytree(self.project, selected)
            (selected / "app.py").write_text(candidate)
            selected_source = inventory(selected)
        return make_bundle(
            self.root,
            self.project,
            source,
            selected,
            selected_source,
            value,
            cases,
            bundle_environment(sys.executable, inherit),
            "accepted",
        )

    def test_loader_full_workflow_and_call_adapter_no_win(self):
        (self.project / "app.py").write_text(
            'def run(x, factor=3):\n    return {"out": x * factor}\n'
        )
        (self.project / "examples.py").write_text(
            "import numpy as np\ndef cases():\n    return [\n"
            '        dict(id="dev", split="discovery", args=[np.arange(8.)]),\n'
            '        dict(id="new", split="evaluation", args=[np.arange(8.) + 2])]\n'
        )
        before = inventory(self.project)
        with contextlib.redirect_stdout(io.StringIO()):
            outcome = optimize_project(
                self.project,
                output=self.root / "run",
                entry="app.py:run",
                input_loader="examples.py:cases",
                budget=100,
            )
        self.assertEqual(outcome.summary["status"], "reference_retained", outcome.summary)
        self.assertEqual(before, inventory(self.project))
        self.assertTrue((Path(outcome.summary["application"]) / "callable.py").exists())
        self.assertNotIn("RuntimeWarning", (self.root / "run/input-loader/stderr.log").read_text())
        with CallableApplication(outcome.summary["application"]) as run:
            first = run(np.arange(8.0), factor=2)["out"]
            pid = run.info["pid"]
            np.testing.assert_array_equal(
                run(np.arange(11.0), factor=4)["out"], np.arange(11.0) * 4
            )
            self.assertNotEqual(run.info["pid"], pid)
            np.testing.assert_array_equal(first, np.arange(8.0) * 2)
        with self.assertRaisesRegex(ValueError, "closed"):
            run(np.arange(8.0))

    def test_loader_rejects_implicit_splits_and_type_changes(self):
        for index, rows in enumerate(
            [
                [dict(id="a", args=[np.arange(3.0)]), dict(id="b", args=[np.arange(3.0)])],
                [dict(id="a", split="discovery", args=[(1, 2)]), dict(id="b", split="evaluation")],
                [
                    dict(id="a", split="discovery", args=[np.arange(6.0)[::2]]),
                    dict(id="b", split="evaluation"),
                ],
            ]
        ):
            folder = self.root / str(index)
            folder.mkdir()
            with self.assertRaises(ValueError):
                freeze(rows, folder)

    def test_record_existing_test_inputs_before_mutation(self):
        recorder = CaseRecorder()
        value = np.arange(4.0)

        def mutate(x):
            x[:] = 0
            return 7

        self.assertEqual(recorder.call("dev", "discovery", mutate, value), 7)
        np.testing.assert_array_equal(recorder.cases[0]["args"][0], np.arange(4.0))
        self.assertTrue((value == 0).all())

    def test_missing_certificate_and_changed_certificate_cannot_enable_optimization(self):
        bundle = self.bundle("def run(x): return x * 2\n", "def run(x): return x * 99\n")
        with CallableApplication(bundle) as run:
            np.testing.assert_array_equal(run(np.arange(8.0)), np.arange(8.0) * 2)
            self.assertEqual(run.info["variant"], "reference")
        (bundle / "callable.json").write_text(
            json.dumps(
                dict(
                    status="accepted",
                    mode="prepared",
                    lifecycle_exact=True,
                    manifest_identity="wrong",
                )
            )
        )
        with CallableApplication(bundle) as run:
            self.assertEqual(run.mode, "oneshot")
            np.testing.assert_array_equal(run(np.arange(8.0)), np.arange(8.0) * 2)

    def test_deployment_auto_with_no_recipe_qualifies_reuse(self):
        (self.project / "app.py").write_text("def run(x): return x * 2\n")
        (self.project / "examples.py").write_text(
            "import numpy as np\ndef cases():\n    return [\n"
            'dict(id="dev", split="discovery", args=[np.arange(8.)]),\n'
            'dict(id="new", split="evaluation", args=[np.arange(8.) + 3])]\n'
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = optimize_project(
                self.project,
                output=self.root / "run",
                entry="app.py:run",
                input_loader="examples.py:cases",
                deployment="auto",
                expected_requests=100,
                budget=110,
            ).summary
        self.assertEqual(result["status"], "reference_retained", result)
        self.assertEqual(result["deployment"]["mode"], "prepared", result["deployment"])
        with CallableApplication(result["application"]) as run:
            self.assertEqual(run.mode, "prepared")
            self.assertIn("reference retained", run.info["fallback_reason"])
            pid = run.info["pid"]
            np.testing.assert_array_equal(run(np.arange(8.0)), np.arange(8.0) * 2)
            np.testing.assert_array_equal(run(np.arange(8.0) + 2), (np.arange(8.0) + 2) * 2)
            self.assertEqual(run.info["pid"], pid)

    def certify(self, bundle, status="reference_retained"):
        certify(
            bundle,
            dict(
                status=status,
                mode="prepared",
                lifecycle_exact=True,
                expected_requests=100,
                timing_scope="regression fixture",
            ),
        )

    def check_changed_environment(self, *, candidate):
        code = (
            "import os\ncount = 0\ndef run(x):\n    global count\n    count += 1\n"
            "    return x + (count if os.environ['REVIEW_STATE'] == 'changed' else 1)\n"
        )
        with patch.dict(os.environ, REVIEW_STATE="stable"):
            bundle = self.bundle(code, code if candidate else None, inherit=["REVIEW_STATE"])
            self.certify(bundle, "accepted" if candidate else "reference_retained")
            with CallableApplication(bundle) as run:
                self.assertEqual(run.mode, "prepared")
                np.testing.assert_array_equal(run(np.arange(8.0)), np.arange(8.0) + 1)
                np.testing.assert_array_equal(run(np.arange(8.0)), np.arange(8.0) + 1)
        with patch.dict(os.environ, REVIEW_STATE="changed"):
            with CallableApplication(bundle) as run:
                self.assertEqual(run.mode, "oneshot")
                self.assertEqual(run.info["variant"], "reference")
                self.assertIn("environment settings changed", run.info["fallback_reason"])
                self.assertEqual(run(5), 6)
                pid = run.info["pid"]
                self.assertEqual(run(5), 6)
                self.assertNotEqual(run.info["pid"], pid)
                self.assertIn("environment settings changed", run.info["fallback_reason"])

    def test_reference_reuse_requires_unchanged_environment(self):
        self.check_changed_environment(candidate=False)

    def test_optimized_fallback_uses_fresh_reference_after_environment_change(self):
        self.check_changed_environment(candidate=True)

    def test_reference_reuse_requires_unchanged_packages(self):
        bundle = self.bundle(
            "count = 0\ndef run(x):\n    global count\n    count += 1\n    return x + count\n"
        )
        path = bundle / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["environment"]["packages"].append(["removed-test-dependency", "1.0"])
        path.write_text(json.dumps(manifest))
        self.certify(bundle)
        with CallableApplication(bundle) as run:
            self.assertEqual(run.mode, "oneshot")
            self.assertIn("packages", run.info["fallback_reason"])
            self.assertEqual(run(5), 6)
            self.assertEqual(run(5), 6)

    def test_invalid_certificate_and_explicit_mode_cannot_enable_import_reuse(self):
        bundle = self.bundle(
            "count = 0\ndef run(x):\n    global count\n    count += 1\n    return x + count\n"
        )
        self.certify(bundle)
        valid = json.loads((bundle / "callable.json").read_text())
        invalid = ["{broken", "[]", "null"] + [
            json.dumps({**valid, **change})
            for change in (
                {"mode": []},
                {"status": []},
                {"runtime_features": []},
                {"input_mode": "fixtures"},
                {"manifest_identity": "wrong"},
            )
        ]
        for content in invalid:
            with self.subTest(certificate=content):
                (bundle / "callable.json").write_text(content)
                with CallableApplication(bundle, mode="prepared") as run:
                    self.assertEqual(run.mode, "oneshot")
                    self.assertEqual(run(5), 6)
                    self.assertEqual(run(5), 6)
                    self.assertIn("fresh reference", run.info["fallback_reason"])
        (bundle / "callable.json").unlink()
        with CallableApplication(bundle, mode="prepared") as run:
            self.assertEqual(run.mode, "oneshot")
            self.assertEqual(run(5), 6)
            self.assertEqual(run(5), 6)

    def test_qualification_cannot_skip_reference_reuse_environment_admission(self):
        with patch.dict(os.environ, REVIEW_STATE="stable"):
            bundle = self.bundle("def run(x): return x\n", inherit=["REVIEW_STATE"])
        with patch.dict(os.environ, REVIEW_STATE="changed"):
            with self.assertRaisesRegex(ValueError, "qualification environment changed"):
                CallableApplication(bundle, mode="prepared", _qualifying=True)

    def test_timeout_and_application_failure_are_never_retried(self):
        marker = self.root / "marker"
        with patch.dict(os.environ, CALL_MARKER=str(marker)):
            bundle = self.bundle(
                'import os\ndef run(x):\n    with open(os.environ["CALL_MARKER"], "a") as f: f.write("once\\n")\n    raise ValueError("failed")\n',
                inherit=["CALL_MARKER"],
            )
            with CallableApplication(bundle) as run:
                with self.assertRaisesRegex(ValueError, "without retry"):
                    run(np.arange(8.0))
        self.assertEqual(marker.read_text(), "once\n")

    def test_deployment_gates_detect_case_and_position_regressions(self):
        def metric(times):
            return dict(
                sequence_seconds=sum(times),
                setup_seconds=0.0,
                first_call_seconds=times[0],
                process_seconds=sum(times),
                peak_rss_bytes=100,
                call_seconds=times,
            )

        rows = {
            "a": [
                dict(ref=metric([1.0, 1.0]), good=metric([0.8, 0.8]), bad=metric([1.1, 0.4]))
                for _ in range(6)
            ]
        }
        self.assertTrue(passes(paired(rows, "ref", "good"), final=True))
        self.assertFalse(passes(paired(rows, "ref", "bad"), final=True))

    def test_optimized_adapter_and_outside_metadata_reference_escape(self):
        (self.project / "app.py").write_text(
            "import time\nimport numpy as np\nresident = np.ones(4_000_000)\n"
            "def run(x):\n    time.sleep(.025)\n    return x * 2\n"
        )
        (self.project / "examples.py").write_text(
            "import numpy as np\ndef cases():\n    return ["
            "dict(id='dev', split='discovery', args=[np.arange(8.)]),"
            "dict(id='new', split='evaluation', args=[np.arange(8.) + 2])]\n"
        )
        plan = dict(
            schema_version=1,
            source_scope=[dict(path="app.py", symbol="run")],
            contract="Exact copied values; delay is unobserved",
            architecture=False,
            max_candidates=1,
            candidates=[
                dict(
                    schema_version=1,
                    hypothesis="remove delay",
                    edits=[dict(path="app.py", symbol="run", body="return x * 2\n")],
                )
            ],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = optimize_project(
                self.project,
                output=self.root / "run",
                entry="app.py:run",
                input_loader="examples.py:cases",
                plan=plan,
                deployment="prepared",
                expected_requests=100,
                budget=150,
            ).summary
        self.assertEqual(result["status"], "accepted", result)
        with CallableApplication(result["application"]) as run:
            np.testing.assert_array_equal(run(np.arange(8.0) + 3), (np.arange(8.0) + 3) * 2)
            self.assertEqual(run.info["variant"], "optimized")
            with self.assertRaisesRegex(ValueError, "metadata"):
                run(np.arange(9.0))
        with CallableApplication(result["application"], variant="reference") as run:
            np.testing.assert_array_equal(run(np.arange(9.0)), np.arange(9.0) * 2)

    def test_loader_source_mutation_and_errors_leave_original_untouched(self):
        (self.project / "app.py").write_text("def run(x): return x\n")
        (self.project / "examples.py").write_text(
            "from pathlib import Path\ndef cases():\n"
            "    Path('app.py').write_text('broken')\n"
            "    return [dict(id='dev', split='discovery', args=[1]),"
            "dict(id='new', split='evaluation', args=[2])]\n"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = optimize_project(
                self.project,
                output=self.root / "run",
                entry="app.py:run",
                input_loader="examples.py:cases",
                budget=20,
            ).summary
        self.assertEqual(result["status"], "failed")
        self.assertIn("changed copied source", result["reason"])
        self.assertEqual((self.project / "app.py").read_text(), "def run(x): return x\n")

    def test_module_state_rejects_reuse_and_keeps_fresh_reference(self):
        (self.project / "app.py").write_text(
            "count = 0\ndef run(x):\n    global count\n    count += 1\n    return x + count\n"
        )
        (self.project / "examples.py").write_text(
            "def cases():\n    return [dict(id='dev', split='discovery', args=[1]), dict(id='new', split='evaluation', args=[2])]\n"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = optimize_project(
                self.project,
                output=self.root / "run",
                entry="app.py:run",
                input_loader="examples.py:cases",
                deployment="auto",
                budget=80,
            ).summary
        self.assertEqual(result["status"], "reference_retained", result)
        self.assertEqual(result["deployment"]["mode"], "oneshot")
        self.assertIn("reference:prepared", result["deployment"]["discovery_failures"])
        with CallableApplication(result["application"]) as run:
            self.assertEqual(run(5), 6)
            self.assertEqual(run(5), 6)

    def test_callable_timeout_stops_owned_worker(self):
        bundle = self.bundle("import time\ndef run(x):\n    time.sleep(5)\n    return x\n")
        with CallableApplication(bundle, timeout=0.6) as run:
            with self.assertRaises(TimeoutError):
                run(np.arange(8.0))
        self.assertIsNone(run.last_receipt)

    def test_exact_loader_bits_and_fortran_layout(self):
        array = np.arange(12.0).reshape(3, 4).T
        rows = freeze(
            [
                dict(id="dev", split="discovery", args=[array]),
                dict(id="new", split="evaluation", args=[np.array([-0.0])]),
            ],
            self.project,
        )
        saved = np.load(self.project / rows[0]["calls"][0]["args"][0]["$npy"])
        self.assertEqual(saved.strides, array.strides)
        self.assertEqual(identity(saved.tobytes().hex()), identity(array.tobytes().hex()))


if __name__ == "__main__":
    unittest.main()
