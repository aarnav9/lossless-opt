"""Prepared lifecycle, exact value transport and source-candidate safety checks."""

import ast
import contextlib
from decimal import Decimal
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lossless.applications import PreparedApplication
from lossless.applications.automatic import Dispatch, common_arithmetic
from lossless.applications.config import initialize, inspect, resolve
from lossless.applications.diagnostics import differences, diagnose_reference
from lossless.applications.prepared_check import qualify_prepared
from lossless.applications.wire import pack, unpack
from lossless.applications.workflow import make_bundle, bundle_environment
from lossless.jobs import identity
from lossless.applications.worker import Observer


class ValueTests(unittest.TestCase):
    def test_transport_preserves_bits_layout_and_types(self):
        bits = np.array([0, 0x8000000000000000, 0x7FF8000000000123], dtype=np.uint64).view(float)
        value = {
            "bits": bits,
            "scalar": bits[-1],
            "complex": np.array([1 - 2j]),
            "f": np.arange(12.0).reshape(3, 4).T,
            "nested": (b"bytes", -0.0, None, True),
        }
        restored = unpack(pack(value))
        self.assertEqual(
            identity(Observer(100000).encode(value)), identity(Observer(100000).encode(restored))
        )
        self.assertEqual(restored["f"].strides, value["f"].strides)
        for invalid in (b"", pack(value)[:-1], pack(value) + b"x"):
            with self.assertRaises(ValueError):
                unpack(invalid)
        with self.assertRaises(ValueError):
            pack(np.array([object()], dtype=object))

    def test_diagnostics_signed_zeros_ulps_complex_and_endian(self):
        row = differences(np.array([1.0]), np.nextafter(np.array([1.0]), np.inf))[0]
        self.assertEqual(row["max_ulp_component_distance"], 1)
        self.assertEqual(differences(0.0, -0.0)[0]["max_ulp_component_distance"], 1)
        a = np.array([complex(1, 2)], dtype=">c16")
        b = np.array([complex(1, 3)], dtype=">c16")
        self.assertEqual(float(differences(a, b)[0]["max_absolute_component_difference"]), 1.0)
        self.assertEqual(differences(np.array([np.nan]), np.array([np.nan])), [])
        self.assertEqual(differences([1], [2])[0]["kind"], "value_or_structure")

    def test_expression_dispatch_evaluates_argument_once_and_keeps_fallback(self):
        tree = ast.parse("def run(get):\n    return np.all(get())\n")
        transform = Dispatch({"np"}, {"all"})
        namespace = {"np": np}
        exec(
            compile(ast.fix_missing_locations(transform.visit(tree)), "<candidate>", "exec"),
            namespace,
        )
        events = []

        def get():
            events.append("arg")
            return [True, False]

        self.assertFalse(namespace["run"](get))
        self.assertEqual(events, ["arg"])
        alias_tree = ast.parse("def run(x):\n    return _value.all(x > 0)\n")
        alias_namespace = {"_value": np}
        exec(
            compile(
                ast.fix_missing_locations(Dispatch({"_value"}, {"all"}).visit(alias_tree)),
                "<alias>",
                "exec",
            ),
            alias_namespace,
        )
        self.assertFalse(alias_namespace["run"](np.array([0, 1])))
        # Original function is captured before argument evaluation changes it.
        original = np.all

        def mutate():
            np.all = lambda value: "wrong"
            return [True]

        try:
            self.assertEqual(namespace["run"](mutate), original([True]))
        finally:
            np.all = original

    def test_arithmetic_exactness_and_custom_type_fallback(self):
        source = "def run(a, b):\n    p = (a + b) * (a + b)\n    return p / (a + b)\n"
        function = ast.parse(source).body[0]
        body = common_arithmetic(function, "np")
        self.assertIsNotNone(body)
        baseline, candidate = {"np": np}, {"np": np}
        exec(source, baseline)
        function.body = ast.parse(body).body
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                "<test>",
                "exec",
            ),
            candidate,
        )
        for a, b in [
            (np.linspace(0.1, 8, 32), np.arange(32.0)),
            (Decimal("1.2"), Decimal("3.4")),
            (np.array([Decimal("1.2")]), np.array([Decimal("3.4")])),
        ]:
            actual, expected = candidate["run"](a, b), baseline["run"](a, b)
            if isinstance(actual, np.ndarray) and actual.dtype.kind != "O":
                self.assertEqual(actual.tobytes(), expected.tobytes())
            else:
                self.assertEqual(actual, expected)
        for invalid in (
            "a = a + b\nreturn a + b",
            "return (global_x + a) * (global_x + a)",
            "return abs(a) + abs(a)",
        ):
            node = ast.parse(
                "def run(a, b):\n" + "\n".join("    " + line for line in invalid.splitlines())
            ).body[0]
            self.assertIsNone(common_arithmetic(node, "np"))


class PreparedTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.project = self.root / "project"
        self.project.mkdir()

    def bundle(
        self, code, *, candidate=None, factory=False, observe=None, cleanup=None, inherit=()
    ):
        (self.project / "app.py").write_text(code)
        np.save(self.project / "x.npy", np.arange(16.0))
        cases = self.root / "cases.json"
        cases.write_text(
            json.dumps(
                [
                    dict(id="dev", split="discovery", args=[{"$npy": "x.npy"}]),
                    dict(id="new", split="evaluation", args=[{"$npy": "x.npy"}]),
                ]
            )
        )
        config = self.root / "job/lossless.json"
        initialize(
            self.project,
            config.parent,
            entry="app.py:make" if factory else "app.py:run",
            factory=factory,
            cases=cases,
        )
        raw = json.loads(config.read_text())
        raw["entry"].update(observe=observe, cleanup=cleanup)
        raw["environment"]["inherit_env"] = list(inherit)
        config.write_text(json.dumps(raw))
        value, cases = resolve(config)
        source = inspect(config)["source"]
        output = self.root / "workflow"
        output.mkdir()
        selected = selected_source = None
        if candidate:
            from lossless.applications.replay import copy_project
            from lossless.applications.discovery import inventory

            selected = self.root / "candidate"
            copy_project(self.project, selected, source)
            (selected / "app.py").write_text(candidate)
            selected_source = inventory(selected)
        result = make_bundle(
            output,
            self.project,
            source,
            selected,
            selected_source,
            value,
            cases,
            bundle_environment(sys.executable, inherit),
            "test_fixture",
        )
        return result, config

    def test_worker_reuses_pid_resets_factory_and_copies_values(self):
        code = """class Processor:
    def __init__(self): self.count = 0
    def __call__(self, x):
        self.count += 1
        return x + self.count
def make(): return Processor()
def observe(fn, result, args, kwargs): return (result, fn.count)
def cleanup(fn): fn.count = 500
"""
        bundle, _ = self.bundle(
            code, factory=True, observe="app.py:observe", cleanup="app.py:cleanup"
        )
        with PreparedApplication(bundle) as app:
            pid = app.info["pid"]
            a, b = app.run_case("new"), app.run_case("new")
            self.assertEqual(a["fingerprint"], b["fingerprint"])
            np.testing.assert_array_equal(a["outputs"][0], np.arange(16.0) + 1)
            self.assertEqual(app.info["pid"], pid)
            (bundle / "reference/app.py").write_text('raise RuntimeError("edited after startup")')
            self.assertEqual(app.run_case("new")["fingerprint"], a["fingerprint"])
        with self.assertRaisesRegex(ValueError, "closed"):
            app.run_case("new")

    def test_uncertified_reference_and_metadata_rejection_before_execution(self):
        code = "def run(x): return x * 2\n"
        bundle, _ = self.bundle(code, candidate=code)
        with PreparedApplication(bundle) as app:
            self.assertEqual(app.info["variant"], "reference")
            self.assertIn("lifecycle", app.info["fallback_reason"])
        with PreparedApplication(bundle, _qualifying=True) as app:
            self.assertEqual(app.info["variant"], "optimized")
            np.save(self.root / "new.npy", np.arange(17.0))
            with self.assertRaisesRegex(ValueError, "metadata"):
                app.run({"args": [{"$npy": "new.npy"}]}, input_root=self.root)
            np.save(self.root / "new.npy", np.linspace(-1, 1, 16))
            result = app.run({"args": [{"$npy": "new.npy"}]}, input_root=self.root)
            np.testing.assert_array_equal(result["outputs"][0], np.linspace(-1, 1, 16) * 2)

    def test_failures_and_timeout_close_without_retry(self):
        code = """import os, time
def run(x):
    with open(os.environ['MARKER'], 'a') as f: f.write('called\\n')
    if x[0] == 0: raise RuntimeError('once')
    time.sleep(10)
    return x
"""
        marker = self.root / "marker"
        with patch.dict(os.environ, MARKER=str(marker)):
            bundle, _ = self.bundle(code, inherit=["MARKER"])
            with PreparedApplication(bundle) as app:
                with self.assertRaisesRegex(ValueError, "without retry"):
                    app.run_case("new")
                self.assertTrue(app._closed)
            self.assertEqual(marker.read_text(), "called\n")
            np.save(self.root / "sleep.npy", np.ones(16))
            with PreparedApplication(bundle) as app:
                app.timeout = 0.1
                with self.assertRaises(TimeoutError):
                    app.run({"args": [{"$npy": "sleep.npy"}]}, input_root=self.root)
                self.assertTrue(app._closed)
            self.assertEqual(marker.read_text(), "called\ncalled\n")

    def test_global_state_cannot_qualify_prepared_lifecycle(self):
        code = "count = 0\ndef run(x):\n    global count\n    count += 1\n    return x + count\n"
        bundle, _ = self.bundle(code, candidate=code)
        with PreparedApplication(bundle, variant="reference") as app:
            fresh = app.run_case("new")
        with contextlib.redirect_stdout(io.StringIO()):
            report = qualify_prepared(
                bundle,
                output=self.root / "qualification",
                expected={"new": fresh["fingerprint"]},
                expected_outputs={"new": identity(Observer(100000).encode(fresh["outputs"]))},
            )
        self.assertEqual(report["status"], "failed")
        self.assertIn("exactness", report["reason"])
        self.assertEqual(json.loads((bundle / "prepared.json").read_text())["status"], "failed")
        with PreparedApplication(bundle) as app:
            self.assertEqual(app.info["variant"], "reference")

    def test_reference_diagnosis_saves_values_and_rejects_recursive_output(self):
        _, config = self.bundle("import os\ndef run(x): return os.urandom(32)\n")
        with contextlib.redirect_stdout(io.StringIO()):
            result = diagnose_reference(config, output=self.root / "diagnostic", repeats=3)
        self.assertEqual(result.summary["status"], "reference_unstable")
        self.assertTrue(result.summary["cases"][0]["attempts"][1]["differences"])
        with self.assertRaisesRegex(ValueError, "separate"):
            diagnose_reference(config, output=self.project / "bad")


if __name__ == "__main__":
    unittest.main()
