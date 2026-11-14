"""Request runtime changes preserve snapshots, ownership, guards and failures."""

from pathlib import Path
import gc
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from lossless.applications import PreparedApplication
from lossless.applications.bundle import signature
from lossless.applications.request_guard import RequestGuard
from lossless.jobs import identity
from lossless.applications import raw_wire
from lossless.applications.worker import Observer
from lossless.applications import prepared_worker
import test_prepared_application as prepared_tests


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = prepared_tests.PreparedTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_request_arrays_live_through_send_and_die_before_next_receive(self):
        bundle, _ = self.fixture.bundle("def run(x): return x\n")
        for layout in ("strided", "C", "F"):
            with self.subTest(layout=layout):
                self._check_request_array_lifetime(bundle, layout)

    def _check_request_array_lifetime(self, bundle, layout):
        refs = []
        original_unpack = raw_wire.unpack
        frames = iter(
            [
                raw_wire.pack(
                    dict(action="values", request={"calls": [{"args": [np.arange(16.0)]}]})
                ),
                raw_wire.pack(dict(action="close")),
            ]
        )
        sent = []

        def decode(frame):
            command = original_unpack(frame)
            if command["action"] == "values":
                refs.append(weakref.ref(command["request"]["calls"][0]["args"][0]))
            return command

        def run(x):
            table = np.arange(128.0).reshape(8, 16).copy(order="F" if layout == "F" else "C")
            result = table[:, ::2] if layout == "strided" else table
            refs.extend([weakref.ref(table), weakref.ref(result)])
            return result

        owner = self

        class Connection:
            def recv_bytes(self, limit):
                owner.assertTrue(all(ref() is None for ref in refs))
                return next(frames)

            def send_bytes(self, frame):
                owner.assertTrue(all(ref() is not None for ref in refs))
                sent.append(original_unpack(frame))

        import random
        import sys

        py_state, np_state = random.getstate(), np.random.get_state()
        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            with (
                patch.object(prepared_worker, "Connection", return_value=Connection()),
                patch.object(prepared_worker, "load_target", return_value=run),
                patch.object(
                    prepared_worker.Path,
                    "cwd",
                    return_value=self.fixture.root / ("worker-" + layout),
                ),
                patch.object(prepared_worker.os, "chdir"),
                patch.object(sys, "path", list(sys.path)),
                patch.object(sys, "dont_write_bytecode", True),
                patch.object(raw_wire, "unpack", side_effect=decode),
            ):
                prepared_worker.serve(0, bundle, "reference", True, ["raw", "buffers", "values"])
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            if gc_enabled:
                gc.enable()
        self.assertEqual([row["status"] for row in sent], ["ready", "completed"])
        expected = np.arange(128.0).reshape(8, 16)
        np.testing.assert_array_equal(
            sent[1]["outputs"][0], expected[:, ::2] if layout == "strided" else expected
        )

    def test_request_scope_releases_arrays_but_preserves_application_cache(self):
        code = """import numpy as np
import weakref
previous = []
cache = np.arange(7.0)
cached = weakref.ref(cache)
generation = 0
class Processor:
    def __init__(self): self.refs = []
    def __call__(self, x):
        assert all(ref() is not None for ref in self.refs), 'released during request'
        base = np.arange(x.size * generation * 2.0)
        result = base[::2]
        self.refs.extend([weakref.ref(x), weakref.ref(base), weakref.ref(result)])
        return result
def make():
    global generation
    assert all(ref() is None for ref in previous), 'retained previous request'
    assert cached() is cache, 'application cache released'
    generation += 1
    return Processor()
def observe(fn, result, args, kwargs):
    observed = result.copy()
    fn.refs.append(weakref.ref(observed))
    return observed
def cleanup(fn):
    global previous
    assert all(ref() is not None for ref in fn.refs), 'released before cleanup'
    previous = fn.refs
"""
        bundle, _ = self.fixture.bundle(
            code, factory=True, observe="app.py:observe", cleanup="app.py:cleanup"
        )
        for mode in ("fixtures", "values"):
            for features in ([], ["observations"], ["raw"], ["raw", "buffers", "dispatcher"]):
                with self.subTest(mode=mode, features=features):
                    enabled = features + (["values"] if mode == "values" else [])
                    with PreparedApplication(
                        bundle,
                        variant="reference",
                        input_mode=mode,
                        _qualifying=True,
                        _features=enabled,
                    ) as app:

                        def run():
                            value = np.arange(16.0) if mode == "values" else {"$npy": "x.npy"}
                            request = {"calls": [{"args": [value]} for _ in range(3)]}
                            return (
                                app.run_values(request)
                                if mode == "values"
                                else app.run(request, input_root=bundle / "reference")
                            )

                        first = run()
                        saved = [value.tobytes() for value in first["outputs"]]
                        second = run()
                        self.assertEqual([value.tobytes() for value in first["outputs"]], saved)
                        for value in first["outputs"]:
                            np.testing.assert_array_equal(value, np.arange(32.0)[::2])
                        for value in second["outputs"]:
                            np.testing.assert_array_equal(value, np.arange(64.0)[::2])

    def test_observation_reuse_preserves_mutation_and_retained_snapshots(self):
        code = """import numpy as np
class Processor:
    def __init__(self): self.buffer=np.empty(16); self.count=0
    def __call__(self,x):
        self.count+=1
        x[:]+=self.count
        self.buffer[:]=x
        return self.buffer
def make():return Processor()
def observe(fn,result,args,kwargs):
    result[0]=-0.0
    return result
def cleanup(fn):fn.buffer[-1]=123
"""
        bundle, _ = self.fixture.bundle(
            code, factory=True, observe="app.py:observe", cleanup="app.py:cleanup"
        )
        request = {"calls": [{"args": [{"$npy": "x.npy"}]} for _ in range(3)]}
        results = []
        for features in [
            [],
            ["observations"],
            ["observations", "dispatcher"],
            ["raw"],
            ["raw", "buffers", "observations"],
        ]:
            with PreparedApplication(bundle, _qualifying=True, _features=features) as app:
                r = app.run(request, input_root=bundle / "reference")
                saved = [x.tobytes() for x in r["outputs"]]
                app.run(request, input_root=bundle / "reference")
                self.assertEqual([x.tobytes() for x in r["outputs"]], saved)
                results.append((r["fingerprint"], saved))
                if "dispatcher" in features:
                    thread = app._dispatch_thread
            if "dispatcher" in features:
                self.assertFalse(thread.is_alive())
        for result in results[1:]:
            self.assertEqual(results[0], result)

    def test_dispatcher_timeout_stops_and_joins(self):
        bundle, _ = self.fixture.bundle(
            "import time\ndef run(x):\n    time.sleep(2)\n    return x\n"
        )
        with PreparedApplication(bundle, _qualifying=True, _features=["dispatcher"]) as app:
            app.timeout = 0.1
            with self.assertRaises(TimeoutError):
                app.run_case("new")
            self.assertFalse(app._dispatch_thread.is_alive())
            self.assertIsNotNone(app._process.poll())

    def test_direct_values_are_owned_copies_and_do_not_interpret_fixture_dicts(self):
        bundle, _ = self.fixture.bundle("def run(x):\n    x[:]+=1\n    return x\n")
        x = np.arange(16.0)
        with PreparedApplication(
            bundle, variant="reference", _qualifying=True, _features=["values"]
        ) as app:
            result = app.run_values({"calls": [{"args": [x]}, {"args": [x]}]})
            np.testing.assert_array_equal(x, np.arange(16.0))
            for output in result["outputs"]:
                np.testing.assert_array_equal(output, x + 1)
            saved = result["outputs"][0].tobytes()
            app.run_values({"calls": [{"args": [x + 2]}]})
            self.assertEqual(result["outputs"][0].tobytes(), saved)
            readonly = x.copy()
            readonly.flags.writeable = False
            for invalid in (x[::2], readonly, x.astype(object)):
                with self.assertRaises(ValueError):
                    app.run_values({"calls": [{"args": [invalid]}]})
            self.assertFalse(app._closed)

    def test_value_dicts_are_literal(self):
        bundle, _ = self.fixture.bundle("def run(x):return x\n")
        with PreparedApplication(
            bundle, variant="reference", _qualifying=True, _features=["values"]
        ) as app:
            literal = {"$npy": "does-not-exist.npy"}
            self.assertEqual(app.run_values({"calls": [{"args": [literal]}]})["outputs"], [literal])

    def test_value_mode_requires_its_own_matching_certificate(self):
        import json
        from lossless.applications.prepared import DEFAULT_FEATURES
        from lossless.jobs import read_json

        code = "def run(x):return x+1\n"
        bundle, _ = self.fixture.bundle(code, candidate=code)
        policy = dict(
            status="accepted",
            manifest_identity=identity(read_json(bundle / "manifest.json")),
            input_mode="fixtures",
            runtime_features=sorted(DEFAULT_FEATURES),
        )
        (bundle / "prepared.json").write_text(json.dumps(policy))
        with PreparedApplication(bundle) as app:
            self.assertEqual(app.info["variant"], "optimized")
            with self.assertRaisesRegex(ValueError, "separate runtime qualification"):
                app.run_values({"calls": [{"args": [np.arange(16.0)]}]})
        with PreparedApplication(bundle, input_mode="values") as app:
            self.assertEqual(app.info["variant"], "reference")
        policy.update(input_mode="values", runtime_features=sorted((*DEFAULT_FEATURES, "values")))
        (bundle / "prepared-values.json").write_text(json.dumps(policy))
        with PreparedApplication(bundle, input_mode="values") as app:
            self.assertEqual(app.info["variant"], "optimized")
            app.run_values({"calls": [{"args": [np.arange(16.0)]}]})
            with self.assertRaisesRegex(ValueError, "run_values"):
                app.run_case("new")
        policy["runtime_features"] = ["values"]
        (bundle / "prepared-values.json").write_text(json.dumps(policy))
        with PreparedApplication(bundle, input_mode="values") as app:
            self.assertEqual(app.info["variant"], "reference")

    def test_prepared_guard_checks_each_new_array_value(self):
        code = "def run(x):return x+1\n"
        bundle, _ = self.fixture.bundle(code, candidate=code)
        with PreparedApplication(bundle, _qualifying=True, _features=["guards", "values"]) as app:
            self.assertEqual(app.info["variant"], "optimized")
            x = np.arange(16.0)
            request = {"calls": [{"args": [x]}, {"args": [x]}]}
            # Match the fixture's sequence length.
            from lossless.jobs import read_json

            count = len(read_json(bundle / "cases.json")[0]["calls"])
            request["calls"] = [{"args": [x]} for _ in range(count)]
            app.run_values(request)
            x[0] = np.nan
            with self.assertRaisesRegex(ValueError, "outside qualification"):
                app.run_values(request)
            x[0] = 2
            app.run_values(request)

    def test_oneshot_rejection_does_not_consume_the_single_request(self):
        code = "def run(x): return x + 1\n"
        bundle, _ = self.fixture.bundle(code, candidate=code)
        with PreparedApplication(
            bundle,
            input_mode="values",
            _qualifying=True,
            _features=["raw", "guards", "values"],
            _lifecycle="callable-oneshot",
        ) as app:
            with self.assertRaisesRegex(ValueError, "outside qualification"):
                app.run_values({"calls": [{"args": [np.arange(17.0)]}]})
            self.assertFalse(app._closed)
            request = {"calls": [{"args": [np.arange(16.0)]}]}
            np.testing.assert_array_equal(
                app.run_values(request)["outputs"][0], np.arange(16.0) + 1
            )
            with self.assertRaisesRegex(ValueError, "oneshot worker already used"):
                app.run_values(request)
            self.assertTrue(app._closed)


class GuardTests(unittest.TestCase):
    def test_matches_existing_guard_for_mutation_metadata_and_nonfinites(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            array = np.arange(24.0).reshape(4, 6)
            np.save(root / "x.npy", array)
            calls = [{"args": [array], "kwargs": {"control": -0.0}}]
            allowed = [identity(signature(calls))]
            cases = [{"calls": [{"args": [{"$npy": "x.npy"}], "kwargs": {"control": -0.0}}]}]
            guard = RequestGuard(cases, root, allowed, scratch_limit=24)
            variants = [
                array,
                array + 2,
                array.T,
                array.astype(np.float32),
                array[:, ::-1],
                np.full((4, 6), np.nan),
                np.full((4, 6), np.inf),
                np.arange(24).reshape(4, 6),
            ]
            readonly = array.copy()
            readonly.setflags(write=False)
            variants.append(readonly)
            for value in variants:
                request = [{"args": [value], "kwargs": {"control": -0.0}}]
                try:
                    expected = identity(signature(request)) in allowed
                except ValueError:
                    expected = False
                self.assertEqual(guard.admits(request), expected)
            array[0, 0] = np.nan
            self.assertFalse(guard.admits(calls))
            array[0, 0] = 0
            self.assertTrue(guard.admits(calls))
            calls[0]["kwargs"]["control"] = 0.0
            self.assertFalse(guard.admits(calls))
            self.assertLessEqual(guard._scratch.size, 24)


class RawWireTests(unittest.TestCase):
    def test_pack_releases_contiguous_arrays_without_cyclic_collection(self):
        enabled = gc.isenabled()
        gc.disable()
        try:
            for order in ("C", "F"):
                for buffered in (False, True):
                    with self.subTest(order=order, buffered=buffered):
                        value = np.arange(24.0).reshape(4, 6).copy(order=order)
                        reference = weakref.ref(value)
                        frame = raw_wire.pack(
                            {"nested": [value]}, bytearray() if buffered else None
                        )
                        del value
                        self.assertIsNone(reference())
                        np.testing.assert_array_equal(
                            raw_wire.unpack(frame)["nested"][0], np.arange(24.0).reshape(4, 6)
                        )
            value = np.arange(24.0)
            reference = weakref.ref(value)
            with self.assertRaisesRegex(ValueError, "unsupported transported value"):
                raw_wire.pack([value, object()])
            del value
            self.assertIsNone(reference())
        finally:
            if enabled:
                gc.enable()

    def test_exact_bits_layout_ownership_and_buffer_growth(self):
        bits = np.array([0, 0x8000000000000000, 0x7FF8000000000123], dtype=np.uint64).view(float)
        value = dict(
            bits=bits,
            scalar=bits[-1],
            f=np.arange(12.0).reshape(3, 4).T,
            big=np.array([1 - 2j], dtype=">c16"),
            empty=np.empty((0, 3)),
            reverse=np.arange(9)[::-2],
            structured=np.array([(2, 3.0)], dtype=[("a", "i4"), ("b", "f8")]),
            nested=(b"abc", -0.0, True, None),
        )
        expected = identity(Observer(100000).encode(value))
        staging = bytearray()
        for buffer in (None, staging):
            frame = raw_wire.pack(value, buffer)
            restored = raw_wire.unpack(frame)
            self.assertEqual(identity(Observer(100000).encode(restored)), expected)
            self.assertEqual(restored["f"].strides, value["f"].strides)
            self.assertTrue(restored["f"].flags.owndata)
            self.assertTrue(restored["f"].flags.writeable)
            del frame
            raw_wire.pack(np.ones(100000), buffer)
            self.assertEqual(identity(Observer(100000).encode(restored)), expected)

    def test_rejects_malformed_and_object_frames(self):
        import json
        import struct

        valid = raw_wire.pack(np.arange(3.0))
        for frame in (b"", valid[:-1], valid + b"x"):
            with self.assertRaises(ValueError):
                raw_wire.unpack(frame)
        with self.assertRaises(ValueError):
            raw_wire.pack(np.array([object()], dtype=object))
        size = struct.unpack_from(">I", valid)[0]
        header = json.loads(valid[4 : 4 + size])
        for change in (
            {"length": -1},
            {"dtype": "O"},
            {"shape": [4]},
            {"order": "K"},
            {"shape": [-1]},
        ):
            edited = json.loads(json.dumps(header))
            edited["arrays"][0].update(change)
            encoded = json.dumps(edited).encode()
            frame = struct.pack(">I", len(encoded)) + encoded + valid[4 + size :]
            with self.assertRaises(ValueError):
                raw_wire.unpack(frame)


if __name__ == "__main__":
    unittest.main()
