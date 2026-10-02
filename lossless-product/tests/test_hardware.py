import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from lossless._native import cuda_info, engine, hardware, operators, providers
from lossless._native.common import write


class HardwareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def campaign(self, **kwargs):
        spec = {
            "operator": "softmax",
            "cases": [
                {
                    "id": "discovery",
                    "split": "discovery",
                    "rows": 3,
                    "columns": 7,
                    "layout": "slice2",
                },
                {"id": "hidden", "split": "evaluation", "rows": 5, "columns": 11, "layout": "f"},
            ],
            "budget_seconds": 60,
            "job_timeout_seconds": 2,
            "compile_timeout_seconds": 3,
            "target_ns": 100000,
            "screening_blocks": 3,
            "confirmation_blocks": 3,
            "max_proposals": 2,
        }
        write(self.root / "spec.json", spec)
        return engine.initialize(self.root / "spec.json", self.root / "run", **kwargs)

    def test_basic_never_starts_a_probe_or_command(self):
        with patch.object(hardware, "command", side_effect=AssertionError("unexpected subprocess")):
            report = hardware.collect(basic=True)
        self.assertEqual(report["accelerators"]["metal"]["status"], "not_queried")
        self.assertTrue(all(v is None for v in report["measurements"].values()))
        self.assertEqual(report["documentation_lookup"]["status"], "not_performed")

    def test_zero_query_budget_keeps_unknowns_without_launching(self):
        with patch.object(
            hardware.subprocess, "Popen", side_effect=AssertionError("must not launch")
        ):
            report = hardware.collect_bounded(timeout=0)
        self.assertIn("budget exhausted", report["collection_error"])
        self.assertEqual(report["accelerators"]["metal"]["status"], "not_queried")

    def test_linux_allowlist_and_missing_probes(self):
        text = json.dumps(
            {
                "lscpu": [
                    {"field": "Model name:", "data": "test CPU"},
                    {"field": "Hostname:", "data": "private"},
                    {"field": "Serial number:", "data": "secret"},
                ]
            }
        )
        self.assertEqual(hardware.parse_lscpu(text), {"Model name": "test CPU"})
        with (
            patch.object(hardware.platform, "system", return_value="Linux"),
            patch.object(hardware, "command", return_value=(None, "unavailable")),
            patch.object(hardware.shutil, "which", return_value=None),
        ):
            report = hardware.collect()
        self.assertEqual(report["accelerators"]["cuda"]["status"], "unavailable")
        self.assertIsNone(report["measurements"]["cache_hit_rate"])

    def test_probe_failure_and_timeout_stay_explicit(self):
        with patch.object(
            hardware.subprocess, "run", side_effect=subprocess.TimeoutExpired("probe", 1)
        ):
            self.assertEqual(hardware.command(["probe"]), (None, "timeout"))
        with patch.object(hardware, "command", return_value=("invalid", None)):
            self.assertEqual(hardware.cuda_probe()["reason"], "invalid_probe_json")
        with patch.object(hardware.shutil, "which", return_value=None):
            self.assertEqual(hardware.metal_probe()["reason"], "clang_not_found")

    def test_existing_report_is_not_overwritten(self):
        target = self.root / "hardware.json"
        target.write_text("original")
        with patch.object(hardware, "collect", side_effect=AssertionError("unnecessary probe")):
            with self.assertRaises(FileExistsError):
                hardware.emit_report(target)
        self.assertEqual(target.read_text(), "original")

    def test_report_validation_and_note_provenance(self):
        path = self.root / "data.json"
        write(path, {"schema_version": 1})
        with self.assertRaises(ValueError):
            hardware.load_report(path)
        note = {
            "claim": "Test claim",
            "source_url": "https://example.org/docs",
            "applies_to": "test device only",
            "retrieved_utc": "2026-10-01T00:00:00Z",
            "limitations": "not measured locally",
            "evidence_type": "vendor_documentation",
        }
        write(path, {"schema_version": 1, "entries": [note]})
        self.assertEqual(hardware.load_notes(path)["entries"][0], note)
        for key, invalid in [
            ("source_url", "file:///private"),
            ("applies_to", ""),
            ("evidence_type", "local_measurement"),
            ("source_url", "https://user:secret@example.org"),
        ]:
            write(path, {"schema_version": 1, "entries": [{**note, key: invalid}]})
            with self.assertRaises(ValueError):
                hardware.load_notes(path)

    def test_request_contains_frozen_context_without_holdout(self):
        profile = hardware.collect(basic=True)
        profile["host"]["fixture"] = "original"
        path = self.root / "hardware.json"
        write(path, profile)
        run = self.campaign(hardware_report=path)
        profile["host"]["fixture"] = "later"
        write(path, profile)
        request = providers.request_context(run)
        self.assertEqual(request["hardware_profile"]["host"]["fixture"], "original")
        self.assertNotIn("hidden", json.dumps(request))
        self.assertEqual(len(request["workload_context"]["discovery_tensors"]), 1)
        self.assertFalse(request["workload_context"]["quantization_allowed"])
        self.assertEqual(request["workload_context"]["backend"], "CPU native C")
        self.assertTrue((run / "harness/lossless/_native/metal_info.m").is_file())
        frozen = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json,sys; from pathlib import Path; from lossless._native.providers import request_context; print(json.dumps(request_context(Path(sys.argv[1]))))",
                str(run),
            ],
            cwd=run / "harness",
            text=True,
            capture_output=True,
            timeout=15,
            check=True,
        )
        self.assertEqual(json.loads(frozen.stdout), request)

    def test_both_context_files_are_frozen(self):
        run = self.campaign()
        for name in ("hardware_profile.json", "hardware_notes.json"):
            path = run / name
            original = path.read_text()
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "frozen contract/harness changed"):
                providers.request_context(run)
            path.write_text(original)
        engine.verify(run)

    def test_layout_context_matches_actual_discovery_generator(self):
        for layout in ("c", "f", "slice2"):
            for rows, cols in ((3, 7), (1, 7), (3, 1)):
                case = {"id": "case", "rows": rows, "columns": cols, "layout": layout}
                context = providers.tensor_context(case)
                x, r, _ = operators.input_arrays("softmax", case, 19)
                self.assertEqual(context["x_and_residual_strides_bytes"], list(x.strides))
                self.assertEqual(x.strides, r.strides)
                self.assertEqual(context["logical_bytes_per_matrix"], x.nbytes)

    def test_cuda_missing_driver_and_init_failure(self):
        with patch.object(cuda_info.C, "CDLL", side_effect=OSError):
            self.assertEqual(cuda_info.discover()["reason"], "cuda_driver_library_not_found")

        class Function:
            def __call__(self, *args):
                return 100

        class Driver:
            def __getattr__(self, name):
                return Function()

        with patch.object(cuda_info.C, "CDLL", return_value=Driver()):
            self.assertEqual(cuda_info.discover()["cuda_error_code"], 100)

    def test_cuda_ffi_units_and_unsupported_attributes(self):
        class Function:
            def __init__(self, call):
                self.call = call

            def __call__(self, *args):
                return self.call(*args)

        def put(pointer, value):
            pointer._obj.value = value
            return 0

        class Driver:
            cuInit = Function(lambda flags: 0)
            cuDeviceGetCount = Function(lambda p: put(p, 1))
            cuDriverGetVersion = Function(lambda p: put(p, 12040))
            cuDeviceGet = Function(lambda p, i: put(p, i))
            cuDeviceTotalMem_v2 = Function(lambda p, d: put(p, 24 * 1024**3))
            cuDeviceGetName = Function(lambda p, n, d: setattr(p, "value", b"Test CUDA GPU") or 0)
            cuDeviceGetAttribute = Function(lambda p, a, d: 1 if a == 106 else put(p, 32))

        with patch.object(cuda_info.C, "CDLL", return_value=Driver()):
            result = cuda_info.discover()
        self.assertEqual(result["driver_api_version"], 12040)
        device = result["devices"][0]
        self.assertEqual(device["total_memory_bytes"], 24 * 1024**3)
        self.assertEqual(device["attributes"]["warp_size"], 32)
        self.assertIsNone(device["attributes"]["max_blocks_per_multiprocessor"])
        self.assertEqual(device["attribute_error_codes"]["max_blocks_per_multiprocessor"], 1)


if __name__ == "__main__":
    unittest.main()
