"""Report display accepts large evidence without widening configuration limits."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lossless.cli import main
from lossless.jobs import read_json
from lossless.reports import read_report


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "report.json"

    def test_cli_preserves_large_nested_report_and_configuration_limit(self):
        report = {
            "kind": "cpu_application_workflow",
            "status": "accepted",
            "selected": "optimized",
            "delivery": [
                {"id": "evaluation", "exact": True, "reference": {"observations": "x" * 4_000_001}}
            ],
            "costs": {"workflow_seconds": 123.5, "payback": None},
            "scope": "Finite evidence only.",
        }
        self.path.write_text(json.dumps(report), encoding="utf-8")
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "JSON input exceeds 4 MB"):
            read_json(self.path)
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            main(["report", str(self.root)])
        self.assertEqual(errors.getvalue(), "")
        self.assertEqual(json.loads(output.getvalue()), report)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)

    def test_report_byte_boundary_includes_multibyte_text(self):
        report = {"label": "é"}
        data = json.dumps(report, ensure_ascii=False).encode("utf-8")
        self.path.write_bytes(data + b" " * (64 - len(data)))
        with patch("lossless.reports._MAX_REPORT_BYTES", 64):
            self.assertEqual(read_report(self.path), report)
            with self.path.open("ab") as stream:
                stream.write(b" ")
            with self.assertRaisesRegex(ValueError, "JSON report exceeds"):
                read_report(self.path)
            # Simulate growth after the initial size check.
            with patch.object(Path, "stat", return_value=SimpleNamespace(st_size=64)):
                with self.assertRaisesRegex(ValueError, "JSON report exceeds"):
                    read_report(self.path)

    def test_cli_rejects_report_over_real_limit(self):
        with self.path.open("wb") as stream:
            stream.truncate(64 * 1024 * 1024 + 1)
        self.assert_cli_error("JSON report exceeds 64 MiB: report.json")

    def test_cli_rejects_invalid_json(self):
        for data, message in [
            (b'{"status":"accepted","status":"failed"}', "duplicate JSON key: status"),
            (b'{"delivery":[{"exact":true,"exact":false}]}', "duplicate JSON key: exact"),
            (b'{"time":NaN}', "nonfinite JSON value: NaN"),
            (b'{"time":Infinity}', "nonfinite JSON value: Infinity"),
            (b'{"time":-Infinity}', "nonfinite JSON value: -Infinity"),
            (b'{"time":1e999}', "Out of range float values"),
            (b'{"status":', "Expecting value"),
            (b'{"text":"\xff"}', "utf-8"),
        ]:
            with self.subTest(data=data):
                self.path.write_bytes(data)
                self.assert_cli_error(message)

    def test_cli_rejects_missing_report(self):
        self.assert_cli_error("report.json")

    def assert_cli_error(self, message):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            with self.assertRaises(SystemExit) as caught:
                main(["report", str(self.root)])
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(output.getvalue(), "")
        self.assertIn(message, errors.getvalue())


if __name__ == "__main__":
    unittest.main()
