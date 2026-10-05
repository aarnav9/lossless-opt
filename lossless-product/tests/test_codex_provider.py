"""CLI transport uses receipts and fails closed on timeout or tool use."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from lossless import codex_provider as codex
from lossless import providers


class CodexProviderTests(unittest.TestCase):
    def test_response_timeout_inherits_effective_job_allowance_without_a_30_minute_cap(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(codex.effective_timeout(), 1800)
            self.assertEqual(codex.effective_timeout(3600), 3600)
        with patch.dict(os.environ, {"LOSSLESS_PROVIDER_TIMEOUT_SECONDS": "3550.5"}):
            self.assertEqual(codex.effective_timeout(), 3550.5)
            self.assertEqual(codex.effective_timeout(7200), 3550.5)
            self.assertEqual(codex.effective_timeout(60), 60)
            self.assertNotIn("LOSSLESS_PROVIDER_TIMEOUT_SECONDS", codex.environment())
        for value in ["0", "-1", "nan", "inf", "invalid"]:
            with patch.dict(os.environ, {"LOSSLESS_PROVIDER_TIMEOUT_SECONDS": value}):
                with self.assertRaises(ValueError):
                    codex.effective_timeout()

    def test_command_receives_the_parent_deadline_allowance(self):
        command = [
            sys.executable,
            "-c",
            "import os,json; print(json.dumps({'timeout': os.environ['LOSSLESS_PROVIDER_TIMEOUT_SECONDS']}))",
        ]
        response, _ = providers.call({}, {"command": command}, timeout=3550.5)
        self.assertEqual(response, {"timeout": "3550.5"})

    def fake_command(self, response, events, sleep=0):
        def command(folder, empty, model, effort):
            script = (
                "import pathlib,sys,time;sys.stdin.read();"
                "pathlib.Path(sys.argv[1]).write_text(sys.argv[2]);"
                "print(sys.argv[3],flush=True);time.sleep(float(sys.argv[4]))"
            )
            return [
                sys.executable,
                "-u",
                "-c",
                script,
                str(folder / "raw_response.json"),
                json.dumps(response),
                "\n".join(json.dumps(e) for e in events),
                str(sleep),
            ]

        return command

    def test_response_uses_receipt_usage_and_does_not_inherit_api_keys(self):
        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "never-inherit", "CODEX_API_KEY": "never-inherit"}
        ):
            self.assertNotIn("OPENAI_API_KEY", codex.environment())
            self.assertNotIn("CODEX_API_KEY", codex.environment())
        response = {
            "schema_version": 1,
            "proposals": [{"id": "test", "hypothesis": "h", "source": "s"}],
        }
        events = [{"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 20}}]
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(codex, "command", side_effect=self.fake_command(response, events)),
        ):
            folder = Path(temp) / "call"
            receipt = codex.invoke(
                {"operator": "copy"}, folder, model="test", effort="xhigh", timeout=3
            )
            self.assertTrue(receipt["eligible"])
            saved = json.loads((folder / "response.json").read_text())
            self.assertEqual(
                saved["usage"], {"input_tokens": 10, "output_tokens": 20, "cost_usd": None}
            )

    def test_forbidden_tools_and_late_complete_output_are_rejected(self):
        response = {
            "schema_version": 1,
            "proposals": [{"id": "test", "hypothesis": "h", "source": "s"}],
        }
        for tool, delay in [(True, 0), (False, 10)]:
            events = [{"type": "turn.completed", "usage": {}}]
            if tool:
                events.append({"type": "item.completed", "item": {"type": "command_execution"}})
            with (
                tempfile.TemporaryDirectory() as temp,
                patch.object(
                    codex, "command", side_effect=self.fake_command(response, events, delay)
                ),
            ):
                folder = Path(temp) / "call"
                receipt = codex.invoke(
                    {"operator": "copy"}, folder, model="test", effort="xhigh", timeout=0.3
                )
                self.assertFalse(receipt["eligible"])
                self.assertFalse((folder / "response.json").exists())
                self.assertEqual(receipt["timed_out"], not tool)

    def test_structured_command_keeps_tools_and_repository_context_disabled(self):
        argv = codex.command(Path("/tmp/call"), "/tmp/empty", "gpt-6.1-sol", "xhigh")
        for flag in [
            "--ignore-user-config",
            "--ephemeral",
            "--skip-git-repo-check",
            "--output-schema",
        ]:
            self.assertIn(flag, argv)
        self.assertIn("project_doc_max_bytes=0", argv)
        self.assertEqual(argv[argv.index("--model") + 1], "gpt-6.1-sol")


if __name__ == "__main__":
    unittest.main()
