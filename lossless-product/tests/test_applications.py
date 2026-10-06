"""Intake must not execute projects; replay must freeze and isolate reference runs."""

import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import lossless
from lossless.artifacts import export
from lossless.applications import initialize, inspect, replay
from lossless.applications.config import resolve
from lossless.applications.discovery import discover, environment, inventory
from lossless.applications.worker import Observer
from lossless.cli import main


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.job = self.root / "job"
        self.cases = self.root / "inputs.json"
        self.cases.write_text(
            json.dumps(
                [
                    {"id": "sample", "split": "discovery", "args": [3]},
                    {"id": "final", "split": "evaluation", "args": [9]},
                ]
            )
        )

    def make(self, source="def run(x):\n    return x * 2\n", **options):
        (self.project / "app.py").write_text(source)
        options.setdefault("entry", "app.py:run")
        initialize(self.project, self.job, cases=self.cases, **options)
        self.config = self.job / "lossless.json"
        return self.config

    def change(self, update):
        value = json.loads(self.config.read_text())
        update(value)
        self.config.write_text(json.dumps(value))

    def run_replay(self, **options):
        with contextlib.redirect_stdout(io.StringIO()):
            return replay(self.config, output=self.root / "run", **options)

    def test_discovery_and_inspection_never_import_and_do_not_guess_inputs(self):
        (self.project / "app.py").write_text(
            "raise RuntimeError('must not import')\ndef run(x):\n return x\n"
        )
        (self.project / ".env").write_text("SECRET=do-not-copy")
        result = initialize(self.project, self.job)
        self.assertEqual(result["status"], "needs_input")
        report = inspect(self.job / "lossless.json")
        self.assertEqual(report["status"], "needs_input")
        self.assertEqual(report["cases"]["discovery"], 0)
        self.assertEqual(report["discovery"]["entry_candidates"][0]["target"], "app.py:run")
        self.assertNotIn(".env", report["source"]["files"])
        self.assertFalse(report["environment"]["device_execution_verified"])
        self.assertFalse((self.project / "__pycache__").exists())

    def test_real_function_replay_preserves_checkout_and_keeps_final_cases_closed(self):
        self.make(
            "from pathlib import Path\ncount=0\ndef run(x):\n global count\n count+=1\n Path('scratch.txt').write_text('local')\n return {'value':x*2,'count':count}\n"
        )
        before = (self.project / "app.py").read_bytes()
        result = self.run_replay()
        self.assertEqual(result.summary["status"], "replayed")
        self.assertEqual([c["id"] for c in result.summary["cases"]], ["sample"])
        self.assertEqual(result.summary["llm_calls"], 0)
        self.assertFalse(result.summary["optimization_performed"])
        self.assertFalse((self.project / "scratch.txt").exists())
        self.assertEqual((self.project / "app.py").read_bytes(), before)
        self.assertEqual(len(result.summary["cases"][0]["attempts"]), 2)
        self.assertTrue((result.directory / "manifest.json").exists())
        with self.assertRaisesRegex(ValueError, "reference replay"):
            export(result.directory, self.root / "artifact")
        with self.assertRaisesRegex(ValueError, "application jobs"):
            lossless.optimize(self.config)

    def test_discovery_prioritizes_application_files_before_scan_limit(self):
        examples = self.project / "docs/examples"
        examples.mkdir(parents=True)
        for index in range(201):
            (examples / f"example_{index:03}.py").write_text("def helper(): pass\n")
        package = self.project / "z_package"
        package.mkdir()
        (package / "app.py").write_text("def run(x): return x\n")
        (self.project / "test_a.py").write_text("def test_run(): pass\n")
        found = discover(self.project, inventory(self.project))
        self.assertTrue(found["scan_truncated"])
        self.assertEqual(found["python_files_scanned"], 200)
        self.assertEqual(found["entry_candidates"][0]["target"], "z_package/app.py:run")
        self.assertTrue(found["entry_candidates"][1]["target"].startswith("docs/examples/"))
        self.assertFalse(
            any(c["target"].startswith("test_a.py:") for c in found["entry_candidates"])
        )

    def test_arrays_mutations_factory_sequence_observer_and_cleanup(self):
        np.save(self.project / "x.npy", np.array([1, 2, 3], dtype=np.float32))
        saved = (self.project / "x.npy").read_bytes()
        self.cases.write_text(
            json.dumps(
                [
                    {
                        "id": "sequence",
                        "split": "discovery",
                        "calls": [
                            {"args": [{"$npy": "x.npy"}]},
                            {"args": [{"$npy": "x.npy"}]},
                        ],
                    }
                ]
            )
        )
        self.make(
            "class Model:\n def __init__(self): self.n=0\n def __call__(self,x):\n  self.n+=1\n  x[0]+=self.n\n  return x*2\ndef run(): return Model()\ndef observe(model,result,args,kwargs): return {'result':result,'state':model.n}\ndef cleanup(model): print('cleaned',model.n)\n",
            factory=True,
        )
        self.change(lambda v: v["entry"].update(observe="app.py:observe", cleanup="app.py:cleanup"))
        result = self.run_replay()
        self.assertEqual(result.summary["status"], "replayed", result.summary)
        calls = result.summary["cases"][0]["attempts"][0]["details"]["calls"]
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(
            calls[0]["observation"]["inputs_before"], calls[0]["observation"]["inputs_after"]
        )
        self.assertNotEqual(calls[0]["observation"]["output"], calls[1]["observation"]["output"])
        self.assertEqual((self.project / "x.npy").read_bytes(), saved)
        self.assertIn(
            "cleaned 2", (result.directory / "attempts/sequence_01/stdout.log").read_text()
        )

    def test_command_argv_stdin_outputs_and_workspace_reset(self):
        (self.project / "app.py").write_text(
            "from pathlib import Path\nimport sys\np=Path('counter.txt')\nn=int(p.read_text())+1 if p.exists() else 1\np.write_text(str(n))\nPath('answer.json').write_text(sys.stdin.read()+sys.argv[1])\nprint(n)\n"
        )
        self.cases.write_text(
            json.dumps(
                [
                    {
                        "id": "command",
                        "split": "discovery",
                        "argv": [";literal-not-shell"],
                        "stdin": "hello",
                    }
                ]
            )
        )
        initialize(self.project, self.job, command=["python", "app.py"], cases=self.cases)
        self.config = self.job / "lossless.json"
        self.change(lambda v: v["entry"].update(outputs=["answer.json"]))
        result = self.run_replay()
        self.assertEqual(result.summary["status"], "replayed", result.summary)
        self.assertFalse((self.project / "counter.txt").exists())
        for attempt in result.summary["cases"][0]["attempts"]:
            self.assertIn("answer.json", attempt["observation"]["files"])
        self.assertEqual((result.directory / "attempts/command_01/stdout.log").read_text(), "1\n")

    def test_unstable_reference_is_reported_without_becoming_accepted(self):
        self.make("import os\ndef run(x): return os.urandom(16)\n")
        result = self.run_replay()
        self.assertEqual(result.summary["status"], "reference_unstable")
        self.assertFalse(result.summary["cases"][0]["repeatable"])

    def test_exceptions_and_timeouts_stop_reference_replay(self):
        self.make("def run(x): raise RuntimeError('reference failed')\n")
        result = self.run_replay()
        self.assertEqual(result.summary["status"], "failed")
        self.assertEqual(len(result.summary["cases"][0]["attempts"]), 1)
        self.assertIn(
            "reference failed", (result.directory / "attempts/sample_01/stderr.log").read_text()
        )
        import shutil

        shutil.rmtree(self.root / "run")
        (self.project / "app.py").write_text("import time\ndef run(x): time.sleep(10)\n")
        self.change(lambda v: v["replay"].update(timeout_seconds=0.3))
        result = self.run_replay()
        self.assertEqual(result.summary["status"], "failed")
        self.assertEqual(result.summary["cases"][0]["attempts"][0]["status"], "timeout")
        self.assertLess(result.summary["elapsed_seconds"], 8)

    def test_output_limit_and_missing_output_fail_closed(self):
        self.make("def run(x): print('x'*100000); return x\n")
        self.change(lambda v: v["replay"].update(max_output_bytes=4096))
        result = self.run_replay()
        self.assertEqual(result.summary["status"], "failed")
        self.assertEqual(result.summary["cases"][0]["attempts"][0]["status"], "output_limit")

    def test_source_changed_while_copying_is_rejected(self):
        self.make()
        module = importlib.import_module("lossless.applications.replay")
        original = module.copy_project

        def changed(root, destination, snapshot):
            (root / "app.py").write_text("def run(x): return 0\n")
            return original(root, destination, snapshot)

        with patch.object(module, "copy_project", side_effect=changed):
            result = self.run_replay()
        self.assertEqual(result.summary["status"], "failed")
        self.assertIn("source changed", result.summary["reason"])
        self.assertEqual(result.summary["cases"], [])

    def test_environment_is_explicit_and_credentials_are_not_inherited_by_default(self):
        self.make(
            "import os\ndef run(x): return [os.getenv('LOSSLESS_PRIVATE_TEST'),os.path.basename(os.environ['HOME'])]\n"
        )
        with patch.dict(os.environ, LOSSLESS_PRIVATE_TEST="private-value"):
            result = self.run_replay()
        self.assertEqual(result.summary["status"], "replayed")
        self.assertNotIn("LOSSLESS_PRIVATE_TEST", result.summary["environment_names"])
        self.assertNotIn("private-value", (result.directory / "report.json").read_text())

    def test_project_snapshot_respects_gitignore_and_rejects_symlinks(self):
        subprocess.run(["git", "init", "-q", str(self.project)], check=True)
        (self.project / ".gitignore").write_text("ignored.txt\n")
        (self.project / "ignored.txt").write_text("skip")
        (self.project / "app.py").write_text("def run(x): return x\n")
        value = inventory(self.project)
        self.assertIn("app.py", value["files"])
        self.assertNotIn("ignored.txt", value["files"])
        (self.project / "link.py").symlink_to(self.project / "app.py")
        with self.assertRaisesRegex(ValueError, "symlink"):
            inventory(self.project)
        self.assertNotIn("link.py", inventory(self.project, ["link.py"])["files"])

    def test_schema_rejects_ambiguous_and_escaping_configuration(self):
        self.make()
        original = self.config.read_text()
        edits = [
            lambda v: v.update(typo=True),
            lambda v: v["entry"].update(target="../app.py:run"),
            lambda v: v["replay"].update(repeats=True),
            lambda v: v["replay"].update(timeout_seconds=0),
            lambda v: v["environment"].update(inherit_env=["PYTHONPATH"]),
        ]
        for edit in edits:
            self.config.write_text(original)
            self.change(edit)
            with self.assertRaises(ValueError):
                resolve(self.config)

    def test_explicit_project_inside_ignored_parent_is_replayed_as_directory(self):
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        (self.root / ".gitignore").write_text("project/\n")
        (self.project / ".env").write_text("PRIVATE=excluded")
        self.make()
        info = inspect(self.config)
        self.assertEqual(info["status"], "ready_to_replay")
        self.assertIn("ignored by enclosing Git", info["source"]["selection"])
        self.assertIsNone(info["source"]["git"]["revision"])
        self.assertIsNone(info["source"]["git"]["dirty"])
        self.assertNotIn(".env", info["source"]["files"])
        result = self.run_replay()
        self.assertEqual(result.summary["status"], "replayed", result.summary)

    def test_exact_observer_distinguishes_signed_zero_and_array_bits(self):
        self.assertNotEqual(Observer(100).encode(0.0), Observer(100).encode(-0.0))
        a = np.array([0x7FC00001], dtype=np.uint32).view(np.float32)
        b = np.array([0x7FC00002], dtype=np.uint32).view(np.float32)
        self.assertNotEqual(Observer(100).encode(a), Observer(100).encode(b))

    def test_src_package_relative_imports_and_factory_arguments(self):
        package = self.project / "src/pkg"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "helper.py").write_text("def multiply(x, factor): return x * factor\n")
        (package / "app.py").write_text(
            "from .helper import multiply\ndef create(factor):\n return lambda x: multiply(x, factor)\n"
        )
        self.make(entry="src/pkg/app.py:create", factory=True)
        self.change(lambda v: v["entry"].update(factory_args=[4]))
        result = self.run_replay()
        self.assertEqual(result.summary["status"], "replayed", result.summary)
        observed = result.summary["cases"][0]["attempts"][0]["details"]["calls"][0]["observation"][
            "output"
        ]
        self.assertEqual(observed, {"type": "int", "value": 12})

    def test_environment_probe_preserves_venv_without_executing_startup_hooks(self):
        import venv

        prefix = self.root / "venv"
        venv.create(prefix, with_pip=False, symlinks=True)
        site = (
            prefix
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        site.mkdir(parents=True, exist_ok=True)
        marker = self.root / "startup-ran"
        (site / "side_effect.pth").write_text(
            f"import pathlib; pathlib.Path({str(marker)!r}).touch()\n"
        )
        dist = site / "intake_probe-1.2.3.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: intake-probe\nVersion: 1.2.3\n"
        )
        detected = environment(prefix / "bin/python")
        self.assertFalse(marker.exists())
        self.assertFalse(detected["startup_hooks_executed"])
        self.assertEqual(detected["environment_prefix"], str(prefix))
        self.assertEqual(detected["packages"]["intake-probe"], "1.2.3")

    def test_explicit_array_fixtures_are_captured_even_when_gitignored(self):
        subprocess.run(["git", "init", "-q", str(self.project)], check=True)
        (self.project / ".gitignore").write_text("*.npy\n")
        np.save(self.project / "input.npy", np.array([1, 2, 3], dtype=np.float32))
        self.cases.write_text('[{"id":"array","split":"discovery","args":[{"$npy":"input.npy"}]}]')
        self.make("def run(x): return x * 2\n")
        info = inspect(self.config)
        self.assertIn("input.npy", info["source"]["files"])
        self.assertIn("input.npy", info["resolved"]["project"]["include"])
        result = self.run_replay()
        self.assertEqual(result.summary["status"], "replayed", result.summary)

    def test_missing_command_output_retains_failure_receipt(self):
        self.cases.write_text('[{"id":"command","split":"discovery","argv":[]}]')
        self.make("print('ok')\n", entry=None, command=["python", "app.py"])
        self.change(lambda v: v["entry"].update(outputs=["missing.json"]))
        result = self.run_replay()
        self.assertEqual(result.summary["status"], "failed")
        receipt = result.directory / "attempts/command_01/receipt.json"
        self.assertIn("declared output missing", json.loads(receipt.read_text())["reason"])
        self.assertFalse(result.summary["cases"][0]["repeatable"])

    def test_exhausted_total_budget_never_invokes_application(self):
        marker = self.root / "should-not-run"
        self.make(
            f"from pathlib import Path\ndef run(x):\n Path({str(marker)!r}).touch()\n return x\n"
        )
        result = self.run_replay(budget=0.001)
        self.assertEqual(result.summary["status"], "failed")
        self.assertIn("deadline", result.summary["reason"])
        self.assertFalse(marker.exists())

    def test_worker_children_do_not_survive_completed_reference(self):
        import time

        marker = self.root / "child-survived"
        child = (
            f"import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).touch()"
        )
        self.make(
            f"import subprocess,sys\ndef run(x):\n subprocess.Popen([sys.executable,'-c',{child!r}])\n return x\n"
        )
        result = self.run_replay()
        self.assertEqual(result.summary["status"], "replayed")
        time.sleep(1.1)
        self.assertFalse(marker.exists())

    def test_cli_generated_job_inside_project_excludes_its_own_outputs(self):
        (self.project / "app.py").write_text("def run(x): return x + 1\n")
        job = self.project / "lossless-job"
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main(
                [
                    "init",
                    "--project",
                    str(self.project),
                    "--entry",
                    "app.py:run",
                    "--cases",
                    str(self.cases),
                    "--output",
                    str(job),
                ]
            )
        self.assertEqual(json.loads(output.getvalue())["status"], "ready_to_replay")
        self.config = job / "lossless.json"
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main(["inspect", str(self.config), "--budget", "10s", "--json"])
        self.assertEqual(json.loads(output.getvalue())["preflight"]["total_seconds"], 10)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main(["inspect", str(self.config)])
        self.assertIn("Reference: your existing application", output.getvalue())
        self.assertIn("LLM calls: 0", output.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            outcome = lossless.replay(self.config, output=job / "runs/first")
        self.assertEqual(outcome.summary["status"], "replayed", outcome.summary)
        self.assertFalse(
            any(
                n.startswith("lossless-job/")
                for n in json.loads((job / "runs/first/source.json").read_text())["files"]
            )
        )


if __name__ == "__main__":
    unittest.main()
