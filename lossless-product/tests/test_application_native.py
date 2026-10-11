"""Native helper admission, reproducible builds and real application evaluation."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from lossless.applications import initialize, native, patches, search

SCOPE = [
    {
        "name": "increment",
        "functions": [{"name": "increment", "returns": "void", "arguments": ["pointer", "size_t"]}],
    }
]
SOURCE = "#include <stddef.h>\nvoid increment(int *x,size_t n){for(size_t i=0;i<n;i++)x[i]+=1;}\n"


class ApplicationNativeTests(unittest.TestCase):
    def test_scope_and_sources_fail_closed(self):
        native.validate_scope(SCOPE)
        for change in [
            dict(name="../escape"),
            dict(functions=[]),
            dict(functions=[dict(name="x", returns=[], arguments=[])]),
        ]:
            with self.assertRaises(ValueError):
                native.validate_scope([{**SCOPE[0], **change}])
        with self.assertRaises(ValueError):
            native.validate_sources([dict(name="other", source=SOURCE)], SCOPE)
        with self.assertRaises(ValueError):
            patches.validate(
                dict(schema_version=1, hypothesis="native", edits=[], native_sources=[]), []
            )

    def test_native_code_changes_candidate_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text("def run(x):\n    return x\n")
            scope = [dict(path="app.py", symbol="run")]
            proposal = dict(
                schema_version=1,
                hypothesis="helper",
                edits=[],
                native_sources=[dict(name="increment", source=SOURCE)],
            )
            _, a = patches.apply(root, root, proposal, scope, SCOPE)
            proposal["native_sources"][0]["source"] += "\n/* independent variant */\n"
            _, b = patches.apply(root, root, proposal, scope, SCOPE)
            self.assertNotEqual(a, b)

    @unittest.skipUnless(shutil.which("clang"), "clang required")
    def test_build_runs_in_fresh_consumer_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = native.build(
                root,
                [dict(name="increment", source=SOURCE)],
                SCOPE,
                timeout=10,
                output=root / "build.json",
            )
            self.assertEqual(len(result["files"]), 3)
            consumer = """import ctypes
from _lossless_native_increment import increment
x=(ctypes.c_int*3)(1,2,3)
increment(ctypes.addressof(x),3)
assert list(x)==[2,3,4]
"""
            subprocess.run([sys.executable, "-c", consumer], cwd=root, check=True, timeout=10)
            with self.assertRaises(ValueError):
                native.build(
                    root,
                    [dict(name="increment", source=SOURCE)],
                    SCOPE,
                    timeout=10,
                    output=root / "again.json",
                )

    @unittest.skipUnless(shutil.which("clang"), "clang required")
    def test_saved_native_candidate_reaches_real_final_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            (project / "app.py").write_text("""def make():
    import time
    def run(x):
        time.sleep(0.004)
        return [v+1 for v in x]
    return run
""")
            cases = [
                dict(id=f"{split}_{i}", split=split, args=[[i, 2, 3]])
                for split in ["discovery", "evaluation"]
                for i in range(2)
            ]
            (root / "cases.json").write_text(json.dumps(cases))
            initialize(
                project,
                root / "job",
                entry="app:make",
                factory=True,
                cases=root / "cases.json",
                python=sys.executable,
            )
            body = """import ctypes
from _lossless_native_increment import increment
def run(x):
    values=(ctypes.c_int*len(x))(*x)
    increment(ctypes.addressof(values),len(x))
    return list(values)
return run
"""
            plan = dict(
                schema_version=1,
                source_scope=[dict(path="app.py", symbol="make")],
                native_helpers=copy.deepcopy(SCOPE),
                contract="Exact increment for bounded small int lists; preserve inputs.",
                architecture=False,
                wall_time_seconds=90,
                final_reserve_seconds=45,
                discovery_pairs=2,
                evaluation_pairs=8,
                max_candidates=1,
                max_setup_increase_seconds=0.5,
                max_peak_rss_ratio=2,
                max_traced_peak_ratio=100,
                candidates=[
                    dict(
                        schema_version=1,
                        hypothesis="Native copy fixture, no external efficacy claim.",
                        edits=[dict(path="app.py", symbol="make", body=body)],
                        native_sources=[dict(name="increment", source=SOURCE)],
                    )
                ],
            )
            result = search(root / "job/lossless.json", plan=plan, output=root / "run")
            self.assertEqual(result.summary["status"], "accepted_experiment", result.summary)
            patch = (root / "run/selected.patch").read_text()
            self.assertIn("_lossless_native_increment.c", patch)
            self.assertIn("_lossless_native_increment.py", patch)
            self.assertTrue((root / "run/candidates/candidate_01/native_builds.json").is_file())


if __name__ == "__main__":
    unittest.main()
