"""Memory of worker processes and teardown belong to the composed lifecycle."""

import os
from pathlib import Path
import sys
import tempfile
import unittest

from lossless.applications.runtime import execute


class LifecycleTests(unittest.TestCase):
    def test_memory_includes_spawned_worker(self):
        # Fault pages in the child and hold them long enough for multiple samples.
        child = "import time;x=bytearray(48*1024**2);time.sleep(.7)"
        parent = (
            "import subprocess,sys;p=subprocess.Popen([sys.executable,'-c',sys.argv[1]]);p.wait()"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            result = execute(
                [sys.executable, "-c", parent, child],
                cwd=path,
                env=os.environ.copy(),
                stdin="",
                folder=path,
                timeout=5,
                max_output_bytes=4096,
                progress=lambda _: None,
                monitor_memory=True,
            )
        self.assertEqual(result["status"], "completed")
        memory = result["process_tree_memory"]
        self.assertTrue(memory["available"])
        self.assertGreaterEqual(memory["processes"], 2)
        self.assertGreater(memory["sum_sampled_peak_rss_bytes"], 48 * 1024**2)

    def test_application_identity_includes_installed_recipes(self):
        from lossless.applications.search import runner_identity

        hashes = runner_identity()
        self.assertIn("execution.py", hashes)
        self.assertIn("retained/modpoly.c", hashes)
        self.assertIn("retained/pybaselines.py", hashes)
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))


if __name__ == "__main__":
    unittest.main()
