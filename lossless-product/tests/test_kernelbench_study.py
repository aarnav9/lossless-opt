import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / "experiments/kernelbench_049"


def module(name):
    spec = importlib.util.spec_from_file_location(name + "_049", ROOT / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


protocol, tasks = module("protocol"), module("tasks")


class KernelBenchStudy(unittest.TestCase):
    def test_twenty_tasks_with_distinct_held_out_shapes(self):
        discovery, final = tasks.cases("discovery"), tasks.cases("evaluation")
        self.assertEqual(len(tasks.TASKS), 20)
        self.assertEqual(len(discovery), 40)
        for a, b in zip(discovery, final):
            self.assertEqual(a["task"], b["task"])
            self.assertNotEqual(a["shapes"], b["shapes"])
            self.assertNotEqual(a["seed"], b["seed"])
        self.assertEqual(
            [sum(k.startswith(f"L{i}_") for k in tasks.TASKS) for i in [1, 2, 3]], [8, 8, 4]
        )

    def test_original_upstream_sources_are_pinned(self):
        manifest = protocol.read(ROOT / "upstream/manifest.json")
        self.assertEqual({r["task"] for r in manifest["tasks"]}, set(tasks.TASKS))
        for row in manifest["tasks"]:
            self.assertEqual(protocol.digest(ROOT / "upstream" / row["source"]), row["sha256"])

    def test_invalid_case_cannot_disappear_from_denominator(self):
        rows = [
            dict(
                case=c,
                checks={n: {"pass": True} for n in ["reference", "candidate"]},
                samples={"reference": [2.0] * 7, "candidate": [1.0] * 7},
                peaks={"reference": [1000], "candidate": [1000]},
            )
            for c in tasks.cases("discovery")
        ]
        self.assertTrue(protocol.assess(rows)["confirmed"])
        self.assertFalse(protocol.assess(rows[:-1])["eligible"])
        rows[0]["checks"]["candidate"]["pass"] = False
        result = protocol.assess(rows)
        self.assertFalse(result["eligible"])
        self.assertIsNone(result["geomean"])
        self.assertEqual(len(result["tasks"]), 20)

    def test_source_scope_and_numerical_contract(self):
        seed = (ROOT / "seed.py").read_text()
        self.assertEqual(len(protocol.source_check(seed)), 64)
        for prefix in [
            "import os\n",
            "import time\n",
            "from ports import execute\n",
            "open('file')\n",
        ]:
            with self.assertRaises(ValueError):
                protocol.source_check(prefix + seed)
        self.assertEqual(protocol.ATOL, 1e-4)
        self.assertEqual(protocol.RTOL, 1e-4)


if __name__ == "__main__":
    unittest.main()
