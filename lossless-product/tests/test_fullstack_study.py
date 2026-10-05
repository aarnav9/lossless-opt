import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "protocol_048", ROOT / "experiments/fullstack_048/protocol.py"
)
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)


class FullStackStudy(unittest.TestCase):
    def test_hidden_inputs_differ(self):
        discovery, final = protocol.cases("discovery"), protocol.cases("evaluation")
        self.assertEqual([c["key"] for c in discovery], [c["key"] for c in final])
        for a, b in zip(discovery, final):
            self.assertNotEqual(a["texts"], b["texts"])
            self.assertNotEqual(a["counts"], b["counts"])

    def test_scope_and_syntax_rejections(self):
        source = (ROOT / "experiments/fullstack_048/seed.py").read_text()
        self.assertEqual(len(protocol.source_check(source)), 64)
        for bad in [
            "import os\n" + source,
            "open('x')\n" + source,
            "from lossless.adapters.mlx.checks import diff\n" + source,
            "class Runner(: pass",
            "pass",
        ]:
            with self.assertRaises((ValueError, SyntaxError)):
                protocol.source_check(bad)

    def test_gate_rejects_incorrect_and_memory_regressions(self):
        record = dict(
            case={"key": "a"},
            exact={"candidate": True, "reference": True},
            samples={"reference": [2.0] * 7, "candidate": [1.0] * 7},
            peaks={"reference": [100000000], "candidate": [100000000]},
            ttft={"reference": [0.1], "candidate": [0.1]},
        )
        self.assertTrue(protocol.assess([record])["confirmed"])
        record["peaks"]["candidate"] = [200000000]
        self.assertFalse(protocol.assess([record])["confirmed"])
        record["exact"]["candidate"] = False
        self.assertFalse(protocol.assess([record])["eligible"])

    def test_freeze_digest_detects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "source.py"
            p.write_text("first")
            before = protocol.digest(p)
            p.write_text("second")
            self.assertNotEqual(before, protocol.digest(p))


if __name__ == "__main__":
    unittest.main()
