"""Lightweight protocol checks; explicit opt-in for CPU tensor integration checks."""

import importlib.util
import os
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1] / "experiments/vision_mechanisms"
SPEC = importlib.util.spec_from_file_location("vision_mechanisms_study", ROOT / "study.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


class VisionProtocolTests(unittest.TestCase):
    def test_disjoint_shapes_and_fixed_families(self):
        a, b = study.cases("discovery"), study.cases("evaluation")
        self.assertEqual(len(a), 16)
        self.assertEqual(len(b), 32)
        self.assertFalse(any(x in b for x in a))
        self.assertEqual({x["family"] for x in b}, {"bicubic", "siglip_loss"})

    def test_speed_is_paired_and_deterministic(self):
        self.assertEqual(study.summary([(2, 1)] * 9)["speedup"], 2)
        self.assertEqual(study.summary([(2, 1)] * 9)["descriptive_95"], [2, 2])

    @unittest.skipUnless(os.environ.get("LOSSLESS_TEST_VISION") == "1", "opt-in CPU PyTorch")
    def test_loss_gradient_and_invalid_controls(self):
        import torch

        torch.set_num_threads(1)
        case = dict(family="siglip_loss", batch=17, dtype="float64", backward=True)
        x = study.input_tensor(case, 13, "extreme")
        before = x.clone()
        ref = study.operation(case, "stock")[0](x)
        candidate = study.operation(case, "diagonal")[0](x)
        self.assertTrue(all(study.difference(a, b)["numerical"] for a, b in zip(ref, candidate)))
        self.assertTrue(study.difference(before, x)["exact"])
        bad = study.operation(case, "softmax_control")[0](x)
        self.assertFalse(study.difference(ref[0], bad[0])["numerical"])
        self.assertFalse(
            study.difference(torch.tensor([float("nan")]), torch.tensor([float("nan")]))["exact"]
        )
        self.assertFalse(study.difference(torch.ones(2), torch.ones(3))["numerical"])

    @unittest.skipUnless(os.environ.get("LOSSLESS_TEST_VISION") == "1", "opt-in CPU PyTorch")
    def test_resize_negative_control_and_independence(self):
        import torch

        torch.set_num_threads(1)
        case = dict(
            family="bicubic", shape=[1, 3, 17, 23], size=[11, 13], antialias=True, layout="strided"
        )
        x = study.input_tensor(case, 14, "normal")
        ref = study.bicubic(x, case, "stock")
        bad = study.bicubic(x, case, "bilinear_control")
        self.assertFalse(study.difference(ref, bad)["numerical"])
        for name in ["channels_last", "separable"]:
            out = study.bicubic(x, case, name)
            self.assertTrue(study.difference(ref, out)["numerical"])
            self.assertTrue(out.is_contiguous())
            self.assertNotEqual(out.data_ptr(), x.data_ptr())


if __name__ == "__main__":
    unittest.main()
