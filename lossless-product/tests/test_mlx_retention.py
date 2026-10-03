"""Pure admission/scheduling checks plus an explicitly enabled local Metal test."""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from lossless.adapters.mlx import runtime, schedule


class MLXPolicyTests(unittest.TestCase):
    def test_wrong_device_or_runtime_cannot_admit(self):
        versions = {"mlx": "0.32.3", "mlx-lm": "0.32.0"}
        with patch(
            "lossless.adapters.mlx.identity.fingerprint",
            side_effect=AssertionError("must reject before model access"),
        ):
            self.assertFalse(runtime.admission("unused", "Apple M3", versions)[0])
            self.assertFalse(runtime.admission("unused", "Apple M2", {"mlx": "different"})[0])

    def test_policy_does_not_allow_precision_or_kernel_changes(self):
        for options in [
            {"allocator_cache_mib": True},
            {"allocator_cache_mib": 1024},
            {"precision": "float16"},
            {"source": "custom kernel"},
        ]:
            with self.assertRaises(ValueError):
                runtime.validate_options(options)

    def test_cohorts_preserve_request_identity_and_capacity(self):
        for count in [1, 7, 8, 9, 17, 33, 64]:
            lengths = [3 + (i % 3) for i in range(count)]
            outputs = [1 + ((i * 7) % 13) for i in range(count)]
            groups = schedule.cohorts(lengths, outputs, "short_first")
            cert = schedule.certificate(lengths, outputs, groups)
            self.assertEqual(sorted(i for group in groups for i in group), list(range(count)))
            self.assertTrue(
                all(len(group) <= 8 and len({lengths[i] for i in group}) == 1 for group in groups)
            )
            self.assertTrue(cert["attains_both"])


@unittest.skipUnless(
    os.environ.get("LOSSLESS_TEST_MLX_MODEL"),
    "set LOSSLESS_TEST_MLX_MODEL for the local Metal retention check",
)
class MetalRetentionTests(unittest.TestCase):
    def test_tokens_probabilities_kv_and_restored_overrides(self):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import BatchKVCache
        from lossless.adapters.mlx.model import GroupedHead
        from lossless.adapters.mlx.checks import diff, cache_diff

        folder = Path(os.environ["LOSSLESS_TEST_MLX_MODEL"]).resolve()
        allowed, reason = runtime.admission(folder, mx.device_info()["device_name"])
        self.assertTrue(allowed, reason)
        stock, tokenizer = load(str(folder))
        mx.eval(stock.parameters())
        grouped = GroupedHead(stock, 4)
        original_clear, original_extract = mx.clear_cache, BatchKVCache.extract
        texts, counts = ["The moon is", "The sun is", "One"], [3, 4, 2]
        reference = runtime.generate(stock, grouped, tokenizer, texts, counts, capture=True)
        candidate = runtime.generate(
            stock,
            grouped,
            tokenizer,
            texts,
            counts,
            capture=True,
            optimized=True,
            profile=True,
            options={"allocator_cache_mib": 256},
        )
        self.assertEqual(reference[0]["output_ids"], candidate[0]["output_ids"])
        self.assertNotIn("profile_stages_seconds", reference[0])
        stages = candidate[0]["profile_stages_seconds"]
        self.assertTrue(all(v >= 0 for v in stages.values()))
        self.assertAlmostEqual(sum(stages.values()), candidate[0]["seconds"], places=8)
        for i in range(len(texts)):
            self.assertTrue(diff(reference[2][i], candidate[2][i])["bitwise"])
            self.assertTrue(cache_diff(reference[1][i], candidate[1][i])["bitwise"])
        self.assertIs(mx.clear_cache, original_clear)
        self.assertIs(BatchKVCache.extract, original_extract)
        stopped = runtime.generate(
            stock, grouped, tokenizer, texts, counts, optimized=True, cancel_after=[1, 2, 1]
        )
        self.assertEqual([len(ids) for ids in stopped[0]["output_ids"]], [1, 2, 1])
        self.assertIn("stock serial", stopped[0]["fallback_reason"])


if __name__ == "__main__":
    unittest.main()
