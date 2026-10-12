"""Optional CPU checks for the stock assisted-generation experiment."""

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    "speculative_study",
    Path(__file__).resolve().parents[1] / "experiments/speculative_050/study.py",
)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def integration_dependencies_available():
    """Only run version-specific cache/generation checks in a tested environment."""
    try:
        return (
            importlib.metadata.version("torch").split("+", 1)[0] in {"2.14.0", "2.14.1"}
            and importlib.metadata.version("transformers") == "4.57.6"
        )
    except importlib.metadata.PackageNotFoundError:
        return False


if os.environ.get("LOSSLESS_REQUIRE_SPECULATIVE_DEPS") == "1":
    if not integration_dependencies_available():
        raise RuntimeError("speculative checks require torch 2.14.0/2.14.1 and transformers 4.57.6")


class SpeculativeStudyTests(unittest.TestCase):
    def test_optional_integration_dependency_gate(self):
        for torch_version, transformers_version, expected in [
            ("2.14.1+cpu", "4.57.6", True),
            ("2.14.0", "4.57.6", True),
            ("2.2.2", "4.57.6", False),
            ("2.14.0", "4.43.3", False),
            ("2.14.0", "5.18.0", False),
        ]:
            versions = {"torch": torch_version, "transformers": transformers_version}
            with self.subTest(versions=versions):
                with patch("importlib.metadata.version", side_effect=versions.__getitem__):
                    self.assertEqual(integration_dependencies_available(), expected)
        with patch(
            "importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError
        ):
            self.assertFalse(integration_dependencies_available())

    def test_recorded_source_matches_published_evidence(self):
        root = Path(__file__).resolve().parents[1]
        recorded = root / "experiments/speculative_050/recorded_study.py"
        evidence = json.loads((root / "docs/assets/speculative-050.json").read_text())
        self.assertEqual(
            hashlib.sha256(recorded.read_bytes()).hexdigest(), evidence["plan"]["source_sha256"]
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional CPU PyTorch")
    def test_incompatible_caches_fail_without_continuation(self):
        import torch

        tensor = torch.zeros(1, 2, 5, 4)
        reference_cache = ((tensor, tensor), (tensor, tensor))

        def value(cache, length=5):
            return {
                "output": SimpleNamespace(
                    sequences=torch.tensor([[1, 2, 3]]),
                    past_key_values=SimpleNamespace(
                        to_legacy_cache=lambda: cache,
                        get_seq_length=lambda: length,
                    ),
                ),
                "logprobs": torch.zeros(3, 1, 8),
            }

        candidates = {
            "shorter_tensors": value(((tensor[:, :, :-1], tensor[:, :, :-1]),) * 2, 4),
            "different_dtype": value(((tensor.double(), tensor.double()),) * 2),
            "mixed_tensor_shapes": value(((tensor[:, :, :-1], tensor), (tensor, tensor))),
            "missing_layer": value(reference_cache[:-1]),
            "missing_value": value(((tensor,), (tensor, tensor))),
            "empty": value((), 0),
            "wrong_length_metadata": value(reference_cache, 4),
        }
        for name, candidate in candidates.items():
            with self.subTest(name=name):
                model = Mock(side_effect=AssertionError("invalid cache reached model"))
                checks = study.compare(value(reference_cache), candidate, model)
                self.assertFalse(checks["exact"])
                self.assertFalse(checks["kv_bitwise"])
                self.assertFalse(checks["continuation_bitwise"])
                self.assertIsNone(checks["kv_max_abs"])
                self.assertIsNone(checks["continuation_max_abs"])
                self.assertEqual(
                    checks["continuation_skipped_reason"], "cache_shape_dtype_or_length_mismatch"
                )
                model.assert_not_called()
                json.dumps(checks, allow_nan=False)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional CPU PyTorch")
    def test_lookup_overshoot_crops_tokens_scores_and_cache(self):
        import torch

        lengths = []
        cache = SimpleNamespace(crop=lambda length: lengths.append(length))
        output = SimpleNamespace(
            sequences=torch.arange(14).reshape(1, 14),
            scores=tuple(torch.zeros(1, 16) for _ in range(9)),
            past_key_values=cache,
        )
        model = SimpleNamespace(generate=lambda **kwargs: output)
        value = study.run(model, {"input_ids": torch.zeros(1, 5, dtype=torch.long)}, 7, "lookup_4")
        self.assertEqual(value["output"].sequences.shape[1], 12)
        self.assertEqual(value["logprobs"].shape[0], 7)
        self.assertEqual(value["surplus_tokens"], 2)
        self.assertEqual(lengths, [11])

    def test_fresh_cases_and_speed_summary(self):
        self.assertFalse(
            set(study.cases("discovery").values()) & set(study.cases("evaluation").values())
        )
        self.assertEqual(study.speed_summary([[2, 1]] * 7)["paired_geomean"], 2)

    @unittest.skipUnless(
        integration_dependencies_available(), "requires torch 2.14.0/2.14.1 and transformers 4.57.6"
    )
    def test_accept_reject_tail_and_exact_negative_control(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.set_num_threads(1)
        torch.manual_seed(50)
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        model = LlamaForCausalLM(config).eval()
        model.config._attn_implementation = "eager"
        same = copy.deepcopy(model)
        other = LlamaForCausalLM(config).eval()
        inputs = {
            "input_ids": torch.tensor([[1, 4, 5, 4, 5]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        for count in [1, 2, 7]:
            reference = study.run(model, inputs, count, "serial")
            self.assertTrue(study.compare(reference, reference, model)["exact"])
            for method, assistant in [("draft_4", same), ("draft_4", other), ("lookup_4", None)]:
                candidate = study.run(model, inputs, count, method, assistant)
                checks = study.compare(reference, candidate, model)
                self.assertTrue(checks["tokens_equal"])
                self.assertEqual(checks["kv_length"], [5 + count - 1] * 2)
        changed = copy.deepcopy(reference)
        changed["logprobs"][0, 0, 0] += 0.125
        self.assertFalse(study.compare(reference, changed, model)["exact"])


if __name__ == "__main__":
    unittest.main()
