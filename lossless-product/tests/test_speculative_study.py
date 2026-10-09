"""Optional CPU checks for the stock assisted-generation experiment."""

import copy
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    "speculative_study",
    Path(__file__).resolve().parents[1] / "experiments/speculative_050/study.py",
)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


class SpeculativeStudyTests(unittest.TestCase):
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

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "optional CPU Transformers")
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
