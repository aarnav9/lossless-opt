"""Collect real hidden states and unchanged dequantized projection weights for CPU studies."""

import argparse
from pathlib import Path
import numpy as np
from lossless._native.common import stamp, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    stamp(f"START estimated=under1min result={out} log={out.parent / (out.name + '.log')}")
    import mlx.core as mx
    from mlx_lm import load

    model, tok = load(a.model)
    mx.eval(model.parameters())
    states = []
    for i, text in enumerate(
        [
            "The moon is",
            "Explain why the sky is blue.",
            "Write a Python function to sort integers.",
            "One plus one equals",
            "The capital city of France is",
            "A short story about a tree.",
            "Describe a triangle.",
            "Testing edge cases in numerical software.",
        ]
    ):
        tokens = mx.array(tok.encode(text, add_special_tokens=False))[None, :]
        h = model.model(tokens)[:, -1, :]
        mx.eval(h)
        states.append(np.array(h)[0])
        stamp(f"HIDDEN request={i + 1}/8")

    def weights(mod):
        if hasattr(mod, "scales"):
            return mx.dequantize(
                mod.weight, mod.scales, mod.biases, group_size=mod.group_size, bits=mod.bits
            )
        return mod.weight

    head = model.model.embed_tokens if model.args.tie_word_embeddings else model.lm_head
    w = weights(head)
    q = weights(model.layers[0].self_attn.q_proj)
    mx.eval(w, q)
    np.savez(out / "projections.npz", hidden=np.stack(states), head=np.array(w), q=np.array(q))
    write(
        out / "manifest.json",
        {
            "hidden_shape": list(np.stack(states).shape),
            "head_shape": list(w.shape),
            "q_shape": list(q.shape),
            "scope": "Real prompts and weights from the pinned quantized model; CPU studies define their own explicit dequantized FP32 projection reference. They do not certify the MLX quantized arithmetic or full-model token equivalence.",
        },
    )
    stamp(f"COMPLETE result={out / 'projections.npz'} log={out.parent / (out.name + '.log')}")


if __name__ == "__main__":
    main()
