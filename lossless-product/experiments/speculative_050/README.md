# Campaign 050: CPU speculative decoding

**Completed:** prompt lookup reached 1.646× across four held-out CPU cases, with
matching tokens but changed probability/KV/continuation bits. No exact promotion.
[Results and raw evidence](../../docs/speculative-050.md).

This research pilot compares stock Transformers greedy decoding with a smaller
draft model and prompt-lookup drafting. It does not change the product adapter.
The target is SmolLM2-360M-Instruct; the assistant is SmolLM2-135M-Instruct.
Both run in FP32 on CPU, with stock SDPA attention and fresh KV caches per call.

Three discovery prompts screen constant draft lengths 2/4/8 and lookup lengths
4/8, using 32 generated tokens. The fastest member of each family is frozen
before evaluating four different prompts with 64 generated tokens and seven
randomized paired repeats. Selection deliberately ignores exactness to diagnose
whether the algorithm offers speed even when its floating-point behavior fails
the stronger contract. A candidate passes the exact speed gate only if every
evaluation probe preserves tokens, full logprob bits, active KV bits and three
forced continuation logits, timed tokens match, and every case's descriptive
paired bootstrap lower endpoint exceeds one.

Every call captures all vocabulary scores and computes full log probabilities.
Timing includes fresh-cache prefill, drafting, verification, generation and
log-softmax. It excludes loading, tokenization, decoding text and the separate
correctness probes. EOS stopping is disabled to ensure equal generated work.
There is no prefix/output reuse between calls. Confidence-based early draft
stopping and adaptive draft lengths are disabled.

Transformers 4.57.6 has two relevant prompt-lookup edge cases: `None` EOS fails
when a lookup matches, and a matching final block can exceed the requested
maximum. The runner supplies an empty stop set and crops surplus tokens, scores
and KV together inside the timed region. Surplus work is charged, and the probe's
surplus count is recorded. No library implementation is patched.

The CPU pilot cannot establish a speedup over the repository's quantized MLX M2
measurements. Token agreement is also insufficient for the product's exact
probability/state contract. Finite probes are not a floating-point equivalence
proof. Bootstrap intervals describe this run, rather than independent machines
or sessions. Latency, peak memory and deployment admission are not qualified.

## Reproduce

From the repository root, create a CPU-only environment with uv:

```sh
uv venv .venv-speculative --python 3.11
uv pip install --python .venv-speculative/bin/python 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-speculative/bin/python 'transformers==4.57.6' 'numpy==2.4.6'
```

Download the pinned public artifacts into the ignored research directory and
record their local paths (approximately 1 GB of downloaded weights):

```sh
HF_HOME="$PWD/research/huggingface" HF_HUB_DISABLE_XET=1 .venv-speculative/bin/python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path('research/results/speculative_050')
root.mkdir(parents=True, exist_ok=True)
revisions = {
    '360M': 'a10cc1512eabd3dde888204e902eca88bddb4951',
    '135M': '12fd25f77366fa6b3b4b768ec3050bf629380bac',
}
models = {}
for size, revision in revisions.items():
    name = f'HuggingFaceTB/SmolLM2-{size}-Instruct'
    path = snapshot_download(name, revision=revision,
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'])
    models[size] = {'id': name, 'revision': revision, 'path': path}
(root / 'models.json').write_text(json.dumps(models, indent=2))
PY
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv-speculative/bin/python -u lossless-product/experiments/speculative_050/study.py --models research/results/speculative_050/models.json --output research/results/speculative_050/new-run --threads 4 --tokens 64 --repeats 7
.venv-speculative/bin/python -m unittest discover -s lossless-product/tests -p test_speculative_study.py -v
```

Use a new output directory each time. It contains a source snapshot, frozen plan
and model revisions, discovery measurements and selection, raw paired timings,
exactness diagnostics, target forward lengths and the final decision.

The regression checks use tiny random Llama models without network access. They
exercise identical/different assistants, short tails, lookup, cache offsets and
a deliberately wrong probability. A separate control checks surplus cropping.

Method references: [Transformers assisted decoding](https://huggingface.co/docs/transformers/v4.57.6/en/generation_strategies#speculative-decoding)
and [prompt lookup implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/generation/candidate_generator.py).
