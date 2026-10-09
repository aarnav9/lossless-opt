# CPU speculative decoding: campaign 050

The CPU pilot finds a workload-specific speedup, but neither speculative method
passes the exact probability/state contract. **Prompt lookup reached 3.830× on
repetition, 1.986× on summarization, and 1.646× across four cases.** The smaller
model draft averaged just 1.016×, with substantial regressions on prose. No
product recipe was changed.

## Held-out results

Speedup is serial time divided by speculative time; below one is slower. Each
entry is the geometric mean of seven paired ratios, with a descriptive paired
bootstrap 95% interval. The suite aggregate weights cases equally in log space.

| Workload | 135M draft, 8-token limit | Prompt lookup, 8-token limit |
| --- | ---: | ---: |
| Explanation | 0.535× [0.446, 0.643] | 0.984× [0.930, 1.058] |
| Python code | 1.578× [1.552, 1.600] | 0.982× [0.968, 0.994] |
| Repetition | 1.482× [1.423, 1.561] | 3.830× [3.748, 3.898] |
| Summarization | 0.851× [0.808, 0.885] | 1.986× [1.816, 2.241] |
| **Four-case geometric mean** | **1.016×** | **1.646×** |

On repetition, prompt lookup reduced median generation time from **2.758 s to
0.723 s**, using ten target forward passes for 64 output tokens. It drafts from
text already in the request/generated prefix, avoiding another model's forward
passes. In the summarization case it can reuse phrases from the source paragraph.
The smaller model needed eight target forward passes on code but still paid for
its own autoregressive generation. On explanation it emitted only about 2.78
output tokens per target call, making eight-token drafting wasteful.
These diagnostics explain the workload dependence; they are not a general
ranking of speculative algorithms or draft models.

## Correctness decision

All eight final probes and all 112 timed calls matched the serial token sequence
and requested count. Each final call generated 64 tokens. Each probe compared
full vocabulary log probabilities and all active KV tensors, then used the same
three forced continuation tokens to compare logits from cloned caches.

**Every speculative probe failed probability, KV and continuation bitwise
agreement.** Maximum observed absolute log-probability difference was
5.341e-5; KV difference was 3.433e-5; continuation-logit difference was 2.337e-5.
Cache lengths agreed. These differences are small numerically but fail the
frozen exact contract. No separate numerical tolerances or task-quality gate was
accepted, and matching these finite greedy outputs does not prove future tokens
will always agree. Both candidates failed the exact speed gate. Prompt lookup
also lacks a speed win across every workload even under a token-only comparison.

Multi-token verification changes the shapes of target matrix/attention
operations compared with single-token decoding. The observed differences are
consistent with that changed floating-point execution; this pilot does not
isolate a particular kernel as the source of every differing bit.

## Protocol and scope

Recorded 8 October 2026 on an **Intel Core i7-12700H**, with CPU-only PyTorch
2.14.1+cpu, Transformers 4.57.6, NumPy 2.4.6, four intra-op threads and one inter-op
thread. Target: SmolLM2-360M-Instruct; assistant: SmolLM2-135M-Instruct, both FP32,
stock SDPA, no sampling, no quantization, and identical tokenizer vocabularies.
Exact model revisions and source hash are in the evidence file.

Three 32-token discovery prompts screened draft lengths 2/4/8 and lookup lengths
4/8. The fastest configuration per family was frozen before four different
64-token evaluation prompts. Discovery selection intentionally ignored
exactness to measure the speed/fidelity tradeoff. Confidence-based early draft
stopping and adaptive draft schedules were disabled; this does not test their
potential to reduce wasted drafting on prose.

Every invocation prefills a fresh cache. Timings include drafting, target
verification, score capture and full log-softmax, and exclude model loading,
tokenization, text decoding and separate validation. Paired order is randomized;
models are warmed before discovery. Earlier aborted setup runs supplied no final
samples. An empty EOS set and timed surplus-output cropping handle two
Transformers 4.57.6 lookup edge cases; the reproduction guide explains them.

This is one process/session on synthetic prompts. CPU affinity, background load
and laptop power/thermal conditions were not controlled. Intervals are
within-session descriptions. Peak memory, first-token latency, other draft
policies and deployment are unqualified. **This is a different model/runtime,
precision and device from the quantized MLX Apple M2 studies**; these ratios
cannot establish a larger M2 gain or be multiplied into its reported speedup.

## Reproduce and inspect

- [Runner, setup and pinned-weight download instructions](../experiments/speculative_050/README.md)
- [Source](../experiments/speculative_050/study.py)
- [Raw paired timings, frozen discovery selection and exactness diagnostics](assets/speculative-050.json)
- [CPU regression checks](../tests/test_speculative_study.py)

The three regression tests passed with the experiment dependencies installed,
covering acceptance/rejection, short tails, cache offsets, a wrong-probability
control and surplus cropping. Ruff and frozen-source/evidence integrity checks
passed. No Apple M2, Metal or CUDA execution was performed.
