# Bicubic interpolation and SigLIP experiments

**Keep these as broader workload coverage; retain stock bicubic resizing and
investigate constant reuse in the SigLIP loss.** The two bicubic rewrites were
slower. Reusing SigLIP's constant sign matrix improved the isolated loss while
preserving tested output and gradient bits. The proposed diagonal rewrite did
not beat that stronger baseline. No product recipe was promoted.

The CPU pilot ran on 9 October 2026 with PyTorch 2.14.0, Python 3.11 and one
intra-op/inter-op thread on the local Apple Silicon host. Sixteen discovery
cases preceded 32 new evaluation cases in each of two fresh processes. All
methods were frozen before discovery, with no subsequent implementation tuning
or winner selection. Three input distributions checked correctness; timing used
normal random inputs and nine randomized interleaved rounds. Total runner time
was approximately 32.4 seconds, excluding authoring, tests and analysis.

## Results

Ratios are equal-case geometric means of paired reference/candidate timings.
Below one means slower. The reference constructs the SigLIP sign matrix on each
loss call, or uses the ordinary PyTorch bicubic call. These are component timings,
not complete image requests or model training steps.

| Implementation and scope | First evaluation | Second evaluation | Correctness |
| --- | ---: | ---: | --- |
| Bicubic channels-last, including both layout copies | 0.693× | 0.701× | All tested outputs bitwise equal |
| Bicubic horizontal then vertical passes | 0.723× | 0.726× | All tested outputs bitwise equal |
| SigLIP cached signs, forward loss | 1.262× | 1.252× | All tested losses bitwise equal |
| SigLIP cached signs, loss plus backward | 1.095× | 1.128× | All tested losses and logit gradients bitwise equal |
| SigLIP diagonal rewrite, forward loss | 1.159× | 1.144× | All tested losses bitwise equal |
| SigLIP diagonal rewrite, loss plus backward | 0.914× | 0.909× | All tested losses and logit gradients bitwise equal |

The diagonal rewrite reached only **0.918×/0.914×** the cached-sign forward
baseline, and **0.835×/0.806×** its forward-plus-backward baseline. Avoiding a
dense sign buffer did not produce an incremental speed win in these cases.
The native BCE-with-logits control passed the numerical checks throughout, but
failed bitwise agreement in 31 of 32 final loss configurations. Its aggregate
speed was similar to cached signs for forward calls and weaker for backward
calls. It belongs under a numerical contract for this reference.

Cached signs are an existing implementation technique, not a new model or an
LLM-discovered algorithm. The buffer costs `batch² × element_size` bytes: at batch
1024, 4 MiB for FP32 or 8 MiB for FP64. Its measured one-time construction cost
repaid in 1–290 calls among the 15 positive-mean first-process cases, and 1–439
calls among all 16 second-process cases. These are **setup-only** payback counts;
they exclude experiment/qualification costs. One first-process case regressed
to 0.949×. Fifteen of sixteen first-process and sixteen of sixteen second-process
descriptive per-case speed intervals lay wholly above one. A blanket deployment
claim is therefore unwarranted.

The full loss and first logit derivatives passed bitwise checks for cached signs
and the diagonal rewrite on every final case/distribution. Inputs and retained
outputs were unchanged. All 64 final different-semantics controls failed the
numerical comparison as intended: changing bicubic to bilinear or changing the
SigLIP loss to softmax cross entropy does not preserve this computation.

## How the ideas fit Lossless

**Bicubic interpolation is a workload to optimize.** It is useful for testing
image preprocessing, positional-embedding resizing, coordinate arithmetic,
separable filters, memory layout and boundary handling. Sampling coordinates,
antialiasing, rounding, padding and output conversion belong in its contract.
The present pilot uses floating-point PyTorch tensors; it does not establish
Pillow/uint8 equivalence or GPU performance. PyTorch documents overshoot and
antialias behavior in its
[interpolation API](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.interpolate.html).

**SigLIP is a vision-language training objective and model family.** Its pairwise
sigmoid objective avoids the global softmax normalization used by contrastive
alternatives. That changes the training objective; it cannot replace an existing
LLM's attention or vocabulary softmax under an exact contract. See the
[original paper](https://arxiv.org/abs/2303.15343) and the
[Transformers loss implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/siglip/modeling_siglip.py).

This pilot evaluates the loss component, including backward computation, without
encoders, pretrained weights, embeddings, an optimizer update or task labels.
It establishes no quality improvement, inference speedup or complete training
speedup. If a component occupies 10% of a request and becomes 1.26× faster, the
whole request becomes only about 1.021× faster. End-to-end profiling must decide
whether optimizing this component is worthwhile.

An existing SigLIP application would be a useful next repository workload:
freeze its checkpoint, processor and returned embeddings/scores; profile complete
requests; compare built-in text-feature reuse and attention backends before
authoring changes. For training, qualify bounded sign reuse with the full model's
parameter gradients and optimizer state. Replacing another model with SigLIP is
a separate task-quality comparison with held-out accuracy/retrieval data.

**The supplied name “lachmanick gca” remains unidentified.** Searches did not
establish a specific source, and the requester had no further identification.
No GCA variant was selected or tested. This item remains open pending a link,
title or recognizable spelling.

## Evidence and reproduction

- [Runner and exact foreground commands](../experiments/vision_mechanisms/README.md)
- [Complete plans, raw timings, checks and summaries](evidence/vision-mechanisms.json.gz)
- [Regression checks](../tests/test_vision_mechanisms.py)

The evidence includes every case and method, source hashes, runtime/build
identities, setup costs and persistent buffer sizes. Timings are exploratory
within-session observations on synthetic tensors, without power/thermal control,
peak-memory measurement or hardware-counter attribution. Numerical tolerances
were fixed at `atol=1e-6`, `rtol=1e-5` before execution. Exact means tested finite
bit agreement, not a universal proof. Four focused tests passed, including
gradient validation and deliberately incorrect controls. No GPU execution,
trained-model evaluation or product default change is included.
