# Bicubic and SigLIP mechanism experiments

Two bounded CPU pilots test whether library choices, separable resizing, or
avoiding a dense label matrix improve these particular computations. The SigLIP
study measures its **pairwise training loss and first derivatives with respect
to the logits**. It does not train a model, evaluate image quality, or change
Lossless's product adapters. [Results](../../docs/vision-mechanisms.md).

## Frozen comparisons

- Bicubic: PyTorch `interpolate` against channels-last with both layout copies
  charged, and two passes (horizontal, then vertical). All use the same
  `align_corners=False` and antialias setting, and return contiguous NCHW data.
  Bilinear interpolation is an intentionally different-output control.
- SigLIP loss: construct dense signs per call, reuse precomputed signs, use the
  native BCE-with-logits routine with cached labels, or compute negative-pair
  log-sigmoids and overwrite the diagonal with positive-pair log-sigmoids.
  The diagonal method retains the reference's row reduction and final mean.
  Softmax cross entropy is an intentionally different-objective control.

The loss is `-mean_i(sum_j(logsigmoid(sign_ij * logits_ij)))`, where the sign is
positive on the diagonal and negative elsewhere. This is an independent
implementation of the formula in [SigLIP](https://arxiv.org/abs/2303.15343), with
the reduction order used by
[Transformers 4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/siglip/modeling_siglip.py).
This pilot has square, paired batches; distributed negatives, rectangular
batches, encoders and higher-order gradients are outside its scope.

All implementations are fixed before discovery. Sixteen discovery cases precede
32 evaluation cases in each of two fresh processes with new input seeds. There
is no implementation revision or winner selection after discovery. Evaluation
covers irregular sizes, degenerate image height, contiguous/strided inputs,
upsampling/downsampling, both antialias settings, FP32/FP64 losses and
forward-only/forward-plus-backward calls. Three distributions test normal values,
large magnitudes, structured values and signed zero. Timings use normal values.

Each method must preserve finite output shape/dtype, inputs and retained outputs.
Bitwise checks and the predeclared numerical relation
`abs(candidate-reference) <= 1e-6 + 1e-5 * abs(reference)` are reported separately.
No observed maximum error is used to set tolerances afterward. Public bicubic
outputs are contiguous; physical gradient strides are not an observable.

Nine randomized, interleaved rounds include allocations, Python dispatch,
layout conversion and (where requested) autograd. Each round batches calls
targeting 5 ms, capped at 100 calls. Input generation and checks are outside
timing. Intervals are descriptive paired bootstraps within one process, not a
deployment certificate. Cached baseline setup and persistent tensor bytes are
recorded; peak memory is not measured. Persistent buffers assume unchanged
shape/dtype/device/label semantics and would need bounded caches in a product.

## Reproduction

The recorded environment uses Python 3.11, PyTorch 2.14.0, CPU execution and one
intra-op/inter-op thread. Existing GPU installations are not used. On an Apple
Silicon Mac, from the repository root:

```sh
python3.11 -m venv .venv-vision
.venv-vision/bin/python -m pip install 'torch==2.14.0' 'numpy==2.4.6'
LOSSLESS_TEST_VISION=1 .venv-vision/bin/python -m unittest discover -s lossless-product/tests -p test_vision_mechanisms.py -v

set -o pipefail
vision_run=research/results/vision_mechanisms-replay
mkdir -p "$vision_run"
.venv-vision/bin/python -u lossless-product/experiments/vision_mechanisms/study.py --split discovery --output "$vision_run/discovery" --log "$vision_run/discovery.log" 2>&1 | tee "$vision_run/discovery.log"
.venv-vision/bin/python -u lossless-product/experiments/vision_mechanisms/study.py --split evaluation --seed 20261109 --output "$vision_run/final-1" --log "$vision_run/final-1.log" 2>&1 | tee "$vision_run/final-1.log"
.venv-vision/bin/python -u lossless-product/experiments/vision_mechanisms/study.py --split evaluation --seed 20261209 --output "$vision_run/final-2" --log "$vision_run/final-2.log" 2>&1 | tee "$vision_run/final-2.log"
.venv-vision/bin/python lossless-product/experiments/vision_mechanisms/summarize.py "$vision_run" "$vision_run/evidence.json.gz"
```

Use a new result directory for another run. The runner refuses to overwrite an
existing run, freezes its source and plan, writes observations after every case,
and prints timestamped progress and result/log paths. Neither network access nor
model weights are needed after installing PyTorch. Ordinary product tests run
only the dependency-free protocol checks; tensor checks require the explicit
`LOSSLESS_TEST_VISION=1` opt-in.
