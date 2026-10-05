# Campaign 049: KernelBench-derived MLX pilot

**Completed:** all 20 tasks passed final numerical/memory checks; two passed their
full per-task speed gates. The selected source averaged 1.149× the frozen
references, but the full-suite gate failed. [Results, graph and evidence](../../docs/kernelbench-049.md).
No product recipe was promoted.

Twenty tasks from [KernelBench](https://github.com/ScalingIntelligence/KernelBench)
revision `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`: eight operators, eight fused
computations, and four graphs (MLP, LeNet, Inception module, Fire module).
Unchanged task sources, per-file hashes and the MIT license are in `upstream/`.
`tasks.py` replaces upstream default sizes with an explicit M2-sized suite. This
is a **KernelBench-derived MLX study**, not an official upstream CUDA score.

## Fixed protocol

- Float32 numerical contract: `abs(candidate-reference) <= 1e-4 +
  1e-4*abs(reference)` for every element, with matching shapes/dtypes and finite
  outputs. This is not Lossless's bitwise inference contract.
- Independent CPU PyTorch execution of the original `Model` classes supplies
  input/parameter fixtures and expected outputs. Every method is checked against
  both that oracle and the fixed direct MLX port. No trained weights are downloaded.
- Forty discovery cases: two shape/layout profiles per task, each with uniform,
  normal, zero, alternating-sign and near-constant inputs. The second profile uses
  transposed MLX views. Forty final cases use new dimensions, weights and seeds;
  the final oracle is generated only after sealing the selection.
- Reference screening compares eager MLX, `mx.compile`, and shapeless compilation.
  Freeze one fastest correct method per task by the geometric mean across its two
  discovery cases. The author receives source, hardware, profiles and feedback.
- One Astra `xhigh` trajectory, a shared 60-minute maximum, 30-minute response
  cap, at most 16 calls. Authoring and candidate evaluation share the budget;
  initial screening/final evaluation are outside it. Reserve 120 seconds for
  measurement; do not start a call with less than another 120 seconds remaining.
  Late responses/measurements are rejected. Invalid proposals are retained.
- The deadline uses macOS `mach_continuous_time` (Linux `CLOCK_BOOTTIME`), including
  sleep. Returned code may implement real MLX/Metal changes and prepack fixed
  parameters. It may fall back to the supplied reference. No output/input-derived
  cache, reduced precision, evaluator changes or global monkeypatches are allowed.
- Select one complete module with the best eligible discovery mean above 1.0;
  otherwise select the unchanged reference fallback. Eligibility requires all 40
  cases correct and warm peak MLX memory <= reference ×1.10 +16 MiB.
- Two fresh final processes, eleven randomized paired timing blocks each. Full
  confirmation requires discovery and both finals: geometric mean >=1.02 and
  every case's lower paired bootstrap 95% interval >1. Per-task confirmation uses
  both cases for that task. No replacement selection after seeing final results.

Calls start with ready MLX tensors and end with a synchronized, materialized MLX
output. Timing includes Python dispatch and layout/weight transformations done in
the call, but excludes NumPy conversion, fixture creation and correctness checks.
Each timing block averages 1–32 individually synchronized calls, calibrated from
the fixed eager reference before timing. Uniform, normal and alternating inputs
rotate identically for all methods. Final timed outputs are checked as well.
Memory includes shared live inputs, parameters and validation arrays. Constructor,
module import and per-case first calls are recorded separately; these first calls
share a worker that has already run the direct port, so they are not isolated
GPU cold starts. Fixed weights can be prepacked, with that cost in setup/first-call.

The worker verifies unchanged inputs, parameters and previously returned outputs.
The AST/import guard catches accidental scope errors; it is not a hostile-code
sandbox or proof of equivalence. The source must be reviewed before interpreting
its result. This is a finite numerical experiment and automatically promotes
nothing into the product. One trajectory cannot establish author repeatability,
model rankings, or optimizer transfer to unseen task families. Compiler screening
is a simple non-LLM control, not an equal-cost randomized search arm.

## Run

Requires the existing Apple M2 environment with MLX 0.32.3, NumPy and CPU PyTorch,
plus the Codex CLI signed in through ChatGPT. Model authoring consumes subscription
allowance; dollar cost is unknown. The run does not use API-key billing or rent a GPU.
From the repository root, use a new output directory:

```sh
export PYTHONPATH="$PWD/lossless-product/src"
.venv-fast-mlx/bin/python -u lossless-product/experiments/kernelbench_049/study.py freeze --output research/results/kernelbench_049 --seconds 3600
set -o pipefail
.venv-fast-mlx/bin/python -u lossless-product/experiments/kernelbench_049/study.py run --output research/results/kernelbench_049 2>&1 | tee research/results/kernelbench_049.log
```

Allow approximately 1–1½ hours for the frozen run; initial development/porting is
additional. Progress is flushed and timestamped every task or at most 30 seconds.
Result and log paths appear at startup and completion. To check the full discovery
ports, unchanged candidate and deliberate wrong-output control without any author
call, use `smoke --output research/results/kernelbench_049_smoke_new`; the recorded
development smoke took approximately 26 seconds. CPU-only protocol regression
tests are in `tests/test_kernelbench_study.py`.

The controller freezes itself, the reference, evaluator, task sources and product
transport before running. All original evidence remains in the ignored output
directory, including numerical fixtures. Published records should retain failures,
source/fixture identities and raw timing samples, redact local paths, and never
include authentication state or raw CLI stderr.
