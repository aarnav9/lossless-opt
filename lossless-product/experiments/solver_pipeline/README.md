# Measured-cost execution-plan pilot

Optional research experiment; no product dependency or default is changed.
Requires Python 3.11+, a C compiler and the separately installed dependencies in
`requirements.txt`. The pinned dependency set was tested with Python 3.11 on
Apple Silicon. GPU support is not implemented here.

The workload is a synthetic six-stage uint8 preprocessing-style pipeline:
clamp high, clamp low, invert, shift, add and XOR. Each contiguous region has
three implementations: allocating NumPy operations, fused native arithmetic,
and a native lookup table. There are 63 region choices. The LLM selects plans
from this fixed catalog; it does not generate new kernels in this first pilot.

## Experiment loop

1. Compile the native implementations and validate every region on all 256
   uint8 values with contiguous/strided input and independent-output checks.
2. Measure ordinary region calls on three discovery inputs. Persist the raw
   samples, measurement IDs and rounded nanosecond costs. Only the evaluator
   supplies costs; the author response schema accepts a hypothesis and IDs.
3. Give both author arms the same catalog, workload, budget and candidate cap.
   The assisted arm also receives a CP-SAT suggestion. Solver time counts
   toward its budget. Each suggestion excludes previously evaluated plans.
4. Measure every proposed complete plan, including allocation, input packing,
   dispatch, result construction and completion. Return the observed timing
   and prediction error to the author. The initial region costs stay frozen;
   there is no fitted cost-model update in this pilot.
5. Freeze both choices before opening held-out shapes/values. Compare against
   the NumPy reference and fully fused native/lookup controls. Identical plans
   share final samples, so arm labels cannot create a false incremental gain.

The inputs are already decoded CPU arrays. Fixture generation, correctness
checks, compilation, calibration, author calls and solver execution are outside
steady-state samples. Compilation and tuning are charged separately; Python
interpreter and module-import startup are not measured. Timed outputs are
discarded. Repeated warm inputs do not represent cold-cache latency, service
queueing or file-to-result performance. CPU dispatch is included; no CUDA
launch-time measurement is claimed.

Each fixed sequence contains one request at each declared size. Model cost is
the sum of measured region medians across that sequence. This is a serial
estimate: it does not model concurrency, contention or nonlinear interactions.
Do not add another launch or allocation penalty to these full-call costs.

The descriptive pilot gates require 2% sequence improvement, no call slower
by more than 3%, and a traced allocation peak within 2x reference plus 32 KiB.
Memory is measured separately with `tracemalloc`, not process RSS. Raw timing
samples are retained, but the pilot does not provide a statistical performance
certificate. A failed final gate retains the reference without reselection.

## Solver and Lean controls

CP-SAT chooses Boolean region variables with exactly one covering each stage.
It minimizes the frozen integer-cost sum. An independent dynamic program
checks the optimum. This simple chain is efficiently solvable by dynamic
programming; OR-Tools is an experimental integration point, not required to
solve this example. A wider graph with real coupled choices would be needed
to establish an advantage from a general solver.

With `--check-lean`, `PlanCost.lean` plus a generated witness verifies a
Bellman-potential certificate: every permitted edge respects the lower bound,
and the chosen valid path attains it. This verifies global optimality within
the supplied finite serial model independently of the CP-SAT status. It does
not prove that the measured coefficients are physical latency bounds, that
the native kernels implement their specifications universally, or that a
chosen plan is globally fastest on hardware. The checker reuses the existing
pinned Lean toolchain; it never downloads one. Omit the flag if Lean is absent.

## Run locally

From the repository root, use a separate environment (do not modify the normal
product environment):

```sh
python3 -m venv .venv-solver-pilot
.venv-solver-pilot/bin/python -m pip install -r lossless-product/experiments/solver_pipeline/requirements.txt
```

The fixed-proposal smoke run should take under two minutes. It exercises the
complete machinery but is **not an LLM-only versus LLM-plus-solver ablation**:

```sh
mkdir -p runs
set -o pipefail
PYTHONPATH=lossless-product/src .venv-solver-pilot/bin/python -u \
  lossless-product/experiments/solver_pipeline/study.py \
  --output runs/solver-smoke --log runs/solver-smoke.log \
  --check-lean 2>&1 | tee runs/solver-smoke.log
```

The live mode uses the repository's existing Codex/ChatGPT provider and saved
sign-in. It sends synthetic operation definitions, measured timings, candidate
IDs and discovery feedback to that provider. It disables author tools and
does not send evaluation results or model weights. A live run requires
authorization for that provider payload. Use a new output directory each time.

This bounded pair allows two proposals and 120 seconds total per arm, with a
45-second limit on each provider response. Expected runtime is two to four
minutes, depending on provider latency:

```sh
mkdir -p runs
set -o pipefail
PYTHONPATH=lossless-product/src .venv-solver-pilot/bin/python -u \
  lossless-product/experiments/solver_pipeline/study.py \
  --output runs/solver-live --log runs/solver-live.log \
  --author codex --model gpt-6-astra --effort medium \
  --max-candidates 2 --budget-seconds 120 --final-reserve-seconds 15 \
  --provider-timeout-seconds 45 --check-lean 2>&1 | tee runs/solver-live.log
```

For longer campaigns, increase the caps and run the foreground Terminal command
yourself. Both arm budgets must remain equal. The runner prints flushed UTC
progress for every catalog batch and proposal; the provider prints progress
every 30 seconds during a quiet response. Startup/completion print result and
log paths. `tee` creates the log; `--log` records the path rather than opening it.

`report.json` records source identities, compiler and dependency versions,
solver bounds, proof status, model errors, memory gates, accepted/retained plans,
tuning time and break-even sequence count. The original sources, frozen cost
catalog, per-candidate records and raw final samples are saved beside it.
Provider failures are explicit; `incomplete_live_pair` cannot establish efficacy.
Compare against `full_native`, `full_lut` and the dynamic-programming optimum,
not just the unfused NumPy reference. One live pair would still be exploratory.

## Recorded local pilot, 10 October 2026

The bounded live pair completed in 39 seconds with two valid GPT-6 Astra
responses per arm. Both arms proposed the same two plans and retained
`native_0_6`, the existing fully fused native control. Solver versus control
speedup was **1.000x**. This example demonstrated no incremental runtime
benefit from the solver. Provider latency varied; the small tuning-time
difference is not evidence of improved search efficiency.

CP-SAT and the independent dynamic program agreed on the 58,782 ns discovery
cost optimum. Lean 4.28.0 checked the generated potential certificate without
`sorryAx` or custom axioms. Complete-plan discovery prediction errors ranged
from -1.5% to +13.6% across the four measured proposals. That mismatch reinforces
the distinction between modeled optimality and observed hardware timing.

All 126 region correctness/ownership checks passed, as did final held-out
per-case and traced-memory gates. Nine pilot regression tests and six existing
provider tests passed. The [compressed evidence](../../docs/evidence/solver-pipeline-pilot.json.gz)
contains the report, raw region/final samples, candidate measurements, cost
catalog, generated Lean witness and checker output. It contains a single
sequential pair on a synthetic CPU workload; it establishes neither CUDA launch
behavior nor broader LLM-plus-solver efficacy.

References: [OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver),
[scheduling objectives](https://developers.google.com/optimization/scheduling/job_shop),
[Lossless timing boundaries](../../docs/benchmarking.md).
