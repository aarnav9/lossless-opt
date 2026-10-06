# External application profiling with a live Astra assessment

On 5 October 2026, the new application profiler measured unmodified pybaselines
v1.2.1 and automatically passed bounded discovery evidence to **GPT-6 Astra at
medium reasoning**, using the existing Codex ChatGPT sign-in. The full study
completed in **89.03 seconds**; the model response took **48.76 seconds**.

This validates profiling and the model-assessment handoff. No candidate source
was generated, applied, accepted or deployed, and no speedup was measured.

## Workload and measurements

The checkout is pinned to `6f8d927e08577a9c5af5f34129b14d57c12725f6` under
BSD-3-Clause, with its notices preserved. See the
[preceding intake study](application-intake-study.md) for repository provenance.

Each of AsLS, SNIP and ModPoly received two discovery cases: one synthetic
4,096-sample spectrum, and an explicitly declared sequence of eight calls on
the same 1,000-sample spectrum. A separate evaluation case remained unopened.
Inputs are JSON numeric lists, so the measured boundary includes the upstream
function's conversion to arrays. An application passing arrays directly may
have different timings and bottlenecks; the case counts are not production
traffic weights.

Each case used five fresh timing processes, one separate `cProfile` process
and one separate `tracemalloc` process: **42 fresh processes** in total.
All six cases passed exact observed repeatability across those passes. The
original source remained unchanged. Hardware/runtime were Apple M2, macOS arm64,
Python 3.11.3, NumPy 1.26.4 and SciPy 1.16.0, with Numba 0.61.2 available.

| Algorithm | Single spectrum median | Eight-call sequence median | Application setup median, single / sequence |
| --- | ---: | ---: | ---: |
| AsLS | 1.619 ms | 3.280 ms | 551.05 / 560.27 ms |
| SNIP | 1.429 ms | 5.002 ms | 560.05 / 561.75 ms |
| ModPoly | 1.122 ms | 2.617 ms | 551.17 / 550.00 ms |

The single and batch cases have different per-call input sizes; these columns
are not a batching speedup comparison. Setup includes application import/factory
and synchronization, but excludes interpreter/harness imports. Call clocks
exclude fixture loading and observation. Observation between sequence calls can
still perturb caches.

Separate profiling time was approximately 1.10–1.47 times clean median sequence
time. These are noisy instrumentation-overhead diagnostics. RSS high-water marks
were approximately 133–140 MiB and include imports/inputs, not just algorithm
allocations. Full clean samples, traced-memory sites and host self/inclusive
times are in the evidence bundle.

Measured host attribution points to different next investigations:

- **AsLS:** `solveh_banded` accounted for about 55% of instrumented self time in
  the single case and 41% in the sequence. A generic Python-loop rewrite would
  miss much of this cost.
- **SNIP:** its clipping loop and SciPy's uniform filter were prominent. The
  self-time fractions do not by themselves establish allocation or bandwidth limits.
- **ModPoly:** repeated polynomial setup/pseudoinverse, iteration/convergence
  work and conversions appear in the profile. This was selected for the live
  model assessment; the other two profiles used no model call.

## What Astra returned

The workflow sent the ModPoly profile and bounded frozen-source excerpts without
raw input arrays or evaluation cases. Codex CLI 0.159.0 used `gpt-6-astra` and
`model_reasoning_effort="medium"`; tools/repository access were disabled.
One completed response passed schema validation and all evidence-ID checks.

Astra proposed four experiments:

1. Amortize application startup if the real deployment repeatedly starts a process.
2. Retain compatible polynomial state, checking grid/order/weights/dtype and masking
   before reuse and preserving ownership of previously returned results.
3. Investigate the repeated convergence-check path while preserving tolerance,
   reduction behavior and iteration/stopping decisions.
4. Investigate reusable clipping/weighting scratch arrays without changing aliasing
   or the lifetime of the previous baseline.

These are advisory hypotheses. The response explicitly requested independent
timing/correctness checks and did not declare an accepted optimization or forecast
a speedup. It noted measurement variability, instrumentation overhead, uncertain
production frequency, limited memory counters and floating-point risks.

The receipt records **20,952 input tokens and 1,344 output tokens**. Dollar cost
is unknown; this used subscription allowance rather than an API-key connection.
The model response allowance was 30 minutes and was not exhausted. Current
[official model documentation](https://developers.openai.com/api/docs/models/gpt-6-astra)
lists medium reasoning support; the recorded live receipt establishes access
for this local account/run, not universal account availability.

## Validation and limits

Product tests cover discovery-only profiling, stateful sequences without hidden
warmup, observer cost outside timing, allocation evidence, deadline exhaustion,
unstable references blocking assessment, configured assessment, disabling it,
invented citations and attempts to return acceptance fields. A subsequent review
also added a regression for applications named `worker.py`, preventing collision
with the profiler's support module. That import-name fix does not change the
recorded upstream workload; the evidence identifies the runner bytes used.

Small MLX and PyTorch fixture profiles also passed. The MLX fixture exercised
materialization/synchronization. The PyTorch fixture calculated a CPU bfloat16
transpose and exercised MPS synchronization; it is not a GPU algorithm benchmark.
CUDA execution and GPU kernel-level attribution were not validated here.

The feature is generic callable profiling with a model assessment, not automatic
workload invention or a source-editing optimizer. Complete-application candidate
qualification remains a later workstream.

## Reproduce

Use the clean checkout and environment described in the
[intake reproduction guide](../experiments/application_intake/README.md), with
current Lossless source installed and Codex signed in through ChatGPT:

```sh
set -o pipefail
python -u lossless-product/experiments/application_intake/profile_study.py \
  --project .lossless/external/pybaselines-v1.2.1 \
  --output .lossless/external/pybaselines-profile-reproduction \
  --model gpt-6-astra --effort medium \
  2>&1 | tee .lossless/external/pybaselines-profile-reproduction.log
```

Expected runtime is roughly 1–5 minutes when the provider responds like this run;
its default maximum response allowance is 30 minutes. `--llm-timeout` changes
that allowance, and `--no-model` runs only deterministic profiling. Output paths
must be fresh. The script prints flushed, timestamped progress and result/log paths.

[Curated evidence](evidence/application-profile-pybaselines.json) includes raw
timing samples, hotspots, memory evidence, runner/source identities, full model
assessment and receipt. Raw local profiles, source snapshots, prompts, responses,
generated jobs and process logs are under `.lossless/external/pybaselines-profile-01`.

[Product usage and measurement boundaries](application-profiling.md).
