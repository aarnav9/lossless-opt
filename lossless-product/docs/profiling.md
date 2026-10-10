# Profile a supported workload

`lossless profile` measures a workload without searching, installing proof tools, or changing acceptance. Registered adapter profiling does not invoke a provider. It writes `report.html`, `report.json`, a sanitized `profile_job.json` and a timestamped `run.log`. Use the HTML report to compare startup, repeated execution, memory and host hotspots; retain JSON for raw samples.

**Existing application jobs:** current source also supports callable application
profiling, with a different report schema and optional `--explain` model assessment.
See [application profiling](application-profiling.md) for its measurement boundaries,
configuration and live external-repository results. The remainder of this guide
describes registered native/MLX adapters.

```sh
lossless init --template softmax --output ./softmax-demo
lossless profile ./softmax-demo/lossless.json --repeats 7 --budget 30s --output ./lossless-runs/softmax-profile
```

The native adapters use generated Gaussian float32 inputs for the configured **discovery** shapes and layouts. MLX uses the actual discovery request texts and token counts. Evaluation cases remain closed. These adapter paths are separate from the newer application callable profiler.

To compare an already exported implementation:

```sh
lossless profile ./softmax-demo/lossless.json --artifact ./softmax-artifact --repeats 7 --budget 30s --output ./lossless-runs/softmax-comparison
```

The artifact must match the adapter. Its files and runtime guards are checked using the normal deployment loader. A guarded reference fallback is reported explicitly. Comparing an artifact against a newly declared comparator does not retroactively change its original acceptance decision.

The Python entry point has the same behavior:

```python
import lossless
result = lossless.profile("softmax-demo/lossless.json", repeats=7, budget=30,
                          output="lossless-runs/softmax-profile")
print(result.summary["status"], result.directory / "report.html")
```

Use a fresh output directory. A separate worker runs under the total wall budget; expiration stops its process group and records `incomplete`. Profiling does not create an exportable optimization artifact. Reports contain workload metadata and local paths, so review them before sharing.

## Native measurements

The named `objective.comparator` is the reference. First-call and warm timings use the ordinary Lossless callable, including its input validation, binding and allocation. This matches new source native jobs' default `objective.timing_scope: "call"`. Bound-buffer calls are measured separately through `operation.bind(...)`; they match explicit `"bound"` jobs and historical preallocated results. Input generation, reference compilation and implementation loading are separately recorded. [Benchmark boundaries and representative workloads](benchmarking.md).

Warm samples are interleaved repeat-block averages, with their raw values and run order retained. Their p95 is descriptive variation among those averages, not a production request-tail percentile. The first invocation of each case occurs in a fresh profiling worker after libraries are loaded; this is not a guarantee of cold operating-system caches.

The worker records the process RSS high-water mark, including imports, input generation and correctness checks. It is not a per-call allocation measurement. A separate `cProfile` pass lists host self-time and inclusive time. Inclusive times overlap and cannot be summed. This instrumentation is excluded from ordinary warm timing.

An artifact comparison reports payback using its recorded search time, newly measured allocating-call savings, and the observed load/first-call excess over warm. Profiling-only compilation of the reference and input generation are excluded. These estimates assume the saving persists; they are not billing forecasts.

## MLX measurements

Use the pinned MLX 0.32.3 / mlx-lm 0.32.0 environment and a local model directory in the job. Optimized artifacts retain the exact model/runtime/device admission rules. The profile includes complete fixed-count generation with probability capture and final KV materialization, plus model-loading time separately.

```sh
lossless profile ./my-mlx-job/lossless.json --artifact ./my-mlx-artifact --repeats 7 --budget 60s --output ./lossless-runs/mlx-profile
```

The report includes per-request first-token and completion latency from batch start, tokens/second, peak warm MLX allocation, and four coarse wall-time stages: input validation/tokenization, execution, final state materialization, and decoding/response construction. Their aggregate fractions partition measured request time. Serial requests include time waiting behind earlier requests.

Artifact comparisons check sampled tokens, full captured probability arrays and active KV against stock serial before warm timing; token decisions are checked during repeats. The first calls share a loaded model, so their order influences observed startup. Cold memory measurements include the held reference outputs needed for exact comparison; use warm peaks when assessing repeated deployment calls. MLX allocation counters exclude memory not tracked by the allocator, including some driver/process memory.

Host profiling reports time spent executing or waiting in host calls. It is **not GPU kernel-level attribution**. The stage clocks do not synchronize every layer or change arithmetic. For kernel-level investigation, use a separate backend profiler after this report identifies a worthwhile request-level target.

Timing, memory and attribution are observations, not optimization acceptance or a universal equivalence proof.

## Recorded local validation

[Campaign 040 evidence](assets/profile-040.json) preserves a native copy profile, an eight-request MLX profile and a fresh graph deployment check on the pinned Apple M2 environment. The selected `decoder_026` artifact passed sampled token/probability/KV equality, eight continuations and cancellation fallback. The discovery profile's seven warm repeats measured 573.684 ms stock serial versus 287.794 ms for the artifact (1.993× throughput). This is one local batch comparison, not another independent acceptance test.

For the first request, median first-token latency increased from 19.230 ms to 91.363 ms; for the last request, it decreased from 472.527 ms to 91.361 ms. Warm MLX allocation peaks were 254.1 and 253.7 MiB. The artifact's execution stage occupied 98.8% of measured batch time; final state materialization occupied 1.05%. Host attribution cannot divide that execution stage into GPU kernels.

Measured payback was 129 identical batches, using the recorded 36.785-second search plus 39.704 ms of relative first-call excess. Shared model loading took another 417 ms and is reported separately. These values show why bulk throughput, queueing latency and setup must be assessed together. Native allocating and bound calls, all MLX samples/orders, phase durations and provenance are retained in the evidence file.
