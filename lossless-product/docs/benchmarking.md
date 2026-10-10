# Benchmark the interface the application uses

New source runs select native candidates using ordinary `operation(...)` calls
by default. Application search also rejects regressions at individual declared
call positions. These changes do not retroactively qualify earlier experiments.

## Measurement boundaries

| Path | Acceptance timing | Reported separately / excluded |
| --- | --- | --- |
| Native `objective.timing_scope: "call"` (default) | The shared exported callable: input/domain guards, allocations, pointer binding, copies and CPU dispatch | Kernel-only diagnostics; compilation, artifact loading and fixture generation excluded |
| Native `objective.timing_scope: "bound"` | Repeated bound calls with existing buffers | Binding, guards and allocation excluded; valid only for applications that actually reuse those buffers |
| Application plan `timing_scope: "calls"` (default) | All declared calls, including the first, internal setup and completion synchronization | Application import/factory setup reported separately |
| Application plan `timing_scope: "setup_and_calls"` | Application import/factory setup plus all declared calls | Interpreter/harness startup, fixture loading and correctness instrumentation excluded |

Native reference and candidate use the same Lossless public call boundary.
Library reference buffers no longer allocate the benchmark's red zones or poison
arrays in the deployment path. This comparison concerns the named library
comparator through Lossless, not arbitrary caller code or bare library timings.
The guarded native validation pass still uses red zones and checks input mutation.
The numerical oracle and tolerances have not changed.

For new public native jobs, the chosen boundary is frozen in resolved JSON,
worker jobs, provider context, reports and exported manifests. The primary
screening/confirmation samples and payback calculation use that boundary.
`kernel_only` measurements cannot override a failed ordinary-call comparison.
Direct low-level research specs retain their historical `bound` default and now
record it explicitly; set `timing_scope` to `call` to use the deployment boundary.

Application setup-plus-call timing sums each process's setup and sequence before
computing statistics. It does not add independently computed medians. Use this
scope for a one-shot job or a handle constructed for every declared sequence.
Use `calls` for an already-running application with a reused model/handle.
Both choices preserve per-case and per-call regression gates. A separate setup
increase limit still applies, and payback uses the selected lifecycle without
charging the same setup increase twice.

## Representative cases and acceptance

A faster total can conceal a slower request. New application search plans default
to `max_call_regression: 0.03`, alongside the existing 3% case regression limit.
Each declared call position is compared across paired fresh-process repetitions;
a candidate must pass both sequence and per-position gates. Missing per-call
timings cannot satisfy that gate. Reports retain raw per-call ratios and medians.
Very short calls can be noisy; choose allowances before search, report them, and
use enough repeats. These are sampled regression checks, not production-tail
guarantees. Historical direct comparison scripts only enable the new gate when
their frozen policy contains `max_call_regression`.

Example fields to merge into an application search plan:

```json
{
  "timing_scope": "setup_and_calls",
  "max_case_regression": 0.03,
  "max_call_regression": 0.03
}
```

Choose the entry function at the application boundary. If users pay for decoding,
padding, batching, transfers, dispatch and output materialization, those operations
belong inside that function. An observer is only for correctness inspection:
placing required application postprocessing there would exclude its time. A
function that receives GPU-ready tensors measures a different interface from one
that receives encoded files or host arrays. Report that distinction.

For batch transformations, freeze at least these regimes as separate cases:

| Regime | Purpose |
| --- | --- |
| No elements changed | Detect added no-op/assembly overhead |
| One element changed | Detect per-item overhead across mostly unchanged items |
| Some elements changed | Exercise mixed dispatch and fallbacks |
| All elements changed | Measure the bulk transformation opportunity |

Cross those regimes with small/large batches, supported shapes/dtypes/layouts,
and held-out values. For variable-length inference, include equal and ragged
lengths, short and long outputs, and actual stopping requirements. Do not invent
production frequencies: report each case and the equal-case suite score until a
real trace supports a weighted aggregate. Separate queueing/concurrency tests
are needed before calling an offline score a serving-latency result.

The application runner already waits for supported accelerator completion inside
call timers. Its explicit synchronization hook covers other runtimes. A new
CPU regression test simulates deferred completion; it is not CUDA validation.
Launch count and device traces remain useful diagnostic evidence, not substitutes
for timing the complete chosen boundary.

## Public suite selection, researched 9 October 2026

**TorchBench is the recommended next model-workload source.** Its collection
provides standardized interfaces to examples of PyTorch workloads and selectable
models/backends. Start with a small, declared inference subset compatible with the
actual test machine, then qualify it on the intended NVIDIA worker. The standard
`get_module()`/example-input interface does not by itself include a user's
preprocessing, queue or service. Add an explicit application wrapper where those
costs matter, and report an adapted result under its own name. Pin the upstream
commit, model files, runtime and cases before comparing candidates.
[TorchBench](https://github.com/pytorch/benchmark).

**Use MLPerf Inference as the serving-scenario reference.** Its LoadGen and
scenario rules define workload arrival and latency/throughput objectives, while
benchmark definitions include datasets and quality targets. This is the stronger
follow-up for service-level claims, with a larger setup/data/runtime commitment.
An application can adopt useful scenario ideas without claiming MLPerf compliance.
Its accuracy rules do not automatically satisfy Lossless's bitwise contract;
retain the declared independent correctness gate as well.
[MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/),
[submission and scenarios](https://docs.mlcommons.org/inference/submission/).

**Keep KernelBench for operator and graph diagnostics.** Upstream exposes
different timing methods, including host-time measurements. Choose and report the
boundary explicitly. KernelBench coverage does not establish full application
latency, and changing timing alone does not supply missing mixed-input regimes.
The existing campaign 049 remains a KernelBench-derived MLX pilot with its
original ready-tensor boundary; no old speedup is relabeled as an application gain.
[KernelBench evaluation guidance](https://github.com/ScalingIntelligence/KernelBench/blob/main/EVAL.md),
[local campaign 049](kernelbench-049.md).

These are researched suite choices, not newly executed or integrated campaigns.
There is no universal suite that replaces a user's representative application
trace. Keep the current public CPU application studies as an additional domain,
and use the same callable/sequence gates when adding a model workload. The
private clip-padding implementation and its NVIDIA measurements were unavailable
for this change; no replication or hardware qualification is claimed.

Before a larger campaign, publish its exact environment, upstream revision,
case/seed split, input/output boundary, baseline and alternative compiler mode,
startup lifecycle, repetition/ordering policy, correctness contract, regression
limits and search-cost accounting. Run long campaigns explicitly in a foreground
Terminal with unbuffered timestamped progress and a named log.
