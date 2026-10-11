# Sequential CPU optimization campaign

**Complete: 100 experiments, twenty per family, plus five frozen final evaluations.**
Each family ran sequentially and used earlier discovery outcomes to guide later
proposals. The compiler-feedback family includes twenty matched pairs, forty
tools-disabled author calls, explicitly approved by the user. Source/diagnostic
payloads went through the signed-in Codex provider; compilation, exactness checks
and timing ran locally.

The completed families found useful, exact speedups on particular workloads.
None passed every predeclared final gate, so their experimental implementations
have not replaced product defaults. Ratios below describe complete request calls
on new input values; setup is measured separately. They are not universal CPU
speedups or improvements against stock pybaselines.

| Family | Trials | Frozen finalist | Final call speedup | Final outcome |
| --- | ---: | --- | ---: | --- |
| Compiled ModPoly region | 20 | 20 | 1.0514× | Exact; flat-input regression |
| Persistent spectrum pool | 20 | 19 | 1.4001× | Exact; per-call regressions and memory limit |
| Ordinary native wrapper | 20 | 17 | 1.2524× | Exact relative to unchanged kernels; one large-copy call regressed |
| Explicit prepared metadata | 20 | 20 | 1.3139× | Exact; one mixed-metadata call exceeded the 3% regression allowance |
| Compiler-feedback authoring | 20 pairs | 16, diagnostics | 0.9743× | Exact; aggregate and several workload regressions |

## What helped

**Move a sufficiently large region into C.** Tiny clipping helpers lost to
binding overhead. Moving the complete iteration loop while preserving the same
OpenBLAS matrix products and dot-product reductions produced an exact discovery
gain around 10%. The final gain was 5.14%, with a 12% aggregate flat-case slowdown
and a worse individual call. A strided-input control caught an early unguarded
source pointer; later variants added a contiguous/aligned fallback and passed
the additional edge cases. This is evidence for larger compiled regions with
guards, not blanket replacement of NumPy operations.

**Use persistent processes for sufficiently large batches.** Four warmed workers,
one BLAS thread each, balanced scheduling, and a serial path for small batches
reached 2.119× on the final batch case and about 1.66× on weighted/long spectra.
The complete suite averaged 1.400×, but some individual calls regressed and the
conservative process-memory estimate exceeded the 512 MiB cap. Pool startup made
setup plus the three measured requests only 0.423–0.464× as fast as serial.
Summed process peaks double-count shared pages; they are not unique physical RAM.
This remains a throughput option for a long-lived, memory-budgeted service.

**Remove native wrapper work while keeping validation.** Raw-address ownership,
direct dispatch, cached constant buffers, and size-dependent domain validation
reduced normal-call overhead. The frozen winner reached 1.2524× overall;
sliced RMSNorm reached 1.561× and small softmax 1.424×. Large copy was 0.996×
overall, but one of its forty declared call positions measured 0.889× and failed
the unchanged gate. Independent numerical-oracle checks still passed. A separate
bounded-cache variant passed ownership, layout, mutation, fallback and retained
output checks; its final performance was not substituted for the frozen winner.

**Make metadata preparation explicit.** Ordinary scan/hash/LRU/front-cache variants
were slower than the existing bounded cache. Explicit owned grid/order handles
with polynomial precomputation were more promising. The selected two-descriptor
variant reached 1.3139× on fresh final inputs, including 1.985× for small signals
and 2.700× for flat signals. It passed setup, memory and eighteen lifecycle
checks. A mixed-metadata call measured 0.959×, just outside the 3% regression
allowance. Setup-inclusive final ratios were 1.049–1.065×. This is a different
API contract: callers prepare owned metadata and reuse handles with changing
signals/weights; it is not a transparent ordinary-call optimization.

**Compiler feedback produced local wins but no qualified global improvement.**
The final authored helper measured 0.9743× against the ordinary application
baseline. Small and mixed-metadata cases reached 1.0919× and 1.0901×, while
weighted, long, batch and flat cases measured 0.9553×, 0.8756×, 0.9346× and
0.8641×. Startup and memory passed, but aggregate speed and regression gates
failed. Setup-inclusive ratios were 0.949–0.988×. The selected implementation
preserved exact outputs, histories and ownership on the eight final profiles
and all ten additional edge-case controls, including strided inputs, signed zero,
extreme finite scales and tight stopping thresholds.
It therefore remains an experimental source, not a product default.

## Numerical and quality candidates

Fusing norm calculations with scalar accumulation changed output/history bits.
The straightforward reduction measured 1.0170× and a four-accumulator reduction
1.0399× in discovery, with maximum observed absolute errors approximately
6.38e-16 and 2.22e-16 respectively. Both fit the fixed numerical envelope
(`atol=1e-10`, `rtol=1e-8`) but failed the performance/regression gate. A scalar
matrix-product replacement was numerical and substantially slower, 0.6962×.
These are recorded numerical leads, not accepted lossless implementations.
No task-quality labels were available, so no quality-preservation claim is made.

## What the authoring comparison measures

Each compiler-feedback experiment has two OpenAI author calls: the same seed C
helper, Python binding, workload metadata and earlier discovery feedback, with
current profiler hotspots and compiler vectorization remarks added to one arm.
Both use the same model, reasoning setting and response allowance, and both
responses are saved before either is benchmarked. Authoring has no tools;
compilation, correctness checks and timing run locally. The comparison tests
whether supplying diagnostics improves proposals, not remote execution speed.
Because both arms use the provider, it does not measure provider authoring
against local-only authoring.

Five target workloads cycle through four rounds. A target's eligible winner can
become its next shared seed. Multiplying these local discovery ratios is only
a selection heuristic; it is not a validated suite-wide speedup. One frozen
source is finally compared with the ordinary application baseline on fresh
cases. Ratios between arms are descriptive: these are adaptive trials with
shared history, not independent replications of a general diagnostic-context
effect. The control is the generic helper-author context, not the earlier full
22-card application checklist.

Source review caught an attribution trap: trial 6 control changed a standalone
GEMV helper that this application's full-loop route never calls. Its measured
gain cannot be credited to that algorithm change; timing variation or code
placement are possible, unproven explanations. Trial 8 control also edited that
unused helper. This finding was supplied equally to both arms before trial 11,
recorded as protocol amendment 006, and did not change any result or gate.

All forty author calls completed, compiled and were evaluated; all forty
preserved exact discovery outputs. Each arm produced three discovery-eligible
candidates. Diagnostics advanced the shared incumbent three times, control
twice. The diagnostics/control ratio was 1.00995× overall, with descriptive
target ratios of 1.1655× small, 0.9473× weighted, 0.9543× long, 0.9923× batch and
1.0050× branches. This is not evidence for enabling diagnostic context as a
universally better default. Pair 16 diagnostics, which outlined the overlap
fallback into a cold function, was frozen for final evaluation. The 1.3791×
cumulative discovery estimate is target-specific and must not be presented as
its final product speedup.

Every call had a 240-second allowance. Actual author time totaled 24.13 minutes
for control and 18.86 minutes for diagnostics, excluding local evaluation.
Provider receipts reported 272,028 versus 317,188 input tokens and 56,821 versus
46,221 output tokens; input counts include 24,576 versus 49,152 cached tokens.
Dollar costs were not reported. The artifact audit passed all twenty pairs:
equal common context/settings, unchanged prompt/response/source/binary identities,
no author tool use, and both responses saved before measurements.

## Product work and validation

The changes below were tested in the development working tree. This research
publication includes the experiment sources and evidence; product integration
is separate, and the published `v0.1.0a1` release assets are unchanged.

Application source search now supports opt-in, declared C helpers alongside
Python body edits. The helper ABI, source bounds, compilation flags, build
diagnostics, generated wrapper, source/binary hashes, and patch contents are
recorded. Candidate binaries load in evaluation workers; normal independent
application gates still decide acceptance. See [application search](application-search.md).
This adds a tested candidate capability without promoting a failed performance
candidate. Native code remains trusted local execution, not a security sandbox
or an equivalence proof.

The campaign also adds exact/numerical/rejected classification controls,
input/retained-output checks, per-call regression gates, fresh final fixtures,
separate memory passes, native independent-oracle checks, and mutation/layout/
eviction conformance checks. The full product suite ran 173 tests: 169 passed and
four optional tests were skipped. Formatting/lint, wheel/source-archive inspection,
and clean installed replay, profiling, source-search rejection and exact export/load
checks passed. All four native-helper tests also passed against the installed wheel,
including a real application search through its final gate.

The initial build environment had an outdated setuptools; an already-installed
compatible backend built the distributions. Clean installation used a cached
public NumPy wheel after sandbox DNS prevented dependency resolution.

## Product priorities from these results

The strongest next product direction is an explicit prepared-metadata API:
it improved setup-inclusive performance and passed the ownership/memory checks,
with one remaining per-call regression to investigate on new fixtures. Keep
preparation cost and cache bounds visible in that API.

For ordinary native calls, combine the wrapper-overhead findings with a bounded
constant-buffer policy, then qualify that new combination independently. The
current results do not establish the performance of that combination. Persistent
pools belong in an explicit long-lived batch API with a worker/memory budget;
startup and the measured memory footprint make them unsuitable as a transparent
default for short requests.

Compiled helper support is now available to source search. Before automatic
deployment, a selected helper still needs reproducible packaging and validation
on its target CPU/BLAS stack. Keep compiler diagnostics optional until a new,
controlled study establishes their value; this run does not justify a universal
default or replacing failed final candidates after inspecting final inputs.

## Method and limits

The application comparator is pinned pybaselines 1.2.1 at revision
`6f8d927e08577a9c5af5f34129b14d57c12725f6`, with the prior continuation and
CPU-control patches. Native trials freeze the existing deployment wrapper and
keep its kernels unchanged. Exact wrapper equivalence therefore does not upgrade
softmax/RMSNorm's underlying numerical contract to mathematical exactness.

All measurements ran sequentially on one Apple M2 with one BLAS thread per
worker. Five discovery profiles use three requests with changing values; eight
fresh final profiles add new dimensions/seeds, float32, flat and sparse cases.
Final qualification uses sixteen paired rounds, at least 1.02× aggregate speedup,
sign-test p ≤ 0.01, no case or declared call position beyond 3% slowdown, and
startup/memory limits. Selection is frozen before final inputs open, with no
replacement selected after seeing a failed final. Setup/import and fixture
decoding are outside request timing; setup is also reported together with calls.

Identical-source controls were near 1× overall, but individual native-call
positions were noisy. The stricter per-position gate was retained anyway.
Consequently a rejected final is a failure to qualify under this protocol,
not proof that the candidate has no useful scope. Finite tests on one CPU do
not prove equivalence for every input or establish portability to other CPUs.

The [research harness](../experiments/cpu_100/README.md) documents commands,
lifecycles, controls and artifact layout. Complete local source snapshots,
proposals, outputs, timings, logs and protocol amendments are under
`.lossless/cpu-100-20261011/` at the repository root. The final portable evidence
is [portable compressed JSON](evidence/cpu-100.json.gz), including every trial,
the five final reports, author sources and receipts, conformance results and
protocol amendments. The [all-trial appendix](cpu-100-experiments.md) lists all
100 discovery experiments. These compact exports complement the full local
artifacts; they do not contain every timing-worker output array.
