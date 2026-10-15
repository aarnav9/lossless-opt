# CPU request overhead — 14 October 2026

All five proposed families were tested sequentially. The retained combination
reuses a duplicate result observation and uses raw array transport. Persistent
dispatch, prepared guards and reusable transport buffers did not earn retention.
Direct arrays are an explicit, separately qualified input mode, because avoiding
files helped some workloads and hurt others. No arithmetic, precision or tolerance
was changed. Every completed study matched exact observations and returned bits.

## What changed and what stayed fast

The frozen combination passed held-out file-request regression and memory checks.
Financial preserved its existing source-optimization advantage. Pybaselines had a
useful runtime improvement, but its small additional source advantage under the
new runtime did not pass the statistical gate.

| Held-out file requests | New optimized runtime / previous optimized runtime | Original source / optimized source, both using new runtime | Decision |
| --- | ---: | ---: | --- |
| NumPy Financial | 1.0069× | **1.3544×**, p=0.015625 | Existing source edge preserved; no demonstrated extra runtime gain |
| pybaselines | **1.1835×**, p=0.015625 | 1.0391×, p=0.34375 | Runtime win; source advantage not independently qualified for this prepared suite |
| Array-copy controls | 1.2344×, p=0.015625 | Identical source in both arms | Mechanism evidence only |

The first ratio means old time divided by new time, so 1.1835× corresponds to about
15.5% less request time. All cases and request positions passed the 3% runtime
regression limit, and memory ratios stayed below 1.2. Financial's case ratios were
1.0026×, 1.0004× and 1.0179×. Pybaselines' were 1.0667×, 1.2218× and 1.2720×.
The array controls use exactly the same copy function in both source arms; apparent
source differences there are measurement variation, never source-optimization wins.

These comparisons deliberately force execution of the previously selected source
candidate inside the research harness. The old pybaselines bundle had not qualified
optimized prepared execution. Thus the 1.1835× result is a comparison of the same
source candidate under two runtimes, not a claim about a previously enabled public
pybaselines deployment. Comparing reference source under both file runtimes gave
1.1594×, but p=0.109375 did not meet the statistical gate. Production still requires
its own prepared certificate before enabling optimized source.

| File case | Previous optimized request median | New optimized request median |
| --- | ---: | ---: |
| Financial width 8 | 113.778 ms | 112.273 ms |
| Financial width 1 | 90.500 ms | 89.867 ms |
| Financial width 32 | 81.688 ms | 81.958 ms |
| pybaselines small | 3.054 ms | 2.878 ms |
| pybaselines weighted | 17.360 ms | 14.345 ms |
| pybaselines branches | 8.651 ms | 7.090 ms |

Ratios use paired sums followed by geometric means; they need not equal ratios of
the individual medians. Startup is excluded from request time and recorded
separately. Median optimized startup was 337→339 ms for Financial and 890→903 ms
for pybaselines in this comparison. These are repeated-request results, not faster
cold launches.

## All five families, including negatives

Each discovery comparison used six paired rounds and three requests per case.
Ratios below compare the same optimized source across runtimes unless stated.

| Family | Financial | pybaselines | Array controls | Outcome |
| --- | ---: | ---: | ---: | --- |
| 1. Reuse duplicate result encoding | 0.9977× | 1.0747× | 1.0783× | Large-array controls helped; pybaselines p=0.109375 was inconclusive. Included in combined validation. |
| 2. Persistent request thread | 0.9715× | 0.7233× | 0.8965× | Rejected; keeping one thread was slower here. |
| 3. Prepared metadata predicates and finite-scan scratch | 0.9772× | 0.9665× | 0.9994× | Rejected; no benefit over current guards. |
| 4. Direct arrays with existing NumPy framing | 1.0190× | 1.2321× | 0.7772× | Mixed. Old arm includes staging caller-owned arrays to files. Try raw transport before final assessment. |
| 5a. Raw output frames, file inputs | 0.9935× | 1.2022× | 1.0722× | Promising; included in combined validation. |
| 5b. Reusable frame buffers versus raw frames | 0.9846× | 0.9803× | 0.9867× | Rejected; measured incrementally against 5a, not against the old runtime. |

The first observation run completed the real projects but stopped when the
synthetic control's declared thread-environment hash was wrong. The harness was
corrected and the entire family replayed. The failed attempt remains in the
archive; no candidate or threshold was changed to repair it.

## Direct arrays: useful, conditional

The frozen live-value combination added raw transport and observation reuse to the
direct-array API. Its old-runtime control includes saving live arrays to `.npy`
because that API requires files; the new reference gets the same direct-array API
as the optimized source. Input creation is outside both timers.

| Held-out live-value boundary | Runtime ratio | Source edge with same new API | Gate outcome |
| --- | ---: | ---: | --- |
| Financial | 1.0056× | **1.3468×**, p=0.015625 | Existing source edge preserved; no extra runtime gain established |
| pybaselines | 1.5934×, p=0.015625 | 1.0952×, p=0.015625 | **Full gates failed** despite favorable averages |
| Array-copy controls | 0.9273× | Identical source | Slower; no broad live-value win |

Pybaselines' weighted and branch cases improved 2.2124× and 1.7521× against staging.
Its small case averaged 1.0436×, but one request position was 0.9690×, just outside
the 1/1.03 regression floor. Source-edge positions of 0.9333× and 0.9448× also failed.
The large array controls measured 0.9557× and 0.8999×. Memory passed. These failures
are not overridden by the aggregate speedup, and the input mode is not selected
automatically. No numerical or quality-only candidate emerged: all completed
variants were exact on this suite, including the slower variants.

## Protocol and scope

Baseline: commit `d13261f57f9f706c61da464f5fe6efd09aaf1d9b`, frozen before edits.
Runtime stages were copied before execution, so authoring the next experiment did
not change the code of a running experiment. Application source candidates stayed
fixed. Financial inputs and pybaselines signals changed using seed `20261014`.
Selection used discovery cases; the combination was frozen before either held-out
measurement. Runs were sequential, without concurrent benchmark jobs or model calls.

Every round used four independent controllers: old reference, old optimized,
new reference and new optimized. Execution and case order rotated. Each request
was checked against two fresh old-reference runs using both observation
fingerprints and actual returned-value hashes. Later requests also had to leave
previously returned values intact. A request includes input loading or staging,
metadata/domain checks, fresh seeded factory creation, calls, input/output/state
observations, cleanup, serialization, transport and decoding.

The source edge always compares implementations with the same new runtime.
Per-case speed, per-request-position regression, sign-test evidence above 1.02×
and memory are reported independently. Research aggregate statistics include
synthetic controls; use the per-project decisions for application claims. Initial
discovery reports did not include position summaries; final reports do. There were
eight completed comparison studies, one preserved setup failure and **3,960 paired
measured requests**, plus separate fresh-reference checks. This is a small, local
CPU pilot, not a universal speed or correctness proof or a multiple-testing-adjusted
claim across all experiments.

Memory is the sum of parent high-water RSS and worker/process-tree peaks. Shared
pages may be counted twice; 20 ms sampling may miss brief descendants. Startup and
import preparation are measured separately. Platform and package fingerprints,
per-round observations, timings, peaks and startup values are in the archive.

## Product integration and evidence

Default prepared requests now use observation reuse and raw frames. Before/after
input snapshots, end-of-sequence retained-output snapshots, cleanup, finite-domain
checks, fresh factories, timeouts and no-retry behavior remain enforced. Raw
decoding creates owned arrays, so later requests cannot overwrite earlier returns.
The negative mechanisms are accessible only through private research flags.

`run-project --prepared --prepared-inputs values` and
`PreparedApplication(..., input_mode='values').run_values(...)` expose the copied
value boundary. Its `prepared-values.json` certificate is separate from the file
certificate and binds the runtime feature set. Missing, mismatched or stale evidence
cannot enable optimized execution. See the [API and ownership contract](cpu-projects.md#pass-live-arrays-without-staging-files).

The local full suite passed **231 tests with 8 optional-dependency skips**.
Added checks cover mutation and retained observations, NaN payloads/signed zero,
endianness/layout, frame corruption, owned arrays, guard mutations, timeout cleanup,
certificate isolation and actual direct-value qualification.

The public workflow was then exercised without research admission overrides.
Fresh Financial inputs (seed `202610143`, three held-out widths) qualified
**1.3677× callable speed and 1.3452× complete live-value request speed**. Pybaselines
qualified 1.3029× callable speed, but its live-value 1.0301× request ratio had
p=0.34375; the public workflow retained the reference for that input mode. These
integration runs confirm both successful deployment and conservative retention.

- [Complete reports, source snapshots, runtime patches and logs](evidence/cpu-request-runtime.json.gz)
- [Sequential harness and reproduction instructions](../experiments/prepared_runtime/README.md)
- [Previous prepared/source study](cpu-prepared-study.md)

Every archived runtime patch was reapplied to the frozen baseline and checked
against its saved Python source hashes. Large generated `.npy` datasets stay in
the local research directory; `restore.py` regenerates them and checks their complete
file hashes against the recorded inventories. Source snapshots include their
upstream licenses, including pybaselines' bundled-code notices.
