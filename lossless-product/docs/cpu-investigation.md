# CPU investigation in application search

Application profiling and source search include a versioned, sourced CPU
investigation catalog. The author receives detected hardware, discovery
measurements and explicit unknowns. Reports distinguish reviewed ideas from
evaluated proposals. Coverage does not establish that every optimization was
tried, that a candidate is fastest, or that its code has been formally verified.
This feature is in source, not the published v0.1.0a1 wheel.

## Configuration

Application jobs default to `profile.architecture: true`. Source-search plans
separately default to `architecture: true`. Set the relevant setting to `false`
to omit the added author context and hardware query. No new hardware file is
required. Profiling still makes no model call unless assessment is configured.

The 30-minute total search default is unchanged. Collection and additional author
work consume that same budget. The CPU query is capped at five seconds and clipped
to remaining time. Clean application timings exclude diagnostic collection.

Profiles save `hardware.json`, `cpu_catalog.json` and `checklist.json`. Search
freezes `author_context.json` in its manifest and writes updated coverage in
`checklist.json` and `report.html`. Provider calls retain exact contexts/responses.

## Collected evidence

| Evidence | Boundary |
| --- | --- |
| CPU, features, core/cache topology and page size | Allowlisted `sysctl` or `lscpu`; missing or permission-denied facts stay unknown |
| Package versions | Selected application interpreter metadata, distinguished from controller fallback |
| Loaded libraries and thread pools | Attribution worker; optional `threadpoolctl`, with paths removed; configured limits are not observed concurrency |
| Input shapes, dtypes, strides and contiguity | At most 32 discovery arrays per sampled call and eight calls per case; no array contents or pointers |
| Process CPU time and page faults | Instrumented calls include profiler overhead; page faults are not translation-cache misses |
| Call counts, self/cumulative time, setup, RSS and traced allocations | Existing separate clean timing, attribution and memory processes |

The hardware query performs no GPU dispatch, model loading, network retrieval or
microbenchmark. Installed packages need not be used; the installed compiler need
not have built a loaded library. Logical array bytes are not physical traffic.

PMU counters, assembly/compiler diagnostics, bandwidth, register spills, core
placement and thermal behavior remain unknown. The catalog states what evidence
would be needed; automatic Instruments, `perf` and LLVM-MCA adapters remain future
work. These measurements are not inferred from Python call stacks.

## Areas and interactions

Version 1 covers 22 areas: work performed, runtime overhead, allocation/ownership,
layout, caches, memory access, address translation, instruction delivery, branches,
execution dependencies, registers, SIMD, arithmetic semantics, fusion, parallel
scheduling, core topology, NUMA, compiler output, libraries, operating conditions,
x86 details and Arm implementation details.

Each card names applicability, evidence needed, transformations, obligations,
validation and interacting cards. Fusion links allocation, memory, registers and
arithmetic: removing a temporary can save traffic while increasing register
pressure or erasing a required rounding boundary. Authors receive candidate
feedback and are prompted to reconsider these interactions after changes.

CPU-family and reported NUMA-node screens are deterministic. General categories
remain investigations even when their benefit or editability is unknown. A shared
Arm ISA does not make Apple cores Neoverse cores; x86 timings do not describe M2.

## Coverage semantics

The first valid assisted source-author response must account for every card.
Later reviews may be incremental. Author context keeps the latest two reviews
and attempts per card plus the current best eligible candidate, with compact
timing summaries and omission counts. Full reviews and timings remain on disk.
Hypotheses and author-proposed exclusions must
cite supplied evidence IDs. Validators reject invented IDs, duplicates and
structured claims of proof or acceptance. Prose remains unverified advice.

| Status | Meaning |
| --- | --- |
| `unknown` | No review or evaluated attempt |
| `not_applicable` | A deterministic hardware screen excludes this category |
| `reviewed_unverified` | Author advice exists; no evaluated implementation is attached |
| `attempted_not_evaluated` | A tagged submission did not complete a recognized comparison, for example a duplicate |
| `evaluated` | A tagged candidate completed a matching, mismatching or unstable-reference comparison; inspect its outcome |

The author tags the hypotheses its edits exercise; the independent evaluator
attaches actual results. An exactness failure is an evaluated attempt, not a
successful transformation. A combined candidate's timing does not establish
which mechanism caused a gain. Search lists unreviewed/untested IDs and its stop
reason, marking unfinished work deferred at that stop. Coverage never changes
the frozen correctness, performance, startup or memory gates.

Assessment-only `profile --explain` receives the catalog but does not return
structured source-search reviews; its coverage remains untested.

## Lean boundaries

Cards link existing `Fusion.lean` and `Movement.lean` statement names and source
identities. These address exact storage, explicit rounding, expression trees,
vector partitioning and a restricted logical-access lower bound. This integration
does not generate/check new proofs: candidate proof status stays `not_checked`.

Implementation evidence requires trusted statements, checked assumptions and
dependencies, machine-range/ownership semantics and correspondence to emitted
code. Compiler/runtime behavior remains a separate boundary. A logical movement
bound is not a bandwidth measurement or a proof of minimum execution time.
See [the formal-reasoning design](design.md#how-formal-reasoning-participates) and
[Lean proof validation](https://lean-lang.org/doc/reference/latest/ValidatingProofs/).

## Sources

The catalog contains original summaries, source links, applicability, reviewed
versions/dates and limitations. Each run freezes its content identity. It does
not download manuals into prompts or bundle restricted instruction tables.

- [Apple guide overview](https://developer.apple.com/documentation/apple-silicon/cpu-optimization-guide?changes=_4_4&language=objc) and [WWDC25 Instruments](https://developer.apple.com/videos/play/wwdc2025/308/). The full guide requires an account/agreement and is not bundled. The session's Processor Trace requires M4/A18-class hardware, so must not be assumed available on M2.
- [Intel manuals](https://www.intel.com/content/www/us/en/developer/articles/technical/intel64-and-ia32-architectures-optimization.html) and [top-down methodology](https://www.intel.com/content/www/us/en/docs/vtune-profiler/cookbook/2024-2/top-down-microarchitecture-analysis-method.html).
- [AMD Zen 5](https://docs.amd.com/v/u/en-US/58455_1.00) and [Arm Neoverse V1](https://developer.arm.com/community/arm-community-blogs/b/servers-and-cloud-computing-blog/posts/arm-neoverse-v1-top-down-methodology), applicable to their named implementations.
- [Agner Fog](https://agner.org/optimize/) and [uops.info](https://www.uops.info/table_overview.html) for x86 guidance and instruction measurements.
- [LLVM-MCA](https://llvm.org/docs/CommandGuide/llvm-mca.html) for scheduling-model estimates, not physical measurements.
- [Berkeley Lab Roofline](https://amcr.lbl.gov/departments/computer-science-department/ppan/roofline-performance-model/) for compute/traffic analysis under stated ceilings and memory levels.

The [matched pilot](cpu-checklist-study.md) measures the joint effect of extra
context and reviews, including their cost. Repeated searches, broader repositories,
optional PMU/compiler adapters and implementation-linked proof checking remain
the next stages.
