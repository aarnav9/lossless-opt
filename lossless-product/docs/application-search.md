# Application source experiments

[Pybaselines study: live authoring, noisy controls, and a scoped 1.058× result](application-search-study.md).
[Broader follow-up: 1.163× beyond the stronger baseline, with a failed allocation gate](application-broader-study.md).
[Continuation with memory feedback and new final cases](application-continuation-study.md).

`lossless search-application` extends application intake and profiling with a
Codex propose → patch → measure → revise loop. It is an experimental source
workflow for trusted Python callables. The original checkout stays unchanged;
proposals execute in fresh copies with local user permissions. These processes
are **not a security sandbox**.

For recognized pinned pybaselines sources, `--retained` replaces `--plan` with
local proposals composing retained source edits, strict compiled helpers,
bounded metadata and independent worker pools. It makes no model calls. See
[the fresh 1.317× exact integration result and reproduction](research-integration.md).
Other projects can use the generic `lossless.execution` and `lossless.retained`
APIs; application authors receive the exact mechanism catalog and must check
applicability. Numerical recipes do not enter exact acceptance implicitly.

Plans may set `discovery_resources: true` to reject startup/memory-ineligible
candidates before freezing a winner; this is enabled for retained composition.
Memory qualification requires the application extra's `psutil`. Separate memory
runs sample the parent and descendants every 50 ms and conservatively sum each
process's sampled peak RSS, counting shared pages more than once and potentially
missing brief peaks. Missing process access fails qualification. Instrumented
memory runs are separate from clean timings. The `setup_and_calls` timing
scope includes teardown, and independent pools are closed before process exit.

```sh
lossless search-application job/lossless.json \
  --plan search-plan.json --output runs/application-search
```

The Python equivalent is `lossless.applications.search(config, plan=plan_path,
output=run_directory)`. A plan explicitly identifies the functions the author may
edit and the domain/lifecycle being measured:

```json
{
  "schema_version": 1,
  "source_scope": [{"path": "package/algorithm.py", "symbol": "Algorithm.run"}],
  "context_symbols": [],
  "contract": "Describe valid inputs, exact observable outputs and mutation, state, and the real invocation lifecycle here.",
  "wall_time_seconds": 1800,
  "final_reserve_seconds": 600,
  "max_candidates": 4,
  "architecture": true,
  "provider": {
    "provider": "codex-chatgpt",
    "model": "gpt-6-astra",
    "effort": "medium",
    "timeout_seconds": 1800
  }
}
```

To remeasure previously authored implementations without an LLM, omit `provider`
and supply `candidates`, a list of schema-version-1 proposal objects with
`hypothesis` and `edits: [{"path": "...", "symbol": "...", "body": "..."}]`.
Source is frozen into the plan, validated and deduplicated exactly like live
proposals. All correctness and timing measurements are fresh; recorded model-call
count and provider time are zero. Prior authoring cost remains a separate cost.

The **30-minute total limit** includes profiling, model responses, candidate
execution and final confirmation. Ten minutes are reserved for final evaluation;
the model response timeout is clipped to the remaining discovery allowance.
`--budget 1h` overrides the total. The model sees its remaining time and candidate
cap. More time permits more work but does not guarantee a faster implementation;
the candidate cap or an author stop can end search early.

For a sustained experiment, increase both limits and reserve final-validation
time explicitly. For example, `wall_time_seconds: 4800`,
`final_reserve_seconds: 1200`, `max_candidates: 128`, and
`honor_author_stop: false` allow up to sixty minutes of discovery profiling,
proposals and measurements, followed by twenty minutes for final validation.
The supported candidate cap is 1–256. The ordinary defaults remain thirty minutes,
four candidates and `honor_author_stop: true`; an empty proposal normally stops
search. With the sustained setting it records an abstention and continues.
Reports record why search stopped. Three consecutive failed provider calls stop
authoring rather than repeatedly consuming the allowance without usable proposals;
an existing discovery winner still goes through the unchanged independent final
gates. The interruption and provider error are recorded even if that patch passes.
Long searches retain the full best eligible and six recent proposals in model
context, with compact older feedback; full artifacts remain on disk.

Intake must already have a valid callable entry and nonempty, separately labeled
discovery and evaluation cases. Lossless profiles discovery inputs automatically,
then shares their measurements, source excerpts and selected function bodies
with Codex. Additional `context_symbols` are read-only. It never sends the
evaluation case descriptions or fixture contents to the author. Ordinary
`profile --explain` remains advice only; it cannot trigger source changes.

Each proposal replaces only executable bodies of the selected functions.
Signatures, decorators, docstrings, module-level imports and other source remain
fixed. An optional declared native-helper scope is described below. Equivalent AST
proposals, including any native source, are deduplicated. The author receives discovery
failures and timings, then can revise. Neither author prose nor its self-reported
performance controls acceptance.

The [CPU investigation catalog](cpu-investigation.md) is enabled by default.
Authors receive hardware facts, discovery layout/library diagnostics, sourced
cards, interactions and correctness obligations. The first valid response must
account for all 22 cards; later reviews can be incremental. Coverage distinguishes
advice from evaluated candidates and lists unfinished work. Set `architecture: false`
in the search plan to use the prior author context. Coverage and proof
template links cannot grant acceptance or prove emitted code.

Before search, Lossless freezes the source, cases, runner hashes, contract,
**original public callable as deployment comparator**, and these default gates:

- Three discovery pairs per case; twelve final pairs per case.
- Exact output and input fingerprints on every measured invocation, including
  previous outputs re-observed after later calls. A changed reference fails.
- At least 1.02× equal-case geometric-mean speedup; no case worse than 1/1.03×.
- No declared call position worse than 1/1.03× (`max_call_regression: 0.03`),
  so an aggregate win cannot hide a slow request inside a sequence.
- Final fixed-sample, one-sided sign-test p ≤ 0.05 for round speedup above 1.02×.
- Median import/factory setup increase ≤ 100 ms per case; separate memory passes
  require RSS and traced allocation peaks ≤ 1.2× reference.

`timing_scope` defaults to `calls`. Set it to `setup_and_calls` for a lifecycle
that pays application import/factory setup on every sequence. Both use complete
calls, per-position regression limits and the same correctness gates. Setup-plus-
sequence times are combined per process before analysis, and payback uses the
chosen boundary. [Measurement policy and public benchmark choices](benchmarking.md).

All gates are configurable in the plan, frozen before execution. Each pair runs
the original and candidate in **separate fresh processes**; AB/BA order is balanced
per case. Each declared call executes once, with completion synchronization and
no hidden warmup. Timers cover the complete callable invocation, including its
internal setup. Fixture loading, fingerprints and observers are outside timers;
their cache effects are a limitation. Import/factory setup and whole worker
elapsed time are reported separately. NumPy import and interpreter startup are
included in worker elapsed time, not import/factory setup.

Only the best discovery-eligible candidate reaches final evaluation. It is frozen
first, and no further model call occurs afterward—even if the final fails. Final
evidence is finite validation, not a universal proof. Sign-test interpretation
assumes independent round signs; correlated drift can violate this. Equal case
weighting does not estimate an unknown production traffic distribution.

`accepted_experiment` produces `selected.patch` and identifies the measured
candidate project. `reference_retained` means no patch passed the full gate.
Failures or exhausted deadlines cannot produce an accepted patch. The report
includes search/model time, raw paired timings, startup/memory and break-even
sequence counts using search time plus extra setup divided by saved call time.
All attempts remain reviewable. Provider dollar cost is unknown under subscription
billing; token usage and elapsed time are recorded when available.

This is **not automatic deployment**. `lossless export` rejects application
experiments: generic domain guards, arbitrary exceptions/aliasing/concurrency,
state-safe fallback, stronger comparator selection and installation validation
remain work. Users still supply representative invocations and a reviewed source
scope; automatic profiling is implemented, fully automatic optimization of an
unknown repository is not. See the [implementation plan](implementation.md).

## Optional compiled helpers

A search plan can declare up to four C helpers with fixed ctypes signatures:

```json
{
  "native_helpers": [{
    "name": "clip",
    "functions": [{
      "name": "clip",
      "returns": "void",
      "arguments": ["pointer", "pointer", "pointer", "size_t"]
    }]
  }]
}
```

This is an addition to the ordinary plan fields. A proposal may then include
`native_sources: [{"name": "clip", "source": "complete C source"}]`, alongside
its function-body edits. The edited body can import the declared function from
`_lossless_native_clip`. Without a declared scope, native source is rejected.
Names cannot replace existing project files. Each helper supports at most sixteen
functions, thirty-two arguments per function and 64 KiB of source. Argument types
are `pointer`, `size_t`, `int`, `int64`, `float`, and `double`; return types also
include `void`.

Lossless compiles these helpers with Clang, strict floating-point flags and a
bounded build timeout inside the search allowance. It records the exact command,
compiler diagnostics, and source, wrapper and binary hashes in `native_builds.json`.
The controller does not load the candidate library. Fresh application workers load
it and run the same independent correctness, complete-call timing, startup and
memory checks as Python proposals. Compilation itself grants no acceptance.

The declared ABI does not validate array pointers or prove the C implementation.
The reviewed callable must guard its supported dtypes, sizes, layouts, alignment
and aliasing, retain buffer owners, and preserve its fallback and output lifetime.
This remains trusted local code execution with user permissions.

For an accepted experiment, `selected.patch` includes the C source and generated
Python wrapper, and the measured candidate project contains the compiled library.
Applying the text patch alone does not build that library. Rebuild from the saved
command for the destination project and rerun its validation before using it.
Application export and automatic deployment remain unsupported.
