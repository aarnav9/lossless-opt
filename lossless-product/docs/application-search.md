# Application source experiments

[Pybaselines study: live authoring, noisy controls, and a scoped 1.058× result](application-search-study.md).
[Broader follow-up: 1.163× beyond the stronger baseline, with a failed allocation gate](application-broader-study.md).
[Continuation with memory feedback and new final cases](application-continuation-study.md).

`lossless search-application` extends application intake and profiling with a
Codex propose → patch → measure → revise loop. It is an experimental source
workflow for trusted Python callables. The original checkout stays unchanged;
proposals execute in fresh copies with local user permissions. These processes
are **not a security sandbox**.

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
fixed. Equivalent AST proposals are deduplicated. The author receives discovery
failures and timings, then can revise. Neither author prose nor its self-reported
performance controls acceptance.

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
