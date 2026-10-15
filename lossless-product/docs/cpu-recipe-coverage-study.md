# Expanded CPU recipes — 15 October 2026

Five paired workload trials completed: the published `0.1.0a3` runtime followed
by expanded recipes, at equal 150-second workflow budgets. The expanded search
found candidates for four workloads previously lacking a recipe, but **none of
those four produced a qualified application speedup**. NumPy Financial retained
its existing approximately 1.35× complete warm-request gain.

The expanded recipes are therefore available in current source (`0.1.0a4.dev0`)
through **`--recipe-coverage expanded`**, with `stable` remaining the default.
The published a3 wheel is unchanged and does not expose this option.

[Portable evidence](evidence/cpu-recipe-coverage.json.gz) includes the frozen
protocol, source used during the campaign, full reports, raw paired measurements,
patches, upstream licenses and source/runtime identities.
[Reproduction](../experiments/cpu_recipe_coverage/README.md) ·
[Harness](../experiments/cpu_recipe_coverage/study.py).

## What was added

The optional candidate generator supports NumPy reductions with axis and keyword
arguments; `argmin`/`argmax`, `cumsum`/`cumprod`, `std`/`var`, and `amin`/`amax`;
and profiled synchronous class methods identified by source line. Existing
candidates remain available individually and in compatible combinations.

The new dispatch captures the callable and argument expressions once, in their
original evaluation order. It forwards axis, dtype, output buffer and other
supported arguments to the corresponding ndarray method. Only exact plain
numeric ndarrays take this path; lists, subclasses and object arrays retain the
original function. Subclass or duck-array `out`/`where` operands also retain
NumPy dispatch. Unknown keyword extensions and argument unpacking are left
untouched. There is no algebra reassociation, changed precision or numerical
tolerance. This is measured finite qualification, not a proof for all inputs.

## Outcomes

Ratios below compare the expanded candidate with equally prepared original
source during deployment evaluation. Rejected ratios are observations, not
delivered speedups. Startup is amortized over 100 expected requests.

| Workload | Candidates: a3 → expanded | Warm / amortized candidate ratio | Delivered result |
| --- | --- | --- | --- |
| FilterPy outer-product sums | 0 → 1 | 1.0181× / 0.9981× | Source-only search passed; complete-request gates rejected it. Original, fresh workers. |
| FilterPy systematic resampling | 0 → 1 | No candidate-vs-original deployment winner | Source search retained original; deployment's original-source choice failed the incumbent guard. Original, fresh workers. |
| FilterPy ensemble Kalman filtering | 0 → 3 | 1.0129× / 0.9980× | Class-method candidates generated and executed; complete-request gates rejected the frozen winner. Original, fresh workers. |
| Perlin fractal noise | 0 → 1 | No candidate-vs-original deployment winner | Source candidate failed memory admission on the large case; final deployment also retained fresh original execution. |
| NumPy Financial rates | 3 → 3 | **1.3467× / 1.3235×** | Existing combined source candidate and prepared worker qualified. |

The a3 Financial control measured 1.3624× warm and 1.3383× amortized against its
equally prepared original. Both versions selected the same source recipe. These
separate runs are consistent with retaining the existing gain; they do not show
that the expanded runtime is faster than a3. No gains are multiplied together.

All ten delivered adapters matched explicit reference results on fresh finite
values with the same metadata. No model calls or relaxed numerical acceptance
were used. Each workload needed a wrapper and Python input loader, with no
upstream algorithm edits. Total workflow times were 13.6–60.5 seconds; paired
trials ran sequentially on the same host. This is agent-operated integration
evidence, not an independent developer study or whole-repository benchmark.

## Why the default remains stable

The three a3 FilterPy runs qualified prepared original execution. Their expanded
runs fell back to fresh workers after final qualification failed. The existing
failure path is conservative about correctness, but it can lose the worker-reuse
benefit from a separate baseline run. Keeping those expanded recipes on by default
would add search time and could make deployment slower despite correct outputs.

After the frozen campaign, an explicit coverage option was added. Candidate
parity checks on all five saved discovery profiles verified that stable mode
generates the a3 candidate sets, and expanded mode generates the exact candidate
sets measured here. Candidate transformations and acceptance thresholds were
not retuned using evaluation results. The evidence retains the pre-option source;
the current reproduction harness explicitly opts into expanded mode when supported.

A next research step is to predeclare and independently qualify a fallback
lifecycle while retaining the existing performance and memory gates. Selecting
another candidate after inspecting failed evaluation would need a new protocol,
not a hidden retry. This remains pending; the current option makes that limitation
visible without changing the default search on these workloads.

## Validation and remaining usability work

The local full suite passed 260 tests with eight optional skips before the final
coverage option. Afterwards, all 11 recipe tests and the CLI coverage-routing
test passed. A final review then added a regression test and fallback guards for
`out`/`where` subclass overrides; all 12 recipe tests passed. None of the measured
workload candidates use those operands, and their generated bodies remain unchanged.
Tests cover dtype/layout/axis combinations, nonfinites, output-buffer
identity and mutation, subclass/object fallback, evaluation order, unknown
extensions, ambiguous class names, decorated scopes and default opt-in behavior.
CI runs the final suite and installed-wheel checks on Linux and macOS.

Two final default-mode workflow controls then kept prepared original execution
for FilterPy outer products and accepted Financial's existing combined candidate.
Fresh values matched through both public adapters. Financial measured 1.4357×
warm / 1.4020× amortized in that validation run. It reused the same workload and
is not an independent performance replication or an additional recipe gain.
[Final default checks, parity results and raw measurements](evidence/cpu-recipe-validation.json.gz).

The [independent developer trial](developer-onboarding-trial.md) has a participant
task, success criteria and feedback template. No independent participants have
completed it yet, and human setup time remains unmeasured.
