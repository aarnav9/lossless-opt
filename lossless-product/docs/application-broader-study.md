# Broader CPU application search

This follow-up tests whether a sustained authoring loop and a wider source scope
improve on the [earlier 1.058× ModPoly experiment](application-search-study.md).
**Result: 1.16255× the strongest existing baseline and 1.70614× stock public
calls on new final workloads, with exact outputs.** Speed and startup gates
passed; the extra cache allocations failed the frozen 20% traced-peak gate, so
the official outcome is **reference retained**. The source is an experimental
candidate, not a promoted deployment recipe.

Authoring was interrupted by the Codex subscription usage limit after 29m16s.
Eleven candidates completed development checks, and the best eligible source was
frozen before any final workload ran. This is not a completed sixty-minute study
or evidence from a matched 30-versus-60-minute comparison.

The [subsequent continuation](application-continuation-study.md) uses the remaining
authoring allowance after subscription access recovered. This study's final cases
become development data there, with a new final suite and unchanged resource gates.

## Workload and comparison

The study-owned spectrum processor uses pinned BSD-3-Clause pybaselines 1.2.1
(`6f8d927e08577a9c5af5f34129b14d57c12725f6`). It prepares input arrays, fits each
spectrum, subtracts its baseline and constructs returned records. Those steps,
cache lookup/build and every first call are inside the application timer. Fixture
decoding, correctness fingerprints and observer hooks are outside; import/factory
setup and complete worker wall time are reported separately. This is a synthetic
application around a real library, not a speed claim for arbitrary repositories.

This is **CPU-only on Apple M2**, with `OPENBLAS_NUM_THREADS=1` on both sides.
There is no GPU, precision reduction or relaxed numerical contract.

Before authoring, three fresh development pairs compare four implementations:

- Unchanged public `polynomial.modpoly` calls.
- Public calls with the previous frozen norm-specialization winner.
- Unchanged public reusable `Baseline` objects, with at most four cached
  grid/order objects guarded by input contents.
- Reusable objects with the previous winner.

The fastest global case-balanced development implementation becomes the frozen
deployment comparator. Final measurements compare the one frozen new candidate
against **all four**, so recovering an existing library facility cannot alone
qualify as a new improvement. Baseline selection is itself measured and recorded.

All fourteen previous cases are now development data, wrapped as two requests
of sixteen fits. Three additional development cases exercise repeated weights,
changing polynomial orders and irregular grids. Eleven new final profiles vary
sizes, seeds, distributions, float32 input, weights, masking, ordering and batch
size, including a single-spectrum request. Each final profile contains three
requests; request order reverses on the second request. Earlier outputs remain
live and are re-observed after the complete sequence.

## Search and acceptance

Subscription-backed GPT-6 Astra at medium reasoning may change eight bodies:
processor construction/preparation/fitting/request execution, ModPoly, polynomial
setup, Vandermonde recalculation and pseudoinverse access. It can change batching,
buffers and instance-local cache behavior. The public interfaces, other source,
comparison inputs, observer, numerical parameters and evaluator remain fixed.
There are no human repairs to model candidate bodies.

The search allowance is **3,600 seconds**, including discovery profiling, model
calls and candidate measurements, with a 128-proposal cap. An empty proposal
records an abstention rather than ending this sustained run. Model responses
have a 600-second individual ceiling clipped to remaining discovery time. Twenty
additional minutes are reserved for the first final comparison, and other
comparator checks have a separate thirty-minute ceiling. Preparatory baseline
measurements are outside the hour and are included in total study cost.

The model receives only development inputs, profiling and previous development
feedback, with remaining time/calls explicit. The best eligible implementation is
frozen once before final cases open. No model call, source revision or reselection
follows final feedback, including a failure.

The observed run made fourteen provider attempts: eleven completed responses and
three quota failures. Provider time was 576.28 seconds, including the failures;
completed responses reported 1,057,419 input tokens, 20,691 output tokens and
3,075 reasoning-output tokens. Dollar cost and starting subscription allowance
are unknown. The CLI reported a usage-limit error, not a model decision to stop
or exhaustion of the configured time allowance.

The initial controller aborted after three provider failures, leaving final
cases unopened. A recovery record then froze `candidate_11`, using exactly the
original maximum-development-score rule among eligible candidates. It was not
edited or reselected. The separate `qualify_broader.py` path evaluates that fixed
source against the original four baselines, with zero further model calls and
unchanged cases and gates. The original failed report remains intact.

The controller now handles this failure more usefully: stop requesting proposals,
record the provider error and incomplete authoring, and independently validate an
already measured winner when one exists. The interruption never grants acceptance
or changes the final gates. This recovery/error-reporting change was made after
the measured authoring run; the application worker and comparison implementation
are unchanged.

Each final comparison uses twelve balanced-order pairs per profile in independent
fresh processes. Acceptance requires:

- Exact returned baseline/corrected arrays, weights, coefficients and full
  tolerance histories, unchanged input observations and retained earlier outputs.
- At least 1.02× equal-case geometric-mean speed, no case below 1/1.03×, and a
  fixed twelve-round one-sided sign-test p-value at most 0.05.
- Median import/factory setup increase at most 100 ms; separate process RSS and
  traced allocation peaks at most 1.2× the comparator.
- Bounded processor state: at most four retained grid/order fitters and 16 MiB
  of counted persistent array/string/byte storage, checked by an uneditable
  observer. This finite object traversal is not a general Python memory proof.
- Unchanged source, fixtures, runner, comparator and frozen-candidate identities.

Every baseline must pass the final gates for the overall study to accept a new
implementation. The final statistical assumptions and finite-case limitations
are the same as [application source search](application-search.md). This produces
experimental evidence and a reviewable patch, not a guarded deployment artifact.

## Observed results

| Frozen comparison | Final paired speedup | Exact outputs | Speed/startup gates | Allocation gate |
| --- | ---: | --- | --- | --- |
| Stock public calls | 1.70614× | Pass | Pass | Fail |
| Public calls + previous patch | 1.60105× | Pass | Pass | Fail |
| Stock reusable objects | 1.27721× | Pass | Pass | Fail |
| Reusable objects + previous patch (deployment comparator) | **1.16255×** | Pass | Pass | Fail |

The strongest-baseline result means **13.98% less request execution time**; the
stock-public result means **41.39% less time**. These are directly measured ratios
from separate comparisons, not products of earlier campaign gains. The eleven
new final profiles used **1,056 fresh timing processes**, plus 88 separate memory
processes. Each timing process executed three complete requests; those requests
contained 49,824 total spectrum fits across the four comparisons. All observed
outputs, input observations and retained earlier outputs matched exactly.
All twelve case-balanced rounds exceeded 1.02× in every comparison (one-sided
fixed sign-test p = 0.000244). Case weighting describes this suite, not a production
traffic distribution.

| New final workload | Strong baseline (ms) | Candidate (ms) | Paired speedup |
| --- | ---: | ---: | ---: |
| small | 10.226 | 7.596 | 1.34470× |
| large | 34.117 | 31.029 | 1.09589× |
| f32 | 8.674 | 7.314 | 1.19320× |
| shared weights | 24.786 | 19.525 | 1.27535× |
| changed weights | 11.321 | 10.011 | 1.12733× |
| mask | 13.677 | 13.438 | 1.01124× |
| descending | 10.640 | 9.301 | 1.13882× |
| orders | 15.401 | 12.375 | 1.24013× |
| irregular | 11.278 | 10.321 | 1.08947× |
| flat zero | 0.988 | 0.901 | 1.12946× |
| single | 0.747 | 0.634 | 1.18003× |

Times are medians of three-request sequences, including their first request and
cache creation. Paired ratios need not equal ratios of the displayed medians.
The single-spectrum case is three requests of one fit each; other cases contain
12–24 fits per request. This demonstrates an in-memory request-path gain. Import
setup is separate; worker wall time also includes evaluator/fixture overhead and
is not evidence of a faster cold CLI application.

The selected source combines bounded weighted/masked pseudoinverse caches,
coefficient-transform reuse, a fast path for an unchanged validated polynomial
order, scalar convergence-dispatch simplification and local alternating baseline
buffers. It does not change model weights, precision, tolerance, iteration order
or returned observations. The final proposal modifies the polynomial method and
two shared setup helpers; it leaves the study-owned processor bodies unchanged.

**The resource tradeoff is real.** Against the strongest comparator, the changed-
weight, masked and descending-weight cases exceeded the allocation gate. The
largest traced-peak increase was **319.6 KiB / 54.02%**. Peak process RSS increased
by at most **0.887%**, and the largest median import/factory setup increase was
**1.913 ms**. Other cases had small or negative traced-peak changes. Across all
four comparators, the maxima were 331.7 KiB extra traced allocation, 57.23% traced
peak increase, 1.61% RSS increase and 6.37 ms extra setup. The persistent-state
observer's existing 16 MiB bound passed. That does not override the separately
frozen 20% traced-allocation limit: the candidate is **not accepted** under this
protocol. A more permissive explicit cache budget would be a different protocol,
requiring its own fresh qualification before promotion.

Preparation plus interrupted authoring took 2,056.37 seconds; frozen qualification
took 1,139.43 seconds, for **3,195.80 seconds of measured controller time**. This
excludes development and the recovery interval. Using median saved request time,
that cost takes approximately 607,000–36.9 million three-request sequences to
repay against the strongest comparator, depending on the case. Against stock
public calls the range is approximately 216,000–7.13 million sequences. These
small request costs favor reusing a qualified recipe over searching afresh for
every occasional application call.

The first 5 development candidates failed at least one speed gate despite exact
outputs. Candidate 6 first passed those gates at 1.14630×; later eligible scores
rose to 1.20084×, 1.26346×, 1.27545× and 1.27646×. Candidate 9's higher aggregate
score did not qualify because one case regressed. This is an observed search
trajectory, not a causal estimate of how much extra time helps other runs.
The final 1.16255× result is lower than development's 1.27646×: new weights, grids,
distributions and request sizes matter.

Keep the broader search controls, interruption recovery, frozen source and raw
measurements. The next release decision should address an explicit cache-memory
budget or a lower-allocation variant, with new final cases. Before another long
live-author study, reduce duplicated model context and discovery process-startup
cost: this run consumed over a million input tokens and ended on subscription
quota before its clock limit.

The [compressed evidence](evidence/application-broader-pybaselines.json.gz)
contains raw paired samples, model proposals/receipts, quota errors, baseline
selection, both reports, recovery provenance and the development frontier.
[The frozen candidate](../experiments/application_intake/broader-candidate.json)
and its [BSD license](../experiments/application_intake/PYBASELINES_LICENSE.txt)
are included. The original upstream checkout, frozen comparators and selected
source remained unchanged. No candidate was edited or reselected after final
cases opened.

Validation: **132 product tests ran: 131 passed and one skipped**. A separate
copy of the selected source passed **308 upstream polynomial/helper tests**,
with eight skipped and 413 unrelated tests deselected (`-k polynomial`). These
upstream tests add compatibility coverage; their tolerances are distinct from
Lossless's exact final comparisons.

## Reproduction

Use the [earlier environment setup](application-search-study.md#reproduce-without-a-model-subscription),
then sign into Codex with ChatGPT for the live author. No API key is used. On
Apple Silicon, allow roughly 75–100 minutes for a one-hour search plus baseline
and final measurements; times depend on the environment and failures.

```sh
set -o pipefail
PYTHONUNBUFFERED=1 PYTHONPATH=lossless-product/src \
  caffeinate -i .venv-pybaselines/bin/python -u \
  lossless-product/experiments/application_intake/broader_study.py \
  --project .lossless/external/pybaselines-v1.2.1 \
  --output .lossless/external/pybaselines-broader-repro \
  --search-seconds 3600 --final-reserve 1200 --max-candidates 128 \
  2>&1 | tee .lossless/external/pybaselines-broader-repro.log
```

`caffeinate -i` prevents idle sleep on macOS; omit it on other platforms. It does
not override closing a laptop lid. Deadlines use a sleep-inclusive clock. Choose
a fresh output path for each run. The command prints timestamped progress and
result/log paths. `--prepare-only` creates frozen inputs/configuration without
calling a model or making a timing claim. New model runs need not return the same
implementation.

To remeasure the selected source **without a model subscription or any LLM calls**,
use the same environment and pinned upstream checkout:

```sh
set -o pipefail
PYTHONUNBUFFERED=1 PYTHONPATH=lossless-product/src \
  caffeinate -i .venv-pybaselines/bin/python -u \
  lossless-product/experiments/application_intake/qualify_broader.py \
  --project .lossless/external/pybaselines-v1.2.1 \
  --candidate lossless-product/experiments/application_intake/broader-candidate.json \
  --comparator reuse_previous \
  --output .lossless/external/pybaselines-broader-qualification-repro \
  --budget 1800 \
  2>&1 | tee .lossless/external/pybaselines-broader-qualification-repro.log
```

Allow approximately 15–25 minutes. The script reconstructs all four baselines,
applies exactly one fixed proposal and rechecks the published final cases. It
does not choose another implementation or relax failed gates. These are known
cases on reproduction, not a new unseen-workload experiment.
