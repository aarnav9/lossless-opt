# Broader CPU search continuation

This follows the [interrupted broader study](application-broader-study.md).
The earlier winner was faster but failed its frozen allocation limit. After the
CLI subscription became available again, GPT-6 Astra at medium reasoning received
the remaining **1,844.48 seconds** of search time, the same implementation scope
and acceptance thresholds, and memory feedback after every exact proposal.

The first session's 1,755.52 seconds plus this allowance total one hour. These are
two sessions separated by final measurements and feedback, **not an uninterrupted
one-hour run or a controlled 30-versus-60-minute comparison**. All candidate bodies
were authored through the subscription CLI. No manual-author fallback was needed.

## Protocol

All 28 previously seen profiles, including the earlier final profiles, became
development cases. Eleven new profiles, with changed sizes, seeds, distributions
and request batches, were frozen before the continuation. Their descriptions and
fixtures were excluded from author context. They opened only after the winner
was frozen, with no subsequent model calls, source edits or reselection.

The deployment comparator remained the prior study's strongest implementation:
public reusable `Baseline` objects plus the earlier convergence patch. The frozen
source is compared against stock public calls, public calls with that patch,
stock reusable objects, and reusable objects with the patch. A fifth comparison
against the interrupted run's cache-heavy candidate is diagnostic: that source
never qualified for deployment under the memory contract.

The CPU, single-OpenBLAS-thread environment, exact observations and complete
request timer are unchanged. Inputs are decoded outside the request timer;
preparation, cache lookup/build, fitting, correction and result construction are
inside. Import/factory setup and worker wall time remain separate. Returned
arrays, parameters, convergence histories, input observations and retained earlier
outputs are all compared exactly.

Three fresh balanced-order pairs per development profile are followed by separate
memory processes. Selection now requires development speed, memory and setup gates
as well as exactness. Final comparisons use twelve pairs per profile and the same
frozen thresholds: aggregate speed at least 1.02×; no case below 1/1.03×; fixed
one-sided sign-test p ≤ 0.05; setup increase ≤ 100 ms; process RSS and traced
allocation peaks ≤ 1.2×. All four deployment comparators must pass. Persistent
state remains limited to four fitters and 16 MiB of counted storage.

## Development trajectory

The retained source was first remeasured on the expanded development suite. Five
new proposals completed; a sixth model response reached the remaining deadline
without a complete eligible response. The 128-candidate cap did not stop search.

| Candidate | Development speedup | Memory gate | Overall development gates |
| --- | ---: | --- | --- |
| Retained cache-heavy source | 1.23262× | Fail | Fail |
| 1: remove large pseudoinverse caches | 1.17351× | Pass | Pass |
| **2: reuse fit-local scratch arrays** | **1.18944×** | **Pass** | **Pass; frozen** |
| 3: move branches and reuse coefficient output | 1.15141× | Pass | Fail: flat case |
| 4: move branches without coefficient-output reuse | 1.17791× | Pass | Fail: masked case |
| 5: one shared weighted-pseudoinverse cache | 1.21924× | Pass | Fail: changed weights |

Candidate 3's flat-case failure included one unusually slow pair; all recorded
samples were retained. Candidate 5's higher aggregate score did not override its
per-case regression. This is adaptive development evidence, not independent
proof of a benefit from more authoring time.

The frozen source changes two function bodies: `_Polynomial.modpoly` and
`_PolyHelper.recalc_vandermonde`. It retains the validated polynomial-order fast
path and a small content-guarded coefficient-transform cache. It removes the
large persistent weighted/masked pseudoinverse caches. Fit-local clipping and
elementwise scratch arrays are reused, with separate baseline buffers and a
fallback for unsupported layouts. Matrix products, dot products, iteration order,
thresholds and output ownership remain subject to the exact evaluator.

Frozen proposal SHA-256:
`bce8c55fc8d8428aa4e8d6635892316b7e14e21c2170cb6c289f8b49c0b69c4a`.

## Final result

The candidate passes every final gate against both reusable-object baselines,
including **1.15188× the strongest comparator (13.19% less request time)**. Its
traced allocation peak is equal or lower on every profile against that comparator.
However, the study's additional allocation gates against public calls fail, so
the overall outcome remains **reference retained**, not a promoted recipe.

| Comparison | Final paired speedup | Exact/speed/setup | Allocation gate |
| --- | ---: | --- | --- |
| Stock public calls | 1.75698× | Pass | Fail |
| Public calls + previous patch | 1.58027× | Pass | Fail |
| Stock reusable objects | 1.28152× | Pass | Pass |
| **Reusable objects + previous patch** | **1.15188×** | **Pass** | **Pass** |

The diagnostic comparison with the earlier cache-heavy winner is **1.00445×**,
with sign-test p = 0.927 against the 1.02× threshold: no demonstrated aggregate
speed improvement. Its shared-weight profile is 0.84339×, reflecting the lost
weighted-pseudoinverse cache benefit. The new source instead reduces traced peak
allocation by 37.99% on changing weights, 31.60% on descending weights and 19.41%
on masked spectra relative to that source. Extra time produced a more conservative
memory/speed tradeoff, not a uniformly faster implementation.

The stock-public comparison means 43.08% less request time; the stock-reusable
comparison means 21.97% less. These ratios are directly measured on the new suite,
not multiplied from prior studies. All twelve case-balanced rounds exceed 1.02×
in each of these four comparisons (fixed one-sided sign-test p = 0.000244).
All eleven profiles are faster against the strongest comparator. Across all five
comparisons there are **1,320 fresh timing processes**, 110 separate memory
processes and 55,800 timed spectrum fits. All observed outputs match. Thermal/system
drift and equal-case weighting retain the limitations of the original protocol.

| New profile | Strong baseline (ms) | Candidate (ms) | Paired speedup |
| --- | ---: | ---: | ---: |
| small | 5.765 | 4.361 | 1.17383× |
| large | 17.067 | 15.321 | 1.10327× |
| f32 | 11.657 | 9.685 | 1.17577× |
| shared weights | 18.757 | 16.141 | 1.15330× |
| changing weights | 9.701 | 8.747 | 1.11031× |
| mask | 12.494 | 11.754 | 1.08949× |
| descending | 10.535 | 9.135 | 1.17695× |
| orders | 11.432 | 9.036 | 1.31627× |
| irregular | 14.602 | 12.831 | 1.13569× |
| flat zero | 1.231 | 1.114 | 1.08620× |
| single | 0.572 | 0.478 | 1.16678× |

Times are medians of complete three-request sequences, including first requests.
Paired geometric-mean ratios need not equal the ratios of displayed medians.
The single case contains one fit per request; other cases contain 10–20 fits.
This remains an in-memory synthetic application around a real BSD-licensed
library, not a cold-command or arbitrary-repository speed guarantee.

Against the strongest comparator, the maximum median setup increase is 23.65 ms
and maximum process RSS increase is 3.31%; both pass. Traced peak allocation is
lower on ten profiles and equal on the remaining one. Against stock reusable
objects, the largest traced increase is 54 bytes (0.0064%).

The two public-call allocation failures are principally the existing reuse
lifecycle's retained fitters, as the separately measured reference peaks show:

| Profile | Stock public peak (KiB) | Stock reuse peak (KiB) | Candidate peak vs reuse (KiB) |
| --- | ---: | ---: | ---: |
| orders | 752.10 | 976.51 | 976.27 |
| irregular | 533.26 | 829.02 | 829.07 |

Against stock public calls the candidate's maxima are +295.76 KiB / 55.46% traced
allocation, +2.71% process RSS and +20.52 ms median setup. Public calls with the
previous patch also fail allocation on these two profiles, while their speed,
setup and RSS checks pass. These findings do not retroactively change the study's
requirement that every deployment comparison pass. Any deployment qualification
using a lifecycle-specific or absolute memory budget must freeze that policy
before a new final evaluation.

The completed continuation responses report 178,256 input tokens (24,576 cached),
9,563 output tokens and 2,713 reasoning-output tokens. Provider calls took 317.74
seconds including the final timeout. That timed-out response may have consumed
additional unreported tokens. Subscription dollars and starting allowance are
unknown. The standalone availability probe is outside the search allowance and
these totals. Final validation uses no model calls.

The continuation took **3,512.88 seconds** of controller time: 1,844.53 seconds
of search plus preparation and final validation. Charging that entire cost to
the median request-time savings gives roughly **1.34–37.6 million three-request
sequences** to break even against the strongest comparator, or 339,000–9.07 million
against stock public calls. Including the parent study's measured controller
time gives 6,708.68 seconds total and roughly 2.56–71.8 million sequences against
the strongest comparator. These are case-dependent estimates, not a production
traffic model; they exclude implementation work, validation tests and waiting
between sessions. Reusing a qualified recipe matters much more economically than
repeating a live search for occasional millisecond-scale calls.

The original checkout, all five comparison sources, fixtures, controller and
measurement modules passed their recorded provenance checks. The frozen candidate
was unchanged after selection. Keep the experimental source, reproducible evidence
and memory-feedback loop; do not advertise this as unconditional deployment
acceptance or a guaranteed benefit from doubling the authoring budget.

The [compressed evidence](evidence/application-continuation-pybaselines.json.gz)
contains the protocol, author context, proposals and receipts, raw development and
final pairs, memory results, frozen-source identities and the executed controller.
The [frozen proposal](../experiments/application_intake/continuation-candidate.json)
retains the [upstream BSD license](../experiments/application_intake/PYBASELINES_LICENSE.txt).

Validation: the selected source passed **308 upstream polynomial/helper tests**,
with eight skipped and 413 deselected. These tests use upstream tolerances and
complement the separate exact comparisons. The unchanged product source passed
**131 product tests, with one skipped**, earlier in this task. Python lint and
format checks passed for all 141 files.

## Reproduction without a model

Use the pinned environment and upstream checkout from the
[original setup instructions](application-search-study.md#reproduce-without-a-model-subscription).
Then remeasure the frozen source with no model calls or subscription:

```sh
set -o pipefail
PYTHONUNBUFFERED=1 PYTHONPATH=lossless-product/src \
  caffeinate -i .venv-pybaselines/bin/python -u \
  lossless-product/experiments/application_intake/qualify_continuation.py \
  --project .lossless/external/pybaselines-v1.2.1 \
  --candidate lossless-product/experiments/application_intake/continuation-candidate.json \
  --output .lossless/external/pybaselines-continuation-repro \
  --budget 2400 \
  2>&1 | tee .lossless/external/pybaselines-continuation-repro.log
```

Allow roughly 20–30 minutes on this environment and use a new output directory.
Omit `caffeinate -i` off macOS. Progress is timestamped and flushed; result/log
paths are printed. `--prepare-only` builds the fixtures and frozen source without
timing or model calls. Its cases and five baseline trees were checked against the
measured run and matched exactly. Published final cases are known on reproduction,
so this command is a remeasurement, not a new unseen-workload experiment.

The live `continue_broader.py` controller additionally requires the retained raw
parent study directory through `--previous-study`; it reuses that run's measured
profile context and records its identity. It is a study-specific continuation
controller, not a general checkpoint/resume feature in the product.
