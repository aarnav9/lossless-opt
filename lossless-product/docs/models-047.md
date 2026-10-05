# Campaign 047: model choice under a common search allowance

**No model established an incremental win over the shared seed or enumeration
under the full predefined gate.** Astra had the highest measured average; Luna
produced similar performance to Sol with about half the search/evaluation time.
Keep the model-generated kernels as research evidence. This pilot does not
justify a new default kernel or a general model ranking.

The actual study took **75 minutes 53 seconds**, including the enumeration
control, separate evaluation and paired diagnostics. All author calls used the
existing ChatGPT-signed-in Codex CLI, not an API key.

## Held-out result

Ratios below are fresh paired geometric means over twelve final shape/layout
cases, using prebound calls on Gaussian input. Each model's choice was frozen
before these cases opened.

| Model | Completed responses / attempts | Valid new kernels | Search + evaluation | Versus common seed | Versus enumeration | Versus retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-6.1 Sol | 2 / 3 | 2 | 29.30 min | 1.00790× | 1.00253× | 1.19621× |
| GPT-6 Astra | 5 / 6 | 5 | 29.47 min | 1.02426× | 1.01963× | 1.21434× |
| GPT-6 Luna | 6 / 6 | 5 | 15.40 min | 1.00914× | 1.00479× | 1.19842× |

All selected kernels passed the tested numerical checks. All three passed the
incremental gate against retained recipes, but **none passed against the common
seed or enumeration**. The gate requires at least 1.02× geometric mean and every
case's paired lower 95% interval above 1×. Astra exceeded the mean threshold
against the seed, but three cases failed the interval condition. Against
enumeration, Astra's 1.01963× mean also fell just below the 1.02× threshold.

The approximately 20% retained-baseline gain is not the incremental value of
these new model calls: the starting pool already contained a strong earlier
kernel. Relative to that available seed, fresh authoring added only 0.8–2.4% in
measured mean, with case-specific losses. Relative to enumeration it added
0.25–1.96%, failing the full gate.

Direct model comparisons also failed the incremental gate:

| Fresh paired comparison | Geometric mean |
| --- | ---: |
| Sol / Astra | 0.98530× |
| Sol / Luna | 0.99677× |
| Astra / Luna | 1.01451× |

Astra's mean was about 1.5% faster than the others in these direct measurements,
but it did not win consistently across all cases. Timing intervals within one
trajectory do not establish repeatability across independent authors.

The enumerated control took **29.02 s discovery + 32.05 s evaluation = 61.07 s**,
selected `enum_u8_two1_scalar0`, and passed all twelve intervals against the
frozen NumPy comparator at 1.91787× geometric mean. The model choices were
1.92579× / 1.94773× / 1.93609× NumPy for Sol / Astra / Luna in their separate
normal evaluation. Those evaluations are distinct from the fresh paired ratios
above; do not combine ratios across measurements.

## Search trajectories and cost

- **Sol:** proposal 1 became the incumbent after a 7.26-minute response. Proposal
  2 took 15.96 minutes and did not improve it; proposal 3 timed out after its
  remaining 5.50-minute response allowance. Search took 29.00 minutes.
- **Astra:** proposal 2 became the incumbent; proposals 3–5 were valid but did
  not improve it. The sixth response timed out after its remaining 2.66-minute
  allowance. Search took 29.00 minutes.
- **Luna:** completed six responses in 14.93 minutes of search, reaching the
  call cap before the time limit. Proposal 3 failed compilation; compiler
  feedback was supplied to proposal 4, which compiled and validated. Proposal 5
  became the incumbent; proposal 6 did not improve it.

Thus the study matches **maximum time and call allowances**, not actual time
spent. It does not answer what Luna would do with a larger call cap and the full
30 minutes. No source was manually repaired, and failures/duplicates/timeouts
consumed slots. Final evaluation and paired diagnostics occurred after all
selections were sealed.

The improvements mainly changed row packing, strided loads and unnecessary
scanning. The selected sources retained scalar `expf` and float64 accumulation;
they did not introduce a new approximate exponential or lower precision.

| Model | Recorded input tokens | Recorded output tokens | Calls missing usage |
| --- | ---: | ---: | ---: |
| Sol | 56,720 | 13,742 | 1 |
| Astra | 139,410 | 51,038 | 1 |
| Luna | 161,586 | 40,325 | 0 |

These are completed-turn CLI receipts, including the compilation-failing Luna
proposal. Timeout usage is **unknown, not zero**, so Sol/Astra totals are lower
bounds. Dollar cost is unknown; no API-price conversion or pro-rated subscription
cost is invented. Probes and the separate live-copy connector smoke are outside
this authoring table and are included separately in the evidence bundle.

## Payback

Search plus normal final evaluation was 1,757.73 s for Sol, 1,768.45 s for Astra
and 924.03 s for Luna. The per-case calculation is:

\[
N_{\mathrm{break-even}} =
\frac{T_{\mathrm{search+evaluation}} + T_{\mathrm{extra\ setup}}}
{T_{\mathrm{reference\ call}} - T_{\mathrm{selected\ call}}}.
\]

Deployment setup was not measured for these research candidates. Against an
already available enumeration winner, the **search-only lower bounds** among
cases with positive measured savings were:

| Model | Finite search-only payback range | Cases with no positive saving |
| --- | ---: | ---: |
| Sol | 816 million–103.8 billion calls | 5 / 12 |
| Astra | 170 million–14.2 billion calls | 1 / 12 |
| Luna | 144 million–19.2 billion calls | 5 / 12 |

These are point-estimate case calculations, not confidence guarantees or a
workload-weighted business case. A losing case has no finite payback under those
measurements. They charge the model search against an already available
reference, without subtracting the control's 61.07-second search/evaluation cost.
Paired diagnostic measurement time is reported in total study elapsed time but
is outside the per-model search/evaluation amounts above. Source receipts and
all per-case payback records remain in the archive.

## What this pilot can establish

One trajectory each used `gpt-6.1-sol`, `gpt-6-astra` and `gpt-6-luna`, `xhigh`,
30-minute maximum search budgets and six one-proposal call slots. Model order
was frozen as Sol → Luna → Astra and run sequentially on Apple M2. The common
starting pool was two retained recipes plus the campaign-044 kernel, originally
authored by Astra. This tests further optimization from that shared start,
not blank-slate authorship or superiority to every later 045/046 kernel.

Nine discovery cases and twelve new final cases span C, Fortran and sliced
layouts. Timing explicitly uses Gaussian input; correctness uses six
finite-input distributions and the independent float64 oracle. This is not the
broader eleven-distribution, ordinary-call qualification used for the
[opt-in recipe in campaign 046](softmax-046.md).

Enumeration varied unrolling, two-value detection and scalar/vector
normalization of the shared seed. It was a fixed six-slot control, not an
exhaustive search over all memory-layout transformations. Equal `xhigh` labels
do not mean equal inference compute. One device, one run order and one trajectory
per model leave provider variability, timing drift and workload transfer open.
No across-author statistical superiority or automatic promotion follows.

The next informative model study would repeat Astra and Luna from the current
strongest retained pool, with enough call slots for both to use a common time
allowance. This pilot gives little reason to lengthen a single run before doing
that replication.

## Evidence and reproduction

[Protocol and commands](../experiments/models_047/README.md) ·
[Codex subscription setup](codex.md) ·
[Compressed evidence](assets/models-047.json.gz).

Evidence SHA-256:
`dcce3fd608d06480bae3cef5a2bb24b836ccd7525e54865da6857d1baa2d2351`.
The archive preserves exact frozen harness and kernel sources, raw timing and
validation JSON, final responses, receipts, sealed choices, model probes and
the live connector smoke. Raw CLI event streams, stderr, auth/runtime databases
and compiled binaries are excluded; local paths are redacted. Later transport
fixes do not rewrite the frozen source snapshot. Freeze a new study when using
newer source.
