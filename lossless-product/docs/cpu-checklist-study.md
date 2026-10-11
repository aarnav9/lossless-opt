# CPU checklist author comparison

This pilot compares the previous application-author context with the new
[CPU investigation catalog](cpu-investigation.md), hardware facts, discovery
diagnostics and structured reviews. Both arms begin with the same already
optimized pybaselines application. The study measures the joint treatment;
it cannot isolate which added component helped.

## Frozen protocol

- Pinned BSD-3-Clause pybaselines 1.2.1, reusable `Baseline` objects and the
  [continuation candidate](application-continuation-study.md) as the common
  experimental comparator. Neither arm gets the other arm's proposals or results.
- GPT-6 Astra, medium reasoning, existing ChatGPT CLI sign-in. No manual author
  substitution. Authors see discovery cases only and the same eight editable
  Python function bodies, contract and acceptance policy.
- Thirty minutes total per arm: twenty minutes for discovery profiling, model
  calls and candidate evaluation, then ten minutes reserved for final checks.
  Hardware collection and additional review cost count against the assisted arm.
  The 128-call cap and disabled early-author-stop avoid the old four-proposal cap.
- Six new development profiles; eight separately frozen final profiles. Each
  contains two complete requests. Shapes, seeds and distributions differ from
  previous studies; changing weights/orders, masking, flat signals and shuffled
  float32 inputs exercise important branches.
- Three fresh-process discovery pairs per case; twelve final pairs. Exact output,
  input and retained-output fingerprints; geometric-mean speedup at least 1.02×;
  no case below 1/1.03×; fixed-sample sign-test p ≤ 0.05. Setup increase at most
  100 ms; traced peak and RSS at most 1.2× reference.
- CPU-only Apple M2 execution, one OpenBLAS thread. Arms run sequentially, with
  seeded order recorded before search. One selected candidate per arm is frozen
  before evaluation opens, with no subsequent author calls or reselection.

The earlier continuation did not qualify against every comparator in its original
study. Here it is an explicitly chosen stronger experimental baseline, not a
newly promoted deployment recipe. This pilot measures additional gains over that
source; it does not re-establish its earlier speedup against stock pybaselines.

## Interpretation

The key outcomes are accepted final improvement, reference retention, regressions,
invalid/duplicate proposals, model calls, search/total/provider time, diagnostic
collection cost and reported tokens. Subscription dollar cost remains unknown.
One paired trajectory on one application cannot establish a general advantage
for the checklist. Final workload cases are not independent author replications.

## Results

**The initial checklist did not improve the selected implementation in this
pilot.** Both selections passed the frozen exactness, performance, setup and
memory gates against the common continuation baseline, but the control was faster.
These are experimental patches, not automatically deployed recipes.

| Outcome | Previous author context | Initial checklist context |
| --- | ---: | ---: |
| Final speedup against the common baseline | **1.094292×** | **1.031511×** |
| Reduction in request time | 8.62% | 3.05% |
| Worst final case speedup | 0.979581× | 0.979566× |
| Fixed final sign-test p against 1.02× | 0.000244 | 0.019287 |
| Completed exact development candidates | 14 | 11 |
| Model calls, including final timeout | 15 | 12 |
| Search time | 20:00 | 20:00 |
| Total search and validation time | 22:57.69 | 22:54.41 |
| Provider time | 12:13.55 | 13:56.77 |
| Reported input tokens | 722,574 | 1,396,722 |
| Reported output tokens | 19,568 | 21,526 |

Both last responses timed out at the discovery deadline. No completed proposal
failed discovery exactness. All final memory gates passed: worst RSS ratios were
1.0323× and 1.0043×, and worst traced-peak ratios were 1.0007× and 1.0004×, for
control and checklist respectively. Worst setup increases were 1.72 ms and
0.59 ms. Small case regressions remained inside the predeclared 3% allowance.
Reported tokens cover completed provider turns; timeout usage may be missing.
Dollar cost is unknown, not zero.

The control selected candidate 13, combining loop specialization, identity-weight
work elimination, bound dot calls and coefficient-buffer reuse. The checklist
selected its first proposal, a guarded identity-weight shortcut. Later checklist
revisions did not beat it under the development gate. The control's stronger
discovery gains appeared late in its search, so the pilot also illustrates why
one early proposal is not a substitute for a sustained loop.

A direct comparison plan was fixed **after the control result but before the
checklist final result**, then executed after both authors stopped. Twelve fresh
paired rounds on the same final suite measured checklist/control speedup
**0.943350×**: the control was **1.060052× faster**. Every round favored control;
round ratios ranged from 0.932170× to 0.962986×. Exact observations matched. This
additional diagnostic cost 160.37 seconds, did not change either selection, and
is not another independent author replication.

The assisted run recorded reviews for all 22 cards: eight had tagged evaluated
candidates, thirteen were reviewed without an evaluated implementation, and one
was excluded by the hardware-family screen. That is coverage evidence, not proof
that eight independent hardware mechanisms improved. The CPU query took 0.274 s
and detected Apple M2/8 GiB, cache/core facts and supported feature flags; the
attribution worker recorded OpenBLAS 0.3.23.dev with one configured thread.
PMU counters and formal implementation proofs remained unavailable.

## Prompt overhead fix after the pilot

Repeated full timing records under several cards caused the assisted prompt to
grow to 561,912 serialized bytes by its last call. After freezing and measuring
both winners, the implementation was changed to keep only the latest two reviews
and attempts per card plus the current best eligible candidate, using compact
measured summaries. Complete evidence remains in local reports.

Replaying the last stored request through that compactor reduced it to 288,781
bytes, **48.6% smaller**. This is a byte-count check, not a new measurement of
provider tokens, latency or optimization quality. The author comparison above
used the original history representation. Both executed context modules and the
compaction receipts are included in the evidence; the changed representation
needs a future matched author replication before claiming an efficacy gain.

The harness supports more independent pairs through `--replicates`, with new
input seeds. Broader repositories, hardware and a docs-only versus structured-
review ablation would help determine when the additional context earns its cost.

[Raw evidence, comparisons and executed context sources](evidence/cpu-checklist-pybaselines.json.gz)
· [Control proposal](../experiments/application_intake/cpu-control-candidate.json)
· [Checklist proposal](../experiments/application_intake/cpu-assisted-candidate.json).
Both saved proposals apply to the prepared continuation baseline, not an
unmodified pybaselines checkout. The BSD-3-Clause notice accompanies them.

After the history fix, the product suite ran 143 tests: 142 passed and one skipped.
Static checks, wheel/source-archive inspection and clean installed-wheel checks
passed. Reapplying each saved proposal reconstructed its exact frozen source
identity. These checks validate integration and provenance; they do not replace
another model comparison of the compacted context.

## Reproduce

Use a source installation and an environment with NumPy, SciPy, pybaselines'
requirements and the signed-in Codex CLI. The command is foreground, unbuffered,
timestamped and logs progress at least once per minute. It can take up to about
one hour per pair, plus preparation and small bookkeeping costs.

From the repository root, replacing the interpreter and pinned checkout paths:

```sh
set -o pipefail
OPENBLAS_NUM_THREADS=1 PYTHONPATH=lossless-product/src \
  /absolute/path/to/venv/bin/python -u \
  lossless-product/experiments/application_intake/checklist_study.py \
  --project /absolute/path/to/pybaselines-v1.2.1 \
  --output .lossless/external/pybaselines-cpu-checklist-new \
  2>&1 | tee .lossless/external/pybaselines-cpu-checklist-new.log
```

The clean checkout must be at commit
`6f8d927e08577a9c5af5f34129b14d57c12725f6`. The script does not download code or
install dependencies. `--budget`, `--final-reserve`, `--max-candidates`, `--model`,
`--effort`, `--order-seed` and `--replicates` are configurable. `--prepare-only`
freezes the protocol without model calls; `--resume` executes that prepared
protocol and never silently reruns a failed/incomplete arm.

Each arm retains source identities, prompts, responses, discovery comparisons,
the frozen finalist, final paired measurements and memory checks. The assisted
arm also retains hardware, catalog and coverage. `summary.json` reports both arms
without changing either arm's independently frozen selection.

For the optional direct comparison, run `compare_checklist.py --study RUN
--declare-only` before the second arm completes, then run the same command without
`--declare-only` after both finish, with `OPENBLAS_NUM_THREADS=1`, the same
interpreter/PYTHONPATH and output piped to `tee RUN/direct-comparison.log`.
Package receipts with `summarize_checklist.py --study RUN --evidence FILE.json.gz`.
Current source uses compact history; reproducing the original pilot's prompts
requires the archived executed context modules in its evidence bundle.
