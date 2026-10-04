# Campaign 044: do author deadlines change results?

Campaign 043 used identical prompts for its five- and 30-minute limits. The
authors were not told the timeout. Successful authors finished after 410.4,
607.6 and 441.4 seconds; the reported 25 minutes added the three authors and
their evaluations. None spent 25 minutes or exhausted the 30-minute allowance.

This study compares **announced 5, 30 and 60 minutes**, plus an **unannounced
60-minute control**, with three independent `gpt-6-astra` / `xhigh` authors per
condition. It retains the one-shot, three-proposal design and numerical softmax
contract from 043. It does not test sustained iterative search.

Authors run sequentially, in seeded randomized blocks. Every author gets the
same discovery context; only the announced allowance differs. The control gets
the original prompt with no duration. All 12 responses and every discovery
selection are frozen before opening any final cases. Final shapes are fresh;
the same cases and numerical distributions apply to every condition.

The evaluator uses retained recipes and deterministic enumeration as controls,
then measures frozen winners directly against retained and against one another.
There are no repairs, retries or default promotions. Incomplete or timed-out
responses count as failed author attempts, even if a response file was written.
Unknown tool events invalidate a response. Token receipts are reported when
available; missing timeout usage and unknown billing are not counted as zero cost.

Report actual elapsed time, completion and validation rates, paired speedups,
search cost and per-shape search-only payback. Announcing time is an instruction
to the model, not a guarantee that it can track a clock or will use the allowance.
Three replicates are exploratory. Timing-sample intervals do not establish a
population-level relationship between model reasoning time and kernel quality.

## Run

The recorded run used 145.3 minutes of authoring and finished the entire pipeline
in 2 hours 34 minutes. Provider speed and model stopping behavior can vary.
The maximum author allowances sum to **7 hours 45 minutes**, plus evaluation and
controller overhead. No weights are loaded; authors use the authenticated Codex
CLI with tools disabled. Native measurement remains local and single-threaded.

From the repository root, using the numerical development environment:

```sh
export PYTHONPATH="$PWD/lossless-product/src"
.venv-integration/bin/python -u lossless-product/experiments/timers_044/study.py freeze --output research/results/timers_044
set -o pipefail
.venv-integration/bin/python -u lossless-product/experiments/timers_044/study.py run --output research/results/timers_044 2>&1 | tee research/results/timers_044.log
```

The runner prints flushed UTC progress at least every 30 seconds during authoring
and each measurement batch, with result/log paths at startup and completion.
It writes `summary.json` and `REPORT.md` in the result directory. Use a new output
directory for a fresh study. Completed receipts are reused on resume; a started
author without a receipt is refused rather than silently retried. Frozen inputs
and harness source hashes must match before resuming.

## Recorded outcome

All three five-minute authors timed out. Every longer-cap author completed and
all 27 returned proposals passed numerical checks. Direct 60/30-minute winner
comparisons were 1.0023×, 1.0283× and 0.9954×; one pair was faster across every
final case. [Full results, cost, limitations and figure](../../docs/timers-044.md).

After a completed run, archive reviewed evidence and regenerate the figure.
From the repository root (with `PYTHONPATH` set as above):

```sh
.venv-integration/bin/python lossless-product/experiments/timers_044/curate.py --study research/results/timers_044 --output lossless-product/docs/assets/timers-044.json.gz
.venv-integration/bin/python lossless-product/scripts/plot_timer_study.py --evidence lossless-product/docs/assets/timers-044.json.gz --output lossless-product/docs/assets
```

Plotting requires the optional `matplotlib` dependency. The archive omits raw CLI
events, temporary runtime/auth metadata and binaries; it retains authored sources,
receipts, frozen inputs, controls, exact study sources and numerical/timing evidence.
