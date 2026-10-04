# Campaign 045: one-hour propose → benchmark → revise pilot

One exploratory trajectory tests whether measured feedback helps beyond the
first response. Model: `gpt-6-astra`, `xhigh`; native numerical softmax on Apple M2.
This is a research harness, not a product default or packaged CLI connector.

**Completed pilot:** three rounds produced nine numerically valid proposals;
a fourth response timed out. Final paired speed was 1.00284× the first-round
choice and failed the incremental gate. See the [report and archived evidence](../../docs/iterate-045.md).

The 60-minute discovery budget includes initialization, seed benchmarks, author
calls, compilation and discovery measurement. Each response allows at most 30
minutes, clipped to remaining search time with 90 seconds reserved for discovery.
Do not start another call with less than three minutes available for its response.
There are at most eight rounds, three proposals each. Caps may remain unused.
Final confirmation and the enumeration control run afterward. Expect roughly
65–75 minutes for the complete pilot; no weights are loaded.

Each round is a fresh isolated CLI response with no tools. The external controller
supplies the incumbent and recent source, all discovery performance summaries,
failure diagnostics, and explicit elapsed/remaining time. Thus revisions receive
prior measured evidence without repository access or held-out cases. This tests
an external stateful loop, not persistence of hidden reasoning between calls.

Both arms start with retained recipes and one frozen campaign-044 kernel:
announced-30-minute author 0, proposal 2. Its historical performance is not sent to
the author. A seeded shuffled enumeration varies exact `expf` unrolling,
two-value detection and scalar/vector normalization of that same kernel. It uses
the same number of new slots attempted by the LLM, including failed responses.
Candidate/timing allowances match; actual elapsed time and tokens remain separate.

The first-round choice is checkpointed before later responses. All loop and
control selections are frozen before any final case opens. Fresh paired final
measurements compare the final choice against the first-round choice, prior
kernel, retained and enumeration. Incremental acceptance requires at least 1.02×
geometric-mean speed and every case's paired lower interval above 1×. This pilot
cannot establish a population-level advantage from one author trajectory.

Nine discovery cases and twelve fresh final cases cover C, Fortran and sliced
layouts. Correctness uses the existing numerical softmax contract and independent
float64 oracle. No approximate exponential, reduced precision, compiler-flag
changes, threads or external libraries are authorized. Duplicate source consumes
a proposal slot; this is exact-source deduplication, not semantic equivalence.
Failures can be revised in a later round and always remain in the record.

Timing uses Gaussian inputs; correctness covers six distributions. The frozen
author prompt lists correctness distributions without explicitly identifying
the timing distribution. This is a recorded protocol limitation, not evidence
that structured-row shortcuts improved the measured workload.

From the repository root:

```sh
export PYTHONPATH="$PWD/lossless-product/src"
.venv-integration/bin/python -u lossless-product/experiments/iterate_045/study.py freeze --output research/results/iterate_045
set -o pipefail
.venv-integration/bin/python -u lossless-product/experiments/iterate_045/study.py run --output research/results/iterate_045 2>&1 | tee research/results/iterate_045.log
```

`freeze` accepts `--seconds` (total discovery, default 3600), `--call-seconds`
(per response, default 1800) and `--rounds` (default 8). Limits are immutable
after freezing. Use a fresh output directory and matching named log for another
experiment; the paths above refer to the completed pilot in the original workspace.

The frozen plan records input and harness digests. Progress is unbuffered and
timestamped at least every 30 seconds during authoring and after measurement jobs.
Paths are printed at startup and completion. `loop.json` checkpoints completed
rounds; wall-clock downtime consumes the original deadline on resume. A partially
recorded round is refused instead of silently retrying a paid author call. Use a
new output directory for a fresh experiment. Do not rerun `run` on a completed
study to regenerate a chart: it can overwrite recorded evaluation wall time
with the duration of skipped jobs. The chart command in the report reads only
the immutable evidence bundle and makes no model or benchmark calls.

Archive a completed run with source/selection integrity checks and path redaction:

```sh
PYTHONPATH="$PWD/lossless-product/src" .venv-integration/bin/python lossless-product/scripts/curate_iterative_study.py --study research/results/iterate_045 --output lossless-product/docs/assets/iterate-045.json.gz
```
