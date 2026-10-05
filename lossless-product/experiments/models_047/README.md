# Campaign 047: model-choice pilot

Compare `gpt-6.1-sol`, `gpt-6-astra` and `gpt-6-luna` through the same iterative
native numerical softmax loop. Each has one trajectory, `xhigh` reasoning,
30 minutes including seed setup, model calls, compilation and discovery, and at
most six calls with one proposal per call. These are matched maximum allowances,
not matched token counts, inference compute or dollar budgets. Models can finish
early at the call cap. Expect roughly 90–110 minutes for the full pilot.

**Completed:** the recorded pilot took 75 minutes 53 seconds. Twelve new kernels
passed validation, one failed compilation and two author calls timed out. None
of the frozen model choices passed the full incremental gate against the shared
seed or enumeration. Luna used all six slots in 14.93 minutes; Sol and Astra
used about 29 minutes each. [Final results and evidence](../../docs/models-047.md).

All models receive the same retained recipes and campaign-044 source, current
incumbent, recent proposals, measured discovery feedback, failures and explicit
remaining time. The shared strong seed was originally authored by Astra; this
tests further optimization from that starting point, not blank-slate authorship.
The prompt explicitly distinguishes Gaussian timing from six-distribution
correctness validation. No tools, browsing or repository access are enabled.

There are nine discovery cases and twelve new final cases covering C, Fortran
and sliced layouts. The numerical contract, compiler and NumPy comparator are
fixed. Source duplicates, invalid proposals and timeouts consume call slots. A
fixed six-slot enumerated search provides an inexpensive control. Its maximum
candidate cap matches; actual model attempts can differ if their time expires.

Model order is a seeded shuffle, executed sequentially on one machine. All model
and control selections seal before any final case opens. Final evaluation and
fresh paired comparisons occur outside the search budget. Incremental comparison
requires at least 1.02× geometric mean and every case's paired lower interval
above 1×. One trajectory per model cannot establish a general model ranking.
No model-generated candidate is automatically promoted by this experiment.

Use an installed current-source Lossless environment and ChatGPT-signed-in Codex
CLI. On the recorded account, minimal probes passed for all three models with
CLI 0.159.0 before freezing the study. Access varies by account/client. This run
uses subscription allowance, not an API key. Dollar cost is recorded as unknown;
missing token usage after a timeout is also unknown, never zero.

The frozen seed uses Arm NEON and was tested on Apple M2 with clang, NumPy 1.26.4
and SciPy 1.16.0. This is not an x86-portable model comparison. Reproducing it on a
different target requires compatible source and a separately frozen experiment.

From the repository root, with a fresh output directory:

```sh
export PYTHONPATH="$PWD/lossless-product/src"
python -u lossless-product/experiments/models_047/study.py freeze --output research/results/models_047_new
set -o pipefail
python -u lossless-product/experiments/models_047/study.py run --output research/results/models_047_new 2>&1 | tee research/results/models_047_new.log
```

`freeze --seconds 1800 --calls 6` makes limits explicit. Freeze records every
input and harness digest. Do not edit frozen product/harness code during the run.
The author transport prints timestamped progress every 30 seconds. Measurement
jobs also report progress; result and log paths are printed at start and finish.

Checkpoints preserve completed rounds. Resume downtime counts against the
original model deadline. A partial author folder is refused instead of silently
retrying a subscription-consuming call. Evaluation wall time includes resume
downtime. A completed run is read-only on rerun, preserving its recorded cost.
Use a new directory for independent repetitions.

Archive with `scripts/curate_046_047.py 047 --study DIRECTORY --output BUNDLE.json.gz`.
Include probe and live-connector-smoke directories using `--extra` if desired.
The curator checks source/selection/receipt integrity, retains final code and raw
timing evidence, and excludes raw CLI event streams, stderr, auth/runtime databases
and binaries. [Protocol, results and limitations](../../docs/models-047.md).
