# Sequential CPU campaign

This repository research harness runs twenty discovery experiments in each of
five families: compiled ModPoly regions, independent-spectrum pools, ordinary
native calls, prepared metadata, and matched author contexts with/without compiler
diagnostics. It is not an installed public CLI. Proposals consume earlier
discovery results; fresh final inputs are generated only after selection.

Run from the repository root with the integration environment and the clean,
pinned checkout at `.lossless/external/pybaselines-v1.2.1`. The application baseline
includes the previous continuation and CPU-control patches. Native trials freeze
the recorded deployment wrapper and keep the compiled numerical kernels fixed.
`frozen_native/` contains the original reference sources and hashes, so subsequent
product changes cannot silently change this study's native comparator.

## Reproduce a published finalist without a model

From a fresh source checkout, prepare Python 3.11 and the recorded dependencies:

```sh
python3.11 -m venv .venv-cpu100
.venv-cpu100/bin/python -m pip install ./lossless-product \
  numpy==1.26.4 scipy==1.16.0 numba==0.61.2 psutil==6.0.0 threadpoolctl==3.3.0
mkdir -p .lossless/external
git clone https://github.com/derb12/pybaselines.git .lossless/external/pybaselines-v1.2.1
git -C .lossless/external/pybaselines-v1.2.1 checkout 6f8d927e08577a9c5af5f34129b14d57c12725f6
set -o pipefail
OPENBLAS_NUM_THREADS=1 PYTHONPATH=lossless-product/src \
  .venv-cpu100/bin/python -u lossless-product/experiments/cpu_100/replay_finalist.py \
  --family 1 --output .lossless/cpu100-replay-f1 \
  2>&1 | tee .lossless/cpu100-replay-f1.log
```

Choose family 1–5 and a new output directory for each run. `--smoke` limits the
run to one case and two pairs (usually under 15 seconds); full replay uses eight
cases and sixteen pairs (roughly 3–5 minutes per family on the study machine).
The frozen selection and source come from the public evidence, including the
family-5 authored C source. No author calls or private run directories are used.
This reproduces correctness and complete-call timing on already-public cases;
it does not repeat the separate memory gate or establish a new holdout result.
Machine/compiler changes can change timing or exactness. The compiled full-loop
path requires the matching NumPy OpenBLAS ILP64 symbols and otherwise falls back.

## Rerun discovery

```sh
set -o pipefail
OPENBLAS_NUM_THREADS=1 PYTHONPATH=lossless-product/src \
  .venv-integration/bin/python -u \
  lossless-product/experiments/cpu_100/study.py --family 1 --count 1 \
  2>&1 | tee -a .lossless/cpu-100-20261011/campaign.log
```

`--prepare-only` freezes the baseline and discovery fixtures. `--controls` runs
the application identical-source control. Run `--family 3 --controls` before the
native-call family. Completed experiments are read from `ledger.json`; `--count`
runs the next requested number sequentially.
Inspect the previous result and revise the next hypothesis before continuing.
Never overwrite an experiment directory. Preserve infrastructure failures
separately before retrying an uncompleted slot.

After all twenty experiments in a family, `--family N --qualify` freezes one
choice and runs sixteen paired rounds on fresh cases, followed by separate
memory checks. `--qualify --resume-memory` resumes an interrupted memory pass
without selecting again or repeating completed final timing. A failed final
does not authorize replacing the selected candidate using those same final inputs.

Non-author discovery trials take roughly 15–60 seconds on the study machine;
final confirmation takes roughly 2–5 minutes. Author trials make two sequential,
tools-disabled Codex CLI calls at equal per-arm budgets, then evaluate their
sources. The complete campaign takes hours. Foreground output is unbuffered,
timestamped and identifies result/log paths. Long runs require explicit user
authorization when launched from Codex.

`--family 5 --preview-author` prepares the next pair's source, profiles and exact
JSON payloads under `author-preview/`, without invoking the provider or counting
a trial. The manifest identifies payload hashes, sizes, destination and data
categories for review. Author requests contain code, workload metadata and
diagnostics; they do not contain the signal-array fixtures.

The receipt distinguishes exact bytes and ownership, a fixed numerical envelope,
and rejected results. No task-quality labels are available. Exact native-wrapper
comparisons are relative to the unchanged kernel; softmax and RMSNorm retain
their original numerical oracle contracts. Independent oracle checks accompany
native correctness runs. Nothing is promoted from discovery timing alone.

Timing measures complete requests on new signal values and retained outputs;
fixture decoding and correctness hashes are outside timing workers. Setup is
reported separately and with calls. Explicit prepared handles have a different
API contract: they own grid/order metadata, charge preparation to setup, and
accept changing signal values and weights. They are not transparent ordinary-call
speedups. Pool initialization, IPC, output assembly, and startup are recorded.

Memory uses separate workers. The conservative process total combines worker
high-water marks with sampled auxiliary-process peaks and double-counts shared
pages; it is not a hard physical-memory bound. Python/NumPy traced allocations
and retained metadata payloads provide additional checks. Read the frozen
protocol and its numbered amendments before interpreting a result.

`conformance.py`, `runtime_conformance.py`, and `native_conformance.py` add
correctness and lifecycle checks outside the hundred performance experiments.
Raw cases, plans, source/binary identities, worker snapshots, logs, output arrays,
timings and reports remain under the run directory. These artifacts can occupy
several gigabytes. Finite validation on one CPU is not a universal equivalence
proof or a performance guarantee for other workloads and machines.

After all twenty author pairs, `audit_authoring.py` checks equal common context,
model/time allowances and compiler settings; saved prompt/response/source/binary
identities; absence of author tool use; and both responses preceding measurement.
This is an artifact/protocol audit, not an equivalence or performance proof.

After all 100 trials and all five final evaluations, `summarize.py` checks the
ledger against per-trial reports and emits portable compressed JSON evidence
and an all-trial Markdown appendix, including the author audit. It refuses to label an incomplete run as
complete. See the current [study report](../../docs/cpu-100-study.md).
