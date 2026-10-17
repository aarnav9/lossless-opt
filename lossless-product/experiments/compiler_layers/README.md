# Compiler-layer ModPoly pilot

Six local experiments exercise CPU targeting, allocation-derived alias contracts,
whole-loop specialization, unused-store elimination, explicit ARM NEON, and a
combination selected from the earlier discovery measurements. The selected
finalist is evaluated once, then confirmed in a separate process without
reselection. See [results and limits](../../docs/cpu-compiler-layers-study.md).

Requirements: macOS arm64, Clang, Python with NumPy/OpenBLAS ILP64, SciPy,
threadpoolctl, and the pinned pybaselines source checkout used by the existing
CPU studies. The recorded environment uses `.venv-integration` and
`.lossless/external/pybaselines-v1.2.1` at commit
`6f8d927e08577a9c5af5f34129b14d57c12725f6`. The runner validates the exact relevant
upstream source hashes. It makes no network/provider calls and does not edit
the upstream checkout or product runtime. It constructs isolated research
subclasses with the retained source bodies inside its own process.

The published Python runners received formatting-only changes after data collection.
The evidence archives retain the exact experiment-time scripts under their repository
paths in `text_files`; existing recorded runs require those versions for hash-checked
confirmation. Fresh runs and confirmation can use the published runners together.
Formatting was checked to preserve the Python syntax tree; recorded measurements
and evidence archives were not rewritten.

From the repository root, with a **new output directory**:

```sh
set -o pipefail
OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  .venv-integration/bin/python -u \
  lossless-product/experiments/compiler_layers/pilot.py \
  --output .lossless/compiler-layers-replay \
  2>&1 | tee .lossless/compiler-layers-replay.log

OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  .venv-integration/bin/python -u \
  lossless-product/experiments/compiler_layers/confirm.py \
  --run .lossless/compiler-layers-replay \
  --output .lossless/compiler-layers-replay-confirm \
  2>&1 | tee .lossless/compiler-layers-replay-confirm.log
```

Both commands print flushed timestamps, case-level progress and result/log paths.
Allow up to five minutes; the recorded runs took tens of seconds. `--smoke` uses
two performance profiles and fewer pairs with different seeds. Smoke results
are harness validation, not accepted performance evidence. The confirmation
command verifies evaluator, source and binary hashes before loading the frozen
finalist. `--seed`, `--size-offset`, and `--pairs` support a further preregistered
replication; do not repeatedly tune these to obtain a passing result.

Exactness is byte agreement on tested output/state behavior, with independently
checked reference repeatability and input/output lifetimes. Numerical-only
results are recorded separately if they occur. No formal equivalence claim is
made. The C ABI relies on the retained Python caller's owned-buffer guarantees.

Timings cover the in-process spectrum application, with new factory creation,
first-request preparation and four warm requests reported separately. They
exclude imports, fixture generation, process startup, worker IPC, and compilation.
The reference is the current retained compiled helper; the frozen finalist is
also compared with the retained Python recipe. Passing against the native
reference alone is insufficient for unconditional product promotion.

`helper.c`, `helper.ll`, `helper.s`, and `build.json` preserve the generated code
and compiler evidence for every variant. `protocol.json`, `discovery.json`,
`selection.json`, correctness receipts, and `report.json` preserve the experiment
decisions and raw paired measurements. Existing result directories are never
overwritten. The public evidence archive also retains the original failed
smoke attempt (the harness initially selected the module wrapper instead of the
class method), the repaired smoke, and the subsequent full runs.

## Direct LLVM IR experiments

`ir_pilot.py` runs five direct IR experiments on the frozen C-level finalist:
two vector-width/interleaving metadata choices, independent square-root
scheduling, prefetch instructions, and vector-loop unrolling. The existing
application evaluator is reused without modifying it. The generated C stays
fixed throughout. See the [IR results](../../docs/cpu-llvm-ir-study.md).

This extension additionally needs LLVM `opt` 21.1.8; the recorded installation is
`/opt/homebrew/opt/llvm@21/bin/opt`. Supply `--opt /absolute/path/to/opt` for another
compatible installation. Different LLVM versions need requalification. No tool
installation or network access is performed by the runner.

```sh
set -o pipefail
OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  .venv-integration/bin/python -u \
  lossless-product/experiments/compiler_layers/ir_pilot.py \
  --output .lossless/llvm-ir-replay \
  2>&1 | tee .lossless/llvm-ir-replay.log

OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  .venv-integration/bin/python -u \
  lossless-product/experiments/compiler_layers/ir_pilot.py \
  --confirm .lossless/llvm-ir-replay \
  --output .lossless/llvm-ir-replay-confirm \
  2>&1 | tee .lossless/llvm-ir-replay-confirm.log
```

Use new result directories. Both commands print timestamps, per-case progress
and result/log paths; allow up to five minutes, though the recorded full runs
took tens of seconds. `--smoke` uses fewer profiles/pairs and different seeds.
Confirmation loads the previously frozen binaries and never reselects a winner.
When discovery finds no eligible candidate, the runner clearly records the
highest-scoring exact candidate as a diagnostic finalist rather than accepting it.

`input.ll`, `optimized.ll`, `edit.diff`, `helper.s`, `metadata.json` and
`build.json` preserve the actual transformations and emitted code. LLVM
`-verify-each` validates IR structure; separate application replay tests sampled
bitwise behavior. Neither constitutes a formal equivalence proof. The original
compiled helper, the previous C-level winner, the unchanged IR roundtrip and the
retained Python recipe are separate comparators; gains against the older helper
cannot be credited to this IR study.
