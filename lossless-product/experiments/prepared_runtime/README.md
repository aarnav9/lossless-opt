# Sequential prepared-runtime study

Five families, followed by frozen combination checks on held-out inputs. No model
calls. Read [results and boundaries](../../docs/cpu-request-runtime-study.md) before
interpreting the numbers. Negative features require private qualification flags;
they are not public deployment settings.

Use Python 3.11 with NumPy 1.26.4, SciPy 1.16 and the `applications` dependencies to
match the original input generation. Timings are local-machine observations. Run
from `lossless-product` with a full Git checkout containing baseline commit
`d13261f57f9f706c61da464f5fe6efd09aaf1d9b`. Each comparison takes about 1–3 minutes on
the measured machine. Run comparisons sequentially, without other benchmarks.

## Reconstruct the recorded source and fixtures

```sh
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export REQUEST_STUDY=/tmp/lossless-request-reproduction
python -u experiments/prepared_runtime/restore.py --output "$REQUEST_STUDY"
```

The output must not exist. This extracts the old runtime with `git archive`, applies
the saved patches, rebuilds inputs using the archived generators, and checks every
runtime/source/fixture file hash. The evidence contains upstream source and
licenses; the numerical arrays are regenerated, not downloaded. Restored bundles
are explicitly research-only and do not carry deployment certificates.

## Run one family

For observation reuse:

```sh
set -o pipefail
python -u experiments/prepared_runtime/study.py \
  --family observations --features observations \
  --baseline "$REQUEST_STUDY/baseline-engine" --engine "$REQUEST_STUDY/engine-01" \
  --bundle financial="$REQUEST_STUDY/fixtures/financial" \
  --bundle pybaselines="$REQUEST_STUDY/fixtures/pybaselines" \
  --bundle arrays="$REQUEST_STUDY/fixtures/arrays" \
  --output "$REQUEST_STUDY/01-observations" --budget 240 \
  2>&1 | tee "$REQUEST_STUDY/01-observations.log"
```

Inspect each report before selecting the next configuration. Use a new output name
for every run. Keep the same three `--bundle` arguments for the remaining families:

| Family | Engine | Features | Other arguments |
| --- | --- | --- | --- |
| Persistent dispatcher | `engine-02` | `dispatcher` | — |
| Prepared guards | `engine-03` | `guards` | — |
| Live arrays, NumPy framing | `engine-04` | `values` | `--values` |
| Raw frames | `engine-05` | `raw` | — |
| Reusable buffers | `engine-05` | `raw buffers` | Use `engine-05` as baseline and `--baseline-features raw` |

The first four rows otherwise use `baseline-engine`. File requests load existing
fixtures inside the timed request. With `--values`, both controllers already own
the arrays: the old API stages them to files inside the timer, while the new API
transports values directly. Those two measurement boundaries must not be merged.

## Validate the frozen combination

```sh
python -u experiments/prepared_runtime/study.py \
  --family combined-fixtures --features observations raw --split evaluation \
  --baseline "$REQUEST_STUDY/baseline-engine" --engine "$REQUEST_STUDY/engine-final" \
  --bundle financial="$REQUEST_STUDY/fixtures/financial" \
  --bundle pybaselines="$REQUEST_STUDY/fixtures/pybaselines" \
  --bundle arrays="$REQUEST_STUDY/fixtures/arrays" \
  --output "$REQUEST_STUDY/heldout-fixtures" --budget 240 \
  2>&1 | tee "$REQUEST_STUDY/heldout-fixtures.log"
```

The live-value confirmation uses `--features observations raw values --values`
and a separate output/log path. Never adapt candidates from held-out timings.
Reports retain per-project source and runtime comparisons, request positions,
memory, startup and exact fingerprints. All rounds and returned values count;
there is no retry of a failed application request.

`prepare.py` can also make a fresh suite from previously selected Financial and
pybaselines application bundles using a different seed. `archive.py` curates all
completed/failed family reports, source patches, sources, generators and logs;
local bulk fixture files remain outside the repository.

## Exercise the public workflow

With local checkouts of NumPy Financial 1.0.0 and pybaselines 1.2.1:

```sh
python -u scripts/check_cpu_project.py --kind numpy-financial \
  --project /path/to/numpy-financial --matrix --seed 202610143 \
  --prepared --prepared-inputs values --budget 240 \
  --output /tmp/lossless-public-financial \
  2>&1 | tee /tmp/lossless-public-financial.log
```

For pybaselines, use `--kind pybaselines --project /path/to/pybaselines`, omit
`--matrix --seed`, and choose new output/log paths. These run source selection and
actual product qualification; they do not turn a research manifest into a certificate.
