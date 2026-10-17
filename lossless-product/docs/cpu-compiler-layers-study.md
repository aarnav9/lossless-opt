# CPU compiler-layer pilot

Six sequential local experiments completed on 2026-10-17. A frozen combination
of alias contracts, loop specialization and removal of an unused temporary write
improved the **existing retained compiled ModPoly helper** by **1.0431×** across
factory setup plus five application requests. A fresh-process confirmation on
new shapes and values measured **1.0390×**. Warm requests measured **1.0455×** and
**1.0367×**, respectively. Both passed the pilot's native-comparator gates.

This is a useful, modest additional CPU gain. It is **not a universal product
speedup or a new default**. The retained Python recipe is still substantially
faster for flat inputs. The finalist therefore failed the gate requiring it to
beat both existing alternatives without a case regression. No production code,
release, or installed dependency was changed, and no author/provider calls were
made.

## Experiments

Discovery used seven application profiles, eight counterbalanced pairs per
profile, and changing signal values in each request. Ratios are geometric means
against the current compiled helper; these are discovery results, not six
independently confirmed wins.

| Experiment | Change | Warm requests | Factory + five requests | Discovery exact |
| --- | --- | ---: | ---: | --- |
| 1 | Target the local CPU with `-mcpu=native` | 0.9952× | 0.9972× | Yes |
| 2 | Expose allocation-derived non-aliasing contracts; separate in-place clipping | 1.0418× | 1.0376× | Yes |
| 3 | Add whole-loop unit/weighted and source-alias specialization | 1.0280× | 1.0388× | Yes |
| 4 | Additionally omit the unused clipped-buffer store for separate-source weighted calls | 1.0439× | 1.0403× | Yes |
| 5 | Use explicit ARM NEON clipping/weighting in the specialized implementation | 1.0266× | 1.0235× | Yes |
| 6 | Combine the earlier measured choices | 1.0360× | 1.0335× | Yes |

Experiment 6 chose experiment 4's source. CPU targeting had not helped, so it
retained the original flags: this was a separately built repeat of experiment
4, not another independent optimization to multiply into the result. Explicit
SIMD did not outperform the selected compiler-generated implementation in
discovery. Neither this result nor the CPU-target result rules out those
techniques on other kernels.

The new C source preserves the loaded NumPy OpenBLAS matrix products and dot
reductions. It does not change floating-point operation order or enable fast
math. The alias contracts follow from the retained caller's independently owned
buffers; the intentional `source == clipped` case has its own implementation.
These are internal caller guarantees, not permission to invoke the C ABI with
arbitrarily overlapping arrays.

## Frozen evaluation and confirmation

Experiment 4 was selected using discovery only. Its source and binary were
frozen before twenty paired rounds on nine evaluation profiles. A second
process loaded the same hashed binaries and used twenty-four paired rounds,
another seed, and another set of lengths. There was no reselection afterward.

| Profile | Evaluation: factory + five requests | Confirmation: factory + five requests |
| --- | ---: | ---: |
| Small spectra | 1.0291× | 1.0259× |
| Changing weights | 1.1006× | 1.0627× |
| Long spectra | 1.0321× | 1.0331× |
| Original-source clipping with weights | 1.0266× | 1.0134× |
| Masking and shuffled inputs | 1.0094× | 1.0129× |
| Batch of spectra | 1.1346× | 1.1114× |
| Flat signals | 1.0039× | 1.0464× |
| Strided-source fallback | 0.9981× | 0.9811× |
| Float32 application inputs | 1.0614× | 1.0691× |
| **Geometric mean** | **1.0431×** | **1.0390×** |

The strided-source profile took the existing Python fallback; its timing changes
must not be attributed to the new native code. Float32 application inputs are
converted by pybaselines to its usual float64 internal computation, then returned
in the original output dtype.

All six variants matched reference output bytes on discovery. The selected
variant matched both the compiled and retained Python references on all nine
evaluation profiles plus ten edge controls: signed zero, tiny/large finite
values, order-zero fallback, one iteration, tight convergence, sparse weights,
read-only input, in-place weighted clipping, and strided-source fallback. It
also matched both references on the nine fresh confirmation profiles. Checks
cover the output tree (baseline, corrected signal, coefficients, weights and
convergence history), reference repeatability, unchanged inputs, and retained
output stability. Native-call traces establish which paths actually executed.

These are finite exactness checks, not an all-input theorem. No numerical-only
candidate arose; no task-quality claim was evaluated.

The predeclared native-comparator gate required at least 1.02× for both warm and
factory-inclusive aggregates, no profile slower than 1/1.03, and an aggregate
paired sign-test p-value at most 0.01. Both runs passed. Identical-source controls
were 1.0055× and 0.9955× factory-inclusive. This is evidence on one machine and
environment, not independent hardware replication.

Against the retained Python recipe, the finalist measured 1.0393× and 1.0441×
factory-inclusive overall, but flat signals measured only **0.7791× / 0.7887×**.
The compiled candidate must remain subject to workload qualification and the
existing reference fallback. Selecting a flat-input threshold using these
evaluation results would require a new experiment and fresh evaluation inputs.

## What the compiler artifacts show

The baseline already contains SIMD instructions. Explicit NEON did not introduce
vectorization where none existed. The specialized source makes the separate and
in-place buffer cases explicit, carries LLVM alias metadata, and removes the
unused store from the separate-source weighted loop. The compiler emits distinct
loop paths with fewer alias checks. CPU targeting changed the recorded LLVM
target from `apple-m1` to `apple-m2`, without a measured benefit here.

The finalist contains more static instructions because it specializes paths;
total instruction count is not a dynamic performance measurement. Experiments
2–4 are close enough that this pilot does not isolate how much of the gain comes
from each mechanism. The evidence supports the combined candidate. It does not
establish that removing the store alone explains the gain.

## Measurement scope and reproduction

Environment: Apple arm64, Apple Clang 21.0.0, Python 3.11.3, NumPy 1.26.4, and
NumPy's loaded OpenBLAS 0.3.23.dev with one BLAS thread. Compilation uses `-O3`,
`-fno-fast-math`, and `-ffp-contract=off` throughout. The source-available pinned
pybaselines 1.2.1 checkout is hash-checked before execution.

Each paired block includes a new application factory and five changing-input
requests. It includes normal application validation, metadata caches, fitting,
correction, result construction, and first-request preparation. Four later
requests form the warm metric. It excludes interpreter/import startup, fixture
generation, worker IPC, and compilation. This is not a timing of the exported
worker/callable bundle. Diagnostic compilation of the finalist (binary, IR and
assembly) took about 0.43 seconds, reported separately. Traced allocation on one
additional profile was approximately 371 KB for both implementations; this is
not a native RSS bound or a broad memory qualification.

The main pilot took about 18 seconds after startup on this machine; confirmation
took about 10 seconds. The initial conservative estimate was 3–5 minutes.

- [Runner and foreground reproduction commands](../experiments/compiler_layers/README.md)
- [Compressed evidence](evidence/cpu-compiler-layers.json.gz): raw paired timings,
  correctness digests, native-path traces, protocols, source/IR/assembly, build
  commands/remarks, evaluator sources and the confirmation results.
- Local full artifacts: `.lossless/compiler-layers-20261017/` and
  `.lossless/compiler-layers-confirm-20261017/`, with adjacent `.log` files.

The next integration candidate is the frozen specialized helper offered through
the existing compiler/source search with the Python alternative preserved. It
needs exported-callable qualification, deployment-cost measurements, and wider
environment testing before promotion. REA, direct assembly patching, PGO and
formal IR equivalence checking were not exercised by this pilot.
