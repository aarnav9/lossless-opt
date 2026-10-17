# Direct LLVM IR pilot

Five local LLVM IR experiments completed on 2026-10-17. **None established an
additional useful speedup over the previous C-level winner.** All five matched
the reference output bits on discovery inputs. The strongest discovery candidate,
vector-loop unrolling, measured only **1.0029×** in evaluation and **1.0012×** in
fresh-process confirmation for factory setup plus five application requests.
Both failed the predeclared 1.02× improvement threshold. No product default,
recipe, or installed dependency changed.

The C-level winner already incorporates alias contracts, loop specialization and
removal of an unused store. It is the primary comparator here, so its previously
measured gain cannot be counted again as an LLVM IR improvement. See the
[preceding compiler-layer study](cpu-compiler-layers-study.md).

## What was changed

The C source was frozen at SHA-256
`b306765fee1e47d1170609b3bd535b40a727e385cd42d3f43e8316a686177bc8`.
All five experiments edited `.ll` files or applied LLVM transformations to them;
none changed the C algorithm. Homebrew LLVM `opt` 21.1.8 was already installed
and accepted the Apple compiler's IR. Apple Clang 21.0.0 emitted the machine code
with its additional middle-end pipeline disabled, preserving the explicit
experimental pipeline.

| Experiment | Direct IR change | Discovery warm requests | Discovery factory + five requests |
| --- | --- | ---: | ---: |
| 1 | Set pointwise-loop vector width 2, interleave count 1 | 0.9815× | 0.9840× |
| 2 | Set vector width 4, interleave count 2 | 0.9963× | 0.9987× |
| 3 | Move four independent numerator square roots after the second dot call | 0.9923× | 0.9938× |
| 4 | Insert 256-byte-ahead `llvm.prefetch` hints in vector loops | 0.9963× | 0.9960× |
| 5 | Request twofold unrolling of eight vector loops | 1.0023× | 1.0031× |

The first two candidates use an additional scalar IR seed emitted from the same
C source, with vectorization and unrolling initially disabled. The runner edits
loop metadata and invokes the vectorizer directly. An unedited automatic
vectorization control measured 0.9871× the C-level winner. Experiment 1 produced
the same normalized assembly instructions as that control, so its explicit hints
did not add an implementation improvement over the automatic choice.

Experiments 3–5 start from the existing optimized IR unless an earlier candidate
passes discovery acceptance. No full-run candidate did, so none became a parent.
The smoke run's parent choices are separate and are not performance evidence.
Square-root scheduling preserved the order of both BLAS calls and the arithmetic
expressions. Prefetch added hints without `inbounds` or dereferenceability
promises for the lookahead address. LLVM documents prefetch as a nontrapping hint
that does not change program behavior. Loop unrolling was confined to pointwise
vector loops, excluding the convergence loop and BLAS reductions.
[LLVM transformation metadata](https://llvm.org/docs/TransformMetadata.html),
[LLVM prefetch semantics](https://llvm.org/docs/LangRef.html#llvm-prefetch-intrinsic).

## Controls and frozen evaluation

Discovery used seven profiles and eight counterbalanced timing pairs per
profile. An unchanged optimized-IR roundtrip measured **0.9999×** the C-level
winner. It is a separate comparator because compiling through IR need not emit
identical instructions to the direct C compilation.

No candidate passed discovery acceptance. Experiment 5, the highest-scoring exact
candidate, was frozen as a **diagnostic finalist**, not an accepted optimization.
It was evaluated on nine fresh profiles with twenty paired rounds, then loaded
unchanged into a fresh process with new shapes, values and twenty-four pairs.
Selection did not change after evaluation.

| Comparator | Evaluation warm | Evaluation factory + five requests | Confirmation warm | Confirmation factory + five requests |
| --- | ---: | ---: | ---: | ---: |
| **Previous C-level winner** | **1.0026×** | **1.0029×** | **1.0013×** | **1.0012×** |
| Unchanged optimized-IR roundtrip | 1.0027× | 1.0022× | 1.0010× | 0.9998× |
| Original retained compiled helper | 1.0465× | 1.0439× | 1.0396× | 1.0367× |
| Retained Python recipe | 1.0413× | 1.0435× | 1.0331× | 1.0383× |

The gain over the original compiled helper largely reflects the preceding
C-level work. It is not a new 4% IR gain. The Python comparison still has the
known flat-input regression: 0.7912× / 0.7953× for factory-inclusive calls.
The original-weighted confirmation case also regressed beyond the 3% allowance
against the unchanged IR roundtrip. No native or product acceptance gate passed.

Acceptance required exact outputs, at least 1.02× on both warm and
factory-inclusive aggregates, no profile slower than 1/1.03, and final aggregate
paired sign-test p-values at most 0.01. Native acceptance compared against the
previous winner, unchanged IR roundtrip and original compiled helper. Product
acceptance additionally required the retained Python comparison to pass.

## Correctness and compiler evidence

All candidates compiled and passed LLVM verification. **`opt -verify-each` checks
IR well-formedness, not semantic equivalence.** Alive2 or another formal
equivalence verifier was not used. Separate application checks established
bitwise agreement on tested inputs, reference repeatability, unchanged inputs,
and stable independently owned output buffers. Numerical tolerance was recorded
separately; no numerical-only candidate arose.

The finalist passed 30 profiles against the previous winner in each final
process: nine application profiles, ten edge controls, and eleven short-tail
lengths spanning 2–33 elements. These include signed zero, tiny/large finite
values, order-zero fallback, one iteration, tight convergence, sparse weights,
read-only input, intentional in-place clipping and strided fallback. The nine
application profiles also matched the retained Python recipe. Native traces
record which paths executed; fallback timings are not credited to IR changes.
No contraction, fast-math flags, or reduction-order changes were introduced.

The emitted assembly confirms that these were real compiler experiments:

- The prefetch candidate contains **18 `prfm` instructions**.
- Unrolling increased the static instruction-line inventory from **765** in the
  unchanged IR control to **1,093**. More static code did not yield a useful
  application improvement.
- Square-root scheduling changed the emitted instruction text while retaining
  eight static square-root sites across the specialized paths.
- The matrix products and dot reductions continue to call the same loaded
  NumPy OpenBLAS library. Its internal kernels were not transformed.

Static instruction counts are not executed instruction counts. These results
do not establish why every candidate lost or rule out IR optimization for other
hotspots. They show no useful incremental win from these five changes on this
already optimized helper.

## Scope and reproduction

Environment: macOS arm64, Apple Clang 21.0.0, LLVM `opt` 21.1.8, Python 3.11.3,
NumPy 1.26.4 and the pinned pybaselines source used by the preceding study, with
one OpenBLAS thread. No downloads, provider calls or REA were needed.

Timings cover an in-process spectrum application: factory creation, initial
metadata preparation, normal validation, fitting, correction and output
construction. They exclude interpreter/import startup, fixture generation,
worker IPC and compilation. This is not exported-callable qualification or a
memory/RSS study. The full run took about 28 seconds; confirmation about 13
seconds, within the initial under-five-minute estimate.

- [Runner and commands](../experiments/compiler_layers/README.md#direct-llvm-ir-experiments)
- [Compressed evidence](evidence/cpu-llvm-ir.json.gz): protocols, raw timings,
  correctness digests/traces, edited and optimized IR, diffs, generated assembly,
  compiler commands, remarks, and evaluator sources.
- Local artifacts: `.lossless/llvm-ir-20261017/` and
  `.lossless/llvm-ir-confirm-20261017/`, with adjacent `.log` files.

The IR runner is reusable research infrastructure. The earlier C-level candidate
remains the stronger implementation lead; these new trials add no accepted
performance improvement to stack onto it.
