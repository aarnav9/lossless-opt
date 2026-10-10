# CPU array preparation experiments

Experiment: exact/lossless CPU optimizations on synthetic numeric arrays, integer plans, and sorted intervals. The measured operations are tensor indexing/copying, Python-to-array conversion, interval comparisons, and statistical reductions; no matrix multiplication or linear-system solve is measured.

Result: the frozen candidates passed the recorded finite output/input checks and startup/memory gates. The table reports isolated CPU operation speedups against each original callable.

| Workload | Evaluation inputs | Original → candidate | Observed speedup |
| --- | --- | --- | --- |
| Indexed tensor assembly | Six contiguous `float32` tensors of shape `[7, 3, 72, 72]`; a repeated/reordered index vector of length 10 produces `[6, 10, 3, 72, 72]` | Gather each input into a temporary tensor and stack → gather directly into final output slices | **2.21x** |
| Integer-array planning | `int64` vectors of length 4,099; batches of 32; full traversal and restart at batch 37; eight derived indices per item | Read tensor elements individually in Python → convert arrays/batch slices in bulk while retaining random draw order | **1.69x** full traversal; **1.71x** restart |
| Sorted interval matching | 4,096 synthetic overlapping intervals `(2i, 2i + 3)`; 64–65 query spans in mixed, late, and unbounded-end cases | Compute overlap against every interval → skip expired intervals and stop at future intervals | **18.21x** mixed; **10.31x** late; **10.71x** unbounded end |
| Dense tensor stack/copy | Sixteen contiguous `[3, 64, 64]` tensors; output `[16, 3, 64, 64]`; `float32` and `float16` | Stack into a temporary and copy into an independent output → stack directly into that output | **1.90x** `float32`; **2.07x** `float16` |
| NaN-aware column statistics | `[4096, 126]` `float32` matrix and `[1024, 42]` `float64` matrix containing NaNs, including an entirely missing column; calculations use `float64` | Compute five quantiles separately → compute them together; retain min/max/mean/std and missing-value handling | **1.80x** ordinary; **1.43x** missing values |

## Measurement and correctness scope

Recorded runtime: Linux x86-64, Python 3.12.12, PyTorch 2.11.0 and NumPy 2.4.4. PyTorch CPU operators used one thread. Each experiment evaluated one saved candidate after two discovery pairs, then six fresh-process reference/candidate evaluation pairs per case with balanced AB/BA ordering. Reported ratios are geometric means of paired warm-call speedups, not ratios of process startup times. Indexed assembly used 32 repetitions per call; stack/copy used 64. Recorded sequences contain two calls.

Tensor inputs use deterministic integer ramps modulo 251, cast to the stated dtype. Planning uses evaluation seeds 731 and 732. Statistical fixtures use NumPy generator seed 173. Discovery uses different fixture sizes, dtypes, or query positions from the stated evaluation cases. Quantiles are 1%, 10%, 50%, 90%, and 99% along axis 0.

Recorded checks compared output values/bits, shapes and types, unchanged inputs, and retained-output ownership as applicable. Planning also preserves final random-generator state and restart behavior. Interval matching retains the first maximum-overlap result. These finite checks do not prove equivalence for arbitrary inputs.

Timing is noisy: an earlier CPU smoke run of indexed assembly measured 1.66x on the same float32 fixture, versus 2.21x in the six-pair confirmation. The missing-value statistics case had one pair where the reference was faster (0.97x). Inputs containing negative zero use the original scalar quantile path to preserve signed-zero bits: on a `[4096, 126]` float64 control, the median paired ratio was 0.97x, so no consistent speedup is claimed for that fallback.

The large interval gains depend on sorted, finite intervals and query location. Stack/copy timings use ordinary pageable CPU memory. None of these ratios measures complete application throughput or guarantees gains on another workload/runtime.

This is a summary-only experiment record. No standalone reproducer or public raw timing bundle accompanies it; it does not introduce a reusable product recipe.
