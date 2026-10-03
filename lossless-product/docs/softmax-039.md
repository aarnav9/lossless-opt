# Softmax qualification, campaign 039

**Decision: keep the campaign 033 layout dispatch experimental.** It passed all tested correctness checks and improved the geometric mean, but seven shape/layout cases repeatedly regressed beyond the rule frozen before measurement. The retained built-in recipe list is unchanged.

The candidate source was frozen by hash. Twenty-five shapes, three layouts (C, Fortran and every-other-column), and three fresh sequential worker processes produced **225 case/process measurements** on Apple M2. Shapes include single rows/columns, tile boundaries and tails, widths through 4,097 and heights through 257. Eleven distributions include Gaussian/uniform, shifted and constant values, peaks, near ties, alternating ±10,000, negative-bound constants, signed tiny values and exponential-underflow steps.

All **14,850 implementation/distribution checks** passed against the independent float64 oracle, numerical tolerances, row sums, red zones and input immutability checks. This is the existing numerical softmax contract, not bitwise identity or a proof for every input. Three processes on one machine do not replace external hardware replication.

Each case screens the two retained recipes, freezes the faster one as its retained comparator, and then measures separate confirmation blocks. NumPy buffered softmax and SciPy's float64 implementation are also timed. Candidate code is not revised using these results.

| Fresh process | Geometric speedup over screened retained winner | Over buffered NumPy | Over float64 SciPy |
| --- | ---: | ---: | ---: |
| 1 | 1.0239× | 2.4862× | 3.7996× |
| 2 | 1.0410× | 2.6043× | 3.9104× |
| 3 | 1.0439× | 2.6779× | 4.1099× |

The promotion rule required every correctness check, at least 1.02× geometric speedup in each process, and no case whose upper paired interval was below 0.95× in two or more processes. The following cases failed the last condition:

| Shape/layout | Speedup by process |
| --- | --- |
| 3×31 C | 0.921×, 0.974×, 0.915× |
| 4×33 F | 0.922×, 0.922×, 0.923× |
| 5×127 F | 0.881×, 0.873×, 0.882× |
| 7×255 C | 0.910×, 0.906×, 0.927× |
| 7×255 F | 0.791×, 0.788×, 0.787× |
| 7×255 slice2 | 0.922×, 0.923×, 0.935× |
| 15×511 F | 0.943×, 0.931×, 0.936× |

For example, the 7×255 Fortran case takes roughly 27% more time despite the positive suite average. Small batches and incomplete tiles are the useful next counterexamples. A future guarded dispatch should be frozen and tested on new cases before promotion; this study did not tune a guard after observing the results.

The run took 65.1 seconds. [Public evidence](assets/softmax-039.json) contains source, the frozen plan, all timing samples/orders, environments and per-implementation correctness summaries. The optional `softmax-039-evidence.json.gz` prerelease asset contains complete per-distribution records. Its SHA-256 is included with the release assets. Bootstrap intervals describe within-case noise and do not correct for all comparisons; repetition reduces some instability but does not establish a device-wide statistical guarantee.

From the repository root, with the product and numerical extra installed:

```sh
set -o pipefail
mkdir -p research/results
PYTHONPATH=lossless-product/src python -u lossless-product/experiments/softmax_039/qualify.py --output research/results/softmax-qualification-replay 2>&1 | tee research/results/softmax-qualification-replay.log
```

Estimated runtime is 2–4 minutes after setup; the original run was faster. The command prints flushed, timestamped progress per case and result/log locations. A new output directory is required. No LLM or model weights are involved. Large run directories and generated libraries stay outside the source distribution.
