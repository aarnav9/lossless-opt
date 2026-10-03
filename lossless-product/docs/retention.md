# Extraction and validation

The source alpha copies reviewed runtime pieces into the product. It never imports the ignored research tree. `resources/PROVENANCE.json` records original source paths, hashes, and selected symbols. Subsequent product adaptations add packaging, strict job resolution, budgets, transport, validation, reports, and deployment guards.

| Research result | Alpha extraction |
| --- | --- |
| Native kernel campaign harness | Persistent frozen contracts, isolated compilation/benchmark workers, independent numerical oracle, red zones, input-mutation checks, screening/confirmation, proposal receipts |
| 017 scheduling | Equal-length cohorts, capacity/scheduling model, Lean lower-bound statements |
| 018 dead projections | Liveness masks and removal of unobserved terminal projections with final active KV preserved |
| 019–020 cache layout/movement | Guarded copy kernels, zero-padding checks, slicing, filtering/extraction and copy independence |
| 021–022 abstract transformations | Scoped fusion/thread-map proof sources and the pinned cache runtime |
| 023 allocator policy | Retained 256 MiB policy; optional LLM comparison against 0/64 MiB |
| Theorem library | Compressed statement index, source project, extraction receipt, Lean/mathlib pins and notices |
| 026/027 reusable graphs | Opt-in decoder/full-head alternatives with bounded specialization counts, fresh incremental gates, startup accounting and deployment checks; see campaigns 032–038 |
| Earlier MPS kernels and certified CPU solves | Not yet exposed as product adapters; retain their research findings in the ledger |

Historical 023 throughput gains around 1.96×/2.38× were scoped to its recorded batches and reference. They are not product-wide guarantees. No CUDA or computer-vision performance gain is claimed. The alpha MLX adapter deliberately preserves its model/runtime/device guards. Broadening them needs fresh evidence.

## Release checks

The CPU suite exercises compiler failures, incorrect kernels, crashes, timeouts, trusted-reference failure, source/contract/result tampering, discovery/evaluation separation, hardware collection failures, provider calls/timeouts, strict JSON, no-win/incomplete outcomes, export/load and reference fallback. It requires clang; SciPy adds the retained numerical baseline regression. Tests use small inputs and finish in approximately a minute on the development machine.

Lean checks compile all seven bundled scoped modules with the pinned toolchain. A short local MLX run compares the extracted recipe against stock serial inference and checks special cache bit patterns. A fresh environment must install the built wheel outside the checkout, run the CPU demo, export/load, and find all bundled resources without the research tree.

Full research performance reruns, fresh multi-GiB proof downloads, and broader backend/machine qualification remain separate release work. Short smoke results establish functionality and sampled correctness; they cannot establish that every historical speedup has been preserved.

## Local verification, 2 October 2026

On Apple M2 / macOS, Python 3.11.3: 41 CPU/policy tests passed with the optional Metal test skipped; all four Metal test-module checks then passed with the local model enabled. All seven retained Lean modules checked with Lean 4.28.0. Ruff static and formatting checks passed. Wheel and source archive inspections excluded research, environments, weights, credentials and generated binaries.

A new environment installed only the wheel and NumPy and passed the no-key demo, export/load and theorem-resource checks outside the checkout. With the optional MLX dependencies installed, the same environment passed GPU optimization, a provider-protocol fixture, export/load, exact token/probability/KV comparisons, and cancellation fallback. Native softmax and RMSNorm also passed installed optimization/export/load; RMSNorm retained its reference when the candidate did not qualify. Provider tests use local protocol fixtures, not paid hosted model calls.

A focused retention run then reused campaign 023 rounds 07/08's synthetic eight-request fixtures (33-token prefixes, requested outputs from 4 through 32 tokens). The installed package completed in about 21 seconds. With seven paired repeats per split, discovery reached 1.952× and evaluation 1.945× stock serial throughput; the descriptive evaluation bootstrap interval was [1.933, 1.988]. Every sampled token, emitted log-probability tensor and active KV state matched bit for bit. This preserves the main approximately 1.96× shrinking-batch result in this local check. It is not a full campaign replay, independent hardware replication, or universal correctness proof.

The input files are in [examples/mlx-fixed-count](../examples/mlx-fixed-count/lossless.json). Set its model path to the pinned local artifact, install the MLX extra and Lean, then run from the repository root:

```sh
lossless optimize lossless-product/examples/mlx-fixed-count/lossless.json --output ./lossless-runs/retention
```

The initial built wheel was approximately 467 KiB; the source archive including design diagrams was approximately 937 KiB. Base package plus NumPy occupied about 38 MiB allocated, or 62 MiB including the new environment's pip/setuptools. With the MLX extra, that environment occupied about 455 MiB, excluding the Python installation, downloaded weights and proof tools. These are local measurements, not cross-platform bounds; rerun `scripts/check_dist.py` for exact archive sizes after documentation changes.

The full first-time mathlib/Lean download was not repeated. Setup orchestration has a pinned-manifest regression test; scoped proof checks used the already installed Lean toolchain. GitHub's CPU matrix provides separate Linux/macOS installation checks; MLX still needs a qualified local Metal worker.

## README performance figure

The [README figure](assets/mlx-retention.png) compares the stock serial MLX reference with the installed alpha's retained exact recipe on the evaluation split described above. Both arms complete the same eight requests. The medians of seven paired warm repeats are **574.769 ms** for the reference and **295.445 ms** for the candidate. Their ratio is **1.945437× throughput**, equivalent to **48.598% less batch completion time**. The recorded descriptive 95% speedup interval is **[1.933331, 1.988043]**.

[Checked-in source data](assets/mlx-retention-evaluation.json) includes all 14 timing samples, paired run orders, all eight exact-comparison records, workload/runtime metadata, and the original local research record's SHA-256. This small evidence file makes the figure reproducible from a clone without the ignored research archive. Regenerating the figure does not rerun or independently reproduce the GPU experiment.

The vertical axis is throughput normalized to the same stock serial reference. The timing labels are medians of complete batch times. The candidate whisker is the source-reported descriptive speedup interval; the reference is normalized to 1×. This is one workload/device comparison, not a ranking of exact, numerical, and quality contracts. No live LLM call generated this measurement.

Model loading and first-use compilation are excluded. Bulk throughput can trade off first-token latency, and matching these finite cases is not a universal correctness proof. Reproducing GPU execution still requires the pinned M2/runtime/model artifact described above; the figure supplies no CUDA, vision, or other-device result.

To redraw the PNG and SVG from the repository root, install the optional plotting dependency in a development environment and run:

```sh
python -m pip install matplotlib
python lossless-product/scripts/plot_retention.py
```

The [plotting script](../scripts/plot_retention.py) validates the exact-comparison flags, sample counts, and ratio of medians before rendering. Matplotlib is not a Lossless runtime dependency.

## Follow-up product checks: campaigns 032–038

The current working tree passed 48 CPU/policy tests with one optional Metal test skipped (49 total). Separate M2 runs passed full token/probability/KV comparison, graph-capacity fallback, patch restoration, 24 graph forced-history checks, artifact export/load, eight continuations and cancellation fallback. This extends the earlier installed-alpha evidence above; the 026/027 options remain opt-in. See the [full follow-up report](frontier-032-038.md) and [curated measurements](assets/frontier-032-038.json) for the native search controls, failures, payback and reproducible commands.

The follow-up wheel and source archive also passed content inspection. A fresh temporary environment installed the wheel with NumPy 2.4.6 outside the checkout and passed the native demo, exact export/load and theorem-resource retrieval. Using the built wheel package with the existing pinned MLX dependencies passed a second graph optimization/export/load run, eight continuation checks and cancellation fallback. The main tests used 17.2 seconds; isolated build requirements were installed because the older research environment setuptools cannot parse current package metadata.
