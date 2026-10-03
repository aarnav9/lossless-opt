# Product hardening and frontier experiments, 2 October 2026

The three requested product fixes are implemented. Six research directions were run as bounded pilots, with manual Codex proposals for the LLM arm. The useful results are an opt-in 027 graph recipe, lower native binding overhead, compatible search-cache reuse, and a promising softmax candidate. Scheduling was inconclusive; vocabulary-head bounds and tensor decomposition lost. These results do not establish general LLM search superiority or universal floating-point correctness.

[Public evidence](assets/frontier-032-038.json) contains timing samples, comparison records, solver outcomes, source hashes and deployment checks. [Experiment source](../experiments/frontier_032) contains the scripts and all three manual C proposals. Raw runs, frozen native harnesses, logs, artifacts and weights remain under local ignored `research/results/frontier_032/`. No weights or compiled binaries enter the public package.

## Product changes

Native jobs resolve and freeze `objective.comparator` before proposing kernels. Defaults are `numpy_copy` or `numpy_buffered`; callers may explicitly choose `native_baseline`, or `scipy_softmax_float64` for softmax with SciPy installed. Discovery selection and fresh evaluation acceptance use that comparator. Other library measurements remain diagnostics. Retaining the reference exports the declared library callable on a compatible environment. A regression test constructs a candidate that is 2× the C baseline but 0.9× the declared library and verifies rejection. This is an explicit deployment choice, not an automatically changing library envelope.

The MLX provider now honors `llm.timeout_seconds`, capped by remaining search time through the same helper as native search. Public repository access is correctly documented. GitHub had no release objects at inspection; the package remains a source alpha pending a tagged release and registry publication. This work does not publish either.

Native compilation and successful discovery validation can be reused under a content-addressed identity covering source, compiler/flags, contract, runtime, hardware and specialization. Evaluation correctness and all timings remain fresh. Corrupt cached entries become misses. Byte-identical proposals under different IDs are rejected before compilation; algebraic or semantic equivalence deduplication is not implemented. Cache storage is trusted local storage, not a defense against malicious candidates or a bounded disk cache.

Native deployment binding now avoids benchmark-only closures, red zones and initialized scratch allocations. The existing `operation.bind(...)` API reuses buffers across calls. MLX admission reuses a model fingerprint only within that admission operation, never as a persistent claim that mutable files remain unchanged.

## 032: graph extraction, complete requests and overhead

026 and 027 are alternatives to the retained recipe. Opt in with `workload.parameters.graph_recipes: ["decoder_026", "fullhead_027"]`; the default remains retained. Each loaded artifact owns a specialization cache with default 64 entries and a hard configuration limit of 128. Unknown specializations beyond capacity execute the retained path. This bounds specialization count, not compiler memory in bytes. Constant-weight capture requires the admitted immutable model and exclusive process scope.

All measurements used Apple M2, MLX 0.32.3, mlx-lm 0.32.0 and the pinned 8-bit SmolLM2 artifact. The historical fixture has eight requests, 33-token prefixes and 4–32 output tokens. Seven interleaved whole-batch repeats include tokenization, execution and required state materialization; loading is separate. TTFT, completion times, first validation-call cost, graph counters and peak MLX allocation are recorded. There is no additive per-layer profiler attribution.

| Evaluation measurement | Retained | 026 | 027 |
| --- | ---: | ---: | ---: |
| First process median batch time | 299.929 ms | 291.629 ms | 288.222 ms |
| Speed relative to retained | 1× | 1.0285× | 1.0406× |
| Descriptive paired bootstrap interval | — | [1.0103, 1.0370] | [1.0171, 1.0498] |
| Fresh process median batch time | 305.615 ms | 294.946 ms | 292.254 ms |
| Fresh speed relative to retained | 1× | 1.0362× | 1.0457× |

The replay had noticeable timing outliers and much wider intervals; all samples are retained. Neither recipe showed a convincing gain on the three-request ragged control. Do not multiply 026 and 027 ratios. First-use cost is not inferred from warm timings: replay discovery first validation calls were 354.5 ms retained, 367.2 ms for 026 and 332.5 ms for 027. Those are single observations after model loading and a serial validation call, not isolated compiler-latency measurements.

The product optimizer independently selected 027 with a 292.242 ms evaluation median against stock serial's 584.022 ms: **1.9984×**. Its incremental interval against retained was **[1.0261, 1.0447]**. Graph candidates must pass the stock comparator gate and an additional >1.01 lower-bound incremental gate against retained in both discovery and evaluation. Export/load preserved sampled tokens, full emitted probability arrays and active KV; eight continuation checks and cancellation fallback passed. A capacity-one test, changing batch shapes, and exception restoration passed. The fresh process added 24 forced-history checks that asserted graph construction actually occurred; initial forced-history checks had only exercised the retained path and are not counted as graph evidence.

Native binding setup fell from **15.81 to 8.61 µs** in its paired microbenchmark. Repeated calls measured **53.32 µs allocating versus 34.31 µs bound**. These are separate measurements, not multiplicative speedups. A repeated search took **5.90 then 4.48 seconds**, with four compile-cache and six discovery-validation-cache hits on the second run. This one cold/warm comparison suggests about 24% lower search wall time, not a general cache speed guarantee.

Reports now expose payback as `(search + extra setup) / seconds saved per call`. The selected MLX artifact estimated **133 eight-request batches** to repay the full 38.77-second search against stock serial after model loading. An existing retained deployment saves only about 11.46 ms per batch in this run: assigning the same whole-search cost gives a **roughly 3,383-batch search-only lower bound** for the incremental change. Native examples needed roughly **177,000–251,000 calls** to repay search alone. Unmeasured deployment setup leaves total native payback null. These estimates assume the measured saving persists and exclude human authoring/provider costs that were not measured.

## 033: does model-generated search help?

The frozen suite compared three candidates per arm: retained recipes, enumerated row-unroll variants, and three manually authored Codex tile/dispatch candidates. Each arm had a 90-second ceiling and the same discovery/evaluation shapes, timing policy and candidate cap. Evaluation shapes were frozen before authoring. Manual proposals used only control discovery feedback, then stayed frozen for all three workload replications.

| Arm | Accepted workload replications | Evaluation geometric speedup over buffered NumPy |
| --- | ---: | --- |
| Retained | 3/3 | 1.504×, 1.483×, 1.531× |
| Enumerated | 2/3 | 1.149×, 1.155×, 1.140×; last failed the per-case interval gate |
| Manual Codex | 3/3 | 1.683×, 1.677×, 1.655× |

All nine proposals were structurally valid. The manual winner was `layout_dispatch`. A separate 31-pair comparison against the retained column kernel measured **1.0433× C layout**, **1.1601× F layout**, and **1.0565× slice2**, with respective descriptive intervals [1.0250, 1.0798], [1.1544, 1.1643], [1.0450, 1.0658]. It uses the existing explicit numerical softmax contract, not a new bitwise guarantee. C-layout samples show warmup/drift; the raw sequence remains visible.

This is one shared assistant authoring session, not three independent model draws. Human/session latency and provider dollar cost cannot be isolated, so accepted improvements per total minute/dollar cannot honestly be computed. The manually generated source was genuinely new in this session, but replaying the checked-in proposals is deterministic replay. A callback used later to test packaging transports that frozen source; it is not another live LLM experiment. The candidate is preserved for further qualification, not silently added to default retained recipes. Next evidence needed: independent blinded authoring runs, randomized arm order and full authoring/cost accounting.

## 034: solver-guided program search

A small integer index IR feeds both Z3 and parenthesized C emission. For four declared finite shapes, the solver checks source/destination bounds, correct source selection and destination bijection for arbitrary copied bits. Three traversal orders passed. Swapped strides, dropped output and off-by-one variants were rejected with counterexamples before compilation. Ordinary unverified search evaluated the same six-member pool using the product's independent checks. Both arms retained NumPy; the legal generated loops did not beat it.

The first successful pilot used separately maintained solver/emitter expressions and measured 6.43 seconds unverified versus 4.50 including proof. It was then replaced and rerun with a shared index IR; the shared-IR run took **6.71 seconds unverified versus 4.51 seconds including 0.13 seconds of proof**, about 33% less. See `verified_ir` evidence for the full comparison. This isolates the value of cheaply rejecting known-invalid proposals, not a demonstration of novel kernel discovery. Both arms have equal proposal ceilings, but the verified arm intentionally spends fewer hardware evaluations. An equal-wall-time reinvestment experiment remains open.

Z3 also confirms a float32 reassociation counterexample at `(33554432, -33554432, 1)` and a 32-bit double-not identity. These do not prove a compiler, arbitrary C memory ownership, arbitrary shapes, GPU execution, or a general rounded-arithmetic rewrite system. No e-graph or Lean/FloatLib integration was attempted. The initial choice of a smaller reassociation counterexample failed an assertion and was corrected; failed logs remain in the local archive.

## 035: joint scheduling and memory lifetimes

Measured 14 compatible cohorts of six real MLX requests with irregular output lengths and arrival times, then exhaustively searched 81 schedule/lifetime states under a declared exported-storage budget. The solver chose the same cohorts as the retained scheduler and avoided the final materialization proxy. Replay medians were 169.3 ms baseline and 161.7 ms solver, but that small five-repeat difference is **inconclusive** and earns no promotion.

The model accounts for exported storage including spare capacity, not logical active KV bytes or actual peak device memory. Its budget excludes singleton states with larger retained capacity. `mx.array` is a materialization proxy, and replay consumes cohort outputs sequentially; this is not a production streaming allocator. The result says the solver can optimize its small declared model, not that it improves real scheduling broadly. Twelve exact token/probability/KV checks passed. A first probe incorrectly assumed cache metadata all had `nbytes`; it was repaired to count arrays only.

## 036: vocabulary-head avoidance

Collected eight real hidden states and the model's dequantized 49,152×576 head. A separate token-only CPU reference uses scalar float32 products/additions, then double log-sum-exp, float32 normalized scores and first-index tie selection. Suffix absolute-value bounds, rounded-reduction error allowances and a conservative normalization gap attempt to avoid low-scoring rows; uncertain cases execute the full reference. Zero/tie, tiny and large-magnitude controls are included.

All 11 token decisions matched that named reference. On real states, the method still computed roughly **94.6–99.9%** of multiplication work and achieved only **0.373–0.393×** reference speed. Zero/tiny cases fell back and were slower still. **Reject this implementation for deployment.** The normalization/transcendental allowance is analytically motivated but not mechanically verified. This is not a certified production selector, not equivalence to MLX's quantized matrix arithmetic, and cannot replace the product's full-probability contract. A tighter, cheap bound would be necessary before investing in integration.

## 037: candidate racing

Real paired timings used retained, manual, enumerated and a deliberately 8×-work slow control. A bounded score with Hoeffding intervals and an alpha-spending schedule accounts for repeated inspection under independent stable-mean sampling assumptions. The offline trace and actual live stopping loop both selected the manual winner; separate confirmation is included.

Live fixed sampling took **3.003 s** and live racing **1.086 s**, including decision overhead. The slow control was eliminated after 34 samples. Close contenders were not eliminated early. This **2.76×** saving depends on the intentionally bad control; it does not show equal gains for a competitive candidate pool. Randomized pairing reduces order effects but does not establish independence or remove thermal drift, so no hardware-level coverage guarantee is claimed. Keep racing experimental until close-contender and drift stress tests validate the decision rule.

## 038: hardware-aware tensor decomposition

Tested 15 rank-seven Strassen 2×2-block variants formed by five shear bases and three accumulation orders. SymPy noncommutative expansion verified each algebraic identity before timing. Real first-layer projection weights supplied the 576×576 matrix, with 8- and 33-row activation workloads. Packing, additions and temporary allocation are included. Evaluation uses fresh perturbations and cancellation controls against a float64 oracle under an explicit numerical tolerance.

Every tested numerical gate passed. The frozen winner achieved only **0.302× NumPy BLAS for 8 rows** and **0.556× for 33 rows**; cancellation controls were similar. Extra additions, memory movement and small GEMMs overwhelmed reduced multiplication count. **Reject for this target.** This is a hardware-aware search over an established rank-seven family, not discovery of a new rank or replication of the 48-multiplication 4×4 result. Numerical equivalence does not satisfy the product's bitwise model contract.

## Packaging and validation

The final CPU suite passed 48 tests with one optional Metal test skipped; separate Metal checks above ran on the qualified M2. Static and formatting checks passed. Wheel and source archive inspection found no research workspace, weights, credentials or generated native binaries. A clean temporary environment installed the wheel plus NumPy 2.4.6 and passed the no-key demo, exact export/load and theorem-resource retrieval outside the checkout. The built wheel's package was also loaded with the existing pinned MLX dependencies and independently selected 027 again; export/load, eight continuations and cancellation passed. This is not a fresh MLX dependency installation or a Linux/GPU compatibility expansion. These validation runs did not create a tagged GitHub release.

## Reproduction

Use a fresh output directory. The complete sequence is estimated at 3–6 minutes after dependencies/model download on the qualified M2; each pilot was under a minute locally apart from setup. It is foreground, unbuffered and logs progress. No hosted provider key is used. GPU experiments require the pinned local model fingerprint, Metal access and the optional MLX dependencies. A new public checkout must obtain the matching model separately; experiments never download or redistribute weights. CPU-only tracks can be run individually.

```sh
python3.11 -m venv .venv-frontier
.venv-frontier/bin/python -m pip install -e './lossless-product[mlx,numerical]' z3-solver==5.1.0.0 sympy==1.12
export PYTHONPATH="$PWD/lossless-product/src"
frontier_python="$PWD/.venv-frontier/bin/python"
frontier_scripts="$PWD/lossless-product/experiments/frontier_032"
frontier_model="$PWD/research/artifacts/fast_smollm2_mlx_8"
frontier_out="$PWD/research/results/frontier-replay-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$frontier_out"
set -o pipefail
run_track() {
  local track="$1"; shift
  "$frontier_python" -u "$frontier_scripts/$track.py" "$@" 2>&1 | tee "$frontier_out/$track.log"
}
for phase in prepare controls manual evaluate; do
  "$frontier_python" -u "$frontier_scripts/native_search.py" --phase "$phase" --output "$frontier_out/search" 2>&1 | tee "$frontier_out/search_$phase.log" || break
done
run_track verified_search --output "$frontier_out/verified_search"
run_track overhead --search "$frontier_out/search" --output "$frontier_out/overhead"
run_track racing --search "$frontier_out/search" --output "$frontier_out/racing"
run_track mlx_study --model "$frontier_model" --output "$frontier_out/mlx_study"
run_track mlx_deploy --model "$frontier_model" --output "$frontier_out/mlx_deploy"
run_track scheduling --model "$frontier_model" --output "$frontier_out/scheduling"
run_track projection_data --model "$frontier_model" --output "$frontier_out/projection_data"
run_track certified_head --data "$frontier_out/projection_data/projections.npz" --output "$frontier_out/certified_head"
run_track tensor_search --data "$frontier_out/projection_data/projections.npz" --output "$frontier_out/tensor_search"
```

The head pilot currently uses macOS `-dynamiclib`; Linux needs the corresponding shared-library flag. Verification is bounded; copy finite-shape solver checks do not qualify unrelated inputs. Bootstrap intervals are descriptive and have no correction for all experiments in this document. The archive preserves failures as well as successes. Future studies should use independent processes and devices, tighter predeclared endpoints, and a workload-derived latency/memory objective before changing defaults.
