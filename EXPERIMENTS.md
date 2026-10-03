# Experiment ledger

Concise research memory for contributors. Read this before repeating an approach. Each entry has one experiment line and one result line; numbered IDs refer to campaigns, which can contain many configurations, controls, repairs, and repeats. Supplemental studies have descriptive IDs.

All results are scoped to the recorded hardware, runtime, model, and workload. **Exact** below means bitwise agreement on tested outputs/state, unless a specific proof is stated; Lean results have narrower declared scopes. Numerical tolerance, certified error, and measured task quality are different contracts. Speedups use the named comparator and cannot be added or multiplied across campaigns. Research qualification does not mean a product default was shipped.

New entries should name their requested category: exact/lossless, numerical, or task quality. Record untested, failed, and inconclusive outcomes explicitly. Categories describe the correctness contract; proof scope and validation coverage are separate evidence. See [contribution guidance](CONTRIBUTING.md) for a short submission format.

Detailed local evidence lives in the ignored `research/` workspace. Numbered reports are `research/results/reports/RESULTS_NNN.md` (`RESULTS.md` for 001); legacy entry paths also exist inside that workspace. Supplemental source locations appear below as plain paths because a public checkout does not contain those files. This ledger is contributor memory, not a replacement for reproducible evidence supporting a published performance claim.

## Numbered campaigns

- **040 — Experiment (product profiling):** Expose bounded CLI/Python profiling for supported native and MLX workloads, separate setup, allocating/bound calls, request latency/memory and instrumented host attribution, then validate against exported artifacts.
- **Result — product feature:** Profiling uses discovery inputs without providers or proof setup; native correctness, artifact integrity and budget regressions pass. Local Metal checks preserve sampled tokens/probabilities/KV and phase accounting. GPU kernel-level attribution remains outside scope. [Usage and limits](lossless-product/docs/profiling.md).

- **039 — Experiment (numerical qualification):** Freeze the manually proposed softmax dispatch and test 75 shape/layout combinations, 11 distributions and three fresh worker processes against a separately screened retained winner plus NumPy/SciPy.
- **Result — retain as experimental:** All 14,850 checks passed; process geomeans were 1.024×/1.041×/1.044× retained, but seven cases repeatedly failed the predeclared regression limit. The 7×255 Fortran case took about 27% more time. Built-in recipes remain unchanged. [Evidence and reproduction](lossless-product/docs/softmax-039.md).

- **038 — Experiment (numerical):** Verify 15 rank-seven tensor-decomposition variants symbolically, then benchmark real 576-wide projection shapes including packing and cancellation controls.
- **Result — rejected:** Best fresh speed was 0.302× NumPy BLAS for 8 rows and 0.556× for 33 rows despite all numerical gates passing. Reduced multiplication count did not repay additions and memory movement; no new tensor rank or bitwise model claim.

- **037 — Experiment (search efficiency):** Compare fixed sampling with a bounded-score sequential race on real native timings, including an intentionally slow correct control and separate confirmation.
- **Result — scoped pilot:** Live search took 3.003 s fixed versus 1.086 s racing and chose the same manual winner. Savings came from rejecting the 8×-work control after 34 samples; close contenders were not rejected early and device noise assumptions remain unverified.

- **036 — Experiment (separate token-only contract):** Test suffix-bound vocabulary-head avoidance on eight real hidden states plus zero/tiny/large controls against a named scalar FP32 normalized-score reference.
- **Result — rejected:** All 11 decisions matched, but real cases still computed 94.6–99.9% of products and ran at 0.373–0.393× reference speed. Bounds are not mechanically certified and do not establish MLX quantized-head equivalence or preserve full probability arrays.

- **035 — Experiment (exact/lossless pilot):** Calibrate 14 MLX cohorts for six irregular requests and enumerate joint scheduling/lifetime choices under a declared exported-storage budget.
- **Result — inconclusive:** The solver searched 81 states and kept the same cohorts, omitting a final materialization proxy. Five-repeat makespan medians were 169.3 versus 161.7 ms with 12 exact checks, insufficient for promotion; model storage/proxy costs are not a production allocator or device-memory proof.

- **034 — Experiment (exact/lossless, bounded proof):** Use a shared integer index IR to prove finite-shape copy mappings with Z3, emit C, and compare verified filtering with ordinary validation on six candidate transformations.
- **Result — no kernel promotion:** Three legal traversals passed and three defective mappings produced counterexamples before compilation; both searches retained NumPy. Filtering reduced measured search work, while a separate FP32 counterexample rejects reassociation. No general-shape theorem, verified compiler or GPU proof.

- **033 — Experiment (numerical):** Compare retained, enumerated and manually authored Codex softmax candidates with three proposals per arm, frozen holdouts and three workload replications.
- **Result — promising candidate:** Manual dispatch passed 3/3 replications versus retained 3/3 and enumeration 2/3; separate paired replay gained 4.3%, 16.0% and 5.6% over retained across C/F/slice layouts. One shared authoring session cannot establish independent-run model superiority or improvements per dollar; no default promotion.

- **032 — Experiment (exact/lossless MLX; native overhead):** Extract bounded opt-in 026/027 alternatives, compare complete requests, validate export/load and continuation, and measure binding plus compilation/evidence reuse.
- **Result — scoped product option:** Product-selected 027 reached 1.998× stock serial with incremental interval [1.026, 1.045] over retained; ragged gains were inconclusive. Fresh graph replay passed 24 forced-history checks; bounded-cache fallback and eight deployment continuations passed. Native binding fell 15.81→8.61 µs; one repeated search fell 5.90→4.48 s. Frozen comparator, configured timeout and public-release messaging fixes shipped in the working tree. [Evidence, limitations and commands for 032–038](lossless-product/docs/frontier-032-038.md).

- **031 — Experiment:** Deblur a real sample photograph with one fixed periodic quadratic model, comparing spatial CG, complex FFT, a strong real-FFT control, and buffer reuse across three sizes, two noise seeds, fresh/cached filters, and an independent process replay.
- **Result — scoped vision/numerical demonstration:** At 512², replay reached about 159× CG including setup or 264–276× cached; those large gains exploit established Fourier structure. Buffer reuse added only a few noisy percent over the strong real-FFT control, preserving tested bytes; all 96 method/case numerical gates passed. No generic vision-model or new Lean-proof claim, and no default promotion.

- **030 — Experiment:** Test six unquantized FP32 model rewrites, structured CPU algorithms, fresh replay, 128 paired task questions, and cancellation/conditioning stress cases.
- **Result — no model promotion:** Selected normalization folding reached 1.0105× in discovery but 0.9998× stock MLX on fresh contexts; task decisions matched but logits/KV changed. Structured CPU wins were regime-dependent, and severe Woodbury/refinement failures require guards and fallback.

- **029 — Experiment:** Explore real-algebra identities on FP32 matrix chains, explicitly low-rank solves, ridge optimization, least squares, and four model rewrites.
- **Result — numerical pilot only:** No useful model speedup; CPU structure helped some cases, but `multi_dot` already captured matrix-chain gains and ill-conditioned normal equations produced about 73% relative error. Real identities are not floating-point equivalence proofs.

- **028 — Experiment:** Run 40 explicit-state graph attempts using dependency deletion, topological scheduling, last-use release, Boolean adjacency matrices, and shared cache metadata against the 027 recipe.
- **Result — rejected:** Corrected variants passed exactness, but finalists failed fresh/replay speed thresholds and first capture rose from about 0.32 s to 2.08 s; two initial cache-aliasing failures are preserved. Fewer graph nodes did not yield faster execution.

- **027 — Experiment:** Run 40 further graph-capture trials with constant weights, smaller compiler keys, guarded mask omission, explicit KV state, and grouped projection/liveness logic.
- **Result — qualified research options:** Two recipes gained about 2.1–2.3% warmed speed against accepted023 on fresh/replayed shrinking batches with tested exactness; startup and compiled-cache growth still need controls. Do not combine these gains with 026 ratios.

- **026 — Experiment:** Run 40 execution-reuse trials comparing decoder compilation groups, explicit state, cache movement, and host-path reuse against accepted023.
- **Result — qualified research option:** Full-decoder compilation plus concatenation gained about 1.4–1.8% warmed speed against accepted023 on fresh/replay workloads with tested exactness; broader transfers were weaker, first capture cost more, and bounded compilation caches remain necessary.

- **025 — Experiment:** Screen native cache-capacity policies and build a reversible BF16 bitmap codec with Lean lane-reconstruction proofs and weight roundtrip tests.
- **Result — no inference promotion:** Cache gains did not reproduce; the codec preserves stored bits but has no GPU inference speed result. Reversible compression alone does not establish faster or bit-equivalent model execution.

- **024 — Experiment:** Compare manual launch-first and Lean-informed ordering of the same eight AutoKernel/MLX candidates, followed by fresh confirmation and memory checks.
- **Result — rejected:** The selected candidate gained only about 0.9–1.0% on fresh/replay bulk and failed retention gates; this order ablation did not demonstrate an incremental benefit from Lean guidance.

- **023 — Experiment:** Profile the exact MLX recipe and compare allocator-cache thresholds with several compilation/fusion candidates across 13 comparisons.
- **Result — retained scoped recipe:** A 256 MiB allocator-cache policy gained 4.6% fresh / 5.0% replay over the preceding exact recipe; total tested bulk throughput reached 1.96×/2.38× stock serial on named workloads, with extra retained memory and worse first-token latency than serial.

- **022 — Experiment:** Run 20 hardware-guided copy/indexing trials; add hardware discovery, 14 scoped indexing/fusion theorems, and a pinned searchable theorem library.
- **Result — no kernel promotion:** All tested bytes matched, but the small copy-launch discovery gain failed confirmation; hardware inventory and real/rational algebra identities do not supply a CUDA evaluator or prove GPU arithmetic equivalence.

- **021 — Experiment:** Adapt pinned AutoKernel orchestration to a local MLX/Metal evaluator and try three manually proposed projection/compilation candidates on raw and existing optimized paths.
- **Result — no incremental gain:** Candidates passed exactness but failed the performance rule; retain both starting paths. The bridge is not a native CUDA/Triton comparison or an upstream optimizer ranking.

- **020 — Experiment:** Test movement-bound-guided copy variants and package an opt-in text-to-text MLX preview with model/runtime admission, state export, and stopping/cancellation fallback.
- **Result — preview validated:** New copy variants lost; the existing recipe reached 1.85× stock serial in a text-to-text replay, 1.16× on varied prompts, and 2.08× on equal-count bulk. The Lean bound is logical copy work, not physical traffic or global speed optimality.

- **019 — Experiment:** Run 20 exact cache/layout trials with zero-padding metadata specialization, guarded prefix views, and a copy-only Metal append kernel generated from a shared indexing expression.
- **Result — retained scoped recipe:** The short-prefix case gained 17.7% over the preceding exact path and 17.3% on replay; mixed buckets gained 13.0%. Most benefit came from metadata handling; memory-heavy exported views and losing kernels remain negative evidence.

- **018 — Experiment:** Use scoped liveness proofs to remove wholly unobserved terminal projection groups while preserving final KV state and live group shapes.
- **Result — retained scoped recipe:** The one-token stress case gained 7.3% over 017 and 8.2% on replay with tested exactness; longer workloads gained little. Removing required final state updates is invalid even when generated tokens match.

- **017 — Experiment:** Prove invocation-count bounds in Lean and regroup compatible fixed-count MLX requests to attain them in the declared scheduling model.
- **Result — retained scoped recipe:** Heterogeneous-count workloads gained 13.8–14.6% over the previous exact batching policy; the no-headroom control tied. The proof concerns invocation counts, while backend equality remains tested.

- **016 — Experiment:** Run 28 MLX batching/model trials and 12 structured CPU trials, with native scheduler controls and cheaper per-RHS numerical certificates.
- **Result — scoped gains:** Restricted exact batching reproduced about 2.33× serial throughput, with a first-token latency cost; padded batching and a 360M BF16 transfer failed. A 512-dimensional CPU solve case reached 1.61–1.99× cached checked LAPACK under conditional error bounds.

- **015 — Experiment:** Run 40 manual MLX and structured CPU trials with stronger native-cache baselines, projection packing, graph solves, and residual-certificate specialization.
- **Result — scoped gains:** Ordinary model improvements were about 0.6–1.0%; a narrow cache case repeated 3.7–4.4% beyond native MLX. CPU preconditioning/certificate wins apply to constructed systems; most earlier prefix-reuse gain was already in the runtime.

- **014 — Experiment:** Run 20 Metal GEMM/GEMV layout, tiling, and fusion rounds inside cached FP32 model generation.
- **Result — numerical gains:** Standalone matrix replacements lost; decode epilogue fusion reached 1.48× original PyTorch MPS and a combined challenger 1.64×, replaying at 1.62× on the named 135M case. New accumulation order changes bits; preserve the earlier exact path separately.

- **013 — Experiment:** Run 20 predeclared MPS trials on two checkpoints, padded inputs, longer prefixes, and a newer PyTorch baseline.
- **Result — retained exact research path:** The revised primary reached 1.20–1.33× original PyTorch 2.14 MPS with tested full-logit/KV equality; the old kernel failed after a runtime upgrade. Earlier 1.9× figures used a substantially slower PyTorch baseline.

- **012 — Experiment:** Integrate real Metal normalization/rotary kernels into cached SmolLM2 generation across 20 adaptive local-GPU rounds.
- **Result — scoped exact gains:** Short cases reached 1.92–1.95× original PyTorch 2.2.2 MPS, while longer/batched cases gained less and CPU could still win; this was a fixed-step loop, not a serving-stack or universal GPU advantage.

- **011 — Experiment:** Run 20 pretrained-model CPU rounds testing native decoder boundaries and projection packing.
- **Result — scoped exact gain:** A complete one-token 135M forward call repeated about 1.15× eager PyTorch with matching tested logits; longer-prefill gains were inconclusive and packing increased memory. Cached generation was not covered.

- **010 — Experiment:** Run 20 sequential native CPU fusion/rewrite rounds on seeded MLP blocks and small stacks, including failures and repairs.
- **Result — scoped numerical gain:** A fresh-seed small-stack repeat reached 1.2785× stock Inductor / 1.7645× eager PyTorch; large-prefill gains remained inconclusive. The 2.179× isolated-region result is not a full-model speedup.

- **009 — Experiment:** Capture native PyTorch graphs and merge gate/up projections, with and without Inductor, across ten CPU configurations.
- **Result — rejected as default:** The primary averaged 0.754× the stronger eager/compiler baseline and the compiled rewrite 0.829×; two compiled wins did not offset eight regressions despite passing correctness checks.

- **008 — Experiment:** Integrate vectorized RMSNorm/residual fusion into 50 ONNX Runtime configurations, separating isolated regions from larger computations.
- **Result — scoped numerical gains:** Larger computations averaged 1.084× optimized ORT; isolated regions gained much more. Do not present the mixed 1.253× average or isolated peaks as whole-model acceleration.

- **007 — Experiment:** Insert the previously winning CPU softmax kernel into complete ONNX Runtime computations with a stronger optimized-runtime baseline.
- **Result — rejected:** Primary speed averaged 0.786× optimized ORT and the parallel variant 0.854×, with no confirmed wins in the revised suite; operator wins over NumPy did not survive integration.

- **006 — Experiment:** Build a resumable frozen-contract controller and search ten native stable-softmax candidates across 24 CPU configurations.
- **Result — scoped operator gain:** The frozen column-traversal candidate reached 1.295× the fixed native baseline and 1.616× buffered NumPy on 12 evaluation configurations; polynomial-exponential variants lost despite passing tolerance checks.

- **005 — Experiment:** Keep the candidate pool fixed and validate two earlier layout gains across 50 fresh CPU shape/stride/padding configurations.
- **Result — neighborhood confirmed:** The small strided/transposed comparison averaged 3.658× its declared reference; the large column-major comparison averaged 1.112× and depended on padding. These are dispatch/workload gains, not newly generated kernels.

- **004 — Experiment:** Add 14 CPU RMSNorm/residual variants and compare an expanded pool with the prior pool and frozen layout default across 50 cases.
- **Result — narrow discovery:** Most gain recovered already-available choices; one new kernel repeated about 1.196× the previous best on a column-major case. New variants added only 0.43% to the full-batch pool average.

- **003 — Experiment:** Compare layout lookup, handwritten rules, learned selection, and guarded measured search across 50 CPU layouts/configurations.
- **Result — prefer simple dispatch:** Layout lookup beat a global kernel by 1.165×; guarded model search added only 1.005× over lookup and required substantial reuse to repay tuning. No broad learned-selector advantage was established.

- **002 — Experiment:** Compare 12 compiled RMSNorm/residual variants with NumPy/NumExpr and a learned selector across 20 CPU workloads.
- **Result — fusion helps, selector loses:** The fixed compiled implementation reached 2.700× the library envelope; the learned selector was 0.954× that compiled baseline and measured search added little. These were tolerance-based CPU operator tests.

- **001 — Experiment:** Compare mathematical matrix-chain planning, a convex-fitted cost model, random search, and exhaustive search across 20 CPU workloads.
- **Result — retain mathematical baseline:** Minimum-operation planning beat naive left association by 1.559×; model-guided search added essentially no gain over that baseline. Established algebraic planning was stronger than the learned policy here.

## Supplemental studies

- **BATCH-VALIDATION — Experiment:** Freeze the 016 batching policy and test 12 new cases / 104 requests, including 720 teacher-forced checks; local report: `research/results/reports/RESULTS_BATCH_VALIDATION.md`.
- **Result — stronger finite evidence:** All tested probability/KV bytes matched; throughput ranged from serial parity to 2.40× serial, while first-token latency could worsen sharply. This is not independent production-traffic evaluation.

- **FAST-MODEL — Experiment:** Compare 11 model/runtime/precision configurations and then test ten MLX optimization configurations; local source: `research/experiments/fast_model/`.
- **Result — stronger stock baseline:** Stock MLX FP32 outperformed our Metal path; reduced precision changed some quality outcomes. Decode-only QKV packing added about 0.7% beyond stock MLX with tested equality; broad packing/fusion variants lost or failed gates.

- **REUSE-AND-CERTIFICATES — Experiment:** Test shared-prefix reuse, speculative generation, and conditionally certified CPU solves; local source: `research/experiments/contract_speedups/`.
- **Result — baseline-sensitive gains:** Aligned reuse reached 2.43× uncached generation, but later testing found only about 0.9% beyond native prefix caching; speculation lost. CPU refinement reached 1.76–1.77× fresh checked FP64 solves but lost when factors were cached.

- **UPSTREAM-COMPATIBILITY — Experiment:** Try the pinned unmodified AutoKernel and KernelAgent GPU entry points on the Apple M2; local source: `research/experiments/upstream_native_check/`.
- **Result — unsupported target:** MPS itself worked, but upstream CUDA/XPU requirements blocked native execution; a Metal adaptation changes the comparison scope. No upstream optimizer performance comparison was run.

## Adding an entry

Append an experiment/result pair under a stable new ID. State the hypothesis and workload in the experiment line; state the decision, comparator, correctness mode, principal result, and useful failure lesson in the result line. Record pending work as pending rather than inventing a result. Use a follow-up ID for new evidence that changes an earlier conclusion, and point back to the earlier entry.

Keep large runs and scratch work under ignored `research/`. Any evidence required to reproduce a public product claim must be separately curated into public fixtures, source, or a versioned evidence bundle. A summary alone is insufficient to reproduce or certify a speedup.

## Product extraction

- **Alpha 0.1.0a1 — Experiment:** Extract the frozen native optimizer and retained 017–023 MLX recipe into an independent package; test wheel installation, provider boundaries, contracts, proofs, export/load, fallback, and the original shrinking-batch fixtures.
- **Result — scoped retention:** Local CPU/Metal checks and seven Lean modules passed; installed MLX reached 1.945× stock serial on the evaluation fixture with exact sampled tokens/probabilities/KV, close to the prior 1.96× result. Full campaign replay, broader model/device qualification, and fresh multi-GiB proof setup remain separate work; see `lossless-product/docs/retention.md`.
