# Executable research integration

The positive research mechanisms are now available in the source package as
scoped candidates, reusable APIs, or existing retained adapters. The catalog
contains 30 mechanism entries, including experimental leads and search tools.
It covers numbered campaigns 001–049 and the application, CPU-100, vision,
certificate and solver studies. Superseded implementations and negative results
remain identified in the [machine-readable catalog](../src/lossless/retained/catalog.json).
These are mechanisms, not 30 independent speedups or 30 unconditional defaults.

The automatic application composition currently recognizes the pinned
**pybaselines 1.2.1** source and its independent-spectrum wrapper. Other projects
can use the generic APIs and the exact-recipe catalog supplied to application
authors; they still need a callable, declared inputs and observations, applicable
source changes, and fresh validation. There is no reliable arbitrary-repository
speed multiplier. Numerical recipes require an explicit numerical contract;
their availability does not relax exact application search.

## Fresh local product result

On 11 October 2026, the public application-search API tried eight compositions
on a copied pybaselines project, without a provider or any model calls. The
reference already reused `Baseline` instances. Discovery selected combined
ModPoly scratch/dispatch edits and polynomial metadata reuse. Only that frozen
winner opened the three new final profiles, with six paired rounds per profile.

| Final profile, three changing-input calls | Reference median sequence | Candidate median sequence | Paired geometric speedup |
| --- | ---: | ---: | ---: |
| 467-point single spectrum | 0.636 ms | 0.459 ms | 1.380× |
| 1,879-point weighted batch of six | 5.680 ms | 4.618 ms | 1.230× |
| 887-point shuffled batch of four, changing orders and branch options | 3.257 ms | 2.561 ms | 1.345× |

The equal-profile geometric speedup was **1.31695×**, approximately **24.1% less
complete-call time**. All output trees, post-call inputs, and retained outputs
matched exactly. Every final round exceeded the 1.02× threshold; the one-sided
sign-test p-value was 0.015625. Startup, peak RSS and traced allocation gates
passed. This is finite workload evidence, not a universal equivalence proof.

The sequences include ordinary validation, allocation and changing signals.
Imports/factory setup are reported separately: roughly 0.47 seconds dominated
these short sequences, so **setup plus three calls was approximately parity**.
This is a reusable-process result, not a 1.32× process-launch claim. Ratios are
paired geometric means, so they need not equal the ratio of displayed medians.

The extra compiled/cache/pool compositions did not improve the winner under
these small and mixed workloads. One compiled four-worker attempt encountered a
process-cleanup permission error; a separate exact conformance replay passed
both discovery cases. That replay did not reopen final selection or qualify a
new performance winner. The initial version-admission and process-monitoring
failures are retained alongside the successful run in the
[curated evidence](evidence/research-integration.json.gz).

## What is available

Inspect the full catalog using `lossless.recipes.catalog()`, optionally filtering
by `contract` and `backend`. New runtime code lives under `lossless.retained` and
does not import the research directory. [Provenance](../src/lossless/retained/PROVENANCE.json)
records original sources and product adaptations.

| Mechanisms | Public entry | Admission and composition |
| --- | --- | --- |
| RMS layout/unrolling and softmax traversal/packing | `recipes.native_candidates` | Numerical native search; all proposals independently evaluated. Large-Fortran 046 is Arm64 scoped. Experimental 033/039 is available but failed historical broad promotion. |
| ModPoly source, compiled loop, bounded metadata and worker combinations | `applications.retained_plan`; CLI `search-application --retained` | Source hashes must match. Eight candidates for the recognized wrapper; only applicable combinations compete. Compilation preserves the loaded NumPy BLAS and checks its ABI/layout. |
| Independent batches | `execution.IndependentMap` | Picklable batch factory, ordered independent records, persistent 2/4 spawned workers, one BLAS thread each; explicit close/context manager and a serial crossover. |
| Prepared metadata and ordinary bounded reuse | `retained.pybaselines.ModPolyPlan`, `ModPolyCache`; `execution.BoundedPlans` | Owned grids, changing signals/weights, bounded entry/payload retention, mutation and eviction checks. Explicit preparation is a different measured lifecycle. |
| Native call overhead | `NativeOperation.optimized_calls()` | Opt-in domain scanner, direct binding and bounded immutable constants; setup compiles locally. Preserves the underlying kernel's exact or numerical contract. Default artifact loading is unchanged. |
| PyTorch CPU RMS/MLP regions and pretrained decoder | `retained.torch_cpu.regions`, `legacy_decoder` | Frozen CPU FP32 models. Exact ATen or numerical reduction alternatives; legacy decoder pins Torch 2.2.2 / Transformers 4.43.3 and one-token no-cache calls. |
| ONNX RMS/residual region | `retained.onnx.rewrite`, `build` | Numerical named graph/axis/epsilon/dtype matcher; caller supplies ORT C headers. Full graph must be revalidated. |
| Current-runtime Metal model kernels | `retained.mps.candidate` | Separate exact 013 and numerical 014 options; Torch 2.14 / Transformers 4.43.3, studied geometries and immutable parameters. Old 012 implementation is superseded. |
| Existing exact MLX scheduling, liveness, movement and graphs | `lossless.optimize`, `mlx.fixed_count` | Existing pinned model/runtime/device and artifact gates remain in effect; graph variants are alternatives. |
| Experimental whole-request MLX source | `retained.mlx_full_request.load_candidate` | Explicit local pinned model, fixed-count token API, exclusive execution, bounded compiled state; 048 failed historical tail/TTFT gates and is not a default. |
| Qualified MLX vision operations | `retained.mlx_vision.candidate` | Numerical 049 max-pooling and convolution/average-pool/sigmoid/sum task scopes only; unsupported inputs use the supplied reference. |
| CPU SigLIP constant reuse | `retained.vision.SiglipLoss` | Exact loss/first-gradient path for square CPU FP32/64 logits; bounded signs with separate inference/autograd caches. No complete training claim. |
| Structured CPU algebra | `retained.numerical` | Explicit numerical matrix-chain planning, `I+UU.T` plans, matrix-free ridge steps and periodic deblurring; structure must be declared, never guessed. Residual guards/fallback for solves. |
| Conditioned least squares | `retained.numerical.RefinedLeastSquaresPlan` | New guarded adaptation of the 030 lead; conditioning checks/refinement and FP64 QR fallback. Fresh speed qualification still required. |
| Conditional certified solves | `retained.certified.CertifiedDensePlan`, `CertifiedPathPlan` | Owned dyadic strictly diagonally dominant SPD systems; checked residual enclosure, fail closed. Certificate assumes the documented rounding model, not a verified BLAS. |
| Reversible BF16 storage | `retained.bf16_codec.encode/decode` | Raw uint16 round trip, including NaN payloads and signed zero; no inference speed claim. |
| Byte pipelines and additive plan selection | `retained.uint8.Uint8Pipeline`; `retained.planning.dynamic_program` | Explicit integer operations; exact native/LUT fusion. DP optimality is limited to the supplied finite additive cost model. |
| Proof filtering and timing races | `retained.copy_proof.candidate`; `retained.racing.BoundedRace` | Optional finite copy-index proofs and experimental discovery-only rejection. Neither grants correctness or final performance acceptance. Racing assumes independent stable timing scores. |

No new task-quality model recipe was qualified by the audited campaigns. Failed
model rewrites, bicubic rewrites, unsupported CUDA claims, unrestricted normal
equations, and the negative CPU-100 compiler-feedback family are not promoted.
Smaller positive observations superseded by stronger retained mechanisms remain
in the research reports; they are not extra gains to multiply together.

## Run the local application integration

From the repository root, install the application extra in your environment and
use an existing checkout at revision
`6f8d927e08577a9c5af5f34129b14d57c12725f6`. The script copies it, generates fresh
fixtures, freezes discovery/final profiles and runs local composition. Expected
runtime on the tested M2 is about two minutes, with a 280-second cap.

```sh
python -m pip install -e './lossless-product[applications]'
set -o pipefail
OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 python -u \
  lossless-product/scripts/check_retained_application.py \
  --project /path/to/pybaselines-v1.2.1 \
  --output ./lossless-runs/retained-pybaselines --budget 280 \
  2>&1 | tee ./retained-pybaselines.log
```

For an already configured, recognized project:

```sh
lossless search-application job/lossless.json --retained --budget 280s \
  --output ./lossless-runs/retained-search
```

`retained_plan` also returns the concrete JSON-compatible proposals for review.
The evaluator retains source patches, native build receipts, per-call samples,
ownership fingerprints, resource checks and failures. It never changes the
input checkout. An accepted application experiment supplies the selected patch
and candidate project; generic `lossless export` still supports only the native
and MLX artifact adapters, not arbitrary application source experiments.

## Validation and packaging

A separate publication checkout subsequently passed the full 193-test run in
110.4 seconds: 189 passed, four optional checks skipped, no failures or errors.
[Publication check receipt and runtime source hashes](evidence/integration-publication-checks.json).

The local regression checks exercised 193 product tests: 189 passed after
focused repairs/replays, with four optional checks skipped. The selected source
also passed 292 upstream polynomial tests, with eight skips. Four earlier
deadline-sensitive cases failed during host suspension/clock jumps and passed
an unchanged-limit replay. Expanding the softmax pool initially changed an old
research replay's seed count; historical no-case requests now retain their
original pool, while declared product searches receive the additional candidate.
The corrected replay, native pool and product workflow checks passed. Failed
records remain in the evidence; no performance gate was relaxed.

Coverage includes the expanded native pool, proofs, memory of spawned workers,
prepared plans, owned buffers, raw-bit codecs, numerical oracles and local
composition. These tests are separate from the successful application
performance run.

Backend smoke scripts under `scripts/check_retained_*.py` accept explicit output
directories and optional local model/header paths. CPU PyTorch checks covered
eight FX calls and two complete pretrained forward calls; ONNX covered nine
full graph calls plus three invalid-epsilon controls; Metal covered four
prefills plus four cached continuations; MLX vision covered six full operation
calls; experimental MLX request checks covered single/ragged requests and three
full-state continuations. These validate sampled functionality, not fresh
historical speedups. No script downloads models or calls an authoring service.

Wheel/source archive inspection and an outside-checkout wheel installation
passed checks of the new resources, base APIs, demo and export/load. Three
additional installed-wheel pybaselines preparation/composition tests also passed
using the existing optional dependencies. Optional model runtimes remain explicit
dependencies; the base package still needs only NumPy. Compiler outputs, model
weights, private research files and environments are not distribution contents.
The published `v0.1.0a1` release binaries have not been replaced.
