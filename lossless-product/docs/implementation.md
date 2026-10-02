# Lossless implementation plan

Build the product as an independently installable project inside `lossless-product`. Extract the research components behind explicit interfaces while retaining the chosen research gains. The first delivery needs a complete optimization workflow, bring-your-own LLM support, and deployment of the selected exact recipes with their checks and reference fallback.

This is a proposed implementation sequence. Only design documentation, illustrative contract/job configurations, and repository hygiene files exist in this folder today.

## Preserve the gains before simplifying the architecture

Use this inventory to decide the initial adapter set. All percentages below describe the cited research workload and comparator, not a future product guarantee. The MLX rows have overlapping dependencies and must not be added or multiplied together.

| Capability to retain | Recorded evidence | Dependency and extraction decision |
| --- | --- | --- |
| Restricted exact MLX batching plus the retained scheduling/cache recipe | The later retained recipe measured 1.96× stock serial on fresh short-prefix bulk and 2.38× on equal-count bulk; first-token latency remained worse than serial | MLX on Metal, pinned model/runtime; preserve the complete composed recipe as the main extraction baseline. `RESULTS_023.md` |
| Lean-guided cohort scheduling | 13.8–14.6% over the previous exact batching policy on heterogeneous-count cases | Part of the MLX recipe; keep scheduling witnesses and original request-order restoration. `RESULTS_017.md` |
| Remove unobserved terminal projections | 7.3% over the preceding recipe on a one-token stress case, 8.2% on replay | Part of the MLX recipe; retain final KV updates and group-liveness guards. `RESULTS_018.md` |
| Cache layout, metadata specialization, short-cache copy | 17.7% over the preceding recipe on a constructed short-prefix workload, 17.3% on replay | Part of the MLX recipe; retain exact state checks, generated indexing expressions, compact state export, and supported length guards. `RESULTS_019.md` |
| Allocator-cache policy | 4.6–5.0% over the previous recipe on fresh/replayed short-prefix bulk | Part of the retained recipe; preserve the memory tradeoff and workload scope. `RESULTS_023.md` |
| Exact graph reuse finalists | Roughly 2.1–2.3% extra warmed speed against the retained recipe on fresh/replayed shrinking batches | Qualified MLX options; retain as candidates while adding bounded caches and startup/amortization rules. `RESULTS_027.md` |
| Earlier exact PyTorch/MPS path | 1.20–1.33× the pinned original PyTorch 2.14 MPS path across tested model/workload cases | Requires a separate PyTorch/MPS adapter; keep in the launch-scope decision. Later tolerance-based GEMV gains cannot inherit its bitwise claim. `RESULTS_013.md` and `RESULTS_014.md` |
| Structured CPU solves with conditional error bounds | 1.61–1.99× cached checked LAPACK on the tested 512-dimensional, sixteen-RHS case across recorded runs/replay | Separate CPU adapter and certified numerical contract; preserve as distinct from bitwise inference. `RESULTS_016.md` |

Most of the recent exact inference gains therefore fit inside one MLX execution adapter with several transformation rules. Supporting the other retained families requires more adapters, not separate optimizer cores. The number of backends should follow this inventory.

For every extracted recipe, compare three complete paths under the same contract and environment: original reference, frozen research implementation, and product implementation. This reveals whether abstraction, dispatch, serialization, copying, or dependency changes consume the gain. Measure startup, warmed execution, memory, and fallbacks separately, with fresh inputs and replay controls.

Record `reference_time / product_time` as the retained speedup and `product_time / research_time - 1` as extraction overhead. Set workload-specific retention tolerances before measuring. Small incremental gains need especially low overhead; a generic 2% overhead allowance could erase the entire graph-reuse improvement. Keep the previous qualified recipe available when a newer one fails this gate. Recipe composition must be revalidated and measured; individual passing transforms do not imply a passing or faster composition.

## Existing evidence and product implications

The following paths are relative to the local `research/` workspace at the outer repository root. They are extraction references, not product runtime dependencies. Its plan/report compatibility paths remain available after cleanup. This table should become a self-contained provenance record before the product is published separately.

| Existing material | What it supplies | Product work needed |
| --- | --- | --- |
| `kernel_lab/engine.py`, `worker.py`, `common.py` | Frozen jobs, subprocess execution, budgets, resume, candidate evaluation, sealing, raw measurements | Separate contracts from operator implementations; consolidate schemas and error types; add deployment lifecycle |
| `kernel_lab/operators.py` | CPU softmax and RMSNorm/residual contracts, fixtures, baselines | Turn into explicit adapters; preserve their numerical scope; do not relabel them bitwise |
| `kernel_lab/hardware.py`, `metal_info.m`, `cuda_info.py` | Hardware inventory with explicit unknowns | Version capability records; distinguish detection from supported execution |
| `kernel_lab/providers.py` | Proposal context, parsing, receipts, model-independent command transport | Add the simple callback API and optional transports; one whole-job budget; redaction and retries |
| `kernel_lab/tensor_gate.py` | Independent finite-case PyTorch comparison | Generalize observation/state interfaces; retain framework-specific checks behind adapters |
| `kernel_lab/upstream.py` | Source admission, hashes, origin and notices | Use within artifact provenance and dependency review |
| `experiments/theory_guided_017` through `memory_bounds_020` | Scheduling, liveness, layout, and movement proofs plus exactness checks | Package individual transformation rules with preconditions; name the unproved backend boundaries |
| `experiments/layout_equivalence_019/indexing.py` | Shared address expression rendered to Lean and Metal | Make expression identity, integer ranges, and generated-source linkage explicit |
| `experiments/memory_bounds_020/api.py` and `release_checks.py` | Text API, identity admission, stopping fallback, relocated-source checks | Replace inherited experiment imports; define session ownership and production interfaces |
| `experiments/profile_guided_023` | Retained MLX recipe and scoped evidence | Extract only supported behavior and remeasure against the original reference |
| `experiments/execution_reuse_026` and `execution_reuse_027` | Qualified warmed graph-capture recipes | Bound compilation caches, invalidate changed weights, measure startup/amortization before default activation |
| `tests/` | Failure, mutation, resume, holdout and state regression coverage | Port behavioral tests with each extracted component; remove campaign-number coupling |

The research controller's `doctor` explicitly reports CPU native C and two supported operators. Its existing model-provider transport is not yet a user-friendly generic optimizer API. GPU inventory does not provide a CUDA evaluator.

Campaign 027 records approximately 2.1–2.3% extra warmed speed on fresh/replayed shrinking batches against the previously accepted recipe, on a pinned Apple M2/model/runtime configuration. It also records greater first-capture cost and retained position-specific compiled functions. Those results qualify candidates for extraction; they do not establish a product-wide speedup or justify enabling them for arbitrary workloads. Source: `RESULTS_027.md` and `experiments/execution_reuse_027/qualified_recipes.json`.

The subsequent explicit-graph campaign 028 did not earn adoption; graph planning greatly increased first capture. Numerical campaigns 029–030 found no confirmed new FP32 model speedup, and their structured CPU opportunities include important conditioning/cancellation failures. They do not supersede the exact recipe or inherit its byte-equality contract. The outer repository's `EXPERIMENTS.md` records these outcomes alongside the earlier studies.

Earlier notes in `KEY.md` already request a general optimization core, a simple bring-your-own-LLM function, explicit correctness policies, prevalidated recipes, fresh evaluation, and reference fallback. The design preserves those requirements. It proposes revisiting the earlier numerical default and integration order around the user's priority of retaining the research gains, rather than treating those choices as already agreed.

No exported conversation transcripts were found in the inspected project files. Architecture notes and reports provide the available saved context.

## Proposed source layout

This is a future layout, not a list of implemented modules. Create modules as their responsibilities are implemented; avoid empty scaffolding for distant features.

```text
lossless-product/
  pyproject.toml
  README.md
  LICENSE
  NOTICE
  src/lossless/
    api.py
    cli.py
    contracts.py
    jobs.py
    search.py
    selection.py
    evidence.py
    artifacts.py
    hardware/
    adapters/
      mlx_metal/
      torch_mps/          # include if its exact recipe is in launch scope
      native_cpu/         # demo and/or selected native workloads
    proofs/
    providers/
    runtime/
    reporting/
  proofs/                 # Lean source and pinned toolchain for supported rules
  examples/               # small inputs and supported integration examples
  tests/                  # unit, integration, packaging, target-specific checks
  docs/
  .github/workflows/
```

Use a Python `src` layout so imports from the checkout do not accidentally conceal missing package contents. Put runtime-required fixtures and recipes into declared package resources. Keep Lean tooling and framework integrations optional. The final package/distribution name must be checked before publication; the proposed `lossless` command is a design name, not a registry claim.

Runtime code must not import `experiments`, `kernel_lab`, or anything through the parent research directory. Replace hard-coded local paths with explicit configuration. Extract through a reviewed allowlist instead of copying the workspace wholesale. Retain required third-party notices with the extracted source.

## Delivery sequence

| Stage | Deliverable | Completion criterion |
| --- | --- | --- |
| 1. Scope and retention baseline | Agreed gain inventory, supported inputs, contracts, launch adapter set, reference fixtures | We know which gains must survive and how to measure retention |
| 2. Shared optimization workflow | Job/evidence schemas, CLI/library, budgets, resume, simple LLM callback and configured transport, sealed selection | A real user-supplied model proposes changes through the independent evaluator; the no-key demo also works |
| 3. Extract selected recipes | MLX scheduling, liveness, layout, allocator policy, associated proofs; other adapters required by launch scope | Correctness checks and performance retention pass against both stock and the frozen research path |
| 4. Reusable deployment | Export/load, guards, reference fallback, bounded caches, explicit state ownership | A separate consumer runs the artifact without research imports; invalidation and fallback work |
| 5. Working alpha release | Independent clean install, examples, reports, proof/evidence checks, license/notices, CI and hardware validation | The complete promised workflow and retained gains reproduce within their declared support matrix |
| 6. Expand support | New models, domains, accelerators, and native-language integrations | Each addition passes the same contract, retention, and deployment gates |

Stages 2–4 are internal milestones toward the first working alpha. A product advertised as combining formal reasoning with measurement needs actual checked obligations connected to its transformations, not just a placeholder proof interface. The first-use exact CPU demonstration may require a small new copy/layout example; the existing softmax/RMSNorm work supplies numerical examples and cannot stand in for it.

Bring-your-own LLM support belongs in the initial alpha; a model subscription is still unnecessary for the demo or reuse of existing artifacts. Recommend strong reasoning-capable models while measuring accepted gains and total search cost. Broader generated-code execution must have an appropriate worker boundary. Multiple adapters can be part of the initial release if they are needed to retain the agreed gains.

Before the alpha, exercise the same job, proposal, evaluation, export, and reload interfaces with a small non-LLM tensor workload. Prefer a vision subgraph on a selected GPU adapter if feasible; a native numerical example can establish the common interface earlier. Neither example establishes broad vision support. Keep tokenizer, sampling, and KV-cache rules inside the relevant workload adapter. Add end-to-end vision benchmarks before advertising vision speedups, including preprocessing, transfers, and postprocessing when those are inside the declared application boundary.

## Acceptance tests that matter

- Contract and source identities are immutable during a search; modified artifacts are rejected.
- Incorrect results, missing state, invalid proofs, compile failures, timeouts, and out-of-budget jobs cannot be promoted. A failed reference stops evaluation.
- Repeating the reference under controlled state establishes the determinism needed for an exact contract; unsupported nondeterminism is diagnosed rather than silently tolerated.
- A sealed candidate is evaluated on fresh data. A rejected finalist cannot be replaced using the same exposed holdout.
- No-win, unsupported-shape, changed-weight, changed-runtime, and missing-dependency paths give actionable outcomes and preserve the reference where it can execute.
- Stateful validation covers reset, mutation, continuation, exceptions, and ownership. A failed candidate cannot corrupt a later reference attempt.
- Timings include the deployment wrapper and required synchronization; cold start, memory, and fallback overhead are visible.
- An isolated wheel install includes its fixtures, recipes, native source resources, and notices. Tests run outside the research checkout.
- A non-LLM workload completes optimization and deployment through the shared interfaces without token-generation fields or dummy model state.
- CPU CI covers the supported OS/compiler matrix. Each advertised accelerator combination has actual hardware validation, with fresh installation and independent reproduction recorded.

Documentation-only planning needs link and consistency checks, not new computation experiments. During implementation, estimate each substantial run before starting it. Follow the workspace preference: runs expected to exceed about five minutes are foreground Terminal commands for the user unless the user explicitly asks Codex to execute them. Supply unbuffered, timestamped progress and `tee` logs with result/log paths.

## GitHub release boundary

The public folder should contain product source, small examples, focused tests, proof source, installation instructions, API/contract documentation, CI, and selected reproducible evidence. Keep the long experiment chronology in the research workspace or a separately published research archive.

Release preparation still needs an original-code license decision, retained third-party notices, a tested install command, an honest support matrix, and a documented contribution workflow. A working-alpha README should lead with a real quickstart and show the exact reference and contract for any performance claim. Preserve enough compact evidence to reproduce a claim; a clean repository should not erase failures or provenance.

No model weights, virtual environments, local credentials, large generated runs, or experimental build products belong in the source release. Dependencies should be declared and installed reproducibly instead of copied from local environments. Publish reviewed evidence bundles separately when raw measurements are too large for the source repository.

No Git repository was initialized and nothing was published during this planning step.
