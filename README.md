# Lossless

**Optimize ML computations under an explicit correctness contract.**

> An optimizer that combines formal reasoning, hardware information, and measured experiments to accelerate computations while preserving an explicit correctness contract.

Bring a supported workload, a JSON contract, and your own LLM. Lossless searches for faster implementations, checks them independently, and exports a guarded implementation with performance and correctness records. Built-in candidates let you try the workflow without an API key.

**Status: GitHub prerelease (`0.1.0a2`), MIT-licensed original code, public source repository.** The long-term scope includes GPU optimization for language models, vision, and linear algebra. The supported workloads below define what runs today; CUDA execution, arbitrary model import, and task-quality acceptance remain future work.

![Recorded installed-alpha MLX comparison: 1.945× throughput, with batch time falling from 574.8 to 295.4 ms and exact agreement on tested tokens, log probabilities and active KV state.](lossless-product/docs/assets/mlx-retention.png)

One recorded evaluation batch on the pinned Apple M2 setup, using the retained recipe. Bars show throughput normalized to stock serial MLX; timing labels show **median batch completion time**. It does not measure live-LLM search gains or establish performance on other models/devices. [Measurements and reproduction scope](lossless-product/docs/retention.md#readme-performance-figure) · [Source data](lossless-product/docs/assets/mlx-retention-evaluation.json).

## Quickstart

Requires **Python 3.10+ and clang** on macOS or Linux. Python dependencies install automatically; the CPU demo needs no model weights, GPU, LLM key, or proof toolchain.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install https://github.com/aarnav9/lossless-opt/releases/download/v0.1.0a2/lossless_opt-0.1.0a2-py3-none-any.whl

lossless doctor
lossless demo --budget 60s --output ./lossless-runs/demo
lossless report ./lossless-runs/demo
lossless export ./lossless-runs/demo --output ./lossless-runs/copy-artifact
```

The commands above install the fixed [v0.1.0a2 release](https://github.com/aarnav9/lossless-opt/releases/tag/v0.1.0a2). For CPU application support, install the wheel with its `applications` extra; see the [CPU quickstart](lossless-product/docs/release-0.1.0a2.md). PyPI and Homebrew distribution remain future work. Source installation: `python -m pip install "./lossless-product[applications]"`.

The demo returns `accepted`, `reference_retained`, or `incomplete`, with `report.json` and `report.html` in the run directory. Completed runs export the accepted implementation or the reference; incomplete runs cannot be exported. A measured speedup is not guaranteed.

## Profile before searching

The current `main` branch selects native candidates using complete public calls by
default, including guards, allocations and dispatch. Application searches also
gate regressions at each declared call position and can include startup costs.
[Benchmark policy and public suite choices](lossless-product/docs/benchmarking.md).

**Local CPU Python/NumPy projects:**
`lossless run-project PATH --entry app.py:run --cases cases.json --output RUN`
profiles representative inputs, tests applicable local recipes under exact checks,
and produces a runnable project copy with a reference fallback. No model calls.
Use `--input-loader examples.py:cases --deployment auto --expected-requests 100`
to supply live examples and select using complete calls, including startup.
Use the result through `CallableApplication(bundle)` as a context-managed callable.
[CPU guide](lossless-product/docs/cpu-projects.md) · [Runnable example](lossless-product/examples/cpu-callable/README.md).
The separate [intake/replay](lossless-product/docs/applications.md),
[optional Codex assessment](lossless-product/docs/application-profiling.md), and
[`search-application`](lossless-product/docs/application-search.md) commands remain available.

Application search includes a [sourced CPU investigation checklist](lossless-product/docs/cpu-investigation.md):
detected hardware, discovery diagnostics, 22 areas with interactions and correctness
obligations, and a report of reviewed, evaluated and unfinished ideas. Coverage
does not establish proof or acceptance; unavailable hardware counters stay unknown.
The first [matched author pilot](lossless-product/docs/cpu-checklist-study.md)
favored the previous author context and led to a fix for repeated prompt history.

```sh
lossless init --template softmax --output ./softmax-demo
lossless profile ./softmax-demo/lossless.json --budget 30s --output ./lossless-runs/profile
```

The HTML/JSON reports show first-call and warm timing, memory, separate host hotspots, and MLX latency/stages where applicable. Profiling uses discovery inputs and makes no LLM calls. Pass `--artifact PATH` to compare an exported implementation. [Profiling guide](lossless-product/docs/profiling.md) · [Release support matrix](lossless-product/docs/release-0.1.0a2.md).

The current `main` branch includes frozen [MLX deployment limits](lossless-product/docs/alpha.md#mlx-deployment-limits-unreleased). [Broader qualification](lossless-product/docs/qualification-041-043.md) includes natural/code/multilingual workloads and a contract-checked stock batching control; the original roughly 2× gain is workload-specific. Historical results keep their recorded hardware and runtime scope.

## CPU application experiment

**Faster spectrum processing with exact tested outputs, on an Apple M2 CPU.**
Live GPT-6 Astra source search optimized polynomial setup and temporary-buffer
reuse in a spectrum-processing application built around pybaselines 1.2.1.
One frozen candidate was tested on eleven new workload profiles, with one
OpenBLAS thread on both sides:

| Comparison | Request speedup | Less request time |
| --- | ---: | ---: |
| Stock public calls | **1.757×** | **43.1%** |
| Stock reusable `Baseline` objects | **1.282×** | **22.0%** |
| Reusable objects + previous optimization | **1.152×** | **13.2%** |

Returned arrays, weights, coefficients and complete convergence histories matched
bit-for-bit; inputs and retained earlier outputs also matched. Both reusable-object
comparisons passed every speed, setup and memory gate. Against the strongest
baseline, traced allocation was equal or lower on every profile.

**This remains experimental.** The overall study retained the reference because
two profiles exceeded the separate allocation limits against public calls, largely
reflecting the existing reuse lifecycle. These are complete in-memory request
timings on a synthetic application around a real library; import/setup is measured
separately. They do not establish exactness or speedups for arbitrary code. Search
used two sessions with intervening feedback, so this is not a controlled timer
comparison. [Results, costs and model-free reproduction](lossless-product/docs/application-continuation-study.md)
· [Raw evidence](lossless-product/docs/evidence/application-continuation-pybaselines.json.gz)
· [Frozen source proposal](lossless-product/experiments/application_intake/continuation-candidate.json).

## Research integrated into the source package

Thirty retained mechanism entries now expose scoped CPU/application, model,
vision, numerical and search candidates. Local composition on pinned pybaselines
measured **1.317× complete-call speed** with exact output/input/ownership checks
and passing startup/memory gates. Setup plus three short calls was near parity;
extra compiled/pool combinations did not improve the selected source edits.
[Integration coverage, evidence and reproduction](lossless-product/docs/research-integration.md).
These are source-package improvements; the existing prerelease assets are unchanged.

## How it works

```mermaid
flowchart LR
    J["Workload + JSON contract + budget"] --> H["Resolve and freeze the job"]
    D["Detected hardware"] --> H
    H --> C["Built-in or LLM-proposed candidate"]
    C --> V["Build, check and measure discovery cases"]
    V -->|Feedback within budget| C
    V --> S["Freeze selection and evaluate separate cases"]
    S --> R["Report: accepted, reference retained, or incomplete"]
    R -->|Completed run| A["Export guarded implementation"]
```

The harness controls the reference, contract, evaluation, and acceptance decision. Your connected LLM receives relevant hardware facts, formal context, and discovery feedback. It proposes changes within the adapter's supported scope. Correctness requirements stay fixed throughout the search.

Optimization happens offline. Your application later loads the exported implementation; running it does not require the search LLM. Target and input guards determine where a replacement can run and when the reference is used.

## Supported today

| Adapter | Workload and target | Contract and scope |
| --- | --- | --- |
| `native.copy` | Float32 tensor copy; CPU | Exact bit patterns, including NaN payloads and signed zero |
| `native.softmax` | Row softmax; CPU | Fixed numerical tolerances against an independent float64 oracle |
| `native.rmsnorm_residual` | RMSNorm plus residual; CPU | Fixed numerical tolerances; unchanged precision |
| `mlx.fixed_count` | Complete fixed-count greedy inference requests; Apple M2 | Exact validation of tokens, emitted log probabilities, active KV state, and request order; pinned runtime and model artifact |

Native adapters currently use generated matrix inputs and accept C candidate source. The MLX adapter retains researched scheduling, dead-projection, cache-movement, and allocator optimizations. Its LLM proposal space currently selects allocator policies. MLX requires the pinned runtime and exact local model artifact described in the [alpha guide](lossless-product/docs/alpha.md#mlx-model-optimization); weights are not bundled.

The CPU suite and isolated package-install checks run on Linux and macOS. GPU validation currently covers the recorded M2 setup. [Retention and validation records](lossless-product/docs/retention.md) explain which research results were extracted, what was tested, and what remains outstanding.

## Configure a workload and connect your LLM

```sh
lossless init --template softmax --output ./my-job
lossless inspect ./my-job/lossless.json
lossless optimize ./my-job/lossless.json --output ./lossless-runs/softmax
```

All user configuration uses **JSON only**. Workload, contract, and budget live in one job file. The contract can instead reference a reusable JSON file. Hardware is detected automatically; an optional JSON hardware profile supplies context that is checked against the worker. Schema version `1` is executable; `draft-1` examples are design proposals.

Add an `llm` section to the generated job to use your model:

```json
{
  "llm": {
    "callback": "my_llm.py:complete",
    "provider": "your-provider",
    "model": "your-model",
    "credential_env": ["MY_MODEL_API_KEY"],
    "timeout_seconds": 1800
  },
  "budget": {
    "wall_time_seconds": 3600,
    "max_candidates": 4,
    "max_llm_calls": 1
  }
}
```

Merge those fields into the generated job, keeping its workload and contract. The total budget includes model calls, compilation, checks, and measurements; each call is also limited by the remaining budget. Built-in candidates count toward `max_candidates`.

Your `complete(prompt: str)` callback returns a proposal JSON string or object. A command alternative reads one JSON request from stdin and writes one proposal batch to stdout. The request contains the contract, reference code or recipe scope, hardware context, and discovery feedback; evaluation inputs are withheld from the prompt. Lossless checks and measures every admitted proposal independently. Keep API keys in environment variables. The `provider` and `model` fields are labels; the callback or command makes the actual model connection.

### Connect Claude or Codex

| Connection | Setup |
| --- | --- |
| Claude API | Install `anthropic` in the same virtual environment, set `ANTHROPIC_API_KEY`, and supply a Python callback using your chosen model. [Minimal callback and job configuration](lossless-product/docs/alpha.md#connecting-claude-or-codex). |
| Codex CLI | Current source ships `python -m lossless.codex_provider`, using saved ChatGPT sign-in and structured proposals. [Ready-to-run job and subscription setup](lossless-product/docs/codex.md). |
| Claude Code | A user-supplied command wrapper invokes `claude -p` with a proposal JSON Schema and extracts `structured_output` from its JSON response. [Official Claude Code documentation](https://code.claude.com/docs/en/headless). |

The callback/command interface is implemented. **The Codex connector is available in current source, outside the published `v0.1.0a2` wheel.** Claude transports remain user-supplied. Codex uses your subscription allowance; API billing is separate. A live copy smoke verified the command path, evaluation and export/load, correctly retaining NumPy when the proposal did not win. [Setup, tested CLI version and costs](lossless-product/docs/codex.md) · [General connection guide](lossless-product/docs/alpha.md#connecting-claude-or-codex).

### What the LLM tests establish

- The CPU suite uses a deterministic provider that returns the reference C implementation as a control candidate. It exercises prompt delivery, compilation, correctness checks, timing, separate evaluation, and export/load.
- The fresh-environment MLX test uses a deterministic provider proposing the 64 MiB allocator policy. The real harness evaluates it; the retained 256 MiB recipe wins that run.
- Regression tests cover malformed responses, timeouts, credential echoes, and withholding evaluation cases from proposal prompts.
- The source Codex connector has a live ChatGPT-signed-in native-copy smoke; it returned an eligible proposal in 37.55 seconds and exported the retained reference. Model access probes passed for GPT-6.1 Sol, GPT-6 Astra and GPT-6 Luna.

A separate manual Codex-guided native study compared authored candidates against retained recipes and enumeration. [Campaign 043](lossless-product/docs/qualification-041-043.md#043-independent-model-authoring) adds independent live Codex CLI sessions; it tests model proposals through the native experiment harness, not a shipped vendor connector.

For a useful workload without model authoring, the [opt-in Fortran softmax recipe](lossless-product/examples/softmax-fortran/README.md) revalidates a frozen kernel locally and exports guards for 19 specific shapes. [Campaign 046](lossless-product/docs/softmax-046.md) qualified it across distributions and fresh processes, including ordinary application calls. It is a numerical CPU recipe; the broader default recipes remain unchanged.

The fixture tests establish protocol and execution behavior, not live-model proposal quality. The measured MLX retained-recipe speedup did not come from a live model call. [Validation record](lossless-product/docs/retention.md#local-verification-2-october-2026).

Capable reasoning models are a sensible starting point; measure their effect on gains and search cost. [Provider protocol and supported proposals](lossless-product/docs/alpha.md#provider-boundary).

## Correctness and formal reasoning

Research contributions can target three explicit contracts:

| Category | Requirement |
| --- | --- |
| Exact / lossless | Preserve the declared output bits, state, and behavior |
| Numerical | Stay within declared error bounds and invariants |
| Task quality | Meet a named metric and allowed change on a specified evaluation dataset |

These categories specify acceptable behavior. Relaxing a contract can admit more candidates, but does not guarantee a larger speedup. Comparing search outcomes requires the same workload, device, reference, and stated search effort.

The current runtime implements the exact and numerical contracts in the support table. Task quality is a research category pending an executable evaluator. A failed exact experiment can motivate a separate numerical or quality experiment; the existing job's contract never relaxes silently.

Bundled Lean proofs cover scoped structural transformations, and the theorem index supplies mathematical context. Lean/mathlib installation is optional and explicit. **Finite validation is not a universal proof of generated code or end-to-end floating-point equivalence.** Reports keep proof scope, validation evidence, and measured performance distinct. See [proof setup and scope](lossless-product/docs/alpha.md#optional-proof-setup).

## Tests and new architectures

The repository already includes [deterministic regression tests](lossless-product/tests), seeded native input generators, independent reference calculations, and separate discovery/evaluation inputs. Public regression cases provide reproducible checks; their presence in the repository does not make them a hidden evaluation set.

For CUDA and later backends, the planned approach combines reviewed contract tests and generators with LLM-proposed additional cases and setup code. Experiment findings can suggest cases such as irregular shapes, padding, or cache aliasing. The runner validates admitted cases, obtains expected results independently, and freezes acceptance criteria. New runtime-specific runners still require tests on their target hardware.

Automatic LLM test generation and backend bootstrapping are **not implemented** in the alpha. See the [proposed test workflow](CONTRIBUTING.md#llm-proposed-tests-and-new-backends) for the trust boundary, folder plan, and hardware qualification steps.

## Contribute an experiment

Ideas, reproductions, failed attempts, and negative results are welcome. Start with a hypothesis; you do not need a successful speedup or your own GPU. Read the [experiment ledger](EXPERIMENTS.md), then use the short [submission format](CONTRIBUTING.md#submit-an-idea-or-result).

Keep the ledger concise: one experiment line and one result line. Small reproduction scripts, generators, and fixtures supporting public claims belong in the tracked repository. Bulk runs, weights, and exploratory artifacts stay in the ignored research workspace.

The next milestone is a bounded PyTorch-on-CUDA workflow with a supported vision example, followed by a short external test on a contributor's NVIDIA GPU. Broader hardware comparisons and public performance rankings follow that first validation. [Contribution and validation plan](CONTRIBUTING.md#next-validation-milestone).

## Repository and license

```text
lossless-product/      Installable package, adapters, proofs, tests, examples and docs
EXPERIMENTS.md         Concise experiment/result memory
CONTRIBUTING.md        Experiment categories, test workflow and development setup
.github/workflows/    CPU checks and isolated package installation
research/             Local research archive; ignored by Git
```

The product runs independently of `research/`. Original code is [MIT licensed](LICENSE); imported components retain their [third-party notices](lossless-product/NOTICE). Package-registry distribution remains pending.

[Architecture and future design](lossless-product/docs/architecture.md) · [Implemented API and configuration](lossless-product/docs/alpha.md) · [Product installation and deployment](lossless-product/README.md)

The [032–038 follow-up report](lossless-product/docs/frontier-032-038.md) records the explicit acceptance comparator, bounded optional graph recipes, manual Codex search, all six frontier pilots, and measured payback.
