# Using the source alpha

This describes the source interface. The `v0.1.0a1` tag preserves the first published alpha; additions explicitly marked unreleased require installation from the current source. Files labeled `draft-1` and the broader design documents remain proposals.

## A job is deterministic setup

Configuration files use JSON exclusively. `lossless init --template copy --output my-job` writes an executable JSON configuration. The required sections are `schema_version: 1`, `workload`, `contract`, and `budget`. Unknown keys, duplicate JSON keys, nonfinite numbers, overlapping discovery/evaluation shapes, and unsupported contract changes are rejected before search.

```json
{
  "schema_version": 1,
  "workload": {
    "adapter": "native.copy",
    "cases": [
      {"id": "discover", "split": "discovery", "rows": 64, "columns": 257, "layout": "c"},
      {"id": "evaluate", "split": "evaluation", "rows": 73, "columns": 259, "layout": "c"}
    ]
  },
  "contract": {
    "preset": "exact",
    "required_evidence": "validated",
    "weight_changes": false,
    "precision_changes": false
  },
  "objective": {"metric": "latency", "min_speedup": 1.02},
  "budget": {"wall_time_seconds": 60, "max_candidates": 4, "max_llm_calls": 1},
  "target": {"device": "auto"},
  "output": {"directory": "./lossless-runs"}
}
```

The registered adapter defines the reference, input generator, accepted observables, exact relation or numeric tolerances, and evaluator. Native inputs currently use generated float32 matrices with C, Fortran, or every-other-column layouts (`c`, `f`, `slice2`). Arbitrary reference functions and model graph import are future work. `softmax` uses `atol=2e-7`, `rtol=4e-5`, nonnegative output and row-sum error at most `2e-6`; `rmsnorm_residual` uses `atol=1e-5`, `rtol=5e-5`, epsilon `1e-5`. Both use an independent float64 oracle. Their presets are `numerical`; accepting them never changes an `exact` request silently.

To split the files, replace the contract object with `{"file": "contract.json"}`. Relative paths resolve against the job file. For optional hardware context, add `"target": {"device": "auto", "profile_file": "hardware.json"}`. Generate that file with `lossless hardware --output hardware.json`. The actual worker still detects its environment; supplied data is context, not permission to ignore device guards. Hardware inventory does not make a CUDA execution backend available.

The harness resolves and freezes these inputs first. It then builds a discovery-only prompt for the user-connected LLM. There is no internal proprietary LLM. Evaluation cases stay out of that prompt and cannot be revised after observing discovery results within the same run.

## Search time and performance (unreleased CLI overrides)

**Treat optimization time as a performance tuning input: more time can help when useful proposals or evaluations are being cut short.** It increases opportunity, not guaranteed speed. Campaign 044's five-minute authors timed out, while all 30- and 60-minute authors completed. One hour did not consistently beat 30 minutes in that one-shot study. [Measured results and cost](timers-044.md).

Time limits belong to the frozen search budget/LLM configuration. The correctness contract defines which outputs are acceptable and remains unchanged. Configure these existing JSON fields, retaining the workload and contract from your job:

```json
{
  "budget": {
    "wall_time_seconds": 28800,
    "max_candidates": 24,
    "max_llm_calls": 12
  },
  "llm": {
    "callback": "provider.py:complete",
    "timeout_seconds": 1800
  }
}
```

| Input | Meaning |
| --- | --- |
| `budget.wall_time_seconds` / `--budget 8h` | Maximum total search allowance, including setup, model calls, compilation and evaluation |
| `llm.timeout_seconds` / `--llm-timeout 30m` | Maximum time for one author response; defaults to 1,800 seconds (30 minutes), clipped to remaining time after the controller reserves evaluation time |
| `budget.max_candidates` | Candidate cap, including retained recipes; at most 100 in this alpha |
| `budget.max_llm_calls` | Upper bound on author calls, not a promise to use them all |

**Current source defaults to 30 minutes per response.** Omit `llm.timeout_seconds` to use 1,800 seconds, or supply any positive finite duration in seconds. CLI `--llm-timeout` and Python `with_time_limits(llm_timeout_seconds=...)` override the configured value. Explicit limits in existing jobs remain effective; the immutable `v0.1.0a1` release retains its previous 30-second default. Resolved jobs record the effective configured timeout even when the input omitted it.

The total search budget remains separate and can cut a response off earlier: the 60-second native template does not become a 30-minute search. Allow enough total time for authoring, compilation and final evaluation. Configure any shorter SDK/CLI-wrapper timeout separately. These allowances are maximums, so a search can stop early. Native search can use feedback across multiple calls within its limits; MLX currently makes at most one allocator-policy proposal call and does not become an autonomous GPU-kernel search when given more time.

Preview CLI overrides without running a provider or changing the source JSON, then use the same overrides to execute:

```sh
lossless inspect my-job/lossless.json --budget 8h --llm-timeout 30m
set -o pipefail
python -u -m lossless optimize my-job/lossless.json --budget 8h --llm-timeout 30m --output lossless-runs/overnight 2>&1 | tee lossless-overnight.log
```

The source CLI prints flushed, timestamped progress every 30 seconds during quiet waits, including LLM authoring. The command saves console output to `lossless-overnight.log`; reports are written under the selected output directory. An LLM callback or command must be configured to use `--llm-timeout`. The source Python API supports the same copy-on-change operation:

```python
import lossless

job = lossless.Workload.from_config("my-job/lossless.json").with_time_limits(
    seconds=8 * 3600, llm_timeout_seconds=30 * 60
)
result = lossless.optimize(job, output="lossless-runs/overnight-python")
```

The original `Workload` and source JSON are unchanged. The resolved JSON records the chosen limits, which participate in job identity and artifact provenance. Resuming requires the same frozen limits; use a new run for a different budget. Budget changes never relax correctness or final evaluation. Report search cost and break-even calls alongside any speed improvement; long searches for tiny kernels can require millions of calls to repay.

## Provider boundary

Choose either a callback or a command:

```json
{"callback": "provider.py:complete", "credential_env": ["MY_MODEL_API_KEY"], "timeout_seconds": 1800}
```

```json
{"command": ["python", "provider.py"], "credential_env": ["MY_MODEL_API_KEY"], "timeout_seconds": 1800}
```

Store that object in the job's `llm` field. Named callbacks run in a fresh interpreter. Native Python callers may also pass an in-memory `llm` callable to `lossless.optimize`; that POSIX callback runs in a forked worker. MLX jobs require the named callback or command form. Command arguments are an argv list; no shell expansion occurs. Commands run from the configuration directory and receive only basic environment variables plus explicitly named credentials. Do not put secrets in the JSON files. Local callbacks are trusted Python code.

The native request includes `response_format`; return a batch such as:

```json
{
  "schema_version": 1,
  "proposals": [
    {"id": "unique_name", "hypothesis": "Why this should improve performance", "source": "complete C source using the supplied ABI", "parameters": {}}
  ],
  "usage": {"input_tokens": 1000, "output_tokens": 300, "cost_usd": null}
}
```

Usage is provider-reported, not a billing enforcement boundary. The controller limits time, model calls and candidate count. It compiles, checks, and measures each admitted source; the model's claimed correctness or speed never admits it. Provider failures are recorded by type without retaining exception text or provider stderr. Configured credential echoes are rejected. Wall-time limits include setup and model calls, with small controller/cleanup overhead possible; native jobs reserve their whole timeout before launching. If the budget cannot complete evaluation, the report says `incomplete` and export is refused.

## Connecting Claude or Codex

The current interface supports user-owned callbacks and commands. It does not include turnkey vendor connectors. The examples below explain the wiring; they have not been exercised against a live Claude/OpenAI service in the product validation runs.

### Claude API callback

Inside your activated virtual environment, install the optional provider SDK and create a CPU job:

```sh
python -m pip install anthropic
lossless init --template copy --output ./my-job
```

Set `ANTHROPIC_API_KEY` in your shell environment. Save this as `my-job/provider.py` and replace `YOUR_CLAUDE_MODEL_ID` with a model available to your account:

```python
import json
from anthropic import Anthropic


def complete(prompt: str) -> dict:
    response = Anthropic(timeout=100, max_retries=0).messages.create(
        model="YOUR_CLAUDE_MODEL_ID",
        max_tokens=8192,
        system=(
            "Return only JSON matching the response_format in the input. "
            "Follow the frozen contract. Do not include Markdown fences."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason != "end_turn":
        raise ValueError("Provider did not return a complete response")
    proposal = json.loads(
        "".join(block.text for block in response.content if block.type == "text")
    )
    proposal["usage"] = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_usd": None,
    }
    return proposal
```

This uses the [official Claude Python SDK](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python). It is a minimal text-to-JSON callback: malformed JSON, incomplete responses, and invalid proposal fields are rejected, rather than repaired automatically. A fuller connector can add provider-side schema-constrained output and bounded retries. Report unknown dollar cost as `null`; token usage is not a billing cap.

Merge these fields into `my-job/lossless.json`, preserving its existing workload and contract:

```json
{
  "llm": {
    "callback": "provider.py:complete",
    "provider": "anthropic",
    "credential_env": ["ANTHROPIC_API_KEY"],
    "timeout_seconds": 120
  },
  "budget": {
    "wall_time_seconds": 180,
    "max_candidates": 4,
    "max_llm_calls": 1
  }
}
```

This short callback example explicitly uses a 120-second Lossless limit and a 100-second SDK timeout; both can be increased for longer reasoning. The callback path resolves relative to the job JSON. The callback selects the model; `llm.provider` and `llm.model` are labels, not built-in SDK routing. The whole-job budget includes setup, provider calls, compilation, checks, and timing. Each call receives at most its configured timeout and the remaining search budget. Built-in candidates also consume candidate slots. Slow responses can exhaust the budget; a completed search can legitimately retain the reference.

```sh
lossless inspect ./my-job/lossless.json
lossless optimize ./my-job/lossless.json --output ./lossless-runs/claude-test
lossless report ./lossless-runs/claude-test
```

### Codex CLI or Claude Code

Current source includes a [ready-to-run Codex connector and job](codex.md) using saved ChatGPT sign-in: `python -m lossless.codex_provider`. Claude Code and custom CLI integrations use a user-supplied command wrapper. A wrapper reads the Lossless request from stdin, asks the CLI for a proposal using an actual JSON Schema for the active adapter, and writes only the proposal batch to stdout. The request's `response_format` illustrates the expected object; it is not itself a JSON Schema.

- **Codex CLI:** `codex exec` accepts stdin context, supports `--output-schema`, and reuses saved CLI authentication. Its ordinary stdout is the final message; `--json` produces a JSONL event stream that must be parsed before returning a proposal. [Official OpenAI documentation](https://learn.chatgpt.com/docs/non-interactive-mode).
- **Claude Code:** `claude -p` supports `--output-format json` with `--json-schema`. Extract the `structured_output` field; do not forward the enclosing session metadata as a Lossless proposal. [Official Claude Code documentation](https://code.claude.com/docs/en/headless).

Once that wrapper exists, replace the callback `llm` object with a command object such as:

```json
{
  "llm": {
    "command": ["python", "provider_command.py"],
    "timeout_seconds": 120
  }
}
```

`provider_command.py` is user-supplied; this filename is not a shipped executable. Preserve the configured job budget. Command arguments are an argv list, not a shell pipeline. Commands inherit the basic environment plus named `credential_env` entries; forward any required custom auth/configuration variables explicitly. Run proposal generation with only the inputs and permissions it needs. Lossless retains responsibility for executing and accepting candidates. Local callbacks and commands are trusted code, and withholding evaluation data from the prompt is not filesystem isolation.

Current source supplies `LOSSLESS_PROVIDER_TIMEOUT_SECONDS` to command transports
with the effective response allowance after applying the configured timeout and
remaining job budget. The bundled Codex connector honors it automatically;
custom commands may use it too. The parent still enforces that deadline.

### What has been tested

The CPU workflow regression sends a deterministic provider the real discovery context and receives the reference C implementation as a control candidate. It then exercises compilation, contract checks, timing, separate evaluation, and export/load. The installed MLX check sends a deterministic command the request and receives a 64 MiB allocator proposal; the retained 256 MiB recipe is selected. Malformed responses, timeout behavior, credential echoes, and evaluation-prompt separation have regression coverage.

Those automated checks use local fixtures. A separate live native-copy smoke now validates the source Codex connector through the public command boundary, separate evaluation and artifact export/load. Its eligible proposal did not beat NumPy and the reference was retained. [Tested CLI, receipt and setup](codex.md). No live Claude or MLX-provider quality claim follows. The measured MLX retention gain comes from the retained recipe. Earlier [campaign 043](qualification-041-043.md#043-independent-model-authoring) measured Codex authoring through a separate research harness. [Recorded validation](retention.md).

## Reports and deployment

New source native jobs default to `objective.timing_scope: "call"`: selection,
confirmation and search payback use ordinary exported calls including guards,
allocation, binding and dispatch. Use `"bound"` only for an application that
actually reuses bound buffers. Kernel-only measurements are separate diagnostics;
old campaigns retain their original scope. [Full benchmark policy](benchmarking.md).

A run directory contains the resolved job, expanded contract, hardware evidence, immutable source/harness hashes, discovery feedback, a frozen selection, evaluation measurements, model-call receipts, `report.json`, `report.html`, and logs. CPU workers have individual logs in `jobs/`; MLX writes its worker progress to `run.log`.

Outcomes are `accepted`, `reference_retained`, or `incomplete`. Acceptance requires correctness, the minimum speedup, and a confirmation interval favoring the candidate. Timing noise can legitimately retain the reference. The deployment comparator is explicit in `objective.comparator` and frozen in the job before proposals or measurements. Native defaults are `numpy_copy` for copy and `numpy_buffered` for softmax/RMSNorm; `native_baseline` is an explicit alternative, and softmax also supports `scipy_softmax_float64` with the numerical extra. Discovery selection and final acceptance use that same comparator, including a per-case confirmation interval. A no-win exported artifact executes the declared comparator when available; unsupported environments use the independent reference. Diagnostic library-envelope comparisons never select a new comparator from evaluation data.

`lossless export RUN --output ARTIFACT` copies the selected implementation, contract and receipt. `lossless.load(ARTIFACT)` verifies hashes and runtime guards. Native bundles contain source plus the locally compiled library; another platform falls back rather than silently recompiling. They contain no model credentials or provider code. Native snapshots support completed-run inspection/resume with the same controller/configuration; unclean interruptions are conservatively charged. MLX interrupted searches require a fresh run. Neither workflow reopens evaluation for iterative tuning.

## MLX model optimization

Install `python -m pip install './lossless-product[mlx]'` from the repository root on Apple Silicon, then `lossless init --template mlx-fixed-count --output my-mlx-job`. Set `workload.parameters.model` to a local model directory. The request JSON contains `text`, `max_tokens`, and a `discovery`/`evaluation` split. The generated template enables scoped proofs; install Lean first or explicitly set `proofs.check` to false if you only require validation evidence.

The admitted optimization currently requires Apple M2, `mlx==0.32.3`, `mlx-lm==0.32.0`, and the exact locally converted 8-bit SmolLM2-135M artifact identified by the shipped file hashes. An arbitrary download of the same model name may have different bytes and will retain stock inference. Weights are unchanged during optimization and are never bundled. This is a retained-recipe adapter, not a generic model converter; broader model admission needs its own validation campaign.

The recipe combines equal-prefix cohorts, serial prefill, grouped heads, removal of dead terminal projections while preserving final KV updates, guarded cache copies, and allocator-cache retention. The LLM can currently choose `allocator_cache_mib` from `0`, `64`, and `256`; arbitrary MLX source proposals are not supported. Comparison checks include every emitted token, emitted log-probability tensor, active KV state, and request order. Cache tests include padding, growth, filtering, independent exports, and exceptional bit patterns.

Use a dedicated process with no concurrent MLX work:

```python
import lossless

operation = lossless.load("mlx-artifact")
# Optional if weights moved:
operation.open(model="/path/to/local/model")
row, caches, log_probabilities = operation.generate(
    ["The moon is", "The sun is"], [8, 8], capture=True
)
print(row["output_texts"])
```

This is an integration point in the application's inference execution path, after loading the exported recipe. Optimization is an earlier offline step. Each call executes complete requests; it does not modify selected tokens or run the search LLM at inference time. Fixed-count mode emits the requested token counts and does not implicitly stop at EOS. Explicit `stop_ids` or `cancel_after` uses the serial reference. Inference may improve bulk throughput while worsening first-token latency. Model loading is timed separately from inference. Internal cache overrides are restored on exit but require an exclusive worker process.

## Optional proof setup

`lossless setup --proofs lean` installs/reuses Lean 4.28.0. `--proofs mathlib` additionally prepares the shipped Lake manifest at its recorded mathlib revision, fetches compatible caches and builds the theorem project. Elan bootstrap and installer use [release 4.2.4](https://github.com/leanprover/elan/releases/tag/v4.2.4); setup records the downloaded bootstrap digest. Setup runs only on an explicit command, never during import, inspection or inference. Existing modified proof workspaces are refused rather than overwritten. Set `LOSSLESS_CACHE_DIR` or pass `--directory` to select the workspace.

For a substantial first-time download, run in your Terminal:

```sh
set -o pipefail
python -u -m lossless setup --proofs mathlib 2>&1 | tee lossless-mathlib-setup.log
```

Setup prints timestamped progress and result/log locations. Proof checks are scoped statements and cannot authorize numerical relaxation. The theorem index is proposal context, not checked evidence for a generated candidate.

## Search reuse, deployment overhead and payback

Native jobs default to a trusted local cache at `./.lossless/cache`, relative to the job. Configure `"cache": {"directory": null, "reuse_validation": false}` to disable it. Cache identities bind source, compiler flags/version, runtime packages, hardware inventory, harness, contract, comparator and (for validation) case/seed and binary identities. Exact duplicate proposal sources are recorded and rejected before compilation/measurement. This does not establish semantic equivalence of different source strings.

Only compilation and successful discovery correctness evidence are reused. Evaluation correctness and every timing remain fresh. Hashes detect accidental corruption, not a malicious cache owner. Native reports contain per-case search-only break-even call counts; total payback stays unknown when deployment setup has not been measured. The bound-buffer API avoids benchmark-only initialization while preserving the complete scratch-buffer ABI.

MLX jobs may opt into `"graph_recipes": ["decoder_026", "fullhead_027"]` inside `workload.parameters`, with `max_candidates` at least three to compare both with the retained recipe. These are alternatives, not additive gains. Each model handle holds at most 64 explicit shape/dtype/index specializations by default; unseen keys after the limit use the retained eager path. The hard maximum is 128 entries. This bounds entry count, not an absolute driver-memory limit. The model is immutable for the lifetime of the handle.

Graph promotion additionally requires a fresh paired interval more than 1% above the retained recipe. Both first validation-call cost and warmed measurements are recorded. MLX reports estimate payback from measured extra first-call cost after model loading. Graphs remain opt-in and require the same exact model/runtime/device admission. Stopping/cancellation uses serial fallback. Provider timeouts honor `llm.timeout_seconds` and the remaining total budget in both backends.

## Workload profiling

`lossless profile JOB --repeats 7 --budget 30s --output DIRECTORY` profiles discovery inputs without invoking providers or proof setup. Pass `--artifact PATH` to compare an exported implementation. Native ordinary and bound calls are separate; MLX records complete requests with probabilities, state, TTFT, memory and coarse phase accounting. See the [profiling guide](profiling.md) for setup costs, p95 interpretation, host attribution and fallback scope.

## MLX deployment limits (unreleased)

Optional `objective.constraints` are frozen before search and checked in discovery, evaluation, and the final decision. They supplement all existing correctness, comparator and graph-incremental gates. For example, merge this objective into an MLX job and set `workload.parameters.repeats` to at least 20:

```json
{
  "metric": "throughput",
  "comparator": "stock_serial",
  "min_speedup": 1.02,
  "constraints": {
    "max_p95_ttft_seconds": 0.15,
    "max_p95_completion_seconds": 1.0,
    "max_peak_mlx_bytes": 536870912,
    "max_break_even_calls": 1000,
    "min_tokens_per_second": 100
  }
}
```

Choose limits for your application; these illustrative values are not universal defaults. Every field is optional. Byte and call limits are positive integers. Unsupported or unavailable measurements fail the gate. These fields currently require `mlx.fixed_count`; native request-tail latency and per-call memory constraints are not implemented.

Latency p95 is computed across warm repeats separately for each request position; the worst position must meet the limit. At least 20 repeats are required for latency gates. This remains a descriptive sampled percentile, not a statistical guarantee about production tails or future arrivals. The profiler's default seven-repeat p95 is insufficient for this acceptance gate.

Memory is the largest MLX-tracked active allocation during warm calls, including the model and returned KV state, with probability capture disabled as in the default deployed interface. Allocator-cache memory, driver allocations and process RSS are excluded; this is not a total system-memory cap. Exact validation separately captures and compares every probability and active KV value.

Payback uses the entire measured search cost plus relative first-call excess over warm, divided by measured per-batch savings. Shared model loading is excluded from deployment setup but included when it occurred during the search. The final check accounts for the completed search rather than only the cost seen at discovery. No positive saving means no finite payback. Unmeasured human/provider billing cost is not invented.

`deployment_metrics`, raw samples and per-limit pass/fail records appear in the report. If no candidate satisfies all gates, Lossless retains the reference; that outcome does **not** certify that the reference meets the requested limits. These are offline acceptance checks for the declared workload, not an online latency/memory enforcement mechanism for arbitrary later requests.
