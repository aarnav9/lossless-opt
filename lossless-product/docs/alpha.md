# Using the source alpha

This is the implemented interface for version `0.1.0a1`. Files labeled `draft-1` and the broader design documents remain proposals.

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

## Provider boundary

Choose either a callback or a command:

```json
{"callback": "provider.py:complete", "credential_env": ["MY_MODEL_API_KEY"], "timeout_seconds": 30}
```

```json
{"command": ["python", "provider.py"], "credential_env": ["MY_MODEL_API_KEY"], "timeout_seconds": 30}
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

The callback path resolves relative to the job JSON. The callback selects the model; `llm.provider` and `llm.model` are labels, not built-in SDK routing. The whole-job budget includes setup, provider calls, compilation, checks, and timing. Each call receives at most its configured timeout and the remaining search budget. Built-in candidates also consume candidate slots. Slow responses can exhaust the budget; a completed search can legitimately retain the reference.

```sh
lossless inspect ./my-job/lossless.json
lossless optimize ./my-job/lossless.json --output ./lossless-runs/claude-test
lossless report ./lossless-runs/claude-test
```

### Codex CLI or Claude Code

Both can be connected through a user-supplied command wrapper. The wrapper reads the Lossless request from stdin, asks the CLI for a proposal using an actual JSON Schema for the active adapter, and writes only the proposal batch to stdout. The request's `response_format` illustrates the expected object; it is not itself a JSON Schema.

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

### What has been tested

The CPU workflow regression sends a deterministic provider the real discovery context and receives the reference C implementation as a control candidate. It then exercises compilation, contract checks, timing, separate evaluation, and export/load. The installed MLX check sends a deterministic command the request and receives a 64 MiB allocator proposal; the retained 256 MiB recipe is selected. Malformed responses, timeout behavior, credential echoes, and evaluation-prompt separation have regression coverage.

No live Claude/Codex run was part of those checks. The measured MLX retention gain comes from the retained recipe, and does not establish live-model proposal quality or search ROI. A bounded live-provider run remains a release-validation task. [Recorded validation](retention.md).

## Reports and deployment

A run directory contains the resolved job, expanded contract, hardware evidence, immutable source/harness hashes, discovery feedback, a frozen selection, evaluation measurements, model-call receipts, `report.json`, `report.html`, and logs. CPU workers have individual logs in `jobs/`; MLX writes its worker progress to `run.log`.

Outcomes are `accepted`, `reference_retained`, or `incomplete`. Acceptance requires correctness, the minimum speedup, and a confirmation interval favoring the candidate. Timing noise can legitimately retain the reference. Library-baseline comparisons are reported alongside native-baseline measurements; acceptance in this alpha is relative to the frozen native baseline. A gain against that baseline does not imply a gain over every library implementation.

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
