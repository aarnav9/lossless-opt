# Use your ChatGPT subscription through Codex CLI

Current source includes `python -m lossless.codex_provider`, a command transport
for the installed Codex CLI. It reuses **ChatGPT sign-in**, requires no API key,
and consumes your account's Codex allowance. It does not make subscription usage
unlimited or turn API usage into a free service. This connector is not in the
immutable `v0.1.0a1` release wheel.

## Ready-to-run path

Install the current source in a virtual environment and install/sign into Codex
CLI using the [official setup instructions](https://learn.chatgpt.com/docs/auth).
From the repository root, with that environment activated:

```sh
python -m pip install ./lossless-product
codex login
codex login status
python -m lossless.codex_provider --model gpt-6.1-sol --probe --timeout 60
lossless inspect lossless-product/examples/codex-softmax/lossless.json
set -o pipefail
python -u -m lossless optimize lossless-product/examples/codex-softmax/lossless.json --output ./lossless-runs/codex-softmax 2>&1 | tee lossless-codex-softmax.log
lossless export ./lossless-runs/codex-softmax --output ./lossless-runs/codex-softmax-artifact
```

The search can take **30 minutes**. Progress is timestamped, the command runs in
the foreground, and the named log captures it. A completed search may retain the
reference. Use a fresh output directory for another run. Install commands may
download Python build dependencies; no model weights are downloaded.

The user supplies one JSON job:
[examples/codex-softmax/lossless.json](../examples/codex-softmax/lossless.json).
It contains workload cases, correctness contract, deployment comparator and
time/candidate/call caps. The provider receives generated discovery context;
the user does not need to write a model prompt or upload a model for this native
softmax example.

The example uses this connection:

```json
{
  "llm": {
    "provider": "codex-chatgpt",
    "model": "gpt-6.1-sol",
    "command": ["python", "-m", "lossless.codex_provider", "--model", "gpt-6.1-sol", "--effort", "xhigh"],
    "timeout_seconds": 1800
  },
  "budget": {
    "wall_time_seconds": 1800,
    "max_candidates": 8,
    "max_llm_calls": 6
  }
}
```

Change **both** the model label and the `--model` argument to choose another
accessible model. The command argument actually selects the model. Use the same
Python environment for `lossless` and `python`; an absolute Python executable
path is also valid. Custom `CODEX_HOME` installations can explicitly forward that
path using `"credential_env": ["CODEX_HOME"]`; authentication contents are never
placed in the job.

`budget.wall_time_seconds` caps the entire search, including model calls and
measurement. `llm.timeout_seconds` caps each response; remaining search time and
evaluation reservation can shorten it. The connector inherits that effective
allowance from Lossless, so changing `llm.timeout_seconds` or using
`--llm-timeout 1h` works without another timeout setting. An optional connector
`--timeout` can impose a shorter extra cap. Standalone invocations outside
Lossless default to 1800 seconds. One proposal is requested per call; built-in
softmax recipes consume two of the eight candidate slots. Runs can finish before
their time allowance.

The connector checks ChatGPT authentication, strips API-key environment
variables, uses an empty working directory and disables CLI tools, apps,
plugins, repository instructions, browsing and subagents. It accepts only a
completed, schema-valid final response, rejects tool events and late responses,
and returns final proposal JSON instead of the CLI's event stream. Lossless
separately compiles, checks and times the C code. This remains trusted local
candidate execution, not an OS sandbox for arbitrary untrusted C.

The native adapters accept C proposals. The MLX connection is limited to the
existing allocator-policy proposal schema; this connector does not add arbitrary
GPU kernel generation. Only native copy and softmax have live connector checks
in this work; MLX live-provider behavior is not established by those checks.

## What was actually checked

On 4 October 2026, Codex CLI **0.159.0** reported ChatGPT sign-in. Minimal
schema-constrained probes succeeded for `gpt-6.1-sol`, `gpt-6-astra`, and
`gpt-6-luna`, all at `xhigh`. Access depends on the account and client; consult the
[official model documentation](https://learn.chatgpt.com/docs/models).

A live native-copy job used the shipped command transport with GPT-6.1 Sol at
`low`, returned an eligible proposal in **37.55 seconds**, passed finite exact
checks, and completed artifact export/load. It did not beat the configured
NumPy comparator, so the result was `reference_retained`. The receipt recorded
13,609 input and 1,087 output tokens, with dollar cost unknown. This smoke verifies
the connection and acceptance boundary, not model optimization quality.

[Campaign 047](models-047.md) compares model choice through the iterative native
research loop. That larger experiment uses a frozen common seed, nine discovery
cases, twelve final cases and matched maximum search allowances; the small ready
example above is not a reproduction of that study.

## DeepSeek and local models

DeepSeek's API documentation lists token prices for V4 models. Free access to a
chat website does not establish free API or CLI automation. A CLI that calls a
paid API retains that API's charges. See the
[official API pricing](https://api-docs.deepseek.com/quick_start/pricing/) and
[model updates](https://api-docs.deepseek.com/updates/).

Local inference avoids provider token charges, but needs sufficient hardware,
RAM, storage and electricity. It can be inexpensive on hardware you already own;
it is not inherently a paid API. A local model can use the existing callback or
command boundary, but none was downloaded or qualified here. For this account,
the verified ready path is the existing ChatGPT-signed-in Codex CLI.
