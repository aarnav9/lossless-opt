# Campaign 048: whole-request MLX implementation search

**Completed:** six exact discovery candidates; the selected implementation reached
approximately 1.098× the frozen references in each of two held-out processes but
failed the full deployment gate. See the [results, graph and evidence](../../docs/fullstack-048.md).
No product recipe was promoted.

One subscription-backed GPT-6 Astra (`xhigh`) author receives source, detected M2
hardware, complete-request discovery profiles and feedback. It may submit an
entire Python/MLX implementation module covering scheduling, cache movement,
scratch buffers, graph execution and Metal kernels. This is a research harness,
not an expansion of the product's allocator-only MLX proposal interface.

The one-hour loop includes authoring and candidate discovery measurement. It has
16 call slots and a 30-minute response cap, shortened by remaining time with a
three-minute measurement reservation. No new call starts with less than 270
seconds left. Initial reference screening and final evaluation are separate.
Every failed, duplicate or timed-out response consumes a slot. No manual code
repairs occur during the loop.

Eight discovery families cover one request, four/eight equal-length requests,
unequal generation counts, ragged prompts, two prompt-length cohorts, a longer
prefill, and natural-language/code requests. Eight unopened counterparts use
different text, lengths and output counts. These are finite constructed requests,
not production traces or a new-model generalization study.
Synthetic texts repeat one word; equal-length rows therefore contain identical
prompts. Natural-language/code requests are distinct. Legitimate within-call work
sharing can exploit the former, so report those families separately rather than
generalizing their gains to unrelated prompts. Reusing outputs/prefill/KV across
separate calls is forbidden in this experiment.

Before authoring, seven repeat blocks compare stock serial, unmodified stock
batching, retained, decoder-026 and fullhead-027. The fastest method passing exact
checks is frozen separately for each discovery family. Those method names also
control the corresponding final family; final timings never select a new
reference. Stock serial is additionally measured throughout.

The fixed controller times tokenization, candidate execution, full probability
and final-state materialization, decoding and response serialization. Every timed
call uses `capture=True`; this differs from prior generation-only warm studies.
All outputs are checked against byte digests from a separate stock process.
Checks include tokens/order, probability shapes/dtypes/bytes, active KV
shapes/dtypes/offsets/bytes, three forced continuation steps, mutation independence
of exported requests, unchanged earlier-call outputs and unchanged weights.
Global implementation overrides must be restored after each call.

Discovery selects the highest geometric mean among exact candidates satisfying
memory and reported-latency limits. One selected source is sealed and evaluated
in two fresh processes with eleven randomized repeat blocks each. A win requires
the discovery gate and both final processes to have a >=1.02 geometric mean,
every per-case paired lower 95% interval above one, warm peak MLX allocation no
more than reference ×1.10 +16 MiB, and maximum backend-reported first-token time
no more than reference ×1.10 +5 ms. No second-choice retry follows final failure.

Reported first-token timestamps come from the implementation, are range checked,
and are not independently instrumented token-emission traces. MLX peaks exclude
some driver/process memory and include live retained validation outputs; paired
calls share that scope. A separate 4 GiB worker RSS limit stops runaway workers.
Constructor and first-call setup are reported separately from warmed timing.
Finite exact checks are not a compiled-code equivalence proof. Python import/AST
checks prevent accidental scope mistakes, **not hostile code execution**; this is
trusted local research. The author has tools disabled and no evaluation inputs.

## Reproduce

Use the existing admitted M2, MLX 0.32.3 / mlx-lm 0.32.0 environment and exact
local 8-bit SmolLM2 artifact. No weights are downloaded. Authoring uses ChatGPT
sign-in through Codex CLI, consumes subscription allowance and reports unknown
dollar cost as null. From the repository root:

```sh
export PYTHONPATH="$PWD/lossless-product/src"
.venv-fast-mlx/bin/python -u lossless-product/experiments/fullstack_048/study.py freeze --model research/artifacts/fast_smollm2_mlx_8 --output research/results/fullstack_048
set -o pipefail
.venv-fast-mlx/bin/python -u lossless-product/experiments/fullstack_048/study.py run --output research/results/fullstack_048 2>&1 | tee research/results/fullstack_048.log
```

Allow roughly 1½–3 hours including reference screening, the maximum one-hour
loop, and fresh confirmation. Progress and result/log paths are timestamped.
Use a new output directory for every run. The frozen source snapshot, workload,
reference choices and final selection are hashed. `smoke` instead runs one small
case and a deliberate wrong-probability negative control in about 20 seconds.

No result automatically modifies default recipes or exports a production
artifact. The experiment answers whether this one author trajectory improves
complete requests on this pinned setup; it cannot rank models or establish
cross-device gains.

## Timing interruption in the recorded run

The Mac entered lid-close sleep during author round three. On the recorded Python
environment, `time.monotonic()` uses `mach_absolute_time()`, which pauses during
system sleep. Consequently the frozen one-hour counter is an awake-time budget;
the actual UTC elapsed duration is longer. The analysis preserves both clocks,
identifies the gap from timestamped progress, and charges full elapsed time in a
separate payback calculation. Remote author compute during the sleep is unknown.
Do not treat this run as an uninterrupted 60-minute wall-clock comparison.

The recorded harness is left frozen. For a future strict wall-time budget, use a
sleep-inclusive monotonic clock and reject late completions after resumption;
Apple documents [the continuous clock](https://developer.apple.com/documentation/kernel/1646199-mach_continuous_time).
No GPU timing worker was active during this observed sleep interval.
