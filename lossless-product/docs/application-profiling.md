# Profile an existing application, then assess the evidence

Current source extends `lossless profile` to application jobs with callable
entries. It automatically freezes discovery inputs/source, measures the original
application, checks observed repeatability and writes a readable report. If a
model connection is configured, it then sends bounded evidence to that model
without a manual report/prompt handoff. This is step 2 of the
[application plan](implementation.md#current-priority-bring-an-existing-application-5-october-2026).

This feature is not in the published `v0.1.0a1` wheel. Source modification and
candidate acceptance now use a separate, explicit
[source-search experiment](application-search.md); guarded application deployment
remains a later stage.

## One workflow

Start with an [application job](applications.md) containing an invocation and
representative inputs. Profiling alone does not make a model call:

```sh
lossless profile .lossless/my-app/lossless.json --repeats 5 --budget 2m --output .lossless/my-app/runs/profile
```

To include an assessment using your existing Codex ChatGPT sign-in:

```sh
set -o pipefail
python -u -m lossless profile .lossless/my-app/lossless.json \
  --explain --model gpt-6-astra --effort medium --budget 30m \
  --output .lossless/my-app/runs/profile-with-assessment \
  2>&1 | tee lossless-application-profile.log
```

The optional assessment shares discovery timings, memory/host attribution and
bounded excerpts of relevant project source with the selected provider. It
does not send raw input arrays, evaluation cases, environment variable values
or the entire repository. Review `assessment_context.json` to see the supplied
evidence; source excerpts may still contain project data. `--no-explain` disables
a configured connection for a run. Model/effort/timeout flags also explicitly
enable assessment. Ordinary `init`, `inspect` and `replay` make no model calls.

Save the choices in the generated JSON to make subsequent profiles automatic:

```json
{
  "profile": {"repeats": 5, "wall_time_seconds": 1800},
  "analysis": {
    "provider": "codex-chatgpt",
    "model": "gpt-6-astra",
    "effort": "medium",
    "timeout_seconds": 1800
  }
}
```

These are optional sections of an application job, not a complete job by
themselves. Both time defaults are 30 minutes. `--budget` caps measurement and
assessment together; `--llm-timeout` limits the response further. Remaining total
time can shorten the model allowance. No retries or model substitutions occur.
The run ends as soon as its work completes, rather than waiting out its allowance.
Model availability and subscription usage depend on the signed-in account.
The [Codex connector](codex.md) excludes API-key environment variables and checks
for ChatGPT authentication. The existing `replay.timeout_seconds` limit also caps
each measurement worker and may be increased for long individual requests.

Python has the same interface:

```python
import lossless

result = lossless.profile(
    ".lossless/my-app/lossless.json", repeats=5, budget=1800,
    output=".lossless/my-app/runs/profile",
    analysis={"provider": "codex-chatgpt", "model": "gpt-6-astra", "effort": "medium"},
)
print(result.summary["status"], result.summary["assessment"]["status"])
```

`analysis=None` uses the job's connection, which is disabled in generated jobs.
`analysis=False` disables it. This first assessment transport is Codex-specific;
it does not change the existing native optimizer's provider interfaces.

## What is measured

For each **discovery** case, the profiler uses five fresh processes by default
(configurable 3–50) for clean timing, one for host attribution and one for traced
memory. Each process executes exactly the declared call sequence once. A factory
is constructed once per process and persists within that sequence. No hidden
warmup or repeated state transition is inserted.

For repeated requests, declare them using the existing case `calls` array. The
report retains first-call and per-position medians as well as the summed sequence
time. Calls can do different work; this is not automatically a warmed latency
distribution or a measured production workload mix.

| Measurement | Boundary |
| --- | --- |
| Application setup | Import, factory construction and completion synchronization; interpreter/harness imports excluded |
| Call/sequence times | Existing callable plus completion synchronization; fixture loading, correctness fingerprints and observer hooks excluded |
| Host hotspots | Separate `cProfile` process, source file/function/line, self and inclusive time; inclusive times overlap |
| Profiler overhead | Instrumented sequence time divided by clean median; noisy diagnostic, not speedup |
| Process memory | RSS lifetime high-water mark including imports, inputs and earlier observations |
| Allocation evidence | Separate `tracemalloc` live-peak/retained measurements and retained source sites; not total allocation traffic |

Source links point into the frozen checkout. Full observations remain in each
attempt's `worker_result.json`; reports retain their fingerprints and compact
measurements. All clean and instrumented processes must match observed outputs
and input states. Mismatches, exceptions, timeouts or incomplete passes prevent
a successful profile and prevent model assessment.

Observation occurs outside call timers but between calls in a sequence, so it
can perturb caches. Tracing can also change timing and allocation behavior.
These effects are why attribution/memory passes are separate and overhead is
reported. Exact repeatability is finite evidence for the original reference,
not proof of arbitrary application semantics.

## Completion and current limits

Synchronous Python/NumPy/SciPy work needs no device hook. If imported, MLX results
are materialized and queued work synchronized; initialized PyTorch CUDA devices
and available MPS are synchronized. The report names the synchronization used.
An explicit no-argument `entry.synchronize`, such as `driver.py:synchronize`,
overrides this inference and must wait for the application's relevant work.
JAX/CuPy/TensorFlow imports require that hook. Custom observers are still needed
for unsupported return types. These hooks execute trusted project code.

Device synchronization adds measurable cost, including when a framework is
imported but does little device work. CPU measurements and small local MLX/MPS
smokes do not qualify arbitrary device programs. GPU kernel attribution, device
allocation accounting and copy/transfer byte counts are not implemented here.
Opaque native library internals are not decomposed by `cProfile`.

Command jobs retain reference replay, but this profiler currently requires a
callable. The callable boundary should include the work the user wants measured;
it does not automatically include an external web server, request serialization
or an undeclared caller pipeline. Missing invocation/input discovery remains an
intake concern rather than something the profiler or assessment silently invents.

## Model authority and failure behavior

The model gets at most eight measured cases, twelve host hotspots per case and
twelve bounded source excerpts, under a 180 KB context limit. Context selects
expensive measured cases without assigning them a production frequency.
Memory context includes the first and largest traced-allocation calls, their
positions, omitted-call count and the full peak range, so long sequences do not
repeat hundreds of nearly identical allocation stacks in the prompt.
Codex runs with tools/repository access disabled in an empty working directory.
No package installation, source edit or benchmark is delegated to it.

The response must satisfy a strict schema. Each proposed opportunity must cite
known evidence IDs, explain its hypothesis, describe a change and propose its
validation. Unknown evidence references, unexpected fields, tools, timeouts and
missing completion receipts fail assessment. The system checks references and
structure; it does not certify the truth of model prose. Numeric measurements
and acceptance remain outside the model's authority.

Successful measurement returns `profiled`; assessment has a separate status.
If the model fails, valid profiling evidence remains available. The CLI exits
nonzero for a failed/incomplete requested assessment or a failed profile. Results
are not exportable as optimized implementations. Token usage and model/effort,
prompt/response hashes, timeout and completion receipts are retained; dollar
cost is unknown for this subscription-backed transport.

[External repository results and live Astra assessment](application-profile-study.md).
