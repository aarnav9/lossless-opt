# Local CPU Python projects

`run-project` takes a local Python/NumPy callable and representative inputs through
configuration, profiling, local candidate selection, exact evaluation, and a
runnable project copy. It makes **zero model calls**. Install current source in
the project's Python environment; the older GitHub release does not contain this
command:

```sh
python -m pip install './lossless-product[applications]'
lossless run-project ./my-project --entry pipeline.py:run \
  --cases ./cases.json --budget 3m --output ./lossless-runs/project
```

Run from the repository root for that installation command. The output directory
must be new and outside the input project. The active Python is the default;
`--python /path/to/venv/bin/python` selects another environment containing Lossless
and your application's dependencies. Dependencies are inspected, not installed
automatically. Omitting the callable or cases generates setup and entry suggestions
with `needs_input`, without importing the application.

## Supply the callable and inputs

The entry can be `pipeline.py:run` or `package.module:run`. Cases are a JSON list:

```json
[
  {"id": "development", "split": "discovery", "args": [{"$npy": "inputs/dev.npy"}]},
  {"id": "held_out", "split": "evaluation", "args": [{"$npy": "inputs/new.npy"}]}
]
```

Array files are relative to the project and load with `allow_pickle=False`.
JSON scalars, lists, string-key dictionaries, and `kwargs` are supported. Referenced
fixtures are included even when gitignored; explicit project exclusions still apply.
`--include` and `--exclude` accept repeatable project-relative paths.

At least two independent cases are required, with both splits represented. If all
split labels are omitted, the first two thirds become discovery and the remainder
evaluation; missing IDs are generated. Supply realistic small/large inputs and the
branches that matter to your application. Input quality determines evidence quality.
Evaluation inputs are frozen before search and withheld from candidate selection.

For a sequence that reuses an object, supply `calls` instead of a single `args`:

```json
{"id": "changing_signals", "split": "evaluation", "calls": [
  {"args": [{"$npy": "inputs/first.npy"}]},
  {"args": [{"$npy": "inputs/second.npy"}]}
]}
```

Use `--factory --entry app.py:make_processor --observe app.py:observe_processor`
for a zero-argument factory returning a callable. The observer signature is
`observe_processor(processor, result, args, kwargs)`; it must read and return the
relevant outputs/state without mutating them. An optional `--cleanup` hook receives
the processor. Each launch starts a new object; calls in one sequence share it.

Only declared environment variables reach the application, plus temporary HOME,
temporary-directory, interpreter, and seed settings. For example, set
`OPENBLAS_NUM_THREADS=1` and pass `--inherit-env OPENBLAS_NUM_THREADS` to qualify
that thread policy. Keep inputs and source stable during a run.

## What is automatic today

The workflow profiles discovery cases and tries applicable local recipes:

- Recognized pybaselines 1.2.1 source: retained source edits, compiled ModPoly helper,
  preparation/cache policies, and independent-spectrum pools in supported combinations.
- Other projects: a small source candidate family replaces selected `np.sum(x)`,
  `np.mean(x)`, other simple reductions, and `np.copy(x)` calls in measured functions
  with guarded ndarray methods, including expression arguments such as `np.all(diff < tol)`.
  Another candidate shares repeated arithmetic expressions in straight-line functions
  with single-assignment locals and numeric type guards. It preserves operator grouping;
  custom numeric types and object arrays retain the original function body. Separate
  and combined candidates compete. This does not enable algebraic reassociation.
- An optional `--plan saved-plan.json` supplies local source/native candidates through
  the existing evaluator. Provider plans are rejected; hosted authoring remains a
  separate explicit `search-application` operation.

This is a limited recipe library, not a general optimizer for every Python program.
Each combination is measured; individual research gains are never multiplied into
a promised project speedup. Candidate or resource failures retain the reference.
Budget exhaustion or broken reference behavior is reported as a failed/incomplete
workflow, never a successful optimization. Long runs print timestamped progress.

Acceptance uses existing exact observations, held-out paired timings, call-position
regression gates, setup costs, traced allocations and process-tree memory checks.
Default searches use six evaluation pairs and require at least 1.02× aggregate
callable speed with statistical evidence. Numerical approximations are not accepted
by this command. Insufficient memory-monitoring permissions prevent qualification.

## Run the result and recover actual outputs

A completed run writes `report.html`, `report.json`, `run.log`, and `application/`.
The application directory contains the frozen reference, an optimized copy if one
qualified, saved cases, a manifest, and a launcher:

```sh
python ./lossless-runs/project/application/run.py --case held_out --output ./results/first
python ./lossless-runs/project/application/run.py --request ./new-request.json --output ./results/new
python ./lossless-runs/project/application/run.py --case held_out --reference --output ./results/original
```

New request JSON uses `args`/`kwargs` or `calls`. Its `$npy` paths are relative to
the request file. Actual returned values are saved as JSON descriptors and NumPy
files, with bit-preserving scalar encoding and no pickle:

```python
from lossless.applications import read_outputs
values = read_outputs('./results/new')  # one return value per call
```

Before execution, the launcher checks source/binary hashes, installed package
versions, Python, host, Lossless runtime, declared environment settings, and input
metadata. Optimized execution admits new **finite numeric array values** only with
measured shapes, dtypes, strides, flags, sequence structure and fixed scalar
controls. An unsupported input or changed environment selects the reference before
running. A modified reference copy stops the launch. There is no retry after a
callable starts, so a partial failure cannot double-apply side effects.

The bundle can move on the same qualified host with the same Python environment.
It requires the Lossless runtime and original project dependencies. Use
`--reference` to disable optimization; the input project was never modified.
Saved returns preserve values, not Python aliases across objects/results. The
launcher executes trusted local code; copying files is not an OS security sandbox.

## Understand the performance report

Qualified speed describes complete callable sequences. Interpreter startup,
launcher environment preparation, source/dependency checks, fixture loading,
observation, and output serialization sit outside that measurement.

The workflow also launches both delivered copies on every evaluation case and
checks their outputs, mutations, and retained observations exactly. These **single
descriptive launch pairs** include the whole process cost. They are not statistical
evidence of an end-to-end launch speedup. Short jobs may see no launch improvement
even when their compute becomes faster.

Correctness evidence is finite and scoped to the supplied observers and cases; it
is not a proof for every future value. General GPU work, arbitrary shell commands,
transparent in-process installation, external side-effect equivalence and general
network services are outside this workflow. A no-win reference bundle is a valid result.

Try the [small NumPy example](../examples/cpu-project/README.md). External validation
uses `scripts/check_cpu_project.py`; see the [experiment record](cpu-project-workflow-study.md)
for positive results, negative results and complete launch costs.

## Reuse a prepared CPU worker

On macOS/Linux, add `--prepared` to `run-project` to qualify a second execution mode:

```sh
lossless run-project ./my-project --entry pipeline.py:run --cases ./cases.json \
  --prepared --budget 5m --output ./lossless-runs/prepared
```

The workflow still validates fresh launches. It additionally compares two equally
persistent workers, reference and candidate, over six paired rounds. Each round
repeats every evaluation case three times, rotating case and variant order. Every
request must match the fresh reference's observations **and actual transported
return values**. A separate `prepared.json` certificate requires complete request
speed, per-case/per-position regression and process-tree memory gates. Source
acceptance and prepared acceptance are separate report fields.

```python
from lossless.applications import PreparedApplication

with PreparedApplication('./lossless-runs/prepared/application') as app:
    print(app.info)  # selected variant, fallback reason and worker PID
    first = app.run_case('held_out')['outputs']
    second = app.run({'args': [{'$npy': 'new.npy'}]}, input_root='./new-inputs')['outputs']
```

The API owns one private process and immutable source snapshot. It reuses imports,
but seeds and creates a **fresh factory for every request**; calls within a request
share that factory. Cleanup runs after each request. JSON requests and `.npy` inputs
are loaded afresh; outputs travel as copied values over a bounded JSON/NumPy channel
without pickle. Responses are limited to 64 MiB. Only one request runs at a time.
The context manager stops the process and removes its private files.

An absent/failed certificate or changed environment selects a reference session.
Once an optimized session starts, unsupported request metadata raises before the
callable executes. Use `PreparedApplication(bundle, variant='reference')` explicitly
for those inputs. Application errors and timeouts stop the worker with **no retry**.
Modules remain loaded, so stateful modules may fail prepared qualification even
when fresh launches qualify. This is finite observed lifecycle evidence, not a
proof about hidden state or external side effects.

Prepared timing includes fixture loading, guards, factory creation, observation,
cleanup, serialization, transport and decoding. Startup is recorded separately;
it must be amortized across enough requests. Memory qualification requires the
`applications` dependencies and working process monitoring. This is a local value
request API, not transparent in-process replacement or a network service.

## Diagnose an unstable reference

If reference profiling fails, the workflow attempts a short discovery-only
diagnostic and still stops exact optimization. To collect additional snapshots:

```sh
lossless diagnose-application ./lossless-runs/project/job/lossless.json \
  --repeats 8 --budget 1m --output ./lossless-runs/reference-diagnostic
```

The diagnostic saves actual returned values, exact fingerprints, differing element
counts, sample values, absolute/relative component differences and finite float32/64
ULP distances. It does not introduce tolerances or approve numerical candidates.
Trace instrumentation can change repeatability; a repeatable diagnostic run does
not overturn the original failure.

The [prepared execution follow-up](cpu-prepared-study.md) records the broader
financial candidates and reference diagnostics.
