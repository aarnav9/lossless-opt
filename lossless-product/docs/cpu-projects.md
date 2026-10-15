# Local CPU Python projects

`run-project` takes a local Python/NumPy callable and representative inputs through
configuration, profiling, local candidate selection, exact evaluation, and a
runnable project copy. It makes **zero model calls**. Install the 0.1.0a2 alpha
or current source in the project's Python environment:

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

## Supply examples directly from Python

Instead of writing JSON and `.npy` files, provide a zero-argument loader in your
project. It must return a list with explicit IDs and discovery/evaluation splits:

```python
# examples.py
import numpy as np

def cases():
    rng = np.random.default_rng(17)
    return [
        dict(id="development", split="discovery", args=[rng.normal(size=(64, 32))]),
        dict(id="held_out", split="evaluation", args=[rng.normal(size=(64, 32))]),
    ]
```

```sh
lossless run-project ./my-project --entry pipeline.py:run \
  --input-loader examples.py:cases --deployment auto --expected-requests 100 \
  --budget 5m --output ./lossless-runs/project
```

The loader runs once, with the selected Python and declared environment, in a
copied project. Its examples are frozen before profiling or selection. The
original project is unchanged. Both `args`/`kwargs` and `calls` sequences work.
The loader supports plain contiguous numeric NumPy arrays and finite JSON
values; it rejects tuples, custom objects, object arrays and reserved `$npy`
dictionaries rather than silently changing their types. Limits: 1 MiB of JSON,
128 MiB of array fixtures, 1,000 cases, and 60 seconds for loader execution.
Direct requests retain the smaller 64 MiB transport limit. Use `--cases` for the
existing fixture path; the two input options are mutually exclusive.

For an existing test helper, import it from the loader and return its input data.
`CaseRecorder` also provides explicit capture before input mutation:

```python
from lossless.applications import CaseRecorder

def cases():
    recorded = CaseRecorder()
    recorded.call("development", "discovery", original_function, development_array)
    recorded.call("held_out", "evaluation", original_function, independent_array)
    return recorded.cases
```

Those calls execute the original function. There is no automatic pytest discovery
or inferred split. Loaders and applications are trusted local code, not sandboxed
code. Do not use loaders with external effects you do not want performed.

## Call the delivered function from your application

```python
from lossless.applications import CallableApplication

with CallableApplication('./lossless-runs/project/application') as run:
    result = run(new_array)  # ordinary args/kwargs and the actual return value
    print(run.info)         # variant, mode, and fallback reason
```

The bundle also contains `callable.py` with an `open_callable(**options)` helper.
Use a context manager to manage worker lifetime. `run_calls([{...}, {...}])`
returns one result per invocation and explicitly shares a factory within that
sequence. Single calls reset the factory. Inputs/results are copied values;
caller aliases and input mutations are not propagated. Supported array layouts
and finite metadata admission are the same as the live-value API below.

A missing/invalid callable certificate defaults to fresh reference execution.
An optimized metadata miss raises **before execution**; open a separate
`CallableApplication(bundle, variant='reference')` for unsupported inputs. There
is no automatic retry after an application error or timeout. A changed source,
package version, host or declared environment prevents optimized admission.
Requalify with the same project command into a new output directory. The original
project and earlier bundles remain available; remove the adapter to revert.

## Select for the deployment you will use

`--deployment` opts into complete adapter measurements:

- `oneshot`: a fresh worker for every call, including startup and cleanup.
- `prepared`: consider reusable imports as well as the fresh-reference fallback.
- `auto`: compare both lifecycles on discovery inputs and freeze one choice.

`auto`/`prepared` explicitly permit import reuse; neither promises preservation
of undeclared module state or external side effects. `--expected-requests N`
amortizes adapter/worker startup and cleanup over N calls. Choose 1 for a single
request. Measured request positions include copying, input validation, observations,
communication and output decoding. Reports retain startup, request and memory
measurements separately; projected amortization is not a measured N-call session.

All exact discovery-matched source candidates can compete, including combinations.
Selection reads discovery timings only. The callable-search winner is an incumbent
comparison, frozen before evaluation. A single source/lifecycle choice is then
frozen and tested over six paired evaluation rounds against the same-mode
reference, fresh reference where applicable, and the incumbent where different.
Performance tests share a 5% error budget across the required comparisons; case,
request-position and process-memory gates still apply. No failed evaluation
starts another selection round. If a source change cannot beat an equally prepared
reference, worker reuse is reported as a runtime change with original source.

`report.json` contains `deployment.frozen_selection`, separate comparisons,
rejected discovery options, and the final reason. `application/callable.json`
binds the result to its manifest, input boundary, runtime features and lifecycle.
It cannot authorize the legacy file/prepared APIs. The source-only and legacy
`--prepared` paths remain available; choose one qualification path per run.

Try the [complete callable example](../examples/cpu-callable/README.md).

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

### Pass live arrays without staging files

For a caller that already holds NumPy arrays, qualify the copied-value boundary:

```sh
lossless run-project ./my-project --entry pipeline.py:run --cases ./cases.json \
  --prepared --prepared-inputs values --budget 5m --output ./lossless-runs/live-arrays
```

```python
with PreparedApplication('./lossless-runs/live-arrays/application', input_mode='values') as app:
    result = app.run_values({'calls': [{'args': [new_signal], 'kwargs': {}}]})
    outputs = result['outputs']
```

`prepared-values.json` qualifies this boundary independently from `prepared.json`.
Both the original and optimized source receive the same runtime improvements.
The timer includes checking, copying and transporting live inputs, all application
work, and transporting outputs; caller input creation and worker startup are separate.
A file-request certificate cannot enable optimized value requests, or vice versa.

Inputs must be plain, aligned, writeable, C/F-contiguous numeric arrays, or supported
Python scalars and containers. `calls` contains 1–100 `args`/`kwargs` entries.
Every array occurrence becomes an independent owned copy. Input mutation stays in
the worker; aliases between caller objects are not preserved. Do not mutate inputs
while a request runs. Dictionaries are literal values, including a `$npy` key.
Unsupported layouts are rejected before execution. Optimized calls still scan
every changing floating-point array for nonfinite values and enforce measured
metadata and fixed scalar controls.

The default transport uses bounded raw array frames with copied, owned outputs.
It preserves dtype, shape, array order and raw bits; structured arrays use the
existing NumPy format. Observation reuse removes one duplicate encoding when the
declared observation is the actual result. Input and retained-output snapshots
remain in place. These changes do not change arithmetic or enable tolerances.
The [request-runtime experiment record](cpu-request-runtime-study.md) includes
negative experiments and separate file/live-input measurements.

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
