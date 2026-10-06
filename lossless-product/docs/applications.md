# Existing-application intake and reference replay

Current source can inspect a **local project**, detect the selected Python
environment, generate a JSON job, and replay an existing function/model object
or command in fresh copied workspaces. This implements the intake/replay stage
of the [eight-part application plan](implementation.md#current-priority-bring-an-existing-application-5-october-2026).
It is not in the published `v0.1.0a1` wheel.

[Callable profiling and optional model assessment](application-profiling.md) are
also available. An explicit [source-search plan](application-search.md) can enable
an experimental author/test/revise loop. Fully automatic optimization, guarded
deployment and LLM-assisted setup remain later work. A successful replay reports
`replayed`, never `accepted`.
The existing native/MLX optimization jobs continue to use their original schema.

## User workflow

Activate the Python environment that normally runs the application and install
current Lossless source there. Alternatively supply `--python /path/to/venv/bin/python`.
Intake/replay does not install dependencies, activate environments, download
models, upload repositories or call an LLM. Profiling can optionally send bounded
evidence/source excerpts to a configured assessment provider.

```sh
lossless init --project /path/to/project --output .lossless/my-application
lossless inspect .lossless/my-application/lossless.json
```

This produces a draft with discovered function candidates and missing setup
instructions. Discovery parses up to 200 Python files of at most 256 KiB each;
application source is scanned before documentation/examples/tools, then tests,
using a deterministic path heuristic. Other candidates remain available within
the scan limit. It does not import the application or infer representative inputs. Select the
entry point and provide inputs to create a runnable job directly:

```sh
lossless init --project /path/to/project --entry package.module:run --cases /path/to/cases.json --output .lossless/my-ready-application
lossless inspect .lossless/my-ready-application/lossless.json
lossless replay .lossless/my-ready-application/lossless.json --budget 2m --output .lossless/my-ready-application/runs/first
```

Use a fresh output directory for each initialization/replay. `--entry` also accepts
`app.py:run`. The callable must be defined in the copied project; `src/` package
layouts are supported. The generated job can instead be edited directly.

`inspect` reports selected environment, file/Git provenance, entry candidates,
case counts, total/per-process replay limits, repeat count, reference and the
scope of observation. It executes only a bounded standard-library environment
probe in the selected interpreter, never the project's imports or tests.
The probe disables Python startup hooks and reads virtual-environment/package
metadata without executing `.pth` or `sitecustomize` code.
Installed-package metadata does not certify that application dependencies or GPU
execution work; explicit replay establishes whether the declared invocation runs.
Use `lossless inspect JOB --json` for the full machine-readable intake record.

## Files and configuration

Initialization writes `lossless.json`, `cases.json`, `environment.json` and
`discovery.json`. Environment/discovery files describe intake; replay obtains a
fresh environment report and freezes the current source/cases independently.
Changes between intake and replay are allowed. Changes while freezing/replaying
invalidate that replay's conclusion.

The executable application schema is separate from native/MLX search jobs:

```json
{
  "schema_version": 1,
  "kind": "application",
  "name": "my-application",
  "project": {"root": "../../my-project", "exclude": []},
  "environment": {"python": "/absolute/path/to/venv/bin/python", "inherit_env": []},
  "entry": {"kind": "callable", "target": "app.py:run", "factory": false},
  "cases": "cases.json",
  "replay": {"repeats": 2, "timeout_seconds": 30, "wall_time_seconds": 120, "seed": 0, "max_output_bytes": 8388608}
}
```

Relative project/cases paths resolve from the job directory. Callable and fixture
paths resolve inside the copied project. Interpreter symlinks are preserved so a
virtual environment keeps its own identity. The application replay budget is
separate from optimization budgets; there is no author response timeout here.

Callable case input:

```json
[
  {"id": "small", "split": "discovery", "args": [[1, 2, 3]], "kwargs": {"scale": 0.5}},
  {"id": "array", "split": "evaluation", "args": [{"$npy": "data/sample.npy"}], "kwargs": {}}
]
```

Arguments support nested JSON values and explicit `$npy` file references.
NumPy fixtures load with `allow_pickle=False`. Explicit `$npy` fixtures are
included in the snapshot even when Git ignores them. For command inputs or other
resource files, use repeatable `--include data/file` options or `project.include`
in the job. Includes select individual files inside the project, remain subject
to snapshot limits, and cannot override secret/cache exclusions or symlink
restrictions.

Each case has an explicit split;
replay runs only discovery by default. `--split evaluation` or `--split all`
explicitly opens those inputs. Public/user-supplied cases are not inherently
hidden holdouts, and replay does not certify statistical independence.

## Model objects, state and lifecycle

Use `--factory --entry app.py:load_model` when a no-argument function constructs
a callable model/object. Construction happens once per case/repeat, followed by
the case's calls. Constructors needing inputs can set `entry.factory_args` and
`entry.factory_kwargs` in the JSON job; these support the same argument/fixture
format as calls. An explicit entry may also use an attribute path such as
`package.module:processor.run`. A sequence can exercise persistent state within a case:

```json
[{"id": "continuation", "split": "discovery", "calls": [
  {"args": [[1, 2]], "kwargs": {}},
  {"args": [[3, 4]], "kwargs": {}}
]}]
```

Each repeat reloads the project in a new process/workspace. Python and NumPy RNGs
are seeded before application import. Other framework RNGs, data acquisition,
external services and nondeterministic kernels require explicit setup in the
user's existing driver. They are not silently made deterministic or tolerated.

Default observers capture the return value and pre/post-call arguments. Supported
outputs are JSON-like structures, bytes and dense NumPy/MLX/PyTorch tensors;
array values are hashed with shape/dtype metadata. MLX values are materialized
and PyTorch tensors copied to CPU for observation. This can initialize a device
and adds overhead. Sparse/quantized tensors, custom objects and hidden state need
an explicit observer. No GPU framework is installed or imported just for discovery.

Optional entry hooks:

```json
{"kind": "callable", "target": "app.py:load_model", "factory": true,
 "observe": "driver.py:observe", "cleanup": "driver.py:cleanup"}
```

`observe(target, result, args, kwargs)` returns the observable JSON/array tree,
which can include state. `cleanup(target)` runs after the call sequence, also on
a call/observation exception; a killed or crashed worker cannot run cleanup.
The target remains the same within a sequence. Input mutations are recorded,
not automatically forbidden. Aliasing, gradients, exception equivalence and
undeclared hidden state are not certified by these replay observations.

## Existing commands

```sh
lossless init --project /path/to/project --cases command-cases.json --output .lossless/command-job --run python app.py --mode batch
```

All Lossless flags precede `--run`; subsequent arguments belong to the program.
There is no shell expansion. `python`/`python3` at the start of the command use
the selected interpreter. Other executables run as supplied. If no cases file is
given, the explicitly supplied command itself becomes one discovery invocation
with empty stdin; no extra workload variants are invented.

Command cases use `argv` (additional arguments) and `stdin` (a string):

```json
[{"id": "batch", "split": "discovery", "argv": ["--size", "16"], "stdin": "sample\n"}]
```

The default observation is successful exit plus exact stdout bytes. Set
`entry.outputs` to workspace-relative output filenames to observe their contents.
Set `entry.observe_stdout` to false for diagnostic/timestamp output only when
output files are declared. Stderr is retained for diagnosis, not compared.
Nonzero exit, missing output, output-limit violations and timeouts fail replay.

## Provenance, execution and reports

Git projects use tracked plus nonignored untracked files; ordinary directories
use a directory walk. An explicitly selected directory ignored entirely by an
enclosing repository also uses a directory walk; the report records this and
does not attribute its source to the enclosing Git revision.
Credentials/config directories, `.env` files, environments,
builds and caches are excluded by default. `--exclude`/`project.exclude` adds
project-relative paths or globs. Source symlinks and submodule directories must
be excluded or supplied as ordinary files. Default snapshot limits are 10,000
files and 256 MiB, configurable under `project.limits.max_files/max_bytes`.
Large weights/external datasets are not magically captured: package a bounded
fixture, raise the explicit snapshot limit, or treat external dependencies as
outside this replay's reproducibility claim.

Each run stores a hashed source snapshot, resolved configuration, selected cases,
environment report, runner hashes, attempt receipts and stdout/stderr logs.
Source is copied again into a fresh temporary workspace per repeat; those working
copies are removed. The frozen reference and reports remain. Known credentials
are not inherited; explicitly required environment variable **names** go in
`environment.inherit_env`. Values are not stored. Each worker gets a fresh HOME
and temporary directory. Source and log files are local and can contain user data.

This executes **trusted local code**, not a sandbox for hostile repositories.
Copied workspaces isolate normal relative file writes, not arbitrary absolute
paths, external databases or network side effects. Child processes in the worker
process group are terminated on completion/timeout. Detached processes and
external state require an appropriate user-controlled execution environment.

Replay uses a sleep-inclusive deadline on macOS/Linux, timestamped progress per
case/repeat and during quiet waits, and reports result/log paths. Output limits
are monitored, not a hard filesystem quota. Timings include startup, setup and
observation; they are diagnostics, not synchronized application benchmarks.

Results are `replayed`, `reference_unstable`, `failed`, or `incomplete`. Even
`replayed` establishes only exact agreement of the **observed reference repeats**;
it does not validate an optimized program or prove all-input equivalence. A
failed/unstable command exits nonzero. `profile` supports callable application
jobs. Use `search-application` with a source plan for candidate experiments;
`optimize` and guarded `export` still reject application jobs/replay results.

[Runnable example](../examples/application/README.md). Python callers can use
`lossless.applications.initialize(...)`, `lossless.applications.inspect(...)`,
and `lossless.replay(config, output=..., repeats=2, budget=120)`.

[External repository validation](application-intake-study.md) records successful
replay of three unmodified pybaselines algorithms across 18 synthetic input cases,
plus an upstream validation-error control. This is intake evidence, not a speedup.
