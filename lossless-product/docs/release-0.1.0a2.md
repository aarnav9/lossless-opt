# Lossless 0.1.0a2 — CPU callable alpha

This alpha adds a local Python/NumPy workflow from representative examples to a
callable application bundle. Source, wheel, source archive and SHA-256 checksums
are attached to the [GitHub prerelease](https://github.com/aarnav9/lossless-opt/releases/tag/v0.1.0a2).
PyPI and Homebrew publication remain future distribution options.

## Install

Use a Python 3.10+ environment on macOS or Linux that contains your application's
dependencies. Python 3.11 is the release-validation environment. Native candidates
also need clang; a callable with no native helper does not require compilation.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'lossless-opt[applications] @ https://github.com/aarnav9/lossless-opt/releases/download/v0.1.0a2/lossless_opt-0.1.0a2-py3-none-any.whl'
lossless doctor
```

## Optimize and use your callable

In your project, supply `app.py:run` and an `examples.py:cases` loader returning
plain numeric arrays and JSON values, with explicit discovery/evaluation splits.
The [complete example](../examples/cpu-callable/README.md) includes both files.

```sh
lossless run-project ./my-project --entry app.py:run \
  --input-loader examples.py:cases --deployment auto --expected-requests 100 \
  --budget 5m --output ./lossless-runs/my-project
```

```python
from lossless.applications import CallableApplication

with CallableApplication('./lossless-runs/my-project/application') as run:
    result = run(new_array)
    print(run.info)
```

`auto` permits reusable imports; `oneshot` keeps a fresh process for each call.
Inputs and returns are copied values, with no caller alias/mutation propagation.
Factories reset per request. Keep the context open for the declared number of
requests to amortize startup. Complete-call selection measures the adapter,
checks exact observations and returned bits, and keeps the original when no
candidate qualifies. It makes zero model calls and enables no numerical tolerance.

## Changes and scope

- Python input loaders freeze examples before search. An explicit `CaseRecorder`
  helper captures inputs from existing test helpers before mutation.
- `CallableApplication` accepts normal args/kwargs, returns actual values, and
  owns worker cleanup. Bundles also include a small `open_callable` helper.
- Complete-call discovery selects source/lifecycle combinations. A frozen final
  compares the choice against the same-mode reference, fresh-reference lifecycle
  and a callable-search incumbent, as applicable. Startup and memory are included;
  repeated request positions retain regression gates. Failed finals do not retry.
- This wheel also includes the CPU recipe integration and prepared-request work
  published since 0.1.0a1: compiled helpers, scoped pools/preparation, generic NumPy
  candidates, bounded direct values, raw transport and observation reuse.

The recipe library remains limited. Exactness evidence covers the provided
inputs and observers, not arbitrary Python, undeclared state, external effects,
all future inputs or a universal equivalence proof. An optimized metadata miss
raises before execution; choose an explicit reference session or requalify.
Changed source, dependencies, host or runtime disable optimized admission.
Historical certificates should be requalified under this release.

See the [CPU guide](cpu-projects.md) and [usability trial record](cpu-usability-study.md)
for measurements, negative outcomes and reproduction details. Native and MLX
contracts remain separate; this CPU alpha does not expand GPU hardware support.
