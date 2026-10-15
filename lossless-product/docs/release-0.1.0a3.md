# Lossless 0.1.0a3 — CPU correctness patch alpha

This patch fixes callable lifecycle admission and exact return-type checking.
The [GitHub prerelease](https://github.com/aarnav9/lossless-opt/releases/tag/v0.1.0a3)
contains the wheel, source archive, checksums and a validation receipt.

```sh
python -m pip install 'lossless-opt[applications] @ https://github.com/aarnav9/lossless-opt/releases/download/v0.1.0a3/lossless_opt-0.1.0a3-py3-none-any.whl'
lossless doctor
lossless run-project ./my-project --entry app.py:run \
  --input-loader examples.py:cases --deployment auto --expected-requests 100 \
  --budget 5m --output ./lossless-runs/my-project-a3
```

Use Python 3.10+ on macOS or Linux with the project's dependencies installed.
Python 3.11 is the CI validation environment. Native helpers additionally need
clang. See the [CPU guide](cpu-projects.md) and
[complete callable example](../examples/cpu-callable/README.md).

```python
from lossless.applications import CallableApplication

with CallableApplication('./lossless-runs/my-project-a3/application') as run:
    result = run(new_array)  # your callable's normal arguments
    print(run.info)
```

## Fixes

- Import reuse requires a matching callable certificate and unchanged environment,
  including reference-only bundles. Changed dependencies or declared settings
  select fresh reference workers before application calls. The fallback reason
  persists across requests. Qualification includes the same startup check.
- Missing, malformed or incompatible certificates use fresh reference execution,
  including when the caller requests prepared mode.
- Exact observations distinguish NumPy scalars, zero-dimensional arrays and
  Python scalar values. A candidate changing that return type is rejected even
  if its bytes match. Diagnostics identify the type change.
- `doctor` and generated bundle instructions describe the callable workflow.
  Python input loaders no longer emit the duplicate-module runpy warning.

## Upgrade and validation

Requalify existing bundles in a new output directory after installing this
runtime. Old certificates cannot authorize the updated implementation. Keep the
context open for repeated calls; inputs and outputs cross a copied-value boundary,
so caller aliases and input mutations do not propagate. Unsupported optimized
metadata raises before execution. There is no retry after execution begins.

The local review ran 249 tests with eight optional skips, followed by 17 callable
tests after the final startup-accounting change; all passed. The release receipt
records final CI and installed-wheel checks.

One rerun of the existing NumPy Financial workload completed in 37 seconds with
no model calls. Against the equally prepared original, complete warm requests
were **1.3535× faster**, or **1.3300×** including startup amortized over 100 requests.
All required performance/memory gates and exact checks passed, including fresh
input values through the public adapter. This is finite evidence on the existing
workload and machine, not a universal speedup or an independent developer study.
[Review and measurements](evidence/cpu-alpha3-review.json.gz) retain the raw paired
measurements; reproduce with `experiments/cpu_usability/study.py --kind numpy-financial`
and the upstream revision recorded in its protocol.

The prior [0.1.0a2 release](release-0.1.0a2.md) remains immutable. This patch does
not add numerical relaxation, GPU targets or package-registry publication.
