# Python inputs to a callable application

From the repository root, in your application's environment:

```sh
python -m pip install './lossless-product[applications]'
lossless run-project lossless-product/examples/cpu-callable/project \
  --entry signals.py:energy --input-loader examples.py:cases \
  --deployment auto --expected-requests 100 --budget 5m \
  --output /tmp/lossless-callable-demo
```

Choose a new output directory for every run. `auto` explicitly permits imports to
be reused between requests; use `oneshot` if your application needs fresh imports.
The count is the expected requests per adapter lifetime, used to account for
startup and cleanup. It is an estimate, not a measured 100-request benchmark.

```python
import numpy as np
from lossless.applications import CallableApplication

with CallableApplication("/tmp/lossless-callable-demo/application") as energy:
    result = energy(np.ones((64, 64)))
    print(result)
    print(energy.info)
```

There may be no faster source candidate. A reference bundle is still runnable.
The complete-call report distinguishes faster computation from reusable-worker
savings. Outputs and inputs cross a **copied-value boundary**; mutations and
aliases do not propagate back to the caller. A factory resets for every request;
use `run_calls` explicitly for a sequence sharing a factory.

An optimized session rejects unmeasured metadata before running. Open
`CallableApplication(path, variant='reference')` for such requests. Failures and
timeouts are never retried. Close the context to release the worker. See the
[CPU guide](../../docs/cpu-projects.md) for supported values and requalification.
