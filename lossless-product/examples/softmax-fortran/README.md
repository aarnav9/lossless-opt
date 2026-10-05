# Qualified numerical softmax recipe: large Fortran arrays

This opt-in example supplies the frozen campaign-045 first-round kernel through
the ordinary Lossless proposal boundary. It needs **no LLM account, API key or
model weights**. The `llm` field names a deterministic local recipe callback;
there is no model call. Built-in candidates and the frozen recipe are checked
against the job's explicit NumPy comparator on the current machine.

Campaign 046 qualified this source on Apple M2, NumPy 1.26.4 and SciPy 1.16.0.
Three fresh confirmation processes measured 1.990–1.999× bound-call speed and
1.374–1.446× ordinary-call speed against separately screened retained/library
references over six timed distributions. Gaussian-only averages were 1.151×
bound / 1.094× ordinary; structured rows accounted for the larger mixed gains.
The 35×1027 shape had isolated ordinary-call losses, so measure your actual
workload before adoption. Full qualification and limits are in
[the report](../../docs/softmax-046.md). These are numerical CPU softmax results,
not exact MLX or whole-model inference speedups.

With current source installed, from the repository root:

```sh
lossless inspect lossless-product/examples/softmax-fortran/lossless.json
set -o pipefail
python -u -m lossless optimize lossless-product/examples/softmax-fortran/lossless.json --output ./lossless-runs/softmax-fortran 2>&1 | tee lossless-softmax-fortran.log
lossless export ./lossless-runs/softmax-fortran --output ./lossless-runs/softmax-fortran-artifact
```

Allow up to two minutes for the job. Check `report.json`: acceptance is measured
on your machine and is not guaranteed. An accepted artifact can then be used:

```python
import lossless
import numpy as np

operation = lossless.load("lossless-runs/softmax-fortran-artifact")
x = np.asfortranarray(np.random.default_rng(1).normal(size=(66, 2039)).astype(np.float32))
y = operation(x)                  # validates, allocates, and calls
bound = operation.bind(x)         # allocate once for repeated calls
y = bound()                      # output overwritten on the next bound call
```

The supplied job lists **19 specific shapes, all Fortran layout**. Those exact
shape/layout pairs become artifact guards. Small Fortran arrays, C/sliced layouts
and unlisted shapes use the independent reference. Numerical inputs must be
finite rank-two float32 arrays with absolute values at most 10,000 and positive
aligned strides. Keep that domain and the bound arrays' shape/strides valid for
every call. The contract uses numerical tolerances, not bitwise equality.

The broad screen was `rows >= 32`, `columns >= 128`, `rows * columns >= 16384`,
Fortran layout. Passing that screen alone does **not** admit every possible shape;
the exported artifact retains the finite case list. The current native loader
also checks OS platform, architecture and NumPy version, and verifies artifact
hashes. It does not fingerprint every CPU model or runtime component. Rebuild
and remeasure when deploying to different hardware.

Editing the job's cases creates a new local experiment; the campaign-046
qualification does not extend automatically. Default softmax recipes are
unchanged. The source digest is checked before the callback submits it.
