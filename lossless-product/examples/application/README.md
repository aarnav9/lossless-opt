# Bring an existing application

`project/app.py` is a normal NumPy application with a function and a command-line
entry point. It has no Lossless imports. With current source installed, from the
repository root:

```sh
lossless init --project lossless-product/examples/application/project --entry app.py:analyze --cases lossless-product/examples/application/cases.json --output .lossless/application-job
lossless inspect .lossless/application-job/lossless.json
lossless replay .lossless/application-job/lossless.json --output .lossless/application-job/runs/first
```

The replay normally takes a few seconds. It executes the discovery case twice
in fresh copied workspaces and compares output and post-call input fingerprints.
The original application stays unchanged; evaluation cases are not run by default.
Look at `report.html` and `report.json` in the output directory.

To exercise the existing command instead:

```sh
lossless init --project lossless-product/examples/application/project --cases lossless-product/examples/application/command-cases.json --output .lossless/command-job --run python app.py
lossless replay .lossless/command-job/lossless.json --output .lossless/command-job/runs/first
```

`--run` must be last; it receives an argv list, with no shell interpretation.
Both forms perform **reference replay only**, not optimization or a performance
comparison. [Full interface, factory/state hooks and limits](../../docs/applications.md).
