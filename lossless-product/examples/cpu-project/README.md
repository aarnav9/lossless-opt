# Short signal windows

From `lossless-product/`, using the project's Python environment with
`pip install '.[applications]'`:

```sh
python examples/cpu-project/prepare.py --output ./lossless-runs/signals-input
lossless run-project ./lossless-runs/signals-input/project \
  --entry signals.py:summarize --cases ./lossless-runs/signals-input/cases.json \
  --budget 3m --output ./lossless-runs/signals
python ./lossless-runs/signals/application/run.py --case new_short --output ./lossless-runs/signals-output
```

Every output path must be new. This is a small synthetic example to exercise the
workflow, not evidence of general project speedups. The local recipe may retain
the reference if it does not pass the exactness, timing, and resource gates.
Use `from lossless.applications import read_outputs` to recover the returned
dictionary of arrays. See the [CPU project guide](../../docs/cpu-projects.md).
