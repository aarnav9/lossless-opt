# Contributing

Read [EXPERIMENTS.md](EXPERIMENTS.md) before repeating a research approach. Keep experimental runners and bulk results in the ignored `research/` workspace. Add a concise experiment/result pair, including negative or inconclusive results. Product code must install and run without that workspace.

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e './lossless-product[dev]'
python -m unittest discover -s lossless-product/tests -v
python -m build lossless-product
```

Use clang for native checks. The CPU suite is small; substantial model campaigns should be run as explicit foreground Terminal jobs with unbuffered, timestamped progress and a named log. Record hardware, runtime versions, workload, comparator, correctness relation and evidence scope for performance claims.

A new adapter needs an executable contract, independent reference, bounded candidate evaluator, disjoint discovery/evaluation inputs, export guards and fallback behavior. Preserve original precision/weights when requested. Never describe finite tests or real-number algebra as a floating-point equivalence proof. A negative/no-win result is valid product behavior.

For extraction, bring the required runtime code, small fixtures, scoped proof sources, pins and notices into `lossless-product/`. Update the retention inventory. Keep user credentials, model weights, compiled caches and generated runs out of commits and archives. Build and install a wheel outside the checkout before changing the supported installation story.

The private alpha has no selected original-code open-source license yet. Public release also requires broader backend qualification and a fresh full proof-toolchain installation check.
