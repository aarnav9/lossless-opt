# CPU callable usability trials

Run each public checkout at the revision in `docs/cpu-usability-study.md`. The
script copies its package, adds an ordinary workload wrapper and an explicitly
split Python input loader, then runs the public workflow without a candidate plan.
The final public adapter is checked on changing input values. Three projects are
new to this workflow; NumPy Financial is the existing performance control.

Example, from `lossless-product`, in an environment with `[applications]` installed:

```sh
set -o pipefail
python -u experiments/cpu_usability/study.py --kind transforms3d \
  --upstream /path/to/transforms3d --budget 360 --output /tmp/usability-transforms3d \
  2>&1 | tee /tmp/usability-transforms3d.log
```

Other kinds: `filterpy`, `perlin-numpy`, `numpy-financial`. Use separate output/log
paths. Each run has a six-minute workflow cap, logs timestamped stages and prints
result/log paths at startup and completion. There are no model calls. Do not run
performance trials concurrently. The study sets BLAS/OpenMP/Accelerate thread
counts to one for both variants.

`protocol.json` freezes the revision, seed, sizes and deployment objective before
search. `result.json` reports setup-script time, overall time, source selection,
deployment comparisons and exact changing-input checks. Raw pair timings and
rejected options remain in `workflow/deployment/`. Input-loader/source edits are
explicitly counted as two integration files; no upstream algorithms are edited.
Human setup time is unknown. This is agent-operated integration evidence, not a
user study or a promise of speedup for the complete upstream projects.

The 14 October release trials repeat the fixed workloads after tightening the
incumbent guard and serializing adapter requests. Earlier trials are retained
alongside final trials in the evidence archive. Neither workload sizes nor source
candidates were tuned after seeing held-out outcomes. Runtime-only savings compare
against an owned fresh worker per call, not ordinary in-process Python.
