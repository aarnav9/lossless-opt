# CPU recipe coverage comparison

Five workloads, each run once with the published `0.1.0a3` runtime and once with
expanded recipes in `0.1.0a4.dev0`. Cases, wrappers, seeds, order and budgets were
frozen before any comparison. There are no model calls, manual candidate plans,
numerical tolerances or retries after failed final evaluation.

The campaign preceded the coverage option. The current harness explicitly requests
expanded coverage when the runtime supports it; a3 retains its original behavior.
The evidence also stores the exact pre-option sources and their hashes. The
curation script verifies those campaign hashes, so curating the historical run
requires its frozen source snapshot, not a later modified checkout.

The FilterPy outer-product and particle-resampling workloads exercise axis
reductions and cumulative sums. Ensemble Kalman filtering additionally exercises
profiled class methods. Perlin noise tests expression arguments and axis reductions;
NumPy Financial is the existing source-performance control. These are application
workloads from three repositories, not arbitrary-repository performance claims.
They are agent-operated tests, not independent developer onboarding evidence.

## Reproduce

Use Python 3.11 with NumPy, SciPy, psutil and threadpoolctl. The evidence records
exact installed versions, source hashes, CPU and upstream revisions. Run on an
otherwise quiet machine; preserve rejected outcomes and complete-request gates.

Clone these upstreams into a directory outside this checkout and check out:

| Repository | Revision |
| --- | --- |
| https://github.com/rlabbe/filterpy | `3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33` |
| https://github.com/pvigier/perlin-numpy | `5e26837db14042e51166eb6cad4c0df2c1907016` |
| https://github.com/numpy/numpy-financial | `247aa4e2ee4fa7db44915790d5a730b9776d10a7` |

Download the wheel and `SHA256SUMS` from
[the a3 release](https://github.com/aarnav9/lossless-opt/releases/tag/v0.1.0a3),
verify the checksum, and extract it into `./baseline-runtime` with
`python -m zipfile -e lossless_opt-0.1.0a3-py3-none-any.whl ./baseline-runtime`.
Using `PYTHONPATH` chooses the runtime without changing dependencies between runs.

From the `lossless-product` folder, run each pair sequentially in a foreground
Terminal. This example runs the outer-product pair; repeat for
`filterpy-resample`, `filterpy-ensemble`, `perlin-numpy`, and `numpy-financial`,
using the corresponding upstream folder. Each pair took about one to three
minutes on the original host. Use a fresh output location.

```sh
set -o pipefail
study_kind=filterpy-outer
upstream_dir=/absolute/path/to/upstreams/filterpy
results_dir=/absolute/path/to/new-results
baseline_dir=/absolute/path/to/baseline-runtime
mkdir -p "$results_dir"
for runtime in baseline expanded; do
  runtime_dir="$baseline_dir"
  if [ "$runtime" = expanded ]; then runtime_dir="$PWD/src"; fi
  result_dir="$results_dir/$study_kind-$runtime"
  PYTHONPATH="$runtime_dir" python -u experiments/cpu_recipe_coverage/study.py \
    --kind "$study_kind" --upstream "$upstream_dir" --budget 150 \
    --output "$result_dir" 2>&1 | tee "$result_dir.log"
done
```

Each run prints timestamped progress and result/log paths at startup and completion.
The harness copies the upstream package and writes two integration files. Its
script setup duration is not human setup time. Inputs are split before search;
evaluation is never used to author new candidates. Deployment compares against
both equally prepared original source and the frozen incumbent when applicable,
with startup, communication, output handling and memory included in admission.

`result.json` reports the delivered variant and verifies fresh finite input values
against the reference through the public adapter. `workflow/report.json`,
`workflow/plan.json`, source-search records and deployment evaluation preserve the
actual decisions and raw measurements. See the
[result report](../../docs/cpu-recipe-coverage-study.md).
