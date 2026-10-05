# Campaign 046: scoped softmax qualification

Freeze the strongest prior sources, screen 75 shape/layout combinations over
three fresh processes, then confirm exactly one selected candidate/scope on new
shapes. Eleven distributions validate numerical output, input immutability and
buffer guards. Six distributions time both ordinary allocating application calls
and prebound calls. A separately screened retained/library reference is frozen
before each timing confirmation. The source and thresholds cannot change during
the study.

**Completed:** `first_045` / `large_f` was the only passing discovery pair.
Nine untouched Fortran shapes passed confirmation in three fresh processes.
The product workflow accepted the opt-in recipe and real export/load guards
passed. [Results, limitations and evidence](../../docs/softmax-046.md).

Expect roughly 8–15 minutes for qualification on the recorded Apple M2. Use a
fresh directory and matching named log. No model calls or downloaded weights:

```sh
python -m pip install './lossless-product[numerical]'
export PYTHONPATH="$PWD/lossless-product/src"
set -o pipefail
python -u lossless-product/experiments/softmax_046/qualify.py --output research/results/softmax_046_new 2>&1 | tee research/results/softmax_046_new.log
```

The qualification harness requires SciPy as well as NumPy and clang. Its frozen
candidate sources use Arm NEON; the recorded replay target is Apple M2, not an
x86 machine. The smaller opt-in product example itself needs only the package's
ordinary NumPy dependency.

The experiment exits without opening final shapes if no discovery scope passes.
It never tries a second candidate after final failure. The broad scope predicate
is a screening condition, not proof for every possible shape. The ready recipe
exports only the explicit 19 validated Fortran shapes:

```sh
python -u -m lossless optimize lossless-product/examples/softmax-fortran/lossless.json --output research/results/softmax_046_deployment_new 2>&1 | tee research/results/softmax_046_deployment_new.log
python -u lossless-product/experiments/softmax_046/deploy.py --run research/results/softmax_046_deployment_new --output research/results/softmax_046_export_new 2>&1 | tee research/results/softmax_046_export_new.log
```

The second command expects this recipe to have been accepted; on another device,
read the product report first. It checks real export/load, native dispatch for all
19 shapes, numerical correctness over 11 distributions, reference fallback and
out-of-domain rejection. It is not a new candidate search.

To curate a completed run, use `scripts/curate_046_047.py 046 --study DIRECTORY
--extra DEPLOYMENT_DIRECTORY --extra EXPORT_DIRECTORY --output BUNDLE.json.gz`.
The archive contains frozen source, raw timing/validation records and deployment
records, with local paths redacted and compiled binaries omitted.
