# CPU project workflow validation — 12 October 2026

This study exercises the new model-free `run-project` workflow, including its
runnable delivery stage. [Portable evidence](evidence/cpu-project-workflow.json.gz)
contains frozen configurations, source/fixture identities, proposal/evaluation
records, exact output fingerprints, memory receipts, and delivery observations.
Hashes refer to original records before path redaction. The reproduction driver is
[`check_cpu_project.py`](../scripts/check_cpu_project.py); the
[usage guide](cpu-projects.md) describes the supported product contract.

All runs used the same local Apple Silicon host, Python 3.11, NumPy 1.26.4 and
SciPy 1.16.0. BLAS/OpenMP thread settings were explicitly fixed to one for the
external projects. No provider or model calls were made. Cases were frozen before
profiling; only discovery measurements fed candidate selection.

| Project/workload | Outcome | Interpretation |
| --- | --- | --- |
| pybaselines 1.2.1, independent-spectrum wrapper | Accepted, **1.3083×** case-balanced callable speed; all delivery fingerprints exact | All eight retained combinations were tried. The source-only candidate won selection; the extra pool/compiler/cache variants were not automatically stacked onto it. |
| NumPy Financial 1.0.0, batched rate calculations | Runnable reference retained | No applicable local source recipe. This is successful intake and delivery, with no optimization claim. |
| TMM, thin-film reflectance/transmittance sweep | Stopped at unstable reference | Repeated unmodified runs produced different output byte fingerprints on identical inputs. No candidate or optimized delivery was approved. |
| Small synthetic signal-window example | Runnable reference retained | One guarded reduction candidate matched exactly but did not pass discovery timing gates. Its 0.8768× discovery ratio is not a qualified slowdown or speedup claim. |

The pybaselines workload uses the fresh fixtures from the existing retained
integration driver: discovery shapes 383/1151, held-out shapes 467/1879/887,
three changing calls per case, weighted spectra and branch/order/sort variations.
The callable result passed six paired held-out rounds: every round exceeded the
1.02× minimum, one-sided sign-test p = 0.015625. Resource and setup gates passed.
The complete workflow took 125.0 seconds under its 280-second budget. These are
finite observations for the declared workload; they do not imply gains on arbitrary
repositories or every future input.

## Launch costs are a separate result

The delivered copies were checked in fresh processes on each evaluation case.
These single descriptive pairs include interpreter/environment startup, source
and dependency checks, fixture loading, observation and output serialization:

| Case | Reference launch | Optimized launch | Reference / optimized |
| --- | ---: | ---: | ---: |
| `new_small` | 0.759 s | 1.036 s | 0.733× |
| `new_weighted` | 0.766 s | 0.928 s | 0.826× |
| `new_branches` | 0.718 s | 0.932 s | 0.770× |

**There was no end-to-end launch speedup.** Compute saved fractions of a
millisecond to about a millisecond, while deployment admission/setup cost much
more. A reusable prepared process or an in-process integration boundary is a
follow-up, with its own lifetime, state, memory and exactness checks. The 1.3083×
result must not be advertised as a command-line launch speedup.

NumPy Financial's reference-only delivery took 0.352/0.362 seconds in its
descriptive pair. Its full workflow took 3.4 seconds. The initial fixture wrapper
omitted the release's required `fv` argument; that failed preflight run is also
archived, and the driver now supplies `fv=0`. It was not an optimization result.

TMM's five reference passes all completed, but their output hashes differed.
For example, first-call hashes started `c82328cef5d7` and `eaf09fb0982e` in clean
timing passes. The cause and numerical magnitude were not established here.
This is an exact-repeatability blocker, not evidence of a useful numerical
optimization. The workflow correctly produced no qualified application bundle.

## Reproduce

Use these source revisions, preserving their upstream licenses:

- [pybaselines](https://github.com/derb12/pybaselines/tree/6f8d927e08577a9c5af5f34129b14d57c12725f6): `6f8d927e08577a9c5af5f34129b14d57c12725f6`.
- [NumPy Financial](https://github.com/numpy/numpy-financial/tree/247aa4e2ee4fa7db44915790d5a730b9776d10a7): `247aa4e2ee4fa7db44915790d5a730b9776d10a7` (`v1.0.0`, pure Python).
- [TMM](https://github.com/sbyrnes321/tmm/tree/462b63b517472b3b18245c98d70e700f4d312866): `462b63b517472b3b18245c98d70e700f4d312866`.

From `lossless-product/`, after installing `.[applications]` in the selected
environment, run one project at a time. On the tested host each job finished in
under five minutes; larger projects may need a user-run Terminal session.

```sh
set -o pipefail
python -u scripts/check_cpu_project.py --kind pybaselines \
  --project /path/to/pybaselines-v1.2.1 --budget 280 \
  --output ./lossless-runs/cpu-project-pybaselines \
  2>&1 | tee ./lossless-runs/cpu-project-pybaselines.log
```

Create `lossless-runs/` before using `tee`. Repeat with `--kind numpy-financial`
and `--kind tmm`, their source paths, and distinct output/log paths. The script
prints timestamped progress and result/log paths at startup and completion.
The [synthetic example](../examples/cpu-project/README.md) runs through the public
CLI. Expect hardware, NumPy builds and background load to change results.

Product regression tests separately cover accepted delivery, new array values,
metadata fallback, modified sources, changed dependencies, relocation, saved
returns, rejected wrong candidates, unstable references, explicit state observers,
and no retry after partial execution. The deliberately delayed test callable is
a deterministic routing test, not performance research evidence.
