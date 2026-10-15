# CPU callable onboarding trials — 14 October 2026

The 0.1.0a2 workflow accepts a Python input loader, freezes explicitly split
examples, selects using complete callable costs, and produces an ordinary
`CallableApplication` interface. Four external project trials completed through
that public path with no model calls, saved candidate plans, or upstream algorithm
edits. Three projects were new to this workflow; NumPy Financial was the existing
performance control. These are bounded workloads, not whole-repository benchmarks.

[Portable evidence](evidence/cpu-usability.json.gz) includes initial and release
trials, protocols, raw paired timings, patches, source/fixture/runtime hashes,
environments and upstream licenses. [Reproduction script](../experiments/cpu_usability/study.py)
· [Commands](../experiments/cpu_usability/README.md) · [User guide](cpu-projects.md).

## Final release-code results

| Workload | Delivered source / lifecycle | Result |
| --- | --- | --- |
| NumPy Financial, batched rate calculations | Combined candidate / prepared | **1.3546× complete warm requests** against equally prepared original source; **1.2902×** including startup/cleanup amortized over 100 requests. All required gates passed. |
| transforms3d, quaternion to axis-angle conversion | Original / fresh worker | Callable-only source search accepted a candidate. Complete-cost discovery preferred prepared original source, but final warm requests were **0.8674×** relative to the incumbent. Rejected by the incumbent guard; no final retry. |
| FilterPy, repeated discrete Bayesian updates | Original / prepared | No applicable source recipe. Reuse qualified and saves repeated worker startup; it does not accelerate the arithmetic. |
| perlin-numpy, fractal noise over changing canvases | Original / fresh worker | No applicable source recipe. Reuse was faster but exceeded the **1.2× memory cap**, so fresh reference execution was retained. |

All four delivered adapters matched explicit reference outputs on new finite
values with the same metadata. Exact checks include observed input mutations and
retained results within the declared copied-value contract. Caller aliases and
input mutation are not preserved by that contract. No numerical tolerance or
precision change was enabled.

Financial's two final comparisons had sign-test p=0.015625, each below its 0.025
threshold. Its selected source was also the frozen callable-search incumbent.
The 1.3546× includes validation, serialization, communication, observations and
returned values. The 100-request figure projects measured startup/cleanup over
100 calls; it is not a measured 100-request session. Held-out shapes were (97, 8)
and (121, 8). Earlier width-1/32 evidence remains in the prior request-runtime
study; this trial does not extend that evidence.

FilterPy's 52.9562× amortized lifecycle ratio compares against starting an owned
Python worker **for every request**. It mainly removes Python/SciPy startup and
is **not** a speedup over ordinary in-process FilterPy. Financial's analogous
cold-to-prepared ratio was 3.8005×; its source improvement is the separate
1.2902× / 1.3546× comparison above.

## Setup, methodology and negative results

Each project needed two integration files: a workload wrapper and an input
loader. No manual JSON/NumPy fixtures or upstream algorithm edits were required.
Final full workflow times on the local macOS host were 22.88 s (transforms3d),
30.48 s (FilterPy), 11.99 s (perlin-numpy), and 35.10 s (Financial). This is
agent-operated integration evidence. Human setup time and independent-user
usability remain unmeasured.

One discovery case and two evaluation cases were frozen before each search.
Deployment discovery used two rounds and two requests per case; final validation
used six paired rounds and three requests per case, alternating option order and
rotating case order. Both amortized cost and raw request positions have regression
gates. The same-mode original and frozen incumbent are comparisons, as applicable;
the distinct final comparisons share a 5% significance budget. A failed final
never starts another selection round. BLAS/OpenMP/Accelerate use one thread.

The initial pass is preserved. It retained prepared original source for
transforms3d and FilterPy, fresh original source for perlin-numpy, and Financial's
1.3521× warm / 1.2881× amortized source gain. Release trials repeated fixed inputs
after making incumbent comparison mandatory even when discovery prefers original
source, and serializing adapter calls. Cases, numerical policy, thresholds and
source candidates were not tuned using held-out performance. These repeated
trials are not independent replications. The final negative transforms3d outcome
is retained.

Tests cover loader errors/source mutation, explicit splits and unsupported
layouts/types, original-source preservation, captured inputs before mutation,
kwargs/returns, independent outputs, missing/mismatched certificates, metadata
rejection and explicit reference escape, modules that cannot reuse imports,
errors/timeouts, and request-position regression rejection. Existing tests cover
changed environments and copied-source integrity. Release checks install the
actual wheel outside the checkout and exercise Python inputs, deployment
qualification and callable results.

The local full suite ran 241 tests with 8 optional skips before two final controls
and guard refinement; the final 12-test onboarding suite passed after those
changes. GitHub runs the expanded full suite on Linux and macOS. See
[validation metadata](evidence/cpu-usability-validation.json).

## Frozen upstream versions

- [transforms3d](https://github.com/matthew-brett/transforms3d/tree/6a43a98e3659d198ff6ce2c90d52ddef50fcf770): `6a43a98e3659d198ff6ce2c90d52ddef50fcf770`.
- [FilterPy](https://github.com/rlabbe/filterpy/tree/3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33): `3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33`.
- [perlin-numpy](https://github.com/pvigier/perlin-numpy/tree/5e26837db14042e51166eb6cad4c0df2c1907016): `5e26837db14042e51166eb6cad4c0df2c1907016`.
- [NumPy Financial](https://github.com/numpy/numpy-financial/tree/247aa4e2ee4fa7db44915790d5a730b9776d10a7): `247aa4e2ee4fa7db44915790d5a730b9776d10a7`.

Remaining gaps: an independent developer using the release on their own project,
and broader optimization coverage. These trials found no new qualified source
improvement outside the known Financial control.
