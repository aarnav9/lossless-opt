# Pybaselines application source search

[The broader follow-up](application-broader-study.md) changes the workload,
baseline and source scope: it finds a larger request-time gain but fails its
allocation gate. It does not supersede this narrower accepted result.

**Result: one scoped, model-authored application-call improvement.** A frozen
candidate reached **1.05848×** the unchanged single-thread reference on fresh
held-out 32-call sequences, approximately **5.52% less call time**. All exact,
startup and memory gates passed. The earlier single-call/default-thread pilot
retained the reference. Neither result establishes an arbitrary-code speedup.

This study connects the existing intake/profile path to an actual model-authored
source change. The target is `pybaselines.polynomial.modpoly` from BSD-3-Clause
pybaselines 1.2.1, pinned to
`6f8d927e08577a9c5af5f34129b14d57c12725f6`. The upstream checkout remains unchanged.
The application boundary is an ordinary public callable/declared sequence, with
imports reported separately; this is not a whole-process or arbitrary-repository
speedup claim. See the [workflow](application-search.md).

## Frozen author loop

One real subscription-backed GPT-6 Astra author, at **medium reasoning**, received
fresh discovery profiles, relevant source, remaining time and previous discovery
feedback. It returned four implementations, with no tools and no evaluator or
final-input access. There were no human edits to these candidate bodies.

Only `_Polynomial.modpoly`'s executable body could change. The public entry,
signature, decorators, source outside that body, numerical parameters, observers,
comparison policy and case split were fixed. All baseline arrays, returned
weights/coefficients/tolerance histories, input mutation and retained previous
outputs are compared exactly. Finite observed agreement is not a universal proof.

The 30-minute limit reserved ten minutes for final evaluation, with a four-candidate
cap. The first run ended at that cap in **240.0 seconds**, including **94.1 seconds
of provider time**; no timeout occurred. It used 153,392 input and 3,147 output
tokens (1,154 reported reasoning-output tokens), with unknown subscription dollar
cost. Finishing early does not show that more search time would be useless.

## Single-call pilot: no accepted change

Six discovery workloads covered short/long spectra, an eight-spectrum sequence,
weights/returned coefficients, peak masking/use-original and a flat spectrum.
Each candidate passed every discovery fingerprint comparison. None passed every
speed gate, so final cases remained unexecuted and the reference was retained.

| Candidate | Change | Discovery aggregate ratio | Gate failure |
| --- | --- | ---: | --- |
| 1 | Reuse clipping, weighted-input and alternating baseline buffers | 0.978× | Overall speed and weighted case |
| 2 | Specialize the two vector norms inside convergence | 1.399× | Weighted case |
| 3 | Specialize norms and bind repeated attributes locally | 1.130× | Weighted case |
| 4 | Combine specialized norms with limited buffer reuse | 1.208× | Flat case |

**Those aggregate ratios are not credible speedup claims.** Timings of these tiny
first calls were noisy. In candidate 2's long case, reference samples were
5.584, 1.022 and 62.428 ms while candidate samples were approximately 0.8 ms.
The paired statistic therefore exaggerated the apparent improvement.

An eight-pair **identical-source A/A control** on long/weighted/flat cases
returned a spurious 1.051× aggregate ratio; its fixed-sample sign-test p-value was
0.637, and the full gate rejected it. The environment's NumPy OpenBLAS pool had
eight threads. Repeating A/A with `OPENBLAS_NUM_THREADS=1` made long-call timings
much steadier but still produced a 1.032× aggregate ratio from first-call noise
(p=0.855; rejected). All three control inputs retained the original bits across
thread settings. Thread count is a plausible contributor, not a proven exclusive
cause of the noise.

## Separate repeated-work qualification

The next protocol freezes **32 explicit calls per case**, including the first,
and single-thread OpenBLAS for both original and candidate. There are no hidden
warmups. This represents repeated spectral processing in an already-running
application; it does not retroactively turn the single-call pilot into a win.

All four exact, frozen Astra proposals are remeasured with **zero new model
calls**. Selection uses six fresh pairs on each of six discovery cases. One winner
is then frozen before twelve fresh pairs on each of eight unexecuted final cases.
Startup, traced allocation and RSS gates still apply. A separate final pass checks
the original eight-thread reference for exact agreement on all fourteen profiles.
Final failures cannot trigger another author or candidate choice.

Candidate 3 won discovery at 1.04822× and then passed final validation at 1.05848×.
It preserves arithmetic order while removing general-purpose vector-norm dispatch
and repeatedly looked-up attributes. It does not add caches or reuse output arrays.
The eight final profiles used **192 fresh timing processes / 6,144 invocations**;
all paired output, input and retained-output fingerprints agreed. All twelve
case-balanced rounds exceeded 1.02× (one-sided sign-test p = 0.000244).

| Final profile | Original median sequence (ms) | Candidate median sequence (ms) | Paired speedup |
| --- | ---: | ---: | ---: |
| Odd small, order 1 | 5.298 | 4.820 | 1.10960× |
| Odd large, order 5 + coefficients | 13.208 | 12.505 | 1.04837× |
| Float32 input | 6.363 | 5.962 | 1.07220× |
| New eight-spectrum pattern | 9.443 | 8.795 | 1.06215× |
| Descending x + weights/coefficients | 7.909 | 7.486 | 1.07205× |
| Peak masking, order 4 | 9.398 | 9.061 | 1.03647× |
| Use-original + weights | 13.076 | 12.231 | 1.05716× |
| Zero spectrum | 3.059 | 2.986 | 1.01252× |

Each sequence has 32 calls; the eight-spectrum pattern cycles through its eight
distinct inputs four times. Paired speedups are geometric means of per-pair ratios,
so they need not equal the ratio of the displayed medians. These are synthetic
spectra, not a production-frequency-weighted benchmark.

Separate memory passes passed on all cases: maximum RSS increase was **1.65%**,
maximum traced-peak increase **4.80%**. The largest median import/factory setup
increase was **1.55 ms**. Default eight-thread outputs also matched all fourteen
single-thread reference profiles, including returned coefficients and histories.
The complete qualification plus that cross-runtime check took **415.35 seconds**;
the search controller portion took 403.01 seconds. Source, inputs, runner and
frozen candidate identities remained unchanged.

The final package also includes a post-measurement fix keeping the public Python
`applications.search` export callable after importing its implementation module.
The measured worker, comparisons and candidate arithmetic are unchanged; the
recorded runner hashes preserve the exact measurement version.

**Payback is substantial.** The controller's own measured qualification cost
requires roughly **477,000–5.55 million 32-call sequences** to repay, depending on
the case. Including the first author pilot, diagnostic worker time, aborted setup
and final cross-runtime check raises the measured total to at least **747 seconds**
and about **884,000–10.30 million sequences** (28.3–329.6 million individual calls).
That excludes development time and control orchestration overhead. Small savings
on sub-millisecond calls favor a reusable qualified recipe over repeating this
search for every occasional workload.

The [machine-readable evidence](evidence/application-search-pybaselines.json)
includes both protocols, every paired sample, receipts, hashes, failures, controls,
memory, startup and costs. This is a successful **source experiment**, not a guarded
deployment artifact. The original upstream checkout was not edited.

The source implementations and original BSD notice are included as
[frozen candidates](../experiments/application_intake/frozen-candidates.json) and
[license](../experiments/application_intake/PYBASELINES_LICENSE.txt).

An initial qualification launch stopped during setup at 24.6 seconds because a
model-context limit was incorrectly applied to saved proposals; it made no model
calls and evaluated no candidate/final case. The corrected path removes that
irrelevant gate. Live assessment also now summarizes repeated allocation stacks
while retaining their positions and full peak range.

## Reproduce without a model subscription

Use Apple Silicon and the recorded NumPy/SciPy environment to compare these numbers.
Other systems can run the experiment, but must establish their own result.

```sh
python3.11 -m venv .venv-pybaselines
.venv-pybaselines/bin/python -m pip install -e ./lossless-product numpy==1.26.4 scipy==1.16.0 numba==0.61.2
mkdir -p .lossless/external
git clone --depth 1 --branch v1.2.1 https://github.com/derb12/pybaselines .lossless/external/pybaselines-v1.2.1
set -o pipefail
.venv-pybaselines/bin/python -u lossless-product/experiments/application_intake/qualify_study.py \
  --project .lossless/external/pybaselines-v1.2.1 \
  --candidates lossless-product/experiments/application_intake/frozen-candidates.json \
  --output .lossless/external/pybaselines-qualification-repro \
  2>&1 | tee .lossless/external/pybaselines-qualification-repro.log
```

Allow approximately 6–9 minutes on the recorded machine; the configurable total
limit is 30 minutes. Result/log paths and timestamped progress are printed.
Choose a fresh output directory for every run. No API key, Codex subscription or
model download is needed to remeasure the frozen proposals.

To run a new live author loop instead, use `search_study.py --project ... --output
...` through the same unbuffered/`tee` pattern, after signing into Codex with
ChatGPT. It defaults to Astra/medium and a 30-minute limit. New model outputs are
not expected to reproduce the exact source proposals above.

`timing_control.py --reference-run RUN --output CONTROL --blas-threads 1` runs
the identical-source discovery diagnostic on a saved search run. It is not a
candidate-selection or final-validation step.

These changes are local source work, not part of the published `v0.1.0a1` wheel.
Guarded automatic deployment and broader repository qualification remain open.

Validation of the completed source: 128 product tests ran (127 passed, one
skipped). A separate copy of the selected candidate passed upstream
`tests/test_polynomial.py::TestModPoly` (21 passed, one skipped) under pytest 8.3.5.
Upstream tests add compatibility coverage; their tolerances are separate from
Lossless's bitwise comparisons.
