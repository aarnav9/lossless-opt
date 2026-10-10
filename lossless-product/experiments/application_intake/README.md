# External application intake: pybaselines

This runs the existing `asls`, `modpoly` and `snip` functions from a clean,
pinned pybaselines checkout through Lossless application intake/replay. It does
not modify upstream code, call an LLM, search candidates or benchmark a speedup.
The numerical input generator is ours; no external dataset is downloaded.

Run from the Lossless repository root with current Lossless source installed in
the active Python environment. Python 3.11.3, NumPy 1.26.4, SciPy 1.16.0 and
optional Numba 0.61.2/llvmlite 0.44.0 were used in the recorded run. Required
packages are NumPy and SciPy; the study records whether optional packages exist.

```sh
git clone --depth 1 --branch v1.2.1 --single-branch https://github.com/derb12/pybaselines.git .lossless/external/pybaselines-v1.2.1
```

The script requires a clean tree at
`6f8d927e08577a9c5af5f34129b14d57c12725f6`, even if the tag later changes.
Keep upstream `LICENSE.txt` and `LICENSES_bundled.txt` with the checkout.
If the checkout already exists, skip cloning and verify its revision/status.

Expected runtime is approximately one minute on the recorded M2, allowing
1–3 minutes for startup variation. It uses two fresh processes per case,
90-second budgets per normal split and a 30-second error-control budget.
Use a fresh output directory each time. In zsh/bash:

```sh
set -o pipefail
python -u lossless-product/experiments/application_intake/study.py \
  --project .lossless/external/pybaselines-v1.2.1 \
  --output .lossless/external/pybaselines-intake-reproduction \
  2>&1 | tee .lossless/external/pybaselines-intake-reproduction.log
```

The script prints flushed, timestamped progress and result/log paths. It writes
`summary.json`, a full `intake.json`, generated jobs/cases, and each replay's
source snapshot, manifest, HTML/JSON report and process logs. Entry points and
cases are selected explicitly; discovery does not choose them automatically.
Evaluation inputs are opened only after each algorithm's discovery replay.
They are public cases, not secret holdouts or a search qualification gate.

The invalid-parameter control intentionally receives `failed`; the overall
study passes only if all normal cases replay consistently, the invalid input
fails with `ValueError`, and the original checkout is unchanged.

See [results and product lessons](../../docs/application-intake-study.md).

The companion `profile_study.py` measures single spectra and declared eight-call
sequences, then automatically requests one Astra/medium assessment through the
Codex subscription connector. See [profiling results and commands](../../docs/application-profile-study.md).

## Current workflow against a strong CPU application baseline

This section measures **additional optimization of an already optimized
application**. It does not measure the total improvement over the original input
repository. See the fixed-output comparison below for that before/after question.

`current_cpu_study.py` exercises the selected source checkout's ordinary
`lossless.applications.search` path. It does not use OR-Tools, the solver pilot,
a hand-selected replacement candidate or modified optimizer logic. Context,
early stopping, timing scope and acceptance thresholds use that checkout's
product defaults. This is a source-checkout experiment, not a measurement of an
unchanged published wheel. The application source-search API remains experimental.

The fixed reference combines the reusable public `Baseline` API with the
previous continuation and CPU-control patches. It is the strongest previously
qualified implementation from that comparison, not a globally optimal algorithm
or a newly promoted product recipe. A discovery-only check first verifies exact
agreement with stock reusable objects. An identical-source A/A control must also
fail the speedup gate before authoring begins.

Three development and three held-out profiles contain four complete requests
each. Every request supplies newly generated signal values, with repeated or
changing grids/orders/weights, first-use setup, float32 input, masking and flat
signals. The spectra are simulated; the library and fitting workflow are real.
This does not represent measured production traffic frequencies.

The timer includes preparation, cache handling, fitting, correction and result
construction. Fixture decoding, process startup and correctness inspection are
excluded; import/factory setup is recorded separately. Acceptance checks exact
observations and retained outputs, final setup, RSS and traced memory. The recorded
local run additionally checked regressions at every declared call position;
GitHub main at `8595c915e5826065bcd9a37a5b03b9f76b84d217` checks whole-case
regression and does not include the local CPU architecture context. Those
differences are part of the versions being tested, not an isolated ablation.
One OpenBLAS thread is fixed for both versions.
Fresh paired processes and a sealed finalist prevent final-input feedback from
entering authoring. No solver follow-up runs automatically after a failure.

The default pilot allows two provider calls with up to 90 seconds per response,
240 seconds total search/validation, and a 90-second final reserve. The
preflight has a separate 45-second cap.
Expected total runtime is roughly three to five minutes; a short unsuccessful
search is not evidence that the application cannot be improved. Longer budgets
should be run explicitly in a foreground Terminal, with the same frozen gates.

Using the existing CPU environment and clean pinned upstream checkout:

```sh
set -o pipefail
OPENBLAS_NUM_THREADS=1 PYTHONPATH=lossless-product/src \
  .venv-integration/bin/python -u \
  lossless-product/experiments/application_intake/current_cpu_study.py \
  --project .lossless/external/pybaselines-v1.2.1 \
  --output .lossless/external/pybaselines-current-cpu-new \
  2>&1 | tee .lossless/external/pybaselines-current-cpu-new.log
```

For a fresh CPU environment, use Python 3.11 and install the recorded numerical
dependencies (no OR-Tools dependency):

```sh
python3.11 -m venv .venv-integration
.venv-integration/bin/python -m pip install numpy==1.26.4 scipy==1.16.0 numba==0.61.2 llvmlite==0.44.0
```

To reproduce the GitHub-main arm, keep its source separate and select it with
`PYTHONPATH`. The experiment driver and frozen baseline patches come from this
working tree; the optimizer and its workers come entirely from the selected
checkout. For a new checkout directory:

```sh
git clone https://github.com/aarnav9/lossless-opt.git .lossless/external/lossless-main-20261010
git -C .lossless/external/lossless-main-20261010 checkout --detach 8595c915e5826065bcd9a37a5b03b9f76b84d217
set -o pipefail
OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=.lossless/external/lossless-main-20261010/lossless-product/src \
  .venv-integration/bin/python -u \
  lossless-product/experiments/application_intake/current_cpu_study.py \
  --project .lossless/external/pybaselines-v1.2.1 \
  --output .lossless/external/pybaselines-main-cpu-new \
  --optimizer-label github-main-8595c91 \
  2>&1 | tee .lossless/external/pybaselines-main-cpu-new.log
```

Run the arms sequentially to avoid timing interference. Use the same seed,
baseline, workload and budgets. `protocol.json` records the imported optimizer
root, checkout commit, Python source status and hashes; `run/plan.json` retains
the actual resolved defaults. The executed driver is copied into the output.

Use `--prepare-only` to freeze inputs and code without running anything, or
`--preflight-only` to perform the local baseline controls without a model call.
Then append `--resume` and use `tee -a` with the same output/log paths to execute
the frozen live search. The runner rejects changed sources/fixtures and an
already-started search instead of silently repeating it. Live authoring uses the
existing ChatGPT provider with benchmark source and development measurements;
it does not send final cases. Result and log paths plus flushed UTC progress are
printed at startup, at each batch, during provider waits, and at completion.

`summary.json` records the current-workflow outcome and preflight controls;
`run/report.json` retains candidate history, paired latency samples, per-call
gates where supported, memory, setup, source identities and search payback. Final
measurements and payback exist only when discovery selects a finalist. The summary also
charges preparation/preflight time separately. There is no manual candidate
repair after selection and no automatic deployment.

### Recorded bounded pilot, 2026-10-10

Evidence is retained in
[`current-cpu-no-solver.json.gz`](../../docs/evidence/current-cpu-no-solver.json.gz).
The baseline, fixtures and requested plan hashes match between the local and
GitHub-main preparations. The optimizer source hashes differ. Main was fetched
at `8595c915e5826065bcd9a37a5b03b9f76b84d217`; local is that commit plus the
working-tree changes recorded by Python source hashes in the protocol.
That historical local run included CPU author context which is not shipped in
this focused timing-and-evidence update. Running this branch therefore does not
reproduce that exact author-context treatment; the archive retains its resolved
plan and source identities. The fixed-output comparison below requires no author
context or live provider.

| Arm | Outcome | Interpretation |
| --- | --- | --- |
| Local, initial 45-second response cap | Both author calls timed out; reference retained | No valid candidate; not a performance result |
| Local, 90-second response cap | First candidate matched discovery outputs at 0.9644x; second candidate's comparison hit the discovery deadline | No accepted improvement; held-out evaluation remained closed |
| GitHub main, 90-second response cap | First candidate matched discovery outputs at 0.9444x; second response requested stop | No accepted improvement; held-out evaluation remained closed |

The 90-second local attempt took 173.2 seconds including preparation and controls
(150.3 seconds inside search). Counting the failed initial attempt, local tuning
cost was 304.2 seconds. The main run took 96.0 seconds including preparation and
controls (71.9 seconds inside search). These costs exclude construction of the
previously frozen strong baseline. No candidate qualified, so no amortization or
final memory improvement is claimed. Dollar cost is unknown: this uses the
signed-in ChatGPT provider, not a measured API bill.

The completed discovery candidate cached one weighted pseudoinverse. It did
not help this workload, which changes weights across fits. The second proposal
hoisted a loop-invariant branch, but its evaluation was incomplete. Three paired
discovery rounds are screening evidence, not a reliable estimate of a small
slowdown or a proof of the optimizer's overall effectiveness. The retry changed
only the provider response cap; it was a declared follow-up, not an independent
replication selected to obtain a favorable result.

Main's candidate also tried pseudoinverse caching, with a different key and cache
policy. It passed finite discovery equality but failed the speed gate; its next
response supplied no edits and normal early stopping retained the reference.
The initial prompt used 33,735 reported input tokens and 26.8 seconds on main,
versus 61,989 tokens and 70.9 seconds locally. Additional local context did not
produce a measured application benefit in this attempt. One sequential pair
cannot attribute the response-time difference to context alone or establish a
general quality difference between versions.

For the retained baseline, the local discovery profile measured complete
four-request sequences of 7.04 ms (stream), 9.34 ms (changing weights), and
3.29 ms (mixed branches). Import/factory setup was about 460–463 ms, recorded
separately. Lifetime process RSS in the timing workers was 129–132 MiB, including
imports and fixtures; separately measured maximum traced per-call live allocation
above entry was 333–736 KiB. These are baseline diagnostics, not improvement claims,
tail-latency estimates, or a fresh-process application speedup.

The fixed baseline agreed exactly with stock reusable objects in discovery,
and the identical-source A/A controls did not pass the speed gate. The short
stock-versus-strong timing check used only two paired rounds and does not itself
establish a new speedup claim. No OR-Tools run was performed, and these results
do not establish that adding a solver is the next useful intervention.

## Final optimized program versus the original input

`measure_cpu_outputs.py` answers the before/after question directly. Its reference
is a clean, pinned pybaselines checkout using the public reusable `Baseline` API
and the same spectrum-processing wrapper. Every upstream file is verified
unchanged in that reference; there are no previous optimization patches in it.

Two optimized programs are fixed before measurement: the continuation proposal
recorded on GitHub main at `8595c915e5826065bcd9a37a5b03b9f76b84d217`, and the
stronger local continuation-plus-CPU-control implementation from prior evidence.
These are saved experimental outputs of Lossless, not a fresh optimizer search,
automatic packaged recipes or a guarantee that a new main run reproduces them.
The local source is also checked against the earlier frozen best implementation.

The primary score is **final application execution speed**. Time spent finding,
compiling or validating a candidate is not charged to each application request
or used to reject a speedup. This run performs no optimization, model calls or
OR-Tools solving. Both programs use the same current measurement harness.

Three new-seed profiles cover streaming batches, changing weights/orders, and
small float32 requests with masking, shuffled/irregular grids and flat signals.
Each profile has four complete requests with new signal values. Twelve paired
fresh-process rounds per profile use balanced randomized reference/candidate
ordering. The first request is included; allocation, preparation, cache lookup
and creation, CPU dispatch, fitting, correction and result construction are
inside request timers. Imports/factory setup are measured separately. Fixture
loading, correctness hashing and interpreter/worker startup are excluded.
Timer and completion-check bookkeeping remain inside the measured boundary;
there is no modeled or artificially added overhead penalty.

An identical-source A/A control runs before both comparisons. All outputs, inputs
and retained outputs must agree exactly. The declared request gates are at least
1.02x case-balanced speedup, at most 3% case/call regression, and sign-test
p <= 0.025 per comparison (two comparisons, family level 0.05). The separate
setup/RSS/traced-memory limits remain 100 ms/1.2x/1.2x. Import-plus-request
statistics are reported as a separate lifecycle diagnostic. Nothing is selected
or repaired using these measurements, and no artifact is automatically deployed.

Use the clean checkouts and numerical environment described above. Expected
measurement runtime is about three to four minutes; the default cap is 285
seconds. From the repository root:

```sh
set -o pipefail
OPENBLAS_NUM_THREADS=1 PYTHONPATH=lossless-product/src \
  .venv-integration/bin/python -u \
  lossless-product/experiments/application_intake/measure_cpu_outputs.py \
  --project .lossless/external/pybaselines-v1.2.1 \
  --main-checkout .lossless/external/lossless-main-20261010 \
  --output .lossless/external/pybaselines-output-vs-stock-new \
  2>&1 | tee .lossless/external/pybaselines-output-vs-stock-new.log
```

`--prepare-only` freezes the three program trees, new fixtures, source identities,
policy and executed driver without benchmarking. `--resume` runs that frozen
experiment and rejects changed source or an already-started measurement. Raw
timing pairs, per-call samples, exactness fingerprints, memory checks and setup
diagnostics are retained under the output alongside `summary.json`. The
`measurement_run_seconds_not_scored` field is operational bookkeeping only.

### Before/after results, 2026-10-10

The corrected comparison found an application speedup over unmodified input.
The best previously qualified local output passed every frozen gate. These
numbers include ordinary request overhead and exclude optimization duration.

| Fixed output versus stock reusable API | Paired request speedup | Less request time | All gates |
| --- | ---: | ---: | --- |
| GitHub-main continuation output | 1.09052x | 8.30% | Fail: small-profile and individual-call regressions |
| Local continuation + CPU-control output | **1.28910x** | **22.43%** | **Pass** |

All measured output/input/retained-output fingerprints matched for both outputs.
Main's fixed-sample sign-test p was 0.019287 and local's was 0.003174, both below
the predeclared 0.025 threshold. Main nevertheless fails the separate regression
checks. The A/A score was 1.000997x and did not qualify as a speedup. All setup and
memory checks passed; the largest local RSS and traced-peak ratios were 1.00765x
and 1.00166x. Source and fixture identities were unchanged after measurement.

| Profile (four complete requests) | Stock median in local comparison | Local optimized median | Local paired speedup |
| --- | ---: | ---: | ---: |
| Streaming batch | 14.268 ms | 11.192 ms | 1.26131x |
| Changing weights/orders | 14.077 ms | 10.386 ms | 1.36324x |
| Small mixed float32/branch cases | 0.910 ms | 0.724 ms | 1.24584x |

Paired speedup is the geometric mean of paired ratios, not the ratio of the
displayed medians. Main's small-profile paired score was 0.90811x, despite a
lower candidate median. Two candidate sequences took 3.542 and 3.232 ms, versus
roughly 0.7–1.1 ms for its other sequences. Those observations were retained;
their cause is not established and the gate was not relaxed or rerun to obtain
a pass. Local's worst per-call paired score was 1.13208x. The sequential
comparisons independently estimate each output versus stock; they do not
constitute a direct paired main-versus-local experiment.

**Lifecycle matters.** The primary result represents requests in a running
application. With import/factory setup plus just four requests included, measured
aggregate ratios were 1.00012x (main) and 1.00366x (local): effectively no useful
startup-inclusive gain established here. This diagnostic includes application
imports/setup but excludes interpreter startup and fixture loading; it is not
whole-process wall time. Reusing a running application amortizes initialization;
the 22.43% result must not be advertised as a fresh-process CLI speedup.

This is finite exactness and timing evidence on one CPU and three new-seed
simulated spectral-processing profiles, not a universal speedup or proof of
equivalence. The saved outputs were chosen before these cases were measured.
Neither was modified, newly searched or automatically deployed in this run.
[Raw protocol, frozen sources and paired evidence](../../docs/evidence/cpu-output-vs-stock.json.gz).
