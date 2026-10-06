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
