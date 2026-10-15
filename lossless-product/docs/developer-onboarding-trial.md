# Independent developer onboarding trial

Status: ready to run; no independent participants or results yet. Automated
project trials do not count as developer onboarding evidence. Use the published
[0.1.0a3 patch alpha](release-0.1.0a3.md) for the first round. Current source has
additional recipes under development; record the exact version used.

## Participants and task

Recruit three Python/NumPy developers who have not worked on Lossless, each with
a local CPU callable and real representative examples. Include one project with
small, infrequent calls where startup can outweigh optimization. Participation
and sharing are voluntary. No telemetry or automatic uploads are required.

Give participants the [CPU guide](cpu-projects.md) and this task, without live
implementation help for the first 30 minutes:

1. Install the wheel into the project's environment and run `lossless doctor`.
2. Choose one callable. Supply a Python input loader with separate discovery and
   evaluation examples. Preserve an independent application test for the result.
3. Run `lossless run-project` with the expected number of calls in their application.
   Save its report even when it retains the original or fails.
4. Use `CallableApplication` inside a small application invocation with fresh
   values. Verify results using the independent test. Keep the context open for
   repeated requests and then close it.
5. Explain which source and worker mode were selected, what the reported speed
   compares against, and how to run the original. Try one changed shape using the
   documented reference path; record whether the behavior was understandable.

Use a new output directory outside the source project. An example foreground
command, after replacing paths and entry names:

```sh
mkdir -p ./lossless-runs
set -o pipefail
printf 'Result: ./lossless-runs/trial/report.json\nLog: ./lossless-runs/trial.log\n'
PYTHONUNBUFFERED=1 lossless run-project ./my-project --entry app.py:run \
  --input-loader examples.py:cases --deployment auto --expected-requests 100 \
  --budget 3m --output ./lossless-runs/trial 2>&1 | tee ./lossless-runs/trial.log
printf 'Result: ./lossless-runs/trial/report.json\nLog: ./lossless-runs/trial.log\n'
```

The workflow emits timestamped phase progress. A three-minute search budget
does not include every intake, delivery and deployment operation. Run it in the
foreground and record the total elapsed time as well.

## Record observations before helping

Use [the feedback template](../experiments/developer_onboarding/feedback.json).
Record installation time, loader preparation time, time to the first working
adapter, total elapsed time, number of changed files and every intervention.
Record blocked attempts and no-win outcomes. Do not rerun with hand-selected
inputs until a speedup appears. Keep the initial attempt distinct from any
assisted follow-up.

A successful usability trial means a working adapter with correct observed
results, an unchanged original, a usable reference path, and an accurate
explanation of the result. A source speedup is a separate performance outcome.
Prepared-worker gains against a new process per call must not be reported as
gains over ordinary in-process Python.

Pilot targets, fixed before recruitment: all three participants can install and
reach an interpretable outcome; at least two integrate the adapter within 30
minutes without implementation help; all three can identify the baseline and
reference path. A three-person pilot identifies friction; it does not estimate
population-wide success rates. Revise documentation or code from the first round
and use new participants for a subsequent unassisted round.

Participants choose what to share. Reports, logs and bundles may contain paths,
source code or fixture details; ask for a minimal redacted summary first. Record
their consent before publishing quotations or project identities. Keep raw
participant responses out of version control unless explicitly approved.
