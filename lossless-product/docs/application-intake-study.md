# Existing GitHub repository: pybaselines intake/replay

On 5 October 2026, the current source application interface replayed three
unmodified algorithms from [pybaselines](https://github.com/derb12/pybaselines),
a scientific library for baseline correction of spectroscopy and other
experimental data. All 18 algorithm/input combinations were repeatable across
two fresh processes each. An invalid-parameter control correctly failed.
The complete check passed again after improving entry-point suggestion ordering.

**This qualifies an intake/replay example, not a speedup.** There was no model
call, candidate implementation, optimization, performance acceptance or export.

## Repository and execution boundary

- Upstream release: `v1.2.1`, pinned to
  `6f8d927e08577a9c5af5f34129b14d57c12725f6`.
- License: [BSD-3-Clause](https://github.com/derb12/pybaselines/blob/6f8d927e08577a9c5af5f34129b14d57c12725f6/LICENSE.txt).
  Local modification/optimization is permitted. Redistribution requires the
  applicable copyright/license notices, and the authors' names cannot imply
  endorsement. The checkout also preserves `LICENSES_bundled.txt`.
- Source snapshot: 150 files, 2,318,757 bytes, including package source and notices.
  Neither upstream implementation files nor the checkout were changed.
- Runtime: Apple M2, macOS arm64, Python 3.11.3, NumPy 1.26.4, SciPy 1.16.0,
  optional Numba 0.61.2 and llvmlite 0.44.0; pentapy absent.
- Invocation: direct existing module functions through generated application jobs.
  No custom application driver or new Lossless backend was required.
- Inputs: our own seeded synthetic spectra, supplied as JSON numeric lists.
  This uses real upstream code, not measured laboratory data.

## Cases and results

Each algorithm received the same six input profiles: noisy spectra with 256,
1,000 and 4,096 samples plus a flat 256-sample signal for discovery; an odd
511-sample spectrum and a new 2,048-sample noise realization for evaluation.
The evaluation split was explicitly opened after each discovery run. These are
public test cases, not secret holdouts or an algorithm-accuracy study.

| Existing function | Discovery cases | Evaluation cases | Successful fresh processes |
| --- | ---: | ---: | ---: |
| `pybaselines.whittaker.asls` | 4/4 | 2/2 | 12/12 |
| `pybaselines.polynomial.modpoly` | 4/4 | 2/2 | 12/12 |
| `pybaselines.smooth.snip` | 4/4 | 2/2 | 12/12 |
| **Total per complete run** | **12/12** | **6/6** | **36/36** |

Returned baseline arrays and parameter dictionaries, plus inputs before/after
each call, matched exact fingerprints within each pair. All 18 fingerprints
also matched between the initial and confirmation studies. The invalid AsLS
parameter `p=2.0` produced a `ValueError` failure receipt on its first attempt;
it did not become a successful replay.

The original checkout remained clean and its selected file hashes unchanged.
Both studies used zero LLM calls. Initial and confirmation study elapsed times
were 41.30 and 44.12 seconds. These include intake, imports, copying and
observation, and are **not algorithm latency or performance comparisons**.

The product regression suite after the discovery fix ran 109 tests: 108 passed
and one was skipped. Formatting and lint checks passed.

## What the external test taught us

The existing callable interface handled this package's decorators, relative
imports, SciPy dependency and tuple/array/dictionary results without an adapter
for each algorithm. All three functions appeared in static discovery.

The first run found a setup problem: alphabetical scanning put documentation
helpers in all twelve displayed suggestion slots. Discovery now scans likely
application source before documentation/examples/tools, then tests. The same
ordering applies before the 200-file scan limit, so large documentation trees
cannot consume that limit before application files. A regression test covers
this. All 515 discovered functions remain available in this repository's full
record; the scan was not truncated.

The ordering is a path heuristic, not semantic API selection. Internal module
helpers can still appear; the user or future setup assistant must select an
invocation and representative inputs. We supplied those explicitly here.
The environment's required dependencies were already available; automatic
installation, source building and missing-dependency recovery were not tested.

The next useful experiment on this checkout is complete-request profiling,
including repeated spectrum batches and setup cost. That will determine whether
time is spent in Python iteration, solver setup, allocation or compiled library
calls before choosing an optimization. No speedup is established by this study.

## Evidence and reproduction

- [Runnable study and exact commands](../experiments/application_intake/README.md).
- [Curated results](evidence/application-intake-pybaselines.json): pinned identities,
  case fingerprints, failure receipt details, source/runner hashes, and discovery
  suggestions before/after the change.
- Raw local runs are under `.lossless/external/pybaselines-intake-01` and
  `.lossless/external/pybaselines-intake-02`; each has generated configuration,
  input files, source snapshots, manifests and per-process reports/logs.

The upstream checkout and raw workspaces remain ignored local downloads. The
tracked reproduction script generates the cases without modifying upstream.
No changes or messages were sent to the upstream project.
