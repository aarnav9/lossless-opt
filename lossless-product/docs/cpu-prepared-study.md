# Prepared CPU requests and broader candidates — 13 October 2026

This follows the [first project workflow study](cpu-project-workflow-study.md).
The changes are available through `run-project --prepared`, `PreparedApplication`,
and `diagnose-application`; see the [usage guide](cpu-projects.md).
All work used local CPU execution with **zero model calls**. Exactness was never
relaxed to retain a faster candidate.

## Results and measurement boundaries

The NumPy Financial candidate combines two independently tested edits: guarded
ndarray dispatch for expression arguments (`np.all(diff < tol)`) and reuse of
identical arithmetic subexpressions in `_g_div_gp`. Arithmetic grouping and precision
remain unchanged. Numeric guards keep the original implementation for custom types.
The source rewrite rule is generic; there is no package-name special case for Financial.

A final integration rerun after the deployment import-path fix used another fresh
seed, `202610132`, and the same frozen candidate-generation rules. It again accepted
**1.3642× callable speed** and **1.3462× complete prepared-request speed** (about 26%
less request time; six rounds, p = 0.015625), with all exact and resource checks
passing. The width-8/1/32 request medians were 151.9/120.8/108.9 ms reference and
112.5/88.9/81.8 ms optimized. Median startup was 116 ms versus 298 ms. Both full
matrix runs are archived, including their runtime identities. The final runnable
bundle corresponds to this last run; the detailed table below records the earlier
matrix experiment.

The initial follow-up on fresh values selected the combination: dispatch alone
measured 1.1129×, arithmetic reuse 1.3288×, and the combination 1.3620× on discovery.
The frozen winner passed six held-out pairs at **1.3920× callable speed**. These
individual gains are not multiplicative. Its cold launcher still lost at 0.6894×.

A second run used seed `202610131`, discovery shape `(80, 8)` and three independent
held-out shapes `(97, 8)`, `(121, 1)`, `(61, 32)`, each with three changing calls.
It qualified **1.3633× callable speed** and **1.3595× complete prepared-request
speed**, compared with an equally persistent reference. All six prepared rounds
exceeded the 1.02× threshold (sign-test p = 0.015625); exact observation, transported
return values, per-case/per-position and memory checks passed.

| Financial case | Reference request | Optimized request | Paired speedup |
| --- | ---: | ---: | ---: |
| Width 8 (`new`) | 152.0 ms | 111.1 ms | 1.3657× |
| Width 1 (`narrow`) | 124.5 ms | 91.8 ms | 1.3636× |
| Width 32 (`wide`) | 110.2 ms | 81.6 ms | 1.3492× |

Times are medians of the average of three full requests per round. Each request
contains three changing callable invocations. Speedups use paired geometric means,
so they need not equal the ratio of these medians. Request time includes input
loading, checks, factory setup, observation, cleanup, serialization, transport and
decoding. It excludes worker startup. The reference is given the same reuse benefit.

Median startup was 113 ms reference and 303 ms optimized. Dividing that extra
190 ms by measured per-request savings suggests a **5–7 request break-even**, a
descriptive estimate rather than a separately qualified production service metric.
Median worker/process-tree peak memory was about 42.1–42.2 MB reference versus
46.3–46.8 MB candidate, below the 1.2× prepared limit. Sampling can miss short-lived
children and double-count shared pages; the report states those limitations.

For **pybaselines**, all eight retained candidates were tried again. Source-only
won and qualified at **1.2722× callable speed**. Prepared execution matched exact
values, but its 1.0186× request ratio failed the 1.02× gate and statistical evidence
(p = 0.65625). The three case ratios were 0.9911×, 1.0469× and 1.0186×; no prepared
speedup was approved. The optimized prepared path therefore stays disabled for this
bundle. These small callable savings are largely obscured by request overhead.

The prepared API reuses module imports in one private worker, creates a fresh
seeded factory per request, and keeps calls within the request on that object.
Six paired rounds exercise repeated requests and rotating case histories against
fresh-reference fingerprints. Qualification has a separate certificate; source
acceptance alone cannot enable optimized prepared execution. Modules with persistent
state that changes results fail this check. Input metadata misses raise before
optimized execution; failed calls/timeouts stop without retrying.

## TMM: quantified reference drift, no optimization approval

The unmodified TMM workload still fails exact repeatability. Eight plain process
probes observed differences up to **6.661338147750939e-16** absolute and **7 ordered
float64 ULP steps** in returned reflectance/transmittance values. A separate run
through the new public diagnostic observed the same maximum absolute difference
and up to 6 ULP steps. These are observed maxima on this suite, not universal bounds.

A return-time snapshot of `coh_tmm` locals found matching bytes for `th_list`,
`kz_list`, `delta`, `t_list` and `r_list`, followed by differing bytes in `M_list`,
the per-layer transfer matrices. In the saved example, two layer-one imaginary
components differ by about 2.78e-17 and 1.11e-16. The next combined transfer matrix
then differs too. This narrows the region to matrix construction; it does **not**
identify a proven faulty primitive or establish a library bug.

Tracing primitive function inputs/outputs changed the drift pattern. Adding a
`dot` wrapper made that traced run repeatable, while its plain control still drifted.
A separate reconstruction varying buffer alignment for `dot` and scalar multiplication
produced identical outputs at every tested offset. The alignment hypothesis was
therefore not reproduced. Instrumentation itself can perturb the relevant behavior.

No faster TMM candidate was approved. The small drift is a potential subject for an
explicit numerical contract, but there is no numerical speedup claim here. The exact
gate remains closed. [Diagnostic evidence](evidence/tmm-reference-diagnostic.json.gz)
contains actual returned array bytes, the saved intermediate example, all probe
reports including negative controls, and the original probe scripts. The portable
[`diagnose_tmm.py`](../scripts/diagnose_tmm.py) supports plain, function, local and
`dot` tracing against a prepared TMM project.

## Evidence, reproduction and product validation

[Workflow evidence](evidence/cpu-prepared-workflows.json.gz) includes proposals,
source and fixture hashes, selected patches, paired timing/memory records, admission
manifests, startup costs and separate prepared results. Historical attempts are kept.
These are finite exact observations for supplied cases, not a proof or a promise of
speedup on arbitrary repositories.

The same Apple Silicon host, Python 3.11, NumPy 1.26.4, SciPy 1.16.0 and upstream
revisions from the [original study](cpu-project-workflow-study.md#reproduce) were
used. BLAS/OpenMP thread variables were set to one. NumPy Financial is v1.0.0.
The pybaselines run deliberately repeats the retained fixture suite as a regression;
its fixtures are not claimed to be new research inputs.

After installing current `.[applications]`, run one project at a time from
`lossless-product/`. Create the log parent first. The tested Financial workflow took
about 42 seconds and pybaselines about 133 seconds, each under the 280-second budget.

```sh
mkdir -p lossless-runs
set -o pipefail
python -u scripts/check_cpu_project.py --kind numpy-financial \
  --project /path/to/numpy-financial --seed 202610132 --matrix --prepared \
  --budget 280 --output lossless-runs/financial-prepared \
  2>&1 | tee lossless-runs/financial-prepared.log
```

Repeat with `--kind pybaselines`, its source path and a fresh output/log path.
Use the public `diagnose-application` command to save actual TMM reference outputs;
use `diagnose_tmm.py --project /path/to/prepared/project --mode locals --output RUN`
for intermediate snapshots. No package weights or hosted authoring are involved.

Product tests cover actual transported bits (NaN payloads, signed zeros, complex
arrays and Fortran layout), expression evaluation once, custom numeric fallback,
fresh factories, reused process identity, private source snapshots, metadata rejection,
partial failure without retry, timeouts, persistent global-state rejection and
reference diagnostics. A delayed synthetic callable verifies positive admission;
its speedup is a routing test, not research evidence. The package-install check also
runs repeated requests through the installed wheel outside the source tree.

The candidate patches retain the upstream [NumPy Financial BSD-3-Clause notice](evidence/NUMPY_FINANCIAL_LICENSE.txt).
