# Contributing

Experiments are the main contribution path. Optimization ideas for ML, vision, and linear algebra are welcome, including ideas that change numerical results or task quality. You can start with an issue describing a hypothesis; a working patch, GPU, or successful speedup is not required. Negative and inconclusive results help others avoid repeating work. Runtime fixes, proof improvements, and documentation are welcome too.

Read [EXPERIMENTS.md](EXPERIMENTS.md) before repeating an approach. Each recorded experiment gets one experiment line and one result line. Keep exploratory runners and bulk results in the ignored `research/` workspace. When a result supports a public claim or reusable feature, contribute its small reproduction script and fixtures to the tracked examples/tests, and link any larger evidence separately. A fresh clone must be able to reproduce the claim without the author's private archive.

## Choose the correctness contract

| Experiment category | What acceptance means |
| --- | --- |
| Exact / lossless | Preserve the contract's declared output bits, state, and behavior, under its stated domain and assumptions. |
| Numerical | Meet explicitly chosen error bounds and invariants; report where output bits differ and any precision changes. |
| Task quality | Meet a declared task metric and allowed degradation on a specified evaluation dataset; report numerical, precision, or weight changes. |

Examples of task metrics include classification accuracy and segmentation IoU. Record the reference score, candidate score, allowed change, dataset version/split, and uncertainty where relevant. Matching one task metric does not establish numerical or bitwise equivalence.

Choose the contract before its acceptance evaluation. An exact candidate that changes bits may motivate a new numerical or task-quality experiment, but its exact result remains a failure. Freeze the new criteria and use fresh evaluation before accepting the new experiment. Incorrect memory accesses, races, or unsupported inputs remain implementation failures. An experiment's category is separate from its outcome: untested, failed, inconclusive, no win, or accepted within the stated scope.

Record evidence separately: finite validation coverage, statistical estimates, numerical certificates, or a proof of a named statement and its assumptions. Never describe finite tests or real-number algebra as a floating-point equivalence proof. Weight and precision changes require explicit permission in the contract.

These categories organize research contributions. The current executable alpha supports only the documented exact and numerical adapter contracts. Task-quality contracts and new transformations need an implemented evaluator before they can be accepted by the product; these guidelines do not add runtime support or relax existing jobs.

## Submit an idea or result

Use this short format in an issue or PR; leave unmeasured fields marked untested:

```text
Experiment: [category] hypothesis, workload, proposed change, hardware/runtime.
Result: [untested / failed / inconclusive / no win / accepted within scope]
        comparator, correctness evidence, timing/memory/quality change, reproduction link.
```

For measured results, include the configuration, inputs or generator, compiler/runtime versions, seeds, discovery/evaluation split, search budget, and timing method. Keep full-model measurements separate from isolated-kernel timings. Report changed precision or weights, failures, and cases where the reference wins. A new idea can start a discussion before this evidence exists.

## LLM-proposed tests and new backends

This is a proposed extension workflow. The alpha has fixed adapter contracts, seeded native input generation, regression tests, and discovery/evaluation separation; it does not currently accept LLM-generated test specifications or create new execution backends automatically.

Use the LLM to turn an experiment hypothesis or failure into additional cases: awkward dimensions, supported strides, padding, repeated calls, aliasing, state resets, or exceptional values allowed by the input domain. It can draft input generators, minimal reproducers, environment probes, and backend setup code. An experiment summary is useful context, but its claims and copied constants are not an oracle for correct results.

Keep the acceptance machinery independently defined:

1. Maintain a versioned baseline suite and seedable generators, with required coverage derived from the contract. Check the suite against known-correct references and deliberately broken implementations, so a checker that always passes is detected.
2. Treat generated case descriptions as proposals. A bounded JSON schema can constrain supported operations, shapes, dtypes, distributions, and resource limits. Cases that fit an already reviewed generator can be admitted automatically; executable generators, reference implementations, and runner changes need code review and validation before joining the trusted suite.
3. Compute expected results from the frozen reference or an appropriate independent oracle. For exact CUDA jobs, compare against the declared CUDA reference on the pinned target/runtime. A CPU result is useful where its semantics match the contract; do not assume identical floating-point results across architectures.
4. Freeze the contract, suite/generator version, target facts, and acceptance policy before candidate search. Keep evaluation inputs and results out of proposal context. A future generalized runner should choose additional evaluation seeds independently after selection is sealed and record them for replay. The current native alpha uses fixed seeded evaluation cases. Public regression tests remain public, and trusted local processes do not provide a security boundary against arbitrary code reading files.
5. If an LLM suggests a new case after seeing evaluation results, retain it as a regression for a subsequent experiment. Use fresh evaluation for that experiment. Keep incorrect, failed, and skipped checks visible; missing hardware means untested, not passed.

Determinism means that recorded inputs, generators, versions, and settings support replay in a qualified environment. It does not mean that a seed guarantees identical results across every framework release or device. PyTorch documents these limits in its [reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html).

As the suite grows, the proposed organization is:

```text
lossless-product/tests/
  fixtures/       Small JSON contracts, case descriptions and input fixtures
  generators/     Reviewed, seeded generators shared where semantics allow
  conformance/    Common contract checks, including known-bad controls
  backends/       Runtime-specific setup, synchronization, state and export checks
```

These subdirectories are a plan, not a claim that a generic conformance suite or CUDA tests already exist. Keep existing tests working while extracting shared helpers. Large model weights and datasets use versioned manifests and hashes rather than entering Git. Runtime dependencies install from explicit, reviewed requirements; LLM-generated setup scripts remain proposals until reviewed and exercised.

For the first CUDA test bundle, capture the actual GPU, driver and runtime, execute a small reference and deliberately incorrect candidate, run a bounded real optimization, and verify export/load in a fresh process. Check candidate timing with the required GPU synchronization; [CUDA operations can execute asynchronously](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution). An LLM can help author this bundle, but only execution on the target device supplies hardware-validation evidence. Scope special-value, state, and race checks to the supported contract and backend. Independent tests and finite comparisons still do not establish universal correctness.

## Develop and validate

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e './lossless-product[dev]'
python -m unittest discover -s lossless-product/tests -v
python -m build lossless-product
```

Use clang for native checks. The CPU suite is small; substantial model campaigns should be run as explicit foreground Terminal jobs with unbuffered, timestamped progress and a named log. Product code must install and run without `research/`.

A new adapter needs an executable contract, independent reference, bounded candidate evaluator, disjoint discovery/evaluation inputs, export guards and fallback behavior. A negative/no-win result is valid product behavior.

For extraction, bring the required runtime code, small fixtures, scoped proof sources, pins and notices into `lossless-product/`. Update the retention inventory. Keep user credentials, model weights, compiled caches and generated runs out of commits and archives. Build and install a wheel outside the checkout before changing the supported installation story.

## Next validation milestone

The planned next adapter is a bounded PyTorch-on-CUDA optimization workflow with a supported vision example. CUDA execution is not implemented in this alpha, and the maintainer has no local NVIDIA GPU. Local CPU checks can validate orchestration and failure handling, but cannot qualify CUDA kernels or performance.

The intended sequence is:

1. Implement the CUDA path and prepare a short, versioned test bundle with explicit setup and output paths. Record the tester's GPU model, OS, driver, and available CUDA/PyTorch environment before selecting supported versions.
2. Have the first external tester run installation, baseline execution, correctness checks, a bounded search, and export/load on their NVIDIA hardware. Keep unexpected results and diagnostics with the report. This is the first external validation, not broad hardware qualification.
3. Fix issues from that run, then expand to public comparative benchmarks, additional GPUs, and larger workloads. Broad performance ranking follows the first external test.

## License

Original code and original contributions use the [MIT License](LICENSE). Retain the [third-party notices](lossless-product/NOTICE) for imported material. The source repository is public. Tagged GitHub releases and package-registry distribution remain pending. A fresh full proof-toolchain installation check remains outstanding alongside broader backend validation.

The [032–038 follow-up report](lossless-product/docs/frontier-032-038.md) contains foreground reproduction commands for the manual proposal replay, Z3 search, graph extraction, scheduling, head bounds, racing and tensor decomposition. Its optional Z3/SymPy dependencies are research tools, not base package dependencies. Replaying preserved manual proposals does not count as a new live model call.
