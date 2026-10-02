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

Original code and original contributions use the [MIT License](LICENSE). Retain the [third-party notices](lossless-product/NOTICE) for imported material. The repository remains a private development alpha until public release; applying MIT does not publish it. A fresh full proof-toolchain installation check remains outstanding alongside broader backend validation.
