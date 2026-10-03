# Repository organization and release plan

> Implementation status: the source alpha is now installable. See [the alpha guide](alpha.md) for supported behavior and [retention status](retention.md). This document describes the broader target design; its draft interfaces and release plans are not all implemented.

Keep the public repository focused on the product, its supported behavior, and enough evidence for contributors to assess it. Preserve the local research separately instead of deleting history. The cleanup completed on 2 October 2026 moves the old workspace under ignored `research/` and keeps a concise public `EXPERIMENTS.md` at the outer repository root.

## Name recommendation

Use **Lossless** as the product name and **`lossless-opt`** as the GitHub repository name. The suffix explains that this is an optimizer while leaving room for kernels, model execution, and numerical workloads. The source alpha now installs the `lossless` CLI.

`lossless` is the shorter alternative but is less descriptive when encountered outside the project. `lossless-product` is useful as the current staging directory; it is an internal organizational name rather than the recommended public repository name. Avoid names tied exclusively to kernels or MLX when the product includes scheduling, memory handling, and other execution targets.

Repository, Python distribution, import, executable, and domain names are separate decisions. A web search is not a reservation or an exhaustive availability check. Confirm the desired owner/repository and distribution names before publication. No repository, package, or domain was created or renamed during this cleanup.

## Current local layout

```text
lossless-opt/
  README.md
  EXPERIMENTS.md
  .gitignore
  lossless-product/
    README.md
    docs/
    examples/
  research/                         # ignored
    WORKSPACE.md
    experiments/
      plans/                        # formerly root EXPERIMENT_PLAN*.md
    results/
      reports/                      # formerly root RESULTS*.md
    notes/                          # literature and hardware research
    artifacts/                      # downloaded models and upstream code
    kernel_lab/
    tests/
    campaigns/
    proposals/
    examples/
    pyproject.toml
    run_*.py and run_*.sh            # preserved historical entry points
    .relocation/                    # move receipt and original source hashes
  .venv-*/                          # existing local environments; ignored
```

Historical runner dependencies are kept together inside `research/`. Compatibility links there retain the old report/plan filenames, literature path, and environment names. The environment directories remain alongside `research/`; the later outer-folder rename changes their absolute paths. Before using them for another campaign, check or recreate environments whose interpreter scripts still refer to the previous workspace path. The original research README and long reports are local history, not the product landing page.

The ignore rule excludes the entire research tree. This workspace was not a Git repository when reorganized, so there were no tracked research files to untrack. If future work imports existing Git history, remember that ignore rules alone do not remove already-tracked content; inspect the index before publishing.

## Public release layout

Keep `lossless-product/` as the extraction boundary while the product is being built. Once the package works independently, make its contents the public repository root and carry over the single experiment ledger and root ignore rules. That promotion is planned, not performed here.

```text
lossless-opt/
  README.md
  EXPERIMENTS.md
  CHANGELOG.md
  LICENSE
  NOTICE
  CONTRIBUTING.md
  pyproject.toml
  src/lossless/
  tests/
  examples/
  proofs/
  docs/
  .github/workflows/
  research/                         # optional local workspace; ignored
```

`EXPERIMENTS.md` records findings, including failures. `CHANGELOG.md` should record released user-facing behavior once releases exist. They serve different purposes; do not duplicate the entire research history into the release changelog.

The [distribution and first use plan](distribution.md) specifies installation channels, the README experience, optional dependencies, measured local sizes, and proposed release-size budgets.

## Next cleanup and release work

| Priority | Work | Done when |
| --- | --- | --- |
| 1 | Finalize the initial supported recipes, input contracts, and target matrix | Each promised gain has a reference, applicable workload, exact/numerical classification, and retention test |
| 2 | Extract the implementation into the product package | No imports or required resources come from ignored research; the package works outside this workspace |
| 3 | Replace provisional examples with a runnable quickstart | A fresh user can diagnose their machine, run a small demo, connect an LLM, inspect a report, and load an artifact |
| 4 | Curate evidence for shipped optimizations | Compact positive/negative fixtures, proof sources, scope, native notices, baselines, and raw timing evidence can be reproduced from public materials |
| 5 | Verify performance retention during extraction | Product and frozen research implementations are compared against the same original reference; wrapper, startup, fallback, and memory costs are included |
| 6 | Consolidate public documentation | README stays short; architecture, configuration, contracts, support limits, and extension interfaces each have one authoritative home |
| 7 | Make generated assets reproducible | Mermaid is the diagram source; a documented render step refreshes committed images, and duplicate diagrams are checked for drift |
| 8 | Add focused quality checks | Formatting/linting, behavioral tests, schema validation, link checks, isolated wheel installation, and documented hardware checks run appropriately |
| 9 | Finish publication metadata | Owner chooses the original-code license; reused code retains notices; contribution/security guidance and issue templates match actual supported workflows |
| 10 | Review the actual publication set | Git status/index and release archives contain no research tree, private configuration, weights, environments, stale builds, or unsupported benchmark claims |

Keep product-only requirements separate from research environment pins. Start the public package in a `src` layout with explicit package resources and optional backend dependencies. Avoid a shell-only quickstart that relies on local model directories or an existing `.venv`.

Prefer small examples that each demonstrate one supported workflow. Keep slow model/hardware tests separate from fast CPU checks. Contributor instructions should require the experiment ledger to be checked before proposing another search, and require every public speed claim to state its reference and contract.

The broad research tree being ignored does not justify hiding code or evidence needed for the released product. Extract those assets deliberately, with their notices, tests, and proof limitations. Large evidence bundles can be published as versioned release assets; contributors must still have enough public information to reproduce the supported behavior.

## What is complete and what remains

Completed: outer-root cleanup, preserved local research, grouped plan/report files, ignore policy, compact experiment ledger through campaign 030 plus supplemental studies, product design/architecture documents, and local Git initialization for the development snapshot.

The source repository is public, with an installable MIT-licensed alpha, CPU CI and isolated wheel-install checks. Live hosted-provider validation, broader hardware qualification, a fresh full proof-toolchain installation, and package-registry distribution remain pending. The first tagged alpha includes release artifacts and checksums. No long-running model experiment is needed merely to organize the repository; performance retention runs belong to the extraction stages and follow the foreground Terminal policy.
