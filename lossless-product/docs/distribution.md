# Lossless distribution and first use plan

> Implementation status: the source alpha is now installable. See [the alpha guide](alpha.md) for supported behavior and [retention status](retention.md). This document describes the broader target design; its draft interfaces and release plans are not all implemented.

Recommend a small Python distribution providing both the `lossless` CLI and `import lossless`. Install it into the user's computation environment. Keep backend dependencies optional, use GitHub for source and contributions, and add Homebrew only after the installation and worker-environment story is stable. The product targets ML and numerical computations; the first demo should work without model weights or an LLM key.

This is a design proposal. The CLI and installable source package now exist in the public source repository; no package registry release has been published. Commands below describe the intended experience after implementation. `lossless-opt` is the proposed distribution/repository name and `lossless` the proposed command/import name; ownership and availability still need verification.

## What the GitHub repository should look like

Keep the product in `lossless-product/` during extraction. Promote its contents to the public repository root once it works independently, carrying over `EXPERIMENTS.md` and the relevant ignore rules. Users should encounter this final layout:

```text
lossless-opt/
  README.md                 Purpose, working quickstart, supported targets
  pyproject.toml            Package metadata, CLI, dependencies and extras
  LICENSE                   MIT license for original code
  NOTICE                    Notices for reused materials
  CONTRIBUTING.md           Setup, tests, adding adapters and recipes
  EXPERIMENTS.md            Concise findings, including negative results
  CHANGELOG.md              Released user-facing changes
  src/lossless/             Core, adapters, runtime, bundled resources
  tests/                    Behavioral, packaging and hardware checks
  examples/                 Small runnable supported workloads
  proofs/                   Public proof sources for shipped claims
  docs/                     Contracts, architecture, support and evidence
  .github/workflows/        CI and release automation
  .gitignore
```

A contributor may create an ignored `research/` directory locally; a fresh clone does not need it. The package must work without the original research workspace. Keep useful fixtures, recipes, proof sources, and enough reproducible evidence public even when the campaign archive stays local. Create example folders as integrations become runnable; do not imply support with empty vision or CUDA directories.

The README should show, in order: the product definition and status; supported workload/device combinations; one install command and a short demo; a real-workload example; what the artifact/report contains; one scoped measured result once reproduced; links to contracts, contribution guidance, and the experiment ledger. Use one compact architecture diagram. Keep campaign histories and alternative installation instructions in linked documents. A design preview must continue saying that there is no runnable product yet.

## Installation channels

| Channel | Intended audience | Recommendation |
| --- | --- | --- |
| Python package using pip or uv | Developers optimizing their own ML or numerical computation | Primary release route; install in the environment that already contains their workload dependencies |
| Git clone plus editable install | Contributors and early source-preview testers | Supported development route once extraction produces an installable package |
| Tagged wheel or source archive | Pinned deployments and offline preparation | Publish alongside releases; offline use also needs compatible dependency wheels and any required toolchains |
| Homebrew | Users wanting a managed CLI installation | Later convenience route after deciding how it selects the workload's execution environment |
| Custom curl installer | Users seeking a bootstrap command | Defer; a separate installer adds maintenance without solving backend compatibility by itself |

Python packaging supports CLI entry points, wheels, optional dependency extras, and editable installs. A wheel is an installable distribution; it does not need to contain the Git history, research archive, or every supported framework. Sources: [PyPA installation guide](https://packaging.python.org/en/latest/tutorials/installing-packages/) and [packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

Default to the user's active project environment so its workload modules and installed frameworks are available. An isolated global tool install has a different environment: uv documents this distinction for `uvx` and recommends project execution when project dependencies are required. Offer an isolated demo route later if useful; do not present it as automatically able to import an arbitrary existing ML project. Sources: [uv tools](https://docs.astral.sh/uv/guides/tools/) and [uv package installation](https://docs.astral.sh/uv/pip/packages/).

Homebrew would require a maintained formula or tap and compatible releases. It should launch a declared worker environment or have its dependencies explicitly installed; discovering another environment's GPU does not make its Python packages importable. A future `brew install OWNER/tap/lossless-opt` would be illustrative until that tap exists. See the [Homebrew formula documentation](https://docs.brew.sh/Formula-Cookbook).

## Download to first result

For an existing project, activate its environment and install the package there. For a standalone demo, the proposed macOS/Linux path is:

```sh
# Future release workflow; not runnable against a published Lossless package yet.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install lossless-opt
lossless doctor
lossless demo --budget 60s
```

An existing uv user can install into the selected environment with `uv pip install lossless-opt`. Publish commands against a verified package name and tested version; use a pinned version for reproducibility examples. Prerelease instructions should explicitly select the prerelease version.

The core should include the CLI/library, config validation, hardware inventory, reports, small templates, and a tiny exact tensor/layout demo using bundled candidates. Aim for completion within the 60-second search budget, with startup separately explained. No weights, paid API, or theorem-prover download is needed for this demo. If a compiler is required, `doctor` should identify it before the demo; the first demo does not promise GPU optimization on unsupported hardware.

`doctor` should report the selected Python environment, device/runtime, installed adapters, usable capabilities, and missing requirements. Detection and support are distinct. It should print one concrete next step for a missing requirement and perform no implicit framework installation or model download.

For actual optimization, the intended flow is:

```sh
# Choose a template that is supported on the selected machine.
lossless init --template native-kernel --output my-job
# Edit my-job/lossless.json, reference code and fixtures; connect the user's LLM.
lossless inspect my-job/lossless.json
lossless optimize my-job/lossless.json --budget 3m --output lossless-runs/first
lossless report lossless-runs/first
lossless export lossless-runs/first --output optimized
```

The native template is a small numerical example. Vision and model-specific templates join the selector when their adapters are implemented and tested. JSON keeps the same workload/contract/objective/LLM/budget structure across those domains. `init` supplies a minimal reference/fixture example and an output ignore rule in a new job folder, without overwriting an existing project configuration. `inspect` shows the resolved contract, hardware, allowed transformations, resource estimate, and LLM sharing policy before tuning.

Users connect either their callback or a supported provider transport. Credentials stay outside the job file. Recommend capable reasoning models, with accepted gain and search cost reported; do not guarantee that model price or size predicts speedup. A reference failure or missing capability gives a diagnostic. No qualifying gain yields a clear reference-retained result with evidence.

The run produces a readable report, structured evidence, and a deployable bundle. Export includes a short adapter-specific integration example so the user knows where to load it. The application then uses the selected implementation for supported inputs and the reference when guards fail. It does not need an LLM connection during ordinary execution. Model weights remain referenced rather than copied by default.

## Backend dependencies and proof tools

Keep one core package initially. Use optional extras or adapter packages for supported integrations only; reserve separate distributions for dependencies or release schedules that actually need them. A proposed `lossless-opt[mlx]` extra would install the MLX integration once shipped. A PyTorch adapter must document the tested framework/device installation separately; an extra name cannot guarantee the correct CUDA runtime or driver.

The core should not pull in all frameworks, CUDA libraries, SciPy, or every provider SDK. Preserve compatible dependencies already present in the user's environment. Publish tested combinations and explain conflicts instead of silently promising that an installation preserves arbitrary existing environments. Compilation and native toolchain requirements belong in adapter documentation and `doctor`.

Bundle small recipe definitions, rules, and proof receipts where needed. Proof source remains public. Independent rechecking and generating new proofs require the appropriate optional checker/toolchain; installing those tools must be an explicit step when the requested job requires them. A shipped receipt records release-time evidence, not an independent local recheck or universal end-to-end equivalence proof. The package need not vendor an entire prover dependency cache.

### What installs automatically

The dependency list is still a proposal; the product has no `pyproject.toml` yet. Separate dependencies by the capabilities a user selects:

| Dependency | When needed | Installation owner |
| --- | --- | --- |
| Python | CLI/library and Python framework adapters | User's existing compatible environment or an explicitly managed environment |
| Core Python libraries | Parsing, schema validation and reporting | pip/uv installs declared package dependencies automatically; exact libraries remain to be selected |
| MLX or PyTorch integration dependencies | Corresponding supported workload adapter | Selected package extra or adapter installation; check existing framework compatibility |
| NumPy/SciPy | Numerical adapters that actually use them | Adapter dependencies, rather than mandatory imports for every job |
| Lean 4, its standard library and Lake | Checking or generating Lean proofs | Lossless setup reuses or installs the pinned toolchain, using Elan where appropriate |
| mathlib and its transitive Lean packages | Proofs or theorem tooling that import mathlib | Lake resolves the pinned project dependencies; setup fetches compatible compiled caches |
| Provider SDK | A built-in LLM transport that needs one | Transport extra; user callback mode does not require every provider SDK |
| Drivers and native compilers | Backend or candidate-build requirements | Detect and explain the platform-specific prerequisite; do not treat these as ordinary Python dependencies |

A Python extra can select Python dependencies, but a declaration such as `[proofs]` would not itself instruct pip to resolve a Lake project or install a GPU driver. Implement toolchain setup explicitly. The proposed proof setup experience is:

```sh
# Proposed product commands, not implemented commands.
python -m pip install lossless-opt
lossless setup --proofs mathlib
lossless doctor
```

The second command should handle the Lean toolchain, Lake dependencies, mathlib cache, and a small proof-check smoke test. A `--proofs lean` profile can install Lean alone. These are installation profiles, not different correctness relations. Show the selected versions, installation paths, expected download/disk use and progress; reuse matching installations and cache them across jobs. Store the toolchain pin and dependency manifest with the shipped proof project. Do not upgrade to an unpinned mathlib/Lean revision during a search. Updates require a tested product compatibility change.

Users should not need to learn Elan or Lake to use this workflow. Underneath, Elan manages Lean toolchains, Lake manages Lean project dependencies, and mathlib supplies mathematical definitions, theorems and tactics. Compatible compiled caches avoid rebuilding the whole library during ordinary setup. Sources: [Elan toolchains](https://lean-lang.org/doc/reference/latest/Build-Tools-and-Distribution/Managing-Toolchains-with-Elan/), [Lake project dependencies](https://lean-lang.org/doc/reference/latest/Build-Tools-and-Distribution/Lake/), and [mathlib project setup and caches](https://leanprover-community.github.io/install/project.html).

Mathlib is a proof library, not the numerical backend that executes a user's GPU computation. The research scheduling proof `experiments/theory_guided_017/Bounds.lean` imports `Std`; the retained layout/movement/fusion proofs likewise use standard or local modules. The separate `experiments/theorem_library/KernelTheorems.lean` imports mathlib matrix and linear-algebra modules. Its research configuration pins Lean 4.28.0 and a specific mathlib commit. These paths are provenance within the ignored research workspace; extract the necessary sources and pins before shipping.

Select dependencies from the actual proof obligations. A job requiring mathlib checks must diagnose a missing mathlib setup before search; it cannot silently downgrade to tests. Reusing an admitted artifact normally needs its execution dependencies, not the proof development environment, unless deployment policy requires an independent proof recheck. A bundled theorem index or receipt is useful context/evidence but does not replace a checker when checking is required.

On 2 October 2026, the existing Lean 4.28.0 toolchain occupied about 2.4 GiB outside this repository. The theorem project's `.lake` tree occupied about 1.7 GiB, including roughly 1.4 GiB inside its mathlib directory. The 1.4 GiB is a subset, not an additional amount. Together the toolchain and project tree occupied about 4.1 GiB; these measurements are allocated installed storage, not clean-install minima or compressed download estimates. The `.lake` tree is already included in the earlier research-directory size. Version, platform, cached imports and retained build artifacts affect the final footprint.

## Measured size and proposed budgets

The historical pre-extraction workspace measurement on 2 October 2026, before adding this plan, found 27 files eligible under the current ignore rules: 613,991 logical bytes, about 600 KiB. A temporary compressed snapshot was about 469 KiB. This consists mainly of design documents, examples and diagrams; it is not a built product package or a measurement of a future Git clone.

| Historical pre-extraction material | Measured size | Intended public delivery at that stage |
| --- | --- | --- |
| Product staging directory | About 780 KiB of allocated disk space | Reviewed docs/examples now; extracted implementation later |
| Research workspace | About 7.2 GiB of allocated disk space | Ignored locally; selected required evidence extracted separately |
| Three local virtual environments | About 2.4 GiB combined | Recreated from dependencies, never copied into the repository |
| Candidate public files | About 0.6 MiB of file contents | Initial source snapshot before the public repository was created |

Allocated disk usage differs from file contents and compressed download size. A clone also carries Git metadata and the requested history; growing binary history can exceed the current checkout. Installed size includes dependencies and unpacked binaries; it is not the wheel download size.

Use these as initial engineering budgets, not predicted release measurements:

| Component | Proposed budget or policy |
| --- | --- |
| Public source snapshot | Under 20 MiB for the early product, excluding Git history |
| Core wheel | Under 5 MiB compressed, excluding dependency downloads |
| Core plus lightweight dependencies | Under 50 MiB installed, excluding Python, GPU frameworks and optional proof tools |
| Backend environments | Measure per supported OS/framework/device combination; report additional download and installed bytes separately |
| Weights and representative datasets | User-supplied or explicitly fetched separately; no default model download |
| Proof toolchains and dependency caches | Optional separate installation with its own size disclosure |
| Generated candidates, compiler caches and run evidence | Explicit output locations, retention settings and disk limits; excluded from source releases |

The current environments illustrate that dependencies can occupy hundreds of MiB or more than a GiB even when our own source is small. They are research environments, not forecasts of any particular release configuration. A remotely connected reasoning LLM needs no local reasoning-model weights; a local reasoning model has its own separately accounted storage.

Measure each release in CI: wheel bytes, source archive bytes, unpacked core size, minimal clean installation, and each supported extra on its own platform. Report dependency download/cache costs separately from installed bytes. Exceeding a core budget should require an explained packaging change, not removal of valuable required correctness or performance code.

## Git ignores and release contents

Git ignore rules keep untracked local research and generated output out of normal staging. They do not erase tracked files or existing history. The public Git history excludes the ignored research workspace. See [Git ignore semantics](https://git-scm.com/docs/gitignore).

Package contents need explicit build configuration and an inspected file list as well. Define core package resources deliberately; exclude research, model weights, local environments, credentials and generated runs from both the wheel and source distribution. Test an installed wheel outside the checkout and ensure required templates, recipes, proof receipts and notices are present. Building from the promoted product directory gives a second, physical boundary around the archive.

Use GitHub release assets or another versioned store for large optional evidence bundles, with recorded content hashes and reproduction instructions. Keep the concise findings in `EXPERIMENTS.md` and the required small tests/proof sources in the repo. Ignoring research is an organizational choice, not a backup; preserve the local archive independently if it is the only copy.

## Release sequence

1. **Design preview:** clean source layout and honest planning docs; no install claim. Keep `lossless-product/` as staging until extraction works.
2. **Installable source alpha:** extracted core and selected adapters, no-key non-LLM demo, BYO LLM workflow, export/load and retention checks. Contributors can clone and use `python -m pip install -e '.[dev]'` from the promoted root once the development extra exists.
3. **Packaged alpha:** publish tested wheels/source distribution under a verified name, with explicit supported targets, measured sizes and a complete quickstart. Keep the license and notices settled before distribution.
4. **Convenience channels:** add Homebrew or an isolated launcher after their workload-environment behavior is implemented and tested. Add more integrations through the same support and evidence gates.

Before declaring the packaged alpha ready, reproduce installation and first use on clean supported machines without the research tree; inspect both archives; verify the no-win/fallback path and LLM connection; and reproduce the retained gains. Do not publish a generic vision or CUDA speed claim based only on the current MLX results.
