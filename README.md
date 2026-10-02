# Lossless

> An optimizer that combines formal reasoning, hardware information, and measured experiments to accelerate computations while preserving an explicit correctness contract.

Lossless is designed for GPU-focused optimization of ML and numerical computations, including language models, computer vision, and linear algebra. Supported CPU workloads also remain in scope. Broader domain support must be implemented and measured; existing research gains apply to their recorded workloads.

**Status: product design and research extraction.** The public product is not yet installable. Proposed interfaces, examples, and diagrams describe the planned behavior; performance entries summarize scoped research, not released product guarantees.

- [Product overview](lossless-product/README.md)
- [Architecture and computation lifecycle](lossless-product/docs/architecture.md)
- [Job configuration and contracts](lossless-product/docs/configuration.md)
- [Experiment ledger](EXPERIMENTS.md): what was tried, what worked, and what failed
- [Repository and release plan](lossless-product/docs/repository.md)
- [Installation, first use and package size plan](lossless-product/docs/distribution.md)

## Repository layout

```text
README.md               Project entry point
EXPERIMENTS.md          Public experiment summaries
lossless-product/      Product design, examples, and future implementation
research/              Local research workspace; ignored by Git
```

`lossless-product/` is the clean product boundary. `research/` contains the historical runners, experiments, plans, detailed reports, raw results, proofs, and downloaded artifacts. It is kept locally but excluded from the proposed public repository. Existing hidden virtual environments also remain local and ignored.

Contributors should read `EXPERIMENTS.md` before proposing a repeat investigation. A new experiment should state what changes relative to prior work and add an experiment/result pair, including a negative or inconclusive result when applicable. An accepted product optimization must bring its required code, tests, proof scope, notices, and reproducible evidence into the public product; it must not depend on the ignored workspace.

For local research, start with `research/WORKSPACE.md` and run historical commands from `research/`. This is the initial development snapshot: design documents and illustrative configurations. The installable optimizer is still to be extracted and implemented. Repository name: **`lossless-opt`**; product name: **Lossless**.
