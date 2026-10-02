# Lossless

> An optimizer that combines formal reasoning, hardware information, and measured experiments to accelerate computations while preserving an explicit correctness contract.

**Installable source alpha (`0.1.0a1`), private development repository.** The product now includes a CLI, JSON contracts, bounded search, bring-your-own LLM callbacks/commands, hardware discovery, validation, reports, and guarded export/load. Supported adapters cover native tensor copy, softmax, RMSNorm plus residual, and the retained exact MLX inference recipe.

```sh
git clone https://github.com/aarnav9/lossless-opt.git
cd lossless-opt
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ./lossless-product
lossless doctor
lossless demo --budget 60s --output ./lossless-runs/demo
```

Python 3.10+ and clang are required for the CPU demo. Python dependencies install automatically; Lean/mathlib, GPU frameworks and weights are optional separate installations. Private-repo access is required. Nothing has been published to PyPI or Homebrew.

The broader product targets GPU optimization for ML, vision, and linear algebra. Arbitrary model graphs, CUDA execution and a vision adapter remain to be implemented. Exact MLX admission currently preserves the researched Apple M2, runtime and model-identity guards. Scoped Lean checks and finite floating-point validation remain distinct evidence.

- [Product quickstart and supported adapters](lossless-product/README.md)
- [Implemented configuration, providers and deployment](lossless-product/docs/alpha.md)
- [Research retention and validation](lossless-product/docs/retention.md)
- [Architecture diagrams and future design](lossless-product/docs/architecture.md)
- [Experiment ledger](EXPERIMENTS.md)
- [Contributing](CONTRIBUTING.md)

```text
lossless-product/      Installable package, adapters, proofs, tests and docs
EXPERIMENTS.md         Concise experiment/result memory
.github/workflows/    CPU checks and isolated package installation
research/             Preserved local research archive; ignored by Git
```

The product runs independently of `research/`. Large experiments, results, model weights, virtual environments and generated optimization runs stay local. Original code is [MIT licensed](LICENSE); retained [third-party notices](lossless-product/NOTICE) ship with the package. Public distribution remains pending.
