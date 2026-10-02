# Lossless

> An optimizer that combines formal reasoning, hardware information, and measured experiments to accelerate computations while preserving an explicit correctness contract.

**Status: product design draft.** This folder establishes the clean product boundary. There is no installable product here yet; commands and APIs in the design are proposals.

The proposed product takes a computation, its reference behavior, representative workloads, a correctness contract, and a resource budget. It returns an optimized implementation with evidence and runtime guards, or keeps the reference when no candidate qualifies.

Lossless is designed for GPU-focused optimization of ML and numerical computations, including language models, computer vision, and linear algebra. Supported CPU workloads also remain in scope. Broader domain support must be implemented and measured; existing research gains apply to their recorded workloads.

The user-connected LLM helps discover optimizations even when the target is a vision model or numerical function. Computer-vision support is a design requirement; vision performance gains have not yet been established.

Start with the [system design and user experience](docs/design.md), including a [concrete contract example](docs/design.md#a-concrete-contract). The [configuration proposal](docs/configuration.md) covers inline or split JSON inputs, optional YAML authoring, hardware discovery, and how the harness builds context for the user's LLM. The [implementation and extraction plan](docs/implementation.md) maps the existing research gains into product components and defines release gates.

The [architecture diagrams and computation lifecycle](docs/architecture.md) show the optimization pipeline, its outputs, where the deployment bundle enters an application, and an LLM inference case study for the retained recipes.

The [distribution and first use plan](docs/distribution.md) recommends package installation for users, source checkout for contributors, optional backends, and explicit size budgets.

The [repository and release plan](docs/repository.md) defines the public layout, local research boundary, naming recommendation, and remaining packaging work. The outer workspace's `EXPERIMENTS.md` keeps the concise experiment/result history for contributors.

The general optimizer direction, bring-your-own LLM support, and preservation of the valuable research gains are product requirements. Interface languages, the initial set of execution targets, and the default correctness policy remain design decisions. The current recommendation is a CLI with a Python orchestration library and native artifacts where the target supports them. Choose adapters around the gains we intend to retain; do not impose a one-backend limit that discards them.

Product code will live entirely inside this folder. It must install and run after this folder is copied out of the research workspace. Research history, weights, generated runs, and local environments stay outside the distributable product. Publication and the original-code license remain to be decided.
