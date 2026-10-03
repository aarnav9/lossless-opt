# Lossless 0.1.0a1 — first public alpha

This prerelease packages the tested native optimizer, the guarded MLX fixed-count adapter, and the new `lossless profile` command. Source, wheel, source archive, checksums and optional softmax qualification evidence are available from the [GitHub prerelease](https://github.com/aarnav9/lossless-opt/releases/tag/v0.1.0a1). PyPI and Homebrew publication remain future work.

## Install and try

Requires Python 3.10+ and clang on macOS or Linux. Use Python 3.11 to match release validation. The base package requires NumPy; GPU models and proof tools are separate.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install https://github.com/aarnav9/lossless-opt/releases/download/v0.1.0a1/lossless_opt-0.1.0a1-py3-none-any.whl
lossless doctor
lossless init --template copy --output ./copy-demo
lossless profile ./copy-demo/lossless.json --budget 30s --output ./lossless-runs/profile
lossless optimize ./copy-demo/lossless.json --budget 60s --output ./lossless-runs/demo
lossless export ./lossless-runs/demo --output ./lossless-runs/copy-artifact
```

A valid outcome can retain the reference. The acceptance comparator is explicit and frozen before search. The CPU demo needs no GPU, model download, LLM key or proof toolchain. Read [profiling](profiling.md) for interpreting timings and [the alpha guide](alpha.md) for connection and deployment details.

## Supported scope

| Adapter | Runtime/hardware | Contract |
| --- | --- | --- |
| `native.copy` | macOS/Linux, clang, NumPy; generated float32 matrices | Tested bitwise copy and independent output |
| `native.softmax` | Same; optional SciPy comparator | Explicit numerical tolerances, finite bounded inputs |
| `native.rmsnorm_residual` | Same | Explicit numerical tolerances, finite bounded inputs |
| `mlx.fixed_count` | Apple M2, MLX 0.32.3 / mlx-lm 0.32.0, exact pinned 8-bit SmolLM2 identity | Sampled bitwise tokens, probabilities and active KV; fixed-count greedy requests in an exclusive process |

Use `lossless doctor` to inspect the current environment. MLX optimization is admitted only for its named model/runtime/device. The wheel includes no model weights. The model fingerprint and qualification scope are in [retention](retention.md); arbitrary models, CUDA, Windows native execution, streaming serving and end-to-end formal verification are not supported by this alpha.

## Included changes

- Native acceptance and retained-reference deployment use the declared comparator. MLX respects configured provider timeouts.
- Compatible native compilation/discovery evidence can be cached; all timing and evaluation checks stay fresh. Duplicate sources are rejected before benchmarking.
- Native bound calls avoid benchmark-only initialization; MLX reuses one fingerprint during admission.
- 026/027 graph alternatives remain opt-in, with bounded specialization counts, fallback, startup accounting and fresh incremental gates.
- `lossless profile` measures discovery inputs, startup/warm timing, memory, latency and separate host attribution without search/provider calls. It can compare a guarded exported artifact.
- Campaign 039 expanded softmax qualification to 225 case/process measurements and 14,850 successful correctness checks. Repeated small/tail regressions kept the candidate experimental; no new default softmax recipe is claimed.

## Validation and limitations

Release checks cover the CPU suite, static/formatting checks, built archive contents, clean wheel installation outside the repository, profiling, exact copy export/load and theorem-resource availability. Local Metal checks cover sampled full-output/state equality, profiling stage accounting, artifact deployment, continuation and cancellation. GitHub's Linux/macOS checks cover CPU paths; they do not qualify additional Metal devices.

Candidate code and provider callbacks run as trusted local processes with timeouts, not inside an operating-system security sandbox. The provider interface is tested; hosted-provider integration and independent LLM-search cost effectiveness remain unvalidated. Timing intervals and finite checks have the scopes stated in their reports.

Historical artifacts can fall back when package/runtime fingerprints change. Reoptimize and export under the installed release before relying on a graph artifact. No optimization is guaranteed to win, and observed bulk throughput can trade off first-token latency.
