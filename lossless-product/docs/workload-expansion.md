# Broader optimization workloads after campaign 048

**Yes: broaden the workload corpus. Actual model training is a separate decision.**
Lossless currently supplies tasks, source, hardware and benchmark feedback to an
existing author model; it does not update that model's weights. More varied
discovery and held-out workloads would improve our experiments immediately.
Calling them training data would obscure that distinction.

The initial **20-task pilot is now implemented and measured in
[campaign 049](kernelbench-049.md)**. All tasks passed final numerical checks;
max pooling and a convolution/average-pooling pipeline passed their full task
speed gates. The remaining recommendations below guide expansion and qualification.

## Where KernelBench helps

[KernelBench](https://github.com/ScalingIntelligence/KernelBench) includes individual
operators, fusion patterns and complete architectures, including vision models.
It is therefore a useful source of tasks beyond language-model inference. Its
native task is translating PyTorch programs into efficient GPU implementations;
the documented runtime targets CUDA and additional GPU backends/DSLs. The
[paper](https://arxiv.org/abs/2502.10517) describes its correctness and performance
evaluation. This does not supply a drop-in Apple MLX evaluator.

Our proposed M2 adaptation would be a **KernelBench-derived MLX suite**, with
explicitly validated ports and its own baselines. We should not report its scores
as an official KernelBench CUDA result. A future CUDA adapter could execute a
pinned upstream task set on supported hardware instead. Neither path was run in
campaign 048, and no cloud GPU was rented.

The initial **20-task pilot** uses eight individual operators, eight
short fusion patterns, and four graphs (two complete small networks and two vision
modules). The broader coverage categories to develop are:

| Family | What it tests beyond current softmax/LLM work |
| --- | --- |
| Convolution, pooling, image blocks | Spatial layouts, channel counts, depthwise/grouped operations |
| Matrix multiplication and epilogues | Rectangular/batched shapes, packing, fused elementwise work |
| Normalization and reductions | Reduction trees, cancellation, small/large dimensions |
| Elementwise chains and tensor movement | Fusion, launch overhead, allocation, strides and views |
| Small CNN/MLP graphs | Whether local gains survive complete-graph execution |

These are proposed coverage categories, not a claim that every named variant is
already ported or supported by Lossless. Choose the exact upstream task IDs and
revision before generating proposals, preserve license/provenance, and retain
failed ports as unsupported rather than silently changing their semantics.

## Separate three kinds of held-out evidence

1. **Implementation discovery:** source and an explicit subset of shapes and
   distributions may be used for optimization. Preserve all failed attempts.
2. **Per-task final evaluation:** freeze the implementation before opening new
   dimensions, strides/layouts, edge cases and random seeds. Include irregular
   dimensions, degenerate sizes, extreme values and cancellation where relevant.
3. **Optimizer transfer:** hold out entire operator/fusion/graph families when
   testing whether a learned recipe selector or future trained author generalizes.
   Merely changing input seeds cannot establish that claim.

Public tasks might already be familiar to an author model; an unseen local seed
does not establish training-data decontamination. Record public provenance and
avoid claiming tasks were absent from model training.

Each task needs a frozen observable contract. Bitwise equality, numerical error
bounds and task-quality metrics remain separate categories. Do not import a
benchmark's numerical tolerance and call the result exact. Compare to the fastest
applicable baseline selected on discovery, including compiled/library paths where
they satisfy the same contract. Record whole-call time, setup/compilation, warm
time, peak memory and payback. Score correctness and speed together; a fast wrong
candidate is a failure.

The upstream [evaluation guide](https://github.com/ScalingIntelligence/KernelBench/blob/main/EVAL.md)
also calls for independent verification and describes timing/checker pitfalls.
We should retain separate reference execution and negative controls, and use
backend-specific synchronized timing. This is particularly relevant when porting
task semantics across frameworks.

## What to collect before considering training

Store complete optimization trajectories: task/source revision, contract,
hardware/runtime/compiler identities, prompt and proposal, generated source,
failures/counterexamples, raw timings, memory/setup observations, frozen baseline,
selection, untouched final result, author time and available usage receipts.
Deduplicate source and related task families before splitting a future training
corpus. An unsuccessful proposal with a precise failure can be useful evidence;
retain it rather than collecting winners only.

First use this corpus to test recipe retrieval and a simple non-LLM search
baseline at matched evaluation budgets. Consider fine-tuning only after the
independent evaluation suite is stable and the corpus is large/diverse enough
to test transfer. More experiments alone do not establish that training will
improve the optimizer.
