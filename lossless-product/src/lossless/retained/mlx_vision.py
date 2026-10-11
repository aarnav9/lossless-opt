"""The two qualified campaign-049 task candidates, under its numerical mode."""


def candidate(task, config, parameters, reference, *, contract):
    import importlib.metadata
    import mlx.core as mx
    from ._mlx_vision import Runner

    if contract != "numerical":
        raise ValueError("campaign 049 acceptance was numerical, including pooling")
    if task not in {"L1_42", "L2_65"}:
        raise ValueError("only max pool and conv/average-pool/sigmoid/sum passed all task gates")
    if importlib.metadata.version("mlx") != "0.32.3" or not mx.metal.is_available():
        raise ValueError("requires the recorded MLX 0.32.3 Metal runtime")
    if task == "L1_42" and config != dict(kernel=4, stride=1, padding=1):
        raise ValueError("retained pooling scope is kernel=4, stride=1, padding=1")
    if task == "L2_65":
        weight = parameters.get("conv.weight")
        if weight is None or weight.ndim != 4 or tuple(weight.shape[-2:]) != (3, 3) or config:
            raise ValueError("requires 3x3 stride-one valid convolution and fixed 4x4 average pool")
    owned = {k: mx.array(v) for k, v in parameters.items()}
    if any(
        v.dtype != mx.float32 or not bool(mx.all(mx.isfinite(v)).item()) for v in owned.values()
    ):
        raise ValueError("finite FP32 immutable parameters required")
    runner = Runner(task, dict(config), owned, reference)

    def run(x):
        if (
            x.dtype != mx.float32
            or x.ndim != 4
            or min(x.shape) < 1
            or max(x.shape) > 2**15
            or x.size > 2**30
        ):
            return reference(x)
        # The direct max-pool kernel uses a finite negative sentinel. Nonfinite
        # inputs need the reference's NaN/Inf semantics, including all -Inf windows.
        if not bool(mx.all(mx.isfinite(x)).item()):
            return reference(x)
        if task == "L2_65" and (
            min(x.shape[2:]) < 6 or x.shape[1] != owned["conv.weight"].shape[1]
        ):
            return reference(x)
        return runner.run(x)

    return run
