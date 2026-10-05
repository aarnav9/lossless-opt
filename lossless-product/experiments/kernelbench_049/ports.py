"""Direct float32 MLX ports of the 20 vendored MIT KernelBench computations.

Public ABI uses the upstream NCHW / row-major logical shapes and parameter names.
Conversions and activation work below are inside the measured call. See upstream/LICENSE.
"""

import mlx.core as mx
import mlx.nn as nn


def execute(task, config, p, *args):
    x = args[0]

    def conv(v, name, stride=1, padding=0, groups=1):
        w = p[name + ".weight"].transpose(0, 2, 3, 1)
        v = mx.conv2d(v.transpose(0, 2, 3, 1), w, stride=stride, padding=padding, groups=groups)
        if name + ".bias" in p:
            v = v + p[name + ".bias"]
        return v.transpose(0, 3, 1, 2)

    def linear(v, name):
        return mx.addmm(p[name + ".bias"], v, p[name + ".weight"].T)

    def pool(v, kernel, stride, padding=0, average=False):
        cls = nn.AvgPool2d if average else nn.MaxPool2d
        return cls(kernel, stride=stride, padding=padding)(v.transpose(0, 2, 3, 1)).transpose(
            0, 3, 1, 2
        )

    if task in {"L1_02", "L1_03"}:
        return x @ args[1]
    if task == "L1_19":
        return mx.maximum(x, 0)
    if task == "L1_40":
        return mx.fast.layer_norm(x, p["ln.weight"], p["ln.bias"], config["eps"])
    if task == "L1_48":
        return mx.mean(x, axis=config["axis"])
    if task == "L1_42":
        return pool(x, config["kernel"], config["stride"], config["padding"])
    if task == "L1_50":
        return conv(x, "conv1", stride=4, padding=2)
    if task == "L1_86":
        return conv(conv(x, "depthwise", padding=1, groups=config["groups"]), "pointwise")
    if task == "L2_01":
        return mx.maximum(conv(x, "conv"), 0) + p["bias"]
    if task == "L2_09":
        return mx.maximum((linear(x, "linear") - 2.0) * 1.5, 0)
    if task == "L2_12":
        y = linear(x, "gemm") * 2.0
        return mx.where(y >= 0, y, y * 0.1)
    if task == "L2_40":
        y = linear(x, "matmul")
        return y * 0.5 + y
    if task == "L2_56":
        return mx.sum(mx.sigmoid(linear(x, "linear")), axis=1, keepdims=True)
    if task == "L2_59":
        y = linear(x, "matmul")
        return y * mx.sigmoid(y) * 2.0
    if task == "L2_65":
        y = pool(conv(x, "conv"), 4, 4, average=True)
        return mx.sum(mx.sigmoid(y), axis=(1, 2, 3))
    if task == "L2_67":
        return mx.mean(nn.gelu(conv(x, "conv")), axis=(2, 3))
    if task == "L3_01":
        for i in range(config["layers"]):
            x = linear(x, f"network.{2 * i}")
            if i < config["layers"] - 1:
                x = mx.maximum(x, 0)
        return x
    if task == "L3_04":
        x = pool(mx.maximum(conv(x, "conv1"), 0), 2, 2)
        x = pool(mx.maximum(conv(x, "conv2"), 0), 2, 2)
        x = x.reshape(x.shape[0], 400)
        x = mx.maximum(linear(x, "fc1"), 0)
        x = mx.maximum(linear(x, "fc2"), 0)
        return linear(x, "fc3")
    if task == "L3_06":
        return mx.concatenate(
            [
                conv(x, "branch1x1"),
                conv(conv(x, "branch3x3.0"), "branch3x3.1", padding=1),
                conv(conv(x, "branch5x5.0"), "branch5x5.1", padding=2),
                conv(pool(x, 3, 1, 1), "branch_pool.1"),
            ],
            axis=1,
        )
    if task == "L3_17":
        x = mx.maximum(conv(x, "squeeze"), 0)
        return mx.concatenate(
            [
                mx.maximum(conv(x, "expand1x1"), 0),
                mx.maximum(conv(x, "expand3x3", padding=1), 0),
            ],
            axis=1,
        )
    raise ValueError(task)


def make(task, config, parameters, mode):
    def run(*args):
        return execute(task, config, parameters, *args)

    if mode == "eager":
        return run
    if mode in {"compiled", "shapeless"}:
        return mx.compile(run, shapeless=mode == "shapeless")
    raise ValueError(mode)
