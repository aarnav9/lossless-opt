"""KernelBench-derived M2 workload definitions; no upstream default sizes are used."""

TASKS = {
    "L1_02": "Matrix multiplication",
    "L1_03": "Batched matrix multiplication",
    "L1_19": "ReLU",
    "L1_40": "Layer normalization",
    "L1_48": "Mean reduction",
    "L1_42": "Max pooling 2D",
    "L1_50": "Convolution 11x11",
    "L1_86": "Depthwise-separable convolution",
    "L2_01": "Convolution / ReLU / bias",
    "L2_09": "Linear / subtract / multiply / ReLU",
    "L2_12": "Linear / multiply / leaky ReLU",
    "L2_40": "Linear / scale / residual",
    "L2_56": "Linear / sigmoid / sum",
    "L2_59": "Linear / swish / scale",
    "L2_65": "Convolution / average pool / sigmoid / sum",
    "L2_67": "Convolution / GELU / global average pool",
    "L3_01": "MLP",
    "L3_04": "LeNet-5",
    "L3_06": "Inception module",
    "L3_17": "Fire module",
}


def cases(split):
    """Two shape/layout cases per task; evaluation is never sent to the author."""
    final = split == "evaluation"
    result = []
    for task in TASKS:
        for size in range(2):
            n = ([3, 7] if final else [2, 4])[size]
            ci = ([7, 17] if final else [8, 16])[size]
            co = ([13, 29] if final else [16, 32])[size]
            hw = ([23, 47] if final else [24, 48])[size]
            m, k, cols = (
                [(37, 127, 83), (257, 513, 385)] if final else [(64, 128, 96), (512, 512, 384)]
            )[size]
            config = {}
            if task in {"L1_02", "L1_03"}:
                shapes = [[m, k], [k, cols]]
                if task == "L1_03":
                    shapes = [[n, *s] for s in shapes]
                init = []
            elif task == "L1_19":
                shapes, init = [[m * 4, k * 4]], []
            elif task == "L1_40":
                shapes, init = [[m, k]], [[k]]
                config = dict(eps=1e-5)
            elif task == "L1_48":
                shapes, init = [[n, m, k]], [1]
                config = dict(axis=1)
            elif task == "L1_42":
                shapes, init = [[n, ci, hw, hw + int(final)]], [4, 1, 1, 1]
                config = dict(kernel=4, stride=1, padding=1)
            elif task == "L1_50":
                shapes, init = [[n, 3, hw * 2, hw * 2]], [1000]
            elif task == "L1_86":
                shapes, init = [[n, ci, hw, hw + int(final)]], [ci, co, 3, 1, 1, 1, False]
                config = dict(groups=ci)
            elif task == "L2_01":
                shapes, init = [[n, ci, hw, hw + int(final)]], [ci, co, 3, [co, 1, 1]]
            elif task.startswith("L2_") and task not in {"L2_65", "L2_67"}:
                shapes = [[m, k]]
                constants = {
                    "L2_09": [2.0, 1.5],
                    "L2_12": [2.0, 0.1],
                    "L2_40": [0.5],
                    "L2_56": [],
                    "L2_59": [2.0],
                }[task]
                init = [k, cols, *constants]
            elif task in {"L2_65", "L2_67"}:
                shapes, init = [[n, ci, hw, hw + int(final)]], [ci, co, 3]
                if task == "L2_65":
                    init.append(4)
            elif task == "L3_01":
                shapes, init = [[m, k]], [k, [cols, k], 31 if final else 32]
                config = dict(layers=3)
            elif task == "L3_04":
                shapes, init = [[n * 4, 1, 32, 32]], [20]
            elif task == "L3_06":
                shapes, init = [[n, ci, hw, hw + int(final)]], [ci, co, ci, co, 4, 8, 8]
            elif task == "L3_17":
                shapes, init = [[n, ci, hw, hw + int(final)]], [ci, 6, co, co]
            else:
                raise ValueError(task)
            result.append(
                dict(
                    id=f"{split}_{task}_{size}",
                    task=task,
                    size=size,
                    shapes=shapes,
                    init=init,
                    config=config,
                    layout="transposed_view" if size else "contiguous",
                    seed=(91000 if final else 19000) + list(TASKS).index(task) * 101 + size,
                )
            )
    return result
