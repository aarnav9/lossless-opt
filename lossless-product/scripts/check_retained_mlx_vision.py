"""Local retained-backend conformance smoke. No downloads or authoring calls."""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
import json
from datetime import datetime, timezone
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from lossless.retained.mlx_vision import candidate

out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)


def stamp(s):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {s}", flush=True)


stamp(f"START estimate=under30sec result={out}/results.json log={out}.log")
mx.set_default_device(mx.gpu)
rng = np.random.default_rng(4220291)
rows = []
for task in ["L1_42", "L2_65"]:
    config = dict(kernel=4, stride=1, padding=1) if task == "L1_42" else {}
    p = (
        {}
        if task == "L1_42"
        else {
            "conv.weight": mx.array(rng.normal(size=(13, 7, 3, 3)).astype("f") * 0.03),
            "conv.bias": mx.array(rng.normal(size=13).astype("f") * 0.1),
        }
    )

    def reference(x):
        if task == "L1_42":
            return nn.MaxPool2d(4, stride=1, padding=1)(x.transpose(0, 2, 3, 1)).transpose(
                0, 3, 1, 2
            )
        y = (
            mx.conv2d(x.transpose(0, 2, 3, 1), p["conv.weight"].transpose(0, 2, 3, 1))
            + p["conv.bias"]
        )
        y = nn.AvgPool2d(4, stride=4)(y)
        return mx.sum(mx.sigmoid(y), axis=(1, 2, 3))

    operation = candidate(task, config, p, reference, contract="numerical")
    for n, h, w, transpose in [(2, 27, 31, False), (3, 29, 25, True), (1, 19, 23, False)]:
        x = mx.array(rng.normal(size=(n, 7, h, w)).astype("f"))
        if transpose:
            x = x.transpose(0, 1, 3, 2)
        expected = reference(x)
        actual = operation(x)
        mx.eval(expected, actual)
        a, b = np.array(expected), np.array(actual)
        np.testing.assert_allclose(b, a, atol=2e-4, rtol=2e-4)
        rows.append(
            dict(
                task=task,
                shape=list(x.shape),
                exact=a.tobytes() == b.tobytes(),
                max_abs=float(np.max(abs(a - b))),
            )
        )
    stamp(f"PASS MLX recipe task={task} cases=3")
(out / "results.json").write_text(json.dumps(dict(checks=rows), indent=2) + "\n")
stamp(f"COMPLETE result={out}/results.json log={out}.log")
