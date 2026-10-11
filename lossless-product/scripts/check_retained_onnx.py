"""Local retained-backend conformance smoke. No downloads or authoring calls."""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--onnx-include", type=Path, required=True)
args = parser.parse_args()
import os, json
from datetime import datetime, timezone

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import onnxruntime as ort
from lossless.retained.onnx import build, rewrite

out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)


def stamp(s):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {s}", flush=True)


stamp(f"START estimate=under30sec result={out}/results.json log={out}.log")
lib = build(out / "native", include_directory=args.onnx_include)
rows = []
for shape in [(3, 17), (7, 31), (1, 4, 19)]:
    rng = np.random.default_rng(79403 + shape[-1])
    w = rng.uniform(0.8, 1.2, shape[-1]).astype("f")
    constants = [
        nh.from_array(w, "region.weight"),
        nh.from_array(np.array(1e-5, dtype="f"), "eps"),
        nh.from_array(np.array(1, dtype="f"), "one"),
    ]
    specs = [
        ("Mul", ["x", "x"], "squared", {}),
        ("ReduceMean", ["squared"], "mean", dict(axes=[-1], keepdims=1)),
        ("Add", ["mean", "eps"], "variance", {}),
        ("Sqrt", ["variance"], "root", {}),
        ("Div", ["one", "root"], "inv", {}),
        ("Mul", ["x", "inv"], "scaled", {}),
        ("Mul", ["scaled", "region.weight"], "weighted", {}),
        ("Add", ["weighted", "r"], "y", {}),
    ]
    nodes = [
        h.make_node(op, inp, [dest], name="/region/" + dest, **attrs)
        for op, inp, dest, attrs in specs
    ]
    model = h.make_model(
        h.make_graph(
            nodes,
            "rms",
            [h.make_tensor_value_info(n, T.FLOAT, shape) for n in ["x", "r"]],
            [h.make_tensor_value_info("y", T.FLOAT, shape)],
            initializer=constants,
        ),
        opset_imports=[h.make_opsetid("", 17)],
        ir_version=8,
    )
    rewritten, record = rewrite(model, contract="numerical")
    assert record["replaced"] == ["region"]
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.register_custom_ops_library(str(lib))
    candidate = ort.InferenceSession(
        rewritten.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )
    reference = ort.InferenceSession(
        model.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )
    for _ in range(3):
        inputs = {n: rng.normal(size=shape).astype("f") for n in ["x", "r"]}
        before = [v.tobytes() for v in inputs.values()]
        expected = reference.run(None, inputs)[0]
        actual = candidate.run(None, inputs)[0]
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=5e-5)
        assert before == [v.tobytes() for v in inputs.values()]
        rows.append(dict(shape=shape, max_abs=float(np.max(abs(actual - expected)))))
    # Corrupt epsilon: must decline source rewrite.
    invalid = onnx.ModelProto()
    invalid.CopyFrom(model)
    invalid.graph.initializer[1].CopyFrom(nh.from_array(np.array(0.01, dtype="f"), "eps"))
    assert not rewrite(invalid, contract="numerical")[1]["replaced"]
stamp("PASS ONNX full graph: 9 numerical calls and 3 invalid-epsilon controls")
(out / "results.json").write_text(
    json.dumps(dict(checks=rows, ort=ort.__version__), indent=2) + "\n"
)
stamp(f"COMPLETE result={out}/results.json log={out}.log")
