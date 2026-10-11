"""Retained narrow named-region ONNX matcher; unsupported graphs decline."""

import copy


def rewrite(model, parallel=False):
    import numpy as np
    import onnx

    result = onnx.shape_inference.infer_shapes(copy.deepcopy(model))
    nodes = list(result.graph.node)
    group = [n for n in nodes if n.name.startswith("/region/")]
    if not group:
        return result, {"replaced": [], "skipped": ["no_named_region"]}

    def reject(reason):
        return result, {"replaced": [], "skipped": [reason]}

    producers = {o: n for n in nodes for o in n.output}
    names = {n.name for n in group}
    constants = {i.name: onnx.numpy_helper.to_array(i) for i in result.graph.initializer}
    for n in nodes:
        if n.op_type == "Constant" and len(n.attribute) == 1 and n.attribute[0].name == "value":
            constants[n.output[0]] = onnx.numpy_helper.to_array(n.attribute[0].t)

    def scalar(name, value):
        return (
            name in constants
            and constants[name].size == 1
            and bool(np.isclose(float(constants[name].reshape(-1)[0]), value, rtol=1e-7, atol=0))
        )

    info = {
        v.name: v.type.tensor_type
        for v in [*result.graph.input, *result.graph.value_info, *result.graph.output]
    }
    if any(
        name not in info or info[name].elem_type != onnx.TensorProto.FLOAT for name in ("x", "r")
    ):
        return reject("dtype")
    shape = lambda name: [d.dim_value for d in info[name].shape.dim]
    if shape("x") != shape("r") or not 1 <= len(shape("x")) <= 8 or not all(shape("x")):
        return reject("shape")
    weight = constants.get("region.weight")
    if weight is None or weight.dtype != np.float32 or weight.shape != (shape("x")[-1],):
        return reject("weight")
    reductions = [n for n in group if n.op_type == "ReduceMean"]
    if len(reductions) != 1:
        return reject("reduction_count")
    attributes = {a.name: onnx.helper.get_attribute_value(a) for a in reductions[0].attribute}
    if attributes.get("axes") != [-1] or attributes.get("keepdims", 1) != 1:
        return reject("reduction_semantics")
    eps = [n for n in group if n.op_type == "Add" and any(scalar(i, 1e-5) for i in n.input)]
    if len(eps) != 1:
        return reject("epsilon")
    pre_add = [n for n in group if n.op_type == "Add" and set(n.input) == {"x", "r"}]
    pre = bool(pre_add)
    z = pre_add[0].output[0] if pre else "x"
    square = producers.get(reductions[0].input[0])
    if square is None or square.op_type != "Mul" or list(square.input) != [z, z]:
        return reject("square")
    if reductions[0].output[0] not in eps[0].input:
        return reject("mean_epsilon_edge")
    roots = [n for n in group if n.op_type == "Sqrt" and list(n.input) == [eps[0].output[0]]]
    if len(roots) != 1:
        return reject("sqrt")
    inverses = [
        n
        for n in group
        if n.op_type == "Div" and scalar(n.input[0], 1.0) and n.input[1] == roots[0].output[0]
    ]
    if len(inverses) != 1:
        return reject("reciprocal")
    scales = [n for n in group if n.op_type == "Mul" and set(n.input) == {z, inverses[0].output[0]}]
    if len(scales) != 1:
        return reject("scale")
    weighted = [
        n
        for n in group
        if n.op_type == "Mul" and set(n.input) == {scales[0].output[0], "region.weight"}
    ]
    if len(weighted) != 1:
        return reject("weight_scale")
    final = weighted[0]
    if not pre:
        adds = [
            n for n in group if n.op_type == "Add" and set(n.input) == {weighted[0].output[0], "r"}
        ]
        if len(adds) != 1:
            return reject("post_residual")
        final = adds[0]
    outputs = [final.output[0]] + ([z] if pre else [])
    internal = {o for n in group for o in n.output}
    external = {i for n in nodes if n.name not in names for i in n.input} | {
        o.name for o in result.graph.output
    }
    if (internal & external) - set(outputs):
        return reject("shared_internal_intermediate")
    allowed = {"Constant", "Add", "Mul", "ReduceMean", "Sqrt", "Div"}
    if any(n.op_type not in allowed for n in group):
        return reject("unsupported_op")
    custom = onnx.helper.make_node(
        "Rms" + ("Pre" if pre else "Post") + ("Parallel" if parallel else "Serial"),
        ["x", "r", "region.weight"],
        outputs,
        name="fused_rms_residual",
        domain="kernel_lab.rms",
    )
    rewritten = []
    for n in nodes:
        if n.name == group[-1].name:
            rewritten.append(custom)
        elif n.name not in names:
            rewritten.append(n)
    del result.graph.node[:]
    result.graph.node.extend(rewritten)
    result.opset_import.append(onnx.helper.make_opsetid("kernel_lab.rms", 1))
    onnx.checker.check_model(result)
    return result, {
        "replaced": ["region"],
        "skipped": [],
        "mode": "pre" if pre else "post",
        "removed_nodes": len(group),
        "outputs_preserved": outputs,
    }
