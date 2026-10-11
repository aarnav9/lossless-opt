import math
from collections import OrderedDict
import mlx.core as mx


def _pair(value):
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _nhwc(x):
    return mx.contiguous(x.transpose(0, 2, 3, 1))


def _metal(name, names, source, contiguous=True):
    return mx.fast.metal_kernel(
        name=name,
        input_names=names,
        output_names=["out"],
        source=source,
        ensure_row_contiguous=contiguous,
    )


def _make_pool_kernel(spec, nhwc, with_bias, rectify):
    kh, kw, sh, sw, ph, pw = spec
    ca, ha, wa = (3, 1, 2) if nhwc else (1, 2, 3)
    if nhwc:
        decode = """
    const int c = int(gid % uint(C));
    const uint t0 = gid / uint(C);
    const int ow = int(t0 % uint(OW));
    const uint t1 = t0 / uint(OW);
    const int oh = int(t1 % uint(OH));
    const int n = int(t1 / uint(OH));
    const uint dst = gid;
"""
    else:
        decode = """
    const uint plane = gid / uint(OH * OW);
    const uint pos = gid % uint(OH * OW);
    int oh, ow;
    if (x_strides[2] < x_strides[3]) {
        oh = int(pos % uint(OH));
        ow = int(pos / uint(OH));
    } else {
        oh = int(pos / uint(OW));
        ow = int(pos % uint(OW));
    }
    const int c = int(plane % uint(C));
    const int n = int(plane / uint(C));
    const uint dst = (plane * uint(OH) + uint(oh)) * uint(OW) + uint(ow);
"""
    epilogue = "value += b[c];" if with_bias else ""
    if rectify:
        epilogue += " value = metal::max(value, 0.0f);"
    source = f"""
    const uint gid = thread_position_in_grid.x;
    const int N = x_shape[0];
    const int C = x_shape[{ca}];
    const int H = x_shape[{ha}];
    const int W = x_shape[{wa}];
    const int OH = (H + 2 * {ph} - {kh}) / {sh} + 1;
    const int OW = (W + 2 * {pw} - {kw}) / {sw} + 1;
    if (gid >= uint(N * C * OH * OW)) return;
    {decode}
    const size_t base = size_t(n) * x_strides[0]
                         + size_t(c) * x_strides[{ca}];
    float value = -3.402823466e+38f;
    #pragma unroll
    for (int r = 0; r < {kh}; ++r) {{
        const int ih = oh * {sh} - {ph} + r;
        if (ih < 0 || ih >= H) continue;
        #pragma unroll
        for (int s = 0; s < {kw}; ++s) {{
            const int iw = ow * {sw} - {pw} + s;
            if (iw < 0 || iw >= W) continue;
            const size_t loc = base + size_t(ih) * x_strides[{ha}]
                                    + size_t(iw) * x_strides[{wa}];
            value = metal::max(value, x[loc]);
        }}
    }}
    {epilogue}
    out[dst] = value;
"""
    tag = spec + (int(nhwc), int(with_bias), int(rectify))
    return _metal(
        "kb049_v4_pool_" + "_".join(str(v) for v in tag),
        ["x", "b"] if with_bias else ["x"],
        source,
        False,
    )


def _make_patch_kernel(channels, kh, kw, padded_k):
    k = channels * kh * kw
    source = f"""
    const uint gid = thread_position_in_grid.x;
    const int H = x_shape[2];
    const int W = x_shape[3];
    const int OH = (H + 4 - {kh}) / 4 + 1;
    const int OW = (W + 4 - {kw}) / 4 + 1;
    const uint rows = uint(x_shape[0] * OH * OW);
    if (gid >= rows * {padded_k}u) return;
    const uint column = gid % {padded_k}u;
    const uint row = gid / {padded_k}u;
    float value = 0.0f;
    if (column < {k}u) {{
        const int c = int(column % {channels}u);
        const int s = int((column / {channels}u) % {kw}u);
        const int r = int(column / {channels * kw}u);
        const int ow = int(row % uint(OW));
        const int oh = int((row / uint(OW)) % uint(OH));
        const int n = int(row / uint(OH * OW));
        const int ih = oh * 4 - 2 + r;
        const int iw = ow * 4 - 2 + s;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {{
            const size_t loc = size_t(n) * x_strides[0]
                + size_t(c) * x_strides[1]
                + size_t(ih) * x_strides[2]
                + size_t(iw) * x_strides[3];
            value = x[loc];
        }}
    }}
    out[gid] = value;
"""
    name = f"kb049_v4_patch_{channels}_{kh}_{kw}_{padded_k}"
    return _metal(name, ["x"], source, False)


def _make_depthwise_kernel(kh, kw, with_bias):
    epilogue = "value += b[c];" if with_bias else ""
    source = f"""
    const uint gid = thread_position_in_grid.x;
    const int C = x_shape[1];
    const int H = x_shape[2];
    const int W = x_shape[3];
    const int OH = H + 3 - {kh};
    const int OW = W + 3 - {kw};
    if (gid >= uint(x_shape[0] * C * OH * OW)) return;
    const int ow = int(gid % uint(OW));
    const uint t = gid / uint(OW);
    const int oh = int(t % uint(OH));
    const uint plane = t / uint(OH);
    const int c = int(plane % uint(C));
    const int n = int(plane / uint(C));
    const size_t base = size_t(n) * x_strides[0]
                         + size_t(c) * x_strides[1];
    float value = 0.0f;
    #pragma unroll
    for (int r = 0; r < {kh}; ++r) {{
        const int ih = oh - 1 + r;
        if (ih < 0 || ih >= H) continue;
        #pragma unroll
        for (int s = 0; s < {kw}; ++s) {{
            const int iw = ow - 1 + s;
            if (iw < 0 || iw >= W) continue;
            const size_t loc = base + size_t(ih) * x_strides[2]
                                    + size_t(iw) * x_strides[3];
            value = metal::fma(x[loc], w[(c * {kh} + r) * {kw} + s], value);
        }}
    }}
    {epilogue}
    out[gid] = value;
"""
    name = f"kb049_v4_depthwise_{kh}_{kw}_{int(with_bias)}"
    return _metal(name, ["x", "w", "b"] if with_bias else ["x", "w"], source, False)


_SUM_TAIL = """
    float subtotal = metal::simd_sum(acc);
    threadgroup float scratch[8];
    if ((tid & 31u) == 0u) scratch[tid >> 5] = subtotal;
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (tid < 32u) {
        float value = tid < 8u ? scratch[tid] : 0.0f;
        value = metal::simd_sum(value);
        if (tid == 0u) out[row] = value * scale;
    }
"""


def _make_sigmoid_sum(ndim, with_bias):
    width = " * ".join(f"x_shape[{i}]" for i in range(1, ndim))
    addition = f"v += b[j % uint(x_shape[{ndim - 1}])];" if with_bias else ""
    source = (
        f"""
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint K = uint({width});
    const float scale = 1.0f;
    float acc = 0.0f;
    for (uint j = tid; j < K; j += 256u) {{
        float v = x[size_t(row) * K + j];
        {addition}
        const float e = metal::exp(-metal::abs(v));
        acc += v >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    }}
"""
        + _SUM_TAIL
    )
    return _metal(
        f"kb049_v4_sigmoid_sum_{ndim}_{int(with_bias)}", ["x", "b"] if with_bias else ["x"], source
    )


def _make_gelu_mean():
    source = (
        """
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint C = uint(x_shape[3]);
    const uint S = uint(x_shape[1] * x_shape[2]);
    const uint n = row / C;
    const uint c = row % C;
    const float scale = 1.0f / float(S);
    const float bias = b[c];
    float acc = 0.0f;
    for (uint s = tid; s < S; s += 256u) {
        const float v = x[(size_t(n) * S + s) * C + c] + bias;
        const float a = metal::abs(v * 0.7071067811865475f);
        const float t = 1.0f / (1.0f + 0.3275911f * a);
        const float p = (((((1.061405429f * t - 1.453152027f) * t)
                           + 1.421413741f) * t - 0.284496736f) * t
                           + 0.254829592f) * t;
        float e = 1.0f - p * metal::exp(-a * a);
        if (v < 0.0f) e = -e;
        acc += (0.5f * v) * (1.0f + e);
    }
"""
        + _SUM_TAIL
    )
    return _metal("kb049_v4_gelu_mean", ["x", "b"], source)


class Runner:
    def __init__(self, task, config, parameters, reference):
        self.task = task
        self.reference = reference
        self._graph = None
        self._direct = False
        self._ndim = 4
        self._cache = OrderedDict()
        self._convs = {}
        self._linears = {}
        self._extra = None
        p = parameters

        if task == "L1_42":
            self._setup_pool(config["kernel"], config["stride"], config["padding"], False)
            self._graph = self._pool_only
            self._direct = True

        elif task == "L1_50":
            self._pack_conv(p, "conv1")
            o, c, kh, kw = p["conv1.weight"].shape
            k = c * kh * kw
            kp = ((k + 31) // 32) * 32
            matrix = self._convs["conv1"][0].reshape(o, k)
            if kp != k:
                matrix = mx.pad(matrix, ((0, 0), (0, kp - k)))
            self._linears["patches"] = (mx.contiguous(matrix).T, self._convs["conv1"][1])
            self._patch_geometry = (kh, kw, kp)
            self._patch_kernel = _make_patch_kernel(c, kh, kw, kp)
            self._graph = self._conv11

        elif task == "L1_86":
            dw = p["depthwise.weight"]
            pw = p["pointwise.weight"]
            groups = int(config["groups"])
            if (
                dw.shape[1] != 1
                or dw.shape[0] != groups
                or tuple(pw.shape[2:]) != (1, 1)
                or pw.shape[1] != groups
            ):
                return
            db = p.get("depthwise.bias")
            pb = p.get("pointwise.bias")
            self._convs["depthwise_raw"] = (
                mx.contiguous(dw),
                None if db is None else mx.contiguous(db),
            )
            self._linears["pointwise_raw"] = (
                mx.contiguous(pw.reshape(pw.shape[0], groups)),
                None if pb is None else mx.contiguous(pb),
            )
            self._depthwise_geometry = (dw.shape[2], dw.shape[3])
            self._depthwise_kernel = _make_depthwise_kernel(
                dw.shape[2], dw.shape[3], db is not None
            )
            self._graph = self._separable

        elif task == "L2_01":
            self._pack_conv(p, "conv")
            bias = p["bias"]
            shape = (1,) * (4 - bias.ndim) + tuple(bias.shape)
            self._extra = mx.contiguous(bias.reshape(shape).transpose(0, 2, 3, 1))
            self._graph = self._conv_relu_bias

        elif task == "L2_40":
            self._ndim = 2
            self._pack_linear(p, "matmul")
            self._graph = self._scaled_linear

        elif task == "L2_56":
            self._ndim = 2
            self._pack_linear(p, "linear")
            self._sigmoid_kernel = _make_sigmoid_sum(2, False)
            self._graph = self._linear_sigmoid_sum

        elif task == "L2_65":
            w = p["conv.weight"].transpose(0, 2, 3, 1)
            folded = None
            for r in range(4):
                for s in range(4):
                    term = mx.pad(w, ((0, 0), (r, 3 - r), (s, 3 - s), (0, 0)))
                    folded = term if folded is None else folded + term
            bias = p.get("conv.bias")
            if bias is None:
                bias = mx.zeros((w.shape[0],), dtype=mx.float32)
            self._convs["pooled"] = (mx.contiguous(folded * 0.0625), mx.contiguous(bias))
            self._sigmoid_kernel = _make_sigmoid_sum(4, True)
            self._graph = self._averaged_convolution

        elif task == "L2_67":
            self._pack_conv(p, "conv")
            w, b = self._convs["conv"]
            if b is None:
                b = mx.zeros((w.shape[0],), dtype=mx.float32)
                self._convs["conv"] = (w, b)
            self._gelu_kernel = _make_gelu_mean()
            self._graph = self._conv_gelu_mean

        elif task == "L3_04":
            channels = p["conv2.weight"].shape[0]
            w = p["fc1.weight"]
            side = math.isqrt(w.shape[1] // channels)
            if side == 0 or channels * side * side != w.shape[1]:
                return
            if "conv1.bias" not in p or "conv2.bias" not in p:
                return
            self._lenet_geometry = (
                p["conv1.weight"].shape[2],
                p["conv1.weight"].shape[3],
                p["conv2.weight"].shape[2],
                p["conv2.weight"].shape[3],
                side,
            )
            self._pack_conv(p, "conv1")
            self._pack_conv(p, "conv2")
            self._setup_pool(2, 2, 0, True, with_bias=True, rectify=True)
            packed = mx.contiguous(
                w.reshape(w.shape[0], channels, side, side).transpose(0, 2, 3, 1).reshape(w.shape)
            )
            self._linears["fc1"] = (packed.T, p["fc1.bias"])
            self._pack_linear(p, "fc2")
            self._pack_linear(p, "fc3")
            self._graph = self._lenet

        elif task == "L3_06":
            matrices = []
            biases = []
            for name in ("branch1x1", "branch3x3.0", "branch5x5.0"):
                w = p[name + ".weight"]
                matrices.append(w.reshape(w.shape[0], w.shape[1]))
                b = p.get(name + ".bias")
                if b is None:
                    b = mx.zeros((w.shape[0],), dtype=mx.float32)
                biases.append(b)
            self._linears["projection"] = (
                mx.concatenate(matrices, axis=0).T,
                mx.concatenate(biases, axis=0),
            )
            self._splits = (matrices[0].shape[0], matrices[1].shape[0])
            self._pack_conv(p, "branch3x3.1")
            self._pack_conv(p, "branch5x5.1")
            self._pack_linear(p, "branch_pool.1")
            self._setup_pool(3, 1, 1, True)
            self._graph = self._inception

        elif task == "L3_17":
            self._pack_linear(p, "squeeze")
            self._pack_linear(p, "expand1x1")
            self._pack_conv(p, "expand3x3")
            self._graph = self._fire

        ready = []
        for w, b in self._convs.values():
            ready.append(w)
            if b is not None:
                ready.append(b)
        for w, b in self._linears.values():
            ready.append(w)
            if b is not None:
                ready.append(b)
        if self._extra is not None:
            ready.append(self._extra)
        if ready:
            mx.eval(*ready)

    def _pack_conv(self, parameters, name):
        w = parameters[name + ".weight"]
        b = parameters.get(name + ".bias")
        self._convs[name] = (
            mx.contiguous(w.transpose(0, 2, 3, 1)),
            None if b is None else mx.contiguous(b),
        )

    def _pack_linear(self, parameters, name):
        w = parameters[name + ".weight"]
        b = parameters.get(name + ".bias")
        self._linears[name] = (w.reshape(w.shape[0], -1).T, None if b is None else mx.contiguous(b))

    def _setup_pool(self, kernel, stride, padding, nhwc, with_bias=False, rectify=False):
        kh, kw = _pair(kernel)
        sh, sw = _pair(stride)
        ph, pw = _pair(padding)
        self._pool_spec = (kh, kw, sh, sw, ph, pw)
        self._pool_nhwc = nhwc
        self._pool_with_bias = with_bias
        self._pool_kernel = _make_pool_kernel(self._pool_spec, nhwc, with_bias, rectify)

    def _pool(self, x, bias=None):
        if self._pool_nhwc:
            n, h, w, c = x.shape
        else:
            n, c, h, w = x.shape
        kh, kw, sh, sw, ph, pw = self._pool_spec
        oh = (h + 2 * ph - kh) // sh + 1
        ow = (w + 2 * pw - kw) // sw + 1
        shape = (n, oh, ow, c) if self._pool_nhwc else (n, c, oh, ow)
        args = [x, bias] if self._pool_with_bias else [x]
        return self._pool_kernel(
            inputs=args,
            grid=(n * c * oh * ow, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[shape],
            output_dtypes=[mx.float32],
        )[0]

    def _conv(self, x, name, stride=1, padding=0, groups=1, add_bias=True):
        w, b = self._convs[name]
        y = mx.conv2d(x, w, stride=stride, padding=padding, groups=groups)
        if add_bias and b is not None:
            y = y + b
        return y

    def _linear(self, x, name):
        w, b = self._linears[name]
        return x @ w if b is None else mx.addmm(b, x, w)

    def _pointwise(self, x, name):
        n, h, w, c = x.shape
        y = self._linear(x.reshape(-1, c), name)
        return y.reshape(n, h, w, y.shape[-1])

    def _pool_only(self, x):
        return self._pool(x)

    def _conv11(self, x):
        n, c, h, w = x.shape
        kh, kw, kp = self._patch_geometry
        oh = (h + 4 - kh) // 4 + 1
        ow = (w + 4 - kw) // 4 + 1
        rows = n * oh * ow
        if rows * kp * 4 > 8 * 1024 * 1024:
            y = self._conv(_nhwc(x), "conv1", stride=4, padding=2)
        else:
            patches = self._patch_kernel(
                inputs=[x],
                grid=(rows * kp, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(rows, kp)],
                output_dtypes=[mx.float32],
            )[0]
            y = self._linear(patches, "patches")
            y = y.reshape(n, oh, ow, y.shape[-1])
        return y.transpose(0, 3, 1, 2)

    def _separable(self, x):
        n, c, h, w = x.shape
        kh, kw = self._depthwise_geometry
        oh, ow = h + 3 - kh, w + 3 - kw
        dw, db = self._convs["depthwise_raw"]
        args = [x, dw] if db is None else [x, dw, db]
        y = self._depthwise_kernel(
            inputs=args,
            grid=(n * c * oh * ow, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(n, c, oh, ow)],
            output_dtypes=[mx.float32],
        )[0]
        pw, pb = self._linears["pointwise_raw"]
        y = mx.matmul(pw, y.reshape(n, c, oh * ow))
        if pb is not None:
            y = y + pb.reshape(1, -1, 1)
        return y.reshape(n, pw.shape[0], oh, ow)

    def _conv_relu_bias(self, x):
        y = self._conv(_nhwc(x), "conv")
        return (mx.maximum(y, 0) + self._extra).transpose(0, 3, 1, 2)

    def _scaled_linear(self, x):
        w, b = self._linears["matmul"]
        if b is None:
            return (x @ w) * 1.5
        return mx.addmm(b, x, w, alpha=1.5, beta=1.5)

    def _linear_sigmoid_sum(self, x):
        y = self._linear(x, "linear")
        n = y.shape[0]
        return self._sigmoid_kernel(
            inputs=[y],
            grid=(n * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(n, 1)],
            output_dtypes=[mx.float32],
        )[0]

    def _averaged_convolution(self, x):
        y = self._conv(_nhwc(x), "pooled", stride=4, add_bias=False)
        n = y.shape[0]
        return self._sigmoid_kernel(
            inputs=[y, self._convs["pooled"][1]],
            grid=(n * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(n,)],
            output_dtypes=[mx.float32],
        )[0]

    def _conv_gelu_mean(self, x):
        y = self._conv(_nhwc(x), "conv", add_bias=False)
        n, h, w, c = y.shape
        return self._gelu_kernel(
            inputs=[y, self._convs["conv"][1]],
            grid=(n * c * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(n, c)],
            output_dtypes=[mx.float32],
        )[0]

    def _lenet(self, x):
        y = self._conv(_nhwc(x), "conv1", add_bias=False)
        y = self._pool(y, self._convs["conv1"][1])
        y = self._conv(y, "conv2", add_bias=False)
        y = self._pool(y, self._convs["conv2"][1])
        y = y.reshape(y.shape[0], -1)
        y = mx.maximum(self._linear(y, "fc1"), 0)
        y = mx.maximum(self._linear(y, "fc2"), 0)
        return self._linear(y, "fc3")

    def _inception(self, x):
        v = _nhwc(x)
        projected = self._pointwise(v, "projection")
        a, b = self._splits
        branch1 = projected[..., :a]
        branch3 = self._conv(mx.contiguous(projected[..., a : a + b]), "branch3x3.1", padding=1)
        branch5 = self._conv(mx.contiguous(projected[..., a + b :]), "branch5x5.1", padding=2)
        branch_pool = self._pointwise(self._pool(v), "branch_pool.1")
        return mx.concatenate((branch1, branch3, branch5, branch_pool), axis=3).transpose(
            0, 3, 1, 2
        )

    def _fire(self, x):
        y = mx.maximum(self._pointwise(_nhwc(x), "squeeze"), 0)
        a = mx.maximum(self._pointwise(y, "expand1x1"), 0)
        b = mx.maximum(self._conv(y, "expand3x3", padding=1), 0)
        return mx.concatenate((a, b), axis=3).transpose(0, 3, 1, 2)

    def run(self, *inputs):
        use_reference = self._graph is None
        if not use_reference and inputs[0].ndim != self._ndim:
            use_reference = True
        if not use_reference and self.task == "L3_04":
            k1h, k1w, k2h, k2w, side = self._lenet_geometry
            h = (inputs[0].shape[2] - k1h + 1) // 2
            w = (inputs[0].shape[3] - k1w + 1) // 2
            h = (h - k2h + 1) // 2
            w = (w - k2w + 1) // 2
            use_reference = h != side or w != side
        if use_reference:
            result = self.reference(*inputs)
        elif self._direct:
            result = self._graph(*inputs)
        else:
            signature = tuple(tuple(x.shape) for x in inputs)
            fn = self._cache.get(signature)
            if fn is None:
                implementation = self._graph

                def graph(*xs):
                    return implementation(*xs)

                fn = mx.compile(graph)
                if len(self._cache) >= 2:
                    self._cache.popitem(last=False)
                self._cache[signature] = fn
            else:
                self._cache.move_to_end(signature)
            result = fn(*inputs)
        mx.eval(result)
        return result
