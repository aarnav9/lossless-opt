"""Hardware-guided launch/address experiments; no floating-point arithmetic."""

from functools import lru_cache
import mlx.core as mx
from lossless.adapters.mlx.indexing import source, VARIANTS


@lru_cache(maxsize=1)
def max_buffer_bytes():
    return mx.device_info()["max_buffer_length"]


@lru_cache(maxsize=5)
def kernel(variant):
    return mx.fast.metal_kernel(
        name="kernel_lab_append_022_" + variant,
        input_names=["oldk", "oldv", "newk", "newv", "meta"],
        output_names=["outk", "outv"],
        source=source(variant),
        compile_options={"math_mode": "safe"},
    )


def append(oldk, oldv, newk, newv, previous, group=256, vector_width=1, variant="flat"):
    if vector_width != 1 or variant not in VARIANTS:
        raise ValueError("unsupported hardware-guided copy configuration")
    if type(group) is not int or not 1 <= group <= 1024:
        raise ValueError("invalid threadgroup")
    B, H, added, D = newk.shape
    if (
        min(B, H, added, D) < 1
        or newk.shape != newv.shape
        or newk.dtype != newv.dtype
        or newk.dtype not in (mx.float32, mx.float16)
    ):
        raise ValueError("matching FP32/FP16 K/V shapes required")
    if oldk is None:
        oldk = mx.zeros((B, H, 0, D), dtype=newk.dtype)
        oldv = mx.zeros((B, H, 0, D), dtype=newv.dtype)
    if (
        oldk.dtype != newk.dtype
        or oldv.dtype != newv.dtype
        or oldk.shape != oldv.shape
        or oldk.shape[:2] != (B, H)
        or oldk.shape[3] != D
        or not 0 <= previous <= oldk.shape[2]
    ):
        raise ValueError("invalid old cache layout")
    capacity, total = oldk.shape[2], previous + added
    size = B * H * total * D
    if max(B * H * capacity * D, size) >= 2**32:
        raise ValueError("outside uint32 address proof bound")
    if size * newk.itemsize > max_buffer_bytes():
        raise ValueError("output exceeds reported Metal buffer limit")
    meta = mx.array([capacity, previous, added, D, size], dtype=mx.uint32)
    grid = (size, 1, 1)
    if variant.startswith("rows"):
        chunk = 2 if variant == "rows_two" else 1
        grid = ((total * D + chunk - 1) // chunk, B * H, 1)
    template = [("T", newk.dtype)]
    if variant in {"fixed_width", "rows_fixed_width"}:
        template.append(("W", D))
    return kernel(variant)(
        inputs=[oldk, oldv, newk, newv, meta],
        template=template,
        grid=grid,
        threadgroup=(group, 1, 1),
        output_shapes=[(B, H, total, D)] * 2,
        output_dtypes=[newk.dtype] * 2,
    )
