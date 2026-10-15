"""Bounded raw array frames. Decoded arrays own their storage; never pickle."""

import io
import json
import math
import struct

from .bundle import restore_value
from .wire import MAX_MESSAGE


def pack(value, staging=None):
    import numpy as np

    arrays, segments = [], []
    total = 0

    def encode(v, depth=0):
        nonlocal total
        if depth > 40:
            raise ValueError("output tree too deep")
        if isinstance(v, (np.ndarray, np.generic)):
            a = np.asarray(v)
            if a.dtype.hasobject:
                raise ValueError("object arrays cannot be transported")
            if total + a.nbytes > MAX_MESSAGE:
                raise ValueError("prepared response exceeds 64 MiB transport limit")
            name = str(len(arrays))
            if a.dtype.fields is not None or a.dtype.subdtype is not None or a.dtype.itemsize == 0:
                stream = io.BytesIO()
                np.save(stream, a, allow_pickle=False)
                segment = stream.getvalue()
                descriptor = dict(name=name, encoding="npy", length=len(segment))
            else:
                order = "F" if a.flags.f_contiguous and not a.flags.c_contiguous else "C"
                # ravel is a view for C/F contiguous arrays, a copy for other layouts.
                segment = memoryview(a.ravel(order=order)).cast("B")
                descriptor = dict(
                    name=name,
                    encoding="raw",
                    length=segment.nbytes,
                    dtype=a.dtype.str,
                    shape=list(a.shape),
                    order=order,
                )
            total += len(segment)
            if total > MAX_MESSAGE:
                raise ValueError("prepared response exceeds 64 MiB transport limit")
            arrays.append(descriptor)
            segments.append(segment)
            return dict(type="numpy_scalar" if isinstance(v, np.generic) else "ndarray", file=name)
        if v is None or type(v) in {bool, int, str}:
            return dict(type=type(v).__name__, value=v)
        if type(v) is float:
            return dict(type="float", bits=struct.pack(">d", v).hex())
        if isinstance(v, bytes):
            return dict(type="bytes", hex=v.hex())
        if isinstance(v, (list, tuple)):
            return dict(type=type(v).__name__, items=[encode(x, depth + 1) for x in v])
        if isinstance(v, dict) and all(isinstance(k, str) for k in v):
            return dict(type="dict", items=[[k, encode(x, depth + 1)] for k, x in v.items()])
        raise ValueError("unsupported transported value")

    row = encode(value)
    header = json.dumps(dict(value=row, arrays=arrays)).encode()
    size = 4 + len(header) + total
    if size > MAX_MESSAGE:
        raise ValueError("prepared response exceeds 64 MiB transport limit")
    parts = [struct.pack(">I", len(header)), header, *segments]
    if staging is None:
        return b"".join(parts)
    if len(staging) < size:
        staging.extend(b"\0" * (size - len(staging)))
    target = memoryview(staging)[:size]
    cursor = 0
    for part in parts:
        target[cursor : cursor + len(part)] = part
        cursor += len(part)
    return target


def unpack(data):
    import numpy as np

    if not 4 <= len(data) <= MAX_MESSAGE:
        raise ValueError("invalid prepared message size")
    size = struct.unpack_from(">I", data)[0]
    if size > len(data) - 4:
        raise ValueError("truncated prepared header")
    header = json.loads(bytes(memoryview(data)[4 : 4 + size]))
    cursor, arrays = 4 + size, {}
    for descriptor in header["arrays"]:
        name, length = descriptor["name"], descriptor["length"]
        if (
            type(name) is not str
            or name in arrays
            or type(length) is not int
            or length < 0
            or cursor + length > len(data)
        ):
            raise ValueError("invalid prepared array frame")
        segment = memoryview(data)[cursor : cursor + length]
        cursor += length
        if descriptor["encoding"] == "raw":
            dtype = np.dtype(descriptor["dtype"])
            shape, order = descriptor["shape"], descriptor["order"]
            if (
                dtype.hasobject
                or dtype.fields is not None
                or dtype.subdtype is not None
                or dtype.itemsize == 0
                or type(shape) is not list
                or len(shape) > 32
                or any(type(x) is not int or x < 0 for x in shape)
                or order not in {"C", "F"}
                or math.prod(shape) * dtype.itemsize != length
            ):
                raise ValueError("invalid raw array metadata")
            # Always owned, even with reusable sender/receiver buffers.
            arrays[name] = (
                np.frombuffer(segment, dtype=dtype).reshape(shape, order=order).copy(order=order)
            )
        elif descriptor["encoding"] == "npy":
            arrays[name] = np.load(io.BytesIO(segment), allow_pickle=False)
        else:
            raise ValueError("unknown array frame encoding")
    if cursor != len(data):
        raise ValueError("unexpected prepared message trailing bytes")
    return restore_value(header["value"], lambda name: arrays[name])
