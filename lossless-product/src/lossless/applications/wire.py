"""Bounded value transport: JSON descriptors plus NumPy arrays, never pickle."""

import io
import json
import struct

from .bundle import restore_value, save_value

MAX_MESSAGE = 64 * 2**20


class Buffers:
    def __init__(self):
        self.files = {}

    def __truediv__(self, name):
        self.files[name] = io.BytesIO()
        return self.files[name]


def pack(value):
    buffers = Buffers()
    row = save_value(value, buffers, [0])
    arrays = [(name, stream.getvalue()) for name, stream in buffers.files.items()]
    header = json.dumps(
        dict(value=row, arrays=[(name, len(data)) for name, data in arrays])
    ).encode()
    size = 4 + len(header) + sum(len(data) for _, data in arrays)
    if size > MAX_MESSAGE:
        raise ValueError("prepared response exceeds 64 MiB transport limit")
    return struct.pack(">I", len(header)) + header + b"".join(data for _, data in arrays)


def unpack(data):
    import numpy as np

    if not 4 <= len(data) <= MAX_MESSAGE:
        raise ValueError("invalid prepared message size")
    size = struct.unpack(">I", data[:4])[0]
    if size > len(data) - 4:
        raise ValueError("truncated prepared header")
    header = json.loads(data[4 : 4 + size])
    cursor, arrays = 4 + size, {}
    for name, length in header["arrays"]:
        if type(length) is not int or length < 0 or cursor + length > len(data) or name in arrays:
            raise ValueError("invalid prepared array frame")
        arrays[name] = data[cursor : cursor + length]
        cursor += length
    if cursor != len(data):
        raise ValueError("unexpected prepared message trailing bytes")
    return restore_value(
        header["value"], lambda name: np.load(io.BytesIO(arrays[name]), allow_pickle=False)
    )
