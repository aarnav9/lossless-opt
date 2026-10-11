"""Original CPU reference for a reversible BF16 bitmap tile format.

Inspired by the exponent coding idea in ZipServ (arXiv:2603.17435), not its
CUDA layout or implementation. Handles raw uint16 representations, including
NaNs and signed zeros. This is not quantization and is not an inference kernel.
"""

import struct


HEADER = struct.Struct("<BBQQQ")  # exponent-window start, count, three bitplanes


def encode(words, start):
    words = list(words)
    if type(start) is not int or not 0 <= start <= 249:
        raise ValueError("seven-exponent window must fit in [0,255]")
    if not 1 <= len(words) <= 64:
        raise ValueError("a tile contains 1 through 64 values")
    if any(type(w) is not int or not 0 <= w <= 65535 for w in words):
        raise ValueError("raw unsigned 16-bit words required")
    planes = [0, 0, 0]
    frequent, outliers = bytearray(), bytearray()
    for index, word in enumerate(words):
        exponent = (word >> 7) & 255
        code = exponent - start + 1 if start <= exponent < start + 7 else 0
        for bit in range(3):
            planes[bit] |= ((code >> bit) & 1) << index
        if code:
            frequent.append(((word >> 8) & 128) | (word & 127))
        else:
            outliers.extend(struct.pack("<H", word))
    return HEADER.pack(start, len(words), *planes) + frequent + outliers


def decode(data):
    if len(data) < HEADER.size:
        raise ValueError("truncated tile header")
    start, count, *planes = HEADER.unpack_from(data)
    if start > 249 or not 1 <= count <= 64 or any(p >> count for p in planes):
        raise ValueError("invalid tile metadata")
    frequent_count = (planes[0] | planes[1] | planes[2]).bit_count()
    expected_bytes = HEADER.size + frequent_count + 2 * (count - frequent_count)
    if len(data) != expected_bytes:
        raise ValueError("incorrect payload size")
    frequent_index = HEADER.size
    outlier_index = HEADER.size + frequent_count
    words = []
    for index in range(count):
        code = sum(((p >> index) & 1) << bit for bit, p in enumerate(planes))
        if code:
            payload = data[frequent_index]
            frequent_index += 1
            words.append(((payload & 128) << 8) | ((start + code - 1) << 7) | (payload & 127))
        else:
            words.append(struct.unpack_from("<H", data, outlier_index)[0])
            outlier_index += 2
    return words
