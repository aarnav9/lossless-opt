"""Sleep-inclusive elapsed clock for deadlines; fail closed on unsupported hosts."""

import ctypes
import sys
import time

if sys.platform == "darwin":
    _lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")

    class _Timebase(ctypes.Structure):
        _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]

    _base = _Timebase()
    _lib.mach_timebase_info(ctypes.byref(_base))
    _lib.mach_continuous_time.restype = ctypes.c_uint64

    def elapsed():
        return _lib.mach_continuous_time() * _base.numer / _base.denom / 1e9

elif hasattr(time, "CLOCK_BOOTTIME"):

    def elapsed():
        return time.clock_gettime(time.CLOCK_BOOTTIME)
else:
    raise RuntimeError("A sleep-inclusive monotonic clock is required for this study")
