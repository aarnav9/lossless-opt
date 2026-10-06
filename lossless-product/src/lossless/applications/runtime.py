"""Bounded process lifetime and sleep-inclusive replay deadlines."""

import ctypes
import os
import signal
import subprocess
import time


def clock():
    if hasattr(time, "CLOCK_BOOTTIME"):
        return time.clock_gettime(time.CLOCK_BOOTTIME)
    if os.uname().sysname == "Darwin":
        return _continuous()
    return time.monotonic()


def _darwin_clock():
    class Timebase(ctypes.Structure):
        _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]

    library = ctypes.CDLL(None)
    info = Timebase()
    library.mach_timebase_info(ctypes.byref(info))
    function = library.mach_continuous_time
    function.restype = ctypes.c_uint64
    return lambda: function() * info.numer / info.denom / 1e9


_continuous = (
    _darwin_clock() if os.name == "posix" and os.uname().sysname == "Darwin" else time.monotonic
)


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def execute(argv, *, cwd, env, stdin, folder, timeout, max_output_bytes, progress):
    started = clock()
    status = "completed"
    with (
        (folder / "stdin.txt").open("wb+") as source,
        (folder / "stdout.log").open("wb") as out,
        (folder / "stderr.log").open("wb") as err,
    ):
        source.write(stdin.encode())
        source.seek(0)
        process = subprocess.Popen(
            argv, cwd=cwd, env=env, stdin=source, stdout=out, stderr=err, start_new_session=True
        )
        heartbeat = started
        try:
            while process.poll() is None:
                elapsed = clock() - started
                if elapsed >= timeout:
                    status = "timeout"
                    break
                if (
                    os.fstat(out.fileno()).st_size + os.fstat(err.fileno()).st_size
                    > max_output_bytes
                ):
                    status = "output_limit"
                    break
                if clock() - heartbeat >= 30:
                    progress(f"REPLAY running elapsed={elapsed:.0f}s allowance={timeout:.0f}s")
                    heartbeat = clock()
                time.sleep(min(0.05, max(0, timeout - elapsed)))
        finally:
            # Kill descendants even when their parent returned; no background
            # process should carry state into the next repeat.
            stop(process)
        if status == "completed" and clock() - started > timeout:
            status = "timeout"
        if os.fstat(out.fileno()).st_size + os.fstat(err.fileno()).st_size > max_output_bytes:
            status = "output_limit"
        if status == "completed" and process.returncode != 0:
            status = "failed"
    return {
        "status": status,
        "returncode": process.returncode,
        "elapsed_seconds": clock() - started,
    }
