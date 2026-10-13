"""Explicit reusable CPU request process, with bounded transport and no retries."""

import json
import math
from multiprocessing.connection import Connection
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

from ..jobs import read_json
from .wire import MAX_MESSAGE, unpack


class PreparedApplication:
    """Own a single worker. Requests use the same JSON/$npy schema as run-project.

    Outputs are copied values. Factories reset per request; imports remain loaded.
    A context manager is required to release the worker and its private snapshot.
    """

    def __init__(self, bundle, *, python=None, variant="auto", timeout=30, _qualifying=False):
        if (
            sys.platform not in {"darwin", "linux"}
            or variant not in {"auto", "reference"}
            or timeout <= 0
            or not math.isfinite(timeout)
        ):
            raise ValueError(
                "prepared requests require macOS/Linux, auto/reference, and positive timeout"
            )
        tick = time.perf_counter()
        self.bundle, self.timeout = Path(bundle).resolve(), timeout
        manifest = read_json(self.bundle / "manifest.json")
        self._temporary = tempfile.TemporaryDirectory(prefix="lossless-prepared-")
        folder = Path(self._temporary.name)
        (folder / "home").mkdir()
        self._log_path = folder / "worker.log"
        self._log = self._log_path.open("wb")
        self._limit = manifest["max_output_bytes"]
        self._lock = threading.Lock()
        self._closed = False
        self.memory_available = True
        self._peaks = {}
        parent, child = socket.socketpair()
        self._connection = Connection(parent.detach())
        env = {
            k: v
            for k, v in os.environ.items()
            if k in {"PATH", "LANG", "SYSTEMROOT", *manifest["inherit_env"]}
        }
        env.update(
            HOME=str(folder / "home"),
            TMPDIR=str(folder) + "/",
            PYTHONHASHSEED=str(manifest["seed"]),
            PYTHONUNBUFFERED="1",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(Path(__file__).resolve().parents[2]),
        )
        try:
            self._process = subprocess.Popen(
                [
                    str(python or sys.executable),
                    "-s",
                    "-m",
                    "lossless.applications.prepared_worker",
                    str(child.fileno()),
                    str(self.bundle),
                    variant,
                    "qualify" if _qualifying else "run",
                ],
                pass_fds=(child.fileno(),),
                env=env,
                cwd=folder,
                stdout=self._log,
                stderr=self._log,
                start_new_session=True,
            )
            child.close()
            self.info = self._exchange(None)
            if self.info.get("status") != "ready":
                raise ValueError("prepared worker did not initialize")
            self.startup_seconds = time.perf_counter() - tick
        except BaseException:
            child.close()
            self.close()
            raise

    def _exchange(self, command):
        finished, result = threading.Event(), {}

        def exchange():
            try:
                if command is not None:
                    data = json.dumps(command, allow_nan=False).encode()
                    if len(data) > 2**20:
                        raise ValueError("request JSON exceeds 1 MiB; use $npy fixtures")
                    self._connection.send_bytes(data)
                result["value"] = unpack(self._connection.recv_bytes(MAX_MESSAGE))
            except BaseException as error:
                result["error"] = error
            finally:
                finished.set()

        thread = threading.Thread(target=exchange, daemon=True)
        thread.start()
        deadline = time.monotonic() + self.timeout
        self._sample_memory()
        while not finished.wait(min(0.02, max(0, deadline - time.monotonic()))):
            self._sample_memory()
            if time.monotonic() >= deadline or self._log_path.stat().st_size > self._limit:
                self.close()
                thread.join(timeout=1)
                raise TimeoutError(
                    "prepared request exceeded time/output limit; worker stopped, no retry"
                )
        if "error" in result:
            log = self._log_path.read_text(errors="replace")[-2000:]
            self.close()
            raise ValueError("prepared worker communication failed: " + log) from result["error"]
        if self._log_path.stat().st_size > self._limit:
            self.close()
            raise ValueError("prepared worker exceeded cumulative output limit")
        return result["value"]

    def _sample_memory(self):
        if not self.memory_available:
            return
        try:
            import psutil

            parent = psutil.Process(self._process.pid)
            for process in [parent, *parent.children(recursive=True)]:
                try:
                    key = process.pid, process.create_time()
                    self._peaks[key] = max(self._peaks.get(key, 0), process.memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
        except ImportError:
            self.memory_available = False
        except Exception:
            self.memory_available = False

    @property
    def peak_tree_rss_bytes(self):
        return sum(self._peaks.values())

    def _run(self, command):
        if self._closed:
            raise ValueError("prepared application is closed")
        if not self._lock.acquire(blocking=False):
            raise ValueError("one request at a time per prepared application")
        tick = time.perf_counter()
        try:
            result = self._exchange(command)
            result["request_seconds"] = time.perf_counter() - tick
            if result["status"] == "outside_scope":
                raise ValueError(result["reason"])
            if result["status"] != "completed":
                self.close()
                raise ValueError(
                    f"prepared application failed without retry: {result.get('error')}: {result.get('reason')}"
                )
            return result
        finally:
            self._lock.release()

    def run_case(self, case):
        return self._run(dict(action="run", case=case))

    def run(self, request, *, input_root):
        return self._run(
            dict(action="run", request=request, input_root=str(Path(input_root).resolve()))
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        process = getattr(self, "_process", None)
        if process is not None:
            # Terminate the owned process group, including application children.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                process.wait(timeout=1)
            process.wait(timeout=2)
        self._connection.close()
        self._log.close()
        self._temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
