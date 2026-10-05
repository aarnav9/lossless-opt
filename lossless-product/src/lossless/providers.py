"""Bounded user-owned LLM transports. Evaluators never delegate acceptance."""

import contextlib
import importlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from .jobs import DEFAULT_LLM_TIMEOUT_SECONDS


def bounded_timeout(connection, remaining_seconds):
    """Same configured timeout and remaining-budget rule for every adapter."""
    return min(
        (connection or {}).get("timeout_seconds", DEFAULT_LLM_TIMEOUT_SECONDS),
        max(0, remaining_seconds),
    )


def _callback_child(connection, callback, prompt, base):
    os.setsid()
    try:
        os.chdir(base)
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            if isinstance(callback, str):
                module, symbol = callback.rsplit(":", 1)
                if module.endswith(".py"):
                    path = (Path(base) / module).resolve()
                    spec = importlib.util.spec_from_file_location("lossless_user_callback", path)
                    loaded = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(loaded)
                else:
                    import sys

                    sys.path.insert(0, str(base))
                    loaded = importlib.import_module(module)
                callback = getattr(loaded, symbol)
            response = callback(prompt)
            text = response if isinstance(response, str) else json.dumps(response, allow_nan=False)
            if len(text.encode()) > 4_000_000:
                raise ValueError("response too large")
            connection.send((True, text))
    except BaseException as error:
        connection.send((False, type(error).__name__))
    finally:
        connection.close()


def _kill(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        if hasattr(process, "terminate"):
            process.terminate()


def call(
    context, connection, *, callback=None, base=Path("."), timeout=DEFAULT_LLM_TIMEOUT_SECONDS
):
    """Return a parsed proposal batch. Only explicit transports may use credentials.

    Callback workers use POSIX fork to support ordinary Python callables without
    serializing executable Python objects. This alpha supports macOS and Linux.
    This contains crashes/timeouts; it is not a security sandbox for supplied code.
    """
    if timeout <= 0:
        raise TimeoutError("LLM budget exhausted")
    connection = connection or {}
    prompt = json.dumps(context, allow_nan=False)
    started = time.monotonic()
    if callback is not None or "callback" in connection:
        selected = callback if callback is not None else connection["callback"]
        # Named callbacks start a fresh interpreter, including when the parent
        # has initialized Metal. Only in-memory CPU callbacks require fork.
        ctx = multiprocessing.get_context("spawn" if isinstance(selected, str) else "fork")
        parent, child = ctx.Pipe(duplex=False)
        process = ctx.Process(target=_callback_child, args=(child, selected, prompt, str(base)))
        process.start()
        child.close()
        try:
            if not parent.poll(timeout):
                raise TimeoutError("LLM callback timed out")
            ok, text = parent.recv()
            if not ok:
                raise ValueError(
                    f"LLM callback failed ({text}); credentials and exception details were not recorded"
                )
        finally:
            parent.close()
            process.join(timeout=0.2)
            if process.is_alive():
                _kill(process)
                process.join(timeout=2)
    else:
        command = connection.get("command")
        if not command:
            raise ValueError("LLM callback or command required")
        env = {
            k: v
            for k, v in os.environ.items()
            if k in {"PATH", "HOME", "TMPDIR", "LANG", *connection.get("credential_env", [])}
        }
        env["PYTHONUNBUFFERED"] = "1"
        # Let command transports honor the resolved allowance, including the
        # remaining search budget, without requiring a second timeout setting.
        env["LOSSLESS_PROVIDER_TIMEOUT_SECONDS"] = str(timeout)
        with (
            tempfile.TemporaryFile() as source,
            tempfile.TemporaryFile() as output,
            open(os.devnull, "wb") as errors,
        ):
            source.write(prompt.encode())
            source.seek(0)
            process = subprocess.Popen(
                command,
                cwd=base,
                stdin=source,
                stdout=output,
                stderr=errors,
                env=env,
                start_new_session=True,
            )
            try:
                while process.poll() is None:
                    if time.monotonic() - started >= timeout:
                        raise TimeoutError("LLM command timed out")
                    if os.fstat(output.fileno()).st_size > 4_000_000:
                        raise ValueError("LLM response exceeds 4 MB")
                    time.sleep(0.05)
                if process.returncode:
                    raise ValueError("LLM command failed; provider stderr is not retained")
                output.seek(0)
                text = output.read(4_000_001).decode()
                if len(text.encode()) > 4_000_000:
                    raise ValueError("LLM response exceeds 4 MB")
            finally:
                if process.poll() is None:
                    _kill(process)
                process.wait()
    # Known configured secrets must never enter proposal files, logs, or prompts.
    for key in connection.get("credential_env", []):
        secret = os.environ.get(key)
        if secret and len(secret) >= 8 and secret in text:
            raise ValueError("provider response contains a configured credential")
    from .jobs import object_pairs

    return json.loads(
        text,
        object_pairs_hook=object_pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite provider response")),
    ), time.monotonic() - started
