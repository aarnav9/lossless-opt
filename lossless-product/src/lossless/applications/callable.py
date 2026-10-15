"""Ordinary callable facade over the explicitly copied-value CPU boundary."""

from pathlib import Path
import time
import threading

from ..jobs import identity, read_json
from .prepared import PreparedApplication


class CallableApplication:
    """Use as ``with CallableApplication(bundle) as run: result = run(x)``.

    Inputs/results are copied values: caller mutations and aliases do not cross
    the boundary. Imports persist only in prepared mode; factories reset for
    each call. Use run_calls for an explicit sequence sharing one factory.
    Unsupported optimized metadata raises before application execution. There
    are no retries; open variant='reference' explicitly when needed.
    """

    def __init__(
        self, bundle, *, python=None, variant="auto", timeout=30, mode=None, _qualifying=False
    ):
        if variant not in {"auto", "reference"} or mode not in {None, "oneshot", "prepared"}:
            raise ValueError("callable requires auto/reference and oneshot/prepared mode")
        self.bundle = Path(bundle).resolve()
        self.python, self.timeout = python, timeout
        manifest = read_json(self.bundle / "manifest.json")
        path = self.bundle / "callable.json"
        policy = read_json(path) if path.exists() else {}
        valid = (
            policy.get("status") in {"accepted", "reference_retained"}
            and policy.get("manifest_identity") == identity(manifest)
            and policy.get("lifecycle_exact") is True
        )
        self.mode = mode or (policy.get("mode", "oneshot") if valid else "oneshot")
        if self.mode not in {"oneshot", "prepared"}:
            raise ValueError("invalid callable certificate mode")
        self.variant = variant
        self._qualifying = _qualifying
        reason = None
        if not _qualifying and not (
            valid and policy.get("mode") == self.mode and policy.get("status") == "accepted"
        ):
            self.variant = "reference"
            reason = "callable deployment has not qualified; re-run run-project with --deployment"
        self.info = dict(variant=self.variant, mode=self.mode, fallback_reason=reason)
        self._session = None
        self._closed = False
        self._lock = threading.Lock()
        self.last_receipt = None
        self.startup_seconds = 0.0
        if self.mode == "prepared":
            tick = time.perf_counter()
            self._session = self._open()
            self.startup_seconds = time.perf_counter() - tick
            self.info.update(self._session.info)
            if reason and not self.info.get("fallback_reason"):
                self.info["fallback_reason"] = reason

    def _open(self):
        return PreparedApplication(
            self.bundle,
            python=self.python,
            variant=self.variant,
            timeout=self.timeout,
            input_mode="values",
            _qualifying=self._qualifying,
            _lifecycle="callable-" + self.mode,
        )

    def _request(self, calls):
        if self._closed:
            raise ValueError("callable application is closed")
        if not self._lock.acquire(blocking=False):
            raise ValueError("one request at a time per callable application")
        self.last_receipt = None
        session = self._session
        try:
            if session is None:
                session = self._open()
            session.timeout = self.timeout
            receipt = session.run_values({"calls": calls})
            receipt["peak_tree_rss_bytes"] = session.peak_tree_rss_bytes
            receipt["memory_available"] = session.memory_available
            self.info.update(session.info)
            self.last_receipt = receipt
            return receipt["outputs"]
        finally:
            try:
                if self.mode == "oneshot" and session is not None:
                    session.close()
            finally:
                self._lock.release()

    def __call__(self, *args, **kwargs):
        return self._request([dict(args=list(args), kwargs=kwargs)])[0]

    def run_calls(self, calls):
        """Return one value per call; the calls share a factory within this request."""
        return self._request(calls)

    def close(self):
        self._closed = True
        if self._session is not None:
            self._session.close()

    def __enter__(self):
        if self._closed:
            raise ValueError("callable application is closed")
        return self

    def __exit__(self, *_):
        self.close()
