"""Ordinary callable facade over the explicitly copied-value CPU boundary."""

from pathlib import Path
import time
import threading

from ..jobs import read_json
from .bundle import callable_policy
from .prepared import DEFAULT_FEATURES, PreparedApplication


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
        policy = callable_policy(self.bundle, manifest, (*DEFAULT_FEATURES, "values"))
        self.mode = mode or (policy["mode"] if policy else "oneshot")
        self.variant = variant
        self._qualifying = _qualifying
        reason = None
        if not _qualifying:
            if policy is None or policy["mode"] != self.mode:
                self.variant, self.mode = "reference", "oneshot"
                reason = (
                    "callable deployment certificate missing, invalid or for another mode; "
                    "using fresh reference; re-run run-project with --deployment"
                )
            elif policy["status"] == "reference_retained":
                self.variant = "reference"
                reason = "reference retained by deployment qualification"
        self._fallback_reason = reason
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
            self._update_info(self._session)

    def _update_info(self, session):
        self.info.update(session.info)
        # The worker rechecks the certificate and environment before imports.
        # A stale prepared qualification uses this worker for one request only.
        if self.mode != self.info["mode"] and self.info.get("fallback_reason"):
            self._fallback_reason = self.info["fallback_reason"]
        self.mode = self.info["mode"]
        if self._fallback_reason and self.info.get("fallback_reason") in {
            None,
            "reference requested or retained",
            "callable deployment has not qualified for this mode; using fresh reference",
        }:
            self.info["fallback_reason"] = self._fallback_reason

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
            self._update_info(session)
            session.timeout = self.timeout
            receipt = session.run_values({"calls": calls})
            receipt["peak_tree_rss_bytes"] = session.peak_tree_rss_bytes
            receipt["memory_available"] = session.memory_available
            self.last_receipt = receipt
            return receipt["outputs"]
        finally:
            try:
                if self.mode == "oneshot" and session is not None:
                    session.close()
                    self._session = None
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
