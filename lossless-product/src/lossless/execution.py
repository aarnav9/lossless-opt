"""Explicit reusable execution candidates; preparation/startup is caller-visible.

These primitives do not assert equivalence or accelerate arbitrary stateful code.
Compare the complete application, including setup, memory and retained outputs.
"""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import os


def payload_bytes(value):
    """Count unique NumPy backing buffers and byte keys in an owned object tree."""
    import numpy as np

    seen = set()

    def size(item):
        if id(item) in seen:
            return 0
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            root = item
            while isinstance(root.base, np.ndarray):
                root = root.base
            if root is not item:
                if id(root) in seen:
                    return 0
                seen.add(id(root))
            return root.nbytes
        if isinstance(item, (bytes, bytearray)):
            return len(item)
        if isinstance(item, dict):
            return sum(size(k) + size(v) for k, v in item.items())
        if isinstance(item, (list, tuple)):
            return sum(size(v) for v in item)
        if hasattr(item, "__dict__") and not isinstance(item, type):
            return size(vars(item))
        return 0

    return size(value)


class BoundedPlans:
    """Bounded FIFO metadata cache. Mutable values are accounted after each call.

    Call ``trim`` after lazily populating a plan. Oversized plans may serve the
    current call but are evicted before return. This bounds retained payload,
    not peak allocation or Python object overhead. Keys must represent *all*
    semantic metadata; never cache answers by signal identity.
    """

    def __init__(self, *, capacity=2, byte_limit=16 * 1024**2):
        if type(capacity) is not int or not 1 <= capacity <= 128:
            raise ValueError("capacity must be an integer in 1..128")
        if type(byte_limit) is not int or byte_limit < 1:
            raise ValueError("byte_limit must be a positive integer")
        self.capacity, self.byte_limit = capacity, byte_limit
        self.entries = {}

    def get(self, key, prepare):
        if key not in self.entries:
            self.entries[key] = prepare()
        value = self.entries[key]
        self.trim()
        return value

    def trim(self):
        while self.entries and (
            len(self.entries) > self.capacity or self.retained_bytes > self.byte_limit
        ):
            del self.entries[next(iter(self.entries))]

    @property
    def retained_bytes(self):
        return payload_bytes(self.entries)

    def clear(self):
        self.entries.clear()


_worker_processor = None
_worker_thread_limit = None


def _initialize(factory, barrier):
    global _worker_processor, _worker_thread_limit
    # A spawned child can import NumPy before its initializer. threadpoolctl
    # handles already-loaded libraries; environment variables handle later ones.
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    from threadpoolctl import threadpool_limits

    _worker_thread_limit = threadpool_limits(limits=1)
    _worker_processor = factory()
    if barrier is not None:
        barrier.wait(timeout=30)


def _chunk(records):
    return _worker_processor(records)


def _ready(_):
    return os.getpid()


class IndependentMap:
    """Persistent spawned workers for independent, order-preserving batches.

    ``factory`` must be a picklable top-level constructor of a callable taking
    a list and returning a same-length list. Its state must be performance-only:
    no cross-record dependencies, side effects or process-sensitive outputs.
    Keep construction under ``if __name__ == '__main__'`` in scripts. Use a
    context manager or ``close``. Exceptions propagate without duplicate replay.
    """

    def __init__(self, factory, *, workers=2, minimum_batch=4, warm=False):
        if type(workers) is not int or not 1 <= workers <= 4:
            raise ValueError("workers must be an integer in 1..4")
        if type(minimum_batch) is not int or minimum_batch < 1:
            raise ValueError("minimum_batch must be positive")
        # Diagnose the optional dependency before launching a child.
        import threadpoolctl  # noqa: F401

        self.factory, self.workers = factory, workers
        self.minimum_batch = minimum_batch
        self.serial = factory()
        self.pool = None
        self.closed = False
        if warm:
            self.start(warm=True)

    def start(self, *, warm=False):
        if self.closed:
            raise RuntimeError("executor is closed")
        if self.pool is None:
            context = get_context("spawn")
            barrier = context.Barrier(self.workers) if warm else None
            self.pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=context,
                initializer=_initialize,
                initargs=(self.factory, barrier),
            )
            if warm:
                try:
                    list(self.pool.map(_ready, range(self.workers)))
                except BaseException:
                    self.close()
                    raise

    def __call__(self, records):
        if self.closed:
            raise RuntimeError("executor is closed")
        if not isinstance(records, (list, tuple)):
            raise ValueError("records must be a finite list or tuple")
        if len(records) < self.minimum_batch or self.workers == 1:
            return self.serial(records)
        self.start()
        width = (len(records) + self.workers - 1) // self.workers
        chunks = [records[i : i + width] for i in range(0, len(records), width)]
        groups = list(self.pool.map(_chunk, chunks))
        if any(not isinstance(g, list) or len(g) != len(c) for g, c in zip(groups, chunks)):
            raise ValueError("independent processor must preserve batch cardinality")
        return [item for group in groups for item in group]

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
        close = getattr(self.serial, "close", None)
        if close:
            close()

    @property
    def fitters(self):
        """Compatibility with the spectrum observer; tree RSS covers children."""
        return getattr(self.serial, "fitters", {})

    def __enter__(self):
        if self.closed:
            raise RuntimeError("executor is closed")
        return self

    def __exit__(self, *_):
        self.close()
