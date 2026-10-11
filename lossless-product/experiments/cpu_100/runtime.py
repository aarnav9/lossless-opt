"""Campaign execution alternatives; never imported by the product runtime."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
import os
import threading

_local = threading.local()


def report_memory():
    # Only memory workers publish PIDs; no instrumentation in timing workers.
    folder = os.environ.get("LOSSLESS_CPU_MEMORY_DIR")
    if folder:
        import json
        from pathlib import Path
        import resource
        import sys

        path = Path(folder) / f"{os.getpid()}.json"
        value = dict(
            pid=os.getpid(),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
        )
        temp = path.with_suffix(f".{threading.get_ident()}.tmp")
        temp.write_text(json.dumps(value))
        temp.replace(path)


def report_pool_helper():
    folder = os.environ.get("LOSSLESS_CPU_MEMORY_DIR")
    if folder:
        import json
        from pathlib import Path
        from multiprocessing import resource_tracker
        import psutil

        pid = resource_tracker._resource_tracker._pid
        if pid:
            try:
                rss = psutil.Process(pid).memory_info().rss
                path = Path(folder) / f"{pid}.json"
                temp = path.with_suffix(".tmp")
                temp.write_text(
                    json.dumps(dict(pid=pid, peak_rss_bytes=rss, auxiliary_sample=True))
                )
                temp.replace(path)
            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                pass


def retained_payload_bytes(value):
    """Unique NumPy backing storage and byte keys, excluding Python overhead."""
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
        if isinstance(item, bytes):
            return len(item)
        if isinstance(item, dict):
            return sum(size(k) + size(v) for k, v in item.items())
        if isinstance(item, (list, tuple)):
            return sum(size(v) for v in item)
        if type(item).__module__.startswith("pybaselines") and hasattr(item, "__dict__"):
            return size(vars(item))
        return 0

    return size(value)


def initialize_worker(reuse=True, barrier=None):
    report_memory()
    import spectrum_workload

    _local.processor = spectrum_workload.make_processor()
    _local.reuse = reuse
    if barrier is not None:
        barrier.wait(timeout=30)


def fit_chunk(records):
    import spectrum_workload

    if not hasattr(_local, "processor"):
        initialize_worker()
    processor = _local.processor if _local.reuse else spectrum_workload.make_processor()
    result = processor(records)
    report_memory()
    return result


def ready(_):
    # Nontrivial bounded job gives the executor time to start all configured workers.
    import time

    time.sleep(0.01)
    report_memory()
    return os.getpid()


class ParallelProcessor:
    def __init__(self, module, options):
        self.serial = module.make_processor()
        self.options = options
        self.pool = None
        if options.get("warm"):
            self.start()
            list(self.pool.map(ready, range(options["workers"] * 2)))
        if options.get("prime_serial"):
            import numpy as np

            grid = np.linspace(0, 1, 33)
            self.serial([dict(x=grid, y=1 + grid * grid, options=dict(poly_order=2, max_iter=2))])
            self.serial.fitters.clear()

    def start(self):
        if self.pool is None:
            opts = self.options
            cls = ThreadPoolExecutor if opts.get("threads") else ProcessPoolExecutor
            context = get_context("spawn")
            kw = {} if opts.get("threads") else {"mp_context": context}
            barrier = (
                (
                    threading.Barrier(opts["workers"])
                    if opts.get("threads")
                    else context.Barrier(opts["workers"])
                )
                if opts.get("warm")
                else None
            )
            self.pool = cls(
                max_workers=opts["workers"],
                initializer=initialize_worker,
                initargs=(opts.get("reuse", True), barrier),
                **kw,
            )
            if not opts.get("threads"):
                report_pool_helper()

    def __call__(self, records):
        o = self.options
        if o.get("early_batch") and len(records) < o.get("minimum_batch", 0):
            return self.serial(records)
        work = sum(r["y"].size for r in records)
        if len(records) < o.get("minimum_batch", 0) or work < o.get("minimum_work", 0):
            return self.serial(records)
        self.start()
        size = o.get("chunk", 1)
        if size == "balanced":
            size = max(1, (len(records) + o["workers"] - 1) // o["workers"])
        chunks = [records[i : i + size] for i in range(0, len(records), size)]
        if o.get("submit"):
            futures = [self.pool.submit(fit_chunk, chunk) for chunk in chunks]
            groups = [f.result() for f in futures]
        else:
            groups = list(self.pool.map(fit_chunk, chunks, chunksize=o.get("transport_chunk", 1)))
        return [item for group in groups for item in group]

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)


class PreparedProcessor:
    def __init__(self, module, options):
        self.module = module
        self.serial = module.make_processor()
        self.options = options
        self.plans = []
        self.front = None
        self.retained_bytes = 0

    def prepare_requests(self, calls):
        """Explicit metadata API: owned grid/order tokens, dynamic y/weights.

        The evaluator charges this once as preparation/setup and separately
        reports setup-inclusive timing. At most capacity distinct tokens are
        admitted; other metadata uses the original ordinary-call fallback.
        """
        import numpy as np

        plans = []
        prepared_calls = []
        for records in calls:
            request = []
            for x, y, options in self.serial.prepare(records):
                descriptor = (x.dtype.str, x.shape, options.get("poly_order", 2))
                bits = np.ascontiguousarray(x).view(np.uint8)
                token = next(
                    (
                        p
                        for p in plans
                        if p["descriptor"] == descriptor
                        and np.array_equal(bits, p["x"].view(np.uint8))
                    ),
                    None,
                )
                if (
                    token is None
                    and len(plans) < self.options["capacity"]
                    and x.size >= self.options.get("minimum_size", 0)
                ):
                    owned = np.ascontiguousarray(x).copy()
                    owned.flags.writeable = False
                    fitter = self.plan(owned, options)
                    token = dict(descriptor=descriptor, x=owned, fitter=fitter)
                    plans.append(token)
                request.append((token, x, y, options))
            prepared_calls.append(request)
        self.explicit_plans = plans
        return prepared_calls

    def run_prepared(self, request):
        results = []
        for token, x, y, options in request:
            if token is None:
                baseline, params = self.module.modpoly(y, x_data=x, **options)
            else:
                if options.get("poly_order", 2) != token["descriptor"][2]:
                    raise ValueError("prepared plan order changed; create a new plan")
                baseline, params = token["fitter"].modpoly(y, **options)
            results.append(dict(baseline=baseline, corrected=y - baseline, parameters=params))
        return results

    def cache_payload_bytes(self):
        return retained_payload_bytes(
            dict(
                plans=self.plans,
                explicit=getattr(self, "explicit_plans", []),
                fallback=vars(self.serial),
            )
        )

    def trim(self):
        # Count retained ndarray backing allocations and byte keys, including
        # polynomial matrices populated lazily by modpoly. Views count once.
        def retained():
            return retained_payload_bytes(self.plans)

        self.retained_bytes = retained()
        while self.plans and (
            len(self.plans) > self.options["capacity"]
            or self.retained_bytes > self.options["byte_limit"]
        ):
            self.plans.pop(0)
            self.retained_bytes = retained()
        if self.front is not None and not any(p is self.front for p in self.plans):
            self.front = None

    def plan(self, x, options):
        import hashlib
        import numpy as np
        from pybaselines import Baseline

        o = self.options
        order = options.get("poly_order", 2)
        descriptor = (x.dtype.str, x.shape, order)
        contiguous = np.ascontiguousarray(x)

        def same(p):
            return descriptor == p["descriptor"] and np.array_equal(
                contiguous.view(np.uint8), p["x"].view(np.uint8)
            )

        if o["last"] and self.front is not None and same(self.front):
            return self.front["fitter"]
        key = None
        if o["key"] == "bytes":
            key = contiguous.tobytes()
        elif o["key"] == "digest":
            key = hashlib.sha256(memoryview(contiguous).cast("B")).digest()
        for i, p in enumerate(self.plans):
            if descriptor == p["descriptor"] and (
                (o["key"] == "bytes" and key == p["key"])
                or (o["key"] == "digest" and key == p["key"] and same(p))
                or (o["key"] == "scan" and same(p))
            ):
                if o["lru"]:
                    self.plans.append(self.plans.pop(i))
                self.front = p
                return p["fitter"]
        # Own the key's bits even when the library itself holds an input view.
        grid = contiguous.copy()
        grid.flags.writeable = False
        fitter = Baseline(x_data=grid if o["own_grid"] else x)
        if (
            o["prewarm"]
            and not options.get("mask_initial_peaks")
            and options.get("weights") is None
        ):
            fitter._setup_polynomial(
                np.zeros(x.size), None, order, calc_vander=True, calc_pinv=True, copy_weights=True
            )
        if o.get("coefficient_prewarm") and options.get("return_coef") and 1 <= order + 1 <= 6:
            from pybaselines.utils import _poly_transform_matrix

            domain = fitter.x_domain
            transform = _poly_transform_matrix(order + 1, domain)
            k = (order + 1, domain.dtype.str, domain.strides, domain.tobytes())
            fitter._modpoly_coef_transform_cache = (k, transform)
        p = dict(descriptor=descriptor, key=key, x=grid, fitter=fitter)
        self.plans.append(p)
        self.front = p
        self.trim()
        return fitter

    def __call__(self, records):
        prepared = self.serial.prepare(records)
        o = self.options
        if o.get("clear_each_call"):
            self.plans.clear()
            self.front = None

        def select(x, opts):
            if x.size < o.get("minimum_size", 0) or (
                o["specialize_order"] and opts.get("weights") is not None
            ):
                return None
            return self.plan(x, opts)

        staged = [select(x, opts) for x, y, opts in prepared] if o["prepare_batch"] else None
        result = []
        for i, (x, y, opts) in enumerate(prepared):
            fitter = staged[i] if staged is not None else select(x, opts)
            baseline, params = (
                self.serial.fit(x, y, opts) if fitter is None else fitter.modpoly(y, **opts)
            )
            result.append(dict(baseline=baseline, corrected=y - baseline, parameters=params))
            if o.get("trim_each_fit", True):
                self.trim()
        if not o.get("trim_each_fit", True):
            self.trim()
        return result


def make_processor(module, config):
    if config.get("family") == 2:
        return ParallelProcessor(module, config["options"])
    if config.get("family") == 4:
        return PreparedProcessor(module, config["options"])
    return module.make_processor()
