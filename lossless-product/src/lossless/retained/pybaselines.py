"""Owned ModPoly metadata plans from the CPU/application studies.

Requires pybaselines 1.2.1. Source optimizations are separately proposed through
``lossless.applications.retained_plan``; this module never monkeypatches a user
installation. Only x/order are prepared; signals and weights stay dynamic.
"""

from ..execution import BoundedPlans, payload_bytes


class ModPolyPlan:
    def __init__(self, x, poly_order=2, *, prewarm=True):
        import numpy as np
        import pybaselines

        # The pinned source checkout lacks setuptools-scm's generated _version
        # file and reports this fallback; retained_plan also checks source SHA256.
        if pybaselines.__version__ not in {"1.2.1", "1.2.1.post1.dev0"}:
            raise ValueError(
                "retained ModPoly plans require pybaselines 1.2.1; requalify other versions"
            )
        if type(poly_order) is not int or poly_order < 0:
            raise ValueError("poly_order must be a nonnegative Python integer")
        x = np.array(x, copy=True, order="C")
        if x.ndim != 1 or not x.size or x.dtype.kind not in "fiu" or not np.isfinite(x).all():
            raise ValueError("x must be a nonempty finite numerical vector")
        x.flags.writeable = False
        self._x, self.poly_order = x, poly_order
        self._fitter = pybaselines.Baseline(x_data=x)
        if prewarm:
            # Same unit-weight matrix preparation as the library. Weighted
            # pseudo-inverses still depend on each call's weights.
            self._fitter._setup_polynomial(
                np.zeros(x.size),
                None,
                poly_order,
                calc_vander=True,
                calc_pinv=True,
                copy_weights=True,
            )

    def __call__(self, y, **options):
        if options.pop("poly_order", self.poly_order) != self.poly_order:
            raise ValueError("prepared polynomial order changed; prepare a new plan")
        return self._fitter.modpoly(y, poly_order=self.poly_order, **options)

    @property
    def retained_bytes(self):
        return payload_bytes(vars(self))


class ModPolyCache:
    """Ordinary-call cache: grid scanning/key creation remains inside the call."""

    def __init__(self, *, capacity=2, byte_limit=16 * 1024**2, prewarm=False):
        self.plans = BoundedPlans(capacity=capacity, byte_limit=byte_limit)
        self.prewarm = prewarm

    def __call__(self, x, y, **options):
        import numpy as np

        x = np.asarray(x)
        order = options.get("poly_order", 2)
        # Preserve ordinary library behavior for unusual order representations.
        if type(order) is not int:
            from pybaselines.polynomial import modpoly

            return modpoly(y, x_data=x, **options)
        key = (x.dtype.str, x.shape, x.tobytes(), order)
        plan = self.plans.get(key, lambda: ModPolyPlan(x, order, prewarm=self.prewarm))
        try:
            return plan(y, **options)
        finally:
            self.plans.trim()
