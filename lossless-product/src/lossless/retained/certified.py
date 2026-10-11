"""Conditional FP64 residual certificates, campaigns 015/016.

The bound assumes the recorded floating-point rounding model; it is not a
BLAS implementation proof. Failed bounds never return accepted solutions.
"""

from dataclasses import dataclass
from fractions import Fraction
import math
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.lapack import get_lapack_funcs

TOL = 1e-8
SCALE = 2**20


def up(x):
    return float(np.nextafter(float(x), math.inf))


@dataclass(frozen=True)
class System:
    a: np.ndarray
    delta: float
    anorm: float

    @classmethod
    def from_units(cls, units):
        units = np.asarray(units)
        if (
            units.ndim != 2
            or units.shape[0] == 0
            or units.shape[0] != units.shape[1]
            or units.dtype.kind not in "iu"
        ):
            raise ValueError("square integer matrix required")
        if not np.array_equal(units, units.T):
            raise ValueError("symmetric matrix required")
        # Keep all integer sums and their FP64 dyadic conversions exact.
        if max(abs(int(units.min())), abs(int(units.max()))) * units.shape[0] >= 2**52:
            raise ValueError("integer row sum exceeds exact range")
        units = units.astype(np.int64)
        rows = np.abs(units).sum(axis=1)
        margins = 2 * np.diag(units) - rows
        if margins.min() <= 0:
            raise ValueError("positive strict diagonal dominance required")
        a = units.astype(np.float64) / SCALE
        a.flags.writeable = False
        return cls(a, int(margins.min()) / SCALE, int(rows.max()) / SCALE)


def certify(system, b, x):
    """Bound ||x-x*||inf under FP64 dot/subtract rounding assumptions.

    Strict DD implies ||A^-1||inf <= 1/delta. The residual is enclosed by
    gamma_(2n+4) (||A||inf ||x||inf + ||b||inf), plus an underflow cushion.
    Every positive scalar operation in the enclosure rounds outward.
    """
    if b.shape != x.shape or b.shape != (len(system.a),):
        raise ValueError("one conforming RHS required")
    if b.dtype != np.float64 or x.dtype != np.float64:
        raise ValueError("FP64 residual operands required")
    if not (np.isfinite(b).all() and np.isfinite(x).all()):
        raise ValueError("finite operands required")
    n = len(x)
    k = 2 * n + 4
    ku = Fraction(k, 2**53)
    if ku >= 1:
        raise ValueError("invalid rounding factor")
    gamma = up(float(ku / (1 - ku)))
    magnitude = up(up(system.anorm * float(np.max(np.abs(x)))) + float(np.max(np.abs(b))))
    if not math.isfinite(magnitude):
        raise ValueError("overflow not certifiable")
    residual = b - system.a @ x
    if not np.isfinite(residual).all():
        raise ValueError("nonfinite computed residual")
    tiny = Fraction.from_float(float(np.finfo(np.float64).tiny))
    underflow = up(float((k + 1) * tiny / (1 - ku)))
    roundoff = up(up(gamma * magnitude) + underflow)
    residual_upper = up(float(np.max(np.abs(residual))) + roundoff)
    bound = up(residual_upper / system.delta)
    return residual, {
        "bound_inf": bound,
        "roundoff_enclosure": roundoff,
        "computed_residual_inf": float(np.max(np.abs(residual))),
        "passed": bound <= TOL,
    }


def factor(a, dtype):
    return cho_factor(
        np.array(a, dtype=dtype, order="F", copy=True),
        lower=True,
        overwrite_a=True,
        check_finite=False,
    )


def solve(system, b, name, factors):
    mixed = "mixed" in name
    fresh = name.startswith("fresh")
    fac = (
        factor(system.a, np.float32 if mixed else np.float64)
        if fresh
        else factors[32 if mixed else 64]
    )
    x = cho_solve(fac, b.astype(np.float32) if mixed else b, check_finite=False).astype(np.float64)
    if name.endswith("plain"):
        return x, {"corrections": 0, "fallback": False, "certificate": None}
    r, cert = certify(system, b, x)
    corrections = 0
    while mixed and not cert["passed"] and corrections < 4:
        correction = cho_solve(fac, r.astype(np.float32), check_finite=False)
        x += correction.astype(np.float64)
        corrections += 1
        r, cert = certify(system, b, x)
    fallback = bool(mixed and not cert["passed"])
    if fallback:
        # Fresh policy includes fallback factorization; cached policy uses its
        # separately accounted FP64 safety factor.
        f64 = factor(system.a, np.float64) if fresh else factors[64]
        x = cho_solve(f64, b, check_finite=False)
        r, cert = certify(system, b, x)
    return x, {"corrections": corrections, "fallback": fallback, "certificate": cert}


class TridiagonalMatrix:
    """Exact representation of constructed path systems for residual products.

    At most seven FP64 scalar operations per residual entry (including b-Ax),
    covered by the existing gamma_(2n+4) envelope for n>=2. No entries are dropped.
    """

    def __init__(self, ab):
        self.diagonal = ab[0].copy()
        self.off = ab[1, :-1].copy()

    def __len__(self):
        return len(self.diagonal)

    def __matmul__(self, x):
        if x.ndim not in (1, 2):
            raise ValueError("vector or RHS matrix required")
        d = self.diagonal if x.ndim == 1 else self.diagonal[:, None]
        e = self.off if x.ndim == 1 else self.off[:, None]
        y = d * x
        y[:-1] += e * x[1:]
        y[1:] += e * x[:-1]
        return y


class PreparedCertificate:
    """Reuse rounding constants; optional operation-count bound for path residuals.

    The tridiagonal product plus subtraction uses at most seven scalar operations
    per entry. k=8 conservatively encloses them; no zero matrix entries are dropped
    approximately. Scalar nextafter operations always round positive bounds up.
    """

    def __init__(self, n, tight=False):
        if n < 2:
            raise ValueError("path certificate requires n>=2")
        self.n, self.tight = n, tight
        self.k = 8 if tight else 2 * n + 4
        ku = Fraction(self.k, 2**53)
        if ku >= 1:
            raise ValueError("invalid rounding factor")
        self.gamma = math.nextafter(float(ku / (1 - ku)), math.inf)
        tiny = Fraction.from_float(float(np.finfo(np.float64).tiny))
        self.underflow = math.nextafter(float((self.k + 1) * tiny / (1 - ku)), math.inf)

    def __call__(self, system, b, x):
        if not isinstance(system.a, TridiagonalMatrix) or len(system.a) != self.n:
            raise ValueError("matching exact tridiagonal representation required")
        if (
            b.shape != x.shape
            or b.shape != (self.n,)
            or b.dtype != np.float64
            or x.dtype != np.float64
        ):
            raise ValueError("conforming FP64 vectors required")
        if not (np.isfinite(b).all() and np.isfinite(x).all()):
            raise ValueError("finite operands required")

        def up(v):
            return math.nextafter(float(v), math.inf)

        magnitude = up(up(system.anorm * float(np.max(np.abs(x)))) + float(np.max(np.abs(b))))
        if not math.isfinite(magnitude):
            raise ValueError("overflow not certifiable")
        residual = b - system.a @ x
        if not np.isfinite(residual).all():
            raise ValueError("nonfinite computed residual")
        rnorm = float(np.max(np.abs(residual)))
        roundoff = up(up(self.gamma * magnitude) + self.underflow)
        bound = up(up(rnorm + roundoff) / system.delta)
        return residual, {
            "bound_inf": bound,
            "roundoff_enclosure": roundoff,
            "computed_residual_inf": rnorm,
            "passed": bound <= TOL,
            "rounding_operations": self.k,
        }

    def many(self, system, b, x):
        if not isinstance(system.a, TridiagonalMatrix) or len(system.a) != self.n:
            raise ValueError("matching exact tridiagonal representation required")
        if (
            b.shape != x.shape
            or b.ndim != 2
            or b.shape[0] != self.n
            or b.shape[1] < 1
            or b.dtype != np.float64
            or x.dtype != np.float64
        ):
            raise ValueError("conforming FP64 RHS matrices required")
        if not (np.isfinite(b).all() and np.isfinite(x).all()):
            raise ValueError("finite operands required")

        def up(v):
            return np.nextafter(v, np.inf)

        magnitude = up(up(system.anorm * np.max(np.abs(x), axis=0)) + np.max(np.abs(b), axis=0))
        if not np.isfinite(magnitude).all():
            raise ValueError("overflow not certifiable")
        residual = b - system.a @ x
        if not np.isfinite(residual).all():
            raise ValueError("nonfinite computed residual")
        rnorm = np.max(np.abs(residual), axis=0)
        roundoff = up(up(self.gamma * magnitude) + self.underflow)
        bound = up(up(rnorm + roundoff) / system.delta)
        return residual, [
            {
                "bound_inf": float(v),
                "roundoff_enclosure": float(e),
                "computed_residual_inf": float(r),
                "passed": bool(v <= TOL),
                "rounding_operations": self.k,
            }
            for v, e, r in zip(bound, roundoff, rnorm)
        ]


class CertifiedDensePlan:
    """Owned dyadic strictly-DD SPD matrix, refinement and checked fallback."""

    def __init__(self, units, *, contract, mixed=True):
        if contract != "numerical":
            raise ValueError("conditional forward error is a numerical contract")
        self._system = System.from_units(units)
        self._mixed = bool(mixed)
        self._factors = {64: factor(self._system.a, np.float64)}
        if mixed:
            try:
                self._factors[32] = factor(self._system.a, np.float32)
            except (ValueError, np.linalg.LinAlgError):
                self._mixed = False

    def __call__(self, b):
        b = np.asarray(b)
        if b.dtype != np.float64 or b.shape != (len(self._system.a),) or not np.isfinite(b).all():
            raise ValueError("matching finite FP64 RHS vector required")
        x, info = solve(
            self._system,
            b,
            "cached_mixed_checked" if self._mixed else "cached64_checked",
            self._factors,
        )
        if not info["certificate"]["passed"]:
            raise ArithmeticError("conditional forward-error bound failed, including fallback")
        return x, info


class CertifiedPathPlan:
    """Batched cached LAPACK path solve with vectorized outward residual bounds.

    Matrix entries are integer units / 2**20. Exact symmetry and strict diagonal
    dominance are checked in setup. No approximate sparsification is performed.
    """

    def __init__(self, diagonal_units, off_diagonal_units, *, contract):
        if contract != "numerical":
            raise ValueError("conditional forward error is a numerical contract")
        d, e = np.asarray(diagonal_units), np.asarray(off_diagonal_units)
        if (
            d.ndim != 1
            or d.size < 2
            or e.shape != (d.size - 1,)
            or d.dtype.kind not in "iu"
            or e.dtype.kind not in "iu"
        ):
            raise ValueError("integer diagonal and subdiagonal vectors required")
        if (
            max(abs(int(d.min())), abs(int(d.max())), abs(int(e.min())), abs(int(e.max()))) * 3
            >= 2**52
        ):
            raise ValueError("dyadic row arithmetic exceeds the exact range")
        d, e = d.astype(np.int64), e.astype(np.int64)
        offsum = np.zeros_like(d)
        offsum[:-1] += np.abs(e)
        offsum[1:] += np.abs(e)
        margin = d - offsum
        if margin.min() <= 0:
            raise ValueError("positive strict diagonal dominance required")
        ab = np.zeros((2, d.size), dtype=np.float64)
        ab[0], ab[1, :-1] = d / SCALE, e / SCALE
        self._system = System(
            TridiagonalMatrix(ab), int(margin.min()) / SCALE, int((d + offsum).max()) / SCALE
        )
        self._certificate = PreparedCertificate(d.size, tight=True)
        pttrf, self._pttrs = get_lapack_funcs(("pttrf", "pttrs"), dtype=np.float64)
        fd, fe, info = pttrf(ab[0].copy(), ab[1, :-1].copy(), overwrite_d=True, overwrite_e=True)
        if info:
            raise ArithmeticError(f"path factorization failed: {info}")
        self._factors = fd, fe

    def __call__(self, b):
        b = np.asarray(b)
        n = len(self._system.a)
        if (
            b.dtype != np.float64
            or b.ndim not in (1, 2)
            or b.shape[0] != n
            or not b.size
            or not np.isfinite(b).all()
        ):
            raise ValueError("matching finite FP64 RHS vector or matrix required")
        rhs = np.array(b[:, None] if b.ndim == 1 else b, order="F", copy=True)
        x, info = self._pttrs(*self._factors, rhs, overwrite_b=False)
        if info:
            raise ArithmeticError(f"path solve failed: {info}")
        certificates = self._certificate.many(self._system, rhs, x)[1]
        if not all(c["passed"] for c in certificates):
            raise ArithmeticError("conditional forward-error bound failed")
        return (x[:, 0] if b.ndim == 1 else x), certificates
