"""Explicit numerical candidates from campaigns 001 and 029–031.

Real-arithmetic identities do not imply bitwise equality. These APIs require
an explicit ``contract='numerical'`` and do not enter exact source search.
No model-quality claim is made. Setup belongs in the measured lifecycle.
"""

import numpy as np


def _contract(contract):
    if contract != "numerical":
        raise ValueError("this recipe requires contract='numerical'; it is not bitwise exact")


def matrix_chain(arrays, *, contract):
    """Use NumPy's established minimum-work parenthesization."""
    _contract(contract)
    return np.linalg.multi_dot(arrays)


class RefinedLeastSquaresPlan:
    """Prepared, conditioned version of the campaign-030 numerical lead.

    Setup computes FP64 singular values. Normal equations are admitted only for
    full-rank matrices with condition <=100; two FP64 residual corrections and
    a measured gradient error estimate check every RHS. Other cases use FP64
    pivoted QR. This extra admission/fallback is new and needs fresh speed tests.
    """

    def __init__(self, x, *, contract, rtol=1e-5, atol=1e-7):
        _contract(contract)
        from scipy import linalg

        x = np.array(x, dtype=np.float32, order="C", copy=True)
        if x.ndim != 2 or min(x.shape) < 1 or x.shape[0] < x.shape[1] or not np.isfinite(x).all():
            raise ValueError("finite overdetermined or square design matrix required")
        if not (np.isfinite(rtol) and np.isfinite(atol) and 0 < rtol <= 1e-3 and 0 <= atol <= 1e-3):
            raise ValueError("invalid measured numerical budget")
        self._x, self._xd = x, x.astype(np.float64)
        singular = linalg.svdvals(self._xd)
        self._minimum = float(singular[-1])
        self.rtol, self.atol = rtol, atol
        self._factor = None
        if self._minimum > 0 and singular[0] / self._minimum <= 100:
            self._factor = linalg.lu_factor(x.T @ x)

    def __call__(self, y):
        from scipy import linalg

        y = np.asarray(y)
        if y.dtype != np.float32 or y.shape != (self._x.shape[0],) or not np.isfinite(y).all():
            raise ValueError("finite FP32 response vector required")
        yd = y.astype(np.float64)
        if self._factor is not None:
            w = linalg.lu_solve(self._factor, self._x.T @ y)
            for _ in range(2):
                residual = self._xd.T @ (yd - self._xd @ w.astype(np.float64))
                w += linalg.lu_solve(self._factor, residual.astype(np.float32))
            gradient = self._xd.T @ (self._xd @ w.astype(np.float64) - yd)
            estimated_error = linalg.norm(gradient) / self._minimum**2
            if np.isfinite(w).all() and estimated_error <= self.atol + self.rtol * linalg.norm(w):
                return w
        reference = linalg.lstsq(self._xd, yd, lapack_driver="gelsy")[0]
        result = reference.astype(np.float32)
        if not np.isfinite(result).all() or linalg.norm(
            result.astype(np.float64) - reference
        ) > self.atol + self.rtol * linalg.norm(reference):
            raise ArithmeticError("least-squares fallback exceeds the measured rounding budget")
        return result


class LowRankIdentityPlan:
    """Solve the explicit system (I + U U.T)x=b, with measured FP64 residual.

    Owns U and its small factor; never approximates an arbitrary dense matrix.
    The residual check is empirical (not outward enclosed). A failed FP32
    candidate falls back to an FP64 dense solve; failure after rounding raises.
    """

    def __init__(self, u, *, contract, rtol=1e-5, atol=1e-7):
        _contract(contract)
        from scipy.linalg import cho_factor

        u = np.array(u, dtype=np.float32, copy=True, order="C")
        if u.ndim != 2 or min(u.shape) < 1 or not np.isfinite(u).all():
            raise ValueError("U must be a nonempty finite matrix")
        if not (np.isfinite(rtol) and np.isfinite(atol) and 0 < rtol <= 1e-3 and 0 <= atol <= 1e-3):
            raise ValueError("invalid numerical residual budget")
        self._u, self._ud = u, u.astype(np.float64)
        self.rtol, self.atol = rtol, atol
        self._factor = cho_factor(np.eye(u.shape[1], dtype=np.float32) + u.T @ u)
        self._dense = None

    def __call__(self, b):
        from scipy.linalg import cho_factor, cho_solve, norm

        b = np.asarray(b)
        if (
            b.dtype != np.float32
            or b.ndim not in (1, 2)
            or b.shape[0] != self._u.shape[0]
            or not b.size
            or not np.isfinite(b).all()
        ):
            raise ValueError("b must be finite FP32 with matching leading dimension")
        x = b - self._u @ cho_solve(self._factor, self._u.T @ b)
        bd = b.astype(np.float64)

        def check(value):
            xd = value.astype(np.float64)
            residual = bd - (xd + self._ud @ (self._ud.T @ xd))
            measured = float(norm(residual))
            return np.isfinite(measured) and measured <= self.atol + self.rtol * float(norm(xd))

        if not check(x):
            if self._dense is None:
                self._dense = cho_factor(np.eye(len(self._u)) + self._ud @ self._ud.T)
            x = cho_solve(self._dense, bd).astype(np.float32)
            if not check(x):
                raise ArithmeticError("measured residual budget failed, including FP64 fallback")
        return x


class PeriodicDeblurPlan:
    """Real-FFT ridge deblurring for the studied 5-tap separable binomial blur.

    Minimizes .5||Hx-y||² + regularization/2||x||² with periodic boundaries.
    Finite FP32 images only. This is a numerical alternative to iterative CG;
    it does not apply to arbitrary kernels, reflect padding or learned models.
    """

    def __init__(self, shape, regularization, *, contract, boundary="wrap"):
        _contract(contract)
        from scipy import fft

        shape = tuple(shape)
        lam = np.float32(regularization)
        if len(shape) != 2 or any(type(n) is not int or n < 5 for n in shape):
            raise ValueError("shape must contain two integer dimensions >=5")
        if boundary != "wrap" or not np.isfinite(lam) or lam <= 0:
            raise ValueError("positive finite FP32 regularization and periodic boundaries required")
        kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32) / np.float32(16)
        impulse = np.zeros(shape, dtype=np.float32)
        for i, a in enumerate(kernel):
            for j, b in enumerate(kernel):
                impulse[(i - 2) % shape[0], (j - 2) % shape[1]] += a * b
        h = fft.rfft2(impulse, workers=1)
        self._gain = np.conj(h) / (np.abs(h) ** 2 + lam)
        self._gain.flags.writeable = False
        self.shape = shape

    def __call__(self, y):
        from scipy import fft

        if (
            not isinstance(y, np.ndarray)
            or y.shape != self.shape
            or y.dtype != np.float32
            or not np.isfinite(y).all()
        ):
            raise ValueError("requires a finite FP32 image of the prepared shape")
        spectrum = fft.rfft2(y, workers=1)
        spectrum *= self._gain
        result = fft.irfft2(spectrum, s=self.shape, workers=1, overwrite_x=True)
        if not np.isfinite(result).all():
            raise ArithmeticError("deblurring overflowed the numerical domain")
        return result


def ridge_steps(x, y, initial, *, regularization, rate, steps, contract):
    """Matrix-free ridge gradients for a declared short, fixed iteration count.

    Avoids Gram construction. This changes FP evaluation order and may lose for
    long trajectories; no convergence or equal-iterate claim is implicit.
    """
    _contract(contract)
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    x, y, w = np.asarray(x), np.asarray(y), np.array(initial, copy=True)
    if x.ndim != 2 or y.shape != (x.shape[0],) or w.shape != (x.shape[1],):
        raise ValueError("incompatible ridge shapes")
    if not all(a.dtype == np.float32 and np.isfinite(a).all() for a in (x, y, w)):
        raise ValueError("finite FP32 operands required")
    if not np.isfinite(regularization) or regularization < 0 or not np.isfinite(rate) or rate <= 0:
        raise ValueError("invalid regularization or step size")
    for _ in range(steps):
        w -= np.float32(rate) * (x.T @ (x @ w - y) + np.float32(regularization) * w)
    if not np.isfinite(w).all():
        raise ArithmeticError("ridge trajectory is nonfinite")
    return w
