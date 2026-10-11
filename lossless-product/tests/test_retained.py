"""Contract, ownership and lifecycle regressions for executable research recipes."""

import importlib.util
import platform
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lossless.execution import BoundedPlans, IndependentMap, payload_bytes
from lossless.recipes import native_candidates


class SquareBatch:
    def __call__(self, records):
        if any(n < 0 for n in records):
            raise ValueError("negative record")
        return [n * n for n in records]


class ExecutionTests(unittest.TestCase):
    def test_search_race_is_rejection_only_and_validates_timings(self):
        from lossless.retained.racing import BoundedRace

        race = BoundedRace(["slow", "tie", "fast"])
        for _ in range(100):
            if "slow" not in race.rejected:
                race.observe("slow", 1, 8)
            self.assertFalse(race.observe("tie", 1, 1)["reject"])
            self.assertFalse(race.observe("fast", 1, 0.5)["reject"])
        self.assertIn("slow", race.rejected)
        with self.assertRaises(ValueError):
            race.observe("fast", 1, float("nan"))

    def test_byte_pipeline_exhaustive_and_strided(self):
        from lossless.retained.uint8 import Uint8Pipeline

        operations = [
            ("min", 224),
            ("max", 32),
            ("invert", 0),
            ("right_shift", 1),
            ("add", 250),
            ("xor", 85),
        ]
        for strategy in ("native", "lookup"):
            pipeline = Uint8Pipeline(operations, strategy=strategy)
            for x in (
                np.array(193, dtype=np.uint8),
                np.arange(256, dtype=np.uint8),
                np.tile(np.arange(256, dtype=np.uint8), (3, 1)).T,
            ):
                before = x.tobytes()
                actual = pipeline(x)
                self.assertEqual(actual.shape, x.shape)
                self.assertEqual(actual.tobytes(), pipeline.reference(x).tobytes())
                retained = actual.tobytes()
                pipeline(np.zeros_like(x))
                self.assertEqual(retained, actual.tobytes())
                self.assertEqual(before, x.tobytes())
                self.assertFalse(np.shares_memory(x, actual))

    def test_catalog_contract_separation_and_additive_planner(self):
        from lossless.recipes import catalog
        from lossless.retained.planning import dynamic_program, validate_plan

        self.assertGreaterEqual(len(catalog()), 25)
        self.assertTrue(all(c["contract"] == "exact" for c in catalog(contract="exact")))
        costs = [
            dict(id="a", start=0, stop=1, cost_ns=10),
            dict(id="b", start=1, stop=2, cost_ns=10),
            dict(id="fused", start=0, stop=2, cost_ns=12),
        ]
        selected = dynamic_program(costs, 2)
        self.assertEqual(selected["plan"], ["fused"])
        with self.assertRaises(ValueError):
            validate_plan(["b", "a"], costs, 2)

    def test_owned_payload_and_lazy_eviction(self):
        a = np.zeros(128)
        self.assertEqual(payload_bytes([a, a[1:]]), a.nbytes)
        plans = BoundedPlans(capacity=2, byte_limit=1100)
        first = plans.get("a", lambda: {"a": a})
        first["new"] = a.copy()
        plans.trim()
        self.assertEqual(plans.retained_bytes, 0)
        plans.get("b", lambda: np.zeros(8))
        plans.get("c", lambda: np.zeros(8))
        plans.get("d", lambda: np.zeros(8))
        self.assertEqual(list(plans.entries), ["c", "d"])

    @unittest.skipUnless(importlib.util.find_spec("threadpoolctl"), "application extra required")
    def test_pool_order_errors_and_close(self):
        with IndependentMap(SquareBatch, workers=2, warm=True) as operation:
            self.assertEqual(operation([3]), [9])
            self.assertEqual(operation(list(range(17))), [n * n for n in range(17)])
            with self.assertRaises(ValueError):
                operation([1, 2, -1, 4])
            self.assertEqual(operation([4, 3, 2, 1]), [16, 9, 4, 1])
        with self.assertRaises(RuntimeError):
            operation([1])
        operation.close()

    @unittest.skipUnless(importlib.util.find_spec("z3"), "z3-solver required")
    def test_bounded_copy_proof_rejects_bad_mapping(self):
        from lossless.retained.copy_proof import candidate

        cases = [dict(rows=7, columns=13, layout=layout) for layout in ("c", "f")]
        for order in ("row", "column", "reverse"):
            self.assertTrue(candidate(cases, order=order)["admitted"])
        for defect in ("swapped", "offbyone", "drop"):
            result = candidate(cases, defect=defect)
            self.assertFalse(result["admitted"])
            self.assertIsNone(result["source"])

    def test_retained_native_pool_independent_oracle_and_redzones(self):
        from lossless._native import operators, worker

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for op in ("softmax", "rmsnorm_residual"):
                proposals = native_candidates(op, [dict(rows=64, columns=257, layout="f")])
                for proposal in proposals:
                    source = root / (proposal["id"] + ".c")
                    binary = source.with_suffix(".so")
                    source.write_text(proposal["source"])
                    result = worker.compile_job(dict(source=str(source), binary=str(binary)))
                    self.assertEqual(result["status"], "ok", result)
                    lib = operators.load_library(binary)
                    for rows, cols, layout, distribution in (
                        (3, 17, "c", "gaussian"),
                        (19, 257, "f", "gaussian"),
                        (7, 31, "slice2", "tiny"),
                        (64, 257, "f", "gaussian"),
                    ):
                        case = dict(rows=rows, columns=cols, layout=layout)
                        arrays = operators.input_arrays(op, case, 820119, distribution)
                        before = operators.digest(arrays)
                        oracle = operators.reference(op, arrays)
                        functions, guards, poison = operators.executors(op, case, arrays, lib, lib)
                        poison()
                        actual = functions["proposal"]()
                        self.assertTrue(
                            operators.check(op, actual, oracle)["passed"], (proposal["id"], case)
                        )
                        self.assertTrue(guards())
                        self.assertEqual(before, operators.digest(arrays))

    def test_softmax_scope_and_source(self):
        small = list(native_candidates("softmax", [dict(rows=4, columns=128, layout="f")]))
        large = list(native_candidates("softmax", [dict(rows=64, columns=256, layout="f")]))
        self.assertNotIn("softmax_fortran_046", [p["id"] for p in small])
        expected = (
            "softmax_fortran_046"
            if platform.machine() in {"arm64", "aarch64"}
            else "softmax_row_16"
        )
        self.assertEqual(large[0]["id"], expected)
        self.assertIn("void kernel", large[0]["source"])

    def test_wrapper_policy_fresh_inputs_invalid_values_and_output_ownership(self):
        from lossless._native import operators, worker
        from lossless._native.deployment import NativeOperation

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for op in ("copy", "softmax", "rmsnorm_residual"):
                source, binary = tmp / (op + ".c"), tmp / (op + ".so")
                source.write_text(operators.baseline_source(op))
                self.assertEqual(
                    worker.compile_job(dict(source=str(source), binary=str(binary)))["status"], "ok"
                )
                cases = [dict(rows=3, columns=19, layout="c"), dict(rows=5, columns=23, layout="f")]
                original = NativeOperation.from_loaded_library(
                    op, operators.load_library(binary), cases
                )
                candidate = original.optimized_calls()
                for case in cases:
                    for seed in (121, 442, 773):
                        arrays = operators.input_arrays(op, case, seed)
                        args = arrays if op == "rmsnorm_residual" else arrays[:1]
                        before = [a.tobytes() for a in args]
                        result = candidate(*args)
                        retained = result.tobytes()
                        self.assertEqual(retained, original(*args).tobytes())
                        candidate(*args)
                        self.assertEqual(retained, result.tobytes())
                        self.assertEqual(before, [a.tobytes() for a in args])
                        if op != "copy":
                            arrays[0][0, 0] = np.nan
                            with self.assertRaises(ValueError):
                                candidate(*args)
                # Out-of-artifact shape must still use the independent fallback.
                arrays = operators.input_arrays(op, dict(rows=7, columns=11, layout="c"), 839)
                args = arrays if op == "rmsnorm_residual" else arrays[:1]
                self.assertEqual(candidate(*args).tobytes(), original(*args).tobytes())
                # Unaligned NumPy views must use safe validation/fallback.
                buffer = bytearray(3 * 19 * 4 + 1)
                x = np.ndarray((3, 19), dtype=np.float32, buffer=buffer, offset=1)
                x[:] = 0.1
                self.assertFalse(x.flags.aligned)
                extra = (
                    (np.zeros_like(x), np.ones(19, dtype=np.float32))
                    if op == "rmsnorm_residual"
                    else ()
                )
                self.assertEqual(candidate(x, *extra).tobytes(), original(x, *extra).tobytes())


@unittest.skipUnless(importlib.util.find_spec("scipy"), "numerical extra required")
class NumericalTests(unittest.TestCase):
    def test_conditioned_least_squares_and_matrix_free_steps(self):
        from lossless.retained.numerical import RefinedLeastSquaresPlan, ridge_steps

        rng = np.random.default_rng(803201)
        x = rng.normal(size=(40, 5)).astype(np.float32)
        y = rng.normal(size=40).astype(np.float32)
        for factor in (1, 1e-6):
            x[:, -1] *= np.float32(factor)
            plan = RefinedLeastSquaresPlan(x, contract="numerical")
            result = plan(y)
            oracle = np.linalg.lstsq(x.astype(float), y.astype(float), rcond=None)[0]
            np.testing.assert_allclose(result, oracle, rtol=1e-5, atol=1e-6)
        initial = np.zeros(5, dtype=np.float32)
        actual = ridge_steps(
            x, y, initial, regularization=0.1, rate=0.001, steps=4, contract="numerical"
        )
        w = np.zeros(5)
        for _ in range(4):
            w -= 0.001 * (x.astype(float).T @ (x.astype(float) @ w - y) + 0.1 * w)
        np.testing.assert_allclose(actual, w, rtol=1e-5, atol=1e-7)
        np.testing.assert_array_equal(initial, np.zeros(5, dtype=np.float32))

    def test_woodbury_owned_metadata_and_residual(self):
        from lossless.retained.numerical import LowRankIdentityPlan

        rng = np.random.default_rng(15031)
        u = (rng.normal(size=(64, 4)) * 0.02).astype(np.float32)
        original = u.copy()
        plan = LowRankIdentityPlan(u, contract="numerical")
        u[:] = 12
        b = rng.normal(size=(64, 2)).astype(np.float32)
        before = b.copy()
        result = plan(b)
        reference = np.linalg.solve(
            np.eye(64) + original.astype(float) @ original.astype(float).T, b
        )
        np.testing.assert_allclose(result, reference, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(b, before)
        with self.assertRaises(ValueError):
            LowRankIdentityPlan(original, contract="exact")
        with self.assertRaises(ValueError):
            plan(np.full((64,), np.nan, dtype=np.float32))

    def test_deblur_independent_dense_oracle_and_ownership(self):
        from scipy.ndimage import convolve1d
        from lossless.retained.numerical import PeriodicDeblurPlan

        shape = (7, 9)
        k = np.array([1, 4, 6, 4, 1], dtype=float) / 16

        def blur(x):
            return convolve1d(convolve1d(x, k, axis=0, mode="wrap"), k, axis=1, mode="wrap")

        h = np.column_stack([blur(row.reshape(shape)).ravel() for row in np.eye(63)])
        y = np.random.default_rng(2991).normal(size=shape).astype(np.float32)
        plan = PeriodicDeblurPlan(shape, 0.003, contract="numerical")
        result = plan(y)
        oracle = np.linalg.solve(
            h.T @ h + float(np.float32(0.003)) * np.eye(63), h.T @ y.ravel()
        ).reshape(shape)
        np.testing.assert_allclose(result, oracle, rtol=1e-4, atol=2e-5)
        retained = result.tobytes()
        plan(y + 1)
        self.assertEqual(result.tobytes(), retained)
        self.assertFalse(np.shares_memory(result, y))
        with self.assertRaises(ValueError):
            PeriodicDeblurPlan(shape, 0.003, contract="numerical", boundary="reflect")

    def test_certified_path_and_dense_reject_invalid_systems(self):
        from lossless.retained.certified import CertifiedPathPlan, CertifiedDensePlan, SCALE

        n = 17
        diagonal = np.full(n, 3 * SCALE, dtype=np.int64)
        off = np.full(n - 1, -SCALE, dtype=np.int64)
        matrix = np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1)
        rhs = np.random.default_rng(5041).normal(size=(n, 5))
        path = CertifiedPathPlan(diagonal, off, contract="numerical")
        result, checks = path(rhs)
        oracle = np.linalg.solve(matrix / SCALE, rhs)
        self.assertTrue(all(c["passed"] for c in checks))
        self.assertLessEqual(np.max(abs(result - oracle)), max(c["bound_inf"] for c in checks))
        dense = CertifiedDensePlan(matrix, contract="numerical")
        d, info = dense(rhs[:, 0])
        self.assertTrue(info["certificate"]["passed"])
        np.testing.assert_allclose(d, oracle[:, 0], atol=1e-8, rtol=0)
        with self.assertRaises(ValueError):
            CertifiedPathPlan(np.ones(n, dtype=int), off, contract="numerical")
        with self.assertRaises(ValueError):
            CertifiedDensePlan(matrix, contract="exact")

    def test_codec_all_words_special_bits_and_malformed_input(self):
        from lossless.retained.bf16_codec import encode, decode

        for start in (0, 120, 249):
            for offset in range(0, 65536, 64):
                words = list(range(offset, offset + 64))
                self.assertEqual(decode(encode(words, start)), words)
        with self.assertRaises(ValueError):
            decode(b"bad")
        with self.assertRaises(ValueError):
            decode(encode([0, 0x8000, 0x7FC1, 0xFFFF], 120) + b"extra")


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required")
class VisionTests(unittest.TestCase):
    def test_sign_reuse_gradient_inference_modes_and_eviction(self):
        import torch
        from torch.nn import functional as F
        from lossless.retained.vision import SiglipLoss

        operation = SiglipLoss(byte_limit=8192, capacity=2)
        for n, dtype in ((13, torch.float32), (17, torch.float64), (7, torch.float32)):
            with torch.inference_mode():
                operation(torch.ones(n, n, dtype=dtype))
            x = torch.randn(n, n, dtype=dtype).T.requires_grad_(True)
            expected = -F.logsigmoid((2 * torch.eye(n, dtype=dtype) - 1) * x).sum(-1).mean()
            actual = operation(x)
            self.assertTrue(torch.equal(expected, actual))
            (a,) = torch.autograd.grad(actual, x)
            (b,) = torch.autograd.grad(expected, x)
            self.assertTrue(torch.equal(a, b))
            self.assertLessEqual(operation.retained_bytes, 8192)


if __name__ == "__main__":
    unittest.main()
