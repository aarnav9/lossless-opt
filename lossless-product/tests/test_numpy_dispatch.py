"""Exact observations and evaluation order for broader automatic CPU recipes."""

import ast
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lossless.applications.automatic import numpy_candidates
from lossless.applications.discovery import inventory
from lossless.applications.numpy_dispatch import ExtendedDispatch
from lossless.applications.patches import scope_sources
from lossless.applications.worker import Observer
from lossless.jobs import identity


def transformed(source):
    tree = ast.parse(source)
    transform = ExtendedDispatch({"np"})
    tree = transform.visit(tree)
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), "<candidate>", "exec"), namespace)
    return namespace, transform.changed


def observed(value):
    return identity(Observer(8 * 2**20).encode(value))


class NumpyDispatchTests(unittest.TestCase):
    def test_exact_dtype_layout_axis_and_nonfinite_observations(self):
        expressions = [
            "np.sum(a, axis=0, keepdims=True)",
            "np.prod(a, -1, dtype=np.complex128)",
            "np.mean(a, axis=(0, 1), keepdims=True)",
            "np.min(a, axis=1, initial=0)",
            "np.max(a, axis=None, initial=2)",
            "np.amin(a)",
            "np.amax(a, axis=0)",
            "np.any(a, axis=1)",
            "np.all(a, axis=0, keepdims=True)",
            "np.argmin(a, axis=-1, keepdims=True)",
            "np.argmax(a)",
            "np.cumsum(a, axis=0)",
            "np.cumprod(a, axis=1)",
            "np.std(a, axis=0, ddof=1)",
            "np.var(a, axis=1, keepdims=True)",
            "np.sum(axis=0, a=a, where=a != 0, initial=3)",
        ]
        for expression in expressions:
            source = f"import numpy as np\ndef run(a):\n    return {expression}\n"
            candidate, changed = transformed(source)
            self.assertEqual(changed, 1, expression)
            reference = {}
            exec(source, reference)
            for dtype in (
                np.bool_,
                np.int8,
                np.uint32,
                np.float16,
                np.float32,
                np.float64,
                np.complex64,
                np.complex128,
            ):
                base = np.arange(12).reshape(3, 4).astype(dtype)
                arrays = [base, np.asfortranarray(base), base[:, ::-1], base.T]
                if np.issubdtype(dtype, np.inexact):
                    unusual = base.copy()
                    unusual.flat[:4] = [-0.0, np.nan, np.inf, -np.inf]
                    arrays.append(unusual)
                for a in arrays:
                    a.flags.writeable = False
                    with self.subTest(expression=expression, dtype=dtype, strides=a.strides):
                        with np.errstate(all="ignore"):
                            self.assertEqual(
                                observed(candidate["run"](a)), observed(reference["run"](a))
                            )

    def test_output_buffer_alias_and_mutation_preserved(self):
        for expression, shape, dtype in (
            ("np.sum(a, axis=0, out=out)", (4,), float),
            ("np.argmax(a, axis=0, out=out)", (4,), np.intp),
            ("np.cumsum(a, axis=1, out=out)", (3, 4), float),
        ):
            source = f"import numpy as np\ndef run(a, out):\n    return {expression}\n"
            candidate, _ = transformed(source)
            namespace = {}
            exec(source, namespace)
            a = np.arange(12.0).reshape(3, 4)
            expected, actual = np.empty(shape, dtype), np.empty(shape, dtype)
            self.assertIs(namespace["run"](a, expected), expected)
            self.assertIs(candidate["run"](a, actual), actual)
            self.assertEqual(observed(actual), observed(expected))
        a, b = np.arange(12.0).reshape(3, 4), np.arange(12.0).reshape(3, 4)
        self.assertIs(candidate["run"](a, a), a)
        namespace["run"](b, b)
        self.assertEqual(observed(a), observed(b))

    def test_subclass_object_and_list_use_original_dispatch(self):
        class Special(np.ndarray):
            def __array_function__(self, func, types, args, kwargs):
                return (func.__name__, tuple(kwargs))

        for operation in ("sum", "mean", "std", "argmax", "cumsum"):
            source = f"import numpy as np\ndef run(a):\n    return np.{operation}(a, axis=0)\n"
            candidate, _ = transformed(source)
            reference = {}
            exec(source, reference)
            for a in ([1, 2], np.array([1, 2], dtype=object), np.arange(3).view(Special)):
                actual, expected = candidate["run"](a), reference["run"](a)
                if isinstance(actual, np.ndarray) and actual.dtype.hasobject:
                    self.assertIs(type(actual), type(expected))
                    np.testing.assert_array_equal(actual, expected)
                else:
                    self.assertEqual(observed(actual), observed(expected))

    def test_arguments_evaluated_once_in_order_including_keyword_array(self):
        source = """import numpy as np
events = []
def value(name, x):
    events.append(name)
    return x
def run(a):
    return np.sum(axis=value('axis', 0), a=value('a', a), initial=value('initial', 2))
"""
        candidate, _ = transformed(source)
        reference = {}
        exec(source, reference)
        self.assertEqual(
            observed(candidate["run"](np.arange(5))), observed(reference["run"](np.arange(5)))
        )
        self.assertEqual(candidate["events"], ["axis", "a", "initial"])
        self.assertEqual(candidate["events"], reference["events"])

    def test_original_function_captured_before_argument_evaluation(self):
        # A list takes fallback, so replacing the module during argument
        # evaluation must not change which original callable receives it.
        source = """from types import SimpleNamespace
import numpy as np
def argument():
    global np
    np = SimpleNamespace(ndarray=np.ndarray, sum=lambda *a, **k: -999)
    return [1, 2]
def run():
    return np.sum(argument(), axis=0)
"""
        candidate, _ = transformed(source)
        self.assertEqual(candidate["run"](), 3)

    def test_unknown_extensions_and_unpacking_are_untouched(self):
        for expression in (
            "np.sum(*args)",
            "np.sum(a, **options)",
            "np.std(a, mean=m)",
            "np.var(a, correction=1)",
            "np.sum(a, a=a)",
            "np.sum(a)",
        ):
            tree = ast.parse(expression, mode="eval")
            expected = ast.dump(tree)
            transform = ExtendedDispatch({"np"})
            actual = transform.visit(copy.deepcopy(tree))
            self.assertEqual(transform.changed, 0)
            self.assertEqual(ast.dump(actual), expected)

    def test_invalid_axis_raises_original_exception_type(self):
        source = "import numpy as np\ndef run(a):\n    return np.sum(a, axis=7)\n"
        candidate, _ = transformed(source)
        reference = {}
        exec(source, reference)
        with self.assertRaises(Exception) as expected:
            reference["run"](np.arange(3))
        with self.assertRaises(type(expected.exception)):
            candidate["run"](np.arange(3))

    def candidates(self, source, rows, *, expanded=True):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "app.py").write_text(source)
            profile = {
                "profiles": [
                    {
                        "attribution": {
                            "top_cumulative": [
                                dict(file="app.py", origin="project", cumulative_seconds=1, **row)
                                for row in rows
                            ]
                        }
                    }
                ]
            }
            result = numpy_candidates(root, inventory(root), profile, expanded=expanded)
            if result[0]:
                scope_sources(root, result[0])
            return result

    def test_profiled_class_methods_use_lines_to_disambiguate(self):
        source = """import numpy as np
class A:
    def run(self, a):
        return np.sum(a, axis=0)
class B:
    def run(self, a):
        return np.cumsum(a)
"""
        scope, candidates = self.candidates(source, [dict(function="run", line=6)])
        self.assertEqual(scope, [dict(path="app.py", symbol="B.run")])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(self.candidates(source, [dict(function="run")]), ([], []))
        self.assertEqual(self.candidates(source, [dict(function="run", line=99)]), ([], []))

    def test_old_candidates_remain_and_new_candidates_compose(self):
        source = """import numpy as np
def run(a):
    return np.sum(a), np.copy(a), np.cumsum(a)
"""
        _, candidates = self.candidates(source, [dict(function="run")])
        self.assertEqual(len(candidates), 5)
        self.assertIn("np.cumsum(a)", candidates[2]["edits"][0]["body"])
        self.assertIn("a.copy(order='K')", candidates[-1]["edits"][0]["body"])
        self.assertIn(".cumsum()", candidates[-1]["edits"][0]["body"])

    def test_decorated_and_shadowed_scopes_are_excluded(self):
        for source in (
            "import numpy as np\n@decorator\ndef run(a):\n    return np.cumsum(a)\n",
            "import numpy as np\ndef run(a):\n    try:\n        return np.cumsum(a)\n    except Exception as np:\n        raise\n",
            "import numpy as np\n@decorator\nclass A:\n    def run(self, a):\n        return np.cumsum(a)\n",
            "import numpy as np\ndef outer():\n    def run(a):\n        return np.cumsum(a)\n    return run\n",
        ):
            self.assertEqual(self.candidates(source, [dict(function="run")]), ([], []))

    def test_expanded_recipes_are_opt_in(self):
        source = "import numpy as np\ndef run(a):\n    return np.sum(a, axis=0)\n"
        self.assertEqual(self.candidates(source, [dict(function="run")], expanded=False), ([], []))
        self.assertEqual(len(self.candidates(source, [dict(function="run")])[1]), 1)
        source = "import numpy as np\nclass A:\n    def run(self, a):\n        return np.sum(a)\n"
        self.assertEqual(self.candidates(source, [dict(function="run")], expanded=False), ([], []))
        self.assertEqual(len(self.candidates(source, [dict(function="run")])[1]), 1)


if __name__ == "__main__":
    unittest.main()
