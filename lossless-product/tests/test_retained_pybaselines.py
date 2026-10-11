"""Prepared metadata and complete composed-application conformance.

Set LOSSLESS_PYBASELINES_PROJECT to the pinned checkout for the source tests.
The user's checkout is copied, never patched in place.
"""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

PROJECT = os.environ.get("LOSSLESS_PYBASELINES_PROJECT")
if PROJECT:
    sys.path.insert(0, str(Path(PROJECT).resolve()))


@unittest.skipUnless(importlib.util.find_spec("pybaselines"), "pybaselines required")
class PreparedTests(unittest.TestCase):
    def test_prepared_changing_signals_weights_orders_and_owned_grids(self):
        from pybaselines import Baseline
        from lossless.retained.pybaselines import ModPolyPlan

        rng = np.random.default_rng(927711)
        for order in (0, 1, 4):
            for dtype in (np.float32, np.float64):
                x = np.linspace(-1, 1, 137)
                x = x[np.random.default_rng(332).permutation(x.size)]
                original = x.copy()
                plan = ModPolyPlan(x, order)
                x[:] = 17
                saved = []
                for weighted in (False, True, False):
                    y = (1 + original**2 + rng.normal(0, 0.1, x.size)).astype(dtype)
                    options = dict(return_coef=True, max_iter=19, use_original=True)
                    if weighted:
                        options["weights"] = rng.uniform(0.15, 1, x.size)
                    expected = Baseline(x_data=original).modpoly(y, poly_order=order, **options)
                    actual = plan(y, **options)
                    self.assertEqual(actual[0].tobytes(), expected[0].tobytes())
                    self.assertEqual(set(actual[1]), set(expected[1]))
                    for key in actual[1]:
                        self.assertEqual(actual[1][key].tobytes(), expected[1][key].tobytes())
                    saved.append((actual[0], actual[0].tobytes()))
                self.assertTrue(all(a.tobytes() == b for a, b in saved))
                with self.assertRaises(ValueError):
                    plan(y, poly_order=order + 1)

    def test_cache_mutation_eviction_and_lazy_growth(self):
        from pybaselines.polynomial import modpoly
        from lossless.retained.pybaselines import ModPolyCache

        cache = ModPolyCache(capacity=2, byte_limit=4096)
        rng = np.random.default_rng(72915)
        x = np.linspace(0, 1, 127)
        for offset in (0.0, 1.0, 2.0, 0.0):
            x[:] = np.linspace(offset, offset + 1, x.size)
            y = rng.normal(size=x.size)
            expected = modpoly(y, x_data=x, poly_order=3, max_iter=10)
            actual = cache(x, y, poly_order=3, max_iter=10)
            self.assertEqual(actual[0].tobytes(), expected[0].tobytes())
            self.assertLessEqual(cache.plans.retained_bytes, 4096)
            self.assertLessEqual(len(cache.plans.entries), 2)


@unittest.skipUnless(PROJECT, "set LOSSLESS_PYBASELINES_PROJECT for source integration")
class CompositionTests(unittest.TestCase):
    def test_recognized_compositions_hash_admission_and_no_shared_edits(self):
        import lossless
        from lossless.applications import retained_plan
        from lossless.applications.search import resolve_plan

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(Path(PROJECT) / "pybaselines", root / "pybaselines")
            shutil.copyfile(
                Path(lossless.__file__).parent / "retained/spectrum_workload.py",
                root / "spectrum_workload.py",
            )
            plan = retained_plan(root)
            resolve_plan(plan, None)
            self.assertEqual(len(plan["candidates"]), 8)
            for candidate in plan["candidates"]:
                for e in candidate["edits"]:
                    if "_lossless_native_modpoly" in e["body"]:
                        self.assertTrue(candidate.get("native_sources"))
            serialized = json.dumps(plan)
            plan["candidates"][3]["edits"][-1]["body"] = "raise RuntimeError('mutated')"
            self.assertEqual(json.dumps(retained_plan(root)), serialized)
            with (root / "pybaselines/polynomial.py").open("a") as stream:
                stream.write("\n# changed revision\n")
            with self.assertRaises(ValueError):
                retained_plan(root)


if __name__ == "__main__":
    unittest.main()
