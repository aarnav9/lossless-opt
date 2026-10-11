"""The checklist cannot turn advisory claims or missing counters into acceptance."""

import copy
import json
import unittest
from unittest.mock import patch

from lossless._native import hardware
from lossless.applications import architecture, cpu_catalog, patches
from lossless.applications.profile_worker import input_layouts, runtime_metadata


class CPUChecklistTests(unittest.TestCase):
    def setUp(self):
        self.cat = architecture.catalog()
        self.hardware = {"host": {"machine": "arm64", "cpu_details": {}}}
        self.coverage = architecture.initial_coverage(self.cat, self.hardware)

    def note(self, **changes):
        return {
            "id": "allocation",
            "assessment": "hypothesis",
            "reason": "Inspect measured allocations",
            "evidence_ids": ["sample:memory"],
            "experiment": True,
            **changes,
        }

    def test_catalog_links_and_sources_are_complete_and_identity_changes(self):
        ids = {c["id"] for c in self.cat["cards"]}
        self.assertEqual(len(ids), len(self.cat["cards"]))
        for card in self.cat["cards"]:
            self.assertLessEqual(set(card["interactions"]), ids)
            self.assertLessEqual(set(card["sources"]), self.cat["sources"].keys())
            for key in ("checks", "evidence_needed", "obligations", "validation"):
                self.assertTrue(card[key])
        changed = copy.deepcopy(cpu_catalog.CARDS)
        changed[0]["checks"] += " changed"
        with patch.object(cpu_catalog, "CARDS", changed):
            self.assertNotEqual(self.cat["identity"], architecture.catalog()["identity"])

    def test_unsupported_and_unknown_hardware_are_distinct(self):
        entries = {e["id"]: e for e in self.coverage["entries"]}
        self.assertEqual(entries["x86_model"]["applicability"], "not_applicable")
        self.assertEqual(entries["arm_model"]["applicability"], "applicable")
        self.assertEqual(entries["numa"]["applicability"], "unknown")
        unknown = architecture.initial_coverage(self.cat, {"host": {}})
        self.assertEqual(
            next(e for e in unknown["entries"] if e["id"] == "x86_model")["status"], "unknown"
        )

    def test_cpu_only_probe_never_queries_an_accelerator(self):
        with (
            patch.object(hardware, "metal_probe", side_effect=AssertionError("GPU query")),
            patch.object(hardware, "cuda_probe", side_effect=AssertionError("GPU query")),
            patch.object(hardware, "command", return_value=(None, "unavailable")),
            patch.object(hardware.shutil, "which", return_value=None),
        ):
            report = hardware.collect(cpu_only=True)
        self.assertTrue(all(v is None for v in report["measurements"].values()))
        self.assertEqual(report["collection_mode"], "cpu_queries")

    def test_review_cannot_claim_proof_acceptance_or_fake_evidence(self):
        for change in (
            {"assessment": "proved"},
            {"assessment": "accepted"},
            {"evidence_ids": ["invented"]},
            {"evidence_ids": []},
            {"id": "imaginary"},
            {"experiment": 1},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                architecture.validate_notes([self.note(**change)], evidence_ids=["sample:memory"])
        with self.assertRaises(ValueError):
            architecture.validate_notes([self.note(), self.note()])
        with self.assertRaises(ValueError):
            architecture.validate_notes([self.note()], required=["work"])

    def test_reviews_and_duplicates_never_count_as_evaluated(self):
        note = self.note()
        architecture.record_review(self.coverage, [note], "candidate_01")
        entry = next(e for e in self.coverage["entries"] if e["id"] == "allocation")
        self.assertEqual(entry["status"], "reviewed_unverified")
        architecture.record_result(
            self.coverage, [note], {"id": "candidate_01", "status": "duplicate"}
        )
        self.assertEqual(entry["status"], "attempted_not_evaluated")
        architecture.record_result(
            self.coverage, [note], {"id": "candidate_02", "status": "mismatch"}
        )
        self.assertEqual(entry["status"], "evaluated")
        self.assertEqual(entry["proof_status"], "not_checked")
        architecture.finish(self.coverage, "discovery_deadline")
        self.assertIn("cache", self.coverage["untested_ids"])
        self.assertNotIn("allocation", self.coverage["untested_ids"])

    def test_abstention_cannot_claim_experiment(self):
        with self.assertRaises(ValueError):
            patches.validate(
                {
                    "schema_version": 1,
                    "hypothesis": "stop",
                    "edits": [],
                    "checklist": [self.note()],
                },
                [],
            )

    def test_author_history_keeps_recent_and_winner_without_repeating_raw_timings(self):
        for i in range(30):
            candidate = f"candidate_{i:02d}"
            architecture.record_review(self.coverage, [self.note()], candidate)
            architecture.record_result(
                self.coverage,
                [self.note()],
                {
                    "id": candidate,
                    "status": "matched",
                    "discovery": {
                        "statistics": {
                            "case_balanced_geomean_speedup": 1.1,
                            "cases": [{"speedup": 1.05}],
                            "raw_samples": [123456789] * 1000,
                        }
                    },
                },
            )
        compact = architecture.context_coverage(self.coverage, "candidate_03")
        row = next(e for e in compact["entries"] if e["id"] == "allocation")
        self.assertEqual(
            [e["candidate"] for e in row["experiments"]],
            ["candidate_03", "candidate_28", "candidate_29"],
        )
        self.assertEqual(row["omitted_attempts"], 27)
        self.assertEqual(row["total_reviews"], 30)
        self.assertEqual(row["experiments"][0]["minimum_case_speedup"], 1.05)
        self.assertNotIn("raw_samples", json.dumps(compact))
        original = next(e for e in self.coverage["entries"] if e["id"] == "allocation")
        self.assertEqual(len(original["experiments"]), 30)
        self.assertIn("raw_samples", json.dumps(original))
        self.assertLess(len(json.dumps(compact)), len(json.dumps(self.coverage)) / 10)

    def test_layout_collector_is_bounded_and_does_not_expose_values(self):
        import numpy as np

        array = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::-2]
        result = input_layouts([array, array])
        self.assertEqual(len(result["arrays"]), 1)
        self.assertEqual(result["arrays"][0]["strides_bytes"], [24, -8])
        self.assertFalse(result["arrays"][0]["c_contiguous"])
        cycle = []
        cycle.append(cycle)
        self.assertEqual(input_layouts(cycle)["arrays"], [])
        self.assertLessEqual(len(input_layouts([array.copy() for _ in range(100)])["arrays"]), 32)
        self.assertNotIn("data", result["arrays"][0])

    def test_runtime_metadata_does_not_expose_library_paths(self):
        result = runtime_metadata()
        self.assertNotIn("filepath", json.dumps(result))
        self.assertIn("loaded_packages", result)


if __name__ == "__main__":
    unittest.main()
