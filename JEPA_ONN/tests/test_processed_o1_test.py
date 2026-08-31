import tempfile
import unittest
import zipfile
from pathlib import Path

from evals.intphys_test.processed_test_support import (
    audit_manifest_records,
    build_task_mapping,
    create_submission_zip,
    plausibility_from_surprise,
)


class ProcessedO1TestSupportTests(unittest.TestCase):
    def test_manifest_requires_exact_four_movies_per_scene(self):
        records = [
            {"sample_id": "O1/0001/1", "path": "videos/a.pt"},
            {"sample_id": "O1/0001/2", "path": "videos/b.pt"},
            {"sample_id": "O1/0001/3", "path": "videos/c.pt"},
        ]
        audit = audit_manifest_records(records, data_root=Path("/unused"),
                                       expected_scene_count=1)
        self.assertFalse(audit["complete_o1"])
        self.assertEqual(audit["missing_count"], 1)
        self.assertEqual(audit["scene_count"], 1)

    def test_task_mapping_is_id_based_and_preserves_task_order(self):
        records = [
            {"sample_id": "O1/0001/2", "path": "videos/b.pt"},
            {"sample_id": "O1/0001/1", "path": "videos/a.pt"},
        ]
        mapping, audit = build_task_mapping(
            ["O1/0001/1", "O1/0001/2"], records
        )
        self.assertEqual(mapping, [("O1/0001/1", "videos/a.pt"),
                                    ("O1/0001/2", "videos/b.pt")])
        self.assertTrue(audit["path_mapping_valid"])

    def test_task_mapping_rejects_duplicate_task_paths(self):
        records = [{"sample_id": "O1/0001/1", "path": "videos/a.pt"}]
        mapping, audit = build_task_mapping(
            ["O1/0001/1", "O1/0001/1"], records
        )
        self.assertEqual(mapping, [])
        self.assertFalse(audit["path_mapping_valid"])
        self.assertEqual(audit["duplicate_count"], 1)

    def test_plausibility_is_bounded_and_decreases_with_surprise(self):
        high = plausibility_from_surprise(0.2)
        low = plausibility_from_surprise(1.2)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertGreater(high, low)

    def test_submission_zip_contains_only_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            answer = root / "answer.txt"
            answer.write_text("O1/0001/1 0.5\n", encoding="utf-8")
            archive = root / "submission_o1.zip"
            create_submission_zip(answer, archive)
            with zipfile.ZipFile(archive) as zf:
                self.assertEqual(zf.namelist(), ["answer.txt"])


if __name__ == "__main__":
    unittest.main()
