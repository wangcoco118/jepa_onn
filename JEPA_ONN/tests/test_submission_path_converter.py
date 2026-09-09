import tempfile
import unittest
import zipfile
from pathlib import Path

from evals.intphys_test.convert_submission_paths import convert_submission_zip


class SubmissionPathConverterTests(unittest.TestCase):
    def test_converts_descriptive_paths_and_preserves_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.zip"
            target = root / "target.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr(
                    "answer.txt",
                    "O1/0001_O1_test_visible_static_nobj1/1 0.5504616899\n"
                    "O2/0042_O2_test_occluded_dynamic_1_nobj2/4 0.1234567890\n",
                )

            convert_submission_zip(source, target)

            with zipfile.ZipFile(target) as archive:
                self.assertEqual(archive.namelist(), ["answer.txt"])
                self.assertEqual(
                    archive.read("answer.txt").decode("utf-8"),
                    "O1/0001/1 0.5504616899\n"
                    "O2/0042/4 0.1234567890\n",
                )

    def test_rejects_colliding_converted_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.zip"
            target = root / "target.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr(
                    "answer.txt",
                    "O1/0001_O1_test_visible_static_nobj1/1 0.1\n"
                    "O1/0001_O1_other_name/1 0.2\n",
                )

            with self.assertRaises(ValueError):
                convert_submission_zip(source, target)


if __name__ == "__main__":
    unittest.main()
