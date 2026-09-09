import csv
import tempfile
import unittest
import zipfile
from pathlib import Path


from evals.intphys_test.build_submission import (
    build_answer_lines,
    resolve_output_dir,
    write_submission_zip,
)


class SubmissionBuilderTests(unittest.TestCase):
    def test_builds_average_and_maximum_lines_with_shared_random_o23(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "per_movie_scores.csv"
            task_path = root / "task.txt"
            task_path.write_text(
                "\n".join(
                    [
                        "O1/0001_O1_test_visible_static_nobj1/1",
                        "O1/0001_O1_test_visible_static_nobj1/2",
                        "O1/0001_O1_test_visible_static_nobj1/3",
                        "O1/0001_O1_test_visible_static_nobj1/4",
                        "O2/0001_O2_test_visible_static_nobj1/1",
                        "O3/0001_O3_test_visible_static_nobj1/1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "sample_id",
                        "plausibility_average",
                        "plausibility_maximum",
                    ],
                )
                writer.writeheader()
                for movie_id, average, maximum in (
                    ("1", "0.10", "0.20"),
                    ("2", "0.30", "0.40"),
                    ("3", "0.50", "0.60"),
                    ("4", "0.70", "0.80"),
                ):
                    writer.writerow(
                        {
                            "sample_id": f"O1/0001/{movie_id}",
                            "plausibility_average": average,
                            "plausibility_maximum": maximum,
                        }
                    )

            average_lines = build_answer_lines(csv_path, task_path, "average", 42)
            maximum_lines = build_answer_lines(csv_path, task_path, "maximum", 42)

            self.assertEqual(average_lines[:4], [
                "O1/0001/1 0.1000000000",
                "O1/0001/2 0.3000000000",
                "O1/0001/3 0.5000000000",
                "O1/0001/4 0.7000000000",
            ])
            self.assertEqual(maximum_lines[:4], [
                "O1/0001/1 0.2000000000",
                "O1/0001/2 0.4000000000",
                "O1/0001/3 0.6000000000",
                "O1/0001/4 0.8000000000",
            ])
            self.assertEqual(average_lines[4:], maximum_lines[4:])
            for line in average_lines[4:]:
                score = float(line.split()[1])
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_output_defaults_to_o1_model_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "model_run"
            run_dir.mkdir()
            csv_path = run_dir / "per_movie_scores.csv"

            self.assertEqual(
                resolve_output_dir(csv_path),
                run_dir / "submission_average_maximum",
            )

    def test_output_cannot_escape_o1_model_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "model_run"
            run_dir.mkdir()
            csv_path = run_dir / "per_movie_scores.csv"

            with self.assertRaises(ValueError):
                resolve_output_dir(csv_path, Path(tmp) / "outside")

    def test_submission_zip_contains_only_answer_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            answer_path = root / "answer.txt"
            zip_path = root / "submission.zip"
            answer_path.write_text("O1/0001/1 0.5\n", encoding="utf-8")

            write_submission_zip(answer_path, zip_path)

            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(archive.namelist(), ["answer.txt"])
                self.assertEqual(archive.read("answer.txt"), b"O1/0001/1 0.5\n")


if __name__ == "__main__":
    unittest.main()
