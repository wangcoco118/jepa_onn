import inspect
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
    def test_builds_lines_from_complete_real_o1_o2_o3_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            answer_path = root / "maximum_surprise_answer.txt"
            task_path = root / "task.txt"
            tasks = [
                "O1/0001_O1_test_visible_static_nobj1/1",
                "O1/0001_O1_test_visible_static_nobj1/2",
                "O2/0001_O2_test_visible_static_nobj1/1",
                "O2/0001_O2_test_visible_static_nobj1/2",
                "O3/0001_O3_test_visible_static_nobj1/1",
                "O3/0001_O3_test_visible_static_nobj1/2",
            ]
            task_path.write_text(
                "\n".join(tasks) + "\n",
                encoding="utf-8",
            )
            answer_path.write_text(
                "\n".join(
                    f"{task} {score}"
                    for task, score in zip(
                        reversed(tasks),
                        ("0.91", "0.81", "0.71", "0.61", "0.51", "0.41"),
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            lines = build_answer_lines(answer_path, task_path)

            self.assertEqual(lines, [
                "O1/0001/1 0.4100000000",
                "O1/0001/2 0.5100000000",
                "O2/0001/1 0.6100000000",
                "O2/0001/2 0.7100000000",
                "O3/0001/1 0.8100000000",
                "O3/0001/2 0.9100000000",
            ])

    def test_missing_real_o2_o3_scores_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            answer_path = root / "average_surprise_answer.txt"
            task_path = root / "task.txt"
            task_path.write_text(
                "\n".join([
                    "O1/0001_O1_test_visible_static_nobj1/1",
                    "O2/0001_O2_test_visible_static_nobj1/1",
                    "O3/0001_O3_test_visible_static_nobj1/1",
                ])
                + "\n",
                encoding="utf-8",
            )
            answer_path.write_text(
                "O1/0001_O1_test_visible_static_nobj1/1 0.5\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "missing real scores.*O2/0001/1"
            ):
                build_answer_lines(answer_path, task_path)

    def test_build_answer_lines_has_no_random_seed_parameter(self):
        self.assertEqual(
            list(inspect.signature(build_answer_lines).parameters),
            ["answer_path", "task_path"],
        )

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
