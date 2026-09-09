"""Build official-format IntPhys submissions from processed O1 scores.

The O1 values come from an existing per_movie_scores.csv. O2 and O3 are
filled with deterministic random values in [0, 1] because this utility is
intended for an O1-focused submission experiment. The official validator is
run separately for the average and maximum submissions.
"""

import argparse
import csv
from datetime import datetime
import math
import random
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from evals.intphys_test.convert_submission_paths import canonical_movie_path


_SCORE_COLUMNS = {
    "average": "plausibility_average",
    "maximum": "plausibility_maximum",
}


def _read_o1_scores(csv_path, score_column):
    required = {"sample_id", score_column}
    scores = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or []))
            raise ValueError(f"CSV is missing required columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            sample_id = (row.get("sample_id") or "").strip()
            if not sample_id.startswith("O1/"):
                continue
            if sample_id in scores:
                raise ValueError(f"duplicate O1 sample_id at row {row_number}: {sample_id}")
            try:
                score = float(row[score_column])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid {score_column} at row {row_number}: {sample_id}"
                ) from exc
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"score outside [0, 1] at row {row_number}: {sample_id}")
            scores[sample_id] = score
    if not scores:
        raise ValueError(f"no O1 scores found in {csv_path}")
    return scores


def _read_tasks(task_path):
    return [
        line.strip()
        for line in Path(task_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sample_id_from_official_task(task):
    parts = task.split("/")
    if len(parts) != 3 or parts[0] != "O1" or parts[2] not in {"1", "2", "3", "4"}:
        raise ValueError(f"invalid O1 official task path: {task}")
    scene_id = parts[1].split("_", 1)[0]
    if len(scene_id) != 4 or not scene_id.isdigit():
        raise ValueError(f"cannot map O1 scene id from task path: {task}")
    return f"O1/{scene_id}/{parts[2]}"


def build_answer_lines(csv_path, task_path, aggregation, seed):
    if aggregation not in _SCORE_COLUMNS:
        raise ValueError(f"unsupported aggregation: {aggregation}")
    o1_scores = _read_o1_scores(csv_path, _SCORE_COLUMNS[aggregation])
    tasks = _read_tasks(task_path)
    rng = random.Random(seed)
    lines = []

    for task in tasks:
        official_task = canonical_movie_path(task)
        block = official_task.split("/", 1)[0]
        if block == "O1":
            sample_id = _sample_id_from_official_task(task)
            if sample_id not in o1_scores:
                raise ValueError(
                    f"missing {aggregation} O1 score for {task} -> {sample_id}"
                )
            score = o1_scores[sample_id]
        elif block in {"O2", "O3"}:
            score = rng.uniform(0.0, 1.0)
        else:
            raise ValueError(f"unsupported block in task file: {task}")
        lines.append(f"{official_task} {score:.10f}")

    return lines


def resolve_output_dir(csv_path, output_dir=None, output_name="submission_average_maximum"):
    run_dir = Path(csv_path).resolve().parent
    if output_dir is None:
        if not output_name or Path(output_name).name != output_name:
            raise ValueError("output_name must be a single directory name")
        resolved = run_dir / output_name
    else:
        resolved = Path(output_dir).resolve()
        if resolved.parent != run_dir:
            raise ValueError(
                "output_dir must be a direct child of the O1 model run directory: "
                f"{run_dir}"
            )
    return resolved


def _infer_model_name(csv_path):
    resolved = Path(csv_path).resolve()
    for parent in resolved.parents:
        if parent.name.startswith(('onn_', 'o1_')):
            return parent.name
    return resolved.parent.name


def _submission_zip_name(csv_path, aggregation, generated_at):
    if aggregation not in _SCORE_COLUMNS:
        raise ValueError(f"unsupported aggregation: {aggregation}")
    model_name = _infer_model_name(csv_path)
    timestamp = generated_at.strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{timestamp}_{aggregation}.zip"


def write_submission_zip(answer_path, zip_path):
    answer_path = Path(answer_path)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(answer_path, arcname="answer.txt")


def run_official_validator(validator_path, zip_path, task_path):
    command = [
        sys.executable,
        str(validator_path),
        str(zip_path),
        str(task_path),
    ]
    completed = subprocess.run(
        command,
        cwd=str(Path(validator_path).parent),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return completed.returncode == 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--o1-csv", required=True, type=Path)
    parser.add_argument("--task-file", required=True, type=Path)
    parser.add_argument("--validator", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="direct child of the O1 model run directory; defaults to "
        "<O1 CSV directory>/submission_average_maximum",
    )
    parser.add_argument(
        "--output-name",
        default="submission_average_maximum",
        help="default output directory name under the O1 model run directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    for path, label in (
        (args.o1_csv, "O1 CSV"),
        (args.task_file, "task file"),
        (args.validator, "validator"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    output_dir = resolve_output_dir(
        args.o1_csv,
        args.output_dir,
        args.output_name,
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; choose a new path: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_tasks = [canonical_movie_path(task) for task in _read_tasks(args.task_file)]
    if len(set(canonical_tasks)) != len(canonical_tasks):
        raise ValueError("task file contains duplicate canonical movie paths")

    failed = False
    generated_at = datetime.now()
    with tempfile.TemporaryDirectory(prefix="intphys_task_") as temp_dir:
        validation_task = Path(temp_dir) / "task.txt"
        validation_task.write_text(
            "\n".join(canonical_tasks) + "\n",
            encoding="utf-8",
        )
        for aggregation in ("average", "maximum"):
            mode_dir = output_dir / aggregation
            mode_dir.mkdir()
            answer_path = mode_dir / "answer.txt"
            zip_path = mode_dir / _submission_zip_name(
                args.o1_csv, aggregation, generated_at
            )
            lines = build_answer_lines(
                args.o1_csv, args.task_file, aggregation, args.seed
            )
            answer_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            write_submission_zip(answer_path, zip_path)
            print(f"[{aggregation}] answer: {answer_path}")
            print(f"[{aggregation}] zip: {zip_path}")
            print(f"[{aggregation}] validator:")
            if not run_official_validator(args.validator, zip_path, validation_task):
                failed = True
                print(f"[{aggregation}] validation failed", file=sys.stderr)
            else:
                print(f"[{aggregation}] validation passed")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
