"""Package complete real IntPhys Test predictions in official format.

Both input answer files must come from real model inference and must cover every
O1, O2, and O3 task exactly once. Missing or extra scores are rejected; this
utility never synthesizes placeholder scores.
"""

import argparse
from datetime import datetime
import math
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from evals.intphys_test.convert_submission_paths import canonical_movie_path


def _read_tasks(task_path):
    return [
        line.strip()
        for line in Path(task_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_real_scores(answer_path):
    scores = {}
    with Path(answer_path).open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.split()
            if len(fields) != 2:
                raise ValueError(
                    f"invalid answer line {line_number}: expected path and score"
                )
            movie_path = canonical_movie_path(fields[0])
            if movie_path in scores:
                raise ValueError(
                    f"duplicate score at line {line_number}: {movie_path}"
                )
            try:
                score = float(fields[1])
            except ValueError as exc:
                raise ValueError(
                    f"invalid score at line {line_number}: {fields[1]!r}"
                ) from exc
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"score outside [0, 1] at line {line_number}: {movie_path}"
                )
            scores[movie_path] = score
    if not scores:
        raise ValueError(f"no real scores found in {answer_path}")
    return scores


def build_answer_lines(answer_path, task_path):
    scores = _read_real_scores(answer_path)
    tasks = _read_tasks(task_path)
    canonical_tasks = [canonical_movie_path(task) for task in tasks]

    if len(set(canonical_tasks)) != len(canonical_tasks):
        raise ValueError("task file contains duplicate canonical movie paths")

    task_set = set(canonical_tasks)
    missing = [task for task in canonical_tasks if task not in scores]
    if missing:
        raise ValueError(
            f"missing real scores for {len(missing)} tasks; first: {missing[0]}"
        )

    extra = sorted(set(scores).difference(task_set))
    if extra:
        raise ValueError(
            f"scores contain {len(extra)} tasks absent from task file; "
            f"first: {extra[0]}"
        )

    return [f"{task} {scores[task]:.10f}" for task in canonical_tasks]


def resolve_output_dir(
    answer_path,
    output_dir=None,
    output_name="submission_average_maximum",
):
    run_dir = Path(answer_path).resolve().parent
    if output_dir is None:
        if not output_name or Path(output_name).name != output_name:
            raise ValueError("output_name must be a single directory name")
        resolved = run_dir / output_name
    else:
        resolved = Path(output_dir).resolve()
        if resolved.parent != run_dir:
            raise ValueError(
                "output_dir must be a direct child of the model run directory: "
                f"{run_dir}"
            )
    return resolved


def _infer_model_name(answer_path):
    resolved = Path(answer_path).resolve()
    for parent in resolved.parents:
        if parent.name.startswith(("onn_", "o1_", "transformer_")):
            return parent.name
    return resolved.parent.name


def _submission_zip_name(answer_path, aggregation, generated_at):
    if aggregation not in {"average", "maximum"}:
        raise ValueError(f"unsupported aggregation: {aggregation}")
    model_name = _infer_model_name(answer_path)
    timestamp = generated_at.strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{timestamp}_{aggregation}.zip"


def write_submission_zip(answer_path, zip_path):
    answer_path = Path(answer_path)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
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
    parser.add_argument(
        "--average-answer",
        required=True,
        type=Path,
        help="real average_surprise_answer.txt from raw Test inference",
    )
    parser.add_argument(
        "--maximum-answer",
        required=True,
        type=Path,
        help="real maximum_surprise_answer.txt from raw Test inference",
    )
    parser.add_argument("--task-file", required=True, type=Path)
    parser.add_argument("--validator", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="direct child of the model run directory",
    )
    parser.add_argument(
        "--output-name",
        default="submission_average_maximum",
        help="output directory name under the model run directory",
    )
    args = parser.parse_args(argv)

    for source, label in (
        (args.average_answer, "average real answer"),
        (args.maximum_answer, "maximum real answer"),
        (args.task_file, "task file"),
        (args.validator, "validator"),
    ):
        if not source.is_file():
            raise FileNotFoundError(f"{label} does not exist: {source}")

    average_parent = args.average_answer.resolve().parent
    maximum_parent = args.maximum_answer.resolve().parent
    if average_parent != maximum_parent:
        raise ValueError(
            "average and maximum answer files must belong to the same model run"
        )

    output_dir = resolve_output_dir(
        args.average_answer,
        args.output_dir,
        args.output_name,
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; choose a new path: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_tasks = [
        canonical_movie_path(task)
        for task in _read_tasks(args.task_file)
    ]
    if len(set(canonical_tasks)) != len(canonical_tasks):
        raise ValueError("task file contains duplicate canonical movie paths")

    failed = False
    generated_at = datetime.now()
    sources = {
        "average": args.average_answer,
        "maximum": args.maximum_answer,
    }
    with tempfile.TemporaryDirectory(prefix="intphys_task_") as temp_dir:
        validation_task = Path(temp_dir) / "task.txt"
        validation_task.write_text(
            "\n".join(canonical_tasks) + "\n",
            encoding="utf-8",
        )
        for aggregation, source_answer in sources.items():
            mode_dir = output_dir / aggregation
            mode_dir.mkdir()
            answer_path = mode_dir / "answer.txt"
            zip_path = mode_dir / _submission_zip_name(
                source_answer, aggregation, generated_at
            )
            lines = build_answer_lines(source_answer, args.task_file)
            answer_path.write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )
            write_submission_zip(answer_path, zip_path)
            print(f"[{aggregation}] answer: {answer_path}")
            print(f"[{aggregation}] zip: {zip_path}")
            print(f"[{aggregation}] validator:")
            if not run_official_validator(
                args.validator, zip_path, validation_task
            ):
                failed = True
                print(
                    f"[{aggregation}] validation failed",
                    file=sys.stderr,
                )
            else:
                print(f"[{aggregation}] validation passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
