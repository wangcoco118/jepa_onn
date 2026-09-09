"""Convert IntPhys submission paths to the canonical short-path format.

Only the movie path token is changed. Score tokens and line order are preserved.
For example, O1/0001_O1_test_visible_static_nobj1/1 becomes O1/0001/1.
"""

import argparse
import re
import zipfile
from pathlib import Path


_SHORT_PATH_RE = re.compile(r"^(O[123])/([0-9]{4})/([1-4])$")
_DESCRIPTIVE_PATH_RE = re.compile(r"^(O[123])/([0-9]{4})_[^/]+/([1-4])$")


def canonical_movie_path(path):
    """Return the official short form while preserving block, scene, and movie."""
    match = _SHORT_PATH_RE.fullmatch(path)
    if match:
        return path

    match = _DESCRIPTIVE_PATH_RE.fullmatch(path)
    if not match:
        raise ValueError(f"unsupported movie path: {path}")
    return f"{match.group(1)}/{match.group(2)}/{match.group(3)}"


def convert_answer_text(answer_text):
    """Convert paths in answer.txt and preserve score text and line order."""
    converted_lines = []
    seen_paths = set()

    for line_number, raw_line in enumerate(answer_text.splitlines(keepends=True), 1):
        body = raw_line.rstrip("\r\n")
        line_ending = raw_line[len(body):]
        fields = re.fullmatch(r"(\S+)([ \t]+)(\S+)([ \t]*)", body)
        if not fields:
            raise ValueError(f"invalid answer.txt line {line_number}: {body!r}")

        old_path, separator, score, trailing_space = fields.groups()
        new_path = canonical_movie_path(old_path)
        if new_path in seen_paths:
            raise ValueError(
                f"duplicate canonical movie path at line {line_number}: {new_path}"
            )
        seen_paths.add(new_path)
        converted_lines.append(
            f"{new_path}{separator}{score}{trailing_space}{line_ending}"
        )

    if not converted_lines:
        raise ValueError("answer.txt is empty")
    return "".join(converted_lines)


def convert_submission_zip(input_zip, output_zip):
    """Write a new ZIP containing only the converted answer.txt."""
    input_zip = Path(input_zip)
    output_zip = Path(output_zip)
    if not input_zip.is_file():
        raise FileNotFoundError(f"input ZIP does not exist: {input_zip}")
    if output_zip.exists():
        raise FileExistsError(f"refusing to overwrite output ZIP: {output_zip}")

    with zipfile.ZipFile(input_zip) as source:
        names = source.namelist()
        if names != ["answer.txt"]:
            raise ValueError(
                f"input ZIP must contain only answer.txt, got: {names}"
            )
        answer_text = source.read("answer.txt").decode("utf-8")

    converted = convert_answer_text(answer_text)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED
    ) as target:
        target.writestr("answer.txt", converted.encode("utf-8"))


def convert_submission_directory(input_dir, output_dir):
    """Convert average and maximum ZIPs into a new sibling directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")

    for mode in ("average", "maximum"):
        input_zip = input_dir / mode / "submission.zip"
        output_zip = output_dir / mode / "submission.zip"
        convert_submission_zip(input_zip, output_zip)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    convert_submission_directory(args.input_dir, args.output_dir)
    for mode in ("average", "maximum"):
        print(f"[{mode}] {args.output_dir / mode / 'submission.zip'}")


if __name__ == "__main__":
    main()

