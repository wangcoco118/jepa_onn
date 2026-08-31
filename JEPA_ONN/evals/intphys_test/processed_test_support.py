"""Pure-Python validation and output helpers for processed O1 inference."""

import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


SAMPLE_ID_RE = re.compile(r"^(?P<block>O1)/(?P<scene>[0-9]{4})/(?P<movie>[1-4])$")
OFFICIAL_PATH_RE = re.compile(
    r"(?:^|/)(?P<block>O[1-3])/(?P<scene>[0-9]{4})/(?P<movie>[1-4])(?:\.[^/\s]+)?$"
)


def parse_sample_id(sample_id):
    match = SAMPLE_ID_RE.fullmatch(str(sample_id or ""))
    if match is None:
        return None
    return match.group("block"), int(match.group("scene")), int(match.group("movie"))


def parse_official_movie_path(path):
    token = str(path or "").strip().replace("\\", "/")
    match = OFFICIAL_PATH_RE.search(token)
    if match is None:
        return None
    return (
        match.group("block"),
        int(match.group("scene")),
        int(match.group("movie")),
    )


def is_safe_relative_path(path):
    value = str(path or "")
    candidate = Path(value)
    return bool(value) and not candidate.is_absolute() and ".." not in candidate.parts


def expected_sample_ids(expected_scene_count=1080):
    return {
        f"O1/{scene:04d}/{movie}"
        for scene in range(1, int(expected_scene_count) + 1)
        for movie in range(1, 5)
    }


def audit_manifest_records(records, data_root, expected_scene_count=1080):
    """Audit manifest rows without relying on directory iteration order."""
    records = list(records)
    expected = expected_sample_ids(expected_scene_count)
    ids = []
    paths = []
    invalid_count = 0
    missing_file_count = 0
    valid_by_id = {}
    path_to_ids = defaultdict(list)

    for record in records:
        if not isinstance(record, dict):
            invalid_count += 1
            continue
        sample_id = str(record.get("sample_id", ""))
        relative_path = str(record.get("path", ""))
        parsed = parse_sample_id(sample_id)
        valid = (
            parsed is not None
            and parsed[0] == "O1"
            and 1 <= parsed[1] <= int(expected_scene_count)
            and is_safe_relative_path(relative_path)
            and record.get("num_frames", 100) == 100
            and record.get("image_size", 128) == 128
        )
        if not valid:
            invalid_count += 1
            continue
        ids.append(sample_id)
        paths.append(relative_path)
        path_to_ids[relative_path].append(sample_id)
        if sample_id not in valid_by_id:
            valid_by_id[sample_id] = record
        if not (Path(data_root) / relative_path).is_file():
            missing_file_count += 1

    id_counts = Counter(ids)
    path_counts = Counter(paths)
    duplicate_id_count = sum(count - 1 for count in id_counts.values() if count > 1)
    duplicate_path_count = sum(count - 1 for count in path_counts.values() if count > 1)
    duplicate_count = max(duplicate_id_count, duplicate_path_count)
    missing_ids = sorted(expected.difference(valid_by_id))
    scene_ids = {
        parse_sample_id(sample_id)[1]
        for sample_id in valid_by_id
        if parse_sample_id(sample_id) is not None
    }
    scene_movie_counts = Counter(
        parse_sample_id(sample_id)[1]
        for sample_id in valid_by_id
        if parse_sample_id(sample_id) is not None
    )
    invalid_scene_counts = {
        str(scene): count
        for scene, count in sorted(scene_movie_counts.items())
        if count != 4
    }
    complete = (
        len(records) == len(expected)
        and len(valid_by_id) == len(expected)
        and not missing_ids
        and not duplicate_count
        and not invalid_count
        and not missing_file_count
        and scene_ids == set(range(1, int(expected_scene_count) + 1))
        and all(count == 4 for count in scene_movie_counts.values())
    )
    return {
        "complete_o1": bool(complete),
        "manifest_sample_count": len(records),
        "expected_sample_count": len(expected),
        "scene_count": len(scene_ids),
        "expected_scene_count": int(expected_scene_count),
        "missing_count": len(missing_ids),
        "missing_sample_ids": missing_ids,
        "duplicate_count": int(duplicate_count),
        "invalid_count": int(invalid_count),
        "missing_file_count": int(missing_file_count),
        "scene_movie_count_errors": invalid_scene_counts,
        "unique_valid_sample_count": len(valid_by_id),
    }


def build_task_mapping(task_lines, records):
    """Map official task paths to manifest files by parsed IDs, preserving order."""
    records_by_id = {}
    for record in records:
        sample_id = str(record.get("sample_id", "")) if isinstance(record, dict) else ""
        if parse_sample_id(sample_id) is not None and sample_id not in records_by_id:
            records_by_id[sample_id] = str(record.get("path", ""))

    lines = [str(line).strip() for line in task_lines if str(line).strip()]
    mapping = []
    seen_ids = set()
    duplicate_count = 0
    invalid_count = 0
    scope_error = False
    missing_count = 0
    for line in lines:
        official_path = line.split()[0]
        parsed = parse_official_movie_path(official_path)
        if parsed is None:
            invalid_count += 1
            continue
        block, scene, movie = parsed
        sample_id = f"{block}/{scene:04d}/{movie}"
        if block != "O1":
            scope_error = True
        if sample_id in seen_ids:
            duplicate_count += 1
            continue
        seen_ids.add(sample_id)
        if sample_id not in records_by_id:
            missing_count += 1
            continue
        mapping.append((official_path, records_by_id[sample_id]))

    valid = (
        bool(lines)
        and len(mapping) == len(lines)
        and not duplicate_count
        and not invalid_count
        and not missing_count
        and not scope_error
        and len({official for official, _ in mapping}) == len(mapping)
    )
    if not valid:
        mapping = []
    return mapping, {
        "task_count": len(lines),
        "mapped_task_count": len(mapping),
        "path_mapping_valid": bool(valid),
        "duplicate_count": int(duplicate_count),
        "invalid_count": int(invalid_count),
        "missing_count": int(missing_count),
        "o1_only": not scope_error,
    }


def plausibility_from_surprise(surprise):
    """Bounded, monotonic local score: lower surprise means higher plausibility."""
    value = float(surprise)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"surprise must be finite and non-negative, got {value!r}")
    score = math.exp(-value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"invalid plausibility score {score!r}")
    return score


def create_submission_zip(answer_path, archive_path):
    answer_path = Path(answer_path)
    archive_path = Path(archive_path)
    if not answer_path.is_file():
        raise FileNotFoundError(answer_path)
    if archive_path.exists():
        raise FileExistsError(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, mode="x", compression=zipfile.ZIP_STORED) as archive:
        archive.write(answer_path, arcname="answer.txt")
