"""Inference-only runner for the processed IntPhys O1 test block.

This entry point deliberately keeps local inference artifacts separate from an
official submission.  It only creates a submission ZIP after an explicit task
list, score protocol file, and validator have been supplied and the validator
has passed.
"""

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

from evals.intphys_test import eval as canonical_eval
from evals.intphys_test.processed_test_support import (
    audit_manifest_records,
    build_task_mapping,
    create_submission_zip,
    is_safe_relative_path,
    parse_sample_id,
    plausibility_from_surprise,
)


LOGGER = logging.getLogger("processed_o1_test")
EXPECTED_SCENES = 1080
EXPECTED_MOVIES = EXPECTED_SCENES * 4
EXPECTED_RGB_FRAMES = 100
EXPECTED_RGB_SIZE = 128


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_manifest(data_root):
    manifest_path = Path(data_root) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json does not exist: {manifest_path}")
    records = _load_json(manifest_path)
    if not isinstance(records, list):
        raise ValueError("processed manifest.json must contain a list of samples")
    return records, manifest_path


def _read_task_list(path):
    if path is None:
        return []
    task_path = Path(path)
    if not task_path.is_file():
        raise FileNotFoundError(f"task list does not exist: {task_path}")
    return task_path.read_text(encoding="utf-8").splitlines()


def _valid_unique_records(records, data_root):
    selected = []
    seen_ids = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        sample_id = str(record.get("sample_id", ""))
        relative_path = str(record.get("path", ""))
        parsed = parse_sample_id(sample_id)
        if (
            parsed is None
            or not is_safe_relative_path(relative_path)
            or sample_id in seen_ids
            or record.get("num_frames", 100) != EXPECTED_RGB_FRAMES
            or record.get("image_size", 128) != EXPECTED_RGB_SIZE
            or not (Path(data_root) / relative_path).is_file()
        ):
            continue
        seen_ids.add(sample_id)
        selected.append(record)
    return selected


def _transformer_eval_config(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    return {
        "predictor_type": "vit_transformer",
        "pretrain": {
            "enc_checkpoint_key": "encoder",
            "pred_checkpoint_key": "predictor",
            "model_name": "vit_large",
            "patch_size": 16,
            "folder": str(checkpoint_path.parent),
            "checkpoint": checkpoint_path.name,
            "write_tag": "vjepa_vitl16_transformer",
            "use_sdpa": True,
            "use_silu": False,
            "wide_silu": False,
            "uniform_power": True,
            "is_causal": False,
            "pred_is_causal": False,
            "pred_depth": 12,
            "tubelet_size": 2,
            "frames_per_clip": 16,
        },
        "data": {
            "batch_size": 1,
            "resolution": 224,
            "stride_sliding_window": 2,
            "use_bfloat16": True,
            "frames_per_clip": 16,
            "context_lengths": [2, 4, 6, 8, 10],
            "frame_steps": 2,
        },
        "predictor": {
            "predictor_dim": 384,
            "output_dim": 1024,
            "num_tokens": 1568,
            "num_chunks": 8,
            "chunk_tokens": 196,
        },
        "normalize_targets": True,
    }


def _checkpoint_config(checkpoint_path, predictor_type="onn_feedback"):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    if predictor_type not in {"onn_feedback", "vit_transformer"}:
        raise ValueError(f"unsupported predictor_type: {predictor_type}")
    if predictor_type == "onn_feedback" and checkpoint_path.name.endswith(".last.pt"):
        raise ValueError("refusing to use a .last.pt checkpoint; pass the best checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if predictor_type == "vit_transformer":
        required_keys = ("encoder", "target_encoder", "predictor")
        missing_keys = [
            key for key in required_keys
            if not isinstance(checkpoint.get(key), dict)
        ]
        if missing_keys:
            raise ValueError(
                "Transformer checkpoint is missing state dictionaries: "
                f"{missing_keys}"
            )
        return checkpoint, _transformer_eval_config(checkpoint_path), checkpoint_path
    if checkpoint.get("checkpoint_kind") != "best":
        raise ValueError(
            "refusing checkpoint without checkpoint_kind=best; "
            f"got {checkpoint.get('checkpoint_kind')!r}"
        )
    if checkpoint.get("experiment_mode") != "onn_feedback":
        raise ValueError("checkpoint is not an onn_feedback checkpoint")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("best checkpoint has no complete saved config")
    if config.get("predictor_type") != "onn_feedback":
        raise ValueError("saved config is not predictor_type=onn_feedback")
    data_cfg = config.get("data", {})
    pretrain_cfg = config.get("pretrain", {})
    predictor_cfg = config.get("predictor", {})
    exact_values = {
        "data.frames_per_clip": (data_cfg.get("frames_per_clip"), 16),
        "pretrain.frames_per_clip": (pretrain_cfg.get("frames_per_clip"), 16),
        "data.resolution": (data_cfg.get("resolution"), 224),
        "data.frame_steps": (data_cfg.get("frame_steps"), 2),
        "data.stride_sliding_window": (data_cfg.get("stride_sliding_window"), 2),
        "pretrain.patch_size": (pretrain_cfg.get("patch_size"), 16),
        "predictor.num_tokens": (predictor_cfg.get("num_tokens"), 1568),
        "predictor.num_chunks": (predictor_cfg.get("num_chunks"), 8),
        "predictor.chunk_tokens": (predictor_cfg.get("chunk_tokens"), 196),
        "predictor.predictor_dim": (predictor_cfg.get("predictor_dim"), 384),
        "predictor.output_dim": (predictor_cfg.get("output_dim"), 1024),
    }
    mismatches = {
        name: {"saved": saved, "required": required}
        for name, (saved, required) in exact_values.items()
        if saved != required
    }
    if mismatches:
        raise ValueError(f"unsupported checkpoint input contract: {mismatches}")
    pretrain_path = Path(pretrain_cfg["folder"]) / pretrain_cfg["checkpoint"]
    if not pretrain_path.is_file():
        raise FileNotFoundError(f"saved encoder checkpoint does not exist: {pretrain_path}")
    return checkpoint, config, pretrain_path


def _freeze_eval(module):
    if module is None:
        return
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _load_models(checkpoint_path, config, device):
    pretrain_cfg = config["pretrain"]
    predictor_cfg = config["predictor"]
    predictor_type = config.get("predictor_type", "onn_feedback")
    if predictor_type == "onn_feedback":
        onn_cfg = config.get("onn", config.get("onn_feedback"))
        if not isinstance(onn_cfg, dict):
            raise ValueError("saved checkpoint has no ONN configuration")
        pretrained_path = Path(pretrain_cfg["folder"]) / pretrain_cfg["checkpoint"]
        predictor_checkpoint = str(checkpoint_path)
    elif predictor_type == "vit_transformer":
        onn_cfg = None
        pretrained_path = Path(checkpoint_path)
        predictor_checkpoint = None
    else:
        raise ValueError(f"unsupported predictor_type: {predictor_type}")
    encoder, target_encoder, predictor = canonical_eval.init_model(
        device=device,
        pretrained=str(pretrained_path),
        model_name=pretrain_cfg["model_name"],
        patch_size=pretrain_cfg["patch_size"],
        crop_size=config["data"]["resolution"],
        frames_per_clip=config["data"]["frames_per_clip"],
        tubelet_size=pretrain_cfg["tubelet_size"],
        use_sdpa=pretrain_cfg.get("use_sdpa", True),
        use_SiLU=pretrain_cfg.get("use_silu", False),
        wide_SiLU=pretrain_cfg.get("wide_silu", False),
        is_causal=pretrain_cfg.get("is_causal", False),
        pred_is_causal=pretrain_cfg.get("pred_is_causal", False),
        uniform_power=pretrain_cfg.get("uniform_power", True),
        enc_checkpoint_key=pretrain_cfg.get("enc_checkpoint_key", "encoder"),
        pred_checkpoint_key=pretrain_cfg.get("pred_checkpoint_key", "predictor"),
        pred_embed_dim=predictor_cfg["predictor_dim"],
        output_mode=predictor_cfg.get("output_mode", "mlp"),
        pred_depth=pretrain_cfg.get("pred_depth", 12),
        optical_qkv={},
        predictor_checkpoint=predictor_checkpoint,
        predictor_type=predictor_type,
        onn_feedback_config=onn_cfg,
    )
    _freeze_eval(encoder)
    _freeze_eval(target_encoder)
    _freeze_eval(predictor)
    return encoder, target_encoder, predictor


class ProcessedO1Dataset(Dataset):
    def __init__(self, records, data_root, transform, frame_step):
        self.records = list(records)
        self.data_root = Path(data_root)
        self.transform = transform
        self.frame_step = int(frame_step)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        started = time.perf_counter()
        path = self.data_root / str(record["path"])
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            rgb = payload.get("rgb") if isinstance(payload, dict) else None
            if not isinstance(rgb, torch.Tensor):
                raise ValueError("sample has no RGB tensor")
            if tuple(rgb.shape) != (EXPECTED_RGB_FRAMES, 3, EXPECTED_RGB_SIZE, EXPECTED_RGB_SIZE):
                raise ValueError(f"RGB shape must be [100,3,128,128], got {tuple(rgb.shape)}")
            if rgb.dtype != torch.uint8:
                raise ValueError(f"RGB dtype must be uint8, got {rgb.dtype}")
            # Match the existing IntPhys evaluator: 49 frames from 0..96 at step 2.
            sampled = rgb[: 99 - (99 % self.frame_step) : self.frame_step]
            sampled = sampled[: 99 // self.frame_step]
            frames_thwc = sampled.permute(0, 2, 3, 1).contiguous()
            clip = self.transform(frames_thwc)
            if tuple(clip.shape) != (3, 49, 224, 224):
                raise ValueError(f"transformed clip must be [3,49,224,224], got {tuple(clip.shape)}")
            return {
                "clip": clip,
                "sample_id": str(record["sample_id"]),
                "relative_path": str(record["path"]),
                "read_time_s": time.perf_counter() - started,
                "error": "",
            }
        except Exception as exc:
            return {
                "clip": None,
                "sample_id": str(record.get("sample_id", "")),
                "relative_path": str(record.get("path", "")),
                "read_time_s": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }


def _keep_items(batch):
    return batch


def _make_fixed_masks(context_length, batch_size, device, patch_size=16, frames_per_clip=16):
    mask_ctxt, mask_tgt, full_mask = canonical_eval.get_time_masks(
        context_length,
        spatial_size=(patch_size, patch_size),
        temporal_dim=frames_per_clip,
        as_bool=False,
    )
    masks_ctxt = mask_ctxt.unsqueeze(0).repeat(batch_size, 1).to(device=device, dtype=torch.long)
    masks_tgt = mask_tgt.unsqueeze(0).repeat(batch_size, 1).to(device=device, dtype=torch.long)
    full_mask = full_mask.unsqueeze(0).repeat(batch_size, 1).to(device=device, dtype=torch.long)
    merged = torch.cat((masks_ctxt, masks_tgt), dim=1)
    if merged.shape[1] != 1568 or torch.unique(merged, dim=1).shape[1] != 1568:
        raise ValueError("fixed evaluation masks do not cover 1568 tokens without overlap")
    return [masks_ctxt], [masks_tgt], [full_mask]


def _infer_batch(clips, config, encoder, target_encoder, predictor, device):
    data_cfg = config["data"]
    pretrain_cfg = config["pretrain"]
    context_lengths = list(data_cfg["context_lengths"])
    frames_per_clip = int(data_cfg["frames_per_clip"])
    stride = int(data_cfg["stride_sliding_window"])
    pieces = clips.unfold(2, frames_per_clip, stride).permute(0, 2, -1, 1, 3, 4).contiguous()
    num_movies = pieces.shape[0]
    num_windows = pieces.shape[1]
    pieces = rearrange(pieces.flatten(0, 1), "b t c h w -> b c t h w").contiguous()
    feature_time_total = 0.0
    onn_time_total = 0.0
    losses_by_context = []
    with canonical_eval.autocast_context(device, bool(data_cfg.get("use_bfloat16", False))):
        for context_length in context_lengths:
            feature_started = time.perf_counter()
            masks_ctxt, masks_tgt, full_mask = _make_fixed_masks(
                context_length,
                pieces.shape[0],
                device,
                patch_size=int(pretrain_cfg["patch_size"]),
                frames_per_clip=frames_per_clip,
            )
            hidden = target_encoder(pieces, full_mask)[0]
            if config.get("normalize_targets", True):
                hidden = F.layer_norm(hidden, (hidden.shape[-1],))
            targets = canonical_eval.apply_masks(hidden, masks_tgt, concat=False)
            context = encoder(pieces, masks_ctxt)
            if data_cfg.get("normalize_context", False):
                context = [F.layer_norm(value, (value.shape[-1],)) for value in context]
            onn_started = time.perf_counter()
            feature_time_total += onn_started - feature_started
            predictions = predictor(context, targets, masks_ctxt, masks_tgt)
            onn_time_total += time.perf_counter() - onn_started
            pred = predictions[0].view(num_movies, num_windows, *predictions[0].shape[1:])
            target = targets[0].view(num_movies, num_windows, *targets[0].shape[1:])
            losses_by_context.append(F.l1_loss(pred, target, reduction="none").mean((2, 3)))
    losses = torch.stack(losses_by_context, dim=1).detach().float().cpu()
    return losses, feature_time_total, onn_time_total


def _aggregate_loss(losses, aggregation):
    if aggregation == "average":
        return float(losses.mean().item())
    if aggregation == "maximum":
        return float(losses.max().item())
    raise ValueError(f"unsupported aggregation {aggregation!r}")


def _write_csv(path, rows):
    fields = [
        "sample_id", "official_movie_path", "relative_path", "status",
        "surprise_average", "surprise_maximum", "plausibility_average",
        "plausibility_maximum", "selected_aggregation", "selected_surprise",
        "selected_plausibility", "error",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _expand_validator_args(values, answer_path, task_path):
    replacements = {"{answer}": str(answer_path), "{task}": str(task_path)}
    return [replacements.get(value, value) for value in (values or [])]


def _run_validator(validator, validator_args, answer_path, task_path):
    if validator is None:
        return False, "validator_not_supplied"
    validator_path = Path(validator)
    if not validator_path.is_file():
        return False, f"validator_missing:{validator_path}"
    if not validator_args:
        return False, "validator_args_not_supplied_use_placeholders_answer_task"
    command = [sys.executable, str(validator_path)] + _expand_validator_args(
        validator_args, answer_path, task_path
    )
    completed = subprocess.run(
        command,
        cwd=str(validator_path.parent),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        LOGGER.info("validator_stdout=%s", completed.stdout.strip())
    if completed.stderr:
        LOGGER.info("validator_stderr=%s", completed.stderr.strip())
    return completed.returncode == 0, f"validator_returncode:{completed.returncode}"


def _write_answer(path, mapping, rows_by_relative_path, aggregation):
    with Path(path).open("w", encoding="utf-8") as handle:
        for official_path, relative_path in mapping:
            row = rows_by_relative_path[relative_path]
            value = row["plausibility_" + aggregation]
            handle.write(f"{official_path} {float(value):.10f}\n")


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="ONN best .pt or trained V-JEPA Transformer .pth.tar checkpoint",
    )
    parser.add_argument(
        "--predictor-type",
        choices=("onn_feedback", "vit_transformer"),
        default="onn_feedback",
    )
    parser.add_argument("--data-root", required=True, help="processed test_o1_rgbd_128_allframes directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-list", help="official task.txt; omit for local-only inference")
    parser.add_argument("--score-script", help="starter-kit score.py; required for official_ready=true")
    parser.add_argument("--validator", help="starter-kit validate.py/validator.py")
    parser.add_argument("--validator-args", nargs="*", help="validator args; use {answer} and {task} placeholders")
    parser.add_argument("--aggregation", choices=("average", "maximum"), default="average")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    log_path = output_dir / "inference.log"
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)
    started = time.perf_counter()

    data_root = Path(args.data_root)
    records, manifest_path = _load_manifest(data_root)
    manifest_audit = audit_manifest_records(records, data_root, EXPECTED_SCENES)
    task_lines = _read_task_list(args.task_list)
    task_mapping, task_audit = build_task_mapping(task_lines, records)
    official_task_valid = bool(task_mapping and task_audit["path_mapping_valid"])
    if args.task_list is None:
        LOGGER.info("task_list_not_supplied local_only=true")
    elif not official_task_valid:
        LOGGER.info("task_mapping_invalid official_ready=false audit=%s", task_audit)

    candidate_records = _valid_unique_records(records, data_root)
    if not candidate_records:
        raise RuntimeError("manifest has no valid processed RGB samples to infer")
    if official_task_valid:
        by_relative_path = {str(record["path"]): record for record in candidate_records}
        ordered_records = [
            by_relative_path[relative_path]
            for _, relative_path in task_mapping
            if relative_path in by_relative_path
        ]
    else:
        ordered_records = candidate_records

    checkpoint, config, _ = _checkpoint_config(
        args.checkpoint,
        predictor_type=args.predictor_type,
    )
    data_cfg = config["data"]
    batch_size = int(args.batch_size or data_cfg.get("batch_size", 1))
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
            raise ValueError(f"GPU {args.gpu} is unavailable")
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    LOGGER.info(
        "run_start official_scope=O1-only manifest=%s manifest_complete_o1=%s "
        "manifest_samples=%d scenes=%d missing=%d duplicates=%d invalid=%d "
        "checkpoint=%s checkpoint_epoch=%s device=%s batch_size=%d",
        manifest_path,
        manifest_audit["complete_o1"],
        manifest_audit["manifest_sample_count"],
        manifest_audit["scene_count"],
        manifest_audit["missing_count"],
        manifest_audit["duplicate_count"],
        manifest_audit["invalid_count"],
        args.checkpoint,
        checkpoint.get("epoch"),
        device,
        batch_size,
    )
    LOGGER.info(
        "input_contract frames_from_each_sample=100 deterministic_indices=0:98:2 "
        "sampled_frames=49 model_frames=16 input_resolution=128 model_resolution=224 "
        "layout=RGB_TCHW_after_transform dtype=float32",
    )

    transform = canonical_eval.make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=[1.0, 1.0],
        random_resize_scale=[1.0, 1.0],
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=int(data_cfg["resolution"]),
    )
    dataset = ProcessedO1Dataset(
        ordered_records,
        data_root,
        transform,
        frame_step=int(data_cfg["frame_steps"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=_keep_items,
    )
    encoder, target_encoder, predictor = _load_models(args.checkpoint, config, device)
    LOGGER.info("model_ready encoder=eval target_encoder=eval predictor=eval gradients=disabled")
    total_batches = len(loader)
    rows = []
    rows_by_relative_path = {}
    failure_count = 0
    processed_count = 0
    for batch_index, items in enumerate(loader, start=1):
        valid_items = [item for item in items if not item["error"]]
        for item in items:
            if item["error"]:
                failure_count += 1
                rows.append({
                    "sample_id": item["sample_id"],
                    "official_movie_path": "",
                    "relative_path": item["relative_path"],
                    "status": "failed",
                    "surprise_average": "",
                    "surprise_maximum": "",
                    "plausibility_average": "",
                    "plausibility_maximum": "",
                    "selected_aggregation": args.aggregation,
                    "selected_surprise": "",
                    "selected_plausibility": "",
                    "error": item["error"],
                })
                LOGGER.error("movie_failed batch=%d/%d sample_id=%s path=%s error=%s", batch_index, total_batches, item["sample_id"], item["relative_path"], item["error"])
        if not valid_items:
            continue
        clips = torch.stack([item["clip"] for item in valid_items], dim=0).to(device, non_blocking=True)
        read_time = sum(float(item["read_time_s"]) for item in valid_items)
        with torch.no_grad():
            losses, feature_time, onn_time = _infer_batch(
                clips, config, encoder, target_encoder, predictor, device
            )
        batch_surprise = []
        batch_plausibility = []
        official_by_relative = {}
        if official_task_valid:
            official_by_relative = {relative: official for official, relative in task_mapping}
        for item, loss_tensor in zip(valid_items, losses):
            average_surprise = _aggregate_loss(loss_tensor, "average")
            maximum_surprise = _aggregate_loss(loss_tensor, "maximum")
            average_plausibility = plausibility_from_surprise(average_surprise)
            maximum_plausibility = plausibility_from_surprise(maximum_surprise)
            selected_surprise = average_surprise if args.aggregation == "average" else maximum_surprise
            selected_plausibility = average_plausibility if args.aggregation == "average" else maximum_plausibility
            row = {
                "sample_id": item["sample_id"],
                "official_movie_path": official_by_relative.get(item["relative_path"], ""),
                "relative_path": item["relative_path"],
                "status": "ok",
                "surprise_average": f"{average_surprise:.10f}",
                "surprise_maximum": f"{maximum_surprise:.10f}",
                "plausibility_average": f"{average_plausibility:.10f}",
                "plausibility_maximum": f"{maximum_plausibility:.10f}",
                "selected_aggregation": args.aggregation,
                "selected_surprise": f"{selected_surprise:.10f}",
                "selected_plausibility": f"{selected_plausibility:.10f}",
                "error": "",
            }
            rows.append(row)
            rows_by_relative_path[item["relative_path"]] = row
            processed_count += 1
            batch_surprise.append(selected_surprise)
            batch_plausibility.append(selected_plausibility)
        elapsed = time.perf_counter() - started
        path_values = [item["relative_path"] for item in valid_items]
        LOGGER.info(
            "batch=%d/%d movie_path_range=%s..%s data_read_time_s=%.3f "
            "feature_time_s=%.3f onn_time_s=%.3f surprise=%.6f plausibility=%.6f "
            "processed=%d failures=%d total_time_s=%.1f",
            batch_index,
            total_batches,
            path_values[0],
            path_values[-1],
            read_time,
            feature_time,
            onn_time,
            sum(batch_surprise) / len(batch_surprise),
            sum(batch_plausibility) / len(batch_plausibility),
            processed_count,
            failure_count,
            elapsed,
        )

    csv_path = output_dir / "per_movie_scores.csv"
    _write_csv(csv_path, rows)
    score_script_exists = bool(args.score_script and Path(args.score_script).is_file())
    answer_path = output_dir / "answer.txt"
    answer_created = False
    validator_passed = False
    validator_message = "not_run"
    if official_task_valid and processed_count == EXPECTED_MOVIES and failure_count == 0:
        _write_answer(answer_path, task_mapping, rows_by_relative_path, args.aggregation)
        answer_created = True
        if score_script_exists:
            validator_passed, validator_message = _run_validator(
                args.validator, args.validator_args, answer_path, args.task_list
            )
        else:
            validator_message = "score_script_not_supplied_or_missing"
    official_ready = bool(
        manifest_audit["complete_o1"]
        and task_audit.get("task_count") == EXPECTED_MOVIES
        and task_audit.get("path_mapping_valid")
        and processed_count == EXPECTED_MOVIES
        and failure_count == 0
        and score_script_exists
        and answer_created
        and validator_passed
    )
    archive_path = output_dir / "submission_o1.zip"
    if official_ready:
        create_submission_zip(answer_path, archive_path)
    audit = {
        "official_scope": "O1-only",
        "official_ready": official_ready,
        "task_count": task_audit.get("task_count", 0),
        "processed_sample_count": processed_count,
        "manifest_sample_count": manifest_audit["manifest_sample_count"],
        "scene_count": manifest_audit["scene_count"],
        "missing_count": manifest_audit["missing_count"],
        "duplicate_count": manifest_audit["duplicate_count"],
        "invalid_count": manifest_audit["invalid_count"],
        "missing_file_count": manifest_audit["missing_file_count"],
        "failure_count": failure_count,
        "manifest_complete_o1": manifest_audit["complete_o1"],
        "task_mapping_valid": task_audit.get("path_mapping_valid", False),
        "task_o1_only": task_audit.get("o1_only", False),
        "score_direction": "lower_surprise_is_higher_plausibility",
        "score_transform": "exp(-aggregated_surprise)",
        "aggregation": args.aggregation,
        "predictor_type": args.predictor_type,
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "saved_config_contract": {
            "frames_per_clip": data_cfg["frames_per_clip"],
            "resolution": data_cfg["resolution"],
            "frame_steps": data_cfg["frame_steps"],
            "stride_sliding_window": data_cfg["stride_sliding_window"],
            "num_tokens": config["predictor"]["num_tokens"],
            "num_chunks": config["predictor"]["num_chunks"],
            "chunk_tokens": config["predictor"]["chunk_tokens"],
        },
        "score_script": str(Path(args.score_script)) if args.score_script else None,
        "score_script_exists": score_script_exists,
        "validator": str(Path(args.validator)) if args.validator else None,
        "validator_result": validator_message,
        "answer_created": answer_created,
        "submission_zip_created": archive_path.is_file(),
        "elapsed_s": time.perf_counter() - started,
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info(
        "run_done official_scope=O1-only official_ready=%s task_count=%d "
        "processed_sample_count=%d scene_count=%d missing_count=%d duplicate_count=%d "
        "failures=%d total_time_s=%.1f",
        official_ready,
        audit["task_count"],
        processed_count,
        audit["scene_count"],
        audit["missing_count"],
        audit["duplicate_count"],
        failure_count,
        audit["elapsed_s"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
